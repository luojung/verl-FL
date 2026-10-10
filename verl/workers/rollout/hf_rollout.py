# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Synchronous text-only HF generation using the actor's live parameters.

FSDP generation is restricted to a single-rank, root-only NO_SHARD wrapper.
Calling a delegated HF generate bypasses the root FSDP forward; nested/sharded
wrappers require a different, separately validated generation implementation.
"""

import contextlib

import torch
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from transformers import GenerationConfig

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import get_response_mask

from .base import BaseRollout

__all__ = ["HFRollout"]


class HFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config, tokenizer=None, *, data_config=None, device=None):
        # Do not construct an async inference engine or copy actor parameters.
        self.config = config
        self.module = module
        self.tokenizer = tokenizer
        self.data_config = data_config if data_config is not None else {}
        self.device = torch.device(device) if device is not None else None
        generation_module = self._generation_module()
        if getattr(getattr(generation_module, "config", None), "is_encoder_decoder", False):
            raise NotImplementedError("Shared HF rollout requires a decoder-only causal language model")

    def _generation_module(self):
        if not isinstance(self.module, FSDP):
            return self.module
        nested = any(isinstance(child, FSDP) for child in self.module.modules() if child is not self.module)
        if (
            self.module.process_group.size() != 1
            or self.module.sharding_strategy != ShardingStrategy.NO_SHARD
            or nested
        ):
            raise NotImplementedError(
                "Shared HF rollout requires single-rank, root-only FSDP NO_SHARD. "
                "Disable nested auto-wrapping for this HF single-card path; "
                "sharded/multi-card HF generation is not supported or validated."
            )
        if not getattr(self.module, "_use_orig_params", False):
            raise NotImplementedError(
                "Shared HF rollout requires FSDP use_orig_params=True. "
                "Direct wrapped-module generation with use_orig_params=False is unsupported: "
                "parameter views may become stale after mixed-precision actor execution or offload/reload."
            )
        # With NO_SHARD and no nested wrappers, these are the live actor tensors.
        # Neither summon_full_params nor private lazy-state resets are needed.
        return self.module.module

    async def resume(self, tags: list[str]):
        # Residency/offload is owned by the synchronous actor worker.
        return None

    async def update_weights(self, weights, **kwargs):
        raise NotImplementedError("HF rollout shares live actor weights; external weight synchronization is invalid")

    async def release(self):
        return None

    def _input_device(self):
        if self.device is not None:
            return self.device
        if isinstance(self.module, FSDP):
            return self.module.compute_device
        try:
            return next(self.module.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _token_ids(self, prompts):
        eos = prompts.meta_info.get("eos_token_id")
        pad = prompts.meta_info.get("pad_token_id")
        if eos is None and self.tokenizer is not None:
            eos = self.tokenizer.eos_token_id
        if pad is None and self.tokenizer is not None:
            pad = self.tokenizer.pad_token_id
        if isinstance(eos, (list, tuple)) and not eos:
            eos = None
        if pad is None and eos is not None:
            pad = eos[0] if isinstance(eos, (list, tuple)) else eos
        if pad is None:
            raise ValueError("HF rollout needs a pad_token_id (or an eos_token_id for padding)")
        return eos, pad

    def _materialize_raw_prompts(self, prompts: DataProto) -> DataProto:
        """Build left-padded tokens without mutating tokenizer settings or the input batch."""
        if any(key in prompts.non_tensor_batch for key in ("multi_modal_data", "multi_modal_inputs")):
            raise NotImplementedError("Shared HF rollout is text-only; multimodal processor inputs are unsupported")
        device = self._input_device()
        tensors = dict(prompts.batch.items()) if prompts.batch is not None else {}
        if "input_ids" not in tensors:
            if self.tokenizer is None or "raw_prompt" not in prompts.non_tensor_batch:
                raise ValueError("HFRollout requires input_ids, or raw_prompt and a tokenizer")
            limit = self.config.prompt_length
            if limit <= 0:
                raise ValueError("prompt_length must be positive")
            truncation = prompts.meta_info.get("hf_prompt_truncation", self.data_config.get("truncation", "error"))
            if truncation not in {"error", "left", "right", "middle"}:
                raise ValueError(f"Unsupported prompt truncation policy: {truncation}")
            template_kwargs = dict(
                prompts.meta_info.get("hf_chat_template_kwargs", self.data_config.get("apply_chat_template_kwargs", {}))
            )
            if template_kwargs.get("return_tensors") is not None or template_kwargs.get("return_dict", False):
                raise ValueError("HF chat-template kwargs must return a flat token list, not tensors or a dictionary")
            template_kwargs.update(tokenize=True, add_generation_prompt=True, padding=False, truncation=False)
            rows = []
            for chat in prompts.non_tensor_batch["raw_prompt"]:
                if isinstance(chat, (str, bytes, dict)):
                    raise ValueError("raw_prompt must contain sequences of chat-message dictionaries")
                messages = list(chat)
                for message in messages:
                    if not isinstance(message, dict):
                        raise ValueError("raw_prompt messages must be dictionaries")
                    content = message.get("content")
                    if isinstance(content, (list, tuple)) and any(
                        not isinstance(part, dict) or part.get("type") != "text" for part in content
                    ):
                        raise NotImplementedError("Shared HF rollout cannot tokenize image/video/audio prompts")
                tokens = list(self.tokenizer.apply_chat_template(messages, **template_kwargs))
                if len(tokens) > limit:
                    if truncation == "error":
                        raise ValueError(f"Prompt has {len(tokens)} tokens, exceeding prompt_length={limit}")
                    if truncation == "left":
                        tokens = tokens[-limit:]
                    elif truncation == "right":
                        tokens = tokens[:limit]
                    else:
                        left = limit // 2
                        tokens = tokens[:left] + tokens[-(limit - left) :]
                if not tokens:
                    raise ValueError("HF rollout does not support empty tokenized prompts")
                rows.append(tokens)
            if not rows:
                raise ValueError("HF rollout requires a non-empty batch")
            _, pad = self._token_ids(prompts)
            width = max(map(len, rows))
            input_ids = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
            attention_mask = torch.zeros_like(input_ids)
            for row, tokens in enumerate(rows):
                input_ids[row, -len(tokens) :] = torch.tensor(tokens, dtype=torch.long, device=device)
                attention_mask[row, -len(tokens) :] = 1
            tensors.update(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=compute_position_id_with_mask(attention_mask),
            )
        else:
            if "attention_mask" not in tensors:
                raise ValueError("Pretokenized HF prompts require an explicit attention_mask (pad may equal eos)")
            tensors["input_ids"] = tensors["input_ids"].to(device)
            tensors["attention_mask"] = tensors["attention_mask"].to(device)

        idx, mask = tensors["input_ids"], tensors["attention_mask"]
        if idx.ndim != 2 or mask.shape != idx.shape or idx.size(0) == 0 or idx.size(1) == 0:
            raise ValueError("HF input_ids/attention_mask must have matching non-empty [batch, sequence] shapes")
        if idx.size(1) > self.config.prompt_length:
            raise ValueError("Pretokenized prompts exceed rollout.prompt_length")
        if not torch.all((mask == 0) | (mask == 1)) or not torch.all(mask[:, -1] == 1):
            raise ValueError("HF generation requires binary, non-empty left-padded attention masks")
        if torch.any(mask[:, 1:] < mask[:, :-1]):
            raise ValueError("HF generation requires left padding, not right padding or holes")
        positions = tensors.get("position_ids")
        if positions is None:
            positions = compute_position_id_with_mask(mask)
        if positions.shape != idx.shape:
            raise ValueError("Shared HF text rollout requires 2-D position_ids matching input_ids")
        tensors["position_ids"] = positions.to(device)
        return DataProto(
            batch=TensorDict(tensors, batch_size=idx.size(0)),
            non_tensor_batch=prompts.non_tensor_batch.copy(),
            meta_info=prompts.meta_info.copy(),
        )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        prompts = self._materialize_raw_prompts(prompts)
        batch_size = prompts.batch.batch_size[0]
        micro_batch_size = self.config.get("micro_batch_size", batch_size)
        if micro_batch_size is None:
            micro_batch_size = batch_size
        if micro_batch_size <= 0:
            raise ValueError("HF micro_batch_size must be positive")
        # split(), unlike chunk(floor(batch_size / micro_batch_size)), enforces
        # the memory bound and supports a final, smaller microbatch.
        return DataProto.concat([self._generate_minibatch(p) for p in prompts.split(micro_batch_size)])

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        is_validate = prompts.meta_info.get("validate", False)
        sampling = self.config.val_kwargs if is_validate else self.config
        do_sample = prompts.meta_info.get("do_sample", sampling.do_sample)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        if response_length <= 0:
            raise ValueError("HF response_length must be positive")
        kwargs = {"do_sample": do_sample, "num_beams": 1, "num_return_sequences": 1}
        if do_sample:
            kwargs.update(
                temperature=prompts.meta_info.get("temperature", sampling.temperature),
                top_p=prompts.meta_info.get("top_p", sampling.top_p),
                top_k=max(0, prompts.meta_info.get("top_k", sampling.top_k)),
            )
        kwargs["repetition_penalty"] = self.config.get("repetition_penalty", 1.0)
        eos_token_id, pad_token_id = self._token_ids(prompts)
        if self.config.get("ignore_eos", False):
            eos_token_id = None
        generation_config = GenerationConfig(
            **kwargs,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            max_new_tokens=response_length,
            use_cache=True,
            output_scores=False,
            return_dict_in_generate=True,
        )
        idx = prompts.batch["input_ids"]
        prompt_length = idx.size(1)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        generator = self._generation_module()
        modes = [(child, child.training) for child in self.module.modules()]
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(self.config.get("dtype", "bfloat16"))
        amp = (
            torch.autocast(device_type=idx.device.type, dtype=dtype)
            if idx.device.type != "cpu" and dtype is not None
            else contextlib.nullcontext()
        )
        try:
            self.module.eval()
            with amp:
                output = generator.generate(
                    input_ids=idx,
                    attention_mask=attention_mask,
                    # HF derives and extends position_ids from the evolving
                    # attention_mask. Passing a fixed prompt-only position_ids
                    # can reuse its final position on every cached decode step.
                    generation_config=generation_config,
                )
        finally:
            # Restore the previous mode, including mixed-mode submodules and
            # exception paths. Never force an eval-only actor into train mode.
            for child, training in modes:
                child.train(training)

        seq = output.sequences
        sequence_length = prompt_length + response_length
        if seq.size(0) != idx.size(0) or not prompt_length <= seq.size(1) <= sequence_length:
            raise ValueError("HF generate returned an unexpected batch or sequence length")
        generated_length = seq.size(1) - prompt_length
        if seq.size(1) < sequence_length:
            padding = torch.full(
                (seq.size(0), sequence_length - seq.size(1)), pad_token_id, device=seq.device, dtype=seq.dtype
            )
            seq = torch.cat((seq, padding), dim=1)
        response = seq[:, prompt_length:]
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device).unsqueeze(0)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        response_mask = (
            get_response_mask(response, eos_token=eos_token_id, dtype=attention_mask.dtype)
            if eos_token_id is not None
            else torch.ones_like(response, dtype=attention_mask.dtype)
        )
        response_mask[:, generated_length:] = 0
        batch = TensorDict(
            {
                "prompts": seq[:, :prompt_length],
                "responses": response,
                "input_ids": seq,
                "attention_mask": torch.cat((attention_mask, response_mask), dim=-1),
                "response_mask": response_mask,
                "position_ids": torch.cat((position_ids, response_position_ids), dim=-1),
            },
            batch_size=seq.size(0),
        )
        return DataProto(batch=batch, meta_info=prompts.meta_info.copy())
