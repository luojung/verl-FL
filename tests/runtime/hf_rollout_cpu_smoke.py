# Copyright 2026 contributors. Licensed under the Apache License, Version 2.0.
"""Offline real-tensor shared HF rollout smoke, independent of single_card_cpu_smoke.py.

Run with a CPU interpreter containing torch, transformers, tensordict and numpy:
    python tests/runtime/hf_rollout_cpu_smoke.py

Two tiny random decoder-only models (Qwen2 and Gemma3 text) use single-rank
Gloo, root-only FSDP1 NO_SHARD/use_orig_params=True. Each generates before and
after a real optimizer forward/backward/update. No weights, tokenizer, network,
installation, accelerator, Ray service or remote command is used.

Production HFRollout, BaseRollout, DataProto and tensor helpers are source-loaded
unchanged, omitting unrelated package/Ray/config/device imports. Real PyTorch,
Transformers models, FSDP, TensorDict and production DataProto methods execute;
there are no model/tensor/protocol doubles. Full VeRL imports, worker lifecycle,
offload, mixed precision, VLM image inputs and DCU/multi-card are NOT validated.
Missing dependencies, unsupported CPU FSDP or a failed case produce failure,
never a partial PASS. --help works without the runtime dependencies.
"""

from __future__ import annotations
import __future__

import argparse
import ast
import gc
import json
import os
import sys
import tempfile
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def source_module(name, relative_path, namespace=None, select=None):
    """Retain production definitions, without initializing unrelated VeRL services."""
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if select is not None and isinstance(node, (ast.Import, ast.ImportFrom)):
            continue  # Selected definitions receive only their actual exercised dependencies.
        if isinstance(node, ast.Import):
            aliases = [alias for alias in node.names if alias.name != "ray"]
            if aliases:
                node.names = aliases
                body.append(node)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and not (node.module or "").startswith("verl"):
                body.append(node)
        elif select is None or (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in select):
            body.append(node)
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__dict__.update(namespace or {})
    sys.modules[name] = module  # dataclass resolves this module's namespace.
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag),
        module.__dict__,
    )
    return module


def load_production(torch):
    # DataProto's exercised constructor/slice/split/concat methods require no Ray,
    # collective/device helpers or union helpers. Their definitions stay intact.
    protocol = source_module("hf_cpu_protocol", "verl/protocol.py")
    positions = source_module(
        "hf_cpu_positions", "verl/utils/model.py", {"torch": torch}, select={"compute_position_id_with_mask"}
    )
    masks = source_module(
        "hf_cpu_masks", "verl/utils/torch_functional.py", {"torch": torch}, select={"get_response_mask"}
    )
    base = source_module(
        "hf_cpu_base",
        "verl/workers/rollout/base.py",
        {"ABC": ABC, "abstractmethod": abstractmethod},
        select={"BaseRollout"},
    )
    rollout = source_module(
        "hf_cpu_rollout",
        "verl/workers/rollout/hf_rollout.py",
        {
            "DataProto": protocol.DataProto,
            "BaseRollout": base.BaseRollout,
            "compute_position_id_with_mask": positions.compute_position_id_with_mask,
            "get_response_mask": masks.get_response_mask,
        },
    )
    return protocol.DataProto, rollout.HFRollout


class Config(dict):
    """Small concrete rollout configuration with the production get/attribute interface."""

    __getattr__ = dict.__getitem__


def tiny_model(name, transformers):
    kwargs = dict(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attention_dropout=0.0,
        use_cache=True,
    )
    if name == "qwen2":
        config = transformers.Qwen2Config(**kwargs)
        model_cls = transformers.Qwen2ForCausalLM
    else:
        config = transformers.Gemma3TextConfig(
            **kwargs,
            head_dim=8,
            sliding_window=16,
            sliding_window_pattern=2,
        )
        model_cls = transformers.Gemma3ForCausalLM
    config._attn_implementation = "eager"
    return model_cls(config).float()


def check_generation(output, prompts, torch, step):
    batch = output.batch
    ids = prompts.batch["input_ids"]
    width = ids.size(1)
    length = prompts.meta_info["response_length"]
    require(batch["responses"].shape == (ids.size(0), length), "Metadata response_length was not honored")
    require(batch["input_ids"].shape == (ids.size(0), width + length), "Unexpected total sequence shape")
    require(torch.equal(batch["prompts"], ids), "Prompt prefix changed")
    require(torch.equal(batch["attention_mask"][:, :width], prompts.batch["attention_mask"]), "Prompt mask changed")
    mask = batch["response_mask"]
    require(mask.dtype == torch.int64 and mask.device.type == "cpu", "Response mask dtype/device changed")
    require(torch.all((mask == 0) | (mask == 1)).item(), "Response mask is not binary")
    require(torch.all(mask[:, 0] == 1).item(), "First generated token must remain active, including EOS")
    require(torch.all(mask[:, 1:] <= mask[:, :-1]).item(), "Response mask reactivates after termination")
    for row, row_mask in zip(batch["responses"].tolist(), mask.tolist(), strict=True):
        if 2 in row:
            eos = row.index(2)
            require(row_mask == [1] * (eos + 1) + [0] * (length - eos - 1), "EOS must be included, tail masked")
        else:
            require(row_mask == [1] * length, "Greedy generation without EOS must reach max_new_tokens")
    require(torch.equal(batch["attention_mask"][:, width:], mask), "Response/attention masks disagree")
    expected_positions = torch.tensor([[0, 0, 0, 1, 2, 3, 4], [0, 0, 1, 2, 3, 4, 5]])
    require(torch.equal(batch["position_ids"], expected_positions), "Position bookkeeping changed")
    require(output.meta_info["global_steps"] == step, "Generation lost step metadata")
    require(output.meta_info["response_length"] == length, "Generation lost response-length metadata")


def run_case(name, rt, DataProto, HFRollout):
    torch = rt.torch
    torch.manual_seed(1729)
    actor = tiny_model(name, rt.transformers)
    wrapper = rt.FSDP(
        actor,
        device_id=torch.device("cpu"),
        sharding_strategy=rt.ShardingStrategy.NO_SHARD,
        use_orig_params=True,
        auto_wrap_policy=None,
    )
    config = Config(
        prompt_length=8,
        response_length=5,
        micro_batch_size=1,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        dtype="float32",
        val_kwargs=Config(do_sample=False, temperature=1.0, top_p=1.0, top_k=-1),
    )
    rollout = HFRollout(wrapper, config)
    require(rollout._generation_module() is actor, "Rollout did not share the live actor")
    optimizer = torch.optim.SGD(wrapper.parameters(), lr=0.05)
    ids = torch.tensor([[0, 0, 5, 6], [0, 7, 8, 9]], dtype=torch.int64)
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.int64)
    prompts = DataProto(
        batch=rt.TensorDict({"input_ids": ids, "attention_mask": mask}, batch_size=[2]),
        meta_info={"eos_token_id": 2, "pad_token_id": 0, "response_length": 3, "global_steps": 0},
    )
    check_generation(rollout.generate_sequences(prompts), prompts, torch, step=0)
    require(wrapper.training and actor.training, "Generation failed to restore train mode")
    before = {key: parameter.detach().clone() for key, parameter in wrapper.named_parameters()}
    labels = ids.masked_fill(mask == 0, -100)
    optimizer.zero_grad(set_to_none=True)
    loss = wrapper(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False).loss
    require(torch.isfinite(loss).item(), "Training loss is not finite")
    loss.backward()
    optimizer.step()
    changed = any(not torch.equal(before[key], parameter.detach()) for key, parameter in wrapper.named_parameters())
    require(changed, "Optimizer did not update any live actor parameters")
    prompts.meta_info["global_steps"] = 1
    check_generation(rollout.generate_sequences(prompts), prompts, torch, step=1)
    require(wrapper.training and actor.training, "Post-update generation failed to restore train mode")
    print(
        json.dumps(
            {
                "model": name,
                "loss": float(loss.detach()),
                "optimizer_changed_weights": changed,
                "generation_checks": 2,
                "response_length": 3,
                "status": "PASS",
            }
        ),
        flush=True,
    )
    del rollout, wrapper, actor, optimizer, before, loss
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=["qwen2", "gemma3"], default=["qwen2", "gemma3"])
    args = parser.parse_args()
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import torch.distributed as dist
    import transformers
    from tensordict import TensorDict
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(dist.is_available() and dist.is_gloo_available(), "This smoke requires local CPU Gloo")
    require(not dist.is_initialized(), "Run standalone; this script owns its local process group")
    rt = SimpleNamespace(
        torch=torch, transformers=transformers, TensorDict=TensorDict, FSDP=FSDP, ShardingStrategy=ShardingStrategy
    )
    DataProto, HFRollout = load_production(torch)
    print(f"RUNTIME torch={torch.__version__} transformers={transformers.__version__}", flush=True)
    with tempfile.TemporaryDirectory(prefix="verl-hf-cpu-gloo-") as directory:
        try:
            dist.init_process_group(
                "gloo",
                init_method=(Path(directory) / "rendezvous").as_uri(),
                rank=0,
                world_size=1,
                timeout=timedelta(seconds=30),
            )
            for name in args.models:
                run_case(name, rt, DataProto, HFRollout)
        finally:
            gc.collect()
            if dist.is_initialized():
                dist.destroy_process_group()
    print("PASS: requested real CPU HF/FSDP cases; NOT DCU/FlagOS or multi-card validation", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
