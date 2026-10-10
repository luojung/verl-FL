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
"""Offline real-CPU compatibility smoke; no pytest, weights, tokenizer or downloads.

Run from any directory with the dedicated CPU interpreter:
    python tests/runtime/single_card_cpu_smoke.py --require-fsdp

Expected runtime: torch 2.10.x CPU, transformers 5.5.x, peft 0.19.x,
safetensors (and their installed dependencies). The script installs nothing.
It disables backend extension autoload for a pure CPU control.
It checks tiny Qwen2 and Gemma3 VLM-text LoRA training/export/reload, both plain
and root-only FSDP1 NO_SHARD/use_orig_params when CPU FSDP is supported.

Scope: eager float32 CPU, one rank, no images or pretrained MedGemma weights.
NOT DCU/FlagOS, FSDP2, multi-rank collectives, whole SFT trainer, checkpoint
resume, torch.compile/fusion, inference engines or production throughput.

Real production model_compat/lora_adapter modules are source-loaded to avoid
VeRL package initialization and its unrelated Ray/trainer dependencies. The
adapter helper's sole fsdp_utils dependency is the unchanged source-extracted
fsdp_version function, bound to REAL installed FSDP1/FSDP2 classes. The full
fsdp_utils module/import stack is therefore NOT validated. There are no mocks.

Exit 0 = all four cases passed; 1 = failure; 2 = plain cases passed but CPU
FSDP was explicitly unavailable (never print a full PASS in that case).
--require-fsdp turns unavailable CPU FSDP into a failure. Exceptions retain
normal tracebacks/exit status, and the local process group is always destroyed.
"""

from __future__ import annotations

import argparse
import ast
import copy
import gc
import importlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    # Unlike assert, checks remain enabled under python -O.
    if not condition:
        raise RuntimeError(message)


def load_source(name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Cannot load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def production_helpers(rt):
    """Execute actual helper implementations, without the full VeRL import stack."""
    fsdp_path = ROOT / "verl/utils/fsdp_utils.py"
    tree = ast.parse(fsdp_path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "fsdp_version"]
    require(len(functions) == 1, "Expected exactly one production fsdp_version function")
    name = "verl.utils.fsdp_utils"
    require(name not in sys.modules, "Run this smoke standalone, before importing VeRL fsdp_utils")
    bridge = ModuleType(name)
    bridge.__dict__.update(FSDP=rt.FSDP, FSDPModule=rt.FSDPModule)
    source = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(source, str(fsdp_path), "exec"), bridge.__dict__)
    sys.modules[name] = bridge
    try:
        compatibility = load_source("single_card_model_compat", "verl/utils/model_compat.py")
        adapters = load_source("single_card_lora_adapter", "verl/utils/checkpoint/lora_adapter.py")
        print("HELPERS: real model_compat + lora_adapter; source-only fsdp_version with real FSDP classes", flush=True)
        yield compatibility, adapters
    finally:
        del sys.modules[name]


def load_runtime():
    # Keep CPU control free of optional device plugins and their global patches.
    # PyTorch reads this before auto-importing torch.backends entry points.
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    # Set before imports. All models are constructed from inline configs.
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[name] = "1"
    import peft
    import torch
    import torch.distributed as dist
    import transformers
    from packaging.version import Version
    from safetensors.torch import load_file, save_file
    from torch.distributed.fsdp import FSDPModule, ShardingStrategy
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    for package, expected in (("torch", (2, 10)), ("transformers", (5, 5)), ("peft", (0, 19))):
        actual = importlib.metadata.version(package)
        print(f"VERSION {package}={actual}", flush=True)
        require(
            Version(actual).release[:2] == expected, f"Expected {package} {expected[0]}.{expected[1]}.x, got {actual}"
        )
    print(f"VERSION safetensors={importlib.metadata.version('safetensors')}", flush=True)
    print(f"VERSION python={sys.version.split()[0]} CUDA={torch.version.cuda} HIP={torch.version.hip}", flush=True)
    require(torch.version.cuda is None and torch.version.hip is None, "Use the dedicated CPU PyTorch build")
    require(not dist.is_initialized(), "This standalone smoke must own its process group")
    return SimpleNamespace(
        torch=torch,
        dist=dist,
        peft=peft,
        transformers=transformers,
        FSDP=FSDP,
        FSDPModule=FSDPModule,
        ShardingStrategy=ShardingStrategy,
        save_file=save_file,
        load_file=load_file,
    )


def configs(rt):
    transformers = rt.transformers
    qwen = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        use_cache=False,
    )
    qwen.architectures = ["Qwen2ForCausalLM"]
    text = transformers.Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        query_pre_attn_scalar=8,
        max_position_embeddings=64,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
        attention_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        use_cache=False,
    )
    vision = transformers.SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=16,
        patch_size=8,
        num_channels=3,
        attention_dropout=0.0,
    )
    gemma = transformers.Gemma3Config(
        text_config=text,
        vision_config=vision,
        mm_tokens_per_image=4,
        boi_token_index=61,
        eoi_token_index=62,
        image_token_index=63,
        tie_word_embeddings=False,
    )
    gemma.architectures = ["Gemma3ForConditionalGeneration"]
    return (("qwen2", qwen, ["q_proj", "v_proj"]), ("gemma3_text_only", gemma, "medgemma-language-linear"))


def make_trace(torch):
    from torch.utils._python_dispatch import TorchDispatchMode

    class AtenTrace(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.calls = Counter()
            self.schemas = {}
            self.sample = None

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            schema = func._schema
            key = schema.name + ("." + schema.overload_name if schema.overload_name else "")
            self.calls[key] += 1
            self.schemas[key] = str(schema)
            if self.sample is None:
                self.sample = next((value for value in args if isinstance(value, torch.Tensor)), None)
            return func(*args, **(kwargs or {}))

    return AtenTrace()


def dump_dispatch(trace, rt):
    require(bool(trace.schemas), "No real ATen dispatch was observed")
    require(any(key in trace.calls for key in ("aten::mm", "aten::addmm")), "No real matrix multiply observed")
    representatives = (
        "aten::embedding",
        "aten::zeros_like",
        "aten::mm",
        "aten::addmm",
        "aten::bmm",
        "aten::_log_softmax",
        "aten::nll_loss_forward",
        "aten::nll_loss_backward",
        "aten::_log_softmax_backward_data",
        "aten::copy_",
        "aten::add_.Tensor",
    )
    print(f"ATEN observed_unique={len(trace.schemas)} total_calls={sum(trace.calls.values())}", flush=True)
    keys_probe = getattr(rt.torch._C, "_dispatch_keys", None)
    if keys_probe is not None and trace.sample is not None:
        try:
            print(f"ATEN input_dispatch_keys={keys_probe(trace.sample)}", flush=True)
        except (RuntimeError, TypeError) as exc:
            print(f"ATEN dispatch-key introspection unavailable: {type(exc).__name__}: {exc}", flush=True)
    else:
        print("ATEN dispatch-key introspection unavailable in this build", flush=True)
    kernel_probe = getattr(rt.torch._C, "_dispatch_has_kernel_for_dispatch_key", None)
    for name in representatives:
        if name not in trace.schemas:
            continue
        print(f"ATEN calls={trace.calls[name]} schema={trace.schemas[name]}", flush=True)
        if kernel_probe is None:
            print("ATEN kernel-registration introspection unavailable in this build", flush=True)
            continue
        for dispatch_key in ("CPU", "AutogradCPU", "CompositeImplicitAutograd", "CompositeExplicitAutograd"):
            try:
                print(f"  registered[{dispatch_key}]={kernel_probe(name, dispatch_key)}", flush=True)
            except RuntimeError as exc:
                print(f"  registered[{dispatch_key}]=unavailable ({exc})", flush=True)
    print("ATEN registrations are read-only evidence, not a claim about DCU/PrivateUse1 routes.", flush=True)


def text_inputs(config, compatibility, rt):
    torch = rt.torch
    ids = torch.tensor([[3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3]], dtype=torch.long, device="cpu")
    extra = compatibility.get_text_forward_kwargs(config, ids)
    if config.model_type == "gemma3":
        require(set(extra) == {"token_type_ids"}, "Gemma3 text token types missing")
        types = extra["token_type_ids"]
        require(
            types.shape == ids.shape and types.dtype == ids.dtype and types.device == ids.device,
            "Gemma3 token types must preserve shape/dtype/device",
        )
        require(torch.count_nonzero(types).item() == 0, "Gemma3 text token types must be zero")
    else:
        require(extra == {}, "Qwen causal head received VLM-only kwargs")
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "position_ids": torch.arange(ids.shape[1], device="cpu").expand_as(ids).contiguous(),
        "use_cache": False,
        "return_dict": True,
        **extra,
    }


def checked_logits(model, inputs, rt):
    logits = model(**inputs).logits
    require(logits.shape == (*inputs["input_ids"].shape, 64), f"Unexpected logits shape: {tuple(logits.shape)}")
    require(logits.device.type == "cpu" and logits.dtype == rt.torch.float32, "Logits must be CPU float32")
    require(rt.torch.isfinite(logits).all().item(), "Non-finite logits")
    return logits


def compare_states(expected, actual, torch, stage):
    require(bool(expected) and set(expected) == set(actual), f"{stage}: empty or mismatched adapter keys")
    for name in expected:
        left, right = expected[name], actual[name]
        require(
            left.shape == right.shape and left.dtype == right.dtype, f"{stage}: tensor metadata mismatch for {name}"
        )
        require(torch.equal(left.cpu(), right.cpu()), f"{stage}: adapter values differ for {name}")


def export_and_reload(model, peft_model, base_state, config, inputs, reference, compatibility, adapters, rt):
    torch = rt.torch
    state = adapters.collect_lora_adapter_state_dict(model)
    require(bool(state), "Refusing to accept an empty adapter export")
    pointers = []
    for name, value in state.items():
        require(
            value.device.type == "cpu" and value.is_contiguous() and not value.requires_grad,
            f"Export tensor is not independent contiguous CPU data: {name}",
        )
        require(value.numel() > 0 and torch.isfinite(value).all().item(), f"Invalid exported tensor: {name}")
        pointers.append(value.data_ptr())
    require(len(set(pointers)) == len(pointers), "Exported tensors share storage")
    live = rt.peft.get_peft_model_state_dict(peft_model)
    compare_states(live, state, torch, "helper vs PEFT")
    for name in state:
        require(state[name].data_ptr() != live[name].data_ptr(), f"Helper returned aliased training data: {name}")
    serialized = adapters.run_lora_export_stage(
        "CPU smoke config serialization",
        lambda: adapters.serialize_lora_adapter_config(peft_model.peft_config["default"]),
    )
    target = peft_model.peft_config["default"].target_modules
    expected_target = target if isinstance(target, str) else sorted(target)
    require(serialized["target_modules"] == expected_target, "Target regex/list changed during serialization")
    require(serialized["peft_type"] == "LORA", "Export is not a LoRA config")
    require(serialized.get("base_model_name_or_path") in (None, ""), "Smoke adapter must not embed a model path")
    with tempfile.TemporaryDirectory(prefix="verl-cpu-adapter-") as directory:
        folder = Path(directory)

        def write_adapter():
            rt.save_file(state, str(folder / "adapter_model.safetensors"))
            (folder / "adapter_config.json").write_text(
                json.dumps(serialized, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
            )

        adapters.run_lora_export_stage("CPU smoke adapter filesystem write", write_adapter)
        disk_state = rt.load_file(str(folder / "adapter_model.safetensors"), device="cpu")
        compare_states(state, disk_state, torch, "safetensors reload")
        require(
            json.loads((folder / "adapter_config.json").read_text(encoding="utf-8")) == serialized,
            "JSON config round trip changed values",
        )
        restored_config = rt.peft.PeftConfig.from_pretrained(str(folder), local_files_only=True)
        restored_target = restored_config.target_modules
        restored_target = restored_target if isinstance(restored_target, str) else sorted(restored_target)
        require(restored_target == expected_target, "PEFT config reload changed target modules")
        fresh = compatibility.get_auto_model_class(config).from_config(
            copy.deepcopy(config), attn_implementation="eager"
        )
        fresh.to(device="cpu", dtype=torch.float32)
        fresh.load_state_dict(base_state, strict=True)
        reloaded = rt.peft.PeftModel.from_pretrained(fresh, str(folder), is_trainable=False, local_files_only=True)
        reloaded.eval()
        compare_states(state, rt.peft.get_peft_model_state_dict(reloaded), torch, "PEFT adapter reload")
        with torch.no_grad():
            actual = checked_logits(reloaded, inputs, rt)
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        max_error = (actual - reference).abs().max().item()
        print(
            f"EXPORT verified_tensors={len(state)} files=adapter_model.safetensors,adapter_config.json "
            f"PEFT_reload_max_logit_error={max_error:.9g}",
            flush=True,
        )


def run_case(label, config, targets, use_fsdp, seed, compatibility, adapters, rt):
    torch = rt.torch
    torch.manual_seed(seed)
    config = copy.deepcopy(config)
    auto_class = compatibility.get_auto_model_class(config)
    base = auto_class.from_config(config, attn_implementation="eager").to(device="cpu", dtype=torch.float32)
    expected_class = "Gemma3ForConditionalGeneration" if config.model_type == "gemma3" else "Qwen2ForCausalLM"
    require(type(base).__name__ == expected_class, f"Wrong model head selected: {type(base).__name__}")
    text_config = base.config.get_text_config()
    require(text_config._attn_implementation == "eager", "Text attention must be eager, not fused/compiled")
    base_state = {name: tensor.detach().cpu().clone() for name, tensor in base.state_dict().items()}
    resolved = compatibility.resolve_lora_target_modules(targets)
    if isinstance(targets, list):
        require(resolved is targets, "Generic target_modules should be unchanged")
    lora_config = rt.peft.LoraConfig(
        task_type=rt.peft.TaskType.CAUSAL_LM, r=2, lora_alpha=4, lora_dropout=0.0, bias="none", target_modules=resolved
    )
    peft_model = rt.peft.get_peft_model(base, lora_config)
    model = peft_model
    if use_fsdp:
        model = rt.FSDP(
            peft_model,
            sharding_strategy=rt.ShardingStrategy.NO_SHARD,
            use_orig_params=True,
            device_id=torch.device("cpu"),
            auto_wrap_policy=None,
            sync_module_states=False,
        )
        require(
            sum(isinstance(module, rt.FSDP) for module in model.modules()) == 1, "Expected a root-only FSDP1 wrapper"
        )
    mode = "fsdp1_root_cpu_no_shard" if use_fsdp else "plain_cpu"
    print(f"CASE {label}/{mode} auto_class={auto_class.__name__} model={expected_class} seed={seed}", flush=True)
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    frozen = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters() if not parameter.requires_grad
    }
    require(
        bool(trainable) and all("lora_" in name for name in trainable),
        "Expected only nonempty LoRA trainable parameters",
    )
    require(bool(frozen), "Base model was not frozen")
    if config.model_type == "gemma3":
        require(
            all("language_model.layers." in name and "vision_tower" not in name for name in trainable),
            "MedGemma language-linear target escaped the language layers",
        )
        require(not hasattr(config, "vocab_size"), "Gemma3 test must cover nested text_config vocabulary")
    before = {name: parameter.detach().clone() for name, parameter in trainable.items()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=0.01, weight_decay=0.0)
    trace = make_trace(torch)
    with trace:
        inputs = text_inputs(config, compatibility, rt)
        model.eval()
        with torch.no_grad():
            initial = checked_logits(model, inputs, rt).clone()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = checked_logits(model, inputs, rt)
        actual_vocab = logits.shape[-1]
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, actual_vocab),
            inputs["input_ids"][:, 1:].contiguous().reshape(-1),
        )
        require(loss.ndim == 0 and torch.isfinite(loss).item(), "Invalid training loss")
        loss.backward()
        nonzero_gradients = 0
        for name, parameter in trainable.items():
            require(parameter.grad is not None, f"Missing LoRA gradient: {name}")
            require(torch.isfinite(parameter.grad).all().item(), f"Non-finite LoRA gradient: {name}")
            nonzero_gradients += int(torch.count_nonzero(parameter.grad).item() > 0)
        require(nonzero_gradients > 0, "All adapter gradients are zero")
        optimizer.step()
        changed = [name for name, parameter in trainable.items() if not torch.equal(before[name], parameter.detach())]
        require(bool(changed), "Optimizer did not change any adapter parameter")
        for name, parameter in model.named_parameters():
            require(torch.isfinite(parameter).all().item(), f"Non-finite parameter after optimizer: {name}")
            if name in frozen:
                require(
                    parameter.grad is None and torch.equal(frozen[name], parameter.detach()),
                    f"Frozen base changed: {name}",
                )
        model.eval()
        with torch.no_grad():
            reference = checked_logits(model, inputs, rt).clone()
        output_delta = (reference - initial).abs().max().item()
        require(output_delta > 0, "Adapter update did not change outputs")
    print(
        f"TRAIN loss={loss.item():.8f} nonzero_gradients={nonzero_gradients} changed_parameters={len(changed)} "
        f"max_logit_delta={output_delta:.9g} actual_vocab={actual_vocab}",
        flush=True,
    )
    dump_dispatch(trace, rt)
    export_and_reload(model, peft_model, base_state, config, inputs, reference, compatibility, adapters, rt)
    print(
        "Bottom-layer execution analysis: CPU int64 inputs -> eager transformer + PEFT LoRA -> CPU float32 logits "
        "-> shifted cross_entropy(actual logits vocab) -> autograd -> AdamW -> production adapter collection "
        "-> CPU safetensors/config -> real PEFT reload; exact observed ATen schemas above. "
        "No compile/fusion, PrivateUse1/DCU routing or whole-trainer claim.",
        flush=True,
    )
    print(f"CASE_PASS {label}/{mode}", flush=True)
    del optimizer, model, peft_model, base
    gc.collect()


def fsdp_supported(rt):
    if not rt.dist.is_available() or not rt.dist.is_gloo_available():
        return False, "Gloo is not available in this PyTorch build"
    probe = None
    try:
        probe = rt.FSDP(
            rt.torch.nn.Linear(4, 4),
            sharding_strategy=rt.ShardingStrategy.NO_SHARD,
            use_orig_params=True,
            device_id=rt.torch.device("cpu"),
            auto_wrap_policy=None,
        )
    except RuntimeError as exc:
        # Only a known capability rejection may become UNSUPPORTED. Other errors
        # (and all actual model/train/export errors) must fail with a traceback.
        if "FSDP needs a non-CPU accelerator device" in str(exc):
            return False, str(exc)
        raise
    finally:
        del probe
    return True, "explicit device_id=cpu accepted"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--require-fsdp",
        action="store_true",
        help="Fail rather than return partial coverage if CPU FSDP is unavailable",
    )
    args = parser.parse_args()
    require(args.threads > 0, "--threads must be positive")
    rt = load_runtime()
    rt.torch.set_num_threads(args.threads)
    rt.torch.set_num_interop_threads(1)
    rt.torch.use_deterministic_algorithms(True)
    print("SCOPE: real CPU eager float32; offline tiny random models; NOT DCU or whole trainer", flush=True)
    with production_helpers(rt) as (compatibility, adapters):
        cases = configs(rt)
        for label, config, targets in cases:
            run_case(label, config, targets, False, args.seed, compatibility, adapters, rt)
        available = rt.dist.is_available() and rt.dist.is_gloo_available()
        with tempfile.TemporaryDirectory(prefix="verl-cpu-gloo-") as directory:
            try:
                if available:
                    rt.dist.init_process_group(
                        "gloo",
                        init_method=(Path(directory) / "rendezvous").as_uri(),
                        rank=0,
                        world_size=1,
                        timeout=timedelta(seconds=60),
                    )
                    require(
                        rt.dist.get_rank() == 0 and rt.dist.get_world_size() == 1, "Expected rank-zero single-rank Gloo"
                    )
                supported, reason = fsdp_supported(rt)
                print(f"FSDP_CPU supported={supported} reason={reason}", flush=True)
                if supported:
                    for label, config, targets in cases:
                        run_case(label, config, targets, True, args.seed, compatibility, adapters, rt)
                elif args.require_fsdp:
                    raise RuntimeError(f"Required CPU FSDP1 unavailable: {reason}")
            finally:
                gc.collect()
                if rt.dist.is_initialized():
                    rt.dist.destroy_process_group()
        if not supported:
            print("PARTIAL: 2 plain CPU cases passed; FSDP1 untested; exit=2", flush=True)
            return 2
    print("PASS: 4 real CPU cases; training + adapter values/config + PEFT output reload verified", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
