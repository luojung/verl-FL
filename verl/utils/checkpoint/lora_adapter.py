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

"""PEFT adapter export helpers; collection is SPMD, filesystem writes are not.

Call collection on *all* model ranks, then write the resulting CPU tensors and
serialized config on rank zero. Do not wrap collection in a rank-zero guard.
"""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum


def serialize_lora_adapter_config(config) -> dict:
    """Return JSON-safe PEFT config without mutating it or splitting regex strings.

    Prefer PEFT's own to_dict(), with dataclass/mapping support for older callers.
    Sets become sorted lists; regex strings, booleans, numbers and None retain
    their JSON types. Unknown values fail rather than being silently stringified.
    """
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    elif is_dataclass(config) and not isinstance(config, type):
        config = asdict(config)
    if not isinstance(config, Mapping):
        raise TypeError("LoRA adapter config must be a mapping, dataclass, or expose to_dict()")

    def convert(value):
        if isinstance(value, Enum):
            return convert(value.value)
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, Mapping):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("LoRA adapter config keys must be strings")
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (set, frozenset)):
            return [convert(item) for item in sorted(value)]
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        raise TypeError(f"Unsupported LoRA adapter config value: {type(value).__name__}")

    return convert(config)


def run_lora_export_stage(stage: str, operation):
    """Run a zero-argument callback and agree on ordinary errors across all ranks.

    Every member of the default process group must call this helper in the same
    order. The callback executes on each rank: for rank-zero-only filesystem IO,
    put the rank guard *inside* the callback, not around this helper invocation.
    Returns the callback's rank-local result; results are not broadcast.

    In single-rank/uninitialized execution, the original exception propagates.
    In multi-rank execution, any callback error raises RuntimeError on all ranks
    with the stage and failing ranks identified, chained to the local exception
    where present. Successful peers wait for this agreement before returning.
    No rollback of partial filesystem writes is performed.

    This cannot recover a backend failure that leaves a peer stuck *inside* a
    collective. Set the current accelerator device before invoking this helper
    (also required by NCCL object gathers).
    """
    import torch.distributed as dist

    result, error = None, None
    try:
        result = operation()
    except Exception as exc:
        error = exc

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        errors = [None] * dist.get_world_size()
        message = None if error is None else f"{type(error).__name__}: {error}"
        dist.all_gather_object(errors, message)
        if any(message is not None for message in errors):
            details = "; ".join(f"rank {rank}: {message}" for rank, message in enumerate(errors) if message is not None)
            raise RuntimeError(f"LoRA export failed during {stage}: {details}") from error
    elif error is not None:
        raise error
    return result


def collect_lora_adapter_state_dict(model, adapter_name: str = "default") -> OrderedDict:
    """Collect a non-empty, independent CPU adapter state on every model rank.

    Uses FSDP1 FULL_STATE_DICT with rank0_only=False (including root-owned and
    nested parameters), or FSDP2's public distributed-checkpoint full-state API.
    PyTorch 2.10's NO_SHARD state_dict hooks replace the FSDP1 config with its
    defaults, ignoring requested CPU offload; the later CPU copy is still needed.
    FSDP2 cpu_offload=False is intentional: cpu_offload=True with a full state
    returns an empty state on nonzero ranks. All ranks must enter these APIs and
    any DTensor.full_tensor() calls before rank-zero-only filesystem writing.

    PEFT owns adapter selection, bias/modules_to_save/embedding inclusion and
    adapter-name removal. Keys are not rewritten with architecture prefixes.
    Clones prevent exported tensors from aliasing training state or each other
    (shared storage is not accepted by safetensors.save_file).

    This gathers the full base state before PEFT filtering, trading peak memory
    for correct public-API behavior. Call outside forward/backward. Exceptions
    propagate to the caller; this function neither logs success nor writes files.
    """
    from peft.utils.save_and_load import get_peft_model_state_dict
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from verl.utils.fsdp_utils import fsdp_version

    peft_model = getattr(model, "_fsdp_wrapped_module", model)

    def validate_adapter():
        if adapter_name not in getattr(peft_model, "peft_config", {}):
            raise ValueError(f"PEFT adapter {adapter_name!r} is not present on the model")
        # An unwrapped PEFT root may still contain sharded children. Detect those
        # too instead of treating their local state_dict as a complete export.
        versions = {fsdp_version(module) for module in model.modules()} - {0}
        if versions not in (set(), {1}, {2}):
            raise ValueError("Mixed FSDP1/FSDP2 models are not supported for LoRA export")
        return versions

    versions = run_lora_export_stage("adapter validation", validate_adapter)

    def full_state():
        if versions == {1}:
            config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
                return model.state_dict()
        if versions == {2}:
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            return get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=False))
        return model.state_dict()

    state = run_lora_export_stage("full-state collection", full_state)

    def filter_adapter():
        adapter_state = get_peft_model_state_dict(peft_model, state_dict=state, adapter_name=adapter_name)
        if not adapter_state:
            raise ValueError(f"Refusing to export empty LoRA adapter {adapter_name!r}")
        return adapter_state

    adapter_state = run_lora_export_stage("PEFT filtering", filter_adapter)
    state = None  # Release full base-model storage before cloning adapter tensors.

    def copy_adapter():
        result = OrderedDict()
        # Sort keys to keep collective order deterministic if any DTensors remain.
        for name, tensor in sorted(adapter_state.items()):
            if hasattr(tensor, "full_tensor"):
                tensor = tensor.full_tensor()
            result[name] = tensor.detach().cpu().contiguous().clone()
        # Check only after possible collective materialization has completed.
        for name, tensor in result.items():
            if tensor.numel() == 0:
                raise ValueError(f"Refusing to export empty LoRA adapter tensor {name!r}")
        return result

    return run_lora_export_stage("CPU tensor materialization", copy_adapter)
