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

"""Dependency-isolated export contract tests; no real torch/PEFT/device execution.

Load the pure helper by file and inject scoped doubles for its optional runtime
imports. The FSDP compatibility entry points are extracted verbatim via AST so
this suite also runs in the pytest/ruff-only review environment.
"""

import ast
import importlib.util
import json
import sys
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def helper():
    spec = importlib.util.spec_from_file_location(
        "isolated_lora_adapter", ROOT / "verl/utils/checkpoint/lora_adapter.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TensorDouble:
    def __init__(self, values=(1, 2)):
        self.values = list(values)
        self.calls = []

    def numel(self):
        return len(self.values)

    def detach(self):
        self.calls.append("detach")
        return self

    def cpu(self):
        self.calls.append("cpu")
        return self

    def contiguous(self):
        self.calls.append("contiguous")
        return self

    def clone(self):
        self.calls.append("clone")
        return TensorDouble(self.values)


class DTensorDouble(TensorDouble):
    def full_tensor(self):
        self.calls.append("full_tensor")
        return self


class ModelDouble:
    def __init__(self, state, version=0, children=(), adapters=("default",)):
        self.state = state
        self.version = version
        self.children = children
        self.peft_config = {name: {} for name in adapters}
        self.state_calls = 0

    def modules(self):
        yield self
        for child in self.children:
            yield from child.modules()

    def state_dict(self):
        self.state_calls += 1
        return self.state


@pytest.fixture
def runtime(monkeypatch):
    events = []
    dist = ModuleType("torch.distributed")
    dist.is_available = lambda: True
    dist.is_initialized = lambda: False
    dist.get_world_size = lambda: 2
    dist.get_rank = Mock(side_effect=AssertionError("collection must not branch on rank"))
    dist.all_gather_object = Mock()

    @contextmanager
    def state_dict_type(model, state_type, config):
        events.append(("enter", model, state_type, config))
        yield
        events.append(("exit", model))

    fsdp = ModuleType("torch.distributed.fsdp")
    fsdp.FullyShardedDataParallel = SimpleNamespace(state_dict_type=state_dict_type)
    fsdp.FullStateDictConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    fsdp.StateDictType = SimpleNamespace(FULL_STATE_DICT="full")

    dcp = ModuleType("torch.distributed.checkpoint.state_dict")
    dcp.StateDictOptions = lambda **kwargs: SimpleNamespace(**kwargs)

    def get_model_state_dict(model, options):
        events.append(("fsdp2", model, options))
        return model.state_dict()

    dcp.get_model_state_dict = get_model_state_dict
    fsdp_utils = ModuleType("verl.utils.fsdp_utils")
    fsdp_utils.fsdp_version = lambda model: model.version
    peft = ModuleType("peft.utils.save_and_load")
    peft.get_peft_model_state_dict = Mock()
    torch = ModuleType("torch")
    torch.distributed = dist
    packages = {
        "torch": torch,
        "torch.distributed": dist,
        "torch.distributed.fsdp": fsdp,
        "torch.distributed.checkpoint": ModuleType("torch.distributed.checkpoint"),
        "torch.distributed.checkpoint.state_dict": dcp,
        "peft": ModuleType("peft"),
        "peft.utils": ModuleType("peft.utils"),
        "peft.utils.save_and_load": peft,
        "verl.utils.fsdp_utils": fsdp_utils,
    }
    for name, module in packages.items():
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(events=events, dist=dist, peft=peft, fsdp_utils=fsdp_utils)


def export_case(runtime, *, version=0, children=(), wrapped=False):
    raw = OrderedDict(
        [
            ("base_model.model.language_model.q_proj.lora_A.default.weight", TensorDouble()),
            ("base_model.model.language_model.layers.0.q_proj.lora_B.default.weight", TensorDouble((3, 4))),
            ("base_model.model.vision_tower.saved.modules_to_save.default.weight", TensorDouble((5,))),
            ("base_model.model.embedding.weight", TensorDouble((6,))),
            ("base_model.model.language_model.q_proj.base_layer.weight", TensorDouble((7,))),
        ]
    )
    # PEFT owns these semantics; the collector must return these keys unchanged.
    exported = OrderedDict(
        [
            ("base_model.model.language_model.q_proj.lora_A.weight", raw[next(iter(raw))]),
            ("base_model.model.language_model.layers.0.q_proj.lora_B.weight", raw[list(raw)[1]]),
            ("base_model.model.vision_tower.saved.weight", raw[list(raw)[2]]),
            ("base_model.model.embedding.weight", raw[list(raw)[3]]),
        ]
    )
    runtime.peft.get_peft_model_state_dict.return_value = exported
    model = ModelDouble(raw, version=version, children=children)
    peft_model = ModelDouble(raw)
    if wrapped:
        model._fsdp_wrapped_module = peft_model
    else:
        peft_model = model
    return model, peft_model, exported


@pytest.mark.parametrize("nested", [False, True])
def test_root_and_nested_fsdp1_public_full_state(helper, runtime, nested):
    children = (ModelDouble({}, version=1),) if nested else ()
    model, peft_model, expected = export_case(runtime, version=1, children=children, wrapped=True)
    result = helper.collect_lora_adapter_state_dict(model)
    assert isinstance(result, OrderedDict)
    assert set(result) == set(expected)
    assert "base_model.model.language_model.q_proj.lora_A.weight" in result  # root-owned adapter
    assert "base_model.model.language_model.layers.0.q_proj.lora_B.weight" in result
    assert model.state_calls == 1
    assert len(runtime.events) == 2  # a single public full-state traversal, not prefix scans
    enter, exit_ = runtime.events
    assert enter[:3] == ("enter", model, "full")
    assert enter[3].offload_to_cpu is True
    assert enter[3].rank0_only is False
    assert exit_ == ("exit", model)
    runtime.peft.get_peft_model_state_dict.assert_called_once_with(
        peft_model, state_dict=model.state, adapter_name="default"
    )
    for key, tensor in expected.items():
        assert result[key] is not tensor
        assert result[key].values == tensor.values
        assert tensor.calls == ["detach", "cpu", "contiguous", "clone"]
    assert not hasattr(model, "_is_root")
    runtime.dist.get_rank.assert_not_called()


def test_unwrapped_peft_root_with_nested_fsdp1(helper, runtime):
    model, _, expected = export_case(runtime, children=(ModelDouble({}, version=1),))
    assert set(helper.collect_lora_adapter_state_dict(model)) == set(expected)
    assert runtime.events[0][1] is model


@pytest.mark.parametrize("nested", [False, True])
def test_fsdp2_uses_all_rank_full_state_without_cpu_offload(helper, runtime, nested):
    model, _, expected = export_case(runtime, version=2, children=(ModelDouble({}, version=2),) if nested else ())
    assert set(helper.collect_lora_adapter_state_dict(model)) == set(expected)
    assert len(runtime.events) == 1
    assert runtime.events[0][0] == "fsdp2"
    options = runtime.events[0][2]
    assert options.full_state_dict is True
    assert options.cpu_offload is False
    runtime.dist.get_rank.assert_not_called()


def test_plain_peft_export_and_named_adapter_delegate_keys_to_peft(helper, runtime):
    model, _, expected = export_case(runtime)
    model.peft_config = {"medical": {}}
    result = helper.collect_lora_adapter_state_dict(model, adapter_name="medical")
    assert set(result) == set(expected)
    runtime.peft.get_peft_model_state_dict.assert_called_once_with(
        model, state_dict=model.state, adapter_name="medical"
    )
    assert runtime.events == []


def test_cpu_clones_break_shared_storage(helper, runtime):
    shared = TensorDouble()
    runtime.peft.get_peft_model_state_dict.return_value = {"lora_A.weight": shared, "lora_B.weight": shared}
    result = helper.collect_lora_adapter_state_dict(ModelDouble({}))
    assert result["lora_A.weight"] is not result["lora_B.weight"]
    result["lora_A.weight"].values[0] = 99
    assert shared.values == [1, 2]
    assert result["lora_B.weight"].values == [1, 2]


def test_remaining_dtensors_materialized_on_all_ranks(helper, runtime):
    tensor = DTensorDouble()
    runtime.peft.get_peft_model_state_dict.return_value = {"lora_A.weight": tensor}
    result = helper.collect_lora_adapter_state_dict(ModelDouble({}))
    assert tensor.calls == ["full_tensor", "detach", "cpu", "contiguous", "clone"]
    assert result["lora_A.weight"].values == tensor.values
    runtime.dist.get_rank.assert_not_called()


def test_empty_export_rejected(helper, runtime):
    runtime.peft.get_peft_model_state_dict.return_value = {}
    with pytest.raises(ValueError, match="empty LoRA adapter"):
        helper.collect_lora_adapter_state_dict(ModelDouble({}))


def test_empty_tensor_rejected_after_remaining_dtensors_gathered(helper, runtime):
    remaining = DTensorDouble()
    runtime.peft.get_peft_model_state_dict.return_value = {
        "a_empty": TensorDouble(()),
        "z_nonempty": remaining,
    }
    with pytest.raises(ValueError, match="empty LoRA adapter tensor 'a_empty'"):
        helper.collect_lora_adapter_state_dict(ModelDouble({}))
    assert remaining.calls[0] == "full_tensor"


def test_missing_adapter_rejected_before_full_state(helper, runtime):
    model = ModelDouble({}, version=1, adapters=("other",))
    with pytest.raises(ValueError, match="not present"):
        helper.collect_lora_adapter_state_dict(model)
    assert runtime.events == []
    assert model.state_calls == 0


def test_mixed_fsdp_versions_fail_explicitly(helper, runtime):
    model = ModelDouble({}, version=1, children=(ModelDouble({}, version=2),))
    with pytest.raises(ValueError, match="Mixed FSDP1/FSDP2"):
        helper.collect_lora_adapter_state_dict(model)


def test_state_collection_error_propagates(helper, runtime):
    model = ModelDouble({}, version=1)
    model.state_dict = Mock(side_effect=RuntimeError("all-gather failed"))
    with pytest.raises(RuntimeError, match="all-gather failed"):
        helper.collect_lora_adapter_state_dict(model)
    runtime.peft.get_peft_model_state_dict.assert_not_called()


def test_successful_multi_rank_collection_enters_every_stage(helper, runtime):
    runtime.dist.is_initialized = lambda: True
    runtime.dist.get_rank = lambda: 1  # non-writer participates fully
    messages = []

    def gather(output, message):
        messages.append(message)
        output[:] = [message, message]

    runtime.dist.all_gather_object.side_effect = gather
    model, _, _ = export_case(runtime, version=1, wrapped=True)
    assert helper.collect_lora_adapter_state_dict(model)
    assert len(messages) == 4
    assert messages == [None] * 4
    assert runtime.events[0][3].rank0_only is False


def test_remote_rank_validation_error_stops_before_full_state(helper, runtime):
    runtime.dist.is_initialized = lambda: True

    def gather(output, message):
        output[:] = [message, "ValueError: adapter missing on peer"]

    runtime.dist.all_gather_object.side_effect = gather
    model, _, _ = export_case(runtime, version=1)
    with pytest.raises(RuntimeError, match="adapter validation.*rank 1:.*adapter missing"):
        helper.collect_lora_adapter_state_dict(model)
    assert model.state_calls == 0
    runtime.peft.get_peft_model_state_dict.assert_not_called()


def test_empty_export_error_coherent_across_ranks(helper, runtime):
    runtime.dist.is_initialized = lambda: True

    def gather(output, message):
        output[:] = [message, message]

    runtime.dist.all_gather_object.side_effect = gather
    runtime.peft.get_peft_model_state_dict.return_value = {}
    with pytest.raises(RuntimeError, match="PEFT filtering.*empty LoRA adapter"):
        helper.collect_lora_adapter_state_dict(ModelDouble({}))
    assert runtime.dist.all_gather_object.call_count == 3


@pytest.mark.parametrize("initialized", [False, True])
def test_public_stage_single_rank_returns_value_and_preserves_original_error(helper, runtime, initialized):
    runtime.dist.is_initialized = lambda: initialized
    runtime.dist.get_world_size = lambda: 1
    value = object()
    assert helper.run_lora_export_stage("configuration", lambda: value) is value
    error = ValueError("invalid config")

    def fail():
        raise error

    with pytest.raises(ValueError, match="invalid config") as raised:
        helper.run_lora_export_stage("configuration", fail)
    assert raised.value is error
    runtime.dist.all_gather_object.assert_not_called()


@pytest.mark.parametrize("rank", [0, 1])
def test_public_stage_rank0_io_guard_is_inside_callback(helper, runtime, rank):
    runtime.dist.is_initialized = lambda: True
    runtime.dist.get_rank = lambda: rank
    writer = Mock(return_value="adapter_model.safetensors")
    events = []

    def write_on_rank0():
        events.append("operation")
        if runtime.dist.get_rank() == 0:
            return writer()
        return None

    def gather(output, message):
        events.append("agreement")
        output[:] = [None, None]
        assert message is None

    runtime.dist.all_gather_object.side_effect = gather
    result = helper.run_lora_export_stage("adapter write", write_on_rank0)
    assert result == ("adapter_model.safetensors" if rank == 0 else None)
    assert writer.call_count == (1 if rank == 0 else 0)
    assert events == ["operation", "agreement"]
    runtime.dist.all_gather_object.assert_called_once()


@pytest.mark.parametrize("rank", [0, 1])
def test_public_stage_rank0_write_failure_raises_on_writer_and_peer(helper, runtime, rank):
    runtime.dist.is_initialized = lambda: True
    runtime.dist.get_rank = lambda: rank
    error = OSError("disk full")
    writer = Mock(side_effect=error)

    def write_on_rank0():
        if runtime.dist.get_rank() == 0:
            writer()

    def gather(output, message):
        output[:] = ["OSError: disk full", None]
        assert message == ("OSError: disk full" if rank == 0 else None)

    runtime.dist.all_gather_object.side_effect = gather
    with pytest.raises(RuntimeError, match="adapter write: rank 0: OSError: disk full") as raised:
        helper.run_lora_export_stage("adapter write", write_on_rank0)
    assert raised.value.__cause__ is (error if rank == 0 else None)
    assert writer.call_count == (1 if rank == 0 else 0)


def test_public_stage_config_serialization_failure_is_coherent(helper, runtime):
    runtime.dist.is_initialized = lambda: True

    def gather(output, message):
        assert message.startswith("TypeError:")
        output[:] = [message, message]

    runtime.dist.all_gather_object.side_effect = gather
    with pytest.raises(RuntimeError, match="config serialization:.*Unsupported LoRA adapter config value"):
        helper.run_lora_export_stage(
            "config serialization", lambda: helper.serialize_lora_adapter_config({"target_modules": object()})
        )


class ConfigType(Enum):
    LORA = "LORA"


@dataclass
class ConfigDouble:
    target_modules: object
    peft_type: ConfigType = ConfigType.LORA
    task_type: str = "CAUSAL_LM"
    inference_mode: bool = False
    r: int = 8
    exclude_modules: object = None


@pytest.mark.parametrize("target", [r".*language_model.*\.(q_proj|v_proj)$", "all-linear"])
def test_config_preserves_regex_and_special_string(helper, target):
    config = ConfigDouble(target)
    result = helper.serialize_lora_adapter_config(config)
    assert result["target_modules"] == target
    assert isinstance(result["target_modules"], str)
    assert result["peft_type"] == "LORA"
    assert result["task_type"] == "CAUSAL_LM"
    assert result["inference_mode"] is False
    assert result["r"] == 8
    assert result["exclude_modules"] is None
    assert json.loads(json.dumps(result)) == result
    assert config.peft_type is ConfigType.LORA


def test_config_serializes_sets_without_mutating_and_preserves_nested_types(helper):
    targets = {"v_proj", "q_proj"}
    config = {
        "target_modules": targets,
        "exclude_modules": frozenset({"head", "vision"}),
        "nested": {"alpha": 0.5, "enabled": True, "targets": ("a", "b")},
    }
    result = helper.serialize_lora_adapter_config(config)
    assert result["target_modules"] == ["q_proj", "v_proj"]
    assert result["exclude_modules"] == ["head", "vision"]
    assert result["nested"] == {"alpha": 0.5, "enabled": True, "targets": ["a", "b"]}
    assert config["target_modules"] is targets
    assert isinstance(config["exclude_modules"], frozenset)
    assert json.loads(json.dumps(result)) == result


def test_config_prefers_peft_to_dict(helper):
    config = SimpleNamespace(to_dict=Mock(return_value={"target_modules": "q.*", "peft_type": ConfigType.LORA}))
    assert helper.serialize_lora_adapter_config(config) == {"target_modules": "q.*", "peft_type": "LORA"}
    config.to_dict.assert_called_once_with()


@pytest.mark.parametrize("value", [object(), {"target_modules": object()}, {1: "bad key"}])
def test_invalid_config_fails_instead_of_stringifying(helper, value):
    with pytest.raises(TypeError):
        helper.serialize_lora_adapter_config(value)


def extract_fsdp_functions(*names, namespace):
    tree = ast.parse((ROOT / "verl/utils/fsdp_utils.py").read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == set(names)
    exec(compile(ast.Module(body=functions, type_ignores=[]), "fsdp_utils.py", "exec"), namespace)
    return namespace


def test_layered_compatibility_entrypoint_uses_new_collector(helper, runtime, monkeypatch):
    monkeypatch.setitem(sys.modules, "verl.utils.checkpoint.lora_adapter", helper)
    model, _, expected = export_case(runtime, version=1)
    functions = extract_fsdp_functions("layered_summon_lora_params", namespace={"OrderedDict": OrderedDict})
    assert set(functions["layered_summon_lora_params"](model)) == set(expected)


def test_vllm_collect_lora_params_layered_contract(helper, runtime, monkeypatch):
    monkeypatch.setitem(sys.modules, "verl.utils.checkpoint.lora_adapter", helper)
    model, _, expected = export_case(runtime, version=1)
    functions = extract_fsdp_functions(
        "layered_summon_lora_params",
        "collect_lora_params",
        namespace={"OrderedDict": OrderedDict, "FSDP": object, "fsdp_version": lambda m: m.version},
    )
    assert set(functions["collect_lora_params"](model, layered_summon=True, base_sync_done=True)) == set(expected)
    with pytest.raises(ValueError, match="base-model is preloaded"):
        functions["collect_lora_params"](model, layered_summon=True, base_sync_done=False)


def test_fsdp_version_handles_torch_without_fsdp2():
    class FSDPDouble:
        pass

    functions = extract_fsdp_functions("fsdp_version", namespace={"FSDP": FSDPDouble, "FSDPModule": None})
    assert functions["fsdp_version"](FSDPDouble()) == 1
    assert functions["fsdp_version"](object()) == 0
