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
"""Dependency-isolated compatibility tests, not device/model runtime validation."""

import ast
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def load_compat(monkeypatch):
    def load(vlm_names=("AutoModelForImageTextToText", "AutoModelForVision2Seq")):
        transformers = ModuleType("transformers")
        torch = ModuleType("torch")
        torch.zeros_like = Mock(side_effect=lambda inputs: inputs.new_zeros(inputs.shape))
        monkeypatch.setitem(sys.modules, "torch", torch)
        names = (
            "AutoModel",
            "AutoModelForCausalLM",
            "AutoModelForTokenClassification",
            "AutoModelForSequenceClassification",
            *vlm_names,
        )
        for name in names:
            setattr(transformers, name, type(name, (), {"_model_mapping": {}}))
        monkeypatch.setitem(sys.modules, "transformers", transformers)
        spec = importlib.util.spec_from_file_location("isolated_model_compat", ROOT / "verl/utils/model_compat.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, transformers

    return load


@pytest.mark.parametrize(
    "exports",
    [
        ("AutoModelForVision2Seq",),
        ("AutoModelForImageTextToText",),
        ("AutoModelForImageTextToText", "AutoModelForVision2Seq"),
    ],
)
def test_medgemma_class_across_export_generations(load_compat, exports):
    helper, transformers = load_compat(exports)
    config = SimpleNamespace(architectures=["Gemma3ForConditionalGeneration"], model_type="gemma3")
    original = set(vars(transformers))
    assert helper.get_auto_model_class(config) is getattr(transformers, exports[0])
    assert set(vars(transformers)) == original  # No global alias or mutation.


@pytest.mark.parametrize("exports", [(), ("AutoModelForVision2Seq",), ("AutoModelForImageTextToText",)])
@pytest.mark.parametrize("architecture", ["Qwen2ForCausalLM", "Qwen3ForCausalLM", "Gemma3ForCausalLM"])
def test_causal_models_do_not_require_vlm_export(load_compat, exports, architecture):
    helper, transformers = load_compat(exports)
    config = SimpleNamespace(architectures=[architecture])
    assert helper.get_auto_model_class(config) is transformers.AutoModelForCausalLM


def test_registered_legacy_class_beats_unregistered_modern_class(load_compat):
    helper, transformers = load_compat()
    config_type = type("OldVLMConfig", (), {})
    transformers.AutoModelForVision2Seq._model_mapping[config_type] = object()
    assert helper.get_auto_model_class(config_type()) is transformers.AutoModelForVision2Seq


@pytest.mark.parametrize(
    "head", ["AutoModelForCausalLM", "AutoModelForTokenClassification", "AutoModelForSequenceClassification"]
)
def test_architecture_heads(load_compat, head):
    helper, transformers = load_compat()
    config = SimpleNamespace(architectures=["Custom" + head.removeprefix("AutoModel")])
    assert helper.get_auto_model_class(config) is getattr(transformers, head)


@pytest.mark.parametrize("remote_name", ["AutoModelForVision2Seq", "AutoModelForImageTextToText"])
@pytest.mark.parametrize("exports", [("AutoModelForVision2Seq",), ("AutoModelForImageTextToText",)])
def test_remote_vlm_auto_maps_with_available_exports(load_compat, remote_name, exports):
    helper, transformers = load_compat(exports)
    config = SimpleNamespace(architectures=["CustomVLM"], auto_map={remote_name: [None, "repo--modeling.CustomVLM"]})
    assert helper.get_auto_model_class(config) is getattr(transformers, exports[0])


def test_remote_map_prefers_task_head_and_matches_exact_class(load_compat):
    helper, transformers = load_compat()
    config = SimpleNamespace(
        architectures=["RemoteForCausalLM"],
        auto_map={
            "AutoModel": "modeling.RemoteForCausalLM",
            "AutoModelForImageTextToText": "modeling.NotRemoteForCausalLM",
            "AutoModelForCausalLM": "modeling.RemoteForCausalLM",
        },
    )
    assert helper.get_auto_model_class(config) is transformers.AutoModelForCausalLM


def test_remote_map_without_architectures(load_compat):
    helper, transformers = load_compat()
    config = SimpleNamespace(architectures=None, auto_map={"AutoModelForCausalLM": "modeling.RemoteLM"})
    assert helper.get_auto_model_class(config) is transformers.AutoModelForCausalLM


@pytest.mark.parametrize("architectures", [None, [], "Qwen3ForCausalLM"])
def test_missing_or_scalar_architectures(load_compat, architectures):
    helper, transformers = load_compat()
    config = SimpleNamespace(architectures=architectures)
    expected = transformers.AutoModelForCausalLM if architectures else transformers.AutoModel
    assert helper.get_auto_model_class(config) is expected


def test_native_causal_mapping_without_architectures(load_compat):
    helper, transformers = load_compat()
    config_type = type("QwenConfig", (), {})
    transformers.AutoModelForCausalLM._model_mapping[config_type] = object()
    assert helper.get_auto_model_class(config_type()) is transformers.AutoModelForCausalLM


def test_missing_vlm_export_fails_locally_without_mutation(load_compat):
    helper, transformers = load_compat(())
    config = SimpleNamespace(architectures=["Gemma3ForConditionalGeneration"])
    with pytest.raises(ImportError, match="no supported vision-language"):
        helper.get_auto_model_class(config)
    assert not hasattr(transformers, "AutoModelForVision2Seq")


@pytest.mark.parametrize("value", [None, "all-linear", "q_proj", ["q_proj", "v_proj"], ("q_proj",), {"q_proj"}])
def test_generic_lora_targets_unchanged(load_compat, value):
    helper, _ = load_compat()
    assert helper.resolve_lora_target_modules(value) is value


def test_medgemma_lora_targets_only_language_linear(load_compat):
    helper, _ = load_compat()
    target = helper.resolve_lora_target_modules("medgemma-language-linear")
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        assert re.fullmatch(target, f"model.language_model.layers.0.self_attn.{name}")
    for name in (
        "model.vision_tower.layers.0.q_proj",
        "lm_head",
        "model.language_model.embed_tokens",
        "model.language_model.layers.0.input_layernorm",
    ):
        assert not re.fullmatch(target, name)


class TensorMetadata:
    def __init__(self, shape=(2, 5), dtype="int64", device="flagos:0", fill=7):
        self.shape, self.dtype, self.device, self.fill = shape, dtype, device, fill

    def new_zeros(self, shape):
        return TensorMetadata(shape, self.dtype, self.device, fill=0)


@pytest.mark.parametrize("shape", [(2, 5), (1, 13)])
def test_gemma3_text_kwargs_preserve_tensor_metadata(load_compat, shape):
    helper, _ = load_compat()
    inputs = TensorMetadata(shape=shape)
    kwargs = helper.get_text_forward_kwargs(SimpleNamespace(model_type="gemma3"), inputs)
    token_types = kwargs["token_type_ids"]
    assert (token_types.shape, token_types.dtype, token_types.device, token_types.fill) == (
        shape,
        "int64",
        "flagos:0",
        0,
    )
    assert inputs.fill == 7
    sys.modules["torch"].zeros_like.assert_called_once_with(inputs)


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3", "gemma3_text", "llama", None])
def test_other_models_do_not_receive_vlm_only_kwargs(load_compat, model_type):
    helper, _ = load_compat()
    assert helper.get_text_forward_kwargs(SimpleNamespace(model_type=model_type), object()) == {}
    assert helper.get_text_forward_kwargs(None, object()) == {}


def extract_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)
    return namespace[name]


def test_native_vlm_mapping_precedes_causal_architecture_suffix(load_compat):
    helper, transformers = load_compat()
    config_type = type("FuyuConfig", (), {"architectures": ["FuyuForCausalLM"]})
    transformers.AutoModelForImageTextToText._model_mapping[config_type] = object()
    assert helper.get_auto_model_class(config_type()) is transformers.AutoModelForImageTextToText


@pytest.mark.parametrize("name", ["AutoModelForVision2Seq", "AutoModelForImageTextToText"])
def test_remote_vlm_keeps_available_explicit_auto_map_key(load_compat, name):
    helper, transformers = load_compat()
    config = SimpleNamespace(architectures=["CustomVLM"], auto_map={name: "modeling.CustomVLM"})
    assert helper.get_auto_model_class(config) is getattr(transformers, name)


def test_generic_model_entrypoint_delegates(load_compat):
    helper, transformers = load_compat()
    function = extract_function(
        "verl/utils/model.py",
        "get_hf_auto_model_class",
        {
            "get_auto_model_class": helper.get_auto_model_class,
        },
    )
    assert function(SimpleNamespace(architectures=["Qwen3ForCausalLM"])) is transformers.AutoModelForCausalLM


@pytest.mark.parametrize("exports", [("AutoModelForVision2Seq",), ("AutoModelForImageTextToText",)])
def test_merger_supports_available_vlm_head(load_compat, exports):
    helper, transformers = load_compat(exports)
    function = extract_function(
        "verl/model_merger/base_model_merger.py",
        "get_transformers_auto_model_class",
        {
            "get_auto_model_class": helper.get_auto_model_class,
            "AutoModel": transformers.AutoModel,
        },
    )
    merger = SimpleNamespace(model_config=SimpleNamespace(architectures=["Gemma3ForConditionalGeneration"]))
    assert function(merger) is getattr(transformers, exports[0])
    with pytest.raises(NotImplementedError, match="Unknown architecture"):
        function(SimpleNamespace(model_config=SimpleNamespace(architectures=None)))


@pytest.mark.parametrize("architecture", ["Qwen3ForCausalLM", "Gemma3ForConditionalGeneration"])
def test_valuehead_fallback_uses_shared_selector(monkeypatch, load_compat, architecture):
    helper, transformers = load_compat()
    transformers.AutoModelForTokenClassification.from_pretrained = Mock(side_effect=ValueError("unsupported config"))
    config = SimpleNamespace(architectures=[architecture])
    expected_class = helper.get_auto_model_class(config)
    base = object()
    expected_class.from_pretrained = Mock(return_value=base)
    adapter = object()
    trl = ModuleType("trl")
    trl.AutoModelForCausalLMWithValueHead = SimpleNamespace(from_pretrained=Mock(return_value=adapter))
    monkeypatch.setitem(sys.modules, "trl", trl)
    patch = Mock()
    function = extract_function(
        "verl/utils/model.py",
        "load_valuehead_model",
        {
            "AutoModelForTokenClassification": transformers.AutoModelForTokenClassification,
            "is_trl_available": lambda: True,
            "get_auto_model_class": helper.get_auto_model_class,
            "patch_valuehead_model": patch,
        },
    )
    assert function("local-model", "bfloat16", config, False) is adapter
    assert expected_class.from_pretrained.call_args.kwargs["config"] is config
    trl.AutoModelForCausalLMWithValueHead.from_pretrained.assert_called_once_with(base)
    patch.assert_called_once_with(adapter)


@pytest.mark.parametrize(
    "error", [OSError("load failure"), RuntimeError("device failure"), KeyboardInterrupt(), SystemExit(2)]
)
def test_valuehead_does_not_swallow_runtime_errors_or_interrupts(load_compat, error):
    _, transformers = load_compat()
    transformers.AutoModelForTokenClassification.from_pretrained = Mock(side_effect=error)
    selector = Mock()
    function = extract_function(
        "verl/utils/model.py",
        "load_valuehead_model",
        {
            "AutoModelForTokenClassification": transformers.AutoModelForTokenClassification,
            "is_trl_available": lambda: True,
            "get_auto_model_class": selector,
        },
    )
    with pytest.raises(type(error)) as exc:
        function("local-model", "bfloat16", object(), False)
    assert exc.value is error
    selector.assert_not_called()
