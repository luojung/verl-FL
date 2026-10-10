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
"""Execute selected production AST functions with explicit dependency doubles.

These tests cover integration decisions and exception flow, not numerical
correctness, distributed execution, or real Transformers/device behavior.
"""

import ast
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
TRAINER = "verl/trainer/fsdp_sft_trainer.py"
ACTOR = "verl/workers/actor/dp_actor.py"


class Config(dict):
    __getattr__ = dict.__getitem__


def load_functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    selected = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    for node in selected:
        node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)
    return namespace


@pytest.fixture
def text_kwargs(monkeypatch):
    transformers = ModuleType("transformers")
    torch = ModuleType("torch")
    torch.zeros_like = Mock(side_effect=lambda inputs: inputs.new_zeros(inputs.shape))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    spec = importlib.util.spec_from_file_location("isolated_sft_compat", ROOT / "verl/utils/model_compat.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper.get_text_forward_kwargs


@pytest.mark.parametrize(
    "device,expected",
    [
        ("cuda", {"num_workers": 8, "pin_memory": True, "pin_memory_device": "cuda"}),
        ("npu", {"num_workers": 8, "pin_memory": True, "pin_memory_device": "npu"}),
        ("flagos", {"num_workers": 0, "pin_memory": False}),
        ("privateuseone", {"num_workers": 0, "pin_memory": False}),
        ("cpu", {"num_workers": 8, "pin_memory": False}),
    ],
)
def test_dataloader_defaults(device, expected):
    ns = load_functions(TRAINER, {"_get_sft_dataloader_kwargs"}, {"logger": Mock()})
    assert ns["_get_sft_dataloader_kwargs"]({}, device) == expected


@pytest.mark.parametrize("device", ["flagos", "privateuseone", "cpu", "custom_accelerator"])
def test_dataloader_disables_explicit_unsupported_pinning(device):
    logger = Mock()
    ns = load_functions(TRAINER, {"_get_sft_dataloader_kwargs"}, {"logger": logger})
    assert ns["_get_sft_dataloader_kwargs"]({"pin_memory": True, "num_workers": 2}, device) == {
        "num_workers": 2,
        "pin_memory": False,
    }
    logger.warning.assert_called_once()


def test_dataloader_explicit_ordinary_override():
    ns = load_functions(TRAINER, {"_get_sft_dataloader_kwargs"}, {"logger": Mock()})
    assert ns["_get_sft_dataloader_kwargs"]({"pin_memory": False, "num_workers": 1}, "cuda") == {
        "num_workers": 1,
        "pin_memory": False,
    }


@pytest.mark.parametrize("device", ["flagos", "cuda"])
def test_train_and_validation_loaders_consume_same_policy(device):
    captured = []

    def loader(**kwargs):
        captured.append(kwargs)
        return object()

    ns = load_functions(
        TRAINER,
        {"_get_sft_dataloader_kwargs", "_build_dataloader"},
        {
            "logger": Mock(),
            "get_device_name": lambda: device,
            "DistributedSampler": Mock(),
            "StatefulDataLoader": loader,
        },
    )
    trainer = SimpleNamespace(
        config=Config(data=Config(train_batch_size=8, micro_batch_size_per_gpu=2), ulysses_sequence_parallel_size=1),
        device_mesh=Mock(get_rank=Mock(return_value=0), size=Mock(return_value=1)),
    )
    ns["_build_dataloader"](trainer, [1, 2], [3])
    assert len(captured) == 2
    for kwargs in captured:
        assert kwargs["pin_memory"] is (device == "cuda")
        assert kwargs["num_workers"] == (8 if device == "cuda" else 0)
        assert ("pin_memory_device" in kwargs) is (device == "cuda")
    assert [kwargs["batch_size"] for kwargs in captured] == [8, 2]


@pytest.mark.parametrize("failed_stage", [None, "collect", "synchronize", "empty_cache", "destroy"])
def test_cleanup_attempts_all_stages_and_surfaces_errors(failed_stage):
    calls = []
    failure = RuntimeError("teardown")

    def operation(name):
        def run():
            calls.append(name)
            if name == failed_stage:
                raise failure

        return run

    device = SimpleNamespace(synchronize=operation("synchronize"), empty_cache=operation("empty_cache"))
    ns = load_functions(
        TRAINER,
        {"_cleanup_sft_runtime"},
        {
            "gc": SimpleNamespace(collect=operation("collect")),
            "get_torch_device": lambda: device,
            "destroy_global_process_group": operation("destroy"),
            "logger": Mock(),
        },
    )
    if failed_stage:
        with pytest.raises(RuntimeError) as exc:
            ns["_cleanup_sft_runtime"]()
        assert exc.value is failure
    else:
        ns["_cleanup_sft_runtime"]()
    assert calls == ["collect", "synchronize", "empty_cache", "destroy", "collect"]


def test_cleanup_allows_cpu_module_without_optional_cache_methods():
    destroy = Mock()
    ns = load_functions(
        TRAINER,
        {"_cleanup_sft_runtime"},
        {
            "gc": SimpleNamespace(collect=lambda: None),
            "get_torch_device": lambda: object(),
            "destroy_global_process_group": destroy,
            "logger": Mock(),
        },
    )
    ns["_cleanup_sft_runtime"]()
    destroy.assert_called_once()


@pytest.mark.parametrize("failed_stage", [None, "initialize", "tokenizer", "dataset", "trainer", "fit"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_run_sft_always_cleans_and_preserves_primary_failure(monkeypatch, capsys, failed_stage, cleanup_fails):
    primary = RuntimeError("primary failure")
    cleanup_error = RuntimeError("cleanup failure")

    def stage(name, result):
        def run(*args, **kwargs):
            if failed_stage == name:
                raise primary
            return result

        return run

    utils = ModuleType("verl.utils")
    utils.hf_tokenizer = stage("tokenizer", object())
    monkeypatch.setitem(sys.modules, "verl.utils", utils)
    cleanup = Mock(side_effect=cleanup_error if cleanup_fails else None)
    trainer = SimpleNamespace(fit=stage("fit", None))
    ns = load_functions(
        TRAINER,
        {"run_sft"},
        {
            "get_device_name": lambda: "flagos",
            "initialize_global_process_group": stage("initialize", (0, 0, 1)),
            "init_device_mesh": Mock(return_value=object()),
            "copy_to_local": Mock(return_value="local-model"),
            "create_sft_dataset": stage("dataset", object()),
            "FSDPSFTTrainer": stage("trainer", trainer),
            "_cleanup_sft_runtime": cleanup,
        },
    )
    config = Config(
        ulysses_sequence_parallel_size=1,
        model=Config(partial_pretrain="model", trust_remote_code=False),
        data=Config(train_files="train", val_files="val"),
    )
    expected = primary if failed_stage else cleanup_error if cleanup_fails else None
    if expected:
        with pytest.raises(RuntimeError) as exc:
            ns["run_sft"](config)
        assert exc.value is expected
    else:
        ns["run_sft"](config)
    cleanup.assert_called_once()
    assert ("VERL_SFT_CLEAN_EXIT" in capsys.readouterr().out) is (expected is None)


def tensor_double(shape=(2, 4)):
    tensor = MagicMock(shape=shape, device="flagos:0")
    for name in ("to", "contiguous", "view", "reshape", "squeeze", "transpose", "unsqueeze"):
        getattr(tensor, name).return_value = tensor
    tensor.__getitem__.return_value = tensor
    tensor.size.return_value = shape[-1]
    tensor.dim.return_value = len(shape)
    return tensor


@pytest.mark.parametrize("model_type", ["gemma3", "qwen3", "gemma3_text"])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_sft_forward_kwargs_and_actual_logit_vocabulary(text_kwargs, model_type, sequence_parallel):
    inputs, packed, sliced = tensor_double(), tensor_double((1, 8)), tensor_double((1, 4))
    logits = tensor_double((2, 4, 17))  # Deliberately differs from config vocab_size.
    loss = MagicMock()
    for method in ("__mul__", "__truediv__", "__add__"):
        getattr(loss, method).return_value = loss
    loss_fct = Mock(return_value=loss)
    model = Mock(return_value=SimpleNamespace(logits=logits))
    fake_torch = SimpleNamespace(
        bfloat16="bfloat16",
        autocast=lambda **kwargs: nullcontext(),
        sum=Mock(return_value=loss),
        roll=Mock(return_value=packed),
    )
    ns = load_functions(
        TRAINER,
        {"_compute_loss_and_backward"},
        {
            "torch": fake_torch,
            "nn": SimpleNamespace(CrossEntropyLoss=Mock(return_value=loss_fct)),
            "nullcontext": nullcontext,
            "get_text_forward_kwargs": text_kwargs,
            "unpad_input": Mock(return_value=(packed, object(), None, None)),
            "index_first_axis": Mock(return_value=packed),
            "rearrange": Mock(return_value=packed),
            "ulysses_pad_and_slice_inputs": Mock(return_value=(sliced, sliced, 0)),
            "get_ulysses_sequence_parallel_world_size": lambda: 2,
            "gather_outputs_and_unpad": Mock(return_value=loss),
            "pad_input": Mock(return_value=loss),
        },
    )
    trainer = SimpleNamespace(
        use_remove_padding=sequence_parallel,
        device_name="flagos",
        config=Config(
            ulysses_sequence_parallel_size=2 if sequence_parallel else 1, data=Config(balance_dp_token=False)
        ),
        model_config=SimpleNamespace(model_type=model_type, vocab_size=999),
        fsdp_model=model,
        sharding_manager=nullcontext(),
    )
    mask = tensor_double()
    ns["_compute_loss_and_backward"](
        trainer, {"input_ids": inputs, "attention_mask": mask, "position_ids": mask, "loss_mask": mask}
    )
    kwargs = model.call_args.kwargs
    assert ("token_type_ids" in kwargs) is (model_type == "gemma3")
    actual_inputs = sliced if sequence_parallel else inputs
    assert kwargs["input_ids"] is actual_inputs
    if model_type == "gemma3":
        actual_inputs.new_zeros.assert_called_once_with(actual_inputs.shape)
    if not sequence_parallel:
        logits.view.assert_called_once_with(-1, 17)
    loss_fct.assert_called_once()
    loss.backward.assert_called_once()


@pytest.mark.parametrize("model_type", ["gemma3", "qwen3"])
@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_actor_kwargs_preserve_explicit_multimodal_token_types(monkeypatch, text_kwargs, model_type, multimodal, fused):
    explicit_types = object()
    model_utils = ModuleType("verl.utils.model")
    model_utils.extract_multi_modal_inputs = lambda value: {"token_type_ids": explicit_types, "pixel_values": object()}
    monkeypatch.setitem(sys.modules, "verl.utils.model", model_utils)
    logits = tensor_double((2, 4, 17))
    output = SimpleNamespace(logits=logits, log_probs=tensor_double(), entropy=tensor_double())
    model = Mock(return_value=output, config=SimpleNamespace(model_type=model_type))
    log_probs = object()
    ns = load_functions(
        ACTOR,
        {"_forward_micro_batch"},
        {
            "torch": SimpleNamespace(autocast=lambda **kwargs: nullcontext()),
            "get_text_forward_kwargs": text_kwargs,
            "logprobs_from_logits": Mock(return_value=log_probs),
        },
    )
    inputs = tensor_double()
    batch = {
        "responses": tensor_double((2, 2)),
        "input_ids": inputs,
        "attention_mask": tensor_double(),
        "position_ids": tensor_double(),
    }
    if multimodal:
        batch["multi_modal_inputs"] = [object()]
    actor = SimpleNamespace(
        actor_module=model,
        use_remove_padding=False,
        use_fused_kernels=fused,
        device_name="flagos",
        param_dtype="bfloat16",
    )
    ns["_forward_micro_batch"](actor, batch, 1.0)
    kwargs = model.call_args.kwargs
    assert ("token_type_ids" in kwargs) is (multimodal or model_type == "gemma3")
    if multimodal:
        assert kwargs["token_type_ids"] is explicit_types
    assert kwargs["use_cache"] is False
    assert ("temperature" in kwargs) is fused


def test_production_imports_are_local_and_no_hard_exit():
    paths = ("verl/__init__.py", "verl/utils/model.py", "verl/model_merger/base_model_merger.py", TRAINER, ACTOR)
    forbidden = {"AutoModelForVision2Seq", "AutoModelForImageTextToText"}
    for path in paths:
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "transformers":
                assert not (forbidden & {name.name for name in node.names}), path
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "_exit", path
    init = (ROOT / "verl/__init__.py").read_text(encoding="utf-8")
    assert "_transformers.AutoModel" not in init


def test_all_model_selection_entrypoints_delegate_to_helper():
    for path, name in (
        ("verl/utils/model.py", "get_hf_auto_model_class"),
        ("verl/utils/model.py", "load_valuehead_model"),
        ("verl/model_merger/base_model_merger.py", "get_transformers_auto_model_class"),
        (TRAINER, "_build_model_optimizer"),
    ):
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
        assert any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "get_auto_model_class"
            for node in ast.walk(function)
        )


@pytest.mark.parametrize("rows", [[], ["only-row"], ["first-row", "second-row"]])
@pytest.mark.parametrize("nested", [False, True])
def test_sft_dataset_keeps_singleton_row_axis_and_uses_positional_series_access(monkeypatch, rows, nested):
    class Series:
        def __init__(self, values):
            self.values = values
            self.iloc = self

        def __len__(self):
            return len(self.values)

        def __getitem__(self, key):
            return self.values[key]

        def tolist(self):
            return self.values

    class NamedSeries(Series):
        def __init__(self, value):
            self.values = [value]
            self.iloc = Series([value])

        def __getitem__(self, key):
            raise KeyError("integer key is a label, not a position")

    class DataFrame:
        def __init__(self, values):
            self.values = values

        def __len__(self):
            return len(self.values)

        def __getitem__(self, key):
            return DataFrame(self.values)

        def squeeze(self, axis=None):
            # Reproduce the historical one-row squeeze() -> scalar failure.
            if axis is None and len(self.values) == 1:
                return self.values[0]
            assert axis == 1
            return Series(self.values)

        def apply(self, function, axis):
            assert axis == 1
            return Series([function(NamedSeries(value)) for value in self.values])

    values = [{"text": value} for value in rows] if nested else rows
    frame = DataFrame(values)
    pandas = ModuleType("pandas")
    pandas.DataFrame = DataFrame
    pandas.core = SimpleNamespace(series=SimpleNamespace(Series=Series))
    pandas.read_parquet = lambda path: frame
    pandas.concat = lambda frames: frame
    numpy = ModuleType("numpy")
    numpy.ndarray = type("ndarray", (), {})
    monkeypatch.setitem(sys.modules, "pandas", pandas)
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    ns = load_functions("verl/utils/dataset/sft_dataset.py", {"_read_files_and_tokenize"}, {"pd": pandas})
    dataset = SimpleNamespace(
        parquet_files=["data"],
        max_samples=-1,
        prompt_key=["prompt"],
        response_key=["response"],
        prompt_dict_keys=["text"] if nested else [],
        response_dict_keys=["text"] if nested else [],
    )
    ns["_read_files_and_tokenize"](dataset)
    assert dataset.prompts == rows
    assert dataset.responses == rows


@pytest.mark.parametrize(
    "device,default_attention",
    [
        ("flagos", "eager"),
        ("privateuseone", "eager"),
        ("cpu", "eager"),
        ("cuda", "flash_attention_2"),
        ("npu", "flash_attention_2"),
    ],
)
@pytest.mark.parametrize("overrides", [{}, {"attn_implementation": "sdpa", "local_files_only": True}])
def test_sft_model_load_preserves_ordinary_defaults_and_explicit_overrides(device, default_attention, overrides):
    tree = ast.parse((ROOT / TRAINER).read_text(encoding="utf-8"))
    function = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_build_model_optimizer"
    )
    init_block = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.With)
        and isinstance(node.items[0].context_expr, ast.Call)
        and isinstance(node.items[0].context_expr.func, ast.Name)
        and node.items[0].context_expr.func.id == "init_context"
    )
    # Execute the actual selection/loading statements, excluding FSDP and PEFT setup.
    load_statements = []
    for node in init_block.body:
        load_statements.append(node)
        if isinstance(node, ast.AnnAssign):
            break
    model_class = SimpleNamespace(from_pretrained=Mock(return_value=object()))
    hf_config = object()
    selector = Mock(return_value=model_class)
    ns = {
        "self": SimpleNamespace(config=Config(model=Config(overrides))),
        "get_auto_model_class": selector,
        "get_device_name": lambda: device,
        "local_model_path": "local-model",
        "config": hf_config,
        "torch_dtype": "bfloat16",
        "trust_remote_code": False,
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *load_statements], type_ignores=[]))
    exec(compile(module, TRAINER, "exec"), ns)
    selector.assert_called_once_with(hf_config)
    kwargs = model_class.from_pretrained.call_args.kwargs
    assert kwargs["attn_implementation"] == overrides.get("attn_implementation", default_attention)
    assert kwargs["local_files_only"] is overrides.get("local_files_only", False)
    assert kwargs["config"] is hf_config
    assert kwargs["trust_remote_code"] is False


@pytest.mark.parametrize("model_type", ["gemma3", "qwen3"])
@pytest.mark.parametrize("fused", [False, True])
def test_actor_packed_text_uses_packed_token_type_shape(text_kwargs, model_type, fused):
    inputs, packed = tensor_double(), tensor_double((1, 7))
    logits = tensor_double((1, 7, 17))
    output = SimpleNamespace(logits=logits, log_probs=packed, entropy=packed)
    model = Mock(return_value=output, config=SimpleNamespace(model_type=model_type))
    mask = tensor_double()
    mask.sum.return_value = 7
    ns = load_functions(
        ACTOR,
        {"_forward_micro_batch"},
        {
            "torch": SimpleNamespace(autocast=lambda **kwargs: nullcontext(), roll=Mock(return_value=packed)),
            "get_text_forward_kwargs": text_kwargs,
            "logprobs_from_logits": Mock(return_value=packed),
            "unpad_input": Mock(return_value=(packed, object(), None, None)),
            "index_first_axis": Mock(return_value=packed),
            "rearrange": Mock(return_value=packed),
            "pad_input": Mock(return_value=tensor_double()),
        },
    )
    actor = SimpleNamespace(
        actor_module=model,
        use_remove_padding=True,
        use_ulysses_sp=False,
        use_fused_kernels=fused,
        device_name="flagos",
        param_dtype="bfloat16",
    )
    batch = {
        "responses": tensor_double((2, 2)),
        "input_ids": inputs,
        "attention_mask": mask,
        "position_ids": tensor_double(),
    }
    ns["_forward_micro_batch"](actor, batch, 1.0)
    kwargs = model.call_args.kwargs
    assert kwargs["input_ids"] is packed
    assert ("token_type_ids" in kwargs) is (model_type == "gemma3")
    if model_type == "gemma3":
        packed.new_zeros.assert_called_once_with((1, 7))
    assert kwargs["attention_mask"] is None
