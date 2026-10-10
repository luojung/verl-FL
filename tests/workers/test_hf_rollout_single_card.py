"""Dependency-isolated HF regressions; no accelerator/Ray/Transformers required.

Execute the production class AST with small tensor/module doubles. These tests
verify control flow, token bookkeeping, and lifecycle, not real PyTorch/FSDP.
"""

import ast
import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    __getattr__ = dict.__getitem__


class Tensor:
    def __init__(self, rows, device="cpu", dtype="long"):
        self.rows = [list(row) for row in rows]
        self.device = SimpleNamespace(type=str(device).split(":")[0], name=str(device))
        self.dtype = dtype

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)

    ndim = 2

    def size(self, dim):
        return self.shape[dim]

    def to(self, device):
        return Tensor(self.rows, getattr(device, "name", device), self.dtype)

    def __getitem__(self, key):
        row, col = key
        rows = self.rows[row] if isinstance(row, slice) else [self.rows[row]]
        return Tensor([r[col] if isinstance(col, slice) else [r[col]] for r in rows], self.device.name, self.dtype)

    def __setitem__(self, key, value):
        row, col = key
        indices = list(range(len(self.rows)))[row] if isinstance(row, slice) else [row]
        for i, index in enumerate(indices):
            if isinstance(value, Tensor):
                self.rows[index][col] = value.rows[i]
            elif isinstance(col, slice):
                selected = list(range(len(self.rows[index])))[col]
                for j in selected:
                    self.rows[index][j] = value
            else:
                self.rows[index][col] = value

    def _binary(self, other, fn):
        out = []
        for i, row in enumerate(self.rows):
            rhs = other.rows[i % len(other.rows)] if isinstance(other, Tensor) else [other]
            width = max(len(row), len(rhs))
            out.append([fn(row[j % len(row)], rhs[j % len(rhs)]) for j in range(width)])
        return Tensor(out, self.device.name, self.dtype)

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def __eq__(self, other):
        return self._binary(other, lambda a, b: a == b)

    def __lt__(self, other):
        return self._binary(other, lambda a, b: a < b)

    def __or__(self, other):
        return self._binary(other, lambda a, b: a or b)

    def unsqueeze(self, dim):
        return self


class TensorDict(dict):
    def __init__(self, source, batch_size):
        super().__init__(source)
        self.batch_size = (batch_size,)


class Proto:
    def __init__(self, batch=None, non_tensor_batch=None, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}

    def split(self, size):
        out = []
        for start in range(0, self.batch.batch_size[0], size):
            stop = min(start + size, self.batch.batch_size[0])
            out.append(
                Proto(
                    TensorDict(
                        {k: Tensor(v.rows[start:stop], v.device.name, v.dtype) for k, v in self.batch.items()},
                        stop - start,
                    ),
                    {k: v[start:stop] for k, v in self.non_tensor_batch.items()},
                    self.meta_info.copy(),
                )
            )
        return out

    @staticmethod
    def concat(parts):
        return Proto(
            TensorDict(
                {k: Tensor([row for p in parts for row in p.batch[k].rows]) for k in parts[0].batch},
                sum(p.batch.batch_size[0] for p in parts),
            ),
            meta_info=parts[0].meta_info.copy(),
        )


class Module:
    def __init__(self, children=(), training=True, error=False, device="cpu"):
        self.children = list(children)
        self.training = training
        self.error = error
        self.device = device
        self.calls = []

    def modules(self):
        yield self
        for child in self.children:
            yield from child.modules()

    def parameters(self):
        yield SimpleNamespace(device=self.device)

    def train(self, value=True):
        for child in self.modules():
            child.training = value
        return self

    def eval(self):
        return self.train(False)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        assert not self.training
        if self.error:
            raise RuntimeError("generation failed")
        return SimpleNamespace(sequences=Tensor([r + [7, 2] for r in kwargs["input_ids"].rows]))


class FakeFSDP(Module):
    def __init__(self, module, world_size=1, strategy="NO_SHARD", use_orig_params=True):
        super().__init__([module])
        self.module = module
        self.process_group = SimpleNamespace(size=lambda: world_size)
        self.sharding_strategy = strategy
        self._use_orig_params = use_orig_params
        self.compute_device = "cpu"

    def generate(self, **kwargs):
        raise AssertionError("FSDP delegated generate must not be used")


def positions(mask):
    result = []
    for row in mask.rows:
        total = 0
        new = []
        for value in row:
            total += value
            new.append(max(total - 1, 0))
        result.append(new)
    return Tensor(result)


def response_mask(response, eos_token, dtype):
    eos = eos_token if isinstance(eos_token, (list, tuple)) else [eos_token]
    result = []
    for row in response.rows:
        active = True
        new = []
        for token in row:
            new.append(int(active))
            if token in eos:
                active = False
        result.append(new)
    return Tensor(result, dtype=dtype)


@pytest.fixture
def hf_class():
    source = (ROOT / "verl/workers/rollout/hf_rollout.py").read_text(encoding="utf-8")
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "HFRollout")
    torch = SimpleNamespace(
        device=lambda d: d,
        long="long",
        bfloat16="bfloat16",
        float16="float16",
        no_grad=lambda: lambda fn: fn,
        full=lambda shape, fill, device="cpu", dtype="long": Tensor(
            [[fill] * shape[1] for _ in range(shape[0])], getattr(device, "name", device), dtype
        ),
        tensor=lambda values, **kw: Tensor([values], kw.get("device", "cpu")),
        zeros_like=lambda t: Tensor([[0] * t.size(1) for _ in t.rows], t.device.name),
        ones_like=lambda t, **kw: Tensor([[1] * t.size(1) for _ in t.rows], t.device.name),
        all=lambda t: all(v for row in t.rows for v in row),
        any=lambda t: any(v for row in t.rows for v in row),
        arange=lambda start, end, **kw: Tensor([list(range(start, end))], kw.get("device", "cpu")),
        cat=lambda ts, dim: Tensor([sum([t.rows[i] for t in ts], []) for i in range(ts[0].size(0))]),
        autocast=lambda **kw: contextlib.nullcontext(),
    )
    namespace = dict(
        torch=torch,
        nn=SimpleNamespace(Module=Module),
        FSDP=FakeFSDP,
        ShardingStrategy=SimpleNamespace(NO_SHARD="NO_SHARD"),
        GenerationConfig=lambda **kw: Config(kw),
        DataProto=Proto,
        TensorDict=TensorDict,
        BaseRollout=object,
        contextlib=contextlib,
        compute_position_id_with_mask=positions,
        get_response_mask=response_mask,
    )
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), str(ROOT / "verl/workers/rollout/hf_rollout.py"), "exec"),
        namespace,
    )
    return namespace["HFRollout"]


@pytest.fixture
def config():
    return Config(
        prompt_length=8,
        response_length=5,
        micro_batch_size=2,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=-1,
        dtype="bfloat16",
        val_kwargs=Config(do_sample=False, temperature=0.3, top_p=0.8, top_k=3),
    )


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0
    padding_side = "right"
    truncation_side = "right"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, chat, **kwargs):
        self.calls.append((chat, kwargs))
        return chat[0]["tokens"]


def raw(rows, **meta):
    return Proto(
        non_tensor_batch={"raw_prompt": [[{"role": "user", "content": "text", "tokens": r}] for r in rows]},
        meta_info=meta,
    )


def token_batch(rows, masks=None, **meta):
    return Proto(
        TensorDict(
            {"input_ids": Tensor(rows), "attention_mask": Tensor(masks or [[1] * len(r) for r in rows])}, len(rows)
        ),
        meta_info=meta,
    )


def test_raw_prompt_tokens_left_padding_and_metadata(hf_class, config):
    tokenizer = Tokenizer()
    rollout = hf_class(Module(), config, tokenizer)
    original = raw([[4, 5], [6]], hf_chat_template_kwargs={"enable_thinking": False}, global_steps=17)
    batch = rollout._materialize_raw_prompts(original)
    assert batch.batch["input_ids"].rows == [[4, 5], [0, 6]]
    assert batch.batch["attention_mask"].rows == [[1, 1], [0, 1]]
    assert batch.batch["position_ids"].rows == [[0, 1], [0, 0]]
    assert tokenizer.calls[0][1] == dict(
        enable_thinking=False, tokenize=True, add_generation_prompt=True, padding=False, truncation=False
    )
    assert tokenizer.padding_side == tokenizer.truncation_side == "right"
    assert original.batch is None
    assert batch.meta_info == original.meta_info
    assert batch.meta_info is not original.meta_info


@pytest.mark.parametrize("policy,expected", [("left", [3, 4, 5]), ("right", [1, 2, 3]), ("middle", [1, 4, 5])])
def test_explicit_truncation_policies(hf_class, config, policy, expected):
    config["prompt_length"] = 3
    batch = hf_class(Module(), config, Tokenizer())._materialize_raw_prompts(
        raw([[1, 2, 3, 4, 5]], hf_prompt_truncation=policy)
    )
    assert batch.batch["input_ids"].rows == [expected]


def test_overlong_prompt_not_silently_truncated(hf_class, config):
    with pytest.raises(ValueError, match="exceeding"):
        hf_class(Module(), config, Tokenizer())._materialize_raw_prompts(raw([list(range(9))]))


@pytest.mark.parametrize("masks", [[[1, 0]], [[0, 0]], [[1, 2]]])
def test_invalid_attention_masks_rejected(hf_class, config, masks):
    with pytest.raises(ValueError, match="attention masks"):
        hf_class(Module(), config)._materialize_raw_prompts(token_batch([[4, 0]], masks))


def test_mask_holes_rejected(hf_class, config):
    with pytest.raises(ValueError, match="holes"):
        hf_class(Module(), config)._materialize_raw_prompts(token_batch([[4, 0, 6]], [[1, 0, 1]]))


def test_response_length_metadata_padding_mask_and_positions(hf_class, config):
    module = Module()
    output = hf_class(module, config, Tokenizer()).generate_sequences(raw([[4, 5]], response_length=3, global_steps=9))
    settings = module.calls[0]["generation_config"]
    assert settings["max_new_tokens"] == 3
    assert settings["use_cache"] is True
    assert settings["output_scores"] is False
    assert settings["return_dict_in_generate"] is True
    assert set(module.calls[0]) == {"input_ids", "attention_mask", "generation_config"}
    assert "position_ids" not in module.calls[0]  # HF extends positions from attention_mask each decode step.
    assert output.batch["responses"].rows == [[7, 2, 0]]
    assert output.batch["response_mask"].rows == [[1, 1, 0]]
    assert output.batch["attention_mask"].rows == [[1, 1, 1, 1, 0]]
    assert output.batch["position_ids"].rows == [[0, 1, 2, 3, 4]]
    assert output.meta_info["global_steps"] == 9


def test_nondivisible_microbatch_is_bounded(hf_class, config):
    module = Module()
    output = hf_class(module, config, Tokenizer()).generate_sequences(raw([[i] for i in range(5)]))
    assert [call["input_ids"].size(0) for call in module.calls] == [2, 2, 1]
    assert output.batch.batch_size == (5,)


@pytest.mark.parametrize("error", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_prior_and_mixed_modes_restored_even_on_failure(hf_class, config, error, training):
    child = Module(training=not training)
    module = Module([child], training=training, error=error)
    rollout = hf_class(module, config, Tokenizer())
    if error:
        with pytest.raises(RuntimeError, match="generation failed"):
            rollout.generate_sequences(raw([[4]]))
    else:
        rollout.generate_sequences(raw([[4]]))
    assert module.training == training
    assert child.training == (not training)


def test_root_only_no_shard_uses_live_wrapped_actor(hf_class, config):
    actor = Module()
    wrapper = FakeFSDP(actor)
    rollout = hf_class(wrapper, config, Tokenizer())
    assert rollout.module is wrapper
    assert rollout._generation_module() is actor
    rollout.generate_sequences(raw([[4]]))
    actor.calls.clear()  # Same actor object after an optimizer update, no state_dict copy.
    rollout.generate_sequences(raw([[5]]))
    assert actor.calls[0]["input_ids"].rows == [[5]]


@pytest.mark.parametrize(
    "world_size,strategy,nested", [(2, "NO_SHARD", False), (1, "FULL_SHARD", False), (1, "NO_SHARD", True)]
)
def test_unsafe_fsdp_topologies_rejected(hf_class, config, world_size, strategy, nested):
    actor = FakeFSDP(Module()) if nested else Module()
    with pytest.raises(NotImplementedError, match="root-only"):
        hf_class(FakeFSDP(actor, world_size, strategy), config, Tokenizer())


def test_no_private_fsdp_reset_or_full_parameter_generation_context():
    source = (ROOT / "verl/workers/rollout/hf_rollout.py").read_text(encoding="utf-8")
    attrs = {n.attr for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Attribute)}
    assert "_reset_lazy_init" not in attrs
    assert "summon_full_params" not in attrs


def test_external_weight_sync_not_silently_ignored(hf_class, config):
    with pytest.raises(NotImplementedError, match="live actor weights"):
        asyncio.run(hf_class(Module(), config).update_weights(iter(())))


def test_validation_defaults_and_metadata_overrides(hf_class, config):
    module = Module()
    rollout = hf_class(module, config, Tokenizer())
    rollout.generate_sequences(raw([[4]], validate=True))
    assert module.calls[-1]["generation_config"]["do_sample"] is False
    rollout.generate_sequences(raw([[4]], validate=True, do_sample=True, temperature=0.4, top_k=-1))
    settings = module.calls[-1]["generation_config"]
    assert settings["temperature"] == 0.4
    assert settings["top_p"] == 0.8
    assert settings["top_k"] == 0
    assert settings["num_return_sequences"] == 1


def test_pad_zero_and_multiple_eos_are_preserved(hf_class, config):
    rollout = hf_class(Module(), config, Tokenizer())
    assert rollout._token_ids(raw([[4]], eos_token_id=[2, 3], pad_token_id=0)) == ([2, 3], 0)


def test_multimodal_chat_rejected(hf_class, config):
    prompts = raw([[4]])
    prompts.non_tensor_batch["raw_prompt"][0][0]["content"] = [{"type": "image", "image": "x"}]
    with pytest.raises(NotImplementedError, match="image/video/audio"):
        hf_class(Module(), config, Tokenizer()).generate_sequences(prompts)


def test_missing_attention_mask_is_not_inferred_from_pad_token(hf_class, config):
    prompts = Proto(TensorDict({"input_ids": Tensor([[2, 4]])}, 1))
    with pytest.raises(ValueError, match="explicit attention_mask"):
        hf_class(Module(), config, Tokenizer()).generate_sequences(prompts)


@pytest.mark.parametrize("chat", ["not a chat", {"role": "user"}, ["not a message"]])
def test_invalid_raw_chat_shapes_raise_useful_error(hf_class, config, chat):
    prompts = Proto(non_tensor_batch={"raw_prompt": [chat]})
    with pytest.raises(ValueError, match="raw_prompt"):
        hf_class(Module(), config, Tokenizer()).generate_sequences(prompts)


def test_encoder_decoder_model_rejected(hf_class, config):
    module = Module()
    module.config = SimpleNamespace(is_encoder_decoder=True)
    with pytest.raises(NotImplementedError, match="decoder-only"):
        hf_class(module, config, Tokenizer())


def test_padding_without_eos_is_masked(hf_class, config):
    tokenizer = Tokenizer()
    tokenizer.eos_token_id = None
    output = hf_class(Module(), config, tokenizer).generate_sequences(raw([[4]], response_length=4))
    assert output.batch["response_mask"].rows == [[1, 1, 0, 0]]


def test_zero_microbatch_rejected(hf_class, config):
    config["micro_batch_size"] = 0
    with pytest.raises(ValueError, match="micro_batch_size"):
        hf_class(Module(), config, Tokenizer()).generate_sequences(raw([[4]]))


def test_zero_response_length_rejected(hf_class, config):
    with pytest.raises(ValueError, match="response_length"):
        hf_class(Module(), config, Tokenizer()).generate_sequences(raw([[4]], response_length=0))


def test_raw_prompt_replaces_stale_positions_without_dropping_other_tensors(hf_class, config):
    prompts = raw([[4, 5]])
    prompts.batch = TensorDict({"position_ids": Tensor([[99, 99]]), "extra": Tensor([[8]])}, 1)
    output = hf_class(Module(), config, Tokenizer())._materialize_raw_prompts(prompts)
    assert output.batch["position_ids"].rows == [[0, 1]]
    assert output.batch["extra"].rows == [[8]]
    assert prompts.batch["position_ids"].rows == [[99, 99]]


def test_inputs_follow_actor_device_not_default_accelerator(hf_class, config):
    rollout = hf_class(Module(device="flagos:3"), config, Tokenizer())
    output = rollout._materialize_raw_prompts(token_batch([[4, 5]]))
    assert output.batch["input_ids"].device.name == "flagos:3"
    assert output.batch["attention_mask"].device.name == "flagos:3"
    assert output.batch["position_ids"].device.name == "flagos:3"


def test_missing_pad_falls_back_to_first_eos(hf_class, config):
    tokenizer = Tokenizer()
    tokenizer.pad_token_id = None
    tokenizer.eos_token_id = [2, 3]
    rollout = hf_class(Module(), config, tokenizer)
    assert rollout._token_ids(raw([[4]])) == ([2, 3], 2)


@pytest.mark.parametrize("kwargs", [{"return_tensors": "pt"}, {"return_dict": True}])
def test_nonflat_chat_template_output_settings_rejected(hf_class, config, kwargs):
    with pytest.raises(ValueError, match="flat token list"):
        hf_class(Module(), config, Tokenizer()).generate_sequences(raw([[4]], hf_chat_template_kwargs=kwargs))


def test_empty_eos_list_and_ignore_eos_are_explicit(hf_class, config):
    module = Module()
    rollout = hf_class(module, config, Tokenizer())
    assert rollout._token_ids(raw([[4]], eos_token_id=[], pad_token_id=0)) == (None, 0)
    config["ignore_eos"] = True
    output = rollout.generate_sequences(raw([[4]], response_length=3))
    assert module.calls[0]["generation_config"]["eos_token_id"] is None
    assert output.batch["response_mask"].rows == [[1, 1, 0]]


def test_chat_template_cannot_silently_truncate(hf_class, config):
    tokenizer = Tokenizer()
    rollout = hf_class(Module(), config, tokenizer)
    rollout.generate_sequences(raw([[4]], hf_chat_template_kwargs={"padding": True, "truncation": True}))
    assert tokenizer.calls[0][1]["padding"] is False
    assert tokenizer.calls[0][1]["truncation"] is False


def test_fsdp_nonoriginal_parameter_views_are_rejected(hf_class, config):
    wrapper = FakeFSDP(Module(), use_orig_params=False)
    with pytest.raises(NotImplementedError, match="requires FSDP use_orig_params=True"):
        hf_class(wrapper, config, Tokenizer())
    assert wrapper._use_orig_params is False  # Validation must not mutate FSDP internals.


def test_fsdp_missing_original_parameter_flag_fails_closed(hf_class, config):
    wrapper = FakeFSDP(Module())
    del wrapper._use_orig_params
    with pytest.raises(NotImplementedError, match="requires FSDP use_orig_params=True"):
        hf_class(wrapper, config, Tokenizer())
