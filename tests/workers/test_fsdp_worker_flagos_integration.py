# Copyright 2026 contributors. Licensed under the Apache License, Version 2.0.
"""Dependency-isolated worker lifecycle regressions (not device validation)."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "verl/workers/fsdp_workers.py"


class Device:
    def __init__(self):
        self.rng = "training"
        self.cache_calls = 0

    def get_rng_state(self):
        return self.rng

    def set_rng_state(self, state):
        self.rng = state

    def empty_cache(self):
        self.cache_calls += 1


class Model:
    def __init__(self):
        self.training = True
        self.child = SimpleNamespace(training=False)

    def modules(self):
        return iter((self, self.child))


def extract_method(name, namespace):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    method.decorator_list = [ast.Name(id="contextmanager", ctx=ast.Load())] if name == "_shared_hf_generation" else []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(SOURCE), "exec"), namespace)
    return namespace[name]


def lifecycle(offload=False, loader=None):
    device, model, calls = Device(), Model(), []
    worker = SimpleNamespace(actor_module_fsdp=model, gen_random_states="generation", _is_offload_param=offload)
    namespace = {
        "contextmanager": contextmanager,
        "get_torch_device": lambda: device,
        "load_fsdp_model_to_gpu": loader or (lambda _: calls.append("load")),
        "offload_fsdp_model_to_cpu": lambda _: calls.append("offload"),
    }
    worker._shared_hf_generation = MethodType(extract_method("_shared_hf_generation", namespace), worker)
    return worker, device, calls


@pytest.mark.parametrize("offload", [False, True])
def test_shared_hf_restores_rng_and_mixed_modes(offload):
    worker, device, calls = lifecycle(offload)
    with worker._shared_hf_generation():
        assert device.rng == "generation"
        device.rng = "next-generation"
        worker.actor_module_fsdp.training = False
        worker.actor_module_fsdp.child.training = True
    assert device.rng == "training"
    assert worker.gen_random_states == "next-generation"
    assert worker.actor_module_fsdp.training is True
    assert worker.actor_module_fsdp.child.training is False
    assert calls == (["load", "offload"] if offload else [])
    assert device.cache_calls == 1


def test_shared_hf_restores_state_when_generation_raises():
    worker, device, calls = lifecycle(True)
    with pytest.raises(ValueError, match="generation failed"):
        with worker._shared_hf_generation():
            device.rng = "after-sampling"
            worker.actor_module_fsdp.training = False
            raise ValueError("generation failed")
    assert device.rng == "training"
    assert worker.gen_random_states == "after-sampling"
    assert worker.actor_module_fsdp.training is True
    assert calls == ["load", "offload"]
    assert device.cache_calls == 1


def test_shared_hf_load_failure_does_not_offload_or_advance_generation_rng():
    def fail_load(_):
        raise RuntimeError("load failed")

    worker, device, calls = lifecycle(True, fail_load)
    with pytest.raises(RuntimeError, match="load failed"):
        with worker._shared_hf_generation():
            pytest.fail("unreachable")
    assert calls == []
    assert worker.gen_random_states == "generation"
    assert device.rng == "training"
    assert device.cache_calls == 1


@pytest.mark.parametrize("world_size,strategy", [(2, "fsdp"), (1, "fsdp2")])
def test_reviewed_flagos_hf_rejects_unvalidated_sharding(world_size, strategy):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    guard = next(
        node for node in init.body if isinstance(node, ast.If) and "Shared HF rollout on flagos" in ast.unparse(node)
    )
    namespace = {
        "device_name": "flagos",
        "world_size": world_size,
        "role": "actor_rollout",
        "self": SimpleNamespace(
            config=SimpleNamespace(rollout=SimpleNamespace(name="hf"), actor=SimpleNamespace(strategy=strategy))
        ),
    }
    with pytest.raises(NotImplementedError, match="single-card FSDP1"):
        exec(compile(ast.Module(body=[guard], type_ignores=[]), str(SOURCE), "exec"), namespace)
    namespace["device_name"] = "cuda"
    exec(compile(ast.Module(body=[guard], type_ignores=[]), str(SOURCE), "exec"), namespace)


@pytest.mark.parametrize("is_lora", [False, True])
def test_shared_hf_always_preserves_original_parameters(is_lora):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorRolloutRefWorker")
    build = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_build_model_optimizer"
    )
    guard = next(
        node for node in build.body if isinstance(node, ast.If) and "self.use_orig_params = True" in ast.unparse(node)
    )
    worker = SimpleNamespace(_is_lora=is_lora, use_orig_params=False)
    namespace = {"self": worker, "shared_hf_single_card": True}
    exec(compile(ast.Module(body=[guard], type_ignores=[]), str(SOURCE), "exec"), namespace)
    assert worker.use_orig_params is True
    worker.use_orig_params = False
    namespace["shared_hf_single_card"] = False
    exec(compile(ast.Module(body=[guard], type_ignores=[]), str(SOURCE), "exec"), namespace)
    assert worker.use_orig_params is False
