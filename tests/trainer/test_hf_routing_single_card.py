"""Run production routing methods in isolation from Ray and accelerator imports."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    __getattr__ = dict.__getitem__


ROLE = SimpleNamespace(ActorRollout="actor_rollout", ActorRolloutRef="actor_rollout_ref", RewardModel="reward_model")


def method(path, class_name, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    # Strip annotation-only dependencies while executing the real implementation.
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace[name]


def module_stub(monkeypatch, name, **attrs):
    module = ModuleType(name)
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def workers(monkeypatch):
    classes = SimpleNamespace(
        sync=type("SyncFSDP", (), {}),
        async_fsdp=type("AsyncFSDP", (), {}),
        async_megatron=type("AsyncMegatron", (), {}),
        engine=type("Engine", (), {}),
        group=type("WorkerGroup", (), {}),
    )
    module_stub(monkeypatch, "verl.single_controller.ray", RayWorkerGroup=classes.group)
    module_stub(monkeypatch, "verl.trainer.ppo.ray_trainer", Role=ROLE)
    module_stub(
        monkeypatch,
        "verl.workers.fsdp_workers",
        ActorRolloutRefWorker=classes.sync,
        AsyncActorRolloutRefWorker=classes.async_fsdp,
    )
    module_stub(monkeypatch, "verl.workers.megatron_workers", AsyncActorRolloutRefWorker=classes.async_megatron)
    module_stub(monkeypatch, "verl.workers.engine_workers", ActorRolloutRefWorker=classes.engine)
    return classes


def config(name="hf", strategy="fsdp", impl="auto", reward_loop=False):
    return Config(
        actor_rollout_ref=Config(rollout=Config(name=name), actor=Config(strategy=strategy, use_kl_loss=False)),
        trainer=Config(use_legacy_worker_impl=impl),
        algorithm=Config(use_kl_in_reward=False),
        reward_model=Config(use_reward_loop=reward_loop, enable=False, enable_resource_pool=False),
        data=Config(truncation="left", apply_chat_template_kwargs={"enable_thinking": False}),
    )


def run_worker_selection(cfg):
    remote = Mock(side_effect=lambda cls: cls)
    fn = method(
        "verl/trainer/main_ppo.py", "TaskRunner", "add_actor_rollout_worker", {"ray": SimpleNamespace(remote=remote)}
    )
    runner = SimpleNamespace(role_worker_mapping={}, mapping={})
    selected = fn(runner, cfg)
    return selected, runner, remote


def test_hf_uses_sync_actor_worker(workers):
    (cls, group), runner, remote = run_worker_selection(config())
    assert cls is workers.sync
    assert group is workers.group
    assert runner.role_worker_mapping[ROLE.ActorRollout] is workers.sync
    assert runner.mapping[ROLE.ActorRollout] == "global_pool"
    remote.assert_called_once_with(workers.sync)


@pytest.mark.parametrize("name", ["vllm", "sglang"])
@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2", "megatron"])
def test_native_async_worker_selection_unchanged(workers, name, strategy):
    (cls, _), _, _ = run_worker_selection(config(name, strategy))
    expected = workers.async_megatron if strategy == "megatron" else workers.async_fsdp
    assert cls is expected


def test_native_new_engine_worker_unchanged(workers):
    (cls, _), _, _ = run_worker_selection(config("vllm", impl="disable"))
    assert cls is workers.engine


@pytest.mark.parametrize("strategy", ["fsdp2", "megatron"])
def test_hf_unsupported_worker_is_rejected_before_ray_creation(workers, strategy):
    with pytest.raises(NotImplementedError, match="legacy FSDP1"):
        run_worker_selection(config(strategy=strategy))


def test_hf_new_engine_rejected(workers):
    with pytest.raises(ValueError, match="use_legacy_worker_impl"):
        run_worker_selection(config(impl="disable"))


def test_hf_async_reward_loop_rejected(workers):
    with pytest.raises(ValueError, match="use_reward_loop=false"):
        run_worker_selection(config(reward_loop=True))


@pytest.mark.parametrize("name", ["hf", "vllm", "sglang"])
@pytest.mark.parametrize("custom", [False, True])
def test_rollout_manager_only_constructed_for_async(monkeypatch, name, custom):
    manager = Mock(return_value=object())
    loader = Mock(return_value=manager)
    module_stub(monkeypatch, "verl.experimental.agent_loop", AgentLoopManager=manager)
    cfg = config(name)
    cfg.actor_rollout_ref.rollout["agent"] = {"agent_loop_manager_class": "custom.Manager"} if custom else {}
    trainer = SimpleNamespace(config=cfg, actor_rollout_wg=object(), resource_pool_manager=Mock())
    fn = method(
        "verl/trainer/ppo/ray_trainer.py",
        "RayPPOTrainer",
        "_init_rollout_manager",
        {"load_class_from_fqn": loader, "Role": ROLE},
    )
    fn(trainer)
    assert trainer.async_rollout_mode == (name != "hf")
    if name == "hf":
        assert trainer.async_rollout_manager is None
        manager.assert_not_called()
        loader.assert_not_called()
    else:
        manager.assert_called_once_with(config=cfg, worker_group=trainer.actor_rollout_wg, rm_resource_pool=None)
        if custom:
            loader.assert_called_once_with("custom.Manager", "AgentLoopManager")
        else:
            loader.assert_not_called()


def test_async_reward_resource_pool_preserved(monkeypatch):
    manager = Mock()
    module_stub(monkeypatch, "verl.experimental.agent_loop", AgentLoopManager=manager)
    cfg = config("vllm")
    cfg.reward_model.update(enable=True, enable_resource_pool=True)
    pool = object()
    trainer = SimpleNamespace(
        config=cfg,
        actor_rollout_wg=object(),
        resource_pool_manager=SimpleNamespace(get_resource_pool=Mock(return_value=pool)),
    )
    fn = method("verl/trainer/ppo/ray_trainer.py", "RayPPOTrainer", "_init_rollout_manager", {"Role": ROLE})
    fn(trainer)
    assert manager.call_args.kwargs["rm_resource_pool"] is pool
    trainer.resource_pool_manager.get_resource_pool.assert_called_once_with(ROLE.RewardModel)


class Proto:
    def __init__(self):
        self.batch = {"input_ids": [1], "attention_mask": [1], "position_ids": [0], "scores": [9]}
        self.non_tensor_batch = {"raw_prompt": ["chat"], "uid": ["u"], "data_source": ["source"]}
        self.meta_info = {"global_steps": 7}
        self.pop_args = None

    def pop(self, batch_keys, non_tensor_batch_keys):
        self.pop_args = (batch_keys, non_tensor_batch_keys)
        out = SimpleNamespace(
            batch={key: self.batch.pop(key) for key in batch_keys},
            non_tensor_batch={key: self.non_tensor_batch.pop(key) for key in non_tensor_batch_keys},
            meta_info=self.meta_info.copy(),
        )
        return out


@pytest.mark.parametrize("async_mode", [False, True])
def test_gen_batch_keeps_correct_tensor_and_reward_routing(async_mode):
    fn = method("verl/trainer/ppo/ray_trainer.py", "RayPPOTrainer", "_get_gen_batch", {})
    trainer = SimpleNamespace(async_rollout_mode=async_mode, config=config("vllm" if async_mode else "hf"))
    batch = Proto()
    generated = fn(trainer, batch)
    assert "raw_prompt" in generated.non_tensor_batch
    assert "uid" in batch.non_tensor_batch
    assert "scores" in batch.batch
    if async_mode:
        assert batch.pop_args[0] == []
        assert "uid" in generated.non_tensor_batch
        assert "hf_prompt_truncation" not in generated.meta_info
    else:
        assert batch.pop_args[0] == ["input_ids", "attention_mask", "position_ids"]
        assert "uid" not in generated.non_tensor_batch
        assert generated.meta_info["hf_prompt_truncation"] == "left"
        assert generated.meta_info["hf_chat_template_kwargs"] == {"enable_thinking": False}
        assert generated.meta_info["global_steps"] == 7


def test_validate_preserves_tokenization_metadata():
    tree = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8"))
    validate = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_validate")
    overwrite = [
        n
        for n in ast.walk(validate)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Attribute)
            and t.attr == "meta_info"
            and isinstance(t.value, ast.Name)
            and t.value.id == "test_gen_batch"
            for t in n.targets
        )
    ]
    assert not overwrite, "Validation must not overwrite the HF tokenization policy added by _get_gen_batch"
