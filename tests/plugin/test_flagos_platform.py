"""FlagOS platform policy tests using mocked Torch-FL, never hardware."""

import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from .test_flagos_helpers import load_platform_modules, source_functions


@pytest.fixture
def platform_modules(monkeypatch):
    torch = ModuleType("torch")
    torch.cpu = ModuleType("torch.cpu")
    torch.cuda = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    monkeypatch.setitem(sys.modules, "torch_musa", None)
    monkeypatch.delenv("VERL_PLATFORM", raising=False)
    monkeypatch.delenv("FLAGOS_VISIBLE_DEVICES_ENV", raising=False)
    return torch, *load_platform_modules(monkeypatch, torch)


def device(available=True, count=1):
    return SimpleNamespace(is_available=lambda: available, device_count=lambda: count)


def test_module_import_does_not_require_torch_fl(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_fl", None)
    torch = ModuleType("torch")
    flagos, _ = load_platform_modules(monkeypatch, torch)
    assert flagos.PlatformFlagOS().device_name == "flagos"


def test_registered_backend_skips_optional_import(platform_modules, monkeypatch):
    torch, flagos, manager = platform_modules
    torch.flagos = device()
    import_module = Mock(side_effect=AssertionError("must not import again"))
    monkeypatch.setattr(flagos.importlib, "import_module", import_module)
    assert manager._detect_platform_name() == "flagos"
    platform = manager.get_platform()
    assert platform.device_module is torch.flagos
    assert manager.get_platform() is platform
    import_module.assert_not_called()


def test_optional_import_registers_backend(platform_modules, monkeypatch):
    torch, flagos, manager = platform_modules

    def register(name):
        assert name == "torch_fl"
        torch.flagos = device()

    monkeypatch.setattr(flagos.importlib, "import_module", register)
    assert manager._detect_platform_name() == "flagos"


@pytest.mark.parametrize("error", [ImportError("missing"), OSError("DLL missing"), RuntimeError("runtime broken")])
def test_explicit_flagos_fails_with_cause_and_no_cached_cpu(platform_modules, monkeypatch, error):
    _, flagos, manager = platform_modules
    monkeypatch.setenv("VERL_PLATFORM", " FLAGOS ")
    monkeypatch.setattr(flagos.importlib, "import_module", Mock(side_effect=error))
    assert manager._detect_platform_name() == "flagos"
    for _ in range(2):
        with pytest.raises(RuntimeError, match="CPU fallback is disabled") as exc:
            manager.get_platform()
        assert exc.value.__cause__ is error
        assert manager._current_platform is None


@pytest.mark.parametrize("available,count", [(False, 1), (True, 0)])
def test_explicit_unavailable_flagos_never_becomes_cpu(platform_modules, monkeypatch, available, count):
    torch, _, manager = platform_modules
    torch.flagos = device(available, count)
    monkeypatch.setenv("VERL_PLATFORM", "flagos")
    with pytest.raises(RuntimeError, match="visible device"):
        manager.get_platform()
    assert manager._current_platform is None


def test_missing_registration_fails_explicitly(platform_modules, monkeypatch):
    _, flagos, manager = platform_modules
    monkeypatch.setattr(flagos.importlib, "import_module", Mock(return_value=object()))
    monkeypatch.setenv("VERL_PLATFORM", "flagos")
    with pytest.raises(RuntimeError, match="CPU fallback") as exc:
        manager.get_platform()
    assert "did not register" in str(exc.value.__cause__)


@pytest.mark.parametrize("error", [ImportError("missing"), OSError("DLL missing"), RuntimeError("broken")])
@pytest.mark.parametrize("fallback", ["cpu", "cuda", "npu", "musa"])
def test_broken_optional_import_preserves_other_detection(platform_modules, monkeypatch, error, fallback):
    torch, flagos, manager = platform_modules
    monkeypatch.setattr(flagos.importlib, "import_module", Mock(side_effect=error))
    if fallback == "cuda":
        torch.cuda = device()
    elif fallback in ("musa", "npu"):
        setattr(torch, fallback, device())
        if fallback == "musa":
            monkeypatch.setitem(sys.modules, "torch_musa", ModuleType("torch_musa"))
    assert manager._detect_platform_name() == fallback


@pytest.mark.parametrize("backend", ["cpu", "cuda", "npu", "musa"])
def test_other_explicit_overrides_do_not_load_torch_fl(platform_modules, monkeypatch, backend):
    _, flagos, manager = platform_modules
    monkeypatch.setenv("VERL_PLATFORM", backend)
    import_module = Mock(side_effect=AssertionError("optional import on explicit override"))
    monkeypatch.setattr(flagos.importlib, "import_module", import_module)
    assert manager._detect_platform_name() == backend
    import_module.assert_not_called()


def test_failed_initialization_is_not_cached(platform_modules, monkeypatch):
    _, _, manager = platform_modules
    broken = SimpleNamespace(ensure_initialized=Mock(side_effect=RuntimeError("init")))
    monkeypatch.setattr(manager, "_create_platform", lambda name: broken)
    monkeypatch.setenv("VERL_PLATFORM", "cpu")
    with pytest.raises(RuntimeError, match="init"):
        manager.get_platform()
    assert manager._current_platform is None


def test_device_api_delegation_and_seed_fallback(platform_modules):
    torch, flagos, _ = platform_modules
    torch.flagos = device()
    for name in ("set_device", "synchronize", "manual_seed"):
        setattr(torch.flagos, name, Mock())
    torch.flagos.current_device = Mock(return_value=2)
    platform = flagos.PlatformFlagOS()
    assert platform.current_device() == 2
    assert platform.device_count() == 1
    platform.set_device(2)
    platform.synchronize()
    platform.synchronize(2)
    platform.manual_seed_all(42)
    torch.flagos.set_device.assert_called_once_with(2)
    assert torch.flagos.synchronize.call_args_list[0].args == ()
    assert torch.flagos.synchronize.call_args_list[1].args == (2,)
    torch.flagos.manual_seed.assert_called_once_with(42)
    torch.flagos.manual_seed_all = Mock()
    del torch.flagos.manual_seed
    platform.manual_seed_all(7)
    torch.flagos.manual_seed_all.assert_called_once_with(7)


def test_profiler_noops_are_explicit_and_warn_once(platform_modules, caplog):
    _, flagos, _ = platform_modules
    platform = flagos.PlatformFlagOS()
    for _ in range(2):
        with platform.nvtx_range("test"):
            pass
        platform.profiler_start()
        platform.profiler_stop()
    assert len(caplog.records) == 3
    assert all("unsupported" in record.message for record in caplog.records)
    assert platform.cudart() is None


def test_capability_is_unknown_not_cuda_fabrication(platform_modules):
    torch, flagos, _ = platform_modules
    torch.flagos = device()
    platform = flagos.PlatformFlagOS()
    assert platform.get_device_capability() == (None, None)
    torch.flagos.get_device_capability = Mock(side_effect=NotImplementedError)
    assert platform.get_device_capability() == (None, None)
    torch.flagos.get_device_capability = Mock(side_effect=RuntimeError("driver"))
    with pytest.raises(RuntimeError, match="driver"):
        platform.get_device_capability()


def test_visibility_and_communication_names(platform_modules, monkeypatch):
    _, flagos, _ = platform_modules
    platform = flagos.PlatformFlagOS()
    monkeypatch.delenv("FLAGOS_VISIBLE_DEVICES_ENV", raising=False)
    assert platform.visible_devices_envvar() == "CUDA_VISIBLE_DEVICES"
    assert platform.communication_backend_name() == "flagos"
    monkeypatch.setenv("FLAGOS_VISIBLE_DEVICES_ENV", "CUDA_VISIBLE_DEVICES")
    assert platform.visible_devices_envvar() == "CUDA_VISIBLE_DEVICES"


@pytest.mark.parametrize("available,count", [(False, 1), (True, 0)])
def test_auto_unavailable_flagos_preserves_cuda(platform_modules, available, count):
    torch, _, manager = platform_modules
    torch.flagos = device(available, count)
    torch.cuda = device()
    assert manager._detect_platform_name() == "cuda"


def test_auto_detection_keeps_priority(platform_modules, monkeypatch):
    torch, _, manager = platform_modules
    torch.flagos = device()
    torch.musa = device()
    torch.npu = device()
    torch.cuda = device()
    monkeypatch.setitem(sys.modules, "torch_musa", ModuleType("torch_musa"))
    assert manager._detect_platform_name() == "flagos"
    torch.flagos = device(False)
    assert manager._detect_platform_name() == "musa"
    torch.musa = device(False)
    assert manager._detect_platform_name() == "npu"
    torch.npu = device(False)
    assert manager._detect_platform_name() == "cuda"
    torch.cuda = device(True, 0)
    assert manager._detect_platform_name() == "cpu"


def test_invalid_override_preserves_legacy_detection_warning(platform_modules, monkeypatch, caplog):
    torch, _, manager = platform_modules
    torch.flagos = device()
    monkeypatch.setenv("VERL_PLATFORM", "not-a-backend")
    assert manager._detect_platform_name() == "flagos"
    assert "Falling back to auto-detection" in caplog.text


def test_optional_probe_does_not_swallow_process_interrupt(platform_modules, monkeypatch):
    _, flagos, manager = platform_modules
    monkeypatch.setattr(flagos.importlib, "import_module", Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        manager._detect_platform_name()


@pytest.mark.parametrize("override", ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUSTOM_DEVICES"])
def test_conflicting_visibility_override_is_rejected_before_initialization(platform_modules, monkeypatch, override):
    _, flagos, manager = platform_modules
    monkeypatch.setenv("FLAGOS_VISIBLE_DEVICES_ENV", override)
    monkeypatch.setenv("VERL_PLATFORM", "flagos")
    load = Mock(side_effect=AssertionError("must reject before loading backend"))
    monkeypatch.setattr(flagos, "_load_flagos_device", load)
    with pytest.raises(ValueError, match="unset the override"):
        flagos.PlatformFlagOS().visible_devices_envvar()
    with pytest.raises(RuntimeError, match="CPU fallback") as exc:
        manager.get_platform()
    assert isinstance(exc.value.__cause__, ValueError)
    assert manager._current_platform is None
    load.assert_not_called()


@pytest.mark.parametrize("error", [ImportError("missing"), RuntimeError("runtime"), OSError("native library")])
def test_availability_returns_false_but_initialization_preserves_error(platform_modules, monkeypatch, error):
    _, flagos, _ = platform_modules
    monkeypatch.setattr(flagos.importlib, "import_module", Mock(side_effect=error))
    platform = flagos.PlatformFlagOS()
    assert platform.is_available() is False
    with pytest.raises(RuntimeError, match="CPU fallback") as exc:
        platform.ensure_initialized()
    assert exc.value.__cause__ is error


def test_get_platform_initializes_flagos_exactly_once(platform_modules, monkeypatch):
    torch, flagos, manager = platform_modules
    torch.flagos = device()
    monkeypatch.setenv("VERL_PLATFORM", "flagos")
    initialize = Mock()
    monkeypatch.setattr(flagos.PlatformFlagOS, "ensure_initialized", initialize)
    selected = manager.get_platform()
    assert manager.get_platform() is selected
    initialize.assert_called_once_with()


@pytest.mark.parametrize("ray_variable", ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"])
@pytest.mark.parametrize("override", ["", "CUDA_VISIBLE_DEVICES"])
def test_worker_normalization_getter_and_sharing_keep_ray_assignment(
    platform_modules, monkeypatch, ray_variable, override
):
    # Execute the unchanged Worker and actor-sharing methods with mocks. This
    # proves the Python visibility contract, not actual Ray process isolation.
    _, flagos, _ = platform_modules
    platform = flagos.PlatformFlagOS()
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        # Track even initially absent variables so writes by the real Worker
        # method cannot leak into subsequent tests during fixture teardown.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv(ray_variable, "2")
    monkeypatch.setenv("FLAGOS_VISIBLE_DEVICES_ENV", override)
    utils = ModuleType("verl.utils.ray_utils")
    utils.ray_noset_visible_devices = lambda: False
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    ns = {
        "os": os,
        "get_device_name": lambda: "flagos",
        "get_visible_devices_keyword": platform.visible_devices_envvar,
    }
    source_functions(
        "verl/single_controller/base/worker.py", ["_setup_env_cuda_visible_devices", "get_cuda_visible_devices"], ns
    )
    ns["_setup_env_cuda_visible_devices"](None)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    assert "HIP_VISIBLE_DEVICES" not in os.environ
    assert "ROCR_VISIBLE_DEVICES" not in os.environ
    assert ns["get_cuda_visible_devices"](None) == "2"
    actor_ns = {
        "Any": object,
        "ray": SimpleNamespace(get=lambda value: value),
        "NodeAffinitySchedulingStrategy": lambda **kwargs: kwargs,
    }
    source_functions("verl/single_controller/ray/base.py", ["__call__"], actor_ns)
    shared = SimpleNamespace(
        get_node_id=SimpleNamespace(remote=lambda: "node"),
        get_cuda_visible_devices=SimpleNamespace(remote=lambda: ns["get_cuda_visible_devices"](None)),
    )
    cls = Mock()
    actor = SimpleNamespace(cls=cls, args=(), kwargs={})
    actor_ns["__call__"](actor, None, 0, sharing_with=shared, device_name="flagos")
    assert cls.options.return_value.remote.call_args.kwargs["cuda_visible_devices"] == "2"
