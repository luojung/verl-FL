"""Isolated execution of telemetry function ASTs, not real device validation."""

import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from .test_flagos_helpers import source_functions


@pytest.fixture
def telemetry():
    torch = SimpleNamespace(cpu=object(), flagos=SimpleNamespace())
    namespace = {"torch": torch, "get_torch_device": lambda: torch.flagos, "get_device_id": lambda: 0}
    source_functions("verl/utils/flops_counter.py", ["get_device_flops"], namespace, assignments=["_DEVICE_FLOPS"])
    source_functions("verl/utils/profiler/performance.py", ["_get_current_mem_info"], namespace)
    return torch, namespace


def test_unknown_flagos_capacity_is_nan_not_zero_mfu(telemetry):
    _, ns = telemetry
    capacity = ns["get_device_flops"]()
    assert math.isnan(capacity)
    assert math.isnan(123 / capacity)


@pytest.mark.parametrize("name", ["Hygon DCU", None, "", 42])
def test_unmapped_or_invalid_flagos_device_name_is_unknown(telemetry, name):
    torch, ns = telemetry
    torch.flagos.get_device_name = lambda: name
    assert math.isnan(ns["get_device_flops"]())


def test_flagos_known_capacity_from_properties(telemetry):
    torch, ns = telemetry
    torch.flagos.current_device = lambda: 0
    torch.flagos.get_device_properties = lambda index: SimpleNamespace(name="AMD MI300X")
    assert ns["get_device_flops"]() == 1336


@pytest.mark.parametrize(
    "error", [NotImplementedError(), RuntimeError("device-name unsupported"), OSError("native library")]
)
def test_unsupported_flagos_name_is_unknown(telemetry, error):
    torch, ns = telemetry
    torch.flagos.get_device_name = Mock(side_effect=error)
    assert math.isnan(ns["get_device_flops"]())


def test_existing_capacity_table_and_cpu_preserved(telemetry):
    torch, ns = telemetry
    for name, capacity in ns["_DEVICE_FLOPS"].items():
        assert ns["get_device_flops"](unit="B", device_name=name) == capacity
    assert math.isinf(ns["get_device_flops"](device_name="unknown legacy GPU"))
    ns["get_torch_device"] = lambda: torch.cpu
    assert ns["get_device_flops"]() == 0.448
    assert ns["_get_current_mem_info"]() == ("0.00",) * 4


@pytest.mark.parametrize("backend", ["cuda", "npu", "musa"])
def test_non_flagos_capacity_and_memory_apis_preserved(telemetry, backend):
    _, ns = telemetry
    current = SimpleNamespace(
        get_device_name=lambda: "H100",
        memory_allocated=lambda: 1024**3,
        memory_reserved=lambda: 2 * 1024**3,
        mem_get_info=lambda: (3 * 1024**3, 8 * 1024**3),
        __name__=f"torch.{backend}",
    )
    ns["get_torch_device"] = lambda: current
    assert ns["get_device_flops"]() == 989
    assert ns["_get_current_mem_info"]() == ("1.00", "2.00", "5.00", "8.00")
    current.mem_get_info = Mock(side_effect=RuntimeError("driver error"))
    with pytest.raises(RuntimeError, match="driver error"):
        ns["_get_current_mem_info"]()


def test_missing_flagos_memory_counters_are_unavailable(telemetry):
    _, ns = telemetry
    assert ns["_get_current_mem_info"]() == ("N/A",) * 4


def test_allocator_usage_never_substitutes_for_global_memory(telemetry):
    torch, ns = telemetry
    torch.flagos.memory_allocated = lambda: 1024**3
    torch.flagos.memory_reserved = lambda: 2 * 1024**3
    torch.flagos.get_device_properties = lambda index: SimpleNamespace(total_memory=8 * 1024**3)
    assert ns["_get_current_mem_info"]() == ("1.00", "2.00", "N/A", "8.00")


def test_available_flagos_global_memory_keeps_true_zero(telemetry):
    torch, ns = telemetry
    torch.flagos.memory_allocated = lambda: 0
    torch.flagos.mem_get_info = lambda: (8 * 1024**3, 8 * 1024**3)
    assert ns["_get_current_mem_info"]() == ("0.00", "N/A", "0.00", "8.00")


@pytest.mark.parametrize("error", [NotImplementedError(), RuntimeError("unsupported"), OSError("native library")])
def test_unimplemented_flagos_memory_metrics_are_independently_unavailable(telemetry, error):
    torch, ns = telemetry
    torch.flagos.memory_allocated = Mock(side_effect=error)
    torch.flagos.memory_reserved = lambda: 1024**3
    torch.flagos.mem_get_info = Mock(side_effect=error)
    torch.flagos.get_device_properties = Mock(side_effect=error)
    assert ns["_get_current_mem_info"]() == ("N/A", "1.00", "N/A", "N/A")


@pytest.mark.parametrize("unit,divisor", [("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)])
def test_flagos_memory_units_and_precision(telemetry, unit, divisor):
    torch, ns = telemetry
    torch.flagos.memory_allocated = lambda: divisor
    assert ns["_get_current_mem_info"](unit, precision=3)[0] == "1.000"


@pytest.mark.parametrize("name", [None, "", "  ", 42])
def test_invalid_name_api_still_tries_properties(telemetry, name):
    torch, ns = telemetry
    torch.flagos.get_device_name = lambda: name
    torch.flagos.current_device = lambda: 2
    torch.flagos.get_device_properties = Mock(return_value=SimpleNamespace(name="AMD MI300X"))
    assert ns["get_device_flops"]() == 1336
    torch.flagos.get_device_properties.assert_called_once_with(2)


@pytest.mark.parametrize("error", [NotImplementedError(), RuntimeError("unsupported"), OSError("native library")])
def test_raising_name_api_still_tries_properties(telemetry, error):
    torch, ns = telemetry
    torch.flagos.get_device_name = Mock(side_effect=error)
    torch.flagos.current_device = lambda: 0
    torch.flagos.get_device_properties = Mock(return_value=SimpleNamespace(name="AMD MI300X"))
    assert ns["get_device_flops"]() == 1336
    torch.flagos.get_device_properties.assert_called_once_with(0)


def test_properties_oserror_also_returns_unknown_capacity(telemetry):
    torch, ns = telemetry
    torch.flagos.current_device = lambda: 0
    torch.flagos.get_device_properties = Mock(side_effect=OSError("native library"))
    assert math.isnan(ns["get_device_flops"]())
