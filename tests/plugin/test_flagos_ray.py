"""Ray scheduling control-flow tests lifted from source; no Ray cluster started."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from .test_flagos_helpers import source_functions


@pytest.fixture
def ray_functions():
    ns = {
        "Any": Any,
        "PlacementGroupSchedulingStrategy": lambda **kwargs: kwargs,
        "sort_placement_group_by_node_ip": lambda groups: groups,
        "ray": SimpleNamespace(get=lambda value: value),
    }
    source_functions(
        "verl/single_controller/ray/base.py",
        ["_ray_accelerator_resource", "get_placement_groups", "__call__"],
        ns,
    )
    return ns


@pytest.mark.parametrize("device,resource", [("flagos", "GPU"), ("cuda", "GPU"), ("musa", "GPU"), ("npu", "NPU")])
def test_placement_and_actor_use_same_accelerator_resource(ray_functions, device, resource):
    ns = ray_functions
    calls = []

    def placement_group(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(ready=lambda: "ready")

    ns["placement_group"] = placement_group
    pool = SimpleNamespace(
        pgs=None,
        name_prefix="test",
        _store=[1],
        max_colocate_count=2,
        use_gpu=True,
        accelerator_type=None,
        detached=False,
    )
    groups = ns["get_placement_groups"](pool, device_name=device)
    assert calls[0]["bundles"] == [{"CPU": 2, resource: 1}]
    assert len(groups) == 1
    cls = Mock()
    actor = SimpleNamespace(cls=cls, args=("input",), kwargs={}, _options={}, _additional_resource={})
    ns["__call__"](actor, "group", 0, num_gpus=0.5, device_name=device)
    options = cls.options.call_args.kwargs
    if resource == "GPU":
        assert options["num_gpus"] == 0.5
        assert "resources" not in options
    else:
        assert options["resources"] == {"NPU": 0.5}
        assert "num_gpus" not in options
    assert "flagos" not in calls[0]["bundles"][0]


@pytest.mark.parametrize("device", ["flagos", "cuda", "musa", "npu", "cpu"])
def test_cpu_only_pool_and_actor_do_not_request_accelerators(ray_functions, device):
    ns = ray_functions
    placement_group = Mock(return_value=SimpleNamespace(ready=lambda: "ready"))
    ns["placement_group"] = placement_group
    pool = SimpleNamespace(
        pgs=None,
        name_prefix="test",
        _store=[1],
        max_colocate_count=2,
        use_gpu=False,
        accelerator_type=None,
        detached=False,
    )
    ns["get_placement_groups"](pool, device_name=device)
    assert placement_group.call_args.kwargs["bundles"] == [{"CPU": 2}]
    cls = Mock()
    actor = SimpleNamespace(cls=cls, args=(), kwargs={}, _options={}, _additional_resource={})
    ns["__call__"](actor, "group", 0, use_gpu=False, device_name=device)
    assert set(cls.options.call_args.kwargs) == {"scheduling_strategy"}


def test_flagos_preserves_extra_constraints_and_placement_cache(ray_functions):
    ns = ray_functions
    placement_group = Mock(return_value=SimpleNamespace(ready=lambda: "ready"))
    ns["placement_group"] = placement_group
    pool = SimpleNamespace(
        pgs=None,
        name_prefix="test",
        _store=[1],
        max_colocate_count=2,
        use_gpu=True,
        accelerator_type="accelerator_type:DCU",
        detached=True,
    )
    groups = ns["get_placement_groups"](pool, device_name="flagos")
    assert placement_group.call_args.kwargs["bundles"] == [{"CPU": 2, "GPU": 1, "accelerator_type:DCU": 1e-4}]
    assert placement_group.call_args.kwargs["lifetime"] == "detached"
    assert ns["get_placement_groups"](pool, device_name="flagos") is groups
    placement_group.assert_called_once()


def test_unknown_custom_resource_mapping_is_unchanged(ray_functions):
    assert ray_functions["_ray_accelerator_resource"]("custom") == "custom"
