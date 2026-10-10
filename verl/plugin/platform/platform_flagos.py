# Copyright (c) 2026 BAAI. All rights reserved.
"""FlagOS / Torch-FL PrivateUse1 platform implementation."""

import importlib
import logging
import os
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from .platform_base import PlatformBase

logger = logging.getLogger(__name__)


def _load_flagos_device() -> ModuleType:
    """Load Torch-FL only if another caller has not already registered it.

    Importing this platform module alone must not require the optional extension.
    Torch-FL import failures deliberately propagate to the explicit selection path.
    """
    if getattr(torch, "flagos", None) is None:
        importlib.import_module("torch_fl")
    device = getattr(torch, "flagos", None)
    if device is None:
        raise RuntimeError("Torch-FL did not register torch.flagos")
    return device


class PlatformFlagOS(PlatformBase):
    def __init__(self) -> None:
        self._warned_unsupported: set[str] = set()

    def ensure_initialized(self) -> None:
        try:
            self.visible_devices_envvar()
            device = _load_flagos_device()
            if not device.is_available() or device.device_count() <= 0:
                raise RuntimeError("no visible FlagOS devices")
        except Exception as exc:
            raise RuntimeError(
                "FlagOS was selected but Torch-FL could not initialize a visible device. "
                "Check the Torch-FL installation, vendor runtime and device visibility; "
                "CPU fallback is disabled for FlagOS."
            ) from exc

    def _warn_unsupported(self, api: str) -> None:
        if api not in self._warned_unsupported:
            logger.warning("FlagOS %s is unsupported by this adapter; no device profiling is performed.", api)
            self._warned_unsupported.add(api)

    @property
    def device_name(self) -> str:
        return "flagos"

    @property
    def device_module(self) -> ModuleType:
        return _load_flagos_device()

    def is_available(self) -> bool:
        """Non-throwing optional-backend probe; initialization remains fail-fast."""
        try:
            device = self.device_module
            return bool(device.is_available() and device.device_count() > 0)
        except Exception:
            logger.debug("FlagOS availability probe failed", exc_info=True)
            return False

    def current_device(self) -> int:
        return int(self.device_module.current_device())

    def device_count(self) -> int:
        return int(self.device_module.device_count())

    def set_device(self, device_index: int) -> None:
        self.device_module.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        if device_index is None:
            self.device_module.synchronize()
        else:
            self.device_module.synchronize(device_index)

    def manual_seed(self, seed: int) -> None:
        self.device_module.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        device = self.device_module
        fn = getattr(device, "manual_seed_all", None)
        if not callable(fn):
            fn = device.manual_seed
        fn(seed)

    def set_allocator_settings(self, settings: str) -> None:
        memory = getattr(self.device_module, "memory", None)
        fn = getattr(memory, "_set_allocator_settings", None) if memory is not None else None
        if callable(fn):
            fn(settings)

    def empty_cache(self) -> None:
        fn = getattr(self.device_module, "empty_cache", None)
        if callable(fn):
            fn()

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        fn = getattr(self.device_module, "get_device_capability", None)
        if callable(fn):
            try:
                return fn(device_index)
            except NotImplementedError:
                pass
        return (None, None)

    def communication_backend_name(self) -> str:
        return "flagos"

    def visible_devices_envvar(self) -> str:
        # Worker normalizes Ray's HIP/ROCR assignments to CUDA_VISIBLE_DEVICES.
        # A different getter would then read an unset variable and could bypass
        # per-actor isolation, including in the colocated actor sharing path.
        override = os.getenv("FLAGOS_VISIBLE_DEVICES_ENV", "CUDA_VISIBLE_DEVICES").strip()
        if override not in ("", "CUDA_VISIBLE_DEVICES"):
            raise ValueError(
                "FLAGOS_VISIBLE_DEVICES_ENV overrides other than CUDA_VISIBLE_DEVICES "
                "are unsupported by the FlagOS/Ray adapter. Worker normalizes "
                "HIP/ROCR visibility to CUDA_VISIBLE_DEVICES; unset the override."
            )
        return "CUDA_VISIBLE_DEVICES"

    @contextmanager
    def nvtx_range(self, msg: str):
        """Compatibility no-op: this adapter does not emit device ranges."""
        self._warn_unsupported("nvtx_range")
        yield

    def profiler_start(self) -> None:
        """Compatibility no-op, not a successful device-profiler start."""
        self._warn_unsupported("profiler_start")

    def profiler_stop(self) -> None:
        """Compatibility no-op: no device profiler was started by this adapter."""
        self._warn_unsupported("profiler_stop")

    def cudart(self) -> Any:
        """No CUDA runtime facade is exposed for the PrivateUse1 device."""
        return None
