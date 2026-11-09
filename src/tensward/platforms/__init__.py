"""Accelerator platforms: one module per vendor behind :class:`Platform`."""

from __future__ import annotations

import re
from typing import Collection

from . import darwin
from .apple import APPLE
from .cpu import CPU
from .nvidia import BANDWIDTH_ONLY, NVIDIA
from .protocol import (
    MIB,
    Device,
    DeviceIdentity,
    GpuSample,
    HardwareSpec,
    Platform,
    ProcessSample,
)

__all__ = [
    "APPLE", "CPU", "MIB", "PLATFORMS", "Device", "DeviceIdentity", "GpuSample", "HardwareSpec",
    "Platform", "ProcessSample", "darwin",
    "datasheet_bandwidth_gbs", "derived_bandwidth", "detect_devices", "detect_platform",
    "hardware_spec", "platform_for",
]  # fmt: skip

PLATFORMS: tuple[Platform, ...] = (NVIDIA, APPLE, CPU)
"""In detection order; the first is a runtime's platform when it is given none."""


def detect_platform() -> tuple[Platform, tuple[Device, ...]] | None:
    """The first registered accelerator platform with a device here, and its devices; None when
    none has one."""
    for platform in PLATFORMS:
        if platform.accelerator and (devices := platform.detect()):
            return platform, devices
    return None


def detect_devices() -> tuple[Device, ...]:
    """The devices of the detected platform; empty when there is none."""
    found = detect_platform()
    return found[1] if found else ()


def hardware_spec(device_name: str) -> HardwareSpec | None:
    """The datasheet row any registered platform has for ``device_name``."""
    return next((s for p in PLATFORMS if (s := p.spec(device_name)) is not None), None)


def datasheet_bandwidth_gbs(device_name: str) -> float | None:
    """The memory bandwidth published for ``device_name``, with or without a full
    :func:`hardware_spec` row."""
    spec = hardware_spec(device_name)
    if spec is not None:
        return float(spec.bandwidth_gbs)
    return next(
        (float(gbs) for pattern, gbs in BANDWIDTH_ONLY if re.search(pattern, device_name)), None
    )


def derived_bandwidth(index: int) -> float | None:
    """Peak memory bandwidth of device ``index`` from the device itself, if a platform can
    read it."""
    return next((b for p in PLATFORMS if (b := p.derived_bandwidth(index)) is not None), None)


def platform_for(supported: Collection[str], *, host_only: bool = False) -> Platform:
    """The detected platform when it is in ``supported``; else the CPU platform if it is, else
    the first registered one that is. With ``host_only`` (the engine is told to use no device)
    the CPU platform, when supported, whatever is detected."""
    if host_only and CPU.name in supported:
        return CPU
    found = detect_platform()
    if found is not None and found[0].name in supported:
        return found[0]
    ranked = sorted((p for p in PLATFORMS if p.name in supported), key=lambda p: p.accelerator)
    return ranked[0]
