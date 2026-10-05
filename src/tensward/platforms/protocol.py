"""What Tensward needs from an accelerator platform, in vendor-neutral terms.

A :class:`Platform` finds its devices, knows their datasheet figures, and tells a runtime how
to pin a process or a container to some of them. Each vendor's specifics live in its own
module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

MIB = 2**20


@dataclass(frozen=True, slots=True)
class HardwareSpec:
    """Dense (not sparsity) datasheet numbers. ``fp8_tflops`` is None when unsupported."""

    label: str
    name_pattern: str
    bandwidth_gbs: float
    fp16_tflops: float
    int8_tops: float
    fp8_tflops: float | None
    sms: int


@dataclass(frozen=True, slots=True)
class DeviceIdentity:
    """What distinguishes one device from another that runs the same engine the same way."""

    index: str
    uuid: str
    name: str
    driver: str | None
    driver_cuda: str | None
    pci_device_id: str | None
    vbios: str | None
    sm_count: int | None


@dataclass(frozen=True, slots=True)
class Device:
    """One device as detected: its memory at that moment, and what identifies it."""

    name: str
    total_bytes: int
    used_bytes: int
    index: str = ""
    uuid: str = ""
    bus_id: str = ""
    driver: str | None = None
    pci_device_id: str | None = None
    vbios: str | None = None


class Platform(Protocol):
    """One accelerator vendor."""

    name: str
    label: str  # the vendor as people write it; device names may start with it

    def detect(self) -> tuple[Device, ...]:
        """The devices visible here, asked afresh; empty when the platform is absent. Never
        raises, and never starts a vendor driver in this process."""
        ...

    def spec(self, device_name: str) -> HardwareSpec | None:
        """The datasheet row for a device of this name; None when the table has none."""
        ...

    def derived_bandwidth(self, index: int) -> float | None:
        """Peak memory bandwidth of device ``index`` in GB/s, from the device's own figures."""
        ...

    def identity(self, device: Device) -> dict[str, Any]:
        """The fields of :class:`DeviceIdentity` for ``device``; one the platform cannot read is
        None. Never raises."""
        ...

    def describe(self, device: Device, identity: Mapping[str, Any]) -> str:
        """``device`` and the driver facts in ``identity`` as one phrase for a person; a fact it
        lacks reads "unknown"."""
        ...

    def select(
        self, devices: tuple[Device, ...], selection: tuple[str, ...] | None
    ) -> tuple[tuple[Device, ...], tuple[str, ...]]:
        """The devices a selection names (all of them without one), and the tokens that name
        none."""
        ...

    def selection_refusal(self, selection: Sequence[str]) -> str | None:
        """Why a device check cannot judge ``selection``; None when it can."""
        ...

    def select_env(
        self, indices: tuple[str, ...] | None, environ: Mapping[str, str]
    ) -> dict[str, str]:
        """Variables that pin a local process to ``indices`` (none for None), given the
        environment it would otherwise get."""
        ...

    def docker_device_args(self, indices: tuple[str, ...] | None) -> list[str]:
        """``docker run`` options that give a container ``indices`` (every device for None)."""
        ...
