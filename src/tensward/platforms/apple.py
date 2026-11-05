"""Apple Silicon GPUs: one Metal device that shares the Mac's memory with the CPU.

Detection reads ``sysctl`` and the platform UUID; it never starts Metal in this process. The
bandwidth rows are the figures Apple publishes for each chip (the specifications page of the
first Mac that carries it); M1 has none, so its row is the upstream-measured figure and says so.
Where the figure depends on the GPU-core bin, one row per bin and the core count picks it. No
arithmetic rate is listed: Apple does not publish one per chip.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

from . import darwin
from .protocol import GIB, Device, DeviceIdentity, GpuSample, HardwareSpec

BRAND = "Apple M"
_CHIP = re.compile(r"^Apple M([1-9])( Pro| Max| Ultra)?$")
MAX_GENERATION = 5


@dataclass(frozen=True, slots=True)
class AppleChip:
    chip: str
    generation: int
    gpu_cores: int | None


def chip_of(device_name: str, gpu_cores: int | None) -> AppleChip | None:
    found = _CHIP.match(device_name.strip())
    if found is None or int(found[1]) > MAX_GENERATION:
        return None
    return AppleChip(device_name.strip().removeprefix("Apple "), int(found[1]), gpu_cores)


def arch_of(device_name: str) -> str | None:
    chip = chip_of(device_name, None)
    return None if chip is None else f"apple_m{chip.generation}"


def _row(chip: str, gbs: float, cores: int | None = None, note: str = "") -> HardwareSpec:
    bin_label = f", {cores}-core GPU" if cores else ""
    return HardwareSpec(f"{chip}{bin_label}{note}", rf"^Apple {chip}$", gbs, None, None, None, None)


# Memory bandwidth in GB/s from Apple's newsroom and specification pages; GPU-core bins are
# separate rows.
APPLE_SPECS: tuple[HardwareSpec, ...] = (
    _row("M1", 68, note=" (about 68 GB/s, provisional: upstream measurements, not a datasheet)"),
    _row("M1 Pro", 200),
    _row("M1 Max", 400),
    _row("M1 Ultra", 800),
    _row("M2", 100),
    _row("M2 Pro", 200),
    _row("M2 Max", 400),
    _row("M2 Ultra", 800),
    _row("M3", 100),
    _row("M3 Pro", 150),
    _row("M3 Max", 300, 30),
    _row("M3 Max", 400, 40),
    _row("M3 Ultra", 819),
    _row("M4", 120),
    _row("M4 Pro", 273),
    _row("M4 Max", 410, 32),
    _row("M4 Max", 546, 40),
    _row("M5", 153),
)


def _system() -> str | None:
    version, build = darwin.os_build()
    return f"{version} ({build})" if version and build else build or version


def _unified_gib(device: Device) -> str:
    return f"{device.total_bytes / GIB:g}"


class ApplePlatform:
    name = "apple"
    label = "Apple"
    bandwidth_figures = True
    accelerator = True
    memory_topology: Literal["unified"] = "unified"

    def detect(self) -> tuple[Device, ...]:
        if not darwin.on_mac():
            return ()
        keys = darwin.sysctl("machdep.cpu.brand_string", "hw.memsize")
        brand = keys.get("machdep.cpu.brand_string", "")
        if not brand.startswith(BRAND) or not keys.get("hw.memsize", "").isdecimal():
            return ()
        total = int(keys["hw.memsize"])
        available = darwin.available_gib()
        used = 0 if available is None else max(total - round(available * GIB), 0)
        return (
            Device(brand, total, used, "0", darwin.platform_uuid_digest() or "", driver=_system()),
        )

    def spec(self, device_name: str) -> HardwareSpec | None:
        if not device_name.startswith(BRAND):
            return None
        rows = [row for row in APPLE_SPECS if re.search(row.name_pattern, device_name)]
        if len(rows) > 1:
            cores = darwin.gpu_cores()
            rows = [row for row in rows if cores is not None and f", {cores}-core GPU" in row.label]
        return rows[0] if len(rows) == 1 else None

    def derived_bandwidth(self, index: int) -> float | None:
        return None

    def identity(self, device: Device) -> dict[str, Any]:
        return asdict(
            DeviceIdentity(
                device.index,
                device.uuid,
                device.name,
                _system(),
                None,
                None,
                None,
                None,
                darwin.gpu_cores(),
            )  # fmt: skip
        )

    def describe(self, device: Device, identity: Mapping[str, Any]) -> str:
        cores = identity.get("gpu_cores")
        gpu = f"{cores}-core GPU" if cores else "GPU cores unknown"
        return (
            f"{device.name}, {gpu}, {_unified_gib(device)} GiB unified, "
            f"macOS {identity.get('driver') or 'unknown'}"
        )

    def select(
        self, devices: tuple[Device, ...], selection: tuple[str, ...] | None
    ) -> tuple[tuple[Device, ...], tuple[str, ...]]:
        if selection is None:
            return devices, ()
        return (
            tuple(d for d in devices if d.index in selection),
            tuple(token for token in selection if token != "0"),
        )

    def selection_refusal(self, selection: Sequence[str]) -> str | None:
        return "a Mac has one GPU" if any(token != "0" for token in selection) else None

    def select_env(
        self, indices: tuple[str, ...] | None, environ: Mapping[str, str]
    ) -> dict[str, str]:
        return {}

    def parse_sample(self, line: str, at_ns: int) -> GpuSample | None:
        return None

    def sampler_commands(self, indices: tuple[str, ...] | None) -> tuple[tuple[str, ...], ...]:
        return ()

    def docker_device_args(self, indices: tuple[str, ...] | None) -> list[str]:
        from ..runtime import RuntimeFailure  # noqa: PLC0415 - runtime imports this package

        raise RuntimeFailure("Metal is not available in containers on macOS; use --runtime local")


APPLE = ApplePlatform()
