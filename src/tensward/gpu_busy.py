"""Whether the GPU (or the RAM, on a CPU or an Apple machine) is free for a server of our own,
and who holds it when it is not."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Collection

from .engines.protocol import Engine
from .errors import GPU_BUSY, PreflightError
from .platforms import cpu, darwin
from .platforms.protocol import GIB, MIB, Platform
from .probes import run_probe
from .settings import Settings

COMPUTE_APPS_QUERY = (
    "nvidia-smi",
    "--query-compute-apps=pid,used_memory,gpu_uuid",
    "--format=csv,noheader,nounits",
)
DEFAULT_KV_MEMORY_FRACTION = 0.9
GPU_WEIGHT_MARGIN = 1.1
OFFLOAD_DEVICE_SHARE = 0.9
HOST_WEIGHT_MARGIN = 1.2


@dataclass(frozen=True, slots=True)
class Holder:
    """A process holding memory on the device."""

    pid: int
    name: str
    used_bytes: int


@dataclass(frozen=True, slots=True)
class Busy:
    """Memory that is not free: ``where`` names the device ("GPU 0") or "RAM"."""

    where: str
    used_bytes: int
    total_bytes: int
    holders: tuple[Holder, ...]
    needs_bytes: int


def needed_bytes(
    engine: Engine, settings: Settings, weight_bytes: int, total_bytes: int, *, host: bool = False
) -> int:
    """Memory the engine claims for a server: vLLM takes a share of the device, llama.cpp the
    weights plus a margin, on the host (CPU or Apple, ``host``) a larger margin."""
    if engine.name == "vllm":
        return int((settings.kv_memory_fraction or DEFAULT_KV_MEMORY_FRACTION) * total_bytes)
    return int(weight_bytes * (HOST_WEIGHT_MARGIN if host else GPU_WEIGHT_MARGIN))


def needs_for(
    engine: Engine,
    settings: Settings,
    weight_bytes: int,
    platform: Platform,
    gpus: tuple[str, ...] | None,
) -> int:
    """The bytes a server of ``engine`` claims on the device (or the RAM) it will use. A model
    larger than the device is offloaded, so llama.cpp claims at most ``OFFLOAD_DEVICE_SHARE`` of
    the device."""
    host = platform.memory_topology != "discrete"
    total = 0
    if not host:
        devices, _ = platform.select(platform.detect(), gpus)
        total = devices[0].total_bytes if devices else 0
    needs = needed_bytes(engine, settings, weight_bytes, total, host=host)
    if platform.memory_topology == "unified":
        needs = weight_bytes  # --fit cuts the KV cache to what is left; macOS keeps its own share
    if not total:
        return needs
    if engine.name == "vllm":
        return min(needs, total)
    return min(needs, int(OFFLOAD_DEVICE_SHARE * total))


def check_free(
    platform: Platform,
    gpus: tuple[str, ...] | None,
    needs: int,
    exclude: Collection[int] = (),
) -> Busy | None:
    """The first device (or the RAM) with less than ``needs`` bytes free, None when all have
    enough or the memory cannot be read. The memory of the processes in ``exclude`` counts as
    free: servers of our own that are about to be replaced."""
    if platform.memory_topology != "discrete":
        return _check_host(platform, needs)
    devices, _ = platform.select(platform.detect(), gpus)
    apps = _compute_apps()
    for device in devices:
        on_device = [h for uuid, h in apps if uuid == device.uuid]
        ours = sum(h.used_bytes for h in on_device if h.pid in exclude)
        if device.total_bytes - device.used_bytes + ours >= needs:
            continue
        holders = tuple(h for h in on_device if h.pid not in exclude)
        used = device.used_bytes - ours
        return Busy(f"GPU {device.index}", used, device.total_bytes, holders, needs)
    return None


def _check_host(platform: Platform, needs: int) -> Busy | None:
    if platform.memory_topology == "unified":
        total = int(darwin.sysctl("hw.memsize").get("hw.memsize", "0") or 0)
        gib = darwin.available_gib()
        available = None if gib is None else int(gib * GIB)
    else:
        memory = cpu._meminfo()
        total, available = memory.get("MemTotal", 0), memory.get("MemAvailable")
    if available is None or not total or available >= needs:
        return None
    return Busy("RAM", total - available, total, (), needs)


def _compute_apps() -> list[tuple[str, Holder]]:
    out = run_probe(COMPUTE_APPS_QUERY)
    found = []
    for line in (out or "").strip().splitlines():
        try:
            pid, used, uuid = (field.strip() for field in line.split(","))
            found.append((uuid, Holder(int(pid), _process_name(int(pid)), int(used) * MIB)))
        except ValueError:
            continue
    return found


def _process_name(pid: int) -> str:
    if sys.platform == "darwin":
        out = run_probe(["ps", "-o", "comm=", "-p", str(pid)])
        return Path(out.strip()).name if out and out.strip() else "unknown"
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip() or "unknown"
    except OSError:
        return "unknown"


def _gb(value: int) -> str:
    return f"{value / GIB:.1f}".removesuffix(".0")


def busy_message(busy: Busy, minutes: int | None) -> str:
    """What is in use, by whom, and what to do about it; ``minutes`` is None for a server that
    keeps running."""
    free = max(busy.total_bytes - busy.used_bytes, 0)
    who = ", ".join(f"{h.name} (pid {h.pid})" for h in busy.holders)
    unseen = " by a process this user cannot see" if busy.where != "RAM" else ""
    by = f" by {who}" if who else unseen
    fix = "Stop it, or run on another machine." if busy.where != "RAM" else "Free some memory."
    runs = f"for ~{minutes} min" if minutes is not None else "that keeps running"
    return (
        f"{busy.where}: {_gb(busy.used_bytes)} of {_gb(busy.total_bytes)} GB in use{by}; "
        f"the server needs {_gb(busy.needs_bytes)} GB and {_gb(free)} GB is free. "
        f"Tensward starts its own server {runs} and needs the "
        f"{'RAM' if busy.where == 'RAM' else 'GPU'} free. It will not stop your server. {fix}"
    )


def refuse_if_busy(
    platform: Platform,
    gpus: tuple[str, ...] | None,
    needs: int,
    minutes: int | None,
    exclude: Collection[int] = (),
) -> None:
    """Raise a ``gpu_busy`` refusal when the memory a server needs is not free (``exclude``:
    pids whose memory counts as free)."""
    busy = check_free(platform, gpus, needs, exclude)
    if busy is not None:
        raise PreflightError(GPU_BUSY, busy_message(busy, minutes))
