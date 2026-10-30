"""The host CPU as a platform: no devices to detect, but the facts an engine running on it
depends on (usable cores, memory, locked-memory limit, instruction sets) and a sampler for a
process's CPU time and memory.

Every source is Linux ``/proc``, ``/sys`` or cgroup files; one that cannot be read is None.
"""

from __future__ import annotations

import math
import os
import re
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .protocol import GIB, Device, HardwareSpec, ProcessSample

CGROUP_ROOT = Path("/sys/fs/cgroup")
CPU_SYSFS = Path("/sys/devices/system/cpu")
NODE_SYSFS = Path("/sys/devices/system/node")
ISA_FLAGS = frozenset({
    "avx", "avx2", "fma", "f16c", "avx_vnni", "avx512f", "avx512bw", "avx512vl", "avx512_vnni",
    "avx512_bf16", "amx_tile", "amx_int8", "amx_bf16",
    "asimd", "asimddp", "asimdhp", "i8mm", "bf16", "sve", "sve2",
})  # fmt: skip


@dataclass(frozen=True, slots=True, kw_only=True)
class HostFacts:
    """What the host offers an engine that runs on its CPU. ``usable_cpus`` is the least of the
    logical CPUs, the cgroup quota, the cpuset and the affinity of the process asked about:
    llama.cpp's default thread count looks at none but the first."""

    logical_cpus: int
    physical_cores: int
    sockets: int
    smt: bool
    quota_cores: float | None  # cgroup cpu.max quota / period
    cpuset_cpus: int | None  # CPUs in cpuset.cpus.effective
    usable_cpus: int
    mem_gib: float
    mem_available_gib: float
    swap_gib: float
    cgroup_mem_gib: float | None  # memory.max
    memlock_gib: float | None  # RLIMIT_MEMLOCK; None is unlimited or unreadable
    isa: frozenset[str]
    numa_nodes: int


def _read(path: Path | str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _cpu_list(text: str | None) -> int | None:
    """How many CPUs a kernel list such as ``0-3,8`` names."""
    if not text:
        return None
    try:
        return sum(
            int(hi) - int(lo) + 1 if sep else 1
            for lo, sep, hi in (part.partition("-") for part in text.split(","))
        )
    except ValueError:
        return None


def _meminfo() -> dict[str, int]:
    """/proc/meminfo in bytes."""
    found: dict[str, int] = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        if match := re.match(r"(\w+):\s+(\d+)\s*kB", line):
            found[match[1]] = int(match[2]) * 1024
    return found


def _cgroup_dirs(pid: int | None) -> list[Path]:
    """The cgroup directories the process belongs to: the v2 one first, then v1 controllers."""
    dirs = []
    for line in (_read(f"/proc/{pid or 'self'}/cgroup") or "").splitlines():
        _, _, rest = line.partition(":")
        controllers, _, path = rest.partition(":")
        if not controllers:
            dirs.append(CGROUP_ROOT / path.lstrip("/"))
        else:
            for controller in controllers.split(","):
                dirs.append(CGROUP_ROOT / controller / path.lstrip("/"))
    return dirs


def _cgroup_file(dirs: Sequence[Path], name: str) -> str | None:
    """``name`` in the process's own cgroup, else in its ancestors up to the root, since a
    limit set above applies below."""
    for base in dirs:
        for directory in (base, *base.parents):
            if directory == CGROUP_ROOT.parent:
                break
            if (text := _read(directory / name)) is not None:
                return text
    return None


def _quota_cores(dirs: Sequence[Path]) -> float | None:
    limits = []
    for base in dirs:
        for directory in (base, *base.parents):
            if directory == CGROUP_ROOT.parent:
                break
            if (text := _read(directory / "cpu.max")) is not None:
                quota, _, period = text.partition(" ")
                if quota != "max" and period.isdecimal() and int(period) > 0 and quota.isdecimal():
                    limits.append(int(quota) / int(period))
            elif (quota_text := _read(directory / "cpu.cfs_quota_us")) is not None:
                period_text = _read(directory / "cpu.cfs_period_us") or ""
                if quota_text.lstrip("-").isdecimal() and period_text.isdecimal():
                    if int(quota_text) > 0 and int(period_text) > 0:
                        limits.append(int(quota_text) / int(period_text))
    return min(limits) if limits else None


def _topology() -> tuple[int, int, int]:
    """(physical cores, sockets, logical CPUs) from sysfs; (0, 0, 0) when unreadable."""
    cores: set[tuple[str, str]] = set()
    sockets: set[str] = set()
    logical = 0
    for topology in sorted(CPU_SYSFS.glob("cpu[0-9]*/topology")):
        package, core = _read(topology / "physical_package_id"), _read(topology / "core_id")
        if package is None or core is None:
            continue
        logical += 1
        sockets.add(package)
        cores.add((package, core))
    return len(cores), len(sockets), logical


def _isa() -> frozenset[str]:
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        key, _, value = line.partition(":")
        if key.strip() in ("flags", "Features"):
            return frozenset(value.split()) & ISA_FLAGS
    return frozenset()


def _memlock(pid: int | None) -> float | None:
    soft: int | str | None
    if pid is None:
        soft = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
        soft = None if soft == resource.RLIM_INFINITY else soft
    else:
        match = re.search(r"Max locked memory\s+(\S+)", _read(f"/proc/{pid}/limits") or "")
        soft = int(match[1]) if match and match[1].isdecimal() else None
    return None if soft is None else int(soft) / GIB


def _affinity(pid: int | None) -> int | None:
    try:
        return len(os.sched_getaffinity(pid or 0))
    except (OSError, AttributeError):
        return None


def host_facts(pid: int | None = None) -> HostFacts:
    """The host's CPU and memory facts. With ``pid``, the cgroup, affinity and locked-memory
    limit are those of that process, otherwise of this one."""
    cores, sockets, topology_cpus = _topology()
    logical = os.cpu_count() or topology_cpus or 1
    dirs = _cgroup_dirs(pid)
    quota = _quota_cores(dirs)
    cpuset = _cpu_list(_cgroup_file(dirs, "cpuset.cpus.effective"))
    terms = [logical, cpuset, _affinity(pid)]
    if quota is not None:
        terms.append(max(1, math.floor(quota)))
    memory = _meminfo()
    cgroup_mem = _cgroup_file(dirs, "memory.max") or _cgroup_file(dirs, "memory.limit_in_bytes")
    cgroup_limit = int(cgroup_mem) if cgroup_mem and cgroup_mem.isdecimal() else None
    if cgroup_limit is not None and cgroup_limit >= 2**60:
        cgroup_limit = None  # the v1 spelling of "no limit"
    cores = cores or logical
    return HostFacts(
        logical_cpus=logical,
        physical_cores=cores,
        sockets=sockets or 1,
        smt=logical > cores,
        quota_cores=quota,
        cpuset_cpus=cpuset,
        usable_cpus=max(1, min(t for t in terms if t is not None)),
        mem_gib=memory.get("MemTotal", 0) / GIB,
        mem_available_gib=memory.get("MemAvailable", 0) / GIB,
        swap_gib=memory.get("SwapTotal", 0) / GIB,
        cgroup_mem_gib=None if cgroup_limit is None else cgroup_limit / GIB,
        memlock_gib=_memlock(pid),
        isa=_isa(),
        numa_nodes=len(list(NODE_SYSFS.glob("node[0-9]*"))) or 1,
    )


def _stat_fields(pid: int) -> list[str] | None:
    text = _read(f"/proc/{pid}/stat")
    # the command name in parentheses may hold spaces; the fields after it start at "state"
    return text.rpartition(")")[2].split() if text else None


class CpuPlatform:
    name = "cpu"
    label = "CPU"
    bandwidth_figures = False
    accelerator = False

    def detect(self) -> tuple[Device, ...]:
        return ()

    def spec(self, device_name: str) -> HardwareSpec | None:
        return None

    def derived_bandwidth(self, index: int) -> float | None:
        return None

    def identity(self, device: Device) -> dict[str, Any]:
        return {}

    def describe(self, device: Device, identity: Mapping[str, Any]) -> str:
        return device.name

    def select(
        self, devices: tuple[Device, ...], selection: tuple[str, ...] | None
    ) -> tuple[tuple[Device, ...], tuple[str, ...]]:
        return devices, tuple(selection or ())

    def selection_refusal(self, selection: Sequence[str]) -> str | None:
        return "the CPU platform has no devices to select" if selection else None

    def select_env(
        self, indices: tuple[str, ...] | None, environ: Mapping[str, str]
    ) -> dict[str, str]:
        return {}

    def docker_device_args(self, indices: tuple[str, ...] | None) -> list[str]:
        return []

    def sampler_commands(self, indices: tuple[str, ...] | None) -> tuple[tuple[str, ...], ...]:
        return ()

    def parse_sample(self, line: str, at_ns: int) -> None:
        return None

    def sample_process(self, pid: int) -> ProcessSample:
        """CPU time, resident set and major faults of ``pid`` and the host's available memory,
        read now."""
        stat = _stat_fields(pid)
        ticks = os.sysconf("SC_CLK_TCK")
        rss = re.search(r"VmRSS:\s+(\d+)\s*kB", _read(f"/proc/{pid}/status") or "")
        available = _meminfo().get("MemAvailable")
        return ProcessSample(
            at_ns=time.monotonic_ns(),
            cpu_user_s=int(stat[11]) / ticks if stat and len(stat) > 12 else None,
            cpu_sys_s=int(stat[12]) / ticks if stat and len(stat) > 12 else None,
            rss_mib=int(rss[1]) / 1024 if rss else None,
            major_faults=int(stat[9]) if stat and len(stat) > 12 else None,
            mem_available_mib=None if available is None else available / 2**20,
        )


CPU = CpuPlatform()
