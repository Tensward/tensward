"""macOS host facts and process identity, read from stock tools (``sysctl``, ``vm_stat``, ``ps``,
``pmset``, ``sw_vers``, ``ioreg``, ``system_profiler``) and the Mach-O header of a binary.

Every source that cannot be read is None. The tools are run through :data:`RUN`, which tests
replace with a runner fed recorded output; the caches below are keyed by that object.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

from .protocol import GIB, MIB, ProcessSample

if TYPE_CHECKING:
    from .cpu import HostFacts

Runner = Callable[[Sequence[str]], str | None]

TOOL_TIMEOUT_S = 5
SYSTEM_READ_PERIOD_S = 5.0
FEATURES = {
    "hw.optional.arm.FEAT_DotProd": "asimddp",
    "hw.optional.arm.FEAT_I8MM": "i8mm",
    "hw.optional.arm.FEAT_BF16": "bf16",
    "hw.optional.arm.FEAT_SME": "sme",
    "hw.optional.arm.FEAT_SME2": "sme2",
}
FAT_MAGIC = 0xCAFEBABE
MACH_O_64 = b"\xcf\xfa\xed\xfe"
CPU_TYPES: dict[int, Literal["arm64", "x86_64"]] = {0x0100000C: "arm64", 0x01000007: "x86_64"}


def _run(argv: Sequence[str]) -> str | None:
    try:
        done = subprocess.run(  # noqa: S603 - list argv, no shell
            list(argv),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=TOOL_TIMEOUT_S,
            stdin=subprocess.DEVNULL,
            env={**os.environ, "TZ": "UTC", "LC_ALL": "C"},
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


RUN: Runner = _run

_CACHE: dict[tuple[Runner, str], Any] = {}
_SYSTEM: dict[Runner, tuple[float, dict[str, Any]]] = {}
_FAULT_KEYWORD: dict[Runner, str | None] = {}  # per runner: the ps keyword that counts faults


def _cached(key: str, compute: Callable[[], Any]) -> Any:
    slot = (RUN, key)
    if slot not in _CACHE:
        _CACHE[slot] = compute()
    return _CACHE[slot]


def _is_darwin() -> bool:
    return sys.platform == "darwin"


def sysctl(*names: str) -> dict[str, str]:
    """The values of the named keys, one ``sysctl -n`` call each; a key that is missing or
    unreadable is absent."""
    found = {}
    for name in names:
        text = RUN(["sysctl", "-n", name])
        if text is not None and text.strip():
            found[name] = text.strip()
    return found


def _arch_keys() -> dict[str, str]:
    """The CPU-type keys; a read that names neither CPU family is not remembered."""
    slot = (RUN, "arch")
    if slot not in _CACHE:
        keys = sysctl("hw.optional.arm64", "hw.optional.x86_64", "sysctl.proc_translated")
        if not {"hw.optional.arm64", "hw.optional.x86_64"} & keys.keys():
            return keys
        _CACHE[slot] = keys
    cached: dict[str, str] = _CACHE[slot]
    return cached


def on_mac() -> bool:
    """A native arm64 macOS process."""
    if not _is_darwin():
        return False
    keys = _arch_keys()
    return keys.get("hw.optional.arm64") == "1" and keys.get("sysctl.proc_translated") != "1"


def unsupported_mac() -> str | None:
    """Why this Mac cannot be analysed, or None off macOS and on a native arm64 Mac."""
    if not _is_darwin():
        return None
    keys = _arch_keys()
    if keys.get("sysctl.proc_translated") == "1":
        return "this Python runs under Rosetta; use an arm64 Python"
    if keys.get("hw.optional.arm64") == "1":
        return None
    if "hw.optional.arm64" in keys or keys.get("hw.optional.x86_64") == "1":
        return "Intel Macs are not supported in this version"
    return "could not read the CPU type of this Mac"


def _int(text: str | None) -> int | None:
    return int(text) if text is not None and text.isdecimal() else None


def _vm_stat() -> dict[str, int]:
    """Page counts by name, plus ``page_size`` in bytes."""
    text = RUN(["vm_stat"])
    if not text:
        return {}
    size = re.search(r"page size of (\d+) bytes", text)
    if not size:
        return {}
    found = {"page_size": int(size[1])}
    for match in re.finditer(r'^"?([^":\n]+)"?:\s+(\d+)\.?\s*$', text, re.MULTILINE):
        found[match[1].strip()] = int(match[2])
    return found


def _available_mib(pages: dict[str, int]) -> float | None:
    keys = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
    if "page_size" not in pages or any(key not in pages for key in keys):
        return None
    return sum(pages[key] for key in keys) * pages["page_size"] / MIB


def available_gib() -> float | None:
    """Memory a process can take without paging: free, inactive, speculative and purgeable."""
    mib = _available_mib(_vm_stat())
    return None if mib is None else mib / 1024


def swap_usage() -> tuple[float, float] | None:
    """(total, used) swap in MiB from ``vm.swapusage``."""
    text = sysctl("vm.swapusage").get("vm.swapusage", "")
    total = re.search(r"total = ([\d.]+)M", text)
    used = re.search(r"used = ([\d.]+)M", text)
    return (float(total[1]), float(used[1])) if total and used else None


def _cpu_seconds(text: str) -> float | None:
    try:
        parts = [float(part) for part in text.split(":")]
    except ValueError:
        return None
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def os_build() -> tuple[str | None, str | None]:
    text = RUN(["sw_vers"]) or ""
    version = re.search(r"^ProductVersion:\s*(\S+)", text, re.MULTILINE)
    build = re.search(r"^BuildVersion:\s*(\S+)", text, re.MULTILINE)
    return (version[1] if version else None, build[1] if build else None)


def power_state() -> tuple[bool | None, bool | None]:
    """(low power mode on, CPU speed held below 100 %). The second is a system state, not proof
    that the GPU throttled."""
    settings = RUN(["pmset", "-g"])
    mode = re.search(r"^\s*(?:lowpowermode|powermode)\s+(\d)", settings or "", re.MULTILINE)
    thermal = RUN(["pmset", "-g", "therm"])
    limit = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", thermal or "")
    return (
        None if mode is None else mode[1] == "1",
        None if thermal is None else bool(limit and int(limit[1]) < 100),
    )


def gpu_cores() -> int | None:
    def read() -> int | None:
        registry = RUN(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"])
        found = re.search(r'"gpu-core-count"\s*=\s*(\d+)', registry or "")
        if found:
            return int(found[1])
        try:
            profile = RUN(["system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"])
            return _int(str(json.loads(profile or "")["SPDisplaysDataType"][0]["sppci_cores"]))
        except (ValueError, KeyError, IndexError, TypeError):
            return None

    cores: int | None = _cached("gpu_cores", read)
    return cores


def platform_uuid_digest() -> str | None:
    """A short digest of the platform UUID, which identifies the machine without being its
    serial number."""
    text = RUN(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"])
    found = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', text or "")
    return hashlib.sha256(found[1].encode()).hexdigest()[:16] if found else None


def process_start(pid: int) -> str | None:
    """The start time ``ps`` prints for ``pid``; None when it is gone or a zombie."""
    argv = ["ps", "-o", "stat=,lstart=", "-p", str(pid)]
    text = RUN(argv) or RUN(argv)
    stat, _, started = (text or "").strip().partition(" ")
    started = started.strip()
    return None if not started or stat.startswith("Z") else started


def mach_o_arch(path: Path) -> Literal["arm64", "x86_64", "universal"] | None:
    """The architecture in a Mach-O header; None for anything else, such as a script."""
    try:
        with open(path, "rb") as binary:
            head = binary.read(8)
            if head[:4] == MACH_O_64:
                return CPU_TYPES.get(int.from_bytes(head[4:8], "little"))
            if len(head) < 8 or int.from_bytes(head[:4], "big") != FAT_MAGIC:
                return None
            count = int.from_bytes(head[4:8], "big")
            if not 1 <= count < 16:  # a Java class file has the same magic
                return None
            entries = [int.from_bytes(binary.read(20)[:4], "big") for _ in range(count)]
            kinds = {CPU_TYPES.get(cpu_type) for cpu_type in entries}
    except OSError:
        return None
    return "universal" if len(kinds) > 1 else kinds.pop()


def host_facts(pid: int | None = None) -> HostFacts:
    from .cpu import HostFacts

    keys = sysctl(
        "hw.logicalcpu", "hw.physicalcpu", "hw.perflevel0.physicalcpu",
        "hw.perflevel1.physicalcpu", "hw.memsize", "iogpu.wired_limit_mb", *FEATURES,
    )  # fmt: skip
    logical = _int(keys.get("hw.logicalcpu")) or os.cpu_count() or 1
    available = _available_mib(_vm_stat())
    swap = swap_usage()
    soft = resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]
    low_power, thermal = power_state()
    version, build = os_build()
    return HostFacts(
        logical_cpus=logical,
        physical_cores=_int(keys.get("hw.physicalcpu")) or logical,
        sockets=1,
        smt=False,
        quota_cores=None,
        cpuset_cpus=None,
        usable_cpus=logical,
        mem_gib=(_int(keys.get("hw.memsize")) or 0) / GIB,
        mem_available_gib=(available or 0.0) / 1024,
        swap_gib=(swap[0] if swap else 0.0) / 1024,
        cgroup_mem_gib=None,
        memlock_gib=None if soft == resource.RLIM_INFINITY else soft / GIB,
        isa=frozenset(flag for key, flag in FEATURES.items() if keys.get(key) == "1"),
        numa_nodes=1,
        p_cores=_int(keys.get("hw.perflevel0.physicalcpu")),
        e_cores=_int(keys.get("hw.perflevel1.physicalcpu")),
        low_power_mode=low_power,
        thermal_limited=thermal,
        wired_limit_mb=_int(keys.get("iogpu.wired_limit_mb")),
        os_family="macos",
        os_build=build or version,
    )


def _system_read(boundary: bool) -> dict[str, Any]:
    """The system-wide figures of a sample: ``vm_stat`` and two sysctls, re-read at most every
    few seconds unless this is the first or last sample of a window."""
    now = time.monotonic()
    held = _SYSTEM.get(RUN)
    if held is not None and not boundary and now - held[0] < SYSTEM_READ_PERIOD_S:
        return held[1]
    pages = _vm_stat()
    swap = swap_usage()
    pressure = "kern.memorystatus_vm_pressure_level"
    level = _int(sysctl(pressure).get(pressure))
    size = pages.get("page_size")
    compressed = pages.get("Pages occupied by compressor")
    figures = {
        "mem_available_mib": _available_mib(pages),
        "swap_used_mib": swap[1] if swap else None,
        "compressed_mib": None if size is None or compressed is None else compressed * size / MIB,
        "pressure_level": level,
    }
    _SYSTEM[RUN] = (now, figures)
    return figures


def _ps_fields(pid: int) -> list[str]:
    """``rss, utime, stime`` and a fault count of ``pid``. macOS ``ps`` prints ``-`` for
    ``majflt``; ``pagein`` counts the page-ins instead, and is remembered per runner. A ``ps``
    that rejects the fault keyword is read without it."""
    keyword = _FAULT_KEYWORD.get(RUN, "majflt")
    while True:
        extra = f",{keyword}=" if keyword else ""
        text = RUN(["ps", "-o", f"rss=,utime=,stime={extra}", "-p", str(pid)])
        fields = (text or "").split()
        if text is None and keyword:
            keyword = None
        elif len(fields) > 3 and fields[3] == "-" and keyword == "majflt":
            keyword = "pagein"
        else:
            break
    if fields:
        _FAULT_KEYWORD[RUN] = keyword
    return fields


def sample_process(pid: int, *, boundary: bool = False) -> ProcessSample:
    """CPU time, resident set and major faults of ``pid`` and the host's memory figures, read
    now."""
    fields = _ps_fields(pid)
    rss = _int(fields[0]) if len(fields) > 0 else None
    return ProcessSample(
        at_ns=time.monotonic_ns(),
        cpu_user_s=_cpu_seconds(fields[1]) if len(fields) > 1 else None,
        cpu_sys_s=_cpu_seconds(fields[2]) if len(fields) > 2 else None,
        rss_mib=None if rss is None else rss / 1024,
        major_faults=_int(fields[3]) if len(fields) > 3 else None,
        **_system_read(boundary),
    )
