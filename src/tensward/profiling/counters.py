"""Level 2: hardware counters (Nsight Compute) on the few kernels that dominate GPU time.

The trace says which kernels take the time; ``ncu`` says what the hardware was doing inside
them. This module is engine-neutral and holds the facts only: which kernels to profile
(:func:`select_kernels`), the exact ``ncu`` command (:func:`ncu_command`), the profiler
qualification (:func:`qualify`) and the parser for ncu's raw CSV page.
Interpreting them (is the kernel tensor-bound, is low occupancy a problem) is left to an
optional analysis plugin (see :mod:`tensward.extensions`).

Nothing here is a speed claim. ncu replays every kernel with flushed caches, and clocks are not
locked (containers may not lock them), so its durations differ from the clean run. A missing or
unavailable counter stays ``None``; a counter that is validly zero stays ``0.0``.

CLI reference: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
Metric reference: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
"""

from __future__ import annotations

import csv
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..platforms import MIB, detect_devices
from .trace import ATTENTION, Trace

COUNTER_KERNELS = 3  # kernels profiled by default
MIN_SHARE = 0.02  # a kernel below this share of GPU time is not worth a launch
LAUNCHES_PER_KERNEL = 4  # ncu --launch-count: profiled launches of each selected kernel
# --launch-count per group run. One engine step launches a GEMM per layer and projection (~130
# for a 32-layer model), and which variants run depends on the batch shapes of each step: on an
# L4, 400 launches covered the selected variant in one group's run but not in another's. 1,200
# single-pass launches reach steady-state prefill and decode in every run, for about a minute.
# (Filtering ncu by demangled name instead matched nothing for templated kernels.)
GROUP_LAUNCHES = 1200
COUNTER_REQUESTS = 32  # enough prefill and decode steps for every group; ncu ends at the count
COUNTER_TIME_LIMIT_S = 300.0  # hard bound on the profiled requests
PROBE_TIMEOUT_S = 180.0
MEMORY_TOLERANCE_MIB = 16  # above the cold level, GPU memory counts as not returned
MEMORY_SETTLE_S = 10.0
# The probe's matrix multiply runs a cuBLAS kernel whatever the GPU generation.
PROBE_KERNELS = "gemm|gemv|cutlass|nvjet|xmma|cublas"

# Short key -> ncu metric name (NCU 2025.3+, checked in the Profiling Guide's metric reference).
# Ratios are "pct_of_peak": the percentage of the unit's peak rate over the named denominator
# (cycles the SM was active, or cycles elapsed). The occupancy limits are blocks per SM.
METRICS: dict[str, str] = {
    "duration_us": "gpu__time_duration.sum",
    "grid": "launch__grid_size",
    "block": "launch__block_size",
    "sm_count": "launch__sm_count",
    "registers": "launch__registers_per_thread",
    "smem_static_b": "launch__shared_mem_per_block_static",
    "smem_dynamic_b": "launch__shared_mem_per_block_dynamic",
    "theoretical_occupancy_pct": "sm__maximum_warps_per_active_cycle_pct",
    "limit_registers": "launch__occupancy_limit_registers",
    "limit_shared_mem": "launch__occupancy_limit_shared_mem",
    "limit_warps": "launch__occupancy_limit_warps",
    "limit_blocks": "launch__occupancy_limit_blocks",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "eligible_warps": "smsp__warps_eligible.avg.per_cycle_active",
    "issue_active_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "dram_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "tensor_pct": "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "passes": "profiler__replayer_passes",  # replay passes ncu needed; always requested
}
# The launch facts are launch or device attributes (plus the kernel's duration), read without
# replaying the kernel; they ride along with the first group. Every other metric is a hardware
# counter, and ncu replays the kernel once per pass the counters of one request need. Replays
# are what left ~130 MiB of GPU memory behind on an L4, so the counters are collected in
# groups that each should fit ONE pass, and each group is qualified on its own. (SM throughput,
# a roll-up of many sub-unit counters, needed 7 passes on an L4 and is not collected.)
LAUNCH_FACTS = (
    "duration_us", "grid", "block", "sm_count", "registers", "smem_static_b", "smem_dynamic_b",
    "theoretical_occupancy_pct", "limit_registers", "limit_shared_mem", "limit_warps",
    "limit_blocks",
)  # fmt: skip
GROUPS: dict[str, tuple[str, ...]] = {
    "occupancy": (*LAUNCH_FACTS, "achieved_occupancy_pct", "eligible_warps", "issue_active_pct"),
    "memory": ("dram_pct",),
    "tensor": ("tensor_pct",),
}
REQUIRED_GROUP = "occupancy"  # without it there is nothing useful to report
GROUP_OF = {key: group for group, keys in GROUPS.items() for key in keys}
OCCUPANCY_LIMITS = {
    "registers": "limit_registers",
    "shared memory": "limit_shared_mem",
    "block size (warps)": "limit_warps",
    "blocks per SM": "limit_blocks",
}
# ncu's base time units -> microseconds (raw page, ``--print-units base``).
# ncu 2025.3 writes "ns" (seen on an L4); older releases spelled units out.
TIME_UNITS_US = {
    **{unit: 1e6 for unit in ("s", "second")},
    **{unit: 1e3 for unit in ("ms", "msecond")},
    **{unit: 1.0 for unit in ("us", "usecond")},
    **{unit: 1e-3 for unit in ("ns", "nsecond")},
}


@dataclass(frozen=True, slots=True)
class SelectedKernel:
    name: str  # as the trace reports it
    share: float  # of all kernel time in the trace
    calls: int


@dataclass(frozen=True, slots=True)
class Launch:
    """One profiled kernel launch: its ncu name and the counters (None = unavailable)."""

    name: str
    values: Mapping[str, float | None]


@dataclass(frozen=True, slots=True)
class KernelCounters:
    """A selected kernel and the median of each counter over its profiled launches."""

    name: str
    share: float
    calls: int
    ncu_name: str | None  # the name ncu prints for this kernel; None when ncu never saw it
    launches: int
    values: dict[str, float | None] = field(default_factory=dict)
    note: str | None = None  # why there are no counters, or a caveat
    unavailable: dict[str, str] = field(default_factory=dict)  # group -> why it has no values

    def value(self, key: str) -> float | None:
        return self.values.get(key)

    @property
    def occupancy_limiter(self) -> str | None:
        """The resource with the lowest blocks-per-SM limit (several named when they tie)."""
        limits = {n: self.values.get(k) for n, k in OCCUPANCY_LIMITS.items()}
        known = {n: v for n, v in limits.items() if v is not None}
        if not known:
            return None
        lowest = min(known.values())
        return " + ".join(n for n, v in known.items() if v == lowest)


@dataclass(frozen=True, slots=True)
class GroupQualification:
    """One metric group's probe: ``reason`` is why it is unavailable (None when it passed)."""

    name: str
    reason: str | None = None
    passes: float | None = None  # replay passes ncu needed (1 is required)
    eager_launches: int = 0
    graph_launches: int = 0
    memory_after_mib: int | None = None

    @property
    def passed(self) -> bool:
        return self.reason is None


@dataclass(frozen=True, slots=True)
class Qualification:
    """Whether this ncu, on this machine, can be trusted with each metric group.

    Counters are usable when the required group passed; ``reason`` is then None."""

    ncu_version: str | None = None
    memory_cold_mib: int | None = None
    groups: tuple[GroupQualification, ...] = ()

    @property
    def passed(self) -> bool:
        return any(g.name == REQUIRED_GROUP and g.passed for g in self.groups)

    @property
    def reason(self) -> str | None:
        if self.passed:
            return None
        required = next((g for g in self.groups if g.name == REQUIRED_GROUP), None)
        return f"group {REQUIRED_GROUP}: {required.reason if required else 'not probed'}"

    @property
    def available(self) -> list[str]:
        return [g.name for g in self.groups if g.passed]

    @property
    def unavailable(self) -> dict[str, str]:
        return {g.name: g.reason for g in self.groups if g.reason}


@dataclass(frozen=True, slots=True)
class CountersSummary:
    """``status``: ``ok`` or ``unavailable`` (``note`` says why). ``analysis`` is whatever the
    analysis plugin returned."""

    status: str
    note: str | None = None
    qualification: Qualification | None = None
    memory_after_mib: int | None = None
    kernels: tuple[KernelCounters, ...] = ()
    analysis: Any = None


@dataclass(frozen=True, slots=True)
class CountersResult:
    """What the kernel counters found, for the run's metrics and report."""

    summary: CountersSummary
    sections: Sequence[str]  # the analysis plugin's report lines
    extended: bool  # an analysis plugin was installed


# --------------------------------------------------------------------------------------
# Which kernels, and the command
# --------------------------------------------------------------------------------------


def select_kernels(trace: Trace, count: int = COUNTER_KERNELS) -> list[SelectedKernel]:
    """The kernels that take the most GPU time, one slot kept for attention.

    Kernel variants are told apart by their full name, so the GEMM that dominates prefill and
    the one that dominates decode are separate picks. Kernels under MIN_SHARE are not worth it.
    """
    spent: defaultdict[str, float] = defaultdict(float)
    calls: Counter[str] = Counter()
    for event in trace.kernels:
        spent[event.name] += event.end - event.start
        calls[event.name] += 1
    total = sum(spent.values())
    ranked = [
        name for name in sorted(spent, key=lambda n: -spent[n]) if spent[name] / total >= MIN_SHARE
    ]
    attention = [name for name in ranked if ATTENTION.search(name)][:1]
    others = [name for name in ranked if name not in attention][: count - len(attention)]
    chosen = sorted(attention + others, key=lambda name: -spent[name])
    return [SelectedKernel(name, spent[name] / total, calls[name]) for name in chosen]


def _function_head(name: str) -> str:
    """The kernel name without its parameter list (the first "(" outside the template)."""
    depth = 0
    for index, char in enumerate(name):
        depth += {"<": 1, ">": -1}.get(char, 0)
        if char == "(" and depth == 0:
            return name[:index]
    return name


def function_name(name: str) -> str:
    """The bare function of a kernel name: ``marlin::Marlin<...>(...)`` -> ``Marlin``."""
    return _function_head(name).partition("<")[0].rsplit("::", 1)[-1].removeprefix("void ")


def normalized(name: str) -> str:
    """A kernel name reduced to what trace and ncu agree on: the function's base name and its
    template arguments. The PyTorch trace and ncu spell the same demangled name differently, so
    this drops ``void``, the namespace of the function (ncu prints ``marlin::Marlin`` in one
    run and ``Marlin`` in another), the parameter list, whitespace, casts (``(bool)1``),
    integer suffixes (``5l``, ``4u``) and writes ``true``/``false`` as ``1``/``0``."""
    _, bracket, arguments = _function_head(name).partition("<")
    arguments = re.sub(r"\([a-z_ ]+\)(?=[\w(-])", "", arguments)
    arguments = re.sub(r"(?<=\d)[lLuU]+\b", "", arguments)
    arguments = re.sub(r"\btrue\b", "1", re.sub(r"\bfalse\b", "0", arguments))
    return function_name(name) + bracket + re.sub(r"\s+", "", arguments)


def match_ncu_name(trace_name: str, ncu_names: Sequence[str]) -> tuple[str | None, str | None]:
    """The name ncu printed for a trace kernel, or (None, why not). Exactly one distinct ncu
    name must normalize to the trace's; a kernel is never guessed."""
    found = {name for name in ncu_names if normalized(name) == normalized(trace_name)}
    if len(found) == 1:
        return found.pop(), None
    if found:
        return None, f"ambiguous: several ncu names match: {sorted(found)}"
    same = {name for name in ncu_names if function_name(name) == function_name(trace_name)}
    seen = sorted({short_name(name) for name in same or ncu_names})
    return None, (
        f"not seen by ncu; ncu printed for this {'function' if same else 'run (other functions)'}: "
        f"{'; '.join(seen) or 'nothing'}"
    )


def ncu_command(
    ncu: str, *, name: str, name_base: str, log_file: Path, group: str, launches: int, gated: bool
) -> list[str]:
    """``ncu`` with one metric group and bounded launches; append the command to profile.

    ``name`` is ncu's ``--kernel-name`` (``regex:...`` or an exact name) on ``name_base``
    (``function`` or ``demangled``). The CSV always prints the demangled name, which is what
    :func:`match_ncu_name` matches. ``--graph-profiling node`` profiles the kernels inside CUDA
    graphs one by one (the default in current versions; explicit here). ``--target-processes
    all`` follows the engine's child processes. ``--clock-control none`` keeps the GPU's own
    clocks: by default ncu locks them, which a container is not allowed to do (the first L4 run
    failed with "Failed to lock GPU clock frequencies"). ``gated`` adds ``--profile-from-start
    off``: only kernels launched between the engine's start and stop profile requests are
    profiled, so none of the startup kernels use up the launch count. The raw CSV page is written
    to ``log_file`` only when ncu exits, which it does itself once ``launches`` are profiled.
    """
    return [
        ncu,
        "--target-processes", "all",
        "--graph-profiling", "node",
        "--clock-control", "none",
        *(["--profile-from-start", "off"] if gated else []),
        "--kernel-name-base", name_base,
        "--kernel-name", name,
        "--print-kernel-base", "demangled",
        "--launch-count", str(launches),
        "--metrics", ",".join(METRICS[key] for key in (*GROUPS[group], "passes")),
        "--csv", "--page", "raw", "--print-units", "base",
        "--log-file", str(log_file),
    ]  # fmt: skip


def group_command(
    ncu: str, kernels: Sequence[SelectedKernel], log_file: Path, group: str
) -> list[str]:
    """``ncu`` collecting one metric group for the first gated launches of the selected kernels'
    functions. The filter uses bare function names, which every spelling shares; the rows are
    attributed to kernels afterwards by :func:`match_ncu_name` (filtering on ncu's full
    demangled name matched nothing on a real A10G)."""
    functions = sorted({function_name(kernel.name) for kernel in kernels})
    return ncu_command(
        ncu, name=f"regex:^({'|'.join(map(re.escape, functions))})$", name_base="function",
        log_file=log_file, group=group, launches=GROUP_LAUNCHES, gated=True,
    )  # fmt: skip


# --------------------------------------------------------------------------------------
# Parsing ncu's raw CSV page
# --------------------------------------------------------------------------------------


def _number(text: str | None) -> float | None:
    """A CSV cell as a number; empty, ``n/a`` and anything unreadable are None, zero is 0.0."""
    try:
        return float((text or "").replace(",", "").strip())
    except ValueError:
        return None


def parse_raw_csv(text: str) -> list[Launch]:
    """The launches of ``ncu --csv --page raw``: a header row, a units row, then one row per
    launch with a column per requested metric."""
    lines = [line for line in text.splitlines() if line and not line.startswith("==")]
    launches: list[Launch] = []
    units: Mapping[str, str] = {}
    for row in csv.DictReader(lines):
        if not row.get("ID"):  # the units row has no launch id
            units = {key: (value or "").strip() for key, value in row.items() if key}
            continue
        values = {key: _number(row.get(metric)) for key, metric in METRICS.items()}
        factor = TIME_UNITS_US.get(units.get(METRICS["duration_us"], ""))
        duration = values["duration_us"]
        values["duration_us"] = None if duration is None or factor is None else duration * factor
        launches.append(Launch(row.get("Kernel Name", ""), values))
    return launches


def kernel_counters(
    selected: SelectedKernel,
    ncu_name: str | None,
    launches: Sequence[Launch],
    note: str | None = None,
) -> KernelCounters:
    """Median of each counter over the launches; a counter no launch reported stays None."""
    values: dict[str, float | None] = {}
    for key in METRICS:
        seen = [v for launch in launches if (v := launch.values[key]) is not None]
        values[key] = statistics.median(seen) if seen else None
    if not launches:
        note = note or "no launch matched the kernel filter"
        values = {}
    return KernelCounters(
        selected.name, selected.share, selected.calls, ncu_name, len(launches), values, note
    )


def merge_groups(
    selected: SelectedKernel,
    ncu_name: str | None,
    parts: Mapping[str, KernelCounters],
    unavailable: Mapping[str, str],
) -> KernelCounters:
    """One kernel from its per-group profiles. A group that was not qualified, not profiled or
    returned nothing keeps its counters None (unknown) and its reason in ``unavailable``."""
    values: dict[str, float | None] = {}
    reasons: dict[str, str] = {}
    for group, keys in GROUPS.items():
        part = parts.get(group)
        if part is not None and any(part.value(key) is not None for key in keys):
            values |= {key: part.value(key) for key in keys}
            values[f"{group}_passes"] = part.value("passes")
            continue
        values |= dict.fromkeys(keys)
        reasons[group] = (part.note if part else unavailable.get(group)) or "no values reported"
    launches = max((part.launches for part in parts.values()), default=0)
    if not launches:
        return KernelCounters(
            selected.name, selected.share, selected.calls, ncu_name, 0, {},
            next((part.note for part in parts.values() if part.note), "no kernel was profiled"),
            reasons,
        )  # fmt: skip
    return KernelCounters(
        selected.name, selected.share, selected.calls, ncu_name, launches, values, None, reasons
    )


# --------------------------------------------------------------------------------------
# Profiler qualification
# --------------------------------------------------------------------------------------

PROBE = """
import sys, torch
a = torch.ones(2048, 2048, device="cuda", dtype=torch.float16)
torch.matmul(a, a)
torch.cuda.synchronize()
if sys.argv[1] == "graph":
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.matmul(a, a)
    graph.replay()
    torch.cuda.synchronize()
"""


PROBE_ADVICE = {
    "ERR_NVGPUCTRPERM": "GPU performance counters are restricted to admin users: run as root / "
    "in a container with --cap-add SYS_ADMIN, or set NVreg_RestrictProfilingToAdminUsers=0; "
    "see docs/cli.md#counters",
    "No module named 'torch'": "counters need Tensward installed in the same Python environment "
    "as the engine (torch)",
}


def probe_failure_reason(stderr: str, ncu_log: str) -> str:
    """Why a failed probe failed: a known cause in words the user can act on, else ncu's own
    ``==ERROR==``/``==WARNING==`` lines and the tail of the probe's stderr."""
    lines = [ln for ln in ncu_log.splitlines() if ln.startswith(("==ERROR==", "==WARNING=="))]
    for marker, advice in PROBE_ADVICE.items():
        if marker in stderr or marker in "\n".join(lines):
            return advice
    return " | ".join([*lines, *stderr.strip().splitlines()[-3:]]) or "no output"


def gpu_memory_mib() -> int | None:
    """Memory in use on the first device, or None when the platform cannot say."""
    devices = detect_devices()
    return devices[0].used_bytes // MIB if devices else None


def memory_returned(cold_mib: int | None) -> tuple[bool, int | None]:
    """Wait briefly for GPU memory to fall back to the cold level; (returned, last reading).
    Unreadable memory never counts as returned."""
    deadline = time.monotonic() + MEMORY_SETTLE_S
    while True:
        used = gpu_memory_mib()
        if cold_mib is None or used is None:
            return False, used
        if used <= cold_mib + MEMORY_TOLERANCE_MIB:
            return True, used
        if time.monotonic() >= deadline:
            return False, used
        time.sleep(0.5)


def ncu_version(ncu: str) -> str | None:
    try:
        done = subprocess.run(
            [ncu, "--version"], capture_output=True, text=True, check=False, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    found = re.search(r"Version\s+(\S+)", done.stdout)
    return found.group(1) if found else None


def _probe(ncu: str, group: str, mode: str, directory: Path) -> tuple[list[Launch], str | None]:
    """Profile a tiny matrix multiply (``eager`` or replayed from a CUDA ``graph``) with the
    exact metric group the real run will use; the launches, or why there are none."""
    log = directory / f"probe_{group}_{mode}.csv"
    command = ncu_command(
        ncu,
        name=f"regex:{PROBE_KERNELS}",
        name_base="demangled",
        log_file=log,
        group=group,
        launches=LAUNCHES_PER_KERNEL,
        gated=False,
    )
    command += [sys.executable, "-c", PROBE, mode]
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=PROBE_TIMEOUT_S
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return [], f"{mode} probe: {type(error).__name__}: {error}"
    text = log.read_text("utf-8", errors="replace") if log.exists() else ""
    if done.returncode != 0:
        why = probe_failure_reason(done.stderr, text)
        return [], f"{mode} probe exited {done.returncode}: {why}"
    return parse_raw_csv(text), None


def qualify(ncu: str, directory: Path) -> Qualification:
    """Check this ncu before trusting it with the model, one metric group at a time.

    Each group runs a tiny no-model probe, eagerly and inside a CUDA graph, as its own ncu
    invocation. A group is qualified only if every metric is present in the output, ncu needed
    a single replay pass, the graph run shows the graph's kernels (node profiling works), and
    GPU memory returns to the cold level after each ncu exits. A failed group is unavailable
    with its reason and the others continue, except that the required group failing ends the
    check, and so does memory left behind (it would fail every later group for the same cause).
    Nothing is relaxed.
    """
    cold = gpu_memory_mib()
    results: list[GroupQualification] = []
    for group in GROUPS:
        result, returned = _qualify_group(ncu, group, directory, cold)
        results.append(result)
        if not returned or (group == REQUIRED_GROUP and not result.passed):
            break
    skipped = [g for g in GROUPS if g not in {r.name for r in results}]
    results += [GroupQualification(g, "not probed: an earlier probe failed") for g in skipped]
    return Qualification(ncu_version(ncu), cold, tuple(results))


def _qualify_group(
    ncu: str, group: str, directory: Path, cold: int | None
) -> tuple[GroupQualification, bool]:
    """The group's result and whether GPU memory was back at the cold level afterwards."""
    eager = graph = 0
    passes: float | None = None
    used: int | None = None
    failure = None
    for mode in ("eager", "graph"):
        launches, failure = _probe(ncu, group, mode, directory)
        returned, used = memory_returned(cold)
        failure = failure or _probe_failure(group, mode, launches, eager)
        if failure is None and not returned:
            failure = f"GPU memory did not return to the cold level after the {mode} probe "
            failure += f"({cold} -> {used} MiB)"
        eager, graph = (len(launches), graph) if mode == "eager" else (eager, len(launches))
        seen = [p for launch in launches if (p := launch.values["passes"]) is not None]
        passes = max(seen, default=passes)
        if failure:
            break
    return GroupQualification(group, failure, passes, eager, graph, used), returned


def _probe_failure(
    group: str, mode: str, launches: Sequence[Launch], eager_launches: int
) -> str | None:
    if not launches:
        return f"{mode} probe: ncu reported no kernel launches"
    wanted = (*GROUPS[group], "passes")
    missing = [
        METRICS[key] for key in wanted if all(launch.values[key] is None for launch in launches)
    ]
    if missing:
        return f"{mode} probe: metrics missing from the output: {', '.join(missing)}"
    passes = max(launch.values["passes"] or 0 for launch in launches)
    if passes != 1:
        return f"{mode} probe: ncu needed {passes:g} replay passes, not 1"
    if mode == "graph" and len(launches) <= eager_launches:
        return "the graph probe showed no kernels beyond the eager one (graph nodes not profiled)"
    return None


# --------------------------------------------------------------------------------------
# Kernel names
# --------------------------------------------------------------------------------------


def short_name(name: str) -> str:
    return _function_head(name).removeprefix("void ")
