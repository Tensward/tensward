"""The profiled launches of ``analyse --trace`` and ``--counters``, each on its own server."""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import httpx

from ..engines import Engine
from ..errors import AnalyseFailure
from ..extensions import extend, load_extender
from ..files import write_json
from ..images import ImageSource
from ..measure import serving, spec_for, timed_requests, warm_up
from ..measurement import Measurement
from ..progress import PhaseStarted, emit
from ..project import ResolvedProject
from ..runtime import (
    Runtime,
    RuntimeFailure,
    ServeSpec,
    profiler,
)
from ..settings import Settings
from .counters import (
    COUNTER_REQUESTS,
    COUNTER_TIME_LIMIT_S,
    REQUIRED_GROUP,
    CountersResult,
    CountersSummary,
    KernelCounters,
    Launch,
    group_command,
    kernel_counters,
    match_ncu_name,
    memory_returned,
    merge_groups,
    parse_raw_csv,
    qualify,
    select_kernels,
)
from .trace import (
    Trace,
    TraceResult,
    TraceSummary,
    TraceUnavailable,
    read_trace,
    summarize_trace,
)

TRACE_REQUESTS = 16  # the traced slice is short: profiler overhead and trace size grow with it
TRACE_TIME_LIMIT_S = 120.0  # hard bound on the profiled requests
INTERRUPTED = "interrupted; the measurement before it is complete"


@dataclass(frozen=True, slots=True)
class Profiles:
    """The profiled launches of one run. ``interrupted``: a Ctrl-C stopped them, and what it
    stopped is marked unavailable so the run can still be written before the interrupt is
    raised again."""

    trace: TraceResult
    counters: CountersResult | None
    interrupted: bool = False


def add_profiles(
    project: ResolvedProject,
    engine: Engine,
    runtime: Runtime,
    settings: Settings,
    run_dir: Path,
    measurement: Measurement,
    ready_timeout_s: float,
    *,
    counters: bool,
) -> Profiles:
    """The trace and, with ``counters``, the kernel counters of the trace's top kernels."""
    traced = TraceResult(TraceSummary("unavailable", INTERRUPTED), (), False, None)
    stopped = CountersResult(CountersSummary("unavailable", INTERRUPTED), (), False)
    counted = stopped if counters else None
    try:
        traced = add_trace(
            project, engine, runtime, settings, run_dir, measurement, ready_timeout_s
        )
        if counters:
            counted = add_counters(
                project, engine, runtime, settings, run_dir,
                replace(measurement, trace=traced.summary), traced.trace, ready_timeout_s,
            )  # fmt: skip
    except KeyboardInterrupt:
        return Profiles(traced, counted, interrupted=True)
    return Profiles(traced, counted)


# --------------------------------------------------------------------------------------
# Trace (Level 1)
# --------------------------------------------------------------------------------------


def add_trace(
    project: ResolvedProject,
    engine: Engine,
    runtime: Runtime,
    settings: Settings,
    run_dir: Path,
    measurement: Measurement,
    ready_timeout_s: float,
) -> TraceResult:
    """Profile a short slice on its own launch; a failed trace never fails the analysis."""
    trace_dir = run_dir / "trace"
    trace_dir.mkdir(mode=0o700)
    detail = "trace: profiling a short slice on a separate launch (diagnostic, not a speed claim)"
    emit(PhaseStarted(phase="trace", detail=detail))
    extender = load_extender()
    sections: list[str] = []
    trace = None
    try:
        asyncio.run(_traced_slice(project, engine, runtime, settings, trace_dir, ready_timeout_s))
        tracing = profiler(engine)
        files = tracing.collect(trace_dir)
        trace = read_trace(files, tracing.step_scope)
        summary = summarize_trace(
            trace,
            len(files),
            expected_quant_kernels=[
                engine.quant_kernel_symbols[kernel.name]
                for kernel in measurement.quant_kernels
                if kernel.name in engine.quant_kernel_symbols
            ],
        )
        if extender and (
            extension := extend(
                extender.analyse, trace, replace(measurement, trace=summary), settings
            )
        ):
            summary = replace(summary, analysis=extension.result)
            sections = list(extension.sections)
    except TraceUnavailable as error:
        summary = TraceSummary("unavailable", str(error))
    except (AnalyseFailure, httpx.HTTPError, TimeoutError, OSError, ValueError) as error:
        summary = TraceSummary("unavailable", f"{type(error).__name__}: {error}")
    return TraceResult(summary, sections, extender is not None, trace)


async def _traced_slice(
    project: ResolvedProject,
    engine: Engine,
    runtime: Runtime,
    settings: Settings,
    trace_dir: Path,
    ready_timeout_s: float,
) -> None:
    """Launch with the profiler on, run TRACE_REQUESTS requests at the workload's concurrency
    under a hard time limit, and stop the profiler so the trace files are written."""
    run_id = trace_dir.parent.name
    spec = spec_for(project, engine, settings, run_id + "-trace", trace_dir)
    tracing = profiler(engine)
    async with serving(
        engine, runtime, spec, ready_timeout_s=ready_timeout_s, record_dirs=[trace_dir],
        phase="trace", images=ImageSource(project.images),
    ) as live:  # fmt: skip
        await warm_up(project, spec, live, run_id)
        url = live.server.endpoint_url
        await tracing.start(live.client, url)
        try:
            await timed_requests(
                project, spec, live, run_id, TRACE_REQUESTS, TRACE_TIME_LIMIT_S, "trace"
            )
        finally:
            await tracing.stop(live.client, url)


# --------------------------------------------------------------------------------------
# Kernel counters (Level 2)
# --------------------------------------------------------------------------------------


def add_counters(
    project: ResolvedProject,
    engine: Engine,
    runtime: Runtime,
    settings: Settings,
    run_dir: Path,
    measurement: Measurement,
    trace: Trace | None,
    ready_timeout_s: float,
) -> CountersResult:
    """Profile the trace's top kernels with ncu; failing to never fails the analysis."""
    counters_dir = run_dir / "counters"
    counters_dir.mkdir(mode=0o700)
    detail = "counters: profiling the top kernels with Nsight Compute, one launch per counter group"
    emit(PhaseStarted(phase="counters", detail=detail))
    try:
        summary = _measure_counters(
            project, engine, runtime, settings, counters_dir, measurement, trace, ready_timeout_s
        )
    except (
        AnalyseFailure,
        RuntimeFailure,
        httpx.HTTPError,
        TimeoutError,
        OSError,
        ValueError,
    ) as error:
        summary = CountersSummary("unavailable", f"{type(error).__name__}: {error}")
    extender = load_extender()
    sections: list[str] = []
    if (
        extender
        and extender.counters
        and summary.status == "ok"
        and (extension := extend(extender.counters, summary, measurement))
    ):
        summary = replace(summary, analysis=extension.result)
        sections = list(extension.sections)
    return CountersResult(summary, sections, extender is not None)


def _measure_counters(
    project: ResolvedProject,
    engine: Engine,
    runtime: Runtime,
    settings: Settings,
    counters_dir: Path,
    measurement: Measurement,
    trace: Trace | None,
    ready_timeout_s: float,
) -> CountersSummary:
    if trace is None or measurement.trace is None or measurement.trace.status != "ok":
        return CountersSummary("unavailable", "there is no trusted trace to pick kernels from")
    if not runtime.wraps_launch:
        return CountersSummary(
            "unavailable", "counters need ncu inside the image; unavailable in the docker runtime"
        )
    ncu = shutil.which("ncu")
    if ncu is None:
        return CountersSummary("unavailable", "ncu not found")
    qualification = qualify(ncu, counters_dir)
    write_json(counters_dir / "qualification.json", asdict(qualification))
    if not qualification.passed:
        return CountersSummary(
            "unavailable", f"profiler qualification failed: {qualification.reason}", qualification
        )
    selected = select_kernels(trace)

    def run(name: str, prefix: list[str], log_file: Path) -> tuple[list[Launch], str | None]:
        """One engine launch under ncu; whatever ncu wrote before a failure still counts."""
        spec = replace(
            spec_for(project, engine, settings, f"{counters_dir.parent.name}-{name}", None),
            launch_prefix=tuple(prefix),
        )
        note = None
        try:
            asyncio.run(_counted_slice(project, engine, runtime, spec, log_file, ready_timeout_s))
        except (AnalyseFailure, httpx.HTTPError, TimeoutError, OSError, ValueError) as error:
            note = f"{type(error).__name__}: {error}"
        text = log_file.read_text("utf-8", errors="replace") if log_file.exists() else ""
        return parse_raw_csv(text), note

    # One gated launch per qualified group (each fits one pass), filtered by the kernels' bare
    # function names; the trace and ncu spell full names differently, so every run's rows are
    # matched to the trace's kernels by name afterwards. The occupancy group runs first.
    groups = sorted(qualification.available, key=lambda group: group != REQUIRED_GROUP)
    runs: dict[str, tuple[list[Launch], str | None]] = {}
    returned, used = True, qualification.memory_cold_mib
    for group in groups:
        if not returned:
            break  # the profiler left memory behind: stop before it disturbs anything else
        log_file = counters_dir / f"{group}.csv"
        runs[group] = run(group, group_command(ncu, selected, log_file, group), log_file)
        returned, used = memory_returned(qualification.memory_cold_mib)
    kernels: list[KernelCounters] = []
    for kernel in selected:
        seen, _ = runs.get(REQUIRED_GROUP, ([], None))
        ncu_name, why = match_ncu_name(kernel.name, [launch.name for launch in seen])
        parts = {}
        for group, (launches, note) in runs.items():
            name, missing = match_ncu_name(kernel.name, [launch.name for launch in launches])
            rows = [launch for launch in launches if launch.name == name]
            parts[group] = kernel_counters(kernel, name or ncu_name, rows, missing or note)
        kernels.append(merge_groups(kernel, ncu_name, parts, qualification.unavailable))
    note = None if returned else "GPU memory did not return after profiling; later kernels skipped"
    return CountersSummary("ok", note, qualification, used, tuple(kernels))


async def _counted_slice(
    project: ResolvedProject,
    engine: Engine,
    runtime: Runtime,
    spec: ServeSpec,
    log_file: Path,
    ready_timeout_s: float,
) -> None:
    """Run the profiled slice between the engine's start and stop profile requests, so ncu
    (``--profile-from-start off``) counts only workload kernels. ncu ends the server itself once
    its launch count is reached, possibly mid-slice: that is expected, not a failure. Stopping
    the server afterwards lets ncu write its report even when the count was not reached."""
    tracing = profiler(engine)
    async with serving(
        engine, runtime, spec, ready_timeout_s=ready_timeout_s, record_dirs=[log_file.parent],
        phase=f"counters ({log_file.stem})", images=ImageSource(project.images), may_exit=True,
    ) as live:  # fmt: skip
        run_id = log_file.parent.parent.name
        await warm_up(project, spec, live, run_id)
        url = live.server.endpoint_url
        await tracing.start(live.client, url)
        try:
            await timed_requests(
                project, spec, live, run_id, COUNTER_REQUESTS, COUNTER_TIME_LIMIT_S, "counters"
            )
            await tracing.stop(live.client, url)
        except (httpx.HTTPError, TimeoutError):
            if live.server.is_running():
                raise
