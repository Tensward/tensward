"""Profiler traces: what the GPU did and where it sat idle, as headline numbers.

Engine-neutral: it reads PyTorch-profiler Chrome traces (``.json`` or ``.json.gz``, complete
events ``ph: "X"``) whatever engine wrote them. GPU events have category ``kernel``,
``gpu_memcpy`` or ``gpu_memset``; CPU events ``cpu_op``, ``user_annotation``, ``python_function``,
``cuda_runtime`` or ``cuda_driver`` (category names from libkineto's ActivityType.h).

This module parses (:func:`read_trace`) and reports the headline facts: GPU busy share, idle
share, gap count and whether the trace can be trusted. Interpreting the gaps is left to an
optional analysis plugin (see :mod:`tensward.extensions`), which receives the parsed
:class:`Trace`.

Profiled timing is diagnostic only: the profiler slows the engine and its numbers are never a
speed claim.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Collection, Iterable, NamedTuple, Sequence

GAP_MIN_US = 50.0  # an idle interval this long counts as a gap
MIN_KERNELS = 50  # fewer kernels than this cannot be a real decode trace
IDLE_HINT_SHARE = 0.10  # idle time above this share of the window is worth a pointer to a fix
GPU_CATEGORIES = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
CPU_CATEGORIES = frozenset(
    {"cpu_op", "user_annotation", "python_function", "cuda_runtime", "cuda_driver"}
)
# Checkpoint quantization kernel (as the engine reports it) -> its name in CUDA kernel symbols.
QUANT_KERNEL_SYMBOLS = {"MarlinLinearKernel": "marlin"}
# The kernels a real model trace must contain. Attention is matched first: fused attention
# kernels are built on cutlass too.
ATTENTION = re.compile(r"flash|fmha|attention|attn|paged|reshape_and_cache|\bmla", re.IGNORECASE)
GEMM = re.compile(
    r"gemm|gemv|marlin|cutlass|cublas|nvjet|xmma|machete|wgmma|s16816|splitk", re.IGNORECASE
)


class TraceUnavailable(Exception):
    """The engine's trace cannot be analysed at all; the message says why."""


class Event(NamedTuple):
    start: float  # microseconds
    end: float
    name: str
    cat: str
    pid: int
    tid: int
    correlation: int | None


@dataclass(frozen=True, slots=True)
class Gap:
    """A stretch of at least GAP_MIN_US with no GPU activity, and the events around it."""

    start: float
    end: float
    prior: Event  # the event that ended last before the gap
    following: Event  # the event that starts after it

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Trace:
    """A parsed trace of one GPU: its events, idle gaps and busy time (all microseconds)."""

    gpu: tuple[Event, ...]
    cpu: tuple[Event, ...]
    gaps: tuple[Gap, ...]
    window: float
    busy_time: float
    step_scope: str  # name of the CPU annotation that wraps one model step

    @property
    def kernels(self) -> list[Event]:
        return [e for e in self.gpu if e.cat == "kernel"]


@dataclass(frozen=True, slots=True)
class TraceSummary:
    """The headline numbers. ``status``: ``ok``, ``untrusted`` (shown, but no recommendation
    may be drawn from it) or ``unavailable``. ``analysis`` is whatever the analysis plugin
    returned, kept with the metrics and read by the plugin's own recipes."""

    status: str
    note: str | None = None
    files: int = 0
    window_ms: float | None = None
    gpu_busy_share: float | None = None
    kernels: int = 0
    gap_count: int = 0
    idle_share: float | None = None
    analysis: Any = None


def load_events(paths: Iterable[Path]) -> list[Event]:
    """The GPU and CPU complete events of every trace file, times in microseconds."""
    events = []
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            document = json.load(handle)
        for raw in document.get("traceEvents", ()):
            cat = raw.get("cat")
            if raw.get("ph") != "X" or (cat not in GPU_CATEGORIES and cat not in CPU_CATEGORIES):
                continue
            start = float(raw["ts"])
            events.append(
                Event(
                    start,
                    start + float(raw.get("dur", 0)),
                    str(raw.get("name", "")),
                    cat,
                    raw.get("pid", 0),
                    raw.get("tid", 0),
                    (raw.get("args") or {}).get("correlation"),
                )
            )
    return events


def _busy_time(events: Iterable[Event]) -> float:
    """Total time at least one event was running (the union of the intervals)."""
    total = 0.0
    frontier = float("-inf")
    for event in sorted(events, key=lambda e: e.start):
        total += max(event.end - max(event.start, frontier), 0.0)
        frontier = max(frontier, event.end)
    return total


def _find_gaps(gpu: Sequence[Event]) -> list[Gap]:
    """Idle intervals of at least GAP_MIN_US between consecutive busy stretches."""
    ordered = sorted(gpu, key=lambda event: event.start)
    gaps: list[Gap] = []
    frontier, last = ordered[0].end, ordered[0]
    for event in ordered[1:]:
        if event.start - frontier >= GAP_MIN_US:
            gaps.append(Gap(frontier, event.start, last, event))
        if event.end > frontier:
            frontier, last = event.end, event
    return gaps


def read_trace(paths: Sequence[Path], step_scope: str) -> Trace:
    """Parse trace files into one GPU's events, gaps and busy time.

    Raises :class:`TraceUnavailable` when there is nothing to analyse.
    """
    if not paths:
        raise TraceUnavailable("the engine wrote no trace file")
    events = load_events(paths)
    gpu = [e for e in events if e.cat in GPU_CATEGORIES]
    if not gpu:
        raise TraceUnavailable("the trace has no GPU events")
    device = Counter(e.pid for e in gpu).most_common(1)[0][0]  # one GPU is modelled
    gpu = [e for e in gpu if e.pid == device]
    if not any(e.cat == "kernel" for e in gpu):
        raise TraceUnavailable("the trace has no GPU kernels")
    window = max(e.end for e in gpu) - min(e.start for e in gpu)
    return Trace(
        gpu=tuple(gpu),
        cpu=tuple(e for e in events if e.cat in CPU_CATEGORIES),
        gaps=tuple(_find_gaps(gpu)),
        window=window,
        busy_time=_busy_time(gpu),
        step_scope=step_scope,
    )


def summarize_trace(
    trace: Trace, files: int, expected_quant_kernels: Collection[str] = ()
) -> TraceSummary:
    """The headline numbers. ``expected_quant_kernels`` are kernel names the engine says it
    selected; a trace without them did not capture the target kernels."""
    kernels = trace.kernels
    window = trace.window
    summary = TraceSummary(
        "ok",
        files=files,
        window_ms=window / 1000,
        gpu_busy_share=trace.busy_time / window if window else None,
        kernels=len(kernels),
        gap_count=len(trace.gaps),
        idle_share=sum(g.length for g in trace.gaps) / window if window else None,
    )
    problem = _untrusted_reason(kernels, expected_quant_kernels)
    return summary if problem is None else replace(summary, status="untrusted", note=problem)


def _untrusted_reason(kernels: Sequence[Event], expected: Collection[str]) -> str | None:
    """Why this trace cannot be trusted to show the model's kernels; None when it can."""
    if len(kernels) < MIN_KERNELS:
        return f"only {len(kernels)} kernels were captured (need {MIN_KERNELS})"
    attention = any(ATTENTION.search(e.name) for e in kernels)
    gemm = any(GEMM.search(e.name) and not ATTENTION.search(e.name) for e in kernels)
    missing = [name for name, found in (("gemm", gemm), ("attention", attention)) if not found]
    if missing:
        return f"no {' or '.join(missing)} kernels were captured, so the profiler missed the model"
    names = " ".join(e.name.lower() for e in kernels)
    absent = [symbol for symbol in expected if symbol not in names]
    if absent:
        return f"the engine selected {', '.join(absent)} kernels but the trace has none"
    return None


def render_trace(summary: TraceSummary, *, analysis_available: bool) -> list[str]:
    """The headline report.md section."""
    lines = ["", "## GPU timeline (profiled - diagnostic only)", ""]
    if summary.status == "unavailable":
        return [*lines, f"- trace unavailable: {summary.note}"]
    lines += [
        "- profiled timing is not a speed claim: the profiler slows the engine, and this ran "
        "on a separate short launch after the clean measurement",
        f"- window: {summary.window_ms:,.1f} ms, GPU busy {summary.gpu_busy_share:.1%}, "
        f"{summary.kernels:,} kernels",
        f"- {summary.gap_count:,} idle gaps of at least {GAP_MIN_US:g} us with no GPU activity: "
        f"{summary.idle_share:.1%} of the window",
    ]  # fmt: skip
    if summary.status == "untrusted":
        lines.append(f"- NOT TRUSTED, no recommendation is drawn from it: {summary.note}")
    if not analysis_available and (summary.idle_share or 0) >= IDLE_HINT_SHARE:
        lines.append(
            "- Cause analysis of GPU idle time and the fixes for it are available with "
            "Tensward Optimize."
        )
    return lines
