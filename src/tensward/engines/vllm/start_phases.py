"""How a vLLM 0.30 server start went, from its log: the seconds each phase ended at, what kind of
start it was, and whether requests queued in the first minute of the load."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping

from .logs import LOADED

STAMP = re.compile(r"\b(\d\d)-(\d\d) (\d\d:\d\d:\d\d)\b")
CAPTURE_SECONDS = re.compile(r"Graph capturing finished in (\d+) secs")
LOAD_LINE = re.compile(
    r"\b(\d\d)-(\d\d) (\d\d:\d\d:\d\d)\b.*Running: (\d+) reqs, Waiting: (\d+) reqs, "
    r"GPU KV cache usage: ([\d.]+)%"
)
# The log lines that end a phase, by the phase's name; the first line of any of its markers.
PHASE_MARKERS = {
    "spawn": ("Initializing a V1 LLM engine",),
    "weights": ("Loading weights took", "Model loading took"),
    "mm_warmup": ("warmup completed in",),
    "compile": ("torch.compile took",),
    "ready": ("Starting vLLM server",),
}
KV_MARKERS = ("Available KV cache memory", "GPU KV cache size")
CAPTURED = "Graph capturing finished"
COLD_JIT_S = 30.0  # a Triton JIT before the second capture this long means empty caches
LOAD_WINDOW_S = 60.0
QUEUED_KV_PERCENT = 95.0
CLASSES = ("cache_hit", "new_set_bf16", "new_set_fp8", "cold")


def _when(month: str, day: str, clock: str) -> datetime:
    """A log stamp, which has no year; a leap one keeps 02-29 valid."""
    return datetime.strptime(f"2000-{month}-{day} {clock}", "%Y-%m-%d %H:%M:%S")


def _stamp(line: str) -> datetime | None:
    found = STAMP.search(line)
    return _when(found[1], found[2], found[3]) if found else None


def phases(log: str) -> dict[str, float] | None:
    """Seconds from launch (the first stamped ``APIServer`` line) to the end of spawn, weights,
    mm_warmup, compile, graphs_1 (the first CUDA graph capture), kv_alloc, gap (the cold Triton
    JIT before the second capture ends: where that capture began), graphs_2 and ready. A phase
    the log does not show is left out; None without a launch stamp."""
    stamped = [(at, line) for line in log.splitlines() if (at := _stamp(line)) is not None]
    launch = next((at for at, line in stamped if "APIServer" in line), None)
    if launch is None:
        return None

    def seconds(at: datetime) -> float:
        return ((at - launch) % timedelta(days=366)).total_seconds()

    def first(markers: tuple[str, ...]) -> datetime | None:
        return next((at for at, line in stamped if any(m in line for m in markers)), None)

    captures = [(at, line) for at, line in stamped if CAPTURED in line]
    found: dict[str, datetime | None] = {
        name: first(markers) for name, markers in PHASE_MARKERS.items()
    }
    found["graphs_1"] = captures[0][0] if captures else None
    found["kv_alloc"] = first(KV_MARKERS)
    found["graphs_2"] = captures[1][0] if len(captures) > 1 else None
    found["gap"] = None
    if len(captures) > 1:
        took = CAPTURE_SECONDS.search(captures[1][1])
        found["gap"] = captures[1][0] - timedelta(seconds=int(took[1])) if took else None
    order = ("spawn", "weights", "mm_warmup", "compile", "graphs_1", "kv_alloc", "gap",
             "graphs_2", "ready")  # fmt: skip
    return {name: seconds(at) for name in order if (at := found[name]) is not None}


def cache_state(log: str, recorded: Mapping[str, Any] | None = None) -> dict[str, bool]:
    """What the start's compile caches held: ``aot_hit`` when the log loaded the compiled graph
    and compiled nothing, ``empty_before`` when the run's ``caches`` record shows every cache
    empty at the launch."""
    return {
        "aot_hit": LOADED in log and "Compiling a graph" not in log,
        "empty_before": bool(recorded and all(e["before"] == 0 for e in recorded.values())),
    }


def start_class(
    phases: Mapping[str, float] | None, caches: Mapping[str, bool], quantization: str | None
) -> str:
    """``cache_hit`` (the compiled graph was loaded), ``cold`` (empty caches, or a Triton JIT
    of COLD_JIT_S or more before the second capture), else ``new_set_fp8`` or ``new_set_bf16``:
    a new compile with warm kernel caches."""
    if caches["aot_hit"]:
        return "cache_hit"
    jit = None
    if phases and "gap" in phases and "kv_alloc" in phases:
        jit = phases["gap"] - phases["kv_alloc"]
    if caches["empty_before"] or (jit is not None and jit >= COLD_JIT_S):
        return "cold"
    return "new_set_fp8" if quantization and quantization.startswith("fp8") else "new_set_bf16"


@dataclass(frozen=True, slots=True)
class LoadLine:
    offset_s: float  # seconds after the first stats line
    waiting: int
    kv_percent: float


@dataclass(frozen=True, slots=True)
class LoadQueue:
    seen: int  # stats lines in the first LOAD_WINDOW_S of the load
    queued: LoadLine | None  # the first of them with requests waiting at the KV limit
    closed: bool  # a later line shows the first LOAD_WINDOW_S are over


def load_queue(log: str) -> LoadQueue:
    """The engine's ``Running/Waiting/KV usage`` lines for the first LOAD_WINDOW_S after the
    first of them, and the first with requests waiting at QUEUED_KV_PERCENT or more KV usage."""
    lines = []
    for found in LOAD_LINE.finditer(log):
        lines.append((_when(*found.group(1, 2, 3)), int(found[4]), int(found[5]), float(found[6])))
    seen, queued, closed = 0, None, False
    for at, _, waiting, kv in lines:
        offset = ((at - lines[0][0]) % timedelta(days=366)).total_seconds()
        if offset > LOAD_WINDOW_S:
            closed = True
            break
        seen += 1
        if queued is None and waiting > 0 and kv >= QUEUED_KV_PERCENT:
            queued = LoadLine(offset, waiting, kv)
    return LoadQueue(seen, queued, closed)
