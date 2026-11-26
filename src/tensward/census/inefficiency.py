"""What the settings can't fix: the share of the census window the GPU sat idle, as launch
gaps and as waiting on the host. The idle share is split once, copies first, then launch gaps
capped by what is left, then host work as the rest; waiting on the host is copies plus host
work, so the two classes add up to the idle share."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from .policy import _in, _version
from .readings import H100_SXM, ReadingContext
from .snapshot import CensusWindow

Kind = Literal["launch_gaps", "host_wait"]
KINDS: tuple[Kind, ...] = ("launch_gaps", "host_wait")
Label = Literal["pageable_copies", "host_gpu_copies", "host_work"]
LABELS: tuple[Label, ...] = ("pageable_copies", "host_gpu_copies", "host_work")
BUCKET_NS = 100_000_000
SHORT_GAPS = 2
MIN_KERNELS = 1000
FLOORS: Mapping[Kind, float] = {"launch_gaps": 0.10, "host_wait": 0.30}
CONFIRMED: frozenset[Kind] = frozenset()
LABEL_CONFIRMED = False
COPY_PART_LABEL = 0.10
PAGEABLE_LABEL = 0.5
VERSIONS: Mapping[str, tuple[str, ...]] = {
    "llamacpp": (">=12246",),
    "vllm": ("==0.30.*", "==0.33.*"),
}
HOST_WAIT_VERSIONS: Mapping[str, tuple[str, ...]] = {"llamacpp": (">=12246",)}
GPUS = frozenset({"NVIDIA L4", H100_SXM})
H2D = 1
HOST_GPU = (1, 2)


@dataclass(frozen=True, slots=True)
class Inefficiency:
    """``share`` of the window is the most a fix of this kind could give back; the report
    shows it at or above ``floor``. ``label`` names the kind of work behind a host wait."""

    kind: Kind
    share: float
    floor: float
    detail: Mapping[str, float] = field(default_factory=dict)
    label: Label | None = None


def _checked(ctx: ReadingContext, versions: Mapping[str, tuple[str, ...]]) -> bool:
    return ctx.gpu_model in GPUS and any(
        _in(_version(ctx.version), spec) for spec in versions.get(ctx.engine, ())
    )


def validated(w: CensusWindow, ctx: ReadingContext) -> bool:
    """One recording process, on an engine version and GPU the classes were checked on."""
    return len(w.pids) == 1 and _checked(ctx, VERSIONS)


def _pageable_h2d(w: CensusWindow) -> float | None:
    h2d = {k: c.bytes for k, c in w.copies_detail.items() if k.split(":")[0] == str(H2D)}
    total = sum(h2d.values())
    if not total:
        return None
    return sum(v for k, v in h2d.items() if k.split(":")[1] == "1") / total


def _launch_detail(w: CensusWindow) -> dict[str, float]:
    launched = sum(k.n for k in w.kernels)
    if not launched:
        return {}
    return {
        "kernel_us": sum(k.ns for k in w.kernels) / launched / 1e3,
        "graph_share": sum(k.graph_n for k in w.kernels) / launched,
    }


def _label(copy_part: float, pageable: float | None) -> Label | None:
    if pageable is None:
        return None
    if copy_part < COPY_PART_LABEL:
        return "host_work"
    return "pageable_copies" if pageable >= PAGEABLE_LABEL else "host_gpu_copies"


def _host_wait(w: CensusWindow, copy_part: float, host_part: float) -> Inefficiency:
    detail = {"copy_part": copy_part, "host_part": host_part}
    if w.scoped and w.requests > 0:
        moved = sum(c.bytes for k, c in w.copies.items() if k in HOST_GPU)
        detail["gb_per_request"] = moved / w.requests / 1e9
    pageable = _pageable_h2d(w) if w.scoped else None
    if pageable is not None:
        detail |= {"pageable_h2d": pageable, "pinned_h2d": 1 - pageable}
    share = copy_part + host_part
    return Inefficiency(
        "host_wait", share, FLOORS["host_wait"], detail, _label(copy_part, pageable)
    )


def idle_split(w: CensusWindow) -> tuple[int, float]:
    """The kernel time of the window's timeline (ns) and the share of the window copies ran
    with no kernel running, at most the idle share; the copy part of the host wait."""
    length = w.end_ns - w.start_ns
    busy = copy_idle = 0
    for bucket in w.timeline:
        if "busy_ns" not in bucket:
            continue
        busy += bucket["busy_ns"]
        copy_idle += max(0, min(bucket.get("copy_ns", 0), BUCKET_NS - bucket["busy_ns"]))
    if length <= 0:
        return busy, 0.0
    return busy, min(copy_idle / length, 1 - busy / length)


def copy_wait_valid(w: CensusWindow, ctx: ReadingContext) -> bool:
    """Where the copy part reads as measured: one recording process, an engine version the
    classes were checked on, CUDA graphs off, no failed or dropped request and enough kernels.
    The GPU model is not part of it: the part is read from the timeline alone."""
    return (
        len(w.pids) == 1
        and any(_in(_version(ctx.version), spec) for spec in VERSIONS.get(ctx.engine, ()))
        and ctx.graphs_off is True
        and not w.failed
        and w.dropped == 0
        and w.n_kernels >= MIN_KERNELS
        and w.end_ns > w.start_ns
        and bool(w.timeline)
    )


def classes(w: CensusWindow, ctx: ReadingContext) -> tuple[Inefficiency, ...] | None:
    """Launch gaps, and the host wait where it was checked and the window can say it (no
    failed request, some kernel time); None outside the validated situation or with too few
    kernels."""
    length = w.end_ns - w.start_ns
    if not validated(w, ctx) or length <= 0 or not w.timeline or len(w.gaps) < SHORT_GAPS:
        return None
    if w.n_kernels < MIN_KERNELS:
        return None
    busy, copies = idle_split(w)
    idle = 1 - busy / length
    short = sum(g.ns for g in w.gaps[:SHORT_GAPS]) / length
    launch = min(short, idle - copies)
    host = max(0.0, idle - copies - launch)
    found = [Inefficiency("launch_gaps", launch, FLOORS["launch_gaps"], _launch_detail(w))]
    valid = not w.failed and busy > 0
    if valid and _checked(ctx, HOST_WAIT_VERSIONS):
        found.append(_host_wait(w, copies, host))
    return tuple(found)


def inefficiencies(w: CensusWindow, ctx: ReadingContext) -> list[Inefficiency] | None:
    """The confirmed classes of the window, or None when the window cannot say."""
    found = classes(w, ctx)
    if found is None:
        return None
    return [c for c in found if c.kind in CONFIRMED]
