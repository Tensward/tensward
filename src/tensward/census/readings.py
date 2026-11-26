"""The census readings: what the window of a census start says about the GPU, each one only
in the situation it was validated in; anywhere else it stays unknown."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ..fact_values import FactSpec
from .snapshot import CensusWindow, KernelRow

H100_SXM = "NVIDIA H100 80GB HBM3"
H100_SMS = 132
COPY_KINDS = {"h2d": 1, "d2h": 2, "d2d": 8}
H100 = {
    "sms": H100_SMS,
    "warps": 64,
    "blocks": 32,
    "regs": 65536,
    "reg_unit": 256,
    "shm": 233472,
    "shm_res": 1024,
}


@dataclass(frozen=True, slots=True)
class ReadingContext:
    engine: str
    gpu_model: str | None
    sm_count: int | None
    graphs_off: bool | None
    server_log: str | None = None
    version: str | None = None
    n_vocab: int | None = None  # the model's vocabulary, the width of one row of logits
    host_weight_bytes: int | None = None  # weights the placement keeps in host memory


@dataclass(frozen=True, slots=True)
class Reading:
    """``compute`` gives the value, or None when the window cannot say; ``validated`` says
    whether the situation is one the reading was proven in; ``rule`` is that situation in words."""

    spec: FactSpec
    compute: Callable[[CensusWindow, ReadingContext], float | int | bool | None]
    rule: str
    validated: Callable[[CensusWindow, ReadingContext], bool]


def per_sm_blocks(k: Mapping[str, float | None], c: Mapping[str, int]) -> tuple[int, int]:
    """Resident blocks per SM and warps per block: the lowest of the warp, block, register and
    shared-memory bounds."""
    tpb = int(k.get("tpb_max") or k.get("tpb") or 1)
    wpb = (tpb + 31) // 32
    blocks = min(c["blocks"], c["warps"] // wpb)
    regs = k.get("regs")
    if regs:
        per_warp = math.ceil(regs * 32 / c["reg_unit"]) * c["reg_unit"]
        blocks = min(blocks, 4 * ((c["regs"] // 4) // per_warp) // wpb)
    shm = k.get("shm")
    if shm:
        blocks = min(blocks, c["shm"] // (int(shm) + c["shm_res"]))
    return max(1, blocks), wpb


def kernel_fill(k: Mapping[str, float | None], c: Mapping[str, int]) -> float:
    """The share of the card's warp slots a kernel's resident warps fill while it runs,
    averaged over its waves."""
    blocks_per_sm, wpb = per_sm_blocks(k, c)
    blocks = max(float(k["blocks_mean"] or 0), 1.0)
    per_wave = c["sms"] * blocks_per_sm
    waves = math.ceil(blocks / per_wave - 1e-9)
    return blocks * wpb / (waves * c["sms"] * c["warps"])


def fullness_union(kernels: Sequence[KernelRow], busy_share: float) -> float | None:
    """How full the GPU is while it is busy: the kernel-time-weighted mean fill times the busy
    share, an estimate of DCGM's SM occupancy."""
    total = sum(k.ns for k in kernels)
    if not total:
        return None
    filled = 0.0
    for k in kernels:
        fill = kernel_fill(
            {
                "blocks_mean": k.blocks / k.n if k.n else 0.0,
                "tpb": k.tpb,
                "tpb_max": k.tpb_max,
                "regs": k.regs,
                "shm": k.shm_static + k.shm_dynamic_max,
            },
            H100,
        )
        filled += k.ns * fill
    return filled / total * busy_share


def _one_process(w: CensusWindow, ctx: ReadingContext) -> bool:
    return len(w.pids) == 1


def _busy_share(w: CensusWindow, ctx: ReadingContext) -> float | None:
    length = w.end_ns - w.start_ns
    if length <= 0 or not w.timeline:
        return None
    return sum(b.get("busy_ns", 0) for b in w.timeline) / length


def _has_kind(w: CensusWindow, kind: str) -> bool:
    return bool(w.after) and all(kind in s.kinds for s in w.after.values())


def _counted(w: CensusWindow, ctx: ReadingContext) -> bool:
    return bool(w.requests > 0 and w.n_kernels > 0 and w.dropped == 0)


def _per_request(w: CensusWindow, value: float) -> float:
    return value / w.requests


def _copies(w: CensusWindow, ctx: ReadingContext) -> float:
    return _per_request(w, sum(w.copies[k].n for k in COPY_KINDS.values() if k in w.copies))


def _bytes(kind: int) -> Callable[[CensusWindow, ReadingContext], float]:
    return lambda w, ctx: _per_request(w, w.copies[kind].bytes if kind in w.copies else 0)


def _kernel_ms(w: CensusWindow, ctx: ReadingContext) -> float:
    return _per_request(w, sum(k.ns for k in w.kernels) / 1e6)


def _on_h100(w: CensusWindow, ctx: ReadingContext) -> bool:
    return (
        _one_process(w, ctx)
        and ctx.gpu_model == H100_SXM
        and ctx.sm_count in (None, H100_SMS)
        and _has_kind(w, "kernels")
        and w.dropped == 0
    )


def _fullness(engine: str) -> Callable[[CensusWindow, ReadingContext], float | None]:
    def compute(w: CensusWindow, ctx: ReadingContext) -> float | None:
        busy = _busy_share(w, ctx)
        return None if busy is None or ctx.engine != engine else fullness_union(w.kernels, busy)

    return compute


def _spec(
    name: str, type_: str, unit: str | None, text: str, engines: tuple[str, ...] = ()
) -> FactSpec:
    return FactSpec(
        name=name,
        type=type_,  # type: ignore[arg-type]
        unit=unit,
        values=None,
        source="measurement",
        description=text,
        engines=engines,
    )


def _counts(w: CensusWindow, ctx: ReadingContext) -> bool:
    return _counted(w, ctx) and _has_kind(w, "memcpy")


def _kernel_time(w: CensusWindow, ctx: ReadingContext) -> bool:
    return _one_process(w, ctx) and ctx.graphs_off is True and _counted(w, ctx)


def _logits_share(w: CensusWindow, ctx: ReadingContext) -> float | None:
    """Bytes the timeline saw copied from device to host in the window, per output token,
    against one row of float logits."""
    if not w.out_tokens or not ctx.n_vocab:
        return None
    d2h = sum(bucket.get("d2h", 0) for bucket in w.timeline)
    return d2h / w.out_tokens / (ctx.n_vocab * 4)


def _logits_valid(w: CensusWindow, ctx: ReadingContext) -> bool:
    return (
        _one_process(w, ctx)
        and _counts(w, ctx)
        and bool(w.timeline)
        and w.out_tokens > 0
        and bool(ctx.n_vocab)
    )


def _host_passes(w: CensusWindow, ctx: ReadingContext) -> float | None:
    """How many times the host weights crossed to the device for each request."""
    if not ctx.host_weight_bytes:
        return None
    h2d = w.copies[COPY_KINDS["h2d"]].bytes if COPY_KINDS["h2d"] in w.copies else 0
    return _per_request(w, h2d) / ctx.host_weight_bytes


def _pageable(w: CensusWindow, ctx: ReadingContext) -> float | None:
    from .inefficiency import _pageable_h2d

    return _pageable_h2d(w)


def _h2d_rate(w: CensusWindow, ctx: ReadingContext) -> float | None:
    """GB a second the host-to-GPU copies moved while they ran."""
    h2d = [c for k, c in w.copies_detail.items() if k.split(":")[0] == str(COPY_KINDS["h2d"])]
    moved, took = sum(c.bytes for c in h2d), sum(c.ns for c in h2d)
    return moved / took if moved and took else None


def _copy_wait(w: CensusWindow, ctx: ReadingContext) -> float | None:
    from .inefficiency import idle_split

    return idle_split(w)[1]


def _copy_wait_valid(w: CensusWindow, ctx: ReadingContext) -> bool:
    from .inefficiency import copy_wait_valid

    return copy_wait_valid(w, ctx)


_COPY_RULE = "the memcpy kind was recorded and no record was dropped"
READINGS: tuple[Reading, ...] = (
    Reading(
        _spec(
            "census.busy_share",
            "float",
            "share",
            "share of the window a kernel was running on the GPU, from the census timeline",
        ),
        _busy_share,
        "one recording process, the timeline covering the window",
        _one_process,
    ),
    Reading(
        _spec("census.kernels_per_request", "float", None, "kernels launched for each request"),
        lambda w, ctx: _per_request(w, w.n_kernels),
        "requests were counted and no record was dropped",
        _counted,
    ),
    Reading(
        _spec(
            "census.copies_per_request", "float", None, "host and device copies for each request"
        ),
        _copies,
        _COPY_RULE,
        _counts,
    ),
    *(
        Reading(
            _spec(
                f"census.{name}_bytes_per_request",
                "float",
                "bytes",
                f"bytes copied {text} for each request",
            ),
            _bytes(COPY_KINDS[name]),
            _COPY_RULE,
            _counts,
        )
        for name, text in (
            ("h2d", "from host to device"),
            ("d2h", "from device to host"),
            ("d2d", "from device to device"),
        )
    ),
    Reading(
        _spec(
            "census.kernel_ms_per_request",
            "float",
            "ms",
            "GPU time the kernels run for each request",
        ),
        _kernel_ms,
        "CUDA graphs off in the census start, one recording process",
        _kernel_time,
    ),
    Reading(
        _spec(
            "census.fullness",
            "float",
            "share",
            "how full the GPU's warp slots are over the window, an estimate of DCGM SM occupancy",
            ("llamacpp",),
        ),
        _fullness("llamacpp"),
        f"{H100_SXM}, one recording process",
        _on_h100,
    ),
    Reading(
        _spec(
            "census.fullness_at_most",
            "float",
            "share",
            "an upper bound of how full the GPU's warp slots are, where persistent kernels "
            "set their own grid",
            ("vllm",),
        ),
        _fullness("vllm"),
        f"{H100_SXM}, one recording process",
        _on_h100,
    ),
    Reading(
        _spec(
            "census.logits_on_host_share",
            "float",
            "share",
            "bytes copied from the GPU to the host per output token, against one row of "
            "float logits (1.0: every token's logits went to the host)",
            ("llamacpp",),
        ),
        _logits_share,
        "one recording process, the memcpy kind recorded, output tokens counted and the "
        "model's vocabulary known (C:RD2r)",
        _logits_valid,
    ),
    Reading(
        _spec(
            "census.host_passes_per_request",
            "float",
            None,
            "how many times the weights in host memory were copied to the GPU for each request",
            ("llamacpp",),
        ),
        _host_passes,
        "the memcpy kind recorded, no record dropped and weights in host memory (C:RD11)",
        lambda w, ctx: _counts(w, ctx) and bool(ctx.host_weight_bytes),
    ),
    Reading(
        _spec(
            "census.copy_wait_share",
            "float",
            "share",
            "share of the window copies between host and GPU ran while no kernel did",
            ("llamacpp",),
        ),
        _copy_wait,
        "one recording process, CUDA graphs off, a llama.cpp build the host-wait class was "
        "checked on, no failed request (C:RD12)",
        _copy_wait_valid,
    ),
    Reading(
        _spec(
            "census.pageable_h2d_share",
            "float",
            "share",
            "share of the host-to-GPU bytes copied from pageable (not pinned) host memory",
            ("llamacpp",),
        ),
        _pageable,
        "the memcpy kind and the source memory recorded, no record dropped",
        lambda w, ctx: _counts(w, ctx) and bool(w.copies_detail),
    ),
    Reading(
        _spec(
            "census.h2d_gb_per_s",
            "float",
            None,
            "GB a second the host-to-GPU copies moved while they ran",
            ("llamacpp",),
        ),
        _h2d_rate,
        "the memcpy kind and the source memory recorded, no record dropped",
        lambda w, ctx: _counts(w, ctx) and bool(w.copies_detail),
    ),
    Reading(
        _spec(
            "census.graphs_off",
            "bool",
            None,
            "the census start ran with CUDA graphs off",
        ),
        lambda w, ctx: ctx.graphs_off,
        "the start's own setting",
        lambda w, ctx: ctx.graphs_off is not None,
    ),
    Reading(
        _spec("census.recording_processes", "int", None, "processes that recorded kernels"),
        lambda w, ctx: len(w.pids),
        "any window",
        lambda w, ctx: bool(w.pids),
    ),
)


def compute_readings(w: CensusWindow, ctx: ReadingContext) -> dict[str, float | int | bool]:
    """Every reading whose situation holds and that the window can compute."""
    found: dict[str, float | int | bool] = {}
    for reading in READINGS:
        if not reading.validated(w, ctx):
            continue
        value = reading.compute(w, ctx)
        if value is not None:
            found[reading.spec.name] = value
    return found
