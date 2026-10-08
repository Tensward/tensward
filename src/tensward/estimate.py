"""How much a suggested change is expected to raise this run's throughput, as a range of new over
old requests per second, for the playbook entries whose estimate held on runs it was not built
from. Internal: not part of the extension API.

In a saturated closed loop, throughput is the running batch over the time one request spends
running (Little's law), so a change's effect is its batch ratio times its service-time ratio."""

from __future__ import annotations

import hashlib
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Mapping, Protocol, Sequence

from .classify import NAMES, Bottleneck
from .playbook import Situation
from .playbook_common import _concurrency_cap, _demand
from .settings import Settings

if TYPE_CHECKING:
    from .engines.protocol import Engine

MIN_WINDOW_S = 30.0  # a shorter window is within the archive's run-to-run noise
MAX_FAILED_SHARE = 0.02  # failed requests leave the loop early, so Little's law misreads them
ETA_LOW, ETA_HIGH = 0.20, 0.75  # decode efficiency bounds, set on the training runs only
KV_FIT = 0.85  # share of the KV pool a window of consecutive requests may fill
KV_RELIEF = 1.05  # a batch this much larger counts as KV-cache relief
PROMOTE_AT = 1.5  # an estimate whose low end reaches this leads Try first
UNCHANGED_BELOW = 1.05  # a ratio this close to 1 reads "unchanged"
FREE_TOOL_CHOICES = (None, "auto", "none")  # any other tool_choice makes the engine call a tool
# The Ceilings.gpu strings each estimate was checked on: a platform row's label, else the
# device name (A10G has no platform row).
CHECKED_GPUS: Mapping[str, frozenset[str]] = {
    "raise-concurrency": frozenset({"T4", "L4", "NVIDIA A10G"}),
    "prefix-caching": frozenset({"NVIDIA A10G"}),
}
HASH_DIGITS = 16


@dataclass(frozen=True, slots=True, kw_only=True)
class Input:
    key: str  # stable id, e.g. "uncached_share"
    value: float
    text: str  # how the report says it


@dataclass(frozen=True, slots=True, kw_only=True)
class Estimate:
    low: float  # throughput ratio new / old, the conservative end
    high: float  # low <= high
    inputs: tuple[Input, ...] = ()
    relieves: frozenset[Bottleneck] = frozenset()  # bottlenecks the promotion may group it under
    note: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class RunInputs:
    """What an estimate needs from the run beyond its situation."""

    arrival: Literal["closed_loop", "open_loop", "capped"]
    gpu_count: int  # GPUs the engine was given
    forced_tools: bool = False  # some prompt sets tool_choice "required" or names a tool
    prompt_order: tuple[int, ...] = ()  # each request's prompt index, in dispatch order
    # By prompt index: the token count and the chained block hashes of the prompt as the engine
    # rendered it, chat template applied.
    prompt_blocks: Mapping[int, tuple[int, tuple[str, ...]]] = field(
        default_factory=dict, hash=False
    )


class Estimator(Protocol):
    def __call__(
        self, situation: Situation, *, proposed: Settings, run: RunInputs | None, engine: Engine
    ) -> Estimate | None: ...


def block_hashes(ids: Sequence[int], block_tokens: int) -> tuple[str, ...]:
    """One hash per full block of ``ids``, each chained to the one before, so two equal hashes
    mean an equal prefix up to the end of that block. A partial last block has none."""
    hashes: list[str] = []
    previous = ""
    for start in range(0, len(ids) - block_tokens + 1, block_tokens):
        block = ",".join(str(token) for token in ids[start : start + block_tokens])
        digest = hashlib.sha256(f"{previous}|{block}".encode()).hexdigest()
        previous = digest[:HASH_DIGITS]
        hashes.append(previous)
    return tuple(hashes)


def _passes_gate(
    entry: str, situation: Situation, proposed: Settings, run: RunInputs | None, engine: Engine
) -> bool:
    """Whether the run is one the estimate was checked on: a closed loop on one GPU with few
    failures, a steady window long enough, no speculation, no grammar and a checked GPU."""
    m, settings = situation.measurement, situation.settings
    if run is None or run.arrival != "closed_loop" or run.gpu_count != 1 or run.forced_tools:
        return False
    if m.replicas != 1 or engine.parallel_degree(settings) != 1:
        return False
    if engine.parallel_degree(proposed) != 1:
        return False
    done = m.succeeded + m.failed
    if not done or m.failed > MAX_FAILED_SHARE * done:
        return False
    if m.window is not None and m.window.kind != "steady":
        return False
    seconds = m.window.seconds if m.window is not None else m.seconds
    if seconds is None or seconds < MIN_WINDOW_S:
        return False
    if m.spec_coverage is not None or engine.lever_value(settings, "speculation") is not None:
        return False
    if m.hybrid_cache or situation.facts.structured:
        return False
    return m.ceilings is not None and m.ceilings.gpu in CHECKED_GPUS[entry]


def _batch_text(new: float, old: float, why: str) -> Input:
    return Input(
        key="batch",
        value=new,
        text=f"about {new:.0f} requests would run at once instead of {old:.0f}, {why}",
    )


def raise_concurrency_estimate(
    situation: Situation,
    *,
    proposed: Settings,
    run: RunInputs | None,
    engine: Engine,
    batch: float | None = None,
) -> Estimate | None:
    """A higher cap runs more requests at once. Each request's service time S = b / X has a
    fixed part, the weight reads of its decode steps, and a part that grows at most in
    proportion to the batch. ``batch`` replaces the predicted batch (the backtest's measured
    one)."""
    if not _passes_gate("raise-concurrency", situation, proposed, run, engine):
        return None
    m, ceilings = situation.measurement, situation.measurement.ceilings
    raised = proposed.max_concurrent_requests
    x, o, b, peak = m.request_throughput, m.output_throughput, m.mean_running, m.peak_running
    if ceilings is None or raised is None or not (x and o and b and peak):
        return None
    weights, bandwidth = ceilings.weight_bytes_per_step, ceilings.bandwidth_gbs
    if not (weights and bandwidth):
        return None
    cap = _concurrency_cap(m, situation.settings) or raised
    demand = _demand(m, cap)
    new = min(demand, raised) * b / peak if batch is None else batch
    if new <= b:
        return None
    service = b / x
    fixed = min(service, o / x * weights / (bandwidth * 1e9))
    grown = fixed + (service - fixed) * new / b
    why = "as the new cap allows" if raised <= demand else f"as the clients keep {demand} in flight"
    return Estimate(
        low=new / b * service / grown,
        high=new / b,
        inputs=(_batch_text(new, b, why),),
        relieves=frozenset({"queueing"}),
    )


def _uncached(
    order: Sequence[int],
    prompts: Mapping[int, tuple[int, tuple[str, ...]]],
    *,
    capacity: int,
    lag: int,
    block_tokens: int,
    min_blocks: int,
) -> float | None:
    """The share of prompt tokens the prefix cache would miss, replaying ``order`` through an
    LRU cache of ``capacity`` blocks. A request's blocks become usable ``lag`` dispatches after
    it (still in flight before that), tail first so that tails are evicted before heads. A
    partial last block is always missed. Only the requests after the first ``lag`` count; None
    when none do."""
    cache: OrderedDict[str, None] = OrderedDict()
    pending: deque[int] = deque()
    missed = total = 0
    for position, index in enumerate(order):
        while pending and len(pending) >= max(lag, 1):
            for key in reversed(prompts[pending.popleft()][1]):
                cache[key] = None
                cache.move_to_end(key)
            while len(cache) > capacity:
                cache.popitem(last=False)
        tokens, blocks = prompts[index]
        hits = 0
        for key in blocks:
            if key not in cache:
                break
            hits += 1
        hits = hits if hits >= min_blocks else 0
        for key in blocks[:hits]:
            cache.move_to_end(key)
        if position >= lag:
            total += tokens
            missed += tokens - hits * block_tokens
        pending.append(index)
    return missed / total if total else None


def _held(
    order: Sequence[int],
    prompts: Mapping[int, tuple[int, tuple[str, ...]]],
    n: int,
    block_tokens: int,
) -> int:
    """The most prompt tokens any ``n`` consecutive requests hold at once (all of them when
    there are fewer): their distinct blocks once, plus their partial last blocks."""
    n = max(1, min(n, len(order)))
    held: Counter[str] = Counter()
    tails = most = 0
    for position, index in enumerate(order):
        tokens, blocks = prompts[index]
        held.update(blocks)
        tails += tokens - len(blocks) * block_tokens
        if position >= n:
            gone_tokens, gone = prompts[order[position - n]]
            held.subtract(gone)
            tails -= gone_tokens - len(gone) * block_tokens
            for key in gone:
                if held[key] <= 0:
                    del held[key]
        if position >= n - 1:
            most = max(most, len(held) * block_tokens + tails)
    return most


def _kv_batch(
    order: Sequence[int],
    prompts: Mapping[int, tuple[int, tuple[str, ...]]],
    limit: int,
    room: float,
    block_tokens: int,
    out: float,
) -> int:
    """The largest batch up to ``limit`` and the run's request count whose every window fits
    ``room`` with half of each request's output; fitting is monotone in the batch, so it is
    searched by halves."""
    low, high = 1, max(min(limit, len(order)), 1)
    while low < high:
        middle = (low + high + 1) // 2
        if _held(order, prompts, middle, block_tokens) + middle * out / 2 <= room:
            low = middle
        else:
            high = middle - 1
    return low


def step_ratio(
    *, tpot: float, floor: float, growth: float, batch: float, uncached: float, eta: float
) -> float:
    """Throughput new / old from the step time. The baseline step ``tpot`` splits into decode,
    the decode floor ``floor`` over efficiency ``eta`` (at most the whole step), and prefill.
    Decode grows by ``growth`` (floor at the new batch over floor at the old), never clipped;
    prefill keeps its ``uncached`` share and grows with the ``batch`` ratio."""
    decode = min(tpot, floor / eta)
    prefill = tpot - decode
    return batch * tpot / (decode * growth + uncached * prefill * batch)


def prefix_caching_estimate(
    situation: Situation,
    *,
    proposed: Settings,
    run: RunInputs | None,
    engine: Engine,
    batch: float | None = None,
) -> Estimate | None:
    """Caching skips the prefill of prompt blocks already held, and holds a shared prefix once,
    which can make room for more running requests. The cache is replayed over the run's own
    send order with the prompts as the engine rendered them. ``batch`` replaces the predicted
    batch (the backtest's measured one)."""
    if not _passes_gate("prefix-caching", situation, proposed, run, engine) or run is None:
        return None
    m, settings = situation.measurement, situation.settings
    ceilings = m.ceilings
    capabilities = engine.capabilities(None)
    if settings.prefix_caching is not False or not capabilities.shared_prefix_kv:
        return None
    if ceilings is None or ceilings.moe_experts is not None or situation.facts.sends_images:
        return None
    order, prompts = run.prompt_order, run.prompt_blocks
    if not order or any(index not in prompts for index in order):
        return None
    x, o, b, peak, tpot = (
        m.request_throughput, m.output_throughput, m.mean_running, m.peak_running, m.tpot_p50_ms
    )  # fmt: skip
    weights, per_sequence, bandwidth = (
        ceilings.weight_bytes_per_step, ceilings.kv_bytes_per_sequence, ceilings.bandwidth_gbs
    )  # fmt: skip
    if not (x and o and b and peak and tpot and weights and per_sequence and bandwidth):
        return None
    if not (m.kv_capacity_tokens and m.kv_block_tokens):
        return None
    pool, block_tokens = m.kv_capacity_tokens, int(m.kv_block_tokens)
    out, fill = o / x, b / peak
    cap = _concurrency_cap(m, settings)
    if cap is None:
        return None
    demand = _demand(m, cap)
    widest = max(b, min(demand, cap) * fill)
    capacity = int((pool - widest * out) // block_tokens)
    if capacity <= 0:
        return None
    fitting = _kv_batch(order, prompts, max(demand, cap), KV_FIT * pool, block_tokens, out)
    limit = min(demand, cap, fitting)
    new = max(b, limit * fill) if batch is None else batch
    lag = round(new)
    min_blocks = -(-capabilities.min_cacheable_prefix_tokens // block_tokens)
    replay = dict(block_tokens=block_tokens, min_blocks=min_blocks)
    # The low end also leaves out the prompt blocks of the requests still in flight.
    running = -(-_held(order, prompts, lag, block_tokens) // block_tokens)
    low_capacity = max(capacity - running, 0)
    first = tuple(dict.fromkeys(order))
    found = [
        u
        for u in (
            _uncached(first, prompts, capacity=low_capacity, lag=lag, **replay),
            _uncached(order, prompts, capacity=low_capacity, lag=lag, **replay),
        )
        if u is not None
    ]
    best = _uncached(order, prompts, capacity=capacity, lag=0, **replay)
    if not found or best is None:
        return None
    worst = max(found)

    def floor(n: float) -> float:
        return (weights + n * per_sequence) / (bandwidth * 1e9) * 1e3

    growth, ratio = floor(new) / floor(b), new / b

    def ends(u: float) -> list[float]:
        """The ratio at both eta bounds: it moves one way in eta, falling as eta falls only
        while decode grows faster than prefill (``growth`` above ``u * ratio``)."""
        return [
            step_ratio(tpot=tpot, floor=floor(b), growth=growth, batch=ratio, uncached=u, eta=eta)
            for eta in (ETA_LOW, ETA_HIGH)
        ]

    low, high = sorted((min(ends(worst)), max(ends(best))))
    inputs = [
        Input(
            key="uncached_share",
            value=worst,
            text=f"{1 - worst:.0%} of prompt tokens would be served from the cache",
        )
    ]
    relieves: set[Bottleneck] = {"prefill"}
    if new > b * KV_RELIEF:
        relieves.add("kv_capacity")
        if limit == cap:
            why = f"as the KV cache would have room for the cap of {cap}"
        elif limit == demand:
            why = f"as the KV cache would have room for all {demand} requests in flight"
        else:
            why = "as many as the KV cache would have room for"
        inputs.append(_batch_text(new, b, why))
    note = ""
    if best >= 0.9:
        note = (
            "as the engine renders these prompts, they differ before most of the text they "
            "share, so the cache can reuse little of it"
        )
    return Estimate(
        low=low,
        high=high,
        inputs=tuple(inputs),
        relieves=frozenset(relieves),
        note=note,
    )


ESTIMATORS: Mapping[str, Estimator] = {
    "raise-concurrency": raise_concurrency_estimate,
    "prefix-caching": prefix_caching_estimate,
}


def lead_bottleneck(estimate: Estimate) -> Bottleneck:
    """The bottleneck a promoted suggestion is grouped under: the first it relieves, in the
    diagnosis's order of classes."""
    return next(name for name in NAMES if name in estimate.relieves)


def _ratio_text(ratio: float) -> str:
    return f"{(ratio - 1) * 100:+.0f}%" if ratio <= 2 else f"x{ratio:.1f}"


def range_text(low: float, high: float) -> str:
    """A range of ratios in words, checked in this order: no change, a possible loss, a range
    from no change, a range within a factor of 3, else its low end."""
    if 1 <= low and high < UNCHANGED_BELOW:
        return "unchanged"
    if low < 1:
        return f"{_ratio_text(low)} to {_ratio_text(high)}"
    if low < UNCHANGED_BELOW:
        return f"unchanged to {_ratio_text(high)}"
    if high <= 3 * low:
        return f"{_ratio_text(low)} to {_ratio_text(high)}"
    return f"at least {_ratio_text(low)}"


def expected_text(estimate: Estimate) -> str:
    """The report's ``expected:`` line, after the label."""
    said = [given.text for given in estimate.inputs] + ([estimate.note] if estimate.note else [])
    detail = f" (this run: {'; '.join(said)})" if said else ""
    return f"throughput {range_text(estimate.low, estimate.high)}{detail}"
