"""How much a suggested change is expected to raise this run's throughput, as a range of new over
old requests per second, for the playbook entries whose estimate held on runs it was not built
from. Internal: not part of the extension API.

Throughput is the decoding batch over the step time. A change raises the batch up to the first of
four limits (the clients' demand, the cap, the KV room and the per-step token budget) and the
line names the one that binds. The step model (``raise-concurrency``, ``raise-prefill-batch``)
splits the measured step into prompt work at the probed prompt capacity and decode, which grows
with the batch from one read of the weights at the low end and stays flat at the high end;
``prefix-caching`` replays the run's prompts through the cache and bounds decode by two decode
efficiencies. Every ratio is capped by the prompt ceiling: requests per second never exceed the
prompt capacity over the prompt tokens each request computes."""

from __future__ import annotations

import hashlib
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field, replace
from math import floor as round_down
from typing import TYPE_CHECKING, Callable, Iterable, Literal, Mapping, Protocol, Sequence

from .classify import NAMES, Bottleneck
from .playbook import Situation
from .pressure import PROMPT_SHARE, Limit, demand
from .settings import Settings

if TYPE_CHECKING:
    from .engines.protocol import Engine
    from .measurement import Measurement

MIN_WINDOW_S = 30.0  # a shorter window is within the archive's run-to-run noise
MAX_FAILED_SHARE = 0.02  # failed requests leave the loop early, so Little's law misreads them
ETA_LOW, ETA_HIGH = 0.20, 0.75  # decode efficiency bounds, set on the training runs only
KV_FIT = 0.85  # share of the KV pool a window of consecutive requests may fill
PROMOTE_AT = 1.5  # an estimate whose low end reaches this leads Try first
FREE_TOOL_CHOICES = (None, "auto", "none")  # any other tool_choice makes the engine call a tool
HASH_DIGITS = 16
SHOWN_FROM = 1.05  # no line for a low end below this
RANGE_WITHIN = 3.0  # a range is shown when its high end is within this factor of its low end
CHECKED_GPUS = frozenset({"NVIDIA A10G", "L4"})  # Ceilings.gpu of the GPUs estimates are checked on
BATCH_LIMITS: tuple[Limit, ...] = ("demand", "cap", "kv", "step_budget")
PREFILL_BATCH_COST = "larger prompt chunks per step can raise the time per output token"


@dataclass(frozen=True, slots=True, kw_only=True)
class Input:
    key: str  # stable id, e.g. "uncached_share"
    value: float
    text: str  # how the report says it


@dataclass(frozen=True, slots=True, kw_only=True)
class Estimate:
    low: float
    high: float
    inputs: tuple[Input, ...] = ()
    relieves: frozenset[Bottleneck] = frozenset()
    note: str = ""
    binding: Limit | None = None  # the limit that sets the new batch at the low end
    named: frozenset[Limit] = frozenset()  # the limits the line names


@dataclass(frozen=True, slots=True, kw_only=True)
class RunInputs:
    """What an estimate needs from the run beyond its situation."""

    arrival: Literal["closed_loop", "open_loop", "capped"]
    gpu_count: int  # GPUs the engine was given
    forced_tools: bool = False  # some prompt sets tool_choice "required" or names a tool
    prompt_order: tuple[int, ...] = ()  # each request's prompt index, in dispatch order
    engine_version: str | None = None
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


def outside_gate(
    situation: Situation, proposed: Settings, run: RunInputs | None, engine: Engine
) -> str | None:
    """Why the run is not one the estimate was checked on (a closed loop on one GPU with few
    failures, a steady window long enough, no speculation, no grammar, no forced tool call, no
    hybrid cache and a checked GPU); None when it is."""
    m, settings = situation.measurement, situation.settings
    if run is None or run.arrival != "closed_loop":
        return "not a closed loop"
    if run.gpu_count != 1 or m.replicas != 1:
        return "not one GPU"
    if engine.parallel_degree(settings) != 1 or engine.parallel_degree(proposed) != 1:
        return "a model spread over GPUs"
    if run.forced_tools:
        return "forced tool calls"
    done = m.succeeded + m.failed
    if not done or m.failed > MAX_FAILED_SHARE * done:
        return "more than 2% failed"
    seconds = m.window.seconds if m.window is not None else m.seconds
    window_kind = m.window.kind if m.window is not None else "steady"
    if window_kind != "steady" or seconds is None or seconds < MIN_WINDOW_S:
        return "window not steady or under 30 s"
    if m.spec_coverage is not None or engine.lever_value(settings, "speculation") is not None:
        return "speculative decoding"
    if situation.facts.structured:
        return "structured output"
    if m.hybrid_cache:
        return "a hybrid KV cache"
    if m.ceilings is None or m.ceilings.gpu not in CHECKED_GPUS:
        return "a GPU the estimates were not checked on"
    return None


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


@dataclass(frozen=True, slots=True, kw_only=True)
class Step:
    """The run's engine step (spec 3.1), in decoding sequences and seconds."""

    b: float  # sequences decoding per step
    t: float
    pl: float  # prompt tokens computed per generated token
    prompt: float  # seconds of each step on prompt work, at the run's prompt capacity
    decode: float
    weights: float  # seconds to read the weights once
    weight_bytes: float
    per_sequence: float
    bandwidth: float
    demand: int
    cap: int
    budget: int
    kv: float
    capacity: float  # R, prompt tokens per second (spec 2.1)
    output: float  # generated tokens per second

    def floor(self, batch: float) -> float:
        return (self.weight_bytes + batch * self.per_sequence) / self.bandwidth


def _step(
    situation: Situation, proposed: Settings, run: RunInputs | None, engine: Engine
) -> Step | None:
    if outside_gate(situation, proposed, run, engine) is not None:
        return None
    m, c = situation.measurement, situation.measurement.ceilings
    cap = situation.settings.max_concurrent_requests or _default_cap(m, run, engine)
    computed, scheduled, clients = m.prompt_tokens_per_step, m.scheduled_tokens_per_step, demand(m)
    rate, budget, kv, out = (
        m.prompt_capacity_tok_s, m.step_budget_tokens, m.mean_kv_usage, m.output_throughput
    )  # fmt: skip
    if c is None or computed is None or not scheduled or not m.step_ms or not rate or not budget:
        return None
    wb, ps, gbs = c.weight_bytes_per_step, c.kv_bytes_per_sequence, c.bandwidth_gbs
    if not out or not kv or not cap or not clients or not wb or not ps or not gbs:
        return None
    b, t, bandwidth = scheduled - computed, m.step_ms / 1e3, gbs * 1e9
    if b <= 0:
        return None
    weights = min(t, wb / bandwidth)
    decode = max(t - computed / rate, weights)
    return Step(
        b=b, t=t, pl=computed / b, prompt=t - decode, decode=decode, weights=weights,
        weight_bytes=wb, per_sequence=ps, bandwidth=bandwidth, demand=clients, cap=cap,
        budget=budget, kv=kv, capacity=rate, output=out,
    )  # fmt: skip


def _default_cap(m: Measurement, run: RunInputs | None, engine: Engine) -> int | None:
    """The engine's own cap when the setup sets none; unknown when the run ran more at once."""
    version = run.engine_version if run is not None else None
    gpu = m.ceilings.gpu if m.ceilings is not None else None
    cap = engine.default_concurrency(version, gpu, m.gpu_memory_gib)
    if cap is None or (m.peak_running is not None and m.peak_running > cap):
        return None
    return cap


def _limits(
    *, clients: float, cap: float, kv_room: float, budget: float, pl: float
) -> dict[Limit, float]:
    """The four batch limits of spec 3.1: what the clients keep in flight, the cap, the KV room
    and the step budget with ``pl`` prompt tokens per generated token."""
    return {"demand": clients, "cap": cap, "kv": kv_room, "step_budget": budget / (1 + pl)}


def _bound(limits: Mapping[Limit, float]) -> Limit:
    return min(BATCH_LIMITS, key=lambda name: (limits[name], BATCH_LIMITS.index(name)))


def _end(
    s: Step, *, cap: int, budget: int, kv_room: float, pl: float, linear: bool
) -> tuple[float, Limit, float, float]:
    """One end of the step model (spec 3.2): the ratio capped by the prompt ceiling, the limit
    that sets the new batch, the batch, and the prompt share of the new step."""
    limits = _limits(clients=s.demand, cap=cap, kv_room=kv_room, budget=budget, pl=pl)
    binding = _bound(limits)
    new = limits[binding]
    prompt = s.prompt * new * pl / (s.b * s.pl) if s.pl else 0.0
    grown = s.weights + (s.decode - s.weights) * new / s.b if linear else s.decode
    decode = max(grown, s.floor(new))
    ratio = new / s.b * s.t / (prompt + decode)
    if pl:
        ratio = min(ratio, s.capacity / (pl * s.output))
    return ratio, binding, new, prompt / (prompt + decode)


def _why(binding: Limit, *, clients: int, cap: int, budget: int) -> str:
    return {
        "demand": f"as the clients keep {clients} in flight",
        "cap": f"as the cap of {cap} allows",
        "kv": "as many as the KV cache would have room for",
        "step_budget": f"as the per-step token budget of {budget} allows",
    }[binding]


def _line(
    low: float, high: float, binding: Limit, new: float, old: float, share: float, *,
    clients: int, cap: int, budget: int, relieves: set[Bottleneck], note: str = "",
) -> Estimate | None:  # fmt: skip
    low, high = sorted((low, high))
    if low < SHOWN_FROM:
        return None
    why = _why(binding, clients=clients, cap=cap, budget=budget)
    text = f"about {new:.0f} requests would run at once instead of {old:.0f}, {why}"
    inputs = [Input(key="batch", value=new, text=text)]
    named: set[Limit] = {binding}
    if share >= PROMPT_SHARE:
        named.add("prompt")
        said = f"prompt processing would take {share:.0%} of each step"
        inputs.append(Input(key="prompt_share", value=share, text=said))
    return Estimate(
        low=low, high=high, inputs=tuple(inputs), relieves=frozenset(relieves), note=note,
        binding=binding, named=frozenset(named),
    )  # fmt: skip


def _stepped(
    s: Step, *, cap: int, budget: int, relieves: set[Bottleneck], note: str = ""
) -> Estimate | None:
    room = KV_FIT * s.b / s.kv
    low, binding, new, share = _end(s, cap=cap, budget=budget, kv_room=room, pl=s.pl, linear=True)
    high, *_ = _end(s, cap=cap, budget=budget, kv_room=room, pl=s.pl, linear=False)
    return _line(
        low, high, binding, new, s.b, share, clients=s.demand, cap=cap, budget=budget,
        relieves=relieves, note=note,
    )  # fmt: skip


def raise_concurrency_estimate(
    situation: Situation, *, proposed: Settings, run: RunInputs | None, engine: Engine
) -> Estimate | None:
    s = _step(situation, proposed, run, engine)
    raised = proposed.max_concurrent_requests
    if s is None or raised is None or raised <= s.cap:
        return None
    return _stepped(s, cap=raised, budget=s.budget, relieves={"queueing"})


def raise_prefill_batch_estimate(
    situation: Situation, *, proposed: Settings, run: RunInputs | None, engine: Engine
) -> Estimate | None:
    s = _step(situation, proposed, run, engine)
    budget = proposed.prefill_batch_tokens
    if s is None or budget is None or budget <= s.budget:
        return None
    return _stepped(s, cap=s.cap, budget=budget, relieves={"step_budget"}, note=PREFILL_BATCH_COST)


def prefix_caching_estimate(
    situation: Situation, *, proposed: Settings, run: RunInputs | None, engine: Engine
) -> Estimate | None:
    """0.3.8's estimate (block replay over the rendered prompts, the step split at both decode
    efficiencies) with the new batch from the four limits, not today's peak, the reason naming
    the limit that binds, and the ratio capped by the prompt ceiling."""
    s = _step(situation, proposed, run, engine)
    m, settings = situation.measurement, situation.settings
    capabilities = engine.capabilities(None)
    if s is None or run is None or m.ceilings is None:
        return None
    if settings.prefix_caching is not False or not capabilities.shared_prefix_kv:
        return None
    if m.ceilings.moe_experts is not None or situation.facts.sends_images:
        return None
    order, prompts = run.prompt_order, run.prompt_blocks
    if not order or any(index not in prompts for index in order):
        return None
    x, o, b, peak, tpot = (
        m.request_throughput, m.output_throughput, m.mean_running, m.peak_running, m.tpot_p50_ms
    )  # fmt: skip
    pool, block = m.kv_capacity_tokens, m.kv_block_tokens
    if not (x and o and b and peak and tpot and pool and block):
        return None
    block_tokens, out, fill = int(block), o / x, b / peak
    widest = max(b, min(s.demand, s.cap) * fill)
    capacity = int((pool - widest * out) // block_tokens)
    if capacity <= 0:
        return None
    fitting = _kv_batch(order, prompts, max(s.demand, s.cap), KV_FIT * pool, block_tokens, out)
    lag = round(max(b, min(s.demand, s.cap, fitting)))
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
        return s.floor(n) * 1e3

    def end(
        u: float, pick: Callable[[Iterable[float]], float]
    ) -> tuple[float, Limit, float] | None:
        limits = _limits(clients=s.demand, cap=s.cap, kv_room=fitting, budget=s.budget, pl=s.pl * u)
        binding = _bound(limits)
        new = limits[binding]
        if new < b:  # a limit below the batch the run held: the inputs disagree
            return None
        ratio = pick(
            step_ratio(
                tpot=tpot, floor=floor(b), growth=floor(new) / floor(b), batch=new / b,
                uncached=u, eta=eta,
            )
            for eta in (ETA_LOW, ETA_HIGH)
        )  # fmt: skip
        if s.pl and u:
            ratio = min(ratio, s.capacity / (s.pl * u * s.output))
        return ratio, binding, new

    lower, upper = end(worst, min), end(best, max)
    if lower is None or upper is None:
        return None
    (low, binding, new), (high, *_) = lower, upper
    # The limit that binds today, with the KV room the gauge shows, is relieved when another
    # limit binds once the cache holds shared blocks once.
    today = _bound(
        _limits(clients=s.demand, cap=s.cap, kv_room=KV_FIT * s.b / s.kv, budget=s.budget, pl=s.pl)
    )
    relieved = today if today != binding else None
    relieves: set[Bottleneck] = {"prefill"}
    relieves |= {"kv_capacity"} if relieved == "kv" else set()
    relieves |= {"step_budget"} if relieved == "step_budget" else set()
    line = _line(
        low, high, binding, new, b, 0.0, clients=s.demand, cap=s.cap, budget=s.budget,
        relieves=relieves,
    )  # fmt: skip
    if line is None:
        return None
    cached = Input(
        key="uncached_share",
        value=worst,
        text=f"{1 - worst:.0%} of prompt tokens would be served from the cache",
    )
    return replace(line, inputs=(cached, *line.inputs))


def on_gpus(gpus: frozenset[str], estimator: Estimator) -> Estimator:
    """The estimator, giving no estimate on a GPU outside ``gpus``, the GPUs it held on."""

    def estimate(
        situation: Situation, *, proposed: Settings, run: RunInputs | None, engine: Engine
    ) -> Estimate | None:
        ceilings = situation.measurement.ceilings
        if ceilings is None or ceilings.gpu not in gpus:
            return None
        return estimator(situation, proposed=proposed, run=run, engine=engine)

    return estimate


ESTIMATORS: Mapping[str, Estimator] = {  # each with the checked GPUs its estimate held on
    "prefix-caching": on_gpus(frozenset({"NVIDIA A10G"}), prefix_caching_estimate),
    "raise-prefill-batch": on_gpus(frozenset({"NVIDIA A10G"}), raise_prefill_batch_estimate),
}


def lead_bottleneck(estimate: Estimate) -> Bottleneck:
    """The bottleneck a promoted suggestion is grouped under: the first it relieves, in the
    diagnosis's order of classes."""
    return next(name for name in NAMES if name in estimate.relieves)


def _ratio_text(ratio: float, *, down: bool) -> str:
    """ "+N%" up to x2, "xN.N" above; the low end rounded down, so "at least" stays true."""

    def cut(value: float) -> int:
        return round_down(round(value, 6)) if down else round(value)

    if ratio <= 2:
        return f"+{cut((ratio - 1) * 100)}%"
    return f"x{cut(ratio * 10) / 10:.1f}"


def range_text(low: float, high: float) -> str:
    lower, upper = _ratio_text(low, down=True), _ratio_text(high, down=False)
    if lower == upper:
        return f"about {lower}"
    if high <= RANGE_WITHIN * low:
        return f"{lower} to {upper}"
    return f"at least {lower}"


def expected_text(estimate: Estimate) -> str:
    said = [given.text for given in estimate.inputs] + ([estimate.note] if estimate.note else [])
    detail = f" (this run: {'; '.join(said)})" if said else ""
    return f"throughput {range_text(estimate.low, estimate.high)}{detail}"
