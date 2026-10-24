"""A step's measuring window and the noise band of what it measured, read from each run's own
``requests.jsonl``: a run's figures, a step compared with its parent on the same prompts (or, when
the parent cannot be read, each run's own noise), and the window a step plans before its launch."""

from __future__ import annotations

import json
import math
import random
import statistics
import tomllib
from dataclasses import dataclass, replace
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

METRICS = (
    "req_s",
    "output_tok_s",
    "ttft_p95_ms",
    "ttft_p95_steady_ms",
    "tpot_p95_ms",
    "e2e_p95_ms",
)
REF_REQUESTS = 236
MIN_REQUESTS = 130
JUDGE_REQUESTS = 150  # a quality-risk step: an answer comparison needs about 100 answered pairs
LOWER_IS_BETTER = frozenset(METRICS) - {"req_s", "output_tok_s"}
WINDOWS_S = (60.0, 90.0, 120.0, 180.0, 300.0)
MIN_PER_CLIENT = 4
BLOCKS = 6
T_975_5 = 2.571
RESAMPLES = 500
STEADY = (0.1, 0.9)
MAX_FAILED = 0.02
NOISE_FILE = "data/noise.toml"
RATES = ("req_s", "output_tok_s")
STEADY_MIN_S = 120.0
BIASED = ("req_s", "tpot_p95_ms")
COUNT_MATCHED_FIRST = False  # under tpot and ttft a closed loop's first run sends its steps' count
JUDGED_WHY = "the answer check needs about 100 answered pairs"
WAVE_BINS = 200
WAVE_SIGNAL = 0.2  # the least autocorrelation of completions that shows a wave
MIN_WAVES = 4  # whole waves after the ramp a run needs before their own bands are recorded
WAVES_PER_WINDOW = 2


@dataclass(frozen=True, slots=True)
class Row:
    index: int
    dispatch_ns: int
    first_ns: int | None
    terminal_ns: int | None
    ok: bool
    tokens: int


@dataclass(frozen=True, slots=True)
class Figures:
    requests: int
    seconds: float
    values: Mapping[str, float | None]
    noise: Mapping[str, float | None]


@dataclass(frozen=True, slots=True)
class Delta:
    metric: str
    change: float
    band: float
    verdict: Literal["better", "worse", "within_noise"]
    matched: bool


@dataclass(frozen=True, slots=True)
class Plan:
    requests: int
    seconds: float | None
    reason: str
    steady: Mapping[str, Any] | None = None  # a steady window: its times, source and bias
    noise_source: Literal["table"] | None = None  # the planned band comes from the noise table
    wave_period_s: float | None = None
    waves_band: Mapping[str, float | None] | None = None  # the parent's own waves, unpaired


@dataclass(frozen=True, slots=True)
class Steady:
    """A closed loop's steady part: it settles long after its clients are all in flight, and a
    request dispatched in its last ``tail_s`` would end in the drain."""

    ramp_s: float
    steady_s: float
    tail_s: float

    @property
    def seconds(self) -> float:
        return self.ramp_s + self.steady_s + self.tail_s


@cache
def _table() -> dict[str, dict[str, float]]:
    return tomllib.loads(files("tensward").joinpath(NOISE_FILE).read_text("utf-8"))


def read_rows(run_dir: Path) -> list[Row] | None:
    """The run's measured requests, or None when it kept no ``requests.jsonl``."""
    path = run_dir / "requests.jsonl"
    if not path.is_file():
        return None
    rows = []
    for line in path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows.append(
            Row(
                index=int(record["request_id"].rsplit(":", 1)[1]),
                dispatch_ns=record["dispatch_ns"],
                first_ns=record.get("first_content_ns"),
                terminal_ns=record.get("terminal_ns"),
                ok=record.get("outcome") == "success",
                tokens=int(record.get("committed_output_tokens") or 0),
            )
        )
    return rows


def _answered(row: Row) -> bool:
    return row.ok and row.terminal_ns is not None and row.terminal_ns > row.dispatch_ns


def _ttft(row: Row) -> float | None:
    if not row.ok or row.first_ns is None:
        return None
    return (row.first_ns - row.dispatch_ns) / 1e6


def _tpot(row: Row) -> float | None:
    if not row.ok or row.first_ns is None or row.terminal_ns is None or row.tokens <= 1:
        return None
    return (row.terminal_ns - row.first_ns) / (row.tokens - 1) / 1e6


def _e2e(row: Row) -> float | None:
    if not row.ok or row.terminal_ns is None:
        return None
    return (row.terminal_ns - row.dispatch_ns) / 1e6


PER_ROW: Mapping[str, Callable[[Row], float | None]] = {
    "ttft_p95_ms": _ttft,
    "ttft_p95_steady_ms": _ttft,
    "tpot_p95_ms": _tpot,
    "e2e_p95_ms": _e2e,
}


def _rank(ordered: Sequence[float], q: float) -> float:
    return ordered[math.ceil(q * len(ordered)) - 1]


def _p95(values: Sequence[float]) -> float:
    return _rank(sorted(values), 0.95)


def _span_s(rows: Sequence[Row]) -> float:
    return (max(r.dispatch_ns for r in rows) - min(r.dispatch_ns for r in rows)) / 1e9


def _steady(rows: Sequence[Row]) -> set[int]:
    start = min(r.dispatch_ns for r in rows)
    span = max(r.dispatch_ns for r in rows) - start
    low, high = start + STEADY[0] * span, start + STEADY[1] * span
    return {r.index for r in rows if low <= r.dispatch_ns <= high}


def _before_drain(rows: Sequence[Row]) -> set[int]:
    """The requests that ended by the run's last dispatch. After it no request replaces one
    that ends, so those still running see the load fall (the drain) and finish faster; a
    shorter run drains on prompts a longer one ran under full load."""
    last = max(r.dispatch_ns for r in rows)
    return {r.index for r in rows if r.terminal_ns is not None and r.terminal_ns <= last}


def _values(rows: Sequence[Row], metric: str) -> list[float]:
    return [v for r in rows if (v := PER_ROW[metric](r)) is not None]


def _t_half(logs: Sequence[float]) -> float:
    return T_975_5 * statistics.stdev(logs) / math.sqrt(BLOCKS)


def _spread(log_stat: Callable[[Sequence[int]], float], blocks: int = BLOCKS) -> float:
    """Half the 2.5-97.5% range of a log statistic over resampled blocks."""
    rng = random.Random(0)
    resampled = sorted(
        log_stat([rng.randrange(blocks) for _ in range(blocks)]) for _ in range(RESAMPLES)
    )
    return (_rank(resampled, 0.975) - _rank(resampled, 0.025)) / 2


def _time_blocks(rows: Sequence[Row]) -> list[list[Row]]:
    start = min(r.dispatch_ns for r in rows)
    span = max(r.dispatch_ns for r in rows) - start
    blocks: list[list[Row]] = [[] for _ in range(BLOCKS)]
    for r in rows:
        at = int((r.dispatch_ns - start) * BLOCKS / span) if span else 0
        blocks[min(BLOCKS - 1, at)].append(r)
    return blocks


def _work(
    blocks: Sequence[Sequence[Row]], metric: str, *, closed: bool
) -> tuple[list[float], list[float]] | None:
    """Each block's answered work on a rate metric and what it is per: the block's request
    seconds in a closed loop, else 1; None when a block answered nothing."""
    answered = [[r for r in block if _answered(r)] for block in blocks]
    if not all(answered):
        return None
    work = [
        float(len(block) if metric == "req_s" else sum(r.tokens for r in block))
        for block in answered
    ]
    return work, [sum(_e2e(r) or 0.0 for r in block) if closed else 1.0 for block in answered]


def _own_noise(rows: Sequence[Row], metric: str, *, closed: bool) -> float | None:
    if metric == "ttft_p95_steady_ms":
        steady = _steady(rows)
        rows = [r for r in rows if r.index in steady]
    if len(rows) < 2 * BLOCKS:
        return None
    blocks = _time_blocks(rows)
    if metric in RATES:
        found = _work(blocks, metric, closed=closed)
        if found is None:
            return None
        rates = [w / p for w, p in zip(*found, strict=True)]
        mean = statistics.fmean(rates)
        return T_975_5 * statistics.stdev(rates) / math.sqrt(BLOCKS) / mean if mean else None
    values = [_values(block, metric) for block in blocks]
    if not all(values):
        return None
    return _spread(lambda ids: math.log(_p95([v for i in ids for v in values[i]])))


def figures(rows: Sequence[Row], *, clients: int, upto: int | None = None) -> Figures:
    """The run's figures over its rows (those with ``index < upto``) and its own block noise;
    ``clients`` 0 is an open loop. Rates count every row; latencies and noise leave out the
    drain."""
    kept = [r for r in rows if upto is None or r.index < upto]
    if not kept:
        return Figures(0, 0.0, dict.fromkeys(METRICS), dict.fromkeys(METRICS))
    span = _span_s(kept)
    done = [r for r in kept if r.ok]
    values: dict[str, float | None] = {
        "req_s": len(done) / span if span else None,
        "output_tok_s": sum(r.tokens for r in done) / span if span else None,
    }
    loaded = _before_drain(rows)
    timed = [r for r in kept if r.index in loaded]
    steady = _steady(kept)
    for metric in PER_ROW:
        subset = (
            timed if metric != "ttft_p95_steady_ms" else [r for r in timed if r.index in steady]
        )
        found = _values(subset, metric)
        values[metric] = _p95(found) if found else None
    noise = {m: _own_noise(timed, m, closed=clients > 0) if timed else None for m in METRICS}
    return Figures(len(kept), span, {m: values[m] for m in METRICS}, noise)


def floor(metric: str, repeats: Sequence[float] = ()) -> float:
    """How far one launch moves from another with nothing changed: the table's, or the
    project's own repeats when wider."""
    table = _table()["floor"][metric]
    if len(repeats) < 2:
        return table
    return max(table, 1.96 * statistics.stdev(repeats) / statistics.fmean(repeats))


def target(metric: str) -> float:
    """The band a step without a prediction aims for."""
    return _table()["target"][metric]


def _index_blocks(pairs: Sequence[tuple[Row, Row]]) -> list[Sequence[tuple[Row, Row]]]:
    cuts = [round(b * len(pairs) / BLOCKS) for b in range(BLOCKS + 1)]
    return [pairs[cuts[b] : cuts[b + 1]] for b in range(BLOCKS)]


def paired(
    step: Sequence[Row], parent: Sequence[Row], metric: str, *, open_loop: bool = False
) -> tuple[float, float] | None:
    """The step's change on ``metric`` against its parent over the same prompts, and its paired
    relative 95% half-width; None when the rows cannot give it."""
    by_index = {r.index: r for r in parent if _answered(r)}
    pairs = [
        (r, by_index[r.index])
        for r in sorted(step, key=lambda r: r.index)
        if _answered(r) and r.index in by_index
    ]
    if metric == "ttft_p95_steady_ms" and step:
        steady = _steady(step)
        pairs = [p for p in pairs if p[0].index in steady]
    if len(pairs) < 2 * BLOCKS:
        return None
    blocks = _index_blocks(pairs)
    if metric in RATES:
        if open_loop:
            spans = [(_span_s([p for _, p in b]), _span_s([s for s, _ in b])) for b in blocks]
            if not all(before and after for before, after in spans):
                return None
            logs = [math.log(before / after) for before, after in spans]
        else:
            logs = [
                statistics.fmean(math.log((_e2e(p) or 1.0) / (_e2e(s) or 1.0)) for s, p in b)
                for b in blocks
            ]
        rate = statistics.fmean(logs)
        if metric == "output_tok_s":
            mean_step = statistics.fmean(s.tokens for s, _ in pairs)
            mean_parent = statistics.fmean(p.tokens for _, p in pairs)
            if not mean_step or not mean_parent:
                return None
            rate += math.log(mean_step / mean_parent)
        return math.exp(rate) - 1, _t_half(logs)
    sides = [
        (_values([s for s, _ in b], metric), _values([p for _, p in b], metric)) for b in blocks
    ]
    if not all(s and p for s, p in sides):
        return None

    def log_ratio(ids: Sequence[int]) -> float:
        after = _p95([v for i in ids for v in sides[i][0]])
        before = _p95([v for i in ids for v in sides[i][1]])
        return math.log(after / before) if after > 0 and before > 0 else 0.0

    return math.exp(log_ratio(range(BLOCKS))) - 1, _spread(log_ratio)


def judge(metric: str, change: float, band: float) -> Literal["better", "worse", "within_noise"]:
    if abs(change) <= band:
        return "within_noise"
    return "better" if (change < 0) == (metric in LOWER_IS_BETTER) else "worse"


def unreadable(name: str, rows: list[Row]) -> str | None:
    if not any(r.ok for r in rows):
        return f"the {name} served no request"
    failed = sum(not r.ok for r in rows) / len(rows)
    if failed > MAX_FAILED:
        return f"the {name} had {failed:.1%} failed requests"
    return None


def compare_matched(step_dir: Path, parent_dir: Path, *, clients: int) -> dict[str, Delta] | str:
    """The step against its parent over the prompts both sent and neither ran in its drain, by
    metric; or why they cannot be matched. ``clients`` 0 is an open loop."""
    step, parent = read_rows(step_dir), read_rows(parent_dir)
    if step is None or parent is None:
        return f"the {'step' if step is None else 'parent'} kept no requests.jsonl"
    why = unreadable("step", step) or unreadable("parent", parent)
    if why:
        return why
    upto = min(max(r.index for r in step), max(r.index for r in parent)) + 1
    loaded = _before_drain(step) & _before_drain(parent)
    step = [r for r in step if r.index < upto and r.index in loaded]
    parent = [r for r in parent if r.index < upto and r.index in loaded]
    return _paired_deltas(step, parent, METRICS, open_loop=clients == 0)


def _paired_deltas(
    step: Sequence[Row], parent: Sequence[Row], metrics: Sequence[str], *, open_loop: bool
) -> dict[str, Delta]:
    deltas = {}
    for metric in metrics:
        found = paired(step, parent, metric, open_loop=open_loop)
        if found is None:
            continue
        change, half = found
        band = math.sqrt(half**2 + 2 * floor(metric) ** 2)
        deltas[metric] = Delta(metric, change, band, judge(metric, change, band), True)
    return deltas


def compare_unmatched(after: Figures, before: Figures, metric: str) -> Delta | None:
    """Two runs' figures on ``metric`` compared without matching their prompts."""
    a, b = after.values.get(metric), before.values.get(metric)
    sa, sb = after.noise.get(metric), before.noise.get(metric)
    if a is None or not b or sa is None or sb is None:
        return None
    change = a / b - 1
    band = math.sqrt(sa**2 + sb**2 + 2 * floor(metric) ** 2)
    return Delta(metric, change, band, judge(metric, change, band), False)


def _projected(metric: str, requests: int) -> float:
    paired_noise = _table()["paired60"][metric] * math.sqrt(REF_REQUESTS / requests)
    return math.sqrt(paired_noise**2 + 2 * floor(metric) ** 2)


def wave_period(rows: Sequence[Row]) -> float | None:
    """The seconds between the waves of completions of a closed loop, about one median request:
    the lag of their strongest autocorrelation after it first falls to zero; None when no lag
    shows a wave."""
    start, last = min(r.dispatch_ns for r in rows), max(r.dispatch_ns for r in rows)
    if last <= start:
        return None
    width = (last - start) / WAVE_BINS
    counts = [0] * WAVE_BINS
    for r in rows:
        if _answered(r) and (r.terminal_ns or 0) <= last:
            counts[min(WAVE_BINS - 1, int(((r.terminal_ns or 0) - start) / width))] += 1
    mean = statistics.fmean(counts)
    moved = [c - mean for c in counts]
    total = sum(m * m for m in moved)
    if not total:
        return None
    lags = range(1, WAVE_BINS // 2)
    corr = [sum(a * b for a, b in zip(moved, moved[lag:])) / total for lag in lags]
    fell = next((i for i, c in enumerate(corr) if c <= 0), None)
    if fell is None:
        return None
    best = max(range(fell, len(corr)), key=corr.__getitem__)
    return lags[best] * width / 1e9 if corr[best] >= WAVE_SIGNAL else None


@dataclass(frozen=True, slots=True)
class Waves:
    """A closed loop's wave period and the noise of its whole waves after the ramp."""

    period_s: float
    blocks: int  # whole waves after the ramp
    noise: Mapping[str, float | None]


def _wave_noise(blocks: Sequence[Sequence[Row]], metric: str) -> float | None:
    if metric in RATES:
        found = _work(blocks, metric, closed=True)
        if found is None:
            return None
        work, per = found
        return _spread(
            lambda ids: math.log(sum(work[i] for i in ids) / sum(per[i] for i in ids)),
            len(blocks),
        )
    values = [_values(block, metric) for block in blocks]
    if not all(values):
        return None
    return _spread(lambda ids: math.log(_p95([v for i in ids for v in values[i]])), len(blocks))


def waves(rows: Sequence[Row], *, clients: int) -> Waves | None:
    """The waves of a closed loop of ``clients`` after its ramp, each one period of dispatch,
    and their noise by metric; None under an open loop or when the run shows no wave."""
    period = wave_period(rows) if clients > 0 and rows else None
    found = steady(rows, clients=clients) if period is not None else None
    if period is None or found is None:
        return None
    start = min(r.dispatch_ns for r in rows) + found.ramp_s * 1e9
    count = max(0, int((max(r.dispatch_ns for r in rows) - start) / 1e9 // period))
    blocks: list[list[Row]] = [[] for _ in range(count)]
    for r in _after_ramp(rows, found.ramp_s):
        at = int((r.dispatch_ns - start) / 1e9 // period)
        if at < count:
            blocks[at].append(r)
    noise = {m: _wave_noise(blocks, m) if count >= 2 else None for m in METRICS}
    return Waves(period, count, noise)


def matched_count(clients: int) -> int:
    """The least requests a closed loop of ``clients`` sends in a step from a parent that held
    no request back, and so in its first run under the tpot and ttft goals."""
    return max(MIN_REQUESTS, MIN_PER_CLIENT * clients, JUDGE_REQUESTS)


POST_RAMP_PER_CLIENT = (
    1.3  # requests per client above matched_count that leave 2 x clients after the ramp
)


def post_ramp_count(clients: int) -> int:
    """The requests a first run sends under the ttft goal so that, truncated at that count, it
    keeps ``WAVES_PER_WINDOW`` x ``clients`` requests dispatched after its ramp and ended before
    its drain: the count a step sends (``matched_count``) and ``POST_RAMP_PER_CLIENT`` more per
    client (about 384 + 1.3 x 96 = 509 at 96 clients, from the stored Gemma replays)."""
    return matched_count(clients) + math.ceil(POST_RAMP_PER_CLIENT * clients)


def post_ramp_requests(rows: Sequence[Row], *, clients: int) -> int | None:
    """The least count of ``rows``' first requests whose own ramp leaves ``WAVES_PER_WINDOW`` x
    ``clients`` requests after it and before its drain; None when the run never does."""
    need = WAVES_PER_WINDOW * clients
    for count in range(matched_count(clients), len(rows) + 1, 1):
        cut = [r for r in rows if r.index < count]
        found = steady(cut, clients=clients) if cut else None
        if found is not None and len(_after_ramp(cut, found.ramp_s)) >= need:
            return count
    return None


def judged(found: Plan, configured: int, prompts: int | None = None) -> Plan:
    """``found`` with at least ``JUDGE_REQUESTS``, above ``configured`` too as far as the prompt
    file's ``prompts`` records go (so never above ``configured`` by more than the file holds
    beyond it). The records a run draws are the configured count plus the warm-up, so on a file
    of about that size the requests above ``configured`` repeat prompts the run already sent,
    and on a larger file the sample is still ``configured`` plus the warm-up of distinct prompts:
    the floor adds time, and whether it adds answered pairs depends on how the answer check
    counts a repeated prompt."""
    least = JUDGE_REQUESTS if prompts is None else min(JUDGE_REQUESTS, max(configured, prompts))
    if found.requests >= least:
        return found
    return replace(
        found,
        requests=least,
        seconds=None,
        reason=f"{least} requests, above the configured {configured}: {JUDGED_WHY}",
    )


def first(
    clients: int,
    configured: int,
    *,
    matched: bool,
    prompts: int | None = None,
    ramped: int | bool = False,
) -> Plan:
    """The window of a setup's first run: with ``matched`` (the tpot and ttft goals) a closed
    loop sends the count its steps send (``matched_count``), else the configured count; with
    ``ramped`` (the ttft goal) at least ``post_ramp_count``, or the given count; at least
    ``JUDGE_REQUESTS`` (``judged``), since its answers are the reference."""
    found = Plan(configured, None, "no earlier run to plan from")
    count = matched_count(clients)
    if ramped:
        count = max(count, post_ramp_count(clients) if ramped is True else ramped)
    if matched and clients > 0 and COUNT_MATCHED_FIRST and count < configured:
        found = Plan(count, None, (
            f"{count} requests, the count a step sends from it, so the two carry the same ramp"
        ))  # fmt: skip
    return judged(found, configured, prompts)


def plan(
    rate: float | None,
    *,
    clients: int,
    configured: int,
    metric: str,
    predicted: float | None,
    only: float | None = None,
    quality_risk: bool = False,
    parent: Waves | None = None,
    prompts: int | None = None,
) -> Plan:
    """The requests a step measures: the shortest window whose projected band on ``metric``
    (the noise table's paired band) meets the precision rule and covers ``WAVES_PER_WINDOW`` of
    the ``parent``'s waves after the ramp, or ``only`` seconds, which no count per client raises;
    at least ``MIN_REQUESTS``, or ``JUDGE_REQUESTS`` for a ``quality_risk`` step, and
    ``MIN_PER_CLIENT`` per client with the seconds those requests take; never more than
    ``configured`` but for that floor (``judged``, over the prompt file's ``prompts`` records). The
    parent's wave period and, with ``MIN_WAVES`` waves, their own unpaired bands are recorded
    beside the plan."""
    minimum = JUDGE_REQUESTS if quality_risk else MIN_REQUESTS
    period = parent.period_s if parent is not None else None
    own = (
        {m: round(v, 4) if v is not None else None for m, v in parent.noise.items()}
        if parent is not None
        and parent.blocks >= MIN_WAVES
        and any(v is not None for v in parent.noise.values())
        else None
    )

    def ruled(found: Plan) -> Plan:
        found = replace(
            found, wave_period_s=round(period, 1) if period is not None else None, waves_band=own
        )
        return judged(found, configured, prompts) if quality_risk else found

    if rate is None or rate <= 0:
        return ruled(Plan(configured, None, "no served run gives a rate to plan from"))
    bar = target(metric) if predicted is None else max(abs(predicted) / 2, target(metric))

    def meets(seconds: float) -> bool:
        n = math.ceil(rate * seconds)
        return n >= minimum and _projected(metric, n) <= bar

    least = math.ceil(WAVES_PER_WINDOW * (period or 0.0))
    windows = sorted({max(seconds, least) for seconds in WINDOWS_S})
    seconds = only if only is not None else next(filter(meets, windows), windows[-1])
    paced = math.ceil(rate * seconds)
    requests = max(minimum, paced + 2 * clients)
    if only is None and MIN_PER_CLIENT * clients > requests:
        requests = MIN_PER_CLIENT * clients
        seconds = math.ceil((requests - 2 * clients) / rate)
        paced = math.ceil(rate * seconds)
    if requests >= configured:
        return ruled(Plan(configured, None, "the configured count is already this short"))
    return ruled(Plan(
        requests,
        seconds,
        f"{seconds:g} s at {rate:.2f} req/s: {requests} requests, "
        f"projected band ±{_projected(metric, paced):.1%} on {metric}",
        noise_source="table",
    ))  # fmt: skip


def _after_ramp(rows: Sequence[Row], ramp_s: float) -> list[Row]:
    """The rows dispatched after the ramp that ended by the run's last dispatch."""
    start = min(r.dispatch_ns for r in rows) + ramp_s * 1e9
    loaded = _before_drain(rows)
    return [r for r in rows if r.dispatch_ns >= start and r.index in loaded]


def steady_rate(rows: Sequence[Row], ramp_s: float) -> float | None:
    """Completions per second from the end of the ramp to the run's last dispatch."""
    start = min(r.dispatch_ns for r in rows) + ramp_s * 1e9
    last = max(r.dispatch_ns for r in rows)
    if last <= start:
        return None
    done = sum(1 for r in rows if _answered(r) and start <= (r.terminal_ns or 0) <= last)
    return done / ((last - start) / 1e9) if done else None


def steady(rows: Sequence[Row], *, clients: int) -> Steady | None:
    """The steady part of a closed loop of ``clients``: the ramp lasts until every client is in
    flight and at least two median requests; None under an open loop, when the clients were
    never all in flight, or when nothing ended after the ramp."""
    if clients <= 0 or not rows:
        return None
    ends = [e for r in rows if (e := _e2e(r)) is not None]
    if not ends:
        return None
    start = min(r.dispatch_ns for r in rows)
    events = sorted(
        [(r.dispatch_ns, 1) for r in rows]
        + [(r.terminal_ns, -1) for r in rows if r.terminal_ns is not None]
    )
    in_flight, filled = 0, None
    for at, move in events:
        in_flight += move
        if in_flight >= clients:
            filled = (at - start) / 1e9
            break
    if filled is None:
        return None
    median = statistics.median(ends) / 1e3
    ramp = max(filled, 2 * median)
    rate = steady_rate(rows, ramp)
    if rate is None:
        return None
    return Steady(ramp, max(STEADY_MIN_S, 4 * median, 4 * clients / rate), median)


def steady_requests(rows: Sequence[Row], found: Steady) -> tuple[int, int]:
    """The requests a step sends to dispatch for ``found.seconds`` at the pace of ``rows``, and
    how many of them it selects; past the end of ``rows``, at their steady rate."""
    start = min(r.dispatch_ns for r in rows)
    sent = [r for r in rows if r.dispatch_ns <= start + found.seconds * 1e9]
    beyond = max(0.0, found.seconds - _span_s(rows))
    rate = steady_rate(rows, found.ramp_s) or 0.0
    return (
        len(sent) + math.ceil(rate * beyond),
        len(_after_ramp(sent, found.ramp_s)) + math.floor(rate * beyond),
    )


def bias(parent_rows: Sequence[Row], parent_steady: Steady, *, clients: int) -> dict[str, Delta]:
    """The parent's steady part of the window a step plans from it against the steady part of
    its whole run, on requests per second (completions after the ramp over that span) and
    TPOT p95."""
    start = min(r.dispatch_ns for r in parent_rows)
    sent = [r for r in parent_rows if r.dispatch_ns <= start + parent_steady.seconds * 1e9]
    short = _after_ramp(sent, parent_steady.ramp_s)
    full = _after_ramp(parent_rows, parent_steady.ramp_s)
    deltas: dict[str, Delta] = {}
    rates = steady_rate(sent, parent_steady.ramp_s), steady_rate(parent_rows, parent_steady.ramp_s)
    if short and rates[0] and rates[1]:
        change, band = rates[0] / rates[1] - 1, _projected("req_s", len(short))
        deltas["req_s"] = Delta("req_s", change, band, judge("req_s", change, band), False)
    metric = "tpot_p95_ms"
    after, before = _values(short, metric), _values(full, metric)
    noise = _own_noise(short, metric, closed=clients > 0)
    if after and before and noise is not None:
        change = _p95(after) / _p95(before) - 1
        band = math.sqrt(noise**2 + 2 * floor(metric) ** 2)
        deltas[metric] = Delta(metric, change, band, judge(metric, change, band), False)
    return deltas


def compare_steady(
    step_dir: Path, parent_dir: Path, *, clients: int, parent_steady: Steady
) -> dict[str, Delta] | str:
    """The step against its parent over the prompts both dispatched after their own ramp and
    ended before their own last dispatch: requests per second as the ratio of each run's
    completions after its ramp over that span, the other metrics paired; or why they cannot be
    compared."""
    step, parent = read_rows(step_dir), read_rows(parent_dir)
    if step is None or parent is None:
        return f"the {'step' if step is None else 'parent'} kept no requests.jsonl"
    why = unreadable("step", step) or unreadable("parent", parent)
    if why:
        return why
    own = steady(step, clients=clients)
    if own is None:
        return "the step never had all its clients in flight"
    after = {r.index: r for r in _after_ramp(step, own.ramp_s)}
    before = {r.index: r for r in _after_ramp(parent, parent_steady.ramp_s)}
    both = sorted(after.keys() & before.keys())
    deltas = _paired_deltas(
        [after[i] for i in both], [before[i] for i in both],
        [m for m in METRICS if m != "req_s"], open_loop=False,
    )  # fmt: skip
    rates = steady_rate(step, own.ramp_s), steady_rate(parent, parent_steady.ramp_s)
    if both and rates[0] and rates[1]:
        change, band = rates[0] / rates[1] - 1, _projected("req_s", len(both))
        deltas = {"req_s": Delta("req_s", change, band, judge("req_s", change, band), True),
                  **deltas}  # fmt: skip
    return deltas


def post_ramp(
    step: Sequence[Row], parent: Sequence[Row], *, clients: int, upto: int
) -> Delta | None:
    """TTFT p95 of the step against its parent over the prompts both sent (index under
    ``upto``), dispatched after both runs' ramps and ended before both drains; None under an
    open loop, when a run never had all its clients in flight, or when fewer prompts pair than
    ``WAVES_PER_WINDOW`` waves of ``clients`` send."""
    own, theirs = steady(step, clients=clients), steady(parent, clients=clients)
    if own is None or theirs is None:
        return None
    after = {r.index: r for r in _after_ramp(step, own.ramp_s) if r.index < upto}
    before = {r.index: r for r in _after_ramp(parent, theirs.ramp_s) if r.index < upto}
    both = sorted(after.keys() & before.keys())
    if len(both) < WAVES_PER_WINDOW * clients:
        return None
    metric = "ttft_p95_ms"
    deltas = _paired_deltas(
        [after[i] for i in both], [before[i] for i in both], (metric,), open_loop=False
    )
    return deltas.get(metric)
