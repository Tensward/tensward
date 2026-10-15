"""A step's measuring window and the noise band of what it measured, read from each run's own
``requests.jsonl``: a run's figures, a step compared with its parent on the same prompts (or, when
the parent cannot be read, each run's own noise), and the window a step plans before its launch."""

from __future__ import annotations

import json
import math
import random
import statistics
import tomllib
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

METRICS = (
    "req_s",
    "output_tok_s",
    "ttft_p95_ms",
    "ttft_p95_steady_ms",
    "tpot_p95_ms",
    "e2e_p95_ms",
)
REF_REQUESTS = 236
MIN_REQUESTS = 120
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


def _spread(log_stat: Callable[[Sequence[int]], float]) -> float:
    """Half the 2.5-97.5% range of a log statistic over resampled blocks."""
    rng = random.Random(0)
    resampled = sorted(
        log_stat([rng.randrange(BLOCKS) for _ in range(BLOCKS)]) for _ in range(RESAMPLES)
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


def _own_noise(rows: Sequence[Row], metric: str, *, closed: bool) -> float | None:
    if metric == "ttft_p95_steady_ms":
        steady = _steady(rows)
        rows = [r for r in rows if r.index in steady]
    if len(rows) < 2 * BLOCKS:
        return None
    blocks = _time_blocks(rows)
    if metric in RATES:
        answered = [[r for r in block if _answered(r)] for block in blocks]
        if not all(answered):
            return None
        work = [
            len(block) if metric == "req_s" else sum(r.tokens for r in block) for block in answered
        ]
        per = [sum(_e2e(r) or 0.0 for r in block) if closed else 1.0 for block in answered]
        rates = [w / p for w, p in zip(work, per, strict=True)]
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
    deltas = {}
    for metric in METRICS:
        found = paired(step, parent, metric, open_loop=clients == 0)
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


def plan(
    rate: float | None,
    *,
    clients: int,
    configured: int,
    metric: str,
    predicted: float | None,
    only: float | None = None,
) -> Plan:
    """The requests a step measures: the shortest window whose projected band on ``metric``
    meets the precision rule, or ``only`` seconds; never more than ``configured``."""
    if rate is None or rate <= 0:
        return Plan(configured, None, "no served run gives a rate to plan from")
    bar = target(metric) if predicted is None else max(abs(predicted) / 2, target(metric))

    def meets(seconds: float) -> bool:
        n = math.ceil(rate * seconds)
        return n >= MIN_REQUESTS and _projected(metric, n) <= bar

    seconds = only if only is not None else next(filter(meets, WINDOWS_S), WINDOWS_S[-1])
    paced = math.ceil(rate * seconds)
    requests = max(MIN_REQUESTS, MIN_PER_CLIENT * clients, paced + 2 * clients)
    if requests >= configured:
        return Plan(configured, None, "the configured count is already this short")
    return Plan(
        requests,
        seconds,
        f"{seconds:g} s at {rate:.2f} req/s: {requests} requests, "
        f"projected band ±{_projected(metric, paced):.1%} on {metric}",
    )
