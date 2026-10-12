"""What the project's earlier runs say about the next change: what was tried and what it did
on the goal, a last change that lost (revert it first), one that helped less than its run
expected (the limit that took over), the best run, and why nothing is left to try; and what
comes after the top change: a short predicted plan, checked by the next run."""

from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Collection, Mapping, Sequence

from .capabilities import Resolved
from .classify import LESS_WORK, NAMES, Bottleneck, saturation
from .context import AnalysisContext, Goal, current
from .errors import PreflightError
from .kept import REGISTERED, answer_changes
from .playbook import GOAL_WORDS, RELEVANT, Entry, Situation, Suggestion, goal_bottleneck
from .playbook_common import projected_kv, running_at_cap
from .rejected import Rejection, carried
from .runs import NOISE_FLOOR, RUN_RECORD, beyond, within_noise
from .settings import Settings
from .thresholds import KV_PEAK

if TYPE_CHECKING:
    from .engines import Engine
    from .estimate import Estimate
    from .suggest import Suggested

METRIC: dict[str, tuple[str, str, bool]] = {  # the field, its words, whether lower is better
    "throughput": ("request_throughput", "requests per second", False),
    "ttft": ("ttft_p95_ms", "TTFT p95", True),
    "tpot": ("tpot_p95_ms", "TPOT p95", True),
}
SHOWN = 5  # the most recent tried changes listed
REVERT = Entry(
    name="revert-last-change", addresses=frozenset(), applies=lambda situation: None,
    apply=lambda situation: situation.settings, evidence="ours",
    costs="none: these settings were measured before", basis="your runs",
)  # fmt: skip
LOSS_FLOOR = {goal: NOISE_FLOOR[field] for goal, (field, _, _) in METRIC.items()}  # no repeats
LOW_HIT_RATE = 0.10  # a prefix-cache hit rate under this leaves prefix caching as a lever
FULL_PROMPT_RATE = 0.80  # prompt work at this share of the prompt rate leaves little to gain
RAISE_BUDGET = "raise-prefill-batch"
WAVE_WORSE = 0.20  # the start-up wave's TTFT p95 above this share over the base's is said
KNEE = 0.25  # a latency step that gains less than this share of the req/s it costs
# On a saturated GPU, the latency goal's trade-off left: its change and what it does.
TRADE_OFF = {
    "ttft": (
        "small-step-budget",
        "a smaller step budget lets first tokens out sooner",
    ),
    "tpot": (
        "lower-concurrency",
        "running fewer requests at once lowers TPOT, at a throughput cost",
    ),
}


def command_args(command: str) -> frozenset[str]:
    """The ``--engine-arg`` values of a ``tensward analyse`` command."""
    words = shlex.split(command)
    return frozenset(words[i + 1] for i, word in enumerate(words[:-1]) if word == "--engine-arg")


@dataclass(frozen=True, slots=True)
class PastRun:
    run_id: str
    engine_args: frozenset[str]
    settings: Mapping[str, Any]
    metrics: Mapping[str, float | None]
    primary: str | None
    expected: Mapping[frozenset[str], tuple[str, float]]  # by command args: (change, low end)
    compile_cache: str | None = None
    kv_capacity_tokens: float | None = None
    plan: tuple[Mapping[str, Any], ...] = ()
    goal: str | None = None  # the goal the plan was made for
    served: bool = True  # False: the run served no request (its change was tried and failed)
    resolved: Mapping[str, Any] = field(default_factory=dict)  # as the engine resolved them
    crossed: frozenset[str] = frozenset()  # the classes its diagnosis found at warning or worse
    step_budget: float | None = None  # tokens the engine scheduled per step at most
    prompt_rate: float | None = None  # prompt tokens per second the engine measured it can compute
    wave_ttft_p95_ms: float | None = None  # the start-up wave's TTFT p95, outside the window
    wave_requests: int | None = None

    @classmethod
    def read(cls, run_dir: Path, snapshot_id: str | None, engine: Engine) -> PastRun | None:
        """The run, when it is complete and of ``snapshot_id``."""
        try:
            run = json.loads((run_dir / RUN_RECORD).read_text("utf-8"))
            metrics = json.loads((run_dir / "metrics.json").read_text("utf-8"))
            report = json.loads((run_dir / "report.json").read_text("utf-8"))
        except (OSError, ValueError):
            return None
        if snapshot_id is not None and run.get("snapshot_id") != snapshot_id:
            return None
        args = frozenset(run.get("engine_args", ()))
        if not metrics.get("succeeded"):
            return cls(run_id=run_dir.name, engine_args=args, settings={}, metrics={},
                       primary=None, expected={}, served=False)  # fmt: skip
        launched = metrics.get("resolved")
        try:
            resolved = asdict(engine.effective(
                Settings.from_json(run.get("settings", {})),
                Resolved(**launched) if isinstance(launched, dict) else None,
            ))  # fmt: skip
        except (PreflightError, TypeError):
            resolved = {}
        blocks = [
            block for section in report.get("sections", ()) for block in section.get("blocks", ())
            if block.get("kind") == "suggestion" and block.get("command") and block.get("expected")
        ]  # fmt: skip
        findings = (metrics.get("diagnosis") or {}).get("findings", ())
        return cls(
            run_id=run_dir.name,
            engine_args=args,
            settings=run.get("settings", {}),
            metrics={name: metrics.get(name) for name, _, _ in METRIC.values()},
            primary=(metrics.get("diagnosis") or {}).get("primary"),
            expected={
                command_args(b["command"]): (b["name"], b["expected"]["low"]) for b in blocks
            },
            compile_cache=run.get("compile_cache"),
            kv_capacity_tokens=metrics.get("kv_capacity_tokens"),
            step_budget=metrics.get("step_budget_tokens"),
            prompt_rate=metrics.get("prompt_capacity_tok_s"),
            wave_ttft_p95_ms=(metrics.get("startup_wave") or {}).get("ttft_p95_ms"),
            wave_requests=(metrics.get("startup_wave") or {}).get("requests"),
            plan=tuple((report.get("goal") or {}).get("plan") or ()),
            goal=(report.get("goal") or {}).get("goal"),
            resolved=resolved,
            crossed=frozenset(
                f["bottleneck"] for f in findings if f.get("state") in ("warning", "critical")
            ),
        )


def past_runs(project: Path, engine: Engine, *, before: str | None = None) -> list[PastRun]:
    """The project's complete runs of its registered snapshot, oldest first; with ``before``,
    only those older than that run. The run being written has no run.json yet, so it is never
    its own history."""
    try:
        snapshot = json.loads((project / "project.json").read_text("utf-8")).get("snapshot_id")
    except (OSError, ValueError):
        snapshot = None
    runs = project / "runs"
    paths = (
        sorted(p for p in runs.iterdir() if before is None or p.name < before)
        if runs.is_dir()
        else []
    )
    found = [PastRun.read(path, snapshot, engine) for path in paths]
    return [run for run in found if run is not None]


def flags(args: frozenset[str]) -> frozenset[str]:
    """The flag names of engine arguments: ``max-num-seqs=64`` -> ``max-num-seqs``."""
    return frozenset(arg.partition("=")[0] for arg in args)


def cap_flag(engine: Engine) -> str | None:
    """The engine argument that sets the concurrency cap (``max-num-seqs`` on vLLM)."""
    found = flags(
        frozenset(engine.engine_args_between(Settings(), Settings(max_concurrent_requests=2)))
    )
    return next(iter(found)) if len(found) == 1 else None


def cold_kv_tokens(runs: Sequence[PastRun], args: frozenset[str], flag: str | None) -> float | None:
    """The smallest KV pool a served cold start among ``runs`` measured with the engine
    arguments ``args`` apart from the cap: what a change of the cap, which restarts cold, gets."""

    def rest(given: frozenset[str]) -> frozenset[str]:
        return frozenset(a for a in given if a.partition("=")[0] != flag)

    pools = [run.kv_capacity_tokens for run in runs
             if run.served and run.compile_cache == "cold" and run.kv_capacity_tokens
             and rest(run.engine_args) == rest(args)]  # fmt: skip
    return min(pools, default=None)


def cold_pools(
    runs: Sequence[PastRun], args: frozenset[str], flag: str | None
) -> tuple[tuple[int | None, float], ...]:
    """The step budget and KV pool of each served cold start among ``runs`` with the engine
    arguments ``args`` apart from the cap, oldest first."""

    def rest(given: frozenset[str]) -> frozenset[str]:
        return frozenset(a for a in given if a.partition("=")[0] != flag)

    return tuple(
        (round(run.step_budget) if run.step_budget else None, run.kv_capacity_tokens)
        for run in runs
        if run.served and run.compile_cache == "cold" and run.kv_capacity_tokens
        and rest(run.engine_args) == rest(args)
    )  # fmt: skip


def parent(args: frozenset[str], before: Sequence[PastRun]) -> PastRun | None:
    """The newest served run of ``before`` that sets only flags ``args`` sets, with other
    arguments, the most arguments first: the run a change was made from (a flag added, or its
    value changed)."""
    found = [
        run for run in before
        if run.served and run.engine_args != args and flags(run.engine_args) <= flags(args)
    ]  # fmt: skip
    return max(found, key=lambda run: (len(run.engine_args), run.run_id), default=None)


def loss_threshold(runs: Sequence[PastRun], goal: str) -> float:
    """How far the goal's metric must move the wrong way to count as a loss: twice the largest
    spread between runs of identical engine arguments, at least 2%; LOSS_FLOOR without repeats."""
    metric = METRIC[goal][0]
    groups: dict[frozenset[str], list[float]] = {}
    for run in runs:
        if (value := run.metrics.get(metric)) is not None:
            groups.setdefault(run.engine_args, []).append(value)
    spreads = [(max(v) - min(v)) / (sum(v) / len(v)) for v in groups.values() if len(v) > 1]
    return max(0.02, 2 * max(spreads)) if spreads else LOSS_FLOOR[goal]


def _gain(after: float | None, before: float | None, goal: str) -> float | None:
    """The change on the goal's metric, positive when better."""
    if after is None or not before:
        return None
    moved = after / before - 1
    return -moved if METRIC[goal][2] else moved


def _moved(after: float | None, before: float | None) -> str:
    return f"{(after / before - 1):+.0%}" if after is not None and before else "unmeasured"


def _shift(after: float, before: float, field: str) -> str:
    """The move as a percent, "unchanged" when it rounds to none, and "within noise" when it is
    inside the field's NOISE_FLOOR."""
    moved = _moved(after, before)
    if not before or moved in ("+0%", "-0%"):
        return "unchanged" if before else moved
    return f"{moved}, within noise" if within_noise(after, before, field) else moved


def metric_figure(value: float, field: str) -> str:
    if field == "request_throughput":
        return f"{value:#.3g}".rstrip(".")
    return f"{value:,.0f} ms" if value >= 100 else f"{value:.1f} ms"


def trade_base(runs: Sequence[PastRun]) -> PastRun | None:
    """The newest served run with the engine arguments of the project's first served run."""
    first = next((run for run in runs if run.served), None)
    if first is None:
        return None
    return [run for run in runs if run.served and run.engine_args == first.engine_args][-1]


def _seconds(ms: float) -> str:
    return f"{ms / 1000:.1f} s"


def trade(
    runs: Sequence[PastRun], situation: Situation, goal: str
) -> tuple[tuple[str, ...], dict[str, Any]] | None:
    """Under a latency goal, the goal metric, req/s and the other latency since the trade base,
    measured, and on a second line the start-up wave when its TTFT p95 rose by more than
    WAVE_WORSE and both waves held two requests or more: the lines for So far and the record
    for report.json."""
    base = trade_base(runs)
    if goal == "throughput" or base is None:
        return None
    other = "tpot" if goal == "ttft" else "ttft"
    shown: list[str] = []
    record: dict[str, Any] = {"since": base.run_id}
    for metric in (goal, "throughput", other):
        field, words, _ = METRIC[metric]
        before, now = base.metrics.get(field), getattr(situation.measurement, field)
        if before is None or now is None:
            continue
        shown.append(
            f"{words} {metric_figure(before, field)} -> {metric_figure(now, field)} "
            f"({_shift(now, before, field)})"
        )
        record[field] = [before, now]
    wave = situation.measurement.startup_wave
    waited: list[str] = []
    if (
        wave is not None
        and wave.ttft_p95_ms is not None
        and base.wave_ttft_p95_ms
        and wave.requests >= 2
        and (base.wave_requests or 0) >= 2
        and wave.ttft_p95_ms > base.wave_ttft_p95_ms * (1 + WAVE_WORSE)
    ):
        sizes = (
            f"{wave.requests} requests"
            if base.wave_requests == wave.requests
            else f"{base.wave_requests} requests in run {base.run_id}, {wave.requests} in this run"
        )
        waited = [
            f"the start-up wave ({sizes} sent at once, not in the figures above) waited longer: "
            f"TTFT p95 {_seconds(base.wave_ttft_p95_ms)} -> {_seconds(wave.ttft_p95_ms)} "
            f"({_moved(wave.ttft_p95_ms, base.wave_ttft_p95_ms)})"
        ]
        record["wave"] = {
            "before_ms": base.wave_ttft_p95_ms,
            "after_ms": wave.ttft_p95_ms,
            "requests": wave.requests,
            "base_requests": base.wave_requests,
        }
    since = f"since run {base.run_id} (the latest run of your starting settings): "
    since += "; ".join(shown)
    return (since, *waited), record


def trade_summary(record: Mapping[str, Any], goal: str) -> str:
    """One line of what the trade record measured: its gains, its costs and what moved within
    noise."""
    gains: list[str] = []
    costs: list[str] = []
    quiet: list[str] = []
    for metric in (goal, "throughput", "tpot" if goal == "ttft" else "ttft"):
        field, words, lower = METRIC[metric]
        if field not in record:
            continue
        before, now = record[field]
        if not before:
            continue
        shift = _moved(now, before)
        moved = f"{words} {'unchanged' if shift in ('+0%', '-0%') else shift}"
        if within_noise(now, before, field):
            quiet.append(moved)
        else:
            (gains if (now < before) == lower else costs).append(moved)
    if wave := record.get("wave"):
        costs.append(f"start-up wave TTFT p95 {_moved(wave['after_ms'], wave['before_ms'])}")
    parts = [
        f"{name}: {', '.join(moved)}"
        for name, moved in (("gains", gains), ("costs", costs), ("within noise", quiet))
        if moved
    ]
    return f"Measured since run {record['since']}: {'; '.join(parts)}."


def _values(args: frozenset[str]) -> dict[str, str]:
    """The value each flag of ``args`` sets: ``max-num-seqs=64`` -> ``{"max-num-seqs": "64"}``."""
    return dict(arg.partition("=")[::2] for arg in args)


def _direction(after: str | None, before: str | None) -> int:
    """1 when the numeric value ``after`` is above ``before``, -1 below, 0 when either is not a
    number or they are equal."""
    try:
        moved = float(after or "") - float(before or "")
    except ValueError:
        return 0
    return (moved > 0) - (moved < 0)


def knee(
    runs: Sequence[PastRun], args: frozenset[str], situation: Situation, goal: str
) -> tuple[dict[str, int], str] | None:
    """Under a latency goal, the newest step back from this run that gained on the goal less
    than KNEE of the req/s it cost and whose flags still hold the values it set (a later step
    that set one again is judged from there): those flags, each with the way it moved its value
    (``_direction``), and its measured trade."""
    if goal == "throughput":
        return None
    field, words, _ = METRIC[goal]
    measured = situation.measurement
    after: Mapping[str, float | None] = {
        field: getattr(measured, field),
        "request_throughput": measured.request_throughput,
    }
    after_args, after_name, index, held = args, "this one", len(runs), _values(args)
    while (base := parent(after_args, runs[:index])) is not None:
        gain = _gain(after.get(field), base.metrics.get(field), goal)
        rate, before = after.get("request_throughput"), base.metrics.get("request_throughput")
        cost = 1 - rate / before if rate and before else None
        made = after_args - base.engine_args
        values = _values(after_args)
        current = all(held.get(flag) == values[flag] for flag in flags(made))
        if current and gain is not None and cost is not None and 0 < cost and gain < KNEE * cost:
            step = "the last step" if after_name == "this one" else "the step"
            base_values = _values(base.engine_args)
            moved = {flag: _direction(values[flag], base_values.get(flag)) for flag in flags(made)}
            return moved, (
                f"{step} ({' '.join(sorted(made))}, run {base.run_id} to {after_name}) moved "
                f"{words} {_moved(after.get(field), base.metrics[field])} for requests per "
                f"second {_moved(rate, before)}"
            )
        index = next(i for i, run in enumerate(runs) if run is base)
        after_args, after_name, after = base.engine_args, f"run {base.run_id}", base.metrics
    return None


@dataclass(frozen=True, slots=True)
class Memory:
    lines: tuple[str, ...]  # what the earlier runs say, newest last
    revert: Suggestion | None  # the last change lost on the goal: go back first
    tried: frozenset[str]  # offered changes left out because a run already measured them
    best: tuple[str, float] | None  # the best run on the goal's metric, this one included
    tried_for: frozenset[str] = frozenset()  # the classes those tried changes addressed
    rejected: frozenset[str] = frozenset()  # offered changes left out: the user rejected them
    ahead: PastRun | None = None  # the best run on the goal's metric, when an earlier one
    behind: bool = False  # this run is worse than ``ahead`` by more than the noise band


def _tried_lines(
    runs: Sequence[PastRun], named: Mapping[str, str], args: frozenset[str], goal: str,
    rejected: Collection[str] = (),
) -> list[str]:  # fmt: skip
    """One line per change the runs measured, oldest first: the newest run of each set of
    arguments against the run it was made from, and whether this run kept the change. The
    changes ``named`` (run id -> offered change) are always listed, the others up to SHOWN; the
    runs ``rejected`` are left to their rejection's line."""
    metric, words, _ = METRIC[goal]
    newest = {run.engine_args: run for run in runs}
    lines: list[str] = []
    seen: set[tuple[frozenset[str], frozenset[str]]] = set()
    for index, run in reversed(list(enumerate(runs))):
        if run is not newest[run.engine_args] or (len(lines) >= SHOWN and run.run_id not in named):
            continue
        if run.run_id in rejected:
            continue
        before = parent(run.engine_args, runs[:index])
        base_args = before.engine_args if before is not None else frozenset()
        if not run.engine_args - base_args or (run.engine_args, base_args) in seen:
            continue
        seen.add((run.engine_args, base_args))
        made = run.engine_args - base_args
        change = " ".join(sorted(made))
        label = f"`{named[run.run_id]}` ({change})" if run.run_id in named else change
        if not run.served:
            lines.insert(0, f"{label} tried in run {run.run_id}: it served no request")
            continue
        now = sorted(a for a in args if a.partition("=")[0] in flags(made) and a not in made)
        fate = (
            "kept in this run"
            if made <= args
            else f"changed to {' '.join(now)} in this run"
            if now
            else "reverted"
        )
        moved = (
            f"{words} {_moved(run.metrics[metric], before.metrics[metric])} "
            f"against run {before.run_id}"
            if before is not None
            else "no earlier run to compare with"
        )
        lines.insert(0, f"{label} tried in run {run.run_id}: {moved}, {fate}")
    return lines


def _revert(engine_args: frozenset[str], project_path: Path, reason: str) -> Suggestion:
    """The first change to try: analyse with ``engine_args`` again (none: the registered setup);
    with a setup kept, the registered setup is ``tensward set registered`` first, since a plain
    analyse would run the kept one."""
    if current().kept is not None and not engine_args:
        command = shlex.join(["tensward", "set", REGISTERED, "--project", str(project_path)])
    else:
        words = ["tensward", "analyse", "--project", str(project_path)]
        for text in sorted(engine_args):
            words += ["--engine-arg", text]
        command = shlex.join([*words, *current().flags])
    return Suggestion(
        entry=REVERT, reason=reason, tier="try_first", basis="your runs", command=command
    )


def remember(
    suggestions: Sequence[Suggestion],
    situation: Situation,
    *,
    goal: Goal,
    args: frozenset[str],
    runs: Sequence[PastRun],
    project_path: Path,
    rejections: Sequence[Rejection] = (),
    refused: Sequence[tuple[Suggestion, Rejection]] = (),
    floor_req_s: float | None = None,
) -> tuple[list[Suggestion], Memory]:
    """``suggestions`` without the changes a run already measured, and what the runs say: one
    line per tried change, a revert when this run's change lost on the goal by more than
    ``loss_threshold`` or is one the user rejected, the change that helped less than its run
    expected, and the best run (never one that carries a rejected change, nor one below
    ``floor_req_s`` requests per second, the declared floor). A rejection gets one line when its
    run is history, its change was ``refused`` here (left out of the suggestions), or this run
    carries it."""
    metric, words, lower = METRIC[goal]
    now = getattr(situation.measurement, metric)
    by_args = {run.engine_args: run for run in runs}
    kept: list[Suggestion] = []
    tried: set[str] = set()
    tried_for: set[str] = set()
    named: dict[str, str] = {}
    for suggestion in suggestions:
        run = by_args.get(command_args(suggestion.command)) if suggestion.command else None
        if run is None:
            kept.append(suggestion)
            continue
        tried.add(suggestion.entry.name)
        tried_for.update(suggestion.addresses)
        if not run.engine_args < args:  # a run this one was made from did not make the change
            named[run.run_id] = suggestion.entry.name
    lines = _tried_lines(runs, named, args, goal, {r.run for r in rejections})
    own = carried(rejections, args)
    left_out = {r.run: s.entry.name for s, r in refused}
    history = {run.run_id for run in runs}
    lines += [
        f"{f'`{left_out[r.run]}` ({r.words})' if r.run in left_out else r.words} tried in run "
        f"{r.run}: {r.because}"
        for r in rejections
        if r.run in history or r.run in left_out or r is own
    ]
    revert = None
    made_from = next((run for run in runs if own and run.run_id == own.made_from), None)
    if own is not None and made_from is not None and made_from.served:
        revert = _revert(
            made_from.engine_args, project_path,
            f"you rejected {own.words} (run {own.run}): go back to run {made_from.run_id}'s "
            "settings",
        )  # fmt: skip
    if (base := parent(args, runs)) is not None:
        then = base.metrics[metric]
        gain = _gain(now, then, goal)
        change = " ".join(sorted(args - base.engine_args))
        lost = gain is not None and gain < 0 and beyond(gain, loss_threshold(runs, goal))
        if revert is None and lost:
            revert = _revert(
                base.engine_args, project_path,
                f"the last change ({change}) moved {words} {_moved(now, then)} "
                f"against run {base.run_id}: go back to that run's settings",
            )  # fmt: skip
        expected = base.expected.get(args)
        before_rate = base.metrics["request_throughput"]
        rate = situation.measurement.request_throughput
        ratio = rate / before_rate if rate and before_rate else None
        if (
            goal == "throughput"
            and revert is None
            and expected
            and ratio is not None
            and ratio < expected[1]
        ):
            took = situation.diagnosis.primary
            lines.append(
                f"`{expected[0]}` helped less than run {base.run_id} expected (measured "
                f"x{ratio:.2f}, expected at least x{expected[1]:.2f}): "
                f"{NAMES[took] if took else 'no limit crossed'} took over, the next target"
            )

    def inside(rate: float | None) -> bool:
        return floor_req_s is None or (rate is not None and rate >= floor_req_s)

    scored = [
        (run.run_id, run.metrics.get(metric))
        for run in runs
        if run.served and carried(rejections, run.engine_args) is None
        and inside(run.metrics.get("request_throughput"))
    ]  # fmt: skip
    rate = situation.measurement.request_throughput
    here = [] if own is not None or not inside(rate) else [("this run", now)]
    known = [(name, value) for name, value in [*scored, *here] if value is not None]
    best = (min if lower else max)(known, key=lambda pair: pair[1]) if known else None
    ahead = next(
        (run for run in runs if best and own is None and run.run_id == best[0]
         and run.engine_args != args),
        None,
    )  # fmt: skip
    behind = False
    if ahead is not None and now is not None and (then := ahead.metrics.get(metric)) is not None:
        gap = _gain(now, then, goal)
        behind = gap is not None and gap < 0 and beyond(gap, loss_threshold(runs, goal))
        if behind:
            lines.append(
                f"this run is worse on {words} than run {ahead.run_id}, the best run so far: "
                f"{metric_figure(then, metric)} there, {metric_figure(now, metric)} here "
                f"({_moved(now, then)})"
            )
    return kept, Memory(
        lines=tuple(lines),
        revert=revert,
        tried=frozenset(tried),
        best=best,
        tried_for=frozenset(tried_for) | {a for s, _ in refused for a in s.addresses},
        rejected=frozenset(left_out.values()),
        ahead=ahead,
        behind=behind,
    )


def stop_reason(
    suggestions: Sequence[Suggestion],
    situation: Situation,
    memory: Memory,
    goal: str,
    left_out: Sequence[str] = (),
    ruled_out: Sequence[str] = (),
    knee: str | None = None,
    applied: Sequence[str] = (),
    blocked: Collection[str] = (),
) -> str | None:
    """Why Try first is empty for ``goal``, and whether more is possible: never just "nothing to
    change". None when Try first has a change. ``knee`` is the measured trade of a last step
    whose further change left Try first for it; ``applied`` the entries whose change this run
    already carries; ``blocked`` the classes whose change Try first shows as blocked."""
    if memory.revert is not None or any(s.tier == "try_first" for s in suggestions):
        return None
    if knee is not None:
        return (
            f"Nothing left in Try first for {GOAL_WORDS[goal]}: the knee is reached; {knee}; "
            "the change that would go further is under Could help."
        )
    diagnosis = situation.diagnosis
    saturated = next((f for f in diagnosis.findings if f.bottleneck == "gpu_saturated"), None)
    if saturated is not None and saturated.state == "critical":
        left = _saturated_left(situation, goal, memory, ruled_out, applied)
        return f"Nothing left in Try first for {GOAL_WORDS[goal]}: {left}."
    if (memory.tried or memory.rejected) and memory.best is not None:
        name, value = memory.best
        best = f"the best run is {name} ({METRIC[goal][1]} {metric_figure(value, METRIC[goal][0])})"
        tried = "tried or rejected" if memory.rejected else "tried"
        if suggestions:
            return (
                f"Every Try-first change Tensward has for {GOAL_WORDS[goal]} here was {tried}; "
                f"{best}; Could help lists untried changes with weaker signals."
            )
        return f"Every change Tensward has for {GOAL_WORDS[goal]} here was {tried}; {best}."
    crossed = [*([diagnosis.primary] if diagnosis.primary else []), *diagnosis.secondary]
    later: dict[str, list[str]] = {}  # a Could-help change: the crossed classes it addresses
    bare, held = [], []
    for b in crossed:
        if (s := next((s for s in suggestions if b in s.addresses), None)) is not None:
            later.setdefault(s.entry.name, []).append(NAMES[b])
        elif b in blocked:
            held.append(NAMES[b])
        else:
            bare.append(NAMES[b])
    found = [f"no change in this engine's playbook applies to {', '.join(bare)}"] if bare else []
    found += [f"the change for {' and '.join(held)} is blocked (see Try first)"] if held else []
    found += [f"for {' and '.join(v)}, `{n}` is under Could help" for n, v in later.items()]
    parts = [f"Nothing left in Try first for {GOAL_WORDS[goal]}: "
             f"{'; '.join(found) or 'no measured signal points to a change for it'}"]  # fmt: skip
    if left_out:
        parts.append(f"left out because they worsen it: {', '.join(f'`{n}`' for n in left_out)}")
    if ruled_out:
        parts.append(f"ruled out here: {', '.join(f'`{n}`' for n in ruled_out)}")
    parts.append(_unexplained(situation, goal))
    return "; ".join(parts) + "."


def _unexplained(situation: Situation, goal: str) -> str:
    """The closing part of a stop with no saturated GPU: when prompt processing is the goal's
    bottleneck, how much of the measured prompt rate it used and the levers left; else where to
    look for more."""
    m = situation.measurement
    share, rate = m.prompt_work_share, m.prompt_capacity_tok_s
    if goal_bottleneck(situation.diagnosis, goal) != "prefill" or not share or not rate:
        return (
            "nothing measured says the GPU is at its limit, so more may be possible: a trace "
            "(`--trace`) or another goal can show where"
        )
    hit = m.prefix_cache_hit_rate
    levers = []
    if hit is None:
        levers.append("prefix caching, if prompts share text")
    elif hit < LOW_HIT_RATE:
        levers.append(f"prefix caching (hit rate {hit:.0%})")
    levers.append("a smaller or quantized model, a faster GPU")
    return (
        f"prompt processing ran at {share:.0%} of the prompt rate this engine measured "
        f"({rate:,.0f} tokens/s), about {max(0.0, 1 - share):.0%} headroom for scheduling "
        f"changes; what is left: {', '.join(levers)}"
    )


def _saturated_left(
    situation: Situation, goal: str, memory: Memory, ruled_out: Sequence[str],
    applied: Sequence[str],
) -> str:  # fmt: skip
    """What a saturated GPU leaves for ``goal``: no server setting adds throughput, and with
    the clients setting the batch the test decides what runs at once; a latency goal keeps one
    trade-off and why it is not in Try first, without citing the saturation (it limits
    throughput, not first-token time)."""
    diagnosis, m = situation.diagnosis, situation.measurement
    gpu = diagnosis.gpu_state
    held = saturation(gpu) if gpu is not None else "the GPU was saturated"
    if goal not in TRADE_OFF:
        if m.peak_in_flight and m.peak_running and round(m.peak_running) >= m.peak_in_flight:
            return (
                f"{held}: the {m.peak_in_flight} clients set the batch (every request in flight "
                f"was running), so the test, not a server setting, decides how much runs at "
                f"once; what else can help is {LESS_WORK}"
            )
        return f"{held}: server settings cannot add throughput here; what can is {LESS_WORK}"
    name, does = TRADE_OFF[goal]
    if name in memory.tried:
        why = "was tried (see So far)"
    elif name in memory.rejected:
        why = "was rejected (see So far)"
    elif name in ruled_out:
        why = "is ruled out here (see Not applicable here)"
    elif name in applied:
        why = "is applied in this run"
    else:
        why = "has no signal here"
    if goal == "ttft":
        steps = (
            f"TTFT p95 {m.ttft_p95_ms:.0f} ms is about {m.ttft_p95_ms / m.step_ms:.1f} decode "
            f"steps of {m.step_ms:.0f} ms; "
            if m.ttft_p95_ms and m.step_ms
            else ""
        )
        return (
            f"{steps}the first-token changes in this playbook were tried or do not apply here; "
            f"{does}, but `{name}` {why}; what else can help is {LESS_WORK}"
        )
    return f"{held}; {does}, but `{name}` {why}; beyond it, what helps is {LESS_WORK}"


MAX_STEPS = 3
SIZED_RAISE, DOUBLED_RAISE = "kv-sized-concurrency", "raise-concurrency"
CAP_RAISES = (SIZED_RAISE, DOUBLED_RAISE)
MEDIA_LEVERS = frozenset({"media_inputs", "media_limits"})
# An estimate's binding limit: its class and the signal that would show it binding.
LIMIT_CLASS: dict[str, tuple[Bottleneck, str]] = {
    "demand": ("load_limited", "requests in flight below the concurrency cap with none waiting"),
    "cap": ("queueing", "requests waiting at the concurrency cap"),
    "kv": ("kv_capacity", "KV-cache usage near full with requests waiting"),
    "step_budget": ("step_budget", "steps filled to the step budget while prompts wait"),
    "prompt": ("prefill", "TTFT rising against TPOT"),
}


def plan_for(
    top: Suggestion, entries: Sequence[Entry], goal: str, estimate: Estimate | None,
    tried: frozenset[str], *, situation: Situation,
) -> tuple[dict[str, Any], ...]:  # fmt: skip
    """At most three steps from ``top``: the limit likely to bind after it (the estimate's
    binding limit under the throughput goal, else the entry's ``then``, checked against this run
    by ``_first_limit``), the next change for that limit among ``entries`` and the signal that
    would confirm it. A later change never undoes an earlier one; under a latency goal the plan
    ends before a step whose limit does not matter to the goal. A step is ``tested`` when its
    limit is not crossed in this run, so the next run can confirm it."""
    then = top.entry.then
    limit, signal = (then.limit, then.signal) if then else (None, None)
    if goal == "throughput" and estimate is not None and estimate.binding is not None:
        limit, signal = LIMIT_CLASS[estimate.binding]
    if limit is not None:
        limit, signal = _first_limit(top, situation, goal, limit, signal)
    args = sorted(command_args(top.command)) if top.command else []
    steps = [{"change": top.entry.name, "args": args, "limit": limit, "signal": signal,
              "tested": _tested(situation, limit)}]  # fmt: skip
    used, planned = _raises_alike(top.entry.name) | set(tried), [top.entry]
    relevant = RELEVANT.get(goal)
    while limit is not None and len(steps) < MAX_STEPS:
        found = next(
            (e for e in entries if limit in e.addresses and e.name not in used
             and not any(_undoes(e, earlier) for earlier in planned)
             and (goal not in e.effects or e.effects[goal].direction == "improves")),
            None,
        )  # fmt: skip
        if found is None:
            break
        limit = found.then.limit if found.then else None
        if relevant is not None and limit is not None and limit not in relevant:
            break
        used |= _raises_alike(found.name)
        planned.append(found)
        steps.append({"change": found.name, "args": [], "limit": limit,
                      "signal": found.then.signal if found.then else None,
                      "tested": _tested(situation, limit)})  # fmt: skip
    return tuple(steps)


def _tested(situation: Situation, limit: Bottleneck | None) -> bool:
    """Whether the next run can test ``limit``: it is not crossed in this run."""
    return limit is not None and not situation.fired(limit)


def _first_limit(
    top: Suggestion, situation: Situation, goal: str, limit: Bottleneck | None, signal: str | None
) -> tuple[Bottleneck | None, str | None]:
    """The limit to name after ``top``: KV capacity only when the KV usage projected at the
    offered cap reaches the classifier's warning level; under a latency goal only a class that
    matters to it; else the first crossed class the change does not address (under a latency
    goal, the goal's own crossed bottleneck, which stays), or none; the projection is the
    signal of either."""
    diagnosis, proposed = situation.diagnosis, top.settings
    note = ""
    if limit == "kv_capacity" and proposed is not None and proposed.max_concurrent_requests:
        cap = proposed.max_concurrent_requests
        projected = projected_kv(situation, cap, caching=proposed.prefix_caching is True)
        if projected is not None:
            pool = situation.measurement.kv_capacity_tokens
            tokens = f" ({projected * pool:,.0f} of {pool:,.0f} tokens)" if pool else ""
            share = "under 1%" if projected < 0.005 else f"about {projected:.0%}"
            note = f"KV-cache usage projected at {share}{tokens} at a cap of {cap}"
            if projected >= KV_PEAK.warning:
                return limit, f"{note}, near full with requests waiting"
            limit = None
    relevant = RELEVANT.get(goal)
    if limit is not None and (relevant is None or limit in relevant):
        return limit, signal
    crossed = [*([diagnosis.primary] if diagnosis.primary else []), *diagnosis.secondary]
    left = [c for c in crossed if c not in top.addresses and (relevant is None or c in relevant)]
    if not left and relevant is not None and (own := goal_bottleneck(diagnosis, goal)) in relevant:
        left = [own]
    return (left[0], note or None) if left else (None, note or None)


def _raises_alike(name: str) -> set[str]:
    """``name``, and the other raise of the concurrency cap when it is one: a plan raises the cap
    once."""
    return set(CAP_RAISES) if name in CAP_RAISES else {name}


def _undoes(entry: Entry, earlier: Entry) -> bool:
    """Whether ``entry`` turns a setting ``earlier`` turned with the opposite effect on a metric."""
    opposite = {("improves", "worsens"), ("worsens", "improves")}
    return bool(entry.levers & earlier.levers) and any(
        (effect.direction, earlier.effects[metric].direction) in opposite
        for metric, effect in entry.effects.items()
        if metric in earlier.effects
    )


def fits(entry: Entry, situation: Situation, ruled_out: Sequence[str]) -> bool:
    """Whether ``entry`` can apply to this engine, model and workload whatever the run measured:
    no quality risk and no change of the answers, not ruled out here, no gate blocks it, and a
    text-only model is not asked to drop media encoders it does not have."""
    facts, settings = situation.facts, situation.settings
    return (
        not (entry.quality_risk or entry.changes_answers)
        and entry.name not in ruled_out
        and not any(gate.blocks(facts, settings) for gate in entry.gates)
        and (facts.media_encoders or not (entry.levers and entry.levers <= MEDIA_LEVERS))
    )


def plan_lines(plan: Sequence[Mapping[str, Any]], offered: Collection[str]) -> tuple[str, ...]:
    """The plan as report lines, each labelled a prediction; a first limit crossed in this run
    is said to still bind, which no run can confirm, and a first step with no limit left shows
    its projection. A later step is named only while it is one of the changes ``offered`` in
    this report."""
    lines = []
    for index, step in enumerate(plan):
        limit = (
            f"{NAMES[step['limit']]} likely binds next (watch for {step['signal']})"
            if step["limit"]
            else ""
        )
        if index == 0 and step["limit"] and not step["tested"]:
            note = f"; {step['signal']}" if step["signal"] else ""
            lines.append(f"then: {NAMES[step['limit']]} likely still binds{note}")
        elif index == 0:
            if limit:
                lines.append(f"then (prediction): {limit}")
            elif step["signal"]:
                lines.append(f"then: no other limit is crossed now; {step['signal']}")
        elif step["change"] in offered:
            after = f", after which {limit}" if limit else ""
            lines.append(f"then (prediction): `{step['change']}`{after}")
        else:
            break
    return tuple(lines)


def checked(
    runs: Sequence[PastRun], args: frozenset[str], situation: Situation, goal: str, *,
    engine: Engine,
) -> tuple[str | None, dict[str, Any] | None]:  # fmt: skip
    """The first prediction of the newest run whose plan's first change this run made, against
    what this run shows, when that plan was made for this run's goal: confirmed when its limit is
    crossed now, and held (still binds, not a test) when that limit was crossed when the plan was
    made too; can't tell for decode bandwidth when this run speculates, as the decode ceiling is
    not modelled then."""
    before = next(
        (run for run in reversed(runs)
         if run.plan and args and frozenset(run.plan[0].get("args") or ()) == args),
        None,
    )  # fmt: skip
    if before is None:
        return None, None
    step = before.plan[0]
    if before.goal != goal or not step.get("limit"):
        return None, None
    if (
        step["limit"] == "decode_bandwidth"
        and engine.lever_value(situation.settings, "speculation") is not None
    ):
        text = (
            f"run {before.run_id} predicted {NAMES['decode_bandwidth']} would bind after "
            f"`{step['change']}`: can't tell, as Tensward does not model the decode ceiling while "
            "speculating"
        )
        return text, {"run": before.run_id, "change": step["change"], "predicted": step["limit"],
                      "actual": goal_bottleneck(situation.diagnosis, goal), "confirmed": False,
                      "held": False, "cant_tell": True}  # fmt: skip
    diagnosis = situation.diagnosis
    actual = goal_bottleneck(diagnosis, goal)
    crossed = situation.fired(step["limit"])
    held = crossed and (not step.get("tested", True) or step["limit"] in before.crossed)
    if held:
        shown = "it does (not a test: it was crossed when the plan was made)"
    elif crossed:
        shown = "confirmed"
    elif actual is None:
        shown = "not confirmed: this run crossed no limit"
    else:
        evidence = next(f.evidence for f in diagnosis.findings if f.bottleneck == actual)
        shown = f"not confirmed: this run shows {NAMES[actual]} ({evidence})"
    said = "said {} still binds" if held else "predicted {} would bind"
    text = (
        f"run {before.run_id} {said.format(NAMES[step['limit']])} after `{step['change']}`: {shown}"
    )
    check = {"run": before.run_id, "change": step["change"], "predicted": step["limit"],
             "actual": actual, "confirmed": crossed and not held, "held": held}  # fmt: skip
    return text, check


def _window_scale(cap_now: int | None, cap_new: int | None, at_cap: bool) -> float:
    """How much longer the same requests take at the lowered cap, when the run held its cap."""
    return max(1.0, cap_now / cap_new) if at_cap and cap_now and cap_new else 1.0


def seconds_to_test(
    context: AnalysisContext, *, cap_now: int | None = None, cap_new: int | None = None,
    at_cap: bool = False,
) -> float | None:  # fmt: skip
    """Seconds to test one change here: the load plus the measured window, which a lowered cap
    the run held stretches."""
    total, load = context.test_s, context.load_s
    if total is None or load is None:
        return None
    return load + (total - load) * _window_scale(cap_now, cap_new, at_cap)


def time_to_test(
    context: AnalysisContext, *, cap_now: int | None = None, cap_new: int | None = None,
    at_cap: bool = False,
) -> str | None:  # fmt: skip
    """What testing one change takes on this setup, from this run's own timings."""
    seconds = seconds_to_test(context, cap_now=cap_now, cap_new=cap_new, at_cap=at_cap)
    total, load = context.test_s, context.load_s
    if seconds is None or total is None or load is None:
        return None
    text = (
        f"about {max(1, round(seconds / 60))} min on this setup (this run: {load:.0f} s to load, "
        f"{(total - load) / 60:.1f} min to warm up and measure"
    )
    if (scale := _window_scale(cap_now, cap_new, at_cap)) > 1:
        text += (
            f"; at a cap of {cap_new} instead of {cap_now} the same requests take "
            f"about x{scale:.1f} as long"
        )
    return text + ")"


def warm_pool(runs: Sequence[PastRun], resolved: Mapping[str, Any]) -> tuple[str, float] | None:
    """The KV pool the newest warm start measured whose resolved settings are ``resolved``."""
    for run in reversed(runs):
        if (
            run.served
            and run.compile_cache == "warm"
            and run.resolved == resolved
            and run.kv_capacity_tokens
        ):
            return run.run_id, run.kv_capacity_tokens
    return None


def _at_prompt_rate(
    suggestions: Sequence[Suggestion], runs: Sequence[PastRun], situation: Situation, goal: str
) -> list[Suggestion]:
    """``suggestions`` with a larger step budget moved from Try first to Could help under the
    throughput and TTFT goals when prompt work ran at FULL_PROMPT_RATE or more of the measured
    prompt rate and no run with a larger budget measured a higher rate: a larger budget changes
    the chunk size, not the rate."""
    m = situation.measurement
    budget, rate, share = m.step_budget_tokens, m.prompt_capacity_tok_s, m.prompt_work_share
    if goal == "tpot" or not budget or not rate or share is None or share < FULL_PROMPT_RATE:
        return list(suggestions)
    if any(run.step_budget and run.step_budget > budget and run.prompt_rate
           and run.prompt_rate > rate for run in runs):  # fmt: skip
        return list(suggestions)
    cost = (
        "prompt processing already runs near the prompt rate this engine measured, and a "
        "larger step budget changes the chunk size, not that rate (no run here with a larger "
        "budget measured a higher one)"
    )
    return [
        replace(s, tier="could_help", cost=f"{cost}; {s.cost or s.entry.costs}")
        if s.tier == "try_first" and s.entry.name == RAISE_BUDGET
        else s
        for s in suggestions
    ]


def _past_knee(
    suggestions: Sequence[Suggestion],
    found: tuple[Mapping[str, int], str] | None,
    args: frozenset[str],
) -> tuple[list[Suggestion], str | None]:
    """``suggestions`` with each Try-first change that moves a flag of a last step past the knee
    further the same way (or sets it at all, when that step's direction is not known) moved to
    Could help, its measured trade leading the cost; and that trade, when one moved."""
    if found is None:
        return list(suggestions), None
    made, words = found
    current = _values(args)
    moved, out = False, []
    for s in suggestions:
        own = _values(command_args(s.command) - args) if s.command else {}
        if s.tier == "try_first" and any(
            flag in made and made[flag] in (0, _direction(value, current.get(flag)))
            for flag, value in own.items()
        ):
            moved = True
            cost = f"knee reached: {words}; {s.cost or s.entry.costs}"
            s = replace(s, tier="could_help", cost=cost)
        out.append(s)
    return out, words if moved else None


def _lowers_req_s(suggestion: Suggestion, situation: Situation) -> bool:
    """Whether the change lowers requests per second: its throughput effect worsens, or it
    lowers the cap on requests in flight (with no cap set, below this run's peak running)."""
    effect = suggestion.entry.effects.get("throughput")
    if effect is not None and effect.direction == "worsens":
        return True
    new = suggestion.settings.max_concurrent_requests if suggestion.settings else None
    cap = situation.settings.max_concurrent_requests
    limit = cap if cap is not None else situation.measurement.peak_running
    return limit is not None and new is not None and new < limit


def _within_floor(
    runs: Sequence[PastRun], args: frozenset[str], floor: Mapping[str, Any], *,
    engine: Engine, registered: Settings, quantized: bool,
) -> PastRun | None:  # fmt: skip
    """The newest served run with other engine arguments whose requests per second were within
    the floor below the base's; never one that carries a rejected change, nor one whose settings
    can change answers against the registered setup's unless they are the kept setup's."""
    context = current()
    least = floor["req_s"][0] * (1 - floor["floor"])
    kept = frozenset(context.kept.engine_args) if context.kept is not None else None
    found = [
        run for run in runs
        if run.served and run.engine_args != args
        and (run.metrics.get("request_throughput") or 0) >= least
        and carried(context.rejected, run.engine_args) is None
        and (run.engine_args == kept or not answer_changes(
            engine, registered, Settings.from_json(run.settings), checkpoint_quantized=quantized
        ))
    ]  # fmt: skip
    return found[-1] if found else None


def floor_req_s(base: PastRun | None, situation: Situation) -> float | None:
    """The least requests per second the declared floor allows: the floor below the trade
    base's (this run's, with no earlier run); None without a floor or under the throughput
    goal."""
    context = current()
    loss = context.max_req_s_loss
    rate = situation.measurement.request_throughput
    before = base.metrics.get("request_throughput") if base is not None else rate
    if context.goal == "throughput" or loss is None or not before:
        return None
    return before * (1 - loss)


def _below_floor(
    suggestions: Sequence[Suggestion], base: PastRun | None, situation: Situation
) -> tuple[list[Suggestion], dict[str, Any] | None]:
    """``suggestions`` under the declared floor on requests per second, judged on measurements:
    each Try-first change that lowers them moves to Could help once this run is more than the
    floor below ``base``'s, else its cost says how much room is left; and that comparison. With
    no earlier run, this run is the base."""
    context = current()
    now = situation.measurement.request_throughput
    before = base.metrics.get("request_throughput") if base is not None else now
    loss = context.max_req_s_loss
    if context.goal == "throughput" or loss is None or not before or now is None:
        return list(suggestions), None
    label = f"run {base.run_id}" if base is not None else "this run"
    drop = 1 - now / before
    crossed = now < before * (1 - loss)
    out = []
    for s in suggestions:
        if s.tier == "try_first" and _lowers_req_s(s, situation):
            cost = s.cost or s.entry.costs
            if crossed:
                s = replace(s, tier="could_help", cost=(
                    f"past your floor: requests per second {now:.3g}, {drop:.0%} below "
                    f"{label}'s {before:.3g} (your floor: {loss:.0%}); {cost}"
                ))  # fmt: skip
            else:
                now_text = "" if base is None else f", and this run made {now:.3g}"
                s = replace(s, cost=(
                    f"requests per second may fall: your floor is {before * (1 - loss):.3g} "
                    f"({loss:.0%} below {label}'s {before:.3g}){now_text}; {cost}"
                ))  # fmt: skip
        out.append(s)
    return out, {
        "base": base.run_id if base else None,
        "floor": loss,
        "req_s": [before, now],
        "crossed": crossed,
    }


def _applied(entries: Sequence[Entry], situation: Situation) -> tuple[str, ...]:
    """The latency trade-off entries whose change this run's settings already carry."""
    names = {name for name, _ in TRADE_OFF.values()}
    return tuple(
        e.name for e in entries if e.name in names and e.apply(situation) == situation.settings
    )


def guided(
    suggested: Suggested,
    situation: Situation,
    *,
    engine: Engine,
    entries: Sequence[Entry],
    project_path: Path | None,
    registered: Settings,
) -> Suggested:
    """``suggested`` with the project's history applied: tried changes out, a revert first after
    a loss or once this run is past the declared floor (never to a run whose settings can change
    answers against ``registered``, the registered setup, unless they are the kept setup's), the
    history lines (under a latency goal, the trade since the latest run of the starting settings
    leads them), a larger step budget at the prompt rate, a change going past the knee or the
    floor moved to Could help and, when Try first is empty, why; the short plan after the top
    change, the check of the parent run's plan, and the time a test takes. A load-limited run's
    Try first is the advice to raise the load, so it gets no stop reason."""
    from .suggest import first_try

    context = current()
    args = frozenset(context.engine_args)
    runs = (
        past_runs(project_path, engine, before=context.reported_run)
        if project_path is not None
        else []
    )
    kept, memory = remember(
        suggested.suggestions, situation, goal=context.goal, args=args, runs=runs,
        project_path=project_path or Path("."),
        rejections=context.rejected,
        refused=suggested.rejected,
        floor_req_s=floor_req_s(trade_base(runs), situation),
    )  # fmt: skip
    traded = trade(runs, situation, context.goal)
    if traded is not None:
        memory = replace(memory, lines=(*traded[0], *memory.lines))
    kept = _at_prompt_rate(kept, runs, situation, context.goal)
    found_knee = knee(runs, args, situation, context.goal)
    kept, past_knee = _past_knee(kept, found_knee, args)
    kept, floor = _below_floor(kept, trade_base(runs), situation)
    if floor is not None and floor["crossed"] and memory.revert is None:
        before = _within_floor(runs, args, floor, engine=engine, registered=registered,
                               quantized=situation.facts.quantization is not None)  # fmt: skip
        if before is not None:
            then, now = floor["req_s"]
            memory = replace(memory, revert=_revert(
                before.engine_args, project_path or Path("."),
                f"this run made {now:.3g} requests per second, {1 - now / then:.0%} below run "
                f"{floor['base']}'s {then:.3g}, past your floor of {floor['floor']:.0%}: go back "
                f"to run {before.run_id}'s settings first (the newest run within the floor)",
            ))  # fmt: skip
    if suggested.carried is not None and memory.revert is None:
        base = parent(args, runs)
        back = f"run {base.run_id}'s settings" if base is not None else "the registered setup"
        memory = replace(memory, revert=_revert(
            base.engine_args if base is not None else frozenset(), project_path or Path("."),
            f"{suggested.carried}: go back to {back} before trying another change",
        ))  # fmt: skip
    if (
        suggested.underloaded is None and memory.revert is None
        and not any(s.tier == "try_first" for s in kept)
    ):  # fmt: skip
        kept = [replace(s, tier="try_first") if s.entry.name in suggested.neutral else s
                for s in kept]  # fmt: skip
        kept, again = _past_knee(kept, found_knee, args)
        past_knee = past_knee or again
    ruled_out = tuple(name for name, _ in suggested.not_applicable)
    runnable = [
        replace(s, tier="could_help")
        if s.tier == "try_first" and not s.command and project_path is not None
        else s
        for s in kept
    ]  # a project's Try first with no command to run gets the stop reason too
    stop = (
        None
        if suggested.underloaded is not None
        else stop_reason(runnable, situation, memory, context.goal, suggested.other_goals,
                         ruled_out, knee=past_knee, applied=_applied(entries, situation),
                         blocked=tuple(suggested.blocked))
    )  # fmt: skip
    if stop is not None and context.goal != "throughput":
        stop = (
            f"{stop[:-1]}; no measured limit calls for a change that leaves "
            f"{GOAL_WORDS[context.goal]} alone."
        )
    if stop is not None and floor is not None and floor["crossed"]:
        stop = (
            f"{stop} Your floor of {floor['floor']:.0%} fewer requests per second than run "
            f"{floor['base']} is reached."
        )
    shown = replace(suggested, suggestions=tuple(kept), memory=memory, floor=floor,
                    trade=traded[1] if traded else None)  # fmt: skip
    top = first_try(situation.diagnosis, shown) if memory.revert is None else None
    candidates = [entry for entry in entries if fits(entry, situation, ruled_out)]
    plan = (
        plan_for(top, candidates, context.goal, suggested.estimate(top),
                 memory.tried | memory.rejected, situation=situation)
        if top else ()
    )  # fmt: skip
    line, check = checked(runs, args, situation, context.goal, engine=engine)
    lines = memory.lines + ((line,) if line else ())
    cap = situation.settings.max_concurrent_requests
    at_cap = running_at_cap(situation.measurement, situation.settings)
    test_times = {
        s.entry.name: text
        for s in kept
        if s.command and s.settings
        and _window_scale(cap, s.settings.max_concurrent_requests, at_cap) > 1
        and (text := time_to_test(context, cap_now=cap,
                                  cap_new=s.settings.max_concurrent_requests, at_cap=at_cap))
    }  # fmt: skip
    return replace(
        shown, memory=replace(memory, lines=lines), stop=stop, plan=plan,
        plan_lines=plan_lines(plan, {s.entry.name for s in kept}), check=check,
        test_time=time_to_test(context),
        test_times=test_times, warm_kv=warm_pool(runs, asdict(situation.settings)),
    )  # fmt: skip
