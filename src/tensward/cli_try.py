"""``tensward try``, ``undo``, ``runs`` and ``export``: the commands around the kept setup."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .cli_options import JSON_OUTPUT, engine_for, runtime_for
from .engines import Engine
from .errors import PROJECT_INPUTS_INVALID, TOO_SLOW, PreflightError
from .files import write_private
from .gpu_busy import needs_for, refuse_if_busy
from .kept import (
    REGISTERED,
    Kept,
    NotKept,
    flag_words,
    keep,
    latest_run,
    pinned_baseline,
    read_kept,
    undo,
)
from .project import ResolvedProject, load_project, project_engine, settings_for
from .rejected import (
    REWORDED,
    NotRejected,
    Rejection,
    read_rejected,
    refused_by,
    reject,
    reject_failed_start,
    standing,
)
from .report.beyond import ABOVE, IN_REPORT, beyond_choices, beyond_summary
from .runids import resolve_run
from .runs import RunFile
from .runtime import MACHINE_PROBLEMS
from .stands import Row, limit_line, number, stands_rows
from .text import columns

NO_RUN = "no analysis to try from yet: run tensward analyse first"
NO_OFFER = "no offer left: every change in the last report was tried or rejected"
NO_CHANGE = "nothing to try: the last report offers no change"
NOTHING_WORTH = "Nothing to try: the last run found no change worth testing for your goal."
NO_COMPARISON = "no comparison"
SLOW_KEPT = "it starts, slow on this machine; you accepted"
SHARE_DROP = 0.02
FAILING_CALLS_MORE = 2
CALLS_TO_JUDGE = 40
REVERTED = 3
METRICS = {
    "req_s": ("Requests/s", "request_throughput"),
    "output_tok_s": ("Output tok/s", "output_throughput"),
    "ttft_p95_ms": ("First word, slowest 1 in 20", "ttft_p95_ms"),
    "tpot_p95_ms": ("Per token, slowest 1 in 20", "tpot_p95_ms"),
    "e2e_p95_ms": ("Full answer, slowest 1 in 20", "e2e_p95_ms"),
}
# What each metric is called in a sentence, and whether more is better.
SAID = {
    "req_s": ("throughput", True),
    "output_tok_s": ("output speed", True),
    "ttft_p95_ms": ("the wait for the first word", False),
    "tpot_p95_ms": ("the time per word", False),
    "e2e_p95_ms": ("the time to a full answer", False),
}
# What a goal metric is called where a change offered for another goal made it worse.
GOAL_SAID = {
    "req_s": "requests per second",
    "ttft_p95_ms": "time to first token",
    "tpot_p95_ms": "time per token",
}
TOOL_FLOOR = 0.95
TOOL_SHARES = {
    "produced_call": "calls made",
    "valid_json": "valid JSON",
    "known_tool": "right tool",
    "schema_valid": "arguments fit the tool's schema",
}


class NothingToTry(Exception):
    """No offer to try; ``str(error)`` is the sentence shown, with the command that fixes it."""


@dataclass(frozen=True, slots=True)
class Offer:
    n: int
    name: str
    engine_args: tuple[str, ...]
    engine_env: tuple[str, ...]
    why: str
    test_seconds: float | None
    changes_answers: bool
    first: bool = False
    side: str | None = None  # the other goal it is offered for, when none is for the goal

    def document(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "name": self.name,
            "engine_args": list(self.engine_args),
            "engine_env": list(self.engine_env),
            "why": self.why,
            **({"side": self.side} if self.side else {}),
        }


@dataclass(frozen=True, slots=True)
class TryVerdict:
    keep: bool
    gate: str | None
    lines: list[str]  # under the table: the goal metric against its noise
    checked: dict[str, Any] | None = None  # answers and tool calls when there is no step
    rows: tuple[Row, ...] = ()  # the table's answers and tool calls rows


SERVE_TEXT = "stop tuning and serve what's kept now"
NO_MORE = "No more changes to try."
MENU_COMMAND = 36  # widest command column; a longer command has its text on the next line
WARMUP_REQUESTS = 8  # four sent, and the first ones answer slower than the rest


def next_choices(
    following: str | None,
    *,
    kept: bool = False,
    reworded: bool = False,
    slow: bool = False,
    after_keep: str | None = None,
) -> list[tuple[str, str]]:
    """The "Next:" menu after a ``try`` step: (command, what it does), in print order.
    ``following`` is the next offer's flags, None when there is none; ``after_keep`` the flags
    of the offer that follows when a reworded change is kept. ``slow``: the change starts where
    the current setup did not, but was too slow to measure fully."""
    choices: list[tuple[str, str]] = []
    if reworded:
        choices.append(
            (
                "tensward try --keep",
                f"keep it, then try {after_keep}"
                if after_keep is not None
                else "keep it; nothing else to try after it",
            )
        )
    if following is not None:
        skip = "skip it and try" if reworded or slow else "try"
        choices.append(("tensward try", f"{skip} {following}"))
    if slow:
        choices.append(("tensward try --keep", "keep it so you can serve it"))
    if kept:
        choices.append(("tensward undo", "drop this change"))
    choices.append(("tensward serve", SERVE_TEXT))
    return choices


def next_lines(
    choices: list[tuple[str, str]], *, more: bool = True, wrap_up: str | None = None
) -> list[str]:
    """The menu as printed; ``more`` False says first that there is no next change, as the
    ``wrap_up`` line when the kept setup has a speed to report."""
    head = [] if more else [wrap_up or NO_MORE]
    width = max([22, *(len(c) + 2 for c, _ in choices if len(c) + 2 <= MENU_COMMAND)])
    lines = [*head, "Next:"]
    for command, text in choices:
        if len(command) + 2 > width:
            lines += [f"  {command}", f"  {'':<{width}}{text}"]
        else:
            lines.append(f"  {command:<{width}}{text}")
    return lines


def next_document(choices: list[tuple[str, str]]) -> dict[str, str]:
    return dict(choices)


def keep_reworded(project: Path) -> tuple[str, str]:
    """``try --keep``: keep the latest run that ``try`` refused only for reworded answers, as
    ``tensward set RUN --accept-answer-changes`` does, or that started where the current setup
    did not but was too slow to measure fully; return its change as flags and why it is kept."""
    from .rejected import clear

    run = latest_run(project)
    refusal = next((r for r in read_rejected(project) if r.run == run), None)
    resolved = load_project(project, verify_weights=False)
    pins = pinned_baseline(project, resolved, project_engine(resolved.record.current_setup))
    if refusal is None and run is not None and _too_slow(project / "runs" / run) is not None:
        kept = read_kept(project)
        if kept is None or kept.run != run:
            keep(project, run, accept_unreviewed=True)
            change = RunFile.read(project / "runs" / run).engine_args
            words = flag_words([a for a in change if a not in pins] or change)
            return words, SLOW_KEPT
    if refusal is None or REWORDED not in (refusal.reason or ""):
        raise NotKept(
            "--keep applies only right after a try that was not kept because its answers are "
            "worded differently (none wrong), or that starts but was too slow to measure here"
        )
    keep(project, refusal.run, accept_unreviewed=True)
    clear(project, refusal.run)
    words = flag_words([a for a in refusal.change if a not in pins] or refusal.change)
    return words, "answers reworded, you accepted"


def _too_slow(run_dir: Path) -> dict[str, Any] | None:
    """The pace a run was cut short at, when its setup started where the current one did not
    but would have taken too long to measure fully; None for any other run."""
    try:
        found = json.loads((run_dir / "window.json").read_text("utf-8")).get("too_slow")
    except (OSError, ValueError, AttributeError):
        return None
    return found if isinstance(found, dict) else None


def offers(project: Path) -> list[Offer]:
    """The changes the newest run of the current setup offers, in report order, without those
    the user rejected. A run of a setup that was reverted is not that run: its offers were
    diagnosed on settings that were thrown away."""
    project = project.expanduser()
    resolved = load_project(project, verify_weights=False)
    engine = project_engine(resolved.record.current_setup)
    kept = read_kept(project)
    current = frozenset((*kept.engine_args, *kept.env)) if kept is not None else frozenset()
    run_dir = _current_run(project, resolved, engine, current)
    if run_dir is None:
        return [_first_offer(project, resolved, engine, current)]
    refusing, _ = standing(read_rejected(project), engine.flag_rules)
    return _report_offers(run_dir, refusing, current, engine)


def offers_if_kept(project: Path, run_id: str) -> tuple[Offer | None, frozenset[str]]:
    """The first change ``try`` would offer once run ``run_id``, refused for reworded answers,
    is kept with ``try --keep``, and the arguments that setup carries."""
    from .history import recorded_env

    project = project.expanduser()
    resolved = load_project(project, verify_weights=False)
    engine = project_engine(resolved.record.current_setup)
    run_dir = project / "runs" / run_id
    run_file = RunFile.read(run_dir)
    current = frozenset(run_file.engine_args) | recorded_env(
        run_file.model_dump(), engine.flag_rules
    )
    others = [r for r in read_rejected(project) if r.run != run_id]
    refusing, _ = standing(others, engine.flag_rules)
    try:
        return _report_offers(run_dir, refusing, current, engine)[0], current
    except (NothingToTry, OSError, ValueError, KeyError):
        return None, current


def _report_offers(
    run_dir: Path, refusing: tuple[Rejection, ...], current: frozenset[str], engine: Engine
) -> list[Offer]:
    found: list[Offer] = []
    suggested = 0
    report = json.loads((run_dir / "report.json").read_text("utf-8"))
    for section in report["sections"]:
        for block in section["blocks"]:
            if block.get("kind") != "suggestion" or not block.get("command"):
                continue
            suggested += 1
            offered = frozenset((*block["engine_args"], *block["engine_env"]))
            if refused_by(refusing, offered, current, engine.flag_rules) is not None:
                continue
            found.append(
                Offer(
                    n=len(found) + 1,
                    name=block["name"],
                    engine_args=tuple(block["engine_args"]),
                    engine_env=tuple(block["engine_env"]),
                    why=block["reason"],
                    test_seconds=block.get("test_seconds"),
                    changes_answers=bool(block.get("changes_answers")),
                    side=block.get("side"),
                )
            )
    if not found:
        raise NothingToTry(NO_OFFER if suggested else NO_CHANGE)
    return found


def _current_run(
    project: Path, resolved: ResolvedProject, engine: Engine, current: frozenset[str]
) -> Path | None:
    """The newest complete run of the current setup, None when there is none."""
    from .history import recorded_env

    runs = project / "runs"
    names = sorted((p.name for p in runs.iterdir()), reverse=True) if runs.is_dir() else []
    for name in names:
        run_dir = runs / name
        if not all((run_dir / f).is_file() for f in ("run.json", "metrics.json", "report.json")):
            continue
        run_file = RunFile.read(run_dir)
        args = frozenset(run_file.engine_args) | recorded_env(
            run_file.model_dump(), engine.flag_rules
        )
        if run_file.snapshot_id == resolved.record.snapshot_id and args == current:
            return run_dir
    return None


def _current_unstarted(project: Path) -> bool:
    """The current setup has served no request: it has no run, or its newest run failed all."""
    from .runs import RunMetrics

    resolved = load_project(project, verify_weights=False)
    kept = read_kept(project)
    current = frozenset((*kept.engine_args, *kept.env)) if kept is not None else frozenset()
    engine = project_engine(resolved.record.current_setup)
    run_dir = _current_run(project, resolved, engine, current)
    return run_dir is None or not RunMetrics.read(run_dir / "metrics.json").succeeded


def _first_offer(
    project: Path, resolved: ResolvedProject, engine: Engine, current: frozenset[str]
) -> Offer:
    """The offload for a model that does not fit the GPU, offered before any run exists."""
    from .project import fit_offer

    refusing, _ = standing(read_rejected(project), engine.flag_rules)
    tried = [word for r in refusing for word in r.change]
    seeded = fit_offer(resolved, engine, project, refused=tried)
    if seeded is None:
        from .project import project_fit_advice

        advice = project_fit_advice(resolved, engine, resolved.record.current_setup.engine_settings)
        raise NothingToTry(advice[0] if advice and advice[1] else NO_RUN)
    name, args, why = seeded
    if refused_by(refusing, frozenset(args), current, engine.flag_rules) is not None:
        raise NothingToTry(NO_OFFER)
    return Offer(1, name, args, (), why, None, True, first=True)


def _tool_gate(before: dict[str, Any], after: dict[str, Any]) -> tuple[str, float, float] | None:
    """The first share of tool-call behaviour that fell by more than 2 points while at least 2
    more calls failed, as its key and the two shares."""
    for key in TOOL_SHARES:
        was, now = before.get(key), after.get(key)
        if was is None or now is None:
            continue
        failing = [
            round(
                (1 - share) * (stats.get("requests" if key == "produced_call" else "calling") or 0)
            )
            for share, stats in ((was, before), (now, after))
        ]
        if was - now > SHARE_DROP and failing[1] - failing[0] >= FAILING_CALLS_MORE:
            return key, was, now
    return None


def _clause(metric: str, change: float) -> str:
    """What a metric did, in words: "throughput rose 6%", "the wait for the first word got 72%
    worse"."""
    name, more_is_better = SAID.get(metric, (metric, True))
    size = f"{abs(change):.0%}"
    if more_is_better:
        return f"{name} {'rose' if change > 0 else 'fell'} {size}"
    return f"{name} got {size} {'worse' if change > 0 else 'better'}"


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


NOTHING_SIGNIFICANT = (
    "Nothing changed significantly: differences this small also happen between two identical runs"
)


def _moved(metric: str, change: float) -> str:
    """A move beyond the noise: "+33%: a real gain", "+6% slower: a real change"."""
    more_is_better = SAID.get(metric, ("", True))[1]
    if (change > 0) == more_is_better:
        return f"{change:+.0%}: a real gain"
    return f"{change:+.0%} {'lower' if more_is_better else 'slower'}: a real change"


def _metric_line(step: dict[str, Any]) -> str:
    """The goal metric's move when it is beyond the measurement noise; a move inside it that the
    report still recommends held your floor and limits."""
    label = METRICS.get(step["metric"], (step["metric"], ""))[0]
    if abs(step["change"]) > step["band"]:
        return f"{label}: {_moved(step['metric'], step['change'])}."
    if step["recommended"]:
        return f"{label}: no significant change; it held your floor and limits."
    return f"{NOTHING_SIGNIFICANT}."


def _goal_line(step: dict[str, Any]) -> str | None:
    """For a change offered for another goal: how the goal metric moved, which must not get
    worse beyond its noise."""
    goal = step.get("goal")
    if not goal:
        return None
    label = METRICS.get(goal["metric"], (goal["metric"], ""))[0]
    if abs(goal["change"]) <= goal["band"]:
        return f"{label}: no significant change."
    return f"{label}: {_moved(goal['metric'], goal['change'])}."


def _only_answers_or_tools_hold_back(step: dict[str, Any], clean: bool) -> bool:
    """Whether a step the report did not recommend has a real gain in the goal metric and no
    other limit or loss against it, so that only its answers or tool calls (``clean`` False)
    are what hold it back."""
    reason = step.get("reason") or ""
    other = any(pattern.search(reason) for pattern in (_LOSS, _FLOOR, _LIMIT))
    return step.get("verdict") == "better" and not other and not clean


def judge_step(step: dict[str, Any] | None, *, tools_on: bool) -> TryVerdict:
    """Whether to keep a step: a comparison exists, the goal metric is better beyond its noise
    band, no answer is broken, at least 95% of the answers are identical (or every difference is
    within what the earlier run's own repeated answers differ by, when that was measured) and,
    with tools, no share of tool-call behaviour fell. The record's ``changes_answers`` plays no
    part: the measurement decides."""
    if step is None or step.get("answers") is None:
        return TryVerdict(False, NO_COMPARISON, ["Not compared with an earlier run."])
    better = bool(step["recommended"])
    answers = step["answers"]
    total = answers["total"]
    share = answers["identical"] / total if total else 0.0
    within_repeats = bool(answers.get("judged")) and not answers.get("changed")
    answers_ok = answers["broken"] == 0 and (share >= 0.95 or within_repeats)
    tools_ok = True
    tool_calls = step.get("tool_calls") if tools_on else None
    tool_row: Row | None = None
    if tool_calls is not None:
        before, after = tool_calls.get("before"), tool_calls["after"]
        calls = after.get("calling") or 0
        failed = _tool_gate(before, after) if before and calls >= CALLS_TO_JUDGE else None
        tools_ok = failed is None
        key = failed[0] if failed else "valid_json"
        if calls < CALLS_TO_JUDGE:
            shown = f"only {calls} calls, too few to judge; check them yourself"
        elif failed:
            shown = f"{TOOL_SHARES[key]} {failed[1]:.0%} → {failed[2]:.0%} ({calls} calls)"
        elif after.get("requests"):
            valid = min(
                (
                    after[k]
                    for k in ("valid_json", "known_tool", "schema_valid")
                    if after.get(k) is not None
                ),
                default=None,
            )
            shown = f"{calls} calls, " + (
                "all valid" if valid is None or valid >= 1 else f"{valid:.0%} valid"
            )
        elif before and before.get(key) is not None and after.get(key) is not None:
            shown = f"{TOOL_SHARES[key]} {before[key]:.0%} → {after[key]:.0%} ({calls} calls)"
        else:
            shown = "no earlier numbers to compare"
        mark = "" if calls < CALLS_TO_JUDGE else "✔" if tools_ok else "✘"
        tool_row = ("Tool calls", "", shown, mark)
    gate = None
    if not better and not _only_answers_or_tools_hold_back(step, answers_ok and tools_ok):
        gate = "better"
    elif not answers_ok:
        gate = "answers"
    elif not tools_ok:
        gate = "tool calls"
    if answers["broken"]:
        said, mark = f"{answers['broken']} of {total} broken", "✘"
    elif not answers["differ"]:
        said, mark = f"all {total} identical", "✔"
    else:
        said = f"{answers['identical']} of {total} identical"
        mark = "✔" if answers_ok else "your call" if gate == "answers" else "✘"
    lines = [_metric_line(step)]
    if (goal_line := _goal_line(step)) is not None:
        lines.append(goal_line)
    if answers_ok and answers["differ"]:
        lines.append(
            f"{answers['differ']} answers reworded, "
            + (
                "no more than the earlier run's own repeats differ."
                if answers.get("judged")
                else "within the 5% allowed."
            )
        )
    rows: list[Row] = [("Answers", "", said, mark)]
    if tool_row is not None:
        rows.append(tool_row)
    return TryVerdict(gate is None, gate, lines, rows=tuple(rows))


_LOSS = re.compile(r"a loss: (?P<what>.+?) (?P<change>[+−-]\d+)% ±(?P<band>\d+(?:\.\d+)?)%")
_FLOOR = re.compile(
    r"(?:past|at) your floor(?: within noise)?: requests per second [\d.]+, (?P<below>\d+)% below "
    r"run \S+'s [\d.]+ \(your floor: (?P<floor>\d+)%\)"
)
_LIMIT = re.compile(
    r"past your (?P<what>.+?) limit: (?P<value>[\d,.]+) ms against (?P<limit>[\d.]+) ms"
)


def _metric_named(words: str) -> str | None:
    from .history import WORDS

    for metric, name in sorted(WORDS.items(), key=lambda item: -len(item[1])):
        if words.startswith(name):
            return "ttft_p95_ms" if metric == "ttft_p95_steady_ms" else metric
    return None


def _not_better(step: dict[str, Any]) -> str:
    """Why a step that is not better is not kept, as one sentence: what the goal metric did and,
    when another limit or metric is what failed, that."""
    metric, change, band = step["metric"], step["change"], step["band"]
    if (kept_on := step.get("goal")) and kept_on["verdict"] == "worse":
        return (
            f"it made {GOAL_SAID.get(kept_on['metric'], kept_on['metric'])} worse "
            f"({kept_on['change']:+.0%})."
        )
    name = SAID.get(metric, (metric,))[0]
    goal = _capital(
        _clause(metric, change) if abs(change) > band else f"no significant change in {name}"
    )
    reason = step.get("reason") or ""
    if found := _LOSS.search(reason):
        other = _metric_named(found["what"])
        moved = float(found["change"].replace("−", "-")) / 100
        if other is not None and other != metric:
            return f"{goal}, but {_clause(other, moved)}."
        return f"{goal}."
    if found := _FLOOR.search(reason):
        return (
            f"{goal}, but it is {found['below']}% below the speed you started from, past the "
            f"{found['floor']}% loss you allowed."
        )
    if found := _LIMIT.search(reason):
        other = _metric_named(found["what"]) or metric
        return (
            f"{goal}, but {SAID.get(other, (other,))[0]} is {found['value']} ms, over your "
            f"{found['limit']} ms limit."
        )
    text = NOTHING_SIGNIFICANT if abs(change) <= band else f"{goal}: not a clear gain"
    if "can change answers" in reason:
        text += ", and this change can alter answers, so it is not worth the risk"
    return text + "."


def _gate_text(gate: str, step: dict[str, Any] | None) -> str:
    """Why the gate failed, as one sentence; the not-kept line and the rejection reason."""
    if step is None or step.get("answers") is None:
        return "There was no earlier run to compare it with."
    if gate == "better":
        return _not_better(step)
    if gate == "answers":
        answers = step["answers"]
        if answers["broken"]:
            return f"{answers['broken']} of {answers['total']} answers came out wrong."
        return f"{answers['differ']} of {answers['total']} {REWORDED}."
    failed = _tool_gate(step["tool_calls"]["before"] or {}, step["tool_calls"]["after"])
    assert failed is not None
    return f"Tool calls got worse: {TOOL_SHARES[failed[0]]} {failed[1]:.0%} → {failed[2]:.0%}."


def _unmatched(run_dir: Path) -> str | None:
    """Why the run in ``run_dir`` was not compared with the one it was made from, in words;
    None when it was, or when its window.json does not say."""
    try:
        matched = json.loads((run_dir / "window.json").read_text("utf-8")).get("matched") or {}
    except (OSError, ValueError):
        return None
    parent = matched.get("parent")
    if parent is None:
        return None
    if (removed := matched.get("removed")) is not None:
        return (
            f"It answered {removed:.0%} more of the prompts than run {parent}, which failed "
            "them, so the two runs did different work and their speed is not compared."
        )
    if why := matched.get("why"):
        return f"It was not compared with run {parent}: {why}."
    return None


def _step(run_dir: Path) -> dict[str, Any] | None:
    try:
        report = json.loads((run_dir / "report.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None
    for section in report["sections"]:
        for block in section["blocks"]:
            if block.get("kind") == "data" and block.get("key") == "step":
                return block  # type: ignore[no-any-return]
    return None


def _first_verdict(run_dir: Path) -> tuple[TryVerdict, str]:
    """The first run of a setup that could not start before, with the reason when it is not kept:
    it is kept when it served requests, its answers hold up on their own (none empty or
    repeating) and, with tools, at least 95% of the tool calls are valid JSON of a known tool.
    There is no earlier run to compare speed or answers with."""
    from . import compare
    from .runs import RunMetrics, answers_in

    metrics = RunMetrics.read(run_dir / "metrics.json")
    lines = []
    if metrics.succeeded:
        lines.append("Your model did not run before; now it runs. Answers checked on their own.")
    answers = [answer for given in answers_in(run_dir).values() for answer in given]
    total = len(answers)
    empty = sum(
        not answer.text and not answer.tool_calls and not answer.thinking for answer in answers
    )
    repeating = sum(compare.repeats_itself(answer.text) for answer in answers)
    answers_ok = total > 0 and not any(compare.broken(answer) for answer in answers)
    checked: dict[str, Any] = {
        "answers": {
            "checked": "on their own",
            "total": total,
            "non_empty": total - empty,
            "repeating": repeating,
            "ok": answers_ok,
        },
        "tool_calls": metrics.tool_calls,
    }
    rows: list[Row] = [
        (
            "Answers",
            "",
            f"{total - empty} of {total} non-empty, "
            + ("none repeating" if not repeating else f"{repeating} repeating"),
            "✔" if answers_ok else "✘",
        )
    ]
    tools_ok = True
    if metrics.tool_calls is not None:
        calls = metrics.tool_calls.get("calling")
        calls = calls if isinstance(calls, int) else 0
        shares = []
        for key in ("valid_json", "known_tool"):
            share = metrics.tool_calls.get(key)
            if isinstance(share, (int, float)):
                shares.append(f"{TOOL_SHARES[key]} {share:.0%}")
                if share < TOOL_FLOOR:
                    tools_ok = False
        shown = f"{', '.join(shares)} of {calls} calls"
        if calls < CALLS_TO_JUDGE:
            rows.append(("Tool calls", "", f"{shown}; too few to judge, check them yourself", ""))
        else:
            rows.append(("Tool calls", "", shown, "✔" if tools_ok else "✘ (95% needed)"))
    made = tuple(rows)
    if metrics.succeeded == 0:
        verdict = TryVerdict(False, "served", lines, checked, made)
        return verdict, "The server answered no request."
    if not answers_ok:
        return TryVerdict(False, "answers", lines, checked, made), (
            f"Its answers look broken: {empty} empty and {repeating} repeating of {total}."
            if total
            else "No answers were recorded."
        )
    if not tools_ok:
        return TryVerdict(
            False, "tool calls", lines, checked, made
        ), "Fewer than 95% of its tool calls are valid JSON of a known tool."
    return TryVerdict(True, None, lines, checked, made), ""


def _words(args: tuple[str, ...], env: tuple[str, ...], current: frozenset[str]) -> str:
    """What an offer changes, as flags: ``--max-num-seqs 48``."""
    shown = []
    for arg in (*args, *env):
        if arg in current:
            continue
        name, given, value = arg.partition("=")
        shown.append(arg if arg in env else f"--{name} {value}" if given else f"--{name}")
    return " ".join(shown) or "the kept setup again"


def _held(project: Path, kept: Kept | None) -> tuple[frozenset[str], frozenset[str]]:
    """The arguments the kept setup carries, and with them those the baseline's launch pinned:
    neither is part of a change tried on top of the current setup."""
    resolved = load_project(project, verify_weights=False)
    pins = pinned_baseline(project, resolved, project_engine(resolved.record.current_setup))
    current = frozenset((*kept.engine_args, *kept.env)) if kept is not None else frozenset()
    return current, pins


UNTIMED = "time not known yet: no request finished"


def untimed(project: Path) -> bool:
    """Whether the newest run sent requests and none of them finished, so no time to test a
    change can be taken from it."""
    from . import window

    run = latest_run(project)
    rows = window.read_rows(project / "runs" / run) if run else None
    return rows is not None and bool(rows) and not any(row.ok for row in rows)


def expected_minutes(
    project: Path, offer: Offer, goal: str, weight_bytes: int, fallback: int
) -> int:
    """Minutes a try takes, from the pace of the newest run: its start, the warm-up, the
    requests the window will send at that pace; ``fallback`` when no run gives a pace, as in
    one where every request failed."""
    from . import window
    from .history import planned_window
    from .project import declared_workload
    from .wizard import load_estimate_s

    run = latest_run(project)
    workload = declared_workload(project)
    rows = window.read_rows(project / "runs" / run) if run else None
    if run is None or workload is None or not rows or not any(r.ok for r in rows):
        return fallback
    try:
        ready = float(
            json.loads((project / "runs" / run / "window.json").read_text("utf-8"))["startup"][
                "ready_s"
            ]
        )
    except (OSError, ValueError, KeyError, TypeError):
        ready = load_estimate_s(weight_bytes)
    try:
        plan = planned_window(
            project / "runs" / run, workload, goal=goal, quality_risk=offer.changes_answers
        )
    except (OSError, ValueError, KeyError, TypeError):
        plan = None
    began = min(r.dispatch_ns for r in rows)
    ended = max(r.terminal_ns or r.dispatch_ns for r in rows)
    per_request = (ended - began) / 1e9 / len(rows)
    requests = plan.requests if plan is not None else len(rows)
    return max(1, math.ceil((ready + (requests + WARMUP_REQUESTS) * per_request) / 60))


def _analysis_arguments(arguments: argparse.Namespace, offer: Offer) -> argparse.Namespace:
    from .cli import build_parser

    analysis = build_parser().parse_args(["analyse"])
    for key, value in vars(arguments).items():
        if key not in ("run", "command", "offer", "json") and hasattr(analysis, key):
            setattr(analysis, key, value)
    analysis.engine_arg = list(offer.engine_args)
    analysis.engine_env = [
        (name, value) for name, _, value in (var.partition("=") for var in offer.engine_env)
    ]
    analysis.yes = True
    analysis.try_step = True
    return analysis


def run_try(arguments: argparse.Namespace) -> int:
    from .cli import _confirmed, _run_analyse
    from .report.build import baseline_words
    from .wizard import analysis_minutes

    project = arguments.project.expanduser()
    as_json = arguments.json or JSON_OUTPUT.get()
    if arguments.keep:
        code = _keep_reworded(project, as_json)
        if code is not None:
            return code
    engine = engine_for(arguments)
    runtime = runtime_for(arguments)  # reads the llama.cpp build the offers are worded for
    try:
        found = offers(project)
    except NothingToTry as error:
        if str(error) == NO_CHANGE:
            return _no_change(project, as_json)
        print(error, file=sys.stderr)
        return 1
    if not 1 <= arguments.offer <= len(found):
        count = "1 offer" if len(found) == 1 else f"{len(found)} offers"
        print(f"there is no offer {arguments.offer}: the last report has {count}", file=sys.stderr)
        return 1
    offer = found[arguments.offer - 1]
    kept = read_kept(project)
    current, pins = _held(project, kept)
    held = current | pins
    weight_bytes = load_project(project, verify_weights=False).artifact.metadata.weight_bytes
    analysis = _analysis_arguments(arguments, offer)
    if offer.test_seconds:  # the estimate the report summary and the Next menu name
        minutes = math.ceil(offer.test_seconds / 60)
    else:
        minutes = expected_minutes(
            project,
            offer,
            analysis.goal,
            weight_bytes,
            analysis_minutes(project, weight_bytes, engine, runtime, offer.engine_args),
        )
    tried = _words(offer.engine_args, offer.engine_env, held)
    shown = None if untimed(project) else minutes
    if not as_json:
        print(f"Trying {tried}{_on_top(offer, held)}", file=sys.stderr)
        print(f"Why: {offer.why.split('. Measured together')[0]}", file=sys.stderr)
        print(
            f"How long it takes: {UNTIMED}." if shown is None else f"This takes about {shown} min.",
            file=sys.stderr,
        )
    if not arguments.allow_busy:
        resolved = load_project(project, verify_weights=False)
        settings = settings_for(resolved, engine, offer.engine_args)
        refuse_if_busy(
            runtime.platform,
            runtime.gpus,
            needs_for(engine, settings, weight_bytes, runtime.platform, runtime.gpus),
            minutes,
        )
    if not _confirmed(arguments, engine, runtime, shown):
        print("Nothing started.", file=sys.stderr)
        return 1
    analysis.unstarted_parent = offer.first or _current_unstarted(project)
    status = _run_analyse(analysis, print_report=False)
    run_id = getattr(analysis, "finished_run", "")
    failure = getattr(analysis, "tool_failure", None)
    if failure is not None:
        step = None
        verdict = TryVerdict(False, "tool calls", [], rows=(("Tool calls", "", failure, "✘"),))
        text = f"Tool calls failed: {failure}."
    elif (started := getattr(analysis, "start_failure", None)) is not None:
        return _start_failed(project, as_json, offer, current, tried, *started)
    elif status != 0 and run_id:
        step = None
        text = "No request succeeded."
        verdict = TryVerdict(False, "requests", [], rows=(("Requests", "", "none succeeded", "✘"),))
    elif status != 0 or not run_id:
        print("tensward try failed. Nothing changed in this project.", file=sys.stderr)
        return 1
    else:
        step = _step(project / "runs" / run_id)
        if (slow := _too_slow(project / "runs" / run_id)) is not None:
            verdict, text = _first_verdict(project / "runs" / run_id)
            if verdict.keep:
                return _slow_start(project, as_json, offer, held, tried, run_id, slow, verdict)
        elif offer.first and step is None:
            verdict, text = _first_verdict(project / "runs" / run_id)
        else:
            tools_on = step is not None and step.get("tool_calls") is not None
            verdict = judge_step(step, tools_on=tools_on)
            unmatched = verdict.gate == NO_COMPARISON and _unmatched(project / "runs" / run_id)
            text = unmatched or (_gate_text(verdict.gate, step) if verdict.gate else "")
    now: Kept | None = None
    if verdict.keep:
        try:
            now, _ = keep(project, run_id, accept_unreviewed=True)
        except NotKept as error:
            verdict = TryVerdict(False, "keep", verdict.lines, verdict.checked, verdict.rows)
            text = f"It could not be kept: {error}."
    if not verdict.keep:
        try:
            reject(project, run_id, reason=text)
        except NotRejected as error:
            print(f"not rejected: {error}", file=sys.stderr)
    try:
        following = offers(project)[0]
    except NothingToTry:
        following = None
    answers = step.get("answers") if step else None
    reworded = verdict.gate == "answers" and answers is not None and not answers["broken"]
    after_keep = _after_keep(project, run_id, pins) if reworded else None
    if as_json:
        choices = _choices(
            following, now, held, kept=verdict.keep, reworded=reworded, after_keep=after_keep
        )
        reason = None if verdict.keep else text or None
        print(_json_result(run_id, offer, verdict, step, following, choices, reason))
        return 0 if verdict.keep else REVERTED
    lines = [f"Tried: {tried}{_on_top(offer, held)}", ""]
    lines += _result_lines(project / "runs", run_id, step, verdict)
    if verdict.keep:
        words = f" ({baseline_words(step['baseline'])})" if step and step.get("baseline") else ""
        lines.append(f"Kept in this project{words}. Your running server is unchanged.")
    elif reworded:
        report = project / "runs" / run_id / "report.md"
        lines.append(f"Not kept: {text} Nothing changes unless you keep it.")
        lines.append(f'Compare them: {report} (section "Answers before / after")')
    elif step is not None and step.get("goal") and verdict.gate == "better":
        lines.append(f"Not kept: {text[:1].lower()}{text[1:]} Nothing changed.")
    else:
        lines.append(f"Not kept. {text} Nothing changed.")
    lines.append("")
    ended = None
    if following is None and after_keep is None:
        ended = wrap_up(project, now if verdict.keep else kept, pending=reworded)
    lines += _menu(
        following,
        now,
        held,
        kept=verdict.keep,
        reworded=reworded,
        wrap_up=ended,
        after_keep=after_keep,
        beyond=_beyond(project / "runs" / run_id) if ended is not None else (),
        summary=_beyond_line(project / "runs" / run_id),
    )
    print("\n".join(lines))
    return 0 if verdict.keep else REVERTED


def _slow_start(
    project: Path,
    as_json: bool,
    offer: Offer,
    held: frozenset[str],
    tried: str,
    run_id: str,
    slow: dict[str, Any],
    verdict: TryVerdict,
) -> int:
    """A change that starts where the current setup did not, with sound answers, but too slow
    to measure fully here: not kept and not rejected; ``try --keep`` keeps it."""
    from .runs import RunMetrics

    metrics = RunMetrics.read(project / "runs" / run_id / "metrics.json")
    speed = (
        f", {number(metrics.output_throughput)} tok/s"
        if metrics.output_throughput is not None
        else ""
    )
    said = (
        "It starts now (your current setup does not), but it is slow on this machine: "
        f"{slow['done']} request{'' if slow['done'] == 1 else 's'} in {slow['seconds']} s{speed}."
    )
    following = next(
        (
            found
            for found in _offers_or_none(project)
            if (found.engine_args, found.engine_env) != (offer.engine_args, offer.engine_env)
        ),
        None,
    )
    # this change is not rejected, so a plain `tensward try` would offer it again
    named = f"tensward try {following.n}" if following else ""
    choices = [
        (named if command == "tensward try" else command, text)
        for command, text in _choices(following, None, held, kept=False, slow=True)
    ]
    if as_json:
        slow_verdict = TryVerdict(False, "slow", [], verdict.checked)
        document = json.loads(_json_result(run_id, offer, slow_verdict, None, following, choices))
        print(json.dumps({**document, "starts": said}, sort_keys=True))
        return REVERTED
    lines = [f"Tried: {tried}{_on_top(offer, held)}", said, ""]
    lines += _result_lines(project / "runs", run_id, None, replace(verdict, lines=[]))
    lines += [
        "Not kept yet; nothing changed. Keep it to serve it: tensward try --keep",
        "",
        *next_lines(choices, more=following is not None),
    ]
    print("\n".join(lines))
    return REVERTED


def _offers_or_none(project: Path) -> list[Offer]:
    try:
        return offers(project)
    except NothingToTry:
        return []


def _recorded(run_dir: Path) -> dict[str, Any] | None:
    try:
        found = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) else None


def _significant(run_dir: Path) -> dict[str, bool]:
    """Per metrics key, whether the run's figure moved beyond its noise against its parent."""
    try:
        deltas = json.loads((run_dir / "window.json").read_text("utf-8")).get("deltas") or []
    except (OSError, ValueError):
        return {}
    keys = {metric: key for metric, (_, key) in METRICS.items()}
    return {
        keys[d["metric"]]: d.get("verdict") != "within_noise"
        for d in deltas
        if isinstance(d, dict) and d.get("metric") in keys
    }


def _result_lines(
    runs: Path, run_id: str, step: dict[str, Any] | None, verdict: TryVerdict
) -> list[str]:
    """The table of the run before and after the change, then what the goal metric did and what
    limits the run."""
    after = _recorded(runs / run_id)
    try:
        parent = json.loads((runs / run_id / "window.json").read_text("utf-8")).get("parent")
    except (OSError, ValueError):
        parent = None
    before = _recorded(runs / parent) if parent else None
    significant = _significant(runs / run_id)
    goal = None
    if step is not None and step.get("metric") in METRICS and step.get("before") is not None:
        key = METRICS[step["metric"]][1]
        goal = (key, step["before"], step["after"], step["change"])
        significant[key] = abs(step["change"]) > step["band"]
    rows = stands_rows(before, after, goal, significant=significant) if after is not None else []
    if before is None:
        header: tuple[str, ...] = ("", "now", "")
        table: list[tuple[str, ...]] = [(label, now, "") for label, now, _, _ in rows]
        table += [(label, now, mark) for label, _, now, mark in verdict.rows]
    else:
        header = ("", "before", "after", "change")
        table = [*rows, *verdict.rows]
    lines = [*columns(header, table), ""] if table else []
    lines += verdict.lines
    if after is not None:
        lines.append(limit_line(before, after))
    return lines


def _on_top(offer: Offer, held: frozenset[str]) -> str:
    """ "(on top of your current setup: --n-gpu-layers 41)" when the offer carries flags of the
    setup it was made on, which the change tried leaves out."""
    carried = [arg for arg in (*offer.engine_args, *offer.engine_env) if arg in held]
    if (
        not carried
        or not _words(offer.engine_args, offer.engine_env, held) != "the kept setup again"
    ):
        return ""
    return f" (on top of your current setup: {flag_words(carried)})"


def _no_change(project: Path, as_json: bool) -> int:
    """``try`` after a report that offers no change: nothing failed, so say so and show the menu."""
    from .analyse_next import SERVE_TEXT as SERVE_AS_IS

    serving = SERVE_TEXT if read_kept(project) is not None else SERVE_AS_IS
    run = latest_run(project)
    choices = [*(_beyond(project / "runs" / run) if run else []), ("tensward serve", serving)]
    if as_json:
        print(json.dumps({"offer": None, "reason": NOTHING_WORTH, "next": next_document(choices)}))
    else:
        summary = _beyond_line(project / "runs" / run) if run else None
        lines = next_lines(choices, more=False, wrap_up=NOTHING_WORTH)
        print("\n".join(_with_summary(lines, summary)))
    return 0


def _keep_reworded(project: Path, as_json: bool) -> int | None:
    """``tensward try --keep``: keep what the last try refused for reworded answers. None when
    there is a next change to try, which the caller then runs; else the menu and an exit code."""
    try:
        words, why = keep_reworded(project)
    except NotKept as error:
        print(error, file=sys.stderr)
        return 1
    said = f"Kept {words} ({why})."
    try:
        following = offers(project)[0]
    except NothingToTry:
        following = None
    if following is not None and why != SLOW_KEPT:
        print(said, file=sys.stderr if as_json else sys.stdout)
        return None
    now = read_kept(project)
    current, pins = _held(project, now)
    choices = _choices(following if why == SLOW_KEPT else None, now, current | pins, kept=True)
    if as_json:
        print(json.dumps({"kept": True, "change": words, "next": next_document(choices)}))
    else:
        more = following is not None
        print("\n".join([said, "", *next_lines(choices, more=more)]))
    return 0


def _choices(
    following: Offer | None,
    now: Kept | None,
    current: frozenset[str],
    *,
    kept: bool,
    reworded: bool = False,
    slow: bool = False,
    after_keep: str | None = None,
) -> list[tuple[str, str]]:
    after = frozenset((*now.engine_args, *now.env)) | current if now else current
    flags = _words(following.engine_args, following.engine_env, after) if following else None
    return next_choices(flags, kept=kept, reworded=reworded, slow=slow, after_keep=after_keep)


def _after_keep(project: Path, run_id: str, pins: frozenset[str]) -> str | None:
    """The flags ``try`` runs next once the reworded run ``run_id`` is kept, None for none."""
    offer, carried = offers_if_kept(project, run_id)
    return _words(offer.engine_args, offer.engine_env, carried | pins) if offer else None


def _menu(
    following: Offer | None,
    now: Kept | None,
    current: frozenset[str],
    *,
    kept: bool,
    reworded: bool = False,
    wrap_up: str | None = None,
    after_keep: str | None = None,
    beyond: Sequence[tuple[str, str]] = (),
    summary: str | None = None,
) -> list[str]:
    """The ``Next:`` menu after a try; ``beyond``, the Beyond flags choices, lead it, and
    ``summary``, the Beyond flags line, stands just above it."""
    choices = _choices(following, now, current, kept=kept, reworded=reworded, after_keep=after_keep)
    more = following is not None or after_keep is not None
    return _with_summary(next_lines([*beyond, *choices], more=more, wrap_up=wrap_up), summary)


def _with_summary(lines: list[str], summary: str | None) -> list[str]:
    """``lines``, a menu, with ``summary`` just above its ``Next:``."""
    if summary is None:
        return lines
    at = lines.index("Next:")
    return [*lines[:at], summary, *lines[at:]]


def _beyond_line(run_dir: Path) -> str | None:
    """The one Beyond flags line of the run in ``run_dir``; None when its report has none."""
    try:
        return beyond_summary(json.loads((run_dir / "report.json").read_text("utf-8")))
    except (OSError, ValueError):
        return None


def _beyond(run_dir: Path) -> list[tuple[str, str]]:
    """The Beyond flags choices of the run in ``run_dir``, pointing to its report.md for the
    steps; none when its report has none."""
    try:
        found = beyond_choices(json.loads((run_dir / "report.json").read_text("utf-8")))
    except (OSError, ValueError):
        return []
    return [(command, text.replace(ABOVE, IN_REPORT)) for command, text in found]


def wrap_up(project: Path, kept: Kept | None, *, pending: bool = False) -> str | None:
    """The last line of a session with no more changes to try, above its one Next menu: the
    kept setup's speed, then, when the last change is better but rewords answers, the decision it
    waits for; None when there is neither."""
    from .runs import RunMetrics

    speed = None
    if kept is not None:
        try:
            speed = RunMetrics.read(project / "runs" / kept.run / "metrics.json").output_throughput
        except Exception:
            speed = None
    now = f"Your model now runs at {number(speed)} tok/s" if speed is not None else None
    if pending:
        decision = (
            "The last change is faster but rewords answers: keep it (tensward try --keep) or "
            "serve without it (tensward serve)."
        )
        return f"{now}. {decision}" if now else decision
    return f"{now}; nothing else helped." if now else None


def _next_unstarted(project: Path) -> int:
    taken = [
        int(entry.run.removeprefix("unstarted-"))
        for entry in read_rejected(project)
        if entry.run.removeprefix("unstarted-").isdigit()
    ]
    return max(taken, default=0) + 1


def _start_failed(
    project: Path,
    as_json: bool,
    offer: Offer,
    current: frozenset[str],
    tried: str,
    run: str,
    error: str,
    ran: bool,
) -> int:
    """A setup that did not run: say why in a line, record the change as rejected so that
    ``try`` does not offer it again, and name the next offer. ``ran``: the server was up, so the
    run did not finish. A machine problem (no C or CUDA compiler) is not the change's fault: the
    change is not rejected, and the fix comes first."""
    why = error.splitlines()[0] if error else "the server did not start"
    slow = TOO_SLOW in why
    if slow:
        why = (
            "it is much slower than your current setup: "
            + why[why.index(TOO_SLOW) :].split(": ", 1)[1]
        )
    machine = next((problem for problem in MACHINE_PROBLEMS if problem in why), None)
    if machine:
        why = why[why.index(machine) :]
    else:
        run = run or f"unstarted-{_next_unstarted(project):04d}"
        reject_failed_start(
            project, run, frozenset((*offer.engine_args, *offer.engine_env)) - current, why
        )
    try:
        following = offers(project)[0]
    except NothingToTry:
        following = None
    held = current | _held(project, None)[1]
    if as_json:
        choices = _choices(following, None, held, kept=False)
        document = json.loads(
            _json_result(run, offer, TryVerdict(False, "start", []), None, following, choices)
        )
        print(json.dumps({**document, "error": why}, sort_keys=True))
        return 1
    if machine:
        lines = [f"Not tested: {why[0].upper()}{why[1:]}", "Nothing changed."]
    else:
        lines = [
            f"Not kept. {tried}: {why}."
            if slow
            else f"Not kept. {tried} {'did not finish' if ran else 'would not start'}: {why}",
            "Nothing changed. Tensward will not offer it again here "
            f"(to allow it again: tensward reject --clear {run}).",
        ]
    lines += ["", *_menu(following, None, held, kept=False)]
    print("\n".join(lines), file=sys.stderr)
    return 1


def _json_result(
    run_id: str,
    offer: Offer,
    verdict: TryVerdict,
    step: dict[str, Any] | None,
    following: Offer | None,
    choices: list[tuple[str, str]],
    reason: str | None = None,
) -> str:
    shown = step or verdict.checked or {}
    document = {
        **({"reason": reason} if reason else {}),
        "run_id": run_id,
        "offer": offer.document(),
        "kept": verdict.keep,
        "gate": verdict.gate,
        "metric": shown.get("metric"),
        "before": shown.get("before"),
        "after": shown.get("after"),
        "change": shown.get("change"),
        "band": shown.get("band"),
        "goal": shown.get("goal"),
        "answers": shown.get("answers"),
        "tool_calls": shown.get("tool_calls"),
        "next_offer": following.document() if following else None,
        "next": next_document(choices),
    }
    return json.dumps(document, sort_keys=True)


def run_undo(arguments: argparse.Namespace) -> int:
    project = arguments.project.expanduser()
    resolved = load_project(project, verify_weights=False)
    pins = pinned_baseline(project, resolved, project_engine(resolved.record.current_setup))
    found = undo(project)
    if found is None:
        print("Nothing to undo: no change is kept in this project.", file=sys.stderr)
        return 1
    undone, restored = found
    back = set(restored.engine_args) | set(restored.env) if restored else set()
    gone = [a for a in undone.engine_args if a not in back and a not in pins]
    gone_env = [v for v in undone.env if v not in back]
    if not gone and not gone_env:
        gone = [a for a in undone.engine_args if a not in pins] or list(undone.engine_args)
        gone_env = list(undone.env)
    change = (
        "going back to your starting setup"
        if undone.run == REGISTERED
        else flag_words(gone, gone_env)
    )
    held = [a for a in restored.engine_args if a not in pins] if restored is not None else []
    previous = (
        flag_words(held, restored.env)
        if restored is not None and (held or restored.env)
        else "your starting setup"
    )
    try:
        following = offers(project)[0]
    except NothingToTry:
        following = None
    if previous == "your starting setup" and following is not None and following.first:
        previous += " (your starting setup does not fit this GPU)"
    lines = [
        f"Undone: {change}. Back to {previous}.",
        "Your running server (if any) keeps its settings until you run tensward serve again.",
        "",
    ]
    pinned = frozenset(pins)
    choices = _choices(following, restored, pinned, kept=restored is not None)
    print("\n".join([*lines, *next_lines(choices, more=following is not None)]))
    return 0


def _parent_args(run_dir: Path) -> frozenset[str] | None:
    """The arguments of the run ``run_dir`` was made from; None when it names none."""
    try:
        parent = json.loads((run_dir / "window.json").read_text("utf-8")).get("parent")
        if not parent:
            return None
        return frozenset(RunFile.read(run_dir.parent / parent).engine_args)
    except (OSError, ValueError, PreflightError):
        return None


def _answers_said(step: dict[str, Any] | None) -> str:
    answers = step.get("answers") if step else None
    if not answers:
        return "-"
    if answers["broken"]:
        return f"{answers['broken']} of {answers['total']} wrong"
    if not answers["differ"]:
        return "same"
    return f"{answers['differ']} of {answers['total']} reworded"


def run_runs(arguments: argparse.Namespace) -> int:
    from .history import recorded_env

    project = arguments.project.expanduser()
    resolved = load_project(project, verify_weights=False)
    rules = project_engine(resolved.record.current_setup).flag_rules
    pins = pinned_baseline(project, resolved, project_engine(resolved.record.current_setup))
    kept = read_kept(project)
    held = kept.run if kept is not None else None
    reverted = {r.run for r in read_rejected(project)}
    superseded = {e["run"] for e in kept.history if "run" in e} if kept is not None else set()
    if kept is not None and kept.previous != REGISTERED:
        superseded.add(kept.previous)
    runs = project / "runs"
    rows = []
    for run_dir in sorted(p for p in runs.iterdir()) if runs.is_dir() else ():
        if not (run_dir / "run.json").is_file():
            continue
        run_file = RunFile.read(run_dir)
        args = [*run_file.engine_args, *sorted(recorded_env(run_file.model_dump(), rules))]
        step = _step(run_dir)
        verdict = "-" if step is None else "better" if step["recommended"] else "not better"
        state = (
            "kept"
            if run_dir.name == held
            else "not kept"
            if run_dir.name in reverted
            else "superseded"
            if run_dir.name in superseded
            else ""
        )
        before = _parent_args(run_dir)
        shown = [a for a in args if a not in (before or ()) and a not in pins] or [
            a for a in args if a not in pins
        ]
        answers = step.get("answers") if step else None
        reworded = (
            answers is not None
            and answers["differ"] > 0
            and answers["identical"] < 0.95 * answers["total"]
        )
        rows.append(
            {
                "id": run_dir.name,
                "started_at": run_file.started_at,
                "change": " ".join(args) or "baseline",
                "verdict": verdict,
                "state": state,
                "shown": flag_words(shown) if shown else "your current setup",
                "result": _result_said(state, bool(args), reworded),
                "speed": "-"
                if step is None
                else f"{SAID.get(step['metric'], (step['metric'],))[0]} {step['change']:+.0%}",
                "answers": _answers_said(step),
            }
        )
    if arguments.json or JSON_OUTPUT.get():
        print(json.dumps(
            [{k: v for k, v in row.items() if k not in ("shown", "result", "speed", "answers")}
             for row in rows], sort_keys=True))  # fmt: skip
        return 0
    try:
        following = offers(project)[0]
    except NothingToTry:
        following = None
    after = (frozenset((*kept.engine_args, *kept.env)) if kept is not None else frozenset()) | pins
    flags = _words(following.engine_args, following.engine_env, after) if following else None
    menu = next_lines(next_choices(flags, kept=kept is not None), more=following is not None)
    print("\n".join([*_table(rows), "", *menu]))
    return 0


def _result_said(state: str, changed: bool, reworded: bool) -> str:
    if not changed:
        return "baseline"
    if state == "kept":
        return "kept, answers reworded (you accepted)" if reworded else "kept"
    if state == "superseded":
        return "kept, then replaced"
    return state or "measured only"


def _table(rows: list[dict[str, str]]) -> list[str]:
    head = {
        "id": "Run",
        "shown": "Change",
        "result": "Result",
        "speed": "Speed",
        "answers": "Answers",
    }
    table = [head, *rows]
    widths = {key: max(len(row[key]) for row in table) for key in head}
    return ["  ".join(f"{row[key]:<{widths[key]}}" for key in head).rstrip() for row in table]


def run_export(arguments: argparse.Namespace) -> int:
    from .export import render
    from .history import recorded_env
    from .serve import served_argv

    project = arguments.project.expanduser()
    resolved = load_project(project, verify_weights=False)
    engine = project_engine(resolved.record.current_setup)
    kept = read_kept(project)
    if arguments.run_id is not None:
        run = resolve_run(project, arguments.run_id)
        if not (project / "runs" / run / "run.json").is_file():
            raise PreflightError(PROJECT_INPUTS_INVALID, f"no run {run} in {project / 'runs'}")
        run_file = RunFile.read(project / "runs" / run)
        env = tuple(sorted(recorded_env(run_file.model_dump(), engine.flag_rules)))
        kept = Kept(
            run=run,
            engine_args=run_file.engine_args,
            answer_changes=(),
            previous="",
            snapshot_id=run_file.snapshot_id,
            env=env,
        )
    source = kept.run if kept is not None else latest_run(project)
    version = None
    if source is not None and (project / "runs" / source / "run.json").is_file():
        version = RunFile.read(project / "runs" / source).engine_version
    port = 8000
    argv = served_argv(project, engine, kept, port=port)
    text = render(
        arguments.format,
        argv=argv,
        engine=engine,
        version=version,
        model_path=resolved.artifact.path,
        port=port,
        run=kept.run if kept is not None else None,
        env=kept.env if kept is not None else (),
    )
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        write_private(arguments.output, text)
        print(f"wrote {arguments.output}", file=sys.stderr)
    else:
        print(text, end="")
    return 0
