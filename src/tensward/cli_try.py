"""``tensward try``, ``undo``, ``runs`` and ``export``: the commands around the kept setup."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import ollama_settings, reasoning_check
from .cli_options import JSON_OUTPUT, engine_for, runtime_for
from .compare import (
    answer_changed,
    answers_pass,
    reasoning_changed,
    reasoning_words,
    rounding,
    rounding_passes,
)
from .engines import Engine
from .engines.llamacpp import snapshots
from .errors import PROJECT_INPUTS_INVALID, TOO_SLOW, TOO_SLOW_FLOOR, PreflightError
from .files import write_private
from .gpu_busy import needs_for, refuse_if_busy
from .history import SHORTER_REPLIES, repair_check, reply_change
from .kept import (
    KEEP_LATEST,
    PINNED_RESULT,
    REGISTERED,
    Kept,
    NotKept,
    flag_words,
    keep,
    latest_run,
    partial_runs,
    pinned_baseline,
    pinned_placement,
    read_kept,
    step_block,
    tuned_words,
    undo,
)
from .presets import read_choices, thinking_off
from .project import ResolvedProject, load_project, project_engine, settings_for
from .recheck import read_judged
from .rejected import (
    PREFILL_GAIN,
    REVIEW_MARK,
    REWORDED,
    SAMPLED,
    TOOLS_CHANGED,
    UNCHECKED,
    UNCONFIRMED,
    NotRejected,
    Rejection,
    read_rejected,
    refused_by,
    reject,
    reject_failed_start,
    standing,
    your_call,
)
from .report.beyond import ABOVE, IN_REPORT, beyond_choices, beyond_summary
from .report.markdown import tried_by_number
from .report.snapshots import lines as snapshot_lines
from .runids import resolve_run
from .runs import RunFile, baseline_runs
from .runtime import MACHINE_PROBLEMS, Runtime
from .stands import ONE_RUN, ONE_RUN_MARK, Row, limit_line, number, stands_rows
from .text import columns

NO_RUN = "no analysis to try from yet: run tensward analyse first"
NO_OFFER = "no offer left: every change in the last report was tried or rejected"
NO_CHANGE = "nothing to try: the last report offers no change"
NOTHING_WORTH = "Nothing to try: the last run found no change worth testing for your goal."
NO_COMPARISON = "no comparison"
NOT_COMPARED = "answers not compared"
SLOW_KEPT = "it starts, slow on this machine; you accepted"
SHARE_DROP = 0.05
SHARE_NOISE_SE = 2.0
FAILING_CALLS_MORE = 2
# Broken output (calls that are not JSON, failed requests) blocks once it exceeds the parent's own
# rate by BROKEN_MORE or more and by more than BROKEN_SE standard errors of that rate: 0 -> 2
# blocks. The share-noise rule never judges it.
BROKEN_MORE = 2
BROKEN_SE = 2.0
BROKEN_CALLS = "valid_json"  # calls that are not JSON cannot be parsed: they block
TOOLS_HELD = "tool calls changed"
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
KEEP_FASTER = "--keep-faster"
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
    census_facts: Mapping[str, Any] = field(default_factory=dict)  # census readings it fired on
    kb_record: str | None = None  # the knowledge record behind it, as "id@rev"
    group: str | None = None  # the bottleneck it is listed under in the report ("fit", ...)

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
KEEP_COMMAND = "tensward keep"
DISCARD_COMMAND = "tensward discard"
DISCARDED = "discarded by you"
PAIRS_SHOWN = 3
PAIR_WIDTH = 100
MENU_COMMAND = 36  # widest command column; a longer command has its text on the next line
WARMUP_REQUESTS = 8  # four sent, and the first ones answer slower than the rest


def next_choices(
    following: str | None,
    *,
    kept: bool = False,
    reworded: bool = False,
    slow: bool = False,
    after_keep: str | None = None,
    unmeasured: bool = False,
) -> list[tuple[str, str]]:
    """The "Next:" menu after a ``try`` step: (command, what it does), in print order.
    ``following`` is the next offer's flags, None when there is none; ``after_keep`` the flags
    of the offer that follows when a reworded change is kept. ``slow``: the change starts where
    the current setup did not, but was too slow to measure fully. ``unmeasured``: the kept
    setup is a repair whose speed is not measured yet, which the next try measures first."""
    choices: list[tuple[str, str]] = []
    if reworded:
        choices.append(
            (
                KEEP_COMMAND,
                f"keep it (then tensward try offers {after_keep})"
                if after_keep is not None
                else "keep it; nothing else to try after it",
            )
        )
        choices.append((DISCARD_COMMAND, "drop it; it is not offered again"))
    if following is not None and unmeasured:
        choices.append(("tensward try", f"measure its speed, then try {following}"))
    elif following is not None:
        skip = "skip it and try" if reworded or slow else "try"
        choices.append(("tensward try", f"{skip} {following}"))
    elif unmeasured:
        choices.append(("tensward analyse", "measure its speed on the full window"))
    if slow:
        choices.append((KEEP_COMMAND, "keep it so you can serve it"))
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


def _tools_failed(step: dict[str, Any]) -> bool:
    """Whether the step's tool calls broke: more of them not valid JSON than its parent's rate
    allows (``broke_more``)."""
    calls = step.get("tool_calls")
    if not calls or not calls.get("before"):
        return False
    return _broken_calls(calls["before"], calls["after"]) is not None


def broke_more(was: int, of_was: int, now: int, of_now: int) -> bool:
    """Whether ``now`` broken of ``of_now`` (unparsable calls, failed requests) rose past the
    parent's ``was`` of ``of_was``: by BROKEN_MORE or more over its rate scaled to ``of_now``, and
    by more than BROKEN_SE standard errors of that rate."""
    if not of_now:
        return False
    rate = was / of_was if of_was else 0.0
    expected = rate * of_now
    spread = BROKEN_SE * of_now * math.sqrt(rate * (1 - rate) / of_was) if of_was else 0.0
    return now - expected >= BROKEN_MORE - 1e-9 and now - expected > spread


def _broken_calls(
    before: dict[str, Any], after: dict[str, Any]
) -> tuple[int, int, int, int] | None:
    """The calls that are not valid JSON before and after, each with its calls, when they rose
    past the parent's rate (``broke_more``); None when they did not or are not known. A parent
    that made no call broke none."""
    was, now = before.get(BROKEN_CALLS), after.get(BROKEN_CALLS)
    of_was, of_now = int(before.get("calling") or 0), int(after.get("calling") or 0)
    if now is None or (was is None and of_was):
        return None
    bad = round((1 - (was or 0.0)) * of_was), round((1 - now) * of_now)
    return (bad[0], of_was, bad[1], of_now) if broke_more(bad[0], of_was, bad[1], of_now) else None


def _broken_words(found: tuple[int, int, int, int]) -> str:
    """ "4 of 91 calls not valid JSON (yours: 0 of 103)"."""
    was, of_was, now, of_now = found
    yours = f"yours: {was} of {of_was}" if of_was else "yours made no calls"
    return f"{now} of {of_now} calls not valid JSON ({yours})"


def _held_all(project: Path) -> list[tuple[Rejection, str, bool]]:
    """The steps ``try`` refused only on answers the user may still accept although the goal
    metric got better beyond its band and no tool-call share fell, oldest first, each with what
    the goal metric did and whether it was made from the setup now kept (or the baseline when
    none is)."""
    kept = read_kept(project)
    runs = project / "runs"
    candidates = []
    for refusal in read_rejected(project):
        step = step_block(runs / refusal.run) if your_call(refusal.reason) else None
        first = (step or {}).get("first") or step or {}
        if step and first.get("verdict") == "better" and step.get("metric") in SAID:
            if not _tools_failed(step) and _again_refusal(runs / refusal.run) is None:
                candidates.append((refusal, step))
    baselines: set[str] = set()
    if candidates and kept is None:
        from .history import recorded_env

        resolved = load_project(project, verify_weights=False)
        rules = project_engine(resolved.record.current_setup).flag_rules
        baselines = {
            d.name
            for d in baseline_runs(project, resolved.record.snapshot_id)
            if not recorded_env(RunFile.read(d).model_dump(), rules)
        }
    held = []
    for refusal, step in candidates:
        current = (
            refusal.made_from == kept.run
            if kept is not None
            else refusal.made_from is None or refusal.made_from in baselines
        )
        again = read_judged(runs / refusal.run)
        gain = _capital(
            _clause(step["metric"], step["change"] if again is None else again["change"])
        )
        gain += f" ({PAIRED})" if _paired(step) else ""
        if again is not None:
            gain += f" {AGAINST_MEAN}"
        held.append((refusal, gain, current))
    return held


def held_steps(project: Path) -> list[tuple[Rejection, str]]:
    """The held steps that can be kept now: those made from the setup now kept."""
    return [(r, gain) for r, gain, current in _held_all(project) if current]


def _held_run(project: Path, name: str) -> str:
    """The held step ``name`` stands for: its run id, or its number or the start of it."""
    name = name.zfill(4) if name.isdigit() else name
    found = [
        (r.run, current)
        for r, _, current in _held_all(project)
        if r.run == name or r.run.startswith(f"{name}-")
    ]
    if len(found) != 1:
        raise NotKept(
            f"{name} is not one step that was held back for its answers; "
            "the steps that were are listed when a session ends"
        )
    if not found[0][1]:
        raise NotKept(
            f"{found[0][0]} was measured on an older setup than the one now kept; "
            "run tensward try to measure it again"
        )
    return found[0][0]


def keep_reworded(project: Path, step: str | None = None) -> tuple[str, str]:
    """``try --keep``: keep the latest run that ``try`` refused only for reworded answers, as
    ``tensward set RUN --accept-answer-changes`` does, or that started where the current setup
    did not but was too slow to measure fully; with ``step``, that held step instead, whichever
    run is the latest. Return its change as flags and why it is kept."""
    from .rejected import clear

    run = _held_run(project, step) if step else latest_run(project)
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
    kept = read_kept(project)
    if refusal is None and kept is not None and kept.run == run:
        if repair_check(project / "runs" / run) is not None:
            raise NotKept(
                f"{run} is kept already, as a repair: requests now succeed, its speed is not "
                "measured yet; tensward try measures it first"
            )
    if refusal is not None and (worse := _again_refusal(project / "runs" / refusal.run)):
        raise NoGain(worse)
    if (
        refusal is None
        or not your_call(refusal.reason)
        or _tools_failed(step_block(project / "runs" / refusal.run) or {})
    ):
        raise NotKept(
            "tensward keep applies only right after a try that was not kept because its answers "
            "differ (none broken), because its gain was inside the noise, or that starts but was "
            "too slow to measure here"
        )
    keep(project, refusal.run, accept_unreviewed=True)
    clear(project, refusal.run)
    words = flag_words([a for a in refusal.change if a not in pins] or refusal.change)
    refused_step = step_block(project / "runs" / refusal.run)
    checked = (refused_step or {}).get("answers") or {}
    if UNCONFIRMED in (refusal.reason or ""):
        return words, "gain not confirmed, you accepted"
    if PREFILL_GAIN in (refusal.reason or ""):
        return words, "faster prompts, you accepted"
    if TOOLS_CHANGED in (refusal.reason or ""):
        return words, "tool calls changed, you accepted"
    if checked.get("reasoning_check"):
        return words, "reasoning differs, you accepted"
    return words, (
        "answers changed, you accepted"
        if answer_changed(checked)
        else "answers reworded, you accepted"
    )


def _too_slow(run_dir: Path) -> dict[str, Any] | None:
    """The pace a run was cut short at, when its setup started where the current one did not
    but would have taken too long to measure fully; None for any other run."""
    try:
        found = json.loads((run_dir / "window.json").read_text("utf-8")).get("too_slow")
    except (OSError, ValueError, AttributeError):
        return None
    return found if isinstance(found, dict) else None


REPAIR_GROUP = "fit"  # the report's "For fit and failed requests" offers
TOO_LONG = "too long"
TOOLS_REFUSED = "tools refused"
REPAIRS = {  # the fit offers that repair failed requests, by the failure each repairs
    "fit-context": TOO_LONG,
    "llamacpp-kv-unified": TOO_LONG,
    "enable-tool-calling": TOOLS_REFUSED,
}


def repair_plan(project: Path, offer: Offer) -> dict[str, Any] | None:
    """The short window of a try that repairs failed requests, None for any other try. It is a
    repair when the offer is one of the report's fit and failed-requests changes that repairs a
    failure (``REPAIRS``), the parent shows that failure, and more than ``MAX_FAILED`` of the
    parent's requests failed, the share above which a run is no reference for a speed
    comparison. It sends the fewest requests a measurement has (``fewest_requests``), more when
    the parent's first failed prompt comes later, so that the check sends at least one prompt
    that failed before."""
    from .project import declared_workload
    from .runs import RunMetrics
    from .window import MAX_FAILED, fewest_requests, read_rows

    if offer.first or offer.group != REPAIR_GROUP or offer.name not in REPAIRS:
        return None
    parent = offers_run(project)
    if parent is None or REPAIRS[offer.name] not in _failures(project, parent):
        return None
    try:
        share = RunMetrics.read(parent / "metrics.json").failed_share
    except PreflightError:
        return None
    rows = read_rows(parent)
    if share is None or share <= MAX_FAILED:
        return None
    workload = declared_workload(project)
    if workload is None:
        return None
    arrival = workload.arrival
    requests = fewest_requests(arrival.concurrency or arrival.max_inflight or 1)
    failed = [row.index for row in rows or () if not row.ok]
    if failed:
        requests = max(requests, min(failed) + 1)
    return {
        "requests": min(requests, workload.request_count),
        "parent": parent.name,
        "parent_failed_share": round(share, 4),
    }


def _failures(project: Path, run_dir: Path) -> set[str]:
    """The failures the run shows that a repair offer can fix: prompts longer than its context
    (``TOO_LONG``), failed requests that offered tools (``TOOLS_REFUSED``)."""
    found: set[str] = set()
    try:
        metrics = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    except (OSError, ValueError):
        return found
    signals = metrics.get("engine_signals") or {}
    if metrics.get("too_long") or (signals.get("llamacpp.context_exceeded") or 0) > 0:
        found.add(TOO_LONG)
    path = run_dir / "requests.jsonl"
    failed = {
        record.get("prompt_id")
        for line in (path.read_text("utf-8").splitlines() if path.is_file() else ())
        if line.strip() and (record := json.loads(line)).get("outcome") != "success"
    }
    prompts = load_project(project, verify_weights=False).prompts if failed else ()
    if any(prompt.id in failed and prompt.tools for prompt in prompts):
        found.add(TOOLS_REFUSED)
    return found


def offers(project: Path) -> list[Offer]:
    """The changes the newest run of the current setup offers, in report order, without those
    the user rejected. A run of a setup that was reverted is not that run: its offers were
    diagnosed on settings that were thrown away."""
    project = project.expanduser()
    resolved = load_project(project, verify_weights=False)
    engine = project_engine(resolved.record.current_setup)
    current = _kept_args(project)
    run_dir = _current_run(project, resolved, engine, current)
    if run_dir is None:
        return [_first_offer(project, resolved, engine, current)]
    refusing, _ = standing(read_rejected(project), engine.flag_rules)
    return _report_offers(run_dir, refusing, current, engine, _kept_before(project))


def _kept_args(project: Path) -> frozenset[str]:
    kept = read_kept(project)
    return frozenset((*kept.engine_args, *kept.env)) if kept is not None else frozenset()


def _kept_before(project: Path) -> frozenset[frozenset[str]]:
    """The setups kept before the current one, each as its arguments and environment."""
    kept = read_kept(project)
    history = kept.history if kept is not None else ()
    return frozenset(
        frozenset((*step.get("engine_args", ()), *step.get("env", ()))) for step in history
    )


SHOWN_FILE = "shown.json"  # the run whose report `tensward report` printed last


def remember_shown(project: Path, run_dir: Path | None) -> None:
    """Note the run whose numbered offers the user last saw, when that is not the run ``try``
    numbers from; any other screen clears it."""
    path = project.expanduser() / SHOWN_FILE
    try:
        same = run_dir is None or run_dir == offers_run(project.expanduser())
    except (PreflightError, OSError, ValueError):
        same = False
    if same:
        path.unlink(missing_ok=True)
    elif run_dir is not None:
        path.write_text(json.dumps({"run": run_dir.name}), "utf-8")


def _shown_offer(project: Path, number: int, found: Sequence[Offer]) -> Offer | str | None:
    """Offer ``number`` of the report ``tensward report`` printed last, when that report is not
    the one ``try`` numbers from: the offer of the same name (and the same rank among those of
    that name) on the kept setup, else why there is none; None when no such report was shown."""
    path = project / SHOWN_FILE
    try:
        run = json.loads(path.read_text("utf-8"))["run"]
        report = json.loads((project / "runs" / run / "report.json").read_text("utf-8"))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    shown = [
        (block.get("name"), tuple(block.get("engine_args") or ()))
        for section in report.get("sections", ())
        for block in section.get("blocks", ())
        if block.get("kind") == "suggestion" and tried_by_number(block.get("command"))
    ]
    if not 1 <= number <= len(shown):
        return f"there is no offer {number}: run {run}'s report, shown last, has {len(shown)}"
    name, args = shown[number - 1]
    same = [o for o in found if o.name == name]
    exact = next((o for o in same if o.engine_args == args), None)
    if exact is not None or len(same) == 1:
        return exact or same[0]
    return (
        f"offer {number} of run {run}'s report ({name}) is not on offer for the kept setup; "
        "`tensward report` shows what is"
    )


def offers_run(project: Path) -> Path | None:
    """The run whose report ``offers`` reads, None when there is none."""
    resolved = load_project(project, verify_weights=False)
    engine = project_engine(resolved.record.current_setup)
    return _current_run(project, resolved, engine, _kept_args(project))


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
        return _report_offers(run_dir, refusing, current, engine, _kept_before(project))[0], current
    except (NothingToTry, OSError, ValueError, KeyError):
        return None, current


def _report_offers(
    run_dir: Path,
    refusing: tuple[Rejection, ...],
    current: frozenset[str],
    engine: Engine,
    before: frozenset[frozenset[str]] = frozenset(),
) -> list[Offer]:
    """The report's offers, without those the user rejected and those that would run the
    current setup again or go back to one kept before it (``before``)."""
    found: list[Offer] = []
    suggested = printed = 0
    report = json.loads((run_dir / "report.json").read_text("utf-8"))
    for section in report["sections"]:
        group = None
        for block in section["blocks"]:
            if block.get("kind") == "text" and not block.get("item"):
                key = str(block.get("key", ""))
                group = key.removeprefix("for.") if key.startswith("for.") else None
            if block.get("kind") != "suggestion" or not block.get("command"):
                continue
            suggested += 1
            printed += tried_by_number(block["command"])
            offered = frozenset((*block["engine_args"], *block["engine_env"]))
            # an offer with no arguments falls back to the kept arguments, so it runs the kept
            # setup again, as the revert to the registered setup did four times on H1
            if (offered or current) == current or offered in before:
                continue
            if refused_by(refusing, offered, current, engine.flag_rules) is not None:
                continue
            found.append(
                Offer(
                    n=printed if tried_by_number(block["command"]) else 0,
                    name=block["name"],
                    engine_args=tuple(block["engine_args"]),
                    engine_env=tuple(block["engine_env"]),
                    why=block["reason"],
                    test_seconds=block.get("test_seconds"),
                    changes_answers=bool(block.get("changes_answers")),
                    side=block.get("side"),
                    census_facts=(block.get("kb") or {}).get("census_facts") or {},
                    kb_record=(block.get("kb") or {}).get("record"),
                    group=group,
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

    run_dir = offers_run(project)
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


def _share_counts(key: str, before: dict[str, Any], after: dict[str, Any]) -> tuple[int, int]:
    field = "requests" if key == "produced_call" else "calling"
    return int(before.get(field) or 0), int(after.get(field) or 0)


def _share_noise(key: str, before: dict[str, Any], after: dict[str, Any]) -> float:
    """How far two runs of one setup can differ on this share by chance: SHARE_NOISE_SE standard
    errors of the difference of the two shares, each over its own count."""
    was, now = before[key], after[key]
    counts = _share_counts(key, before, after)
    if not all(counts):
        return 1.0
    return SHARE_NOISE_SE * math.sqrt(was * (1 - was) / counts[0] + now * (1 - now) / counts[1])


def _tool_gate(
    before: dict[str, Any], after: dict[str, Any]
) -> tuple[str, float, float, float] | None:
    """Calls that are not valid JSON past the parent's rate (``_broken_calls``: they block,
    whatever their share), else the first other share of tool-call behaviour that fell beyond
    what two runs of one setup differ by and by at least SHARE_DROP while at least
    FAILING_CALLS_MORE more calls failed, as its key, the two shares and that noise; a smaller
    move is information, shown and not judged."""
    if _broken_calls(before, after) is not None:
        return BROKEN_CALLS, before[BROKEN_CALLS], after[BROKEN_CALLS], 0.0
    for key in (k for k in TOOL_SHARES if k != BROKEN_CALLS):
        was, now = before.get(key), after.get(key)
        if was is None or now is None:
            continue
        counts = _share_counts(key, before, after)
        failing = [round((1 - share) * n) for share, n in zip((was, now), counts)]
        noise = _share_noise(key, before, after)
        if (
            was - now >= SHARE_DROP - 1e-9
            and was - now > noise
            and failing[1] - failing[0] >= FAILING_CALLS_MORE
        ):
            return key, was, now, noise
    return None


def _share_moves(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """The shares of tool-call behaviour that moved by a point or more, in words, each said to be
    inside the noise only when it is; calls not valid JSON are counted, not judged by the noise."""
    moves = []
    for key in TOOL_SHARES:
        was, now = before.get(key), after.get(key)
        if was is None or now is None or f"{was:.0%}" == f"{now:.0%}":
            continue
        moved = f"{TOOL_SHARES[key]} {was:.0%} → {now:.0%}"
        if key == BROKEN_CALLS:
            counts = int(before.get("calling") or 0), int(after.get("calling") or 0)
            bad = round((1 - was) * counts[0]), round((1 - now) * counts[1])
            moved = (
                f"{_broken_words((bad[0], counts[0], bad[1], counts[1]))}, not past your "
                "current setup's rate"
            )
        elif abs(was - now) <= _share_noise(key, before, after):
            moved += ", inside the noise"
        elif now > was:
            moved += ", past the noise, up"
        elif was - now < SHARE_DROP - 1e-9:
            moved += f", past the noise but under the {SHARE_DROP:.0%} points a fall is judged at"
        else:
            moved += f", past the noise with fewer than {FAILING_CALLS_MORE} more calls failing"
        moves.append(moved)
    return moves


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
UNPAIRED = "the two runs' prompts could not be paired, and unpaired runs can differ by up to"


def _moved(metric: str, change: float, note: str = "") -> str:
    """A move beyond the noise: "+33%: a real gain", "+6% slower: a real change"; ``note``
    follows the figure."""
    more_is_better = SAID.get(metric, ("", True))[1]
    if (change > 0) == more_is_better:
        return f"{change:+.0%}{note}: a real gain"
    return f"{change:+.0%}{note} {'lower' if more_is_better else 'slower'}: a real change"


def _same_cache(step: dict[str, Any]) -> str:
    """After a paired figure that kept the prompts a cache change moved, the figure without them
    where it differs: " (prompts the cache changed included); same prompts +15%"."""
    same = step.get("same_cache")
    if not same or step.get("again") or f"{same['change']:+.0%}" == f"{step['change']:+.0%}":
        return ""
    return f" (prompts the cache changed included); same prompts {same['change']:+.0%}"


def _metric_line(step: dict[str, Any]) -> str:
    """The goal metric's move when it is beyond the measurement noise; a move inside it that the
    report still recommends held your floor and limits."""
    label = METRICS.get(step["metric"], (step["metric"], ""))[0]
    if SHORTER_REPLIES in (step.get("reason") or ""):
        return f"{label}: {_shorter(step)}."
    if abs(step["change"]) > step["band"]:
        note = _same_cache(step) + _per_token(step)
        return f"{label}: {_paired_word(step)}{_moved(step['metric'], step['change'], note)}."
    if step["recommended"]:
        return f"{label}: no significant change; it held your floor and limits."
    if prefill_gain(step):
        return f"{label}: {_inside(step)}."
    if not _paired(step):
        return f"{label}: {step['change']:+.0%}, but {UNPAIRED} {step['band']:.0%}."
    return f"{NOTHING_SIGNIFICANT}."


def _per_token(step: dict[str, Any]) -> str:
    """ "; per output token +18%": the paired output rate beside a paired requests per second."""
    tokens = step.get("per_token")
    return f"; per output token {tokens['change']:+.0%}" if tokens else ""


def _shorter(step: dict[str, Any]) -> str:
    """A requests-per-second gain that per output token is none: both figures, and why."""
    tokens = step["per_token"]
    shorter = reply_change(step["change"], tokens["change"])
    where = (
        f"inside its {tokens['band']:.0%} band"
        if tokens["verdict"] == "within_noise"
        else f"past its {tokens['band']:.0%} band"
        if tokens["verdict"] == "better"
        else f"worse past its {tokens['band']:.0%} band"
    )
    return (
        f"{change_words(step)}, but replies {shorter:.0%} shorter and per output token "
        f"{tokens['change']:+.0%} ({where}): the gain {SHORTER_REPLIES}, not a faster server"
    )


def _inside(step: dict[str, Any]) -> str:
    """ "paired +2%, inside its 17% band": a goal move within the noise, with its figure."""
    return f"{change_words(step)}, inside its {step['band']:.0%} band"


def _paired(step: dict[str, Any]) -> bool:
    """Whether the step's band comes from its prompts paired with the earlier run's."""
    return "(not matched:" not in (step.get("reason") or "")


PAIRED = "paired"
NOTE_WIDTH = 80  # a terminal's width: the note under the table wraps there


def _paired_word(step: dict[str, Any]) -> str:
    """ "paired " before a goal change made request by request over the prompts both runs sent,
    which the whole runs' figures beside it need not give."""
    return f"{PAIRED} " if _paired(step) else ""


def change_words(step: dict[str, Any]) -> str:
    """The goal metric's change as the verdict judged it: "paired +21%", "+6%"."""
    return f"{_paired_word(step)}{step['change']:+.0%}"


def _paired_note(run_dir: Path, step: dict[str, Any]) -> str | None:
    """Under the table, once: what "paired" compared."""
    if not _paired(step):
        return None
    try:
        matched = json.loads((run_dir / "window.json").read_text("utf-8")).get("matched") or {}
    except (OSError, ValueError):
        matched = {}
    upto, left = matched.get("upto"), matched.get("uncached") or 0
    prompts = f"the {upto} prompts both runs sent" if upto else "the prompts both runs sent"
    if left:
        prompts += f", {left} left out (only one run had them in its prompt cache)"
    rescaled = (
        ", then set against the mean of your current setup's two runs" if step.get("again") else ""
    )
    return (
        f"{PAIRED}: {METRICS[step['metric']][0]} compared request by request over {prompts}"
        f"{rescaled}; the figures beside it are the whole runs'."
    )


def unconfirmed_gain(step: dict[str, Any]) -> bool:
    """Whether a step held on speed alone moved its goal metric the right way inside a band its
    prompts could not be paired to narrow, with no other limit, loss or answer risk against it and
    its answers and tool calls passing: the user may keep it with ``try --keep``."""
    metric, change = step["metric"], step["change"]
    reason = step.get("reason") or ""
    goal = step.get("goal")
    return (
        not step["recommended"]
        and not _paired(step)
        and "can change answers" not in reason
        and 0 < abs(change) <= step["band"]
        and (change > 0) == SAID.get(metric, ("", True))[1]
        and step.get("loss") is None
        and not (goal and goal["verdict"] == "worse")
        and not any(p.search(reason) for p in (_FLOOR, _LIMIT))
        and step.get("answers") is not None
        and answers_pass(step["answers"])
        and not _tools_failed(step)
    )


def prefill_gain(step: dict[str, Any]) -> bool:
    """Whether a step not kept on its goal metric, which stayed inside its band, made the prompts
    faster past their own band without making prompt and decode time together longer, with no
    other limit or loss against it, no broken answer and its tool calls passing: the user may
    keep it for the prompt time, at the decode cost and answer changes shown."""
    phases = step.get("phases") or {}
    prompt, decode = phases.get("prompt"), phases.get("decode")
    reason, goal, answers = step.get("reason") or "", step.get("goal"), step.get("answers")
    return (
        not step["recommended"]
        and prompt is not None
        and decode is not None
        and math.log1p(prompt["change"]) < -prompt["band"]
        and prompt["after_s"] + decode["after_s"] <= prompt["before_s"] + decode["before_s"]
        and abs(step["change"]) <= step["band"]
        and step.get("loss") is None
        and not (goal and goal["verdict"] == "worse")
        and not any(p.search(reason) for p in (_FLOOR, _LIMIT))
        and answers is not None
        and bool(answers["total"])
        and not answers["broken"]
        and not _tools_failed(step)
    )


def _prefill_words(step: dict[str, Any]) -> str:
    """The prompt-time gain and the decode cost of a ``prefill_gain`` step."""
    phases = step["phases"]
    prompt, decode = phases["prompt"], phases.get("decode")
    rate = (
        f", {prompt['before_tok_s']:,.0f} to {prompt['after_tok_s']:,.0f} prompt tok/s"
        if "before_tok_s" in prompt
        else ""
    )
    cost = ""
    if decode:
        moved = "inside" if abs(decode["change"]) <= decode["band"] else "past"
        cost = f"; decode time {decode['change']:+.0%} ({moved} its {decode['band']:.0%} band)"
    answers = step["answers"]
    differ = (
        f"; {answers['differ']} of {answers['total']} answers differ, none broken"
        if not answers_pass(answers)
        else ""
    )
    return (
        f"prompts x{prompt['before_s'] / prompt['after_s']:.2f} faster (paired, "
        f"{prompt['before_s']:,.0f} to {prompt['after_s']:,.0f} s on the same prompts{rate}, past "
        f"its {prompt['band']:.0%} band; your current setup measured once){cost}{differ}: "
        f"{PREFILL_GAIN}"
    )


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
    other = (
        step.get("loss") is not None
        or any(p.search(reason) for p in (_FLOOR, _LIMIT))
        or SHORTER_REPLIES in reason
    )
    return step.get("verdict") == "better" and not other and not clean


def judge_step(step: dict[str, Any] | None, *, tools_on: bool) -> TryVerdict:
    """Whether to keep a step: a comparison exists, the goal metric is better beyond its noise
    band, no answer is broken, with tools no share of tool-call behaviour fell, and every answer
    is identical (or every difference is within what the earlier run's own repeated answers
    differ by, when that was measured). A step that fails only on answers that differ, none
    broken, is held for the user to keep or discard (gate ``answers``). A prompt compared on its
    reasoning passes only when identical, and a prompt that only thought holds the step back;
    the finished prompts are judged on their own. A step none of whose answers could be
    compared is not kept, as one with no earlier run is not. The record's ``changes_answers``
    plays no part: the measurement decides."""
    if step is None or step.get("answers") is None:
        return TryVerdict(False, NO_COMPARISON, ["Not compared with an earlier run."])
    better = bool(step["recommended"])
    answers = step["answers"]
    total = answers["total"]
    answers_ok = answers_pass(answers)
    tools_ok, tools_broken = True, False
    tool_calls = step.get("tool_calls") if tools_on else None
    tool_row: Row | None = None
    if tool_calls is not None:
        before, after = tool_calls.get("before"), tool_calls["after"]
        calls = after.get("calling") or 0
        broke = _broken_calls(before, after) if before else None
        failed = (
            _tool_gate(before, after) if broke or (before and calls >= CALLS_TO_JUDGE) else None
        )
        tools_ok = failed is None
        tools_broken = broke is not None
        key = failed[0] if failed else "valid_json"
        moves = _share_moves(before, after) if before else []
        if broke is not None:
            shown = f"{_broken_words(broke)}: past your current setup's rate"
        elif calls < CALLS_TO_JUDGE:
            shown = f"only {calls} calls, too few to judge; check them yourself"
        elif failed:
            shown = (
                f"{TOOL_SHARES[key]} {failed[1]:.0%} → {failed[2]:.0%} ({calls} calls), "
                f"past the ±{failed[3]:.0%} noise"
            )
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
            if moves:
                shown += f"; {'; '.join(moves)}"
        elif before and before.get(key) is not None and after.get(key) is not None:
            shown = f"{TOOL_SHARES[key]} {before[key]:.0%} → {after[key]:.0%} ({calls} calls)"
        else:
            shown = "no earlier numbers to compare"
        mark = (
            "✘"
            if tools_broken
            else ""
            if calls < CALLS_TO_JUDGE
            else "✔"
            if tools_ok
            else "✘"
            if tools_broken
            else REVIEW_MARK
        )
        tool_row = ("Tool calls", "", shown, mark)
    gate = None
    if not better and not _only_answers_or_tools_hold_back(step, answers_ok and tools_ok):
        gate = "better"
    elif not total:
        gate = NOT_COMPARED
    elif answers["broken"]:
        gate = "answers"
    elif tools_broken:
        gate = "tool calls"
    elif not tools_ok:
        gate = TOOLS_HELD
    elif not answers_ok:
        gate = "answers"
    if not total:
        said, mark = "none could be compared", "✘"
    elif answers["broken"]:
        said, mark = f"{_broken_of(answers)} broken", "✘"
    elif not answers["differ"]:
        said, mark = f"all {total} identical", "✔"
    else:
        said = f"{answers['identical']} of {total} identical"
        mark = "✔" if answers_ok else REVIEW_MARK if gate in ("answers", TOOLS_HELD) else ""
    checked = answers.get("reasoning_check")
    found = reasoning_check.row(checked) if checked else None
    if answers.get("answer_check") and (found is None or found[0] != "changed"):
        found = reasoning_check.answer_row(answers) or found
    if total and not answers["broken"] and found:
        said, mark = found
    lines = [_metric_line(step)]
    if (goal_line := _goal_line(step)) is not None:
        lines.append(goal_line)
    if reasoning := answers.get("reasoning"):
        lines.append(reasoning_words(reasoning) + ".")
    if checked:
        lines += reasoning_check.lines(checked)
    if answers.get("answer_check"):
        lines += reasoning_check.answer_lines(answers)
    unsure = reasoning["prompts"] - reasoning["identical"] if reasoning else 0
    reworded = answers["differ"] - unsure - len(rounding(answers))
    if answers_ok and reworded and answers.get("judged"):
        lines.append(
            f"{reworded} answers reworded, no more than the earlier run's own repeats differ."
        )
    rows: list[Row] = [("Answers", "", said, mark)]
    if tool_row is not None:
        rows.append(tool_row)
    return TryVerdict(gate is None, gate, lines, rows=tuple(rows))


_FLOOR = re.compile(
    r"(?:past|at) your floor(?: within noise)?: requests per second [\d.]+, (?P<below>\d+)% below "
    r"run \S+'s [\d.]+ \(your floor: (?P<floor>\d+)%\)"
)
_LIMIT = re.compile(
    r"past your (?P<what>.+?) limit: (?P<value>[\d,.]+) ms against (?P<limit>[\d.]+) ms"
)


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
    if loss := step.get("loss"):
        if loss["metric"] != metric:
            return f"{goal}, but {_clause(loss['metric'], loss['change'])}."
        return f"{goal}."
    if found := _FLOOR.search(reason):
        return (
            f"{goal}, but it is {found['below']}% below the speed you started from, past the "
            f"{found['floor']}% loss you allowed."
        )
    if found := _LIMIT.search(reason):
        from .history import named_metric

        other = named_metric(found["what"]) or metric
        return (
            f"{goal}, but {SAID.get(other, (other,))[0]} is {found['value']} ms, over your "
            f"{found['limit']} ms limit."
        )
    if SHORTER_REPLIES in reason and step.get("per_token"):
        text = f"{METRICS.get(metric, (name,))[0]}: {_shorter(step)}"
    elif abs(change) > band:
        text = f"{goal}{f' ({PAIRED})' if _paired(step) else ''}: not a clear gain"
    elif prefill_gain(step):
        text = f"{METRICS.get(metric, (name,))[0]}: {_inside(step)}"
    elif _paired(step):
        text = NOTHING_SIGNIFICANT
    else:
        text = f"{_capital(_clause(metric, change))}, but {UNPAIRED} {band:.0%}"
    if unconfirmed_gain(step):
        text += f", {UNCONFIRMED}"
    if prefill_gain(step):
        text += f"; {_prefill_words(step)}"
    elif "can change answers" in reason:
        text += ", and this change can alter answers, so it is not worth the risk"
    return text + "."


def _sampling(project: Path) -> tuple[float, bool] | None:
    """The temperature the workload samples at when it is above 0, and whether it has no seed;
    None at temperature 0."""
    workload = load_project(project, verify_weights=False).config.workload
    return (workload.temperature, workload.seed is None) if workload.temperature > 0 else None


def _sampled_words(answers: dict[str, Any], temperature: float, unseeded: bool) -> str:
    """The answers of a workload that samples: they vary by sampling, how many differ and, when
    the earlier run's own repeats were measured, how many differ more than those do."""
    seed = " with no seed" if unseeded else ""
    text = (
        f"At temperature {temperature:g}{seed}, {SAMPLED}, so an unchanged setup also rewords "
        f"them: {answers['differ']} of {answers['total']} differ, none broken"
    )
    if judged := answers.get("judged"):
        text += (
            f"; {answers.get('changed', 0)} of {judged} differ more than two runs of your "
            "current setup do"
        )
    return text + "."


def _gate_text(
    gate: str, step: dict[str, Any] | None, sampled: tuple[float, bool] | None = None
) -> str:
    """Why the gate failed, as one sentence; the not-kept line and the rejection reason.
    ``sampled``: the temperature of a workload that samples and whether it has no seed, where an
    unchanged setup also rewords answers."""
    if step is None or step.get("answers") is None:
        return "There was no earlier run to compare it with."
    if gate == "better":
        return _not_better(step)
    if gate == "answers":
        answers = step["answers"]
        if answers["broken"]:
            return f"{_broken_of(answers)} answers came out wrong."
        checked = answers.get("reasoning_check")
        if reasoning_changed(answers):
            return reasoning_check.departs(checked)
        if answer_changed(answers):
            return reasoning_check.answer_departs(answers["answer_check"])
        reasoned = answers.get("reasoning")
        if reasoned and (unsure := reasoned["prompts"] - reasoned["identical"]):
            found = f"; {reasoning_check.gate_words(checked)}" if checked else ""
            return (
                f"{unsure} of {answers['total']} prompt{'' if answers['total'] == 1 else 's'} "
                f"ran out of output tokens while thinking and their reasoning differs{found}, so "
                f"{UNCHECKED}."
            )
        if thought := answers.get("thought"):
            return (
                f"The model only thought on {thought} prompt{'' if thought == 1 else 's'}, so "
                f"{UNCHECKED}."
            )
        if rounding_passes(answers):
            change = answers["answer_check"]["change"]
            return reasoning_check.rounding_words(flag_words(change) or "these settings")
        if sampled is not None:
            return _sampled_words(answers, *sampled)
        return f"{answers['differ']} of {answers['total']} {REWORDED}."
    before, after = step["tool_calls"]["before"] or {}, step["tool_calls"]["after"]
    if gate == "tool calls" and (broke := _broken_calls(before, after)) is not None:
        return (
            f"Tool calls broke: {_broken_words(broke)}, more than your current setup's rate "
            "allows, so some calls cannot be parsed."
        )
    failed = _tool_gate(before, after)
    assert failed is not None
    moved = f"{TOOL_SHARES[failed[0]]} {failed[1]:.0%} → {failed[2]:.0%}"
    text = (
        f"{_capital(moved)}, {TOOLS_CHANGED} (±{failed[3]:.0%}); check that your "
        "agent still gets the calls it needs before you keep it."
    )
    answers = step["answers"]
    if answers.get("differ") and not answers_pass(answers):
        text += f" {_gate_text('answers', step, sampled)}"
    return text


def _pairs(run_dir: Path) -> tuple[Path, list[dict[str, str]]]:
    """The run's report and its answers that differ from the earlier run's, none broken, as
    (prompt id, first answer before, first answer after), from its ``quality.json``; none when
    it has no such file."""
    try:
        found = json.loads((run_dir / "quality.json").read_text("utf-8")).get("pairs") or []
    except (OSError, ValueError, AttributeError):
        found = []
    pairs = [
        {"prompt_id": p["prompt_id"], "before": p["before"][0], "after": p["after"][0]}
        for p in found
        if isinstance(p, dict)
        and p.get("identical") is False
        and not any((p.get("broken") or {}).values())
        and p.get("before")
        and p.get("after")
    ]
    return run_dir / "report.md", pairs


def _cut(text: str) -> str:
    shown = json.dumps(text, ensure_ascii=False)
    return shown if len(shown) <= PAIR_WIDTH else f'{shown[: PAIR_WIDTH - 2]}…"'


def _pair_lines(run_dir: Path) -> list[str]:
    """Up to ``PAIRS_SHOWN`` answers that differ, before and after, side by side."""
    report, pairs = _pairs(run_dir)
    if not pairs:
        return [f'Compare them: {report} (section "Answers before / after")']
    shown = pairs[:PAIRS_SHOWN]
    lines = [
        f'{len(shown)} of the {len(pairs)} that differ (all in {report}, "Answers before / after"):'
    ]
    for pair in shown:
        lines += [
            f"  {pair['prompt_id']}",
            f"    before: {_cut(pair['before'])}",
            f"    after:  {_cut(pair['after'])}",
        ]
    return lines


def _floor_cut(run_dir: Path) -> tuple[bool, str | None]:
    """Whether the run in ``run_dir`` was cut to its fewest requests, and the run it was made
    from, by its window.json."""
    try:
        window = json.loads((run_dir / "window.json").read_text("utf-8"))
    except (OSError, ValueError):
        return False, None
    parent = window.get("parent")
    return "floor_cut" in window, parent if isinstance(parent, str) else None


def _few_compared(run_dir: Path, step: dict[str, Any] | None) -> str | None:
    """How many prompts the answers of a run were compared on when it, or the run it was made
    from, was cut to its fewest requests; None for any other run."""
    answers = step.get("answers") if step else None
    if not answers or not answers["total"]:
        return None
    cut, parent = _floor_cut(run_dir)
    if not cut and (parent is None or not _floor_cut(run_dir.parent / parent)[0]):
        return None
    total = answers["total"]
    return (
        f"Answers compared on {total} prompt{'' if total == 1 else 's'}: this setup is measured "
        "on few requests, so the check rests on few answers."
    )


def _not_compared(run_dir: Path, model: Path | None = None) -> str:
    """Why none of the answers of the run in ``run_dir`` could be compared with the earlier
    run's, what makes them comparable, and how to keep the change unchecked."""
    from .runs import answers_in

    answers = [answer for given in answers_in(run_dir).values() for answer in given]
    keep_anyway = f"To keep it without the check: tensward set {run_dir.name}."
    if answers and all(a.thinking and a.finish_reason == "length" for a in answers):
        return (
            f"No answer could be compared: all {len(answers)} answers ran out of output tokens "
            "while thinking. Raise max_tokens in your prompts, or turn thinking off for the "
            f"check ({thinking_off(model)}), and run tensward init "
            f"again. {keep_anyway}"
        )
    return f"No answer could be compared with the earlier run's. {keep_anyway}"


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
        ids = matched.get("prompts") or []
        which = (
            f" ({', '.join(ids)})"
            if 0 < len(ids) <= 3
            else f" ({len(ids)} prompts; see requests.jsonl in both runs)"
            if ids
            else ""
        )
        return (
            f"It answered {removed:.0%} more of the prompts than run {parent}, which failed "
            f"them{which}, so the two runs did different work and their speed is not compared."
        )
    if why := matched.get("why"):
        return f"It was not compared with run {parent}: {why}."
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


REPAIR_NOT_COMPARED = (
    "Speed and the answer comparison do not apply: the setup before it {failed}, so there is "
    "no speed or set of answers to compare with."
)


def repair_label(record: dict[str, Any]) -> str:
    """The plain label of a kept repair."""
    return (
        f"repair: requests now succeed ({record['succeeded']} of {record['sent']}); "
        "speed not compared yet"
    )


def _repair_verdict(run_dir: Path, record: dict[str, Any]) -> tuple[TryVerdict, str]:
    """A repair check, with the reason when it is not kept: the server started, served at least
    ``1 - MAX_FAILED`` of the requests it was sent, and its answers hold up on their own (none
    empty or repeating; tool calls as for a first run). Its speed and its answers are not
    compared with the setup before it, whose requests mostly failed."""
    from .history import answered_now
    from .runs import RunMetrics
    from .window import MAX_FAILED

    first, text = _first_verdict(run_dir)
    metrics = RunMetrics.read(run_dir / "metrics.json")
    sent = metrics.succeeded + metrics.failed
    record |= {"succeeded": metrics.succeeded, "sent": sent}
    fixed = len(answered_now(run_dir.parent / record["parent"], run_dir))
    answered = f"{metrics.succeeded} of {sent} answered"
    if fixed:
        answered += f", {fixed} prompt{'' if fixed == 1 else 's'} that failed before"
    ok = sent > 0 and (metrics.failed_share or 0.0) <= MAX_FAILED
    rows = (("Requests", "", answered, "✔" if ok else "✘"), *first.rows)
    lines = [
        f"Repair check: {sent} requests, to see that requests now succeed.",
        REPAIR_NOT_COMPARED.format(
            failed=f"failed {record['parent_failed_share']:.0%} of its requests"
        ),
    ]
    if not first.keep and first.gate == "served":
        return replace(first, lines=lines, rows=rows), text
    if not ok:
        verdict = TryVerdict(False, "requests", lines, first.checked, rows)
        return verdict, f"{metrics.failed} of {sent} requests still failed."
    return replace(first, lines=lines, rows=rows), text


def _fails_requests(run_dir: Path, parent: Path | None) -> str | None:
    """Why a step that is not a repair check cannot be kept when it failed more requests than its
    parent's rate allows (``broke_more``), or it or its parent failed more than ``MAX_FAILED`` of
    its requests; None otherwise."""
    from .runs import RunMetrics
    from .window import MAX_FAILED

    step = RunMetrics.read(run_dir / "metrics.json")
    before = RunMetrics.read(parent / "metrics.json") if parent is not None else None
    yours = before.failed_share if before is not None else None
    if before is not None and broke_more(
        before.failed, before.succeeded + before.failed, step.failed, step.succeeded + step.failed
    ):
        return (
            f"{step.failed} of {step.succeeded + step.failed} requests failed (yours: "
            f"{before.failed} of {before.succeeded + before.failed}), more than your current "
            "setup's rate allows."
        )
    if max(step.failed_share or 0.0, yours or 0.0) <= MAX_FAILED:
        return None
    return (
        f"{step.failed} of {step.succeeded + step.failed} requests failed"
        + (f" (yours: {yours:.0%})" if yours is not None else "")
        + "; speed is not compared on a setup that fails requests: try the fit offers first."
    )


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
    project: Path,
    offer: Offer,
    goal: str,
    weight_bytes: int,
    fallback: int,
    requests: int | None = None,
) -> int:
    """Minutes a try takes, from the pace of the newest run: its start, the warm-up, the
    requests the window will send at that pace; ``fallback`` when no run gives a pace, as in
    one where every request failed. With ``requests``, the window sends that many, each at the
    pace of the newest run's answered requests (a failed request returns at once)."""
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
    if requests is not None:
        arrival = workload.arrival
        clients = arrival.concurrency or arrival.max_inflight or 1
        answered = [(r.terminal_ns or r.dispatch_ns) - r.dispatch_ns for r in rows if r.ok]
        per_request = sum(answered) / len(answered) / 1e9 / clients
    else:
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
    analysis.tried = {"census_facts": dict(offer.census_facts), "kb_record": offer.kb_record}
    return analysis


def _check_parent(project: Path, engine: Engine, offer: Offer, kept: Kept | None) -> Path | None:
    """The run the reasoning check of this try compares against: the try's parent, which after
    an undo or a try not kept is not the newest run."""
    from .history import env_args
    from .project import registered_env

    env = {name: value for name, _, value in (var.partition("=") for var in offer.engine_env)}
    if not env and kept is not None:
        env = {name: value for name, _, value in (var.partition("=") for var in kept.env)}
    registered = registered_env(project)
    inherited = {name: value for name, value in os.environ.items() if name not in registered}
    args = frozenset(offer.engine_args or (kept.engine_args if kept else ())) | env_args(
        {**inherited, **env}, engine.flag_rules
    )
    return reasoning_check.parent_dir(project, engine, args, kept)


def run_try(arguments: argparse.Namespace) -> int:
    from .census.refresh import expected_refresh_s, refresh_census
    from .cli import _confirmed, _run_analyse
    from .report.build import baseline_words
    from .wizard import analysis_minutes

    project = arguments.project.expanduser()
    as_json = arguments.json or JSON_OUTPUT.get()
    if arguments.keep:
        code = _keep_reworded(
            project, as_json, None if arguments.keep == KEEP_LATEST else arguments.keep
        )
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
    offer = next((o for o in found if o.n == arguments.offer), None)
    if (shown := _shown_offer(project, arguments.offer, found)) is not None:
        offer = shown if isinstance(shown, Offer) else None
        if offer is None:
            print(shown, file=sys.stderr)
            return 1
    elif offer is None and arguments.offer == 1:
        offer = next((o for o in found if o.n), found[0])
    if offer is None:
        numbers = ", ".join(str(o.n) for o in found if o.n) or "none"
        print(
            f"there is no offer {arguments.offer}: the last report's offers are {numbers}",
            file=sys.stderr,
        )
        return 1
    kept = read_kept(project)
    current, pins = _held(project, kept)
    held = current | pins
    weight_bytes = load_project(project, verify_weights=False).artifact.metadata.weight_bytes
    analysis = _analysis_arguments(arguments, offer)
    repair = repair_plan(project, offer)
    analysis.repair = repair
    fallback = analysis_minutes(project, weight_bytes, engine, runtime, offer.engine_args)
    if repair is not None:
        from .project import declared_workload
        from .wizard import run_minutes, workload_plan

        planned = {**workload_plan(declared_workload(project)), "requests": repair["requests"]}
        fallback = run_minutes(weight_bytes, **planned)
        minutes = expected_minutes(
            project, offer, analysis.goal, weight_bytes, fallback, requests=repair["requests"]
        )
    elif offer.test_seconds:  # the estimate the report summary and the Next menu name
        minutes = math.ceil(offer.test_seconds / 60)
    else:
        minutes = expected_minutes(project, offer, analysis.goal, weight_bytes, fallback)
    source = offers_run(project)
    if repair is None and source is not None and (short := repair_check(source)) is not None:
        return _measure_repaired(
            arguments, project, engine, runtime, offer, short, minutes, as_json
        )
    tried = _words(offer.engine_args, offer.engine_env, held)
    placed = pinned_placement(
        project, load_project(project, verify_weights=False), offer.engine_args, held
    )
    base = _check_parent(project, engine, offer, kept) if repair is None else None
    checks, check_s = reasoning_check.estimate(base) if base else (False, None)
    replays, replay_s = reasoning_check.answer_estimate(base) if base else (False, None)
    minutes += math.ceil(check_s / 60) if check_s else 0
    shown_min = None if untimed(project) else minutes
    refresh_s = (
        expected_refresh_s(
            source, engine, runtime,
            settings_for(load_project(project, verify_weights=False), engine, offer.engine_args),
        )
        if repair is None
        else None
    )  # fmt: skip
    if not as_json:
        print(f"Trying {tried}{_on_top(offer, held)}", file=sys.stderr)
        why = offer.why.split(". Measured together")[0]
        print(f"Why: {placed[0]}; {why}" if placed else f"Why: {why}", file=sys.stderr)
        if repair is not None:
            failed = f"{repair['parent_failed_share']:.0%} of your current setup's requests failed"
            print(
                f"A repair check: {repair['requests']} requests, to see that requests now "
                f"succeed. Speed and the answer comparison do not apply here: {failed}.",
                file=sys.stderr,
            )
        more = (
            ""
            if not replays
            else f", up to {math.ceil(replay_s / 60)} min more if answers differ"
            if replay_s
            else ", a few minutes more if answers differ"
        )
        print(
            f"How long it takes: {UNTIMED}."
            if shown_min is None
            else f"This takes about {shown_min} min{more}.",
            file=sys.stderr,
        )
        if repair is None and not offer.first and shown_min is not None:
            print(
                "If it looks better, your current setup is measured once more right after it, "
                f"about {shown_min} min more, so a machine that drifted does not pass for a gain.",
                file=sys.stderr,
            )
        if refresh_s is not None:
            print(
                "If it is kept, a GPU census start may follow, about "
                f"{max(1, round(refresh_s / 60))} min more (a diagnostic, not a speed claim).",
                file=sys.stderr,
            )
        if checks:
            print(
                "It includes the reasoning check."
                if check_s
                else "Then the reasoning check, not timed on this setup yet.",
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
    if not _confirmed(arguments, engine, runtime, shown_min):
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
        step = step_block(project / "runs" / run_id)
        if (slow := _too_slow(project / "runs" / run_id)) is not None:
            verdict, text = _first_verdict(project / "runs" / run_id)
            if verdict.keep:
                return _slow_start(project, as_json, offer, held, tried, run_id, slow, verdict)
        elif repair is not None:
            step = None
            verdict, text = _repair_verdict(project / "runs" / run_id, repair)
        elif offer.first and step is None:
            verdict, text = _first_verdict(project / "runs" / run_id)
        else:
            tools_on = step is not None and step.get("tool_calls") is not None
            verdict = judge_step(step, tools_on=tools_on)
            if (few := _few_compared(project / "runs" / run_id, step)) is not None:
                verdict = replace(verdict, lines=[*verdict.lines, few])
            unmatched = verdict.gate == NO_COMPARISON and _unmatched(project / "runs" / run_id)
            if verdict.gate == NOT_COMPARED:
                model = load_project(project, verify_weights=False).artifact.path
                text = _not_compared(project / "runs" / run_id, model)
            else:
                sampled = _sampling(project) if verdict.gate in ("answers", TOOLS_HELD) else None
                text = unmatched or (
                    _gate_text(verdict.gate, step, sampled) if verdict.gate else ""
                )
    if (
        repair is None
        and failure is None
        and status == 0
        and not offer.first
        and (failing := _fails_requests(project / "runs" / run_id, source)) is not None
    ):
        verdict = TryVerdict(False, "requests", verdict.lines, verdict.checked, verdict.rows)
        text = failing
    answers = step.get("answers") if step else None
    held_on = verdict.gate
    review = held_on in ("answers", TOOLS_HELD) and answers is not None and not answers["broken"]
    again: dict[str, Any] | None = None
    if (verdict.keep or review) and step is not None and repair is None and not offer.first:
        again = _again(project, engine, runtime, run_id, step, analysis.ready_timeout)
    if again is AGAIN_FAILED:
        again = None
        verdict = replace(verdict, lines=[*verdict.lines, NOT_AGAIN])
    if again is not None and step is not None:
        step = {**step, "before": again["mean"], "change": again["change"], "again": True}
        said = _again_line(step, again)
        if again["better"]:
            verdict = replace(verdict, lines=[_metric_line(step), said, *verdict.lines[1:]])
        else:
            verdict = TryVerdict(False, "better", [said], verdict.checked, verdict.rows)
            text = " ".join(filter(None, (_again_unconfirmed(step), text if review else "")))
            review = False
    forced = review and getattr(arguments, "keep_faster", False)
    if forced:
        verdict = replace(verdict, keep=True, gate=None)
        review = False
    now: Kept | None = None
    if verdict.keep:
        try:
            now, _ = keep(project, run_id, accept_unreviewed=True)
        except NotKept as error:
            verdict = TryVerdict(False, "keep", verdict.lines, verdict.checked, verdict.rows)
            text = f"It could not be kept: {error}."
    if verdict.keep and repair is None:  # a repair's census comes with its full measurement
        refresh_census(
            project, project / "runs" / run_id, source=source, engine=engine, runtime=runtime,
            ready_timeout_s=analysis.ready_timeout,
        )  # fmt: skip
    if not verdict.keep:
        try:
            reject(project, run_id, reason=text)
        except NotRejected as error:
            print(f"not rejected: {error}", file=sys.stderr)
    try:
        following = offers(project)[0]
    except NothingToTry:
        following = None
    reworded = review
    keepable = reworded or (
        verdict.gate == "better"
        and step is not None
        and (
            unconfirmed_gain(step)
            or prefill_gain(step)
            or (again is not None and _again_gain(step, again))
        )
    )
    after_keep = _after_keep(project, run_id, pins) if keepable else None
    if as_json:
        choices = _choices(
            following, now, held, kept=verdict.keep, reworded=keepable, after_keep=after_keep
        )
        reason = None if verdict.keep else text or None
        result = _json_result(run_id, offer, verdict, step, following, choices, reason)
        if repair is not None:
            result = json.dumps({**json.loads(result), "repair": repair}, sort_keys=True)
        if again is not None:
            result = json.dumps({**json.loads(result), "parent_again": again}, sort_keys=True)
        if reworded:
            sample = _pairs(project / "runs" / run_id)[1][:PAIRS_SHOWN]
            result = json.dumps(
                {**json.loads(result), "review": True, "pairs": sample}, sort_keys=True
            )
        print(result)
        return 0 if verdict.keep else REVERTED
    lines = [f"Tried: {tried}{_on_top(offer, held)}", ""]
    lines += _result_lines(project / "runs", run_id, step, verdict, alone=repair is not None)
    if placed:
        lines.append(PINNED_RESULT.format(rest=placed[1], both=placed[0]))
    if (layers := _layer_line(project / "runs", run_id, offer.engine_args)) is not None:
        lines.append(layers)
    if verdict.keep and repair is not None:
        lines.append(
            f"Kept in this project as a {repair_label(repair)}. Your running server is unchanged."
        )
        lines += _ollama_lines(project, (*offer.engine_args, *offer.engine_env))
    elif verdict.keep:
        words = f" ({baseline_words(step['baseline'])})" if step and step.get("baseline") else ""
        if forced:
            what = "tool calls changed" if held_on == TOOLS_HELD else "answers differ"
            lines.append(f"Kept although {what} ({KEEP_FASTER}): {text}")
        lines.append(f"Kept in this project{words}. Your running server is unchanged.")
        lines += _ollama_lines(project, (*offer.engine_args, *offer.engine_env))
    elif reworded:
        what = "tool calls changed" if held_on == TOOLS_HELD else "answers differ"
        faster = (
            f"Faster, {what}" if step and step.get("verdict") == "better" else what.capitalize()
        )
        lines.append(f"{faster}: review. {text}")
        if answers and answers.get("differ"):
            lines += _pair_lines(project / "runs" / run_id)
        lines.append("Not applied: nothing changes unless you keep it.")
    elif step is not None and verdict.gate == "better" and prefill_gain(step):
        lines.append(f"Held for you: {text}")
        lines.append("Not applied: nothing changes unless you keep it.")
    elif step is not None and step.get("goal") and verdict.gate == "better":
        lines.append(f"Not kept: {text[:1].lower()}{text[1:]} Nothing changed.")
    else:
        lines.append(f"Not kept. {text} Nothing changed.")
    lines.append("")
    ended = None
    if following is None and after_keep is None:
        ended = wrap_up(
            project,
            now if verdict.keep else kept,
            pending=reworded,
            reasoned=bool(answers and answers.get("reasoning_check")),
        )
    lines += _menu(
        following,
        now,
        held,
        kept=verdict.keep,
        reworded=keepable,
        wrap_up=ended,
        after_keep=after_keep,
        beyond=_beyond(project / "runs" / run_id) if ended is not None else (),
        summary=_beyond_line(project / "runs" / run_id),
        unmeasured=verdict.keep and repair is not None,
    )
    print("\n".join(lines))
    return 0 if verdict.keep else REVERTED


def _measure_repaired(
    arguments: argparse.Namespace,
    project: Path,
    engine: Engine,
    runtime: Runtime,
    offer: Offer,
    short: dict[str, Any],
    minutes: int,
    as_json: bool,
) -> int:
    """A try from a setup kept as a repair, whose speed its short check did not measure: measure
    that setup on the full window first, then try the change against that measurement, never
    against the short check. When the full measurement no longer offers the change, say so."""
    from .census.start import expected_census_s
    from .cli import _confirmed, _run_analyse, build_parser
    from .history import last_load_s
    from .wizard import analysis_minutes, load_estimate_s

    weight_bytes = load_project(project, verify_weights=False).artifact.metadata.weight_bytes
    kept = read_kept(project)
    args = kept.engine_args if kept is not None else ()
    workload_minutes = expected_minutes(
        project, offer, getattr(arguments, "goal", None) or "throughput", weight_bytes,
        analysis_minutes(project, weight_bytes, engine, runtime, args),
        requests=_configured(project),
    )  # fmt: skip
    census_s = expected_census_s(
        engine, runtime, settings_for(load_project(project, verify_weights=False), engine, args),
        last_load_s(project) or load_estimate_s(weight_bytes),
    )  # fmt: skip
    workload_minutes += math.ceil(census_s / 60) if census_s else 0
    current, pins = _held(project, kept)
    tried = _words(offer.engine_args, offer.engine_env, current | pins)
    print(
        f"Your current setup was kept as a repair: {short.get('requests')} requests checked that "
        "requests succeed; its speed is not measured yet. First it is measured on the full "
        f"window, about {workload_minutes} min; then {tried} is tried, about {minutes} min "
        f"(about {workload_minutes + minutes} min in all).",
        file=sys.stderr,
    )
    if not _confirmed(arguments, engine, runtime, workload_minutes + minutes):
        print("Nothing started.", file=sys.stderr)
        return 1
    again = build_parser().parse_args(["analyse", "--window", "full", "--remeasure"])
    for key, value in vars(arguments).items():
        if key not in ("run", "command", "offer", "json", "keep") and hasattr(again, key):
            setattr(again, key, value)
    again.yes = True
    _run_analyse(again, print_report=False)
    run_id = getattr(again, "finished_run", "")
    if not run_id or not (project / "runs" / run_id / "report.json").is_file():
        print(
            "The full measurement of your current setup did not finish. Nothing changed.",
            file=sys.stderr,
        )
        return 1
    from .runs import RunMetrics
    from .window import MAX_FAILED

    found = _offers_or_none(project)
    same = next(
        (
            o.n or 1 for o in found
            if (o.engine_args, o.engine_env) == (offer.engine_args, offer.engine_env)
            and (o.n or o is found[0])
        ),
        None,
    )  # fmt: skip
    metrics = RunMetrics.read(project / "runs" / run_id / "metrics.json")
    measured = f"Measured your current setup on the full window (run {run_id})"
    if (metrics.failed_share or 0.0) > MAX_FAILED or same is None:
        following = found[0] if found else None
        said = (
            f"{measured}: {metrics.failed} of {metrics.succeeded + metrics.failed} requests "
            "failed, more than its repair check showed, so its speed is no reference; "
            f"{tried} is not tried."
            if (metrics.failed_share or 0.0) > MAX_FAILED
            else f"{measured}; its report no longer offers {tried}."
        )
        choices = _choices(following, kept, current | pins, kept=kept is not None)
        if as_json:
            print(json.dumps({"offer": None, "reason": said, "next": next_document(choices)}))
        else:
            print("\n".join([said, "", *next_lines(choices, more=following is not None)]))
        return 0
    speed = metrics.output_throughput
    print(
        f"{measured}" + (f": {number(speed)} tok/s." if speed is not None else "."),
        file=sys.stderr,
    )
    if (note := _before_repair(project / "runs", run_id, short.get("parent"))) is not None:
        print(note, file=sys.stderr)
    arguments.offer, arguments.yes = same, True
    return run_try(arguments)


def _before_repair(runs: Path, run_id: str, parent: str | None) -> str | None:
    """The full measurement's time per token next to that of the requests the run before the
    repair answered: a note, since a larger context can move layers to the CPU and
    ``--kv-unified`` was slower in every measured run, but the two windows are not matched."""
    from .history import _clients
    from .window import figures, read_rows

    if parent is None:
        return None
    clients = _clients(runs.parent)
    now, before = read_rows(runs / run_id), read_rows(runs / parent)
    answered = [row for row in before or () if row.ok]
    if not now or not answered:
        return None
    after = figures(now, clients=clients).values.get("tpot_p95_ms")
    then = figures(answered, clients=clients).values.get("tpot_p95_ms")
    if after is None or then is None:
        return None
    return (
        f"Per token, slowest 1 in 20: {number(after)} ms now; {number(then)} ms on the "
        f"{len(answered)} requests run {parent} answered before the repair (a note, not a "
        "comparison: a larger context can move layers to the CPU, and --kv-unified was slower "
        "in every measured run)."
    )


def _configured(project: Path) -> int | None:
    from .project import declared_workload

    workload = declared_workload(project)
    return workload.request_count if workload is not None else None


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
    named = (f"tensward try {following.n}" if following.n else "tensward try") if following else ""
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
        f"Not kept yet; nothing changed. Keep it to serve it: {KEEP_COMMAND}",
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


def _again(
    project: Path,
    engine: Engine,
    runtime: Runtime,
    run_id: str,
    step: dict[str, Any],
    ready_timeout_s: float,
) -> dict[str, Any] | None:
    """The try's parent measured once more right after it (A-B-A) and the step's goal change
    judged against the mean of its two runs, kept in the try's parent-again.json; None when that
    cannot be done, AGAIN_FAILED when the parent was started again and did not measure."""
    from .recheck import common_drift, judged_again, measure_parent, record_judged

    if step.get("metric") not in METRICS or not step.get("before"):
        return None
    step_dir = project / "runs" / run_id
    try:
        parent = json.loads((step_dir / "window.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if not parent.get("parent"):
        return None
    figures = measure_parent(project, engine, runtime, step_dir, parent["parent"], ready_timeout_s)
    if figures is None:
        return AGAIN_FAILED
    up, key = SAID.get(step["metric"], ("", True))[1], METRICS[step["metric"]][1]
    if (
        step["metric"].endswith("_p95_ms")
        and (drift := common_drift(step_dir, project / "runs" / parent["parent"], step["metric"]))
        is not None
    ):
        figures = {**figures, key: step["before"] * (1 + drift)}
    judged = judged_again({**step, "better_up": up}, key, figures)
    if judged is None:
        return AGAIN_FAILED
    record_judged(step_dir, judged)
    return judged


AGAIN_FAILED: dict[str, Any] = {}  # the parent was started again and gave no figures
NOT_AGAIN = (
    "Your current setup could not be measured once more, so this verdict is from the first "
    "run of each alone."
)


def _again_line(step: dict[str, Any], again: dict[str, Any]) -> str:
    """What the second run of the parent measured, and when it moved more than the noise."""
    label = METRICS[step["metric"]][0]
    moved = "0%" if abs(again["drift"]) < 0.005 else f"{again['drift']:+.0%}"
    head = f"Your current setup, measured again right after: {label} {moved} against its first run"
    if abs(again["drift"]) > step["band"]:
        head += f", more than the {step['band']:.0%} noise floor: the machine drifted"
    return f'{head}; "before" is the mean of its two runs.'


def _again_gain(step: dict[str, Any], again: dict[str, Any]) -> bool:
    """Whether a change the A-B-A no longer confirms still moved the right way, inside its band:
    the user may keep it."""
    up = SAID.get(step["metric"], ("", True))[1]
    return not again["better"] and (again["change"] > 0) == up and again["change"] != 0


AGAINST_MEAN = "against the mean of your current setup's two runs"


class NoGain(NotKept):
    """The try the parent's second measurement found no gain: it cannot be kept."""


def _again_refusal(run_dir: Path) -> str | None:
    """Why the try in ``run_dir`` cannot be kept when its parent, measured once more right after
    it, made its goal change no gain at all; None otherwise."""
    again, step = read_judged(run_dir), step_block(run_dir)
    if again is None or step is None or step.get("metric") not in SAID or again["better"]:
        return None
    if _again_gain(step, again):
        return None
    return (
        f"{run_dir.name} is no gain: your current setup, measured once more right after it, "
        f"puts {METRICS[step['metric']][0]} at {_paired_word(step)}{again['change']:+.0%} "
        f"{AGAINST_MEAN}; "
        "tensward try measures it again"
    )


def _again_unconfirmed(step: dict[str, Any]) -> str:
    label = METRICS[step["metric"]][0]
    against = f"{label}: {change_words(step)} {AGAINST_MEAN}"
    if (step["change"] > 0) == SAID.get(step["metric"], ("", True))[1] and step["change"]:
        return f"{against}, inside the {step['band']:.0%} noise, {UNCONFIRMED}."
    return f"{against}: not a gain."


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
    runs: Path,
    run_id: str,
    step: dict[str, Any] | None,
    verdict: TryVerdict,
    *,
    alone: bool = False,
) -> list[str]:
    """The table of the run before and after the change, then what the goal metric did and what
    limits the run; with ``alone``, the run's own figures only, as for a repair check."""
    after = _recorded(runs / run_id)
    try:
        parent = json.loads((runs / run_id / "window.json").read_text("utf-8")).get("parent")
    except (OSError, ValueError):
        parent = None
    before = _recorded(runs / parent) if parent and not alone else None
    significant = _significant(runs / run_id)
    goal = None
    if step is not None and step.get("metric") in METRICS and step.get("before") is not None:
        key = METRICS[step["metric"]][1]
        goal = (key, step["before"], step["after"], change_words(step))
    rows = stands_rows(before, after, goal, significant=significant) if after is not None else []
    if before is None:
        header: tuple[str, ...] = ("", "now", "")
        table: list[tuple[str, ...]] = [(label, now, "") for label, now, _, _ in rows]
        table += [(label, now, mark) for label, _, now, mark in verdict.rows]
    else:
        header = ("", "before", "after", "change")
        table = [*rows, *verdict.rows]
    lines = [*columns(header, table), ""] if table else []
    notes = [ONE_RUN] if any(row[-1].endswith(ONE_RUN_MARK) for row in rows) else []
    if goal is not None and step is not None and before is not None:
        if (note := _paired_note(runs / run_id, step)) is not None:
            notes += textwrap.wrap(note, NOTE_WIDTH)
    lines[-1:-1] = notes
    lines += verdict.lines
    if after is not None:
        lines.append(limit_line(before, after))
    if parent and not alone:
        lines += _snapshot_lines(runs / parent, runs / run_id)
    return lines


def _ollama_lines(project: Path, engine_args: Sequence[str]) -> list[str]:
    """The kept change in Ollama's settings, when the project was imported from Ollama."""
    choices = read_choices(project)
    if choices is None or choices.ollama is None:
        return []
    try:
        setup = json.loads((project / "project.json").read_text("utf-8"))["current_setup"]
        parallel = setup["settings"].get("max_concurrent_requests")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        parallel = None
    return ollama_settings.lines(
        choices.ollama, engine_args, parallel if isinstance(parallel, int) else None
    )


def _snapshot_lines(before: Path, after: Path) -> list[str]:
    """The snapshot numbers of a step on a hybrid model, from the logs of the run it was made
    from and of the step."""
    try:
        old, new = (snapshots.read((d / "server.log").read_text("utf-8")) for d in (before, after))
    except OSError:
        return []
    shown = snapshot_lines(old, new)
    return ["", *shown] if shown else []


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

    kept = read_kept(project)
    serving = SERVE_TEXT if kept is not None else SERVE_AS_IS
    run = latest_run(project)
    choices = [*(_beyond(project / "runs" / run) if run else []), ("tensward serve", serving)]
    said = NOTHING_WORTH
    if kept is not None and repair_check(project / "runs" / kept.run) is not None:
        said = wrap_up(project, kept) or said
        choices.insert(0, ("tensward analyse", "measure its speed on the full window"))
    if as_json:
        print(json.dumps({"offer": None, "reason": said, "next": next_document(choices)}))
    else:
        summary = _beyond_line(project / "runs" / run) if run else None
        lines = next_lines(choices, more=False, wrap_up=said)
        print("\n".join(_with_summary(lines, summary)))
    return 0


def _keep_reworded(
    project: Path, as_json: bool, step: str | None = None, *, then_try: bool = True
) -> int | None:
    """``tensward try --keep``: keep what the last try held for answers that differ. None when
    there is a next change to try, which the caller then runs; else the menu and an exit code.
    ``then_try`` False (``tensward keep``): always the menu, and with no ``step`` the newest held
    step when the latest run is not one."""
    try:
        words, why = keep_reworded(project, step)
    except NotKept as error:
        held = held_steps(project) if step is None and not then_try else []
        if isinstance(error, NoGain):  # the try just refused is no gain: keep nothing else
            held = []
        if not held and not then_try and step is None:
            done = _already_kept(project, as_json)
            if done is not None:
                return done
        if not held:
            print(error, file=sys.stderr)
            if not then_try and not as_json:
                print("\n".join(["", *_nothing_held_menu(project)]), file=sys.stderr)
            return 1
        try:
            words, why = keep_reworded(project, held[-1][0].run)
        except NotKept as again:
            print(again, file=sys.stderr)
            return 1
    said = f"Kept {words} ({why})."
    try:
        following = offers(project)[0]
    except NothingToTry:
        following = None
    now = read_kept(project)
    if not as_json:
        said = "\n".join([said, *_ollama_lines(project, now.engine_args if now else ())])
    if following is not None and why != SLOW_KEPT and then_try:
        print(said, file=sys.stderr if as_json else sys.stdout)
        return None
    current, pins = _held(project, now)
    shown = following if why == SLOW_KEPT or not then_try else None
    choices = _choices(shown, now, current | pins, kept=True)
    if as_json:
        print(json.dumps({"kept": True, "change": words, "next": next_document(choices)}))
    else:
        more = following is not None
        print("\n".join([said, "", *next_lines(choices, more=more)]))
    return 0


def _nothing_held_menu(project: Path) -> list[str]:
    """The menu after ``tensward keep`` found nothing held: the next change, else serving."""
    now = read_kept(project)
    try:
        try:
            following: Offer | None = offers(project)[0]
        except NothingToTry:
            following = None
        current, pins = _held(project, now)
    except Exception:  # noqa: BLE001 - the menu never hides the line above it
        return next_lines(
            [("tensward analyse", "measure your setup"), ("tensward serve", "serve it")]
        )
    return _menu(following, now, current | pins, kept=now is not None)


def _already_kept(project: Path, as_json: bool) -> int | None:
    """``tensward keep`` when the last try kept its change itself (better, or
    ``--keep-faster``): say so with the menu; None when the last run is not the kept one."""
    now = read_kept(project)
    if now is None or now.run != latest_run(project):
        return None
    try:
        following = offers(project)[0]
    except NothingToTry:
        following = None
    current, pins = _held(project, now)
    choices = _choices(following, now, current | pins, kept=True)
    words = tuned_words(project, now) or now.run
    said = f"Already kept: {words} (run {now.run}); nothing is waiting for tensward keep."
    if as_json:
        print(json.dumps({"kept": True, "already": True, "run": now.run,
                          "next": next_document(choices)}))  # fmt: skip
    else:
        print("\n".join([said, "", *next_lines(choices, more=following is not None)]))
    return 0


def run_keep(arguments: argparse.Namespace) -> int:
    """``tensward keep [STEP]``: keep the change the last try held for answers that differ (or
    the held step STEP), then show the menu."""
    project = arguments.project.expanduser()
    step = None if arguments.step == KEEP_LATEST else arguments.step
    code = _keep_reworded(project, arguments.json or JSON_OUTPUT.get(), step, then_try=False)
    return 1 if code is None else code


def run_discard(arguments: argparse.Namespace) -> int:
    """``tensward discard [STEP]``: drop the change the last try held for answers that differ
    (or the held step STEP); it is no longer offered."""
    project = arguments.project.expanduser()
    as_json = arguments.json or JSON_OUTPUT.get()
    if arguments.step != KEEP_LATEST:
        try:
            run = _held_run(project, arguments.step)
        except NotKept as error:
            print(error, file=sys.stderr)
            return 1
    else:
        latest = latest_run(project)
        held = [r.run for r, _, _ in _held_all(project)]
        if not held:
            print(f"Nothing to discard: no step is waiting for {KEEP_COMMAND}.", file=sys.stderr)
            return 1
        run = latest if latest in held else held[-1]
    try:
        rejection = reject(project, run, reason=DISCARDED)
    except NotRejected as error:
        print(f"not discarded: {error}", file=sys.stderr)
        return 1
    try:
        following = offers(project)[0]
    except NothingToTry:
        following = None
    now = read_kept(project)
    current, pins = _held(project, now)
    choices = _choices(following, now, current | pins, kept=now is not None)
    if as_json:
        print(json.dumps({"discarded": run, "change": rejection.flags,
                          "next": next_document(choices)}))  # fmt: skip
        return 0
    said = f"Discarded {rejection.flags} (run {run}); it is not offered again. Nothing changed."
    print("\n".join([said, "", *next_lines(choices, more=following is not None)]))
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
    unmeasured: bool = False,
) -> list[tuple[str, str]]:
    after = frozenset((*now.engine_args, *now.env)) | current if now else current
    flags = _words(following.engine_args, following.engine_env, after) if following else None
    return next_choices(
        flags, kept=kept, reworded=reworded, slow=slow, after_keep=after_keep,
        unmeasured=unmeasured,
    )  # fmt: skip


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
    unmeasured: bool = False,
) -> list[str]:
    """The ``Next:`` menu after a try; ``beyond``, the Beyond flags choices, lead it, and
    ``summary``, the Beyond flags line, stands just above it."""
    choices = _choices(
        following, now, current, kept=kept, reworded=reworded, after_keep=after_keep,
        unmeasured=unmeasured,
    )  # fmt: skip
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


def wrap_up(
    project: Path, kept: Kept | None, *, pending: bool = False, reasoned: bool = False
) -> str | None:
    """The last line of a session with no more changes to try, above its one Next menu: the
    kept setup's speed, then, when the last change is better but rewords answers (or, when
    ``reasoned``, its reasoning is the user's call), the decision it waits for; None when there
    is neither."""
    from .runs import RunMetrics

    speed = metrics = None
    if kept is not None:
        try:
            metrics = RunMetrics.read(project / "runs" / kept.run / "metrics.json")
            speed = metrics.output_throughput
        except Exception:
            speed = None
    now = f"Your model now runs at {number(speed)} tok/s" if speed is not None else None
    rest = "; nothing else helped."
    if kept is not None and metrics is not None and repair_check(project / "runs" / kept.run):
        sent = metrics.succeeded + metrics.failed
        now = f"Kept as a {repair_label({'succeeded': metrics.succeeded, 'sent': sent})}"
        rest = " (tensward analyse measures it)."
    last = latest_run(project) if pending else None
    held = [
        f"Held back, {r.run}: {gain}. {r.reason} "
        + (
            f"Keep it: {KEEP_COMMAND} {r.run}"
            if current
            else "It was measured on an older setup; tensward try measures it again."
        )
        for r, gain, current in _held_all(project)
        if r.run != last
    ]
    if pending:
        why = "its reasoning differs" if reasoned else "its answers differ"
        decision = (
            f"The last change is faster but {why}: keep it ({KEEP_COMMAND}), drop it "
            f"({DISCARD_COMMAND}) or serve without it (tensward serve)."
        )
        return "\n".join([f"{now}. {decision}" if now else decision, *held])
    if held:
        return "\n".join([f"{now}{rest}" if now else NO_MORE, *held])
    return f"{now}{rest}" if now else None


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
    if TOO_SLOW_FLOOR in why:
        why = why[why.index(TOO_SLOW_FLOOR) :]
    elif slow:
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
        return 1 if machine else REVERTED
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
    return 1 if machine else REVERTED


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


def _broken_of(answers: dict[str, Any]) -> str:
    """``N of M`` broken answers: M counts answers (each prompt may be asked several times)."""
    return f"{answers['broken']} of {answers.get('answered') or answers['total']}"


def _answers_said(step: dict[str, Any] | None) -> str:
    answers = step.get("answers") if step else None
    if not answers:
        return "-"
    if not answers["total"]:
        return "not compared"
    if answers["broken"]:
        return f"{_broken_of(answers)} wrong"
    if not answers["differ"]:
        return "same"
    if rounding(answers) and answers_pass(answers):
        return f"{answers['differ']} of {answers['total']} differ by rounding"
    return f"{answers['differ']} of {answers['total']} differ"


def run_runs(arguments: argparse.Namespace) -> int:
    from .history import recorded_env
    from .running import read_running

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
    stopped = {p.run: p for p in partial_runs(project)}
    for run_dir in sorted(p for p in runs.iterdir()) if runs.is_dir() else ():
        if (cut := stopped.get(run_dir.name)) is not None:
            rows.append({
                "id": run_dir.name, "started_at": None, "change": "-", "verdict": "-",
                "state": "partial", "shown": "-", "result": cut.said(), "speed": "-",
                "answers": "-",
            })  # fmt: skip
            continue
        if not (run_dir / "run.json").is_file():
            continue
        run_file = RunFile.read(run_dir)
        args = [*run_file.engine_args, *sorted(recorded_env(run_file.model_dump(), rules))]
        step = step_block(run_dir)
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
        repaired = repair_check(run_dir) is not None
        answers = step.get("answers") if step else None
        checked = answers is not None and "reasoning_check" in answers
        reworded = answers is not None and answers["differ"] > 0 and not answers_pass(answers)
        rows.append(
            {
                "id": run_dir.name,
                "started_at": run_file.started_at,
                "change": " ".join(args) or "baseline",
                "verdict": verdict,
                "state": state,
                "shown": flag_words(shown) if shown else "your current setup",
                "result": "kept, repair: requests now succeed"
                if repaired and state == "kept"
                else _result_said(state, bool(args), reworded, checked),
                "speed": "not compared"
                if repaired
                else "-"
                if step is None
                else f"{SAID.get(step['metric'], (step['metric'],))[0]} {change_words(step)}"
                + (f" {AGAINST_MEAN}" if step.get("again") else ""),
                "answers": _answers_said(step),
            }
        )
    if arguments.json or JSON_OUTPUT.get():
        print(json.dumps(
            [{k: v for k, v in row.items() if k not in ("shown", "result", "speed", "answers")}
             for row in rows], sort_keys=True))  # fmt: skip
        return 0
    if (running := read_running(project)) is not None:
        choices = [
            ("tensward stop", "stop it and its server"),
            ("tensward runs", "check again later; its run is listed when it ends"),
        ]
        article = "A" if running.noun == "try" else "An"
        said = f"{article} {running.noun} is running ({running.said()})."
        print("\n".join([*([*_table(rows), ""] if rows else []), said, *next_lines(choices)]))
        return 0
    if all(row["state"] == "partial" for row in rows):
        from .cli import first_analysis_minutes

        choices = [
            (
                "tensward analyse",
                f"measure your setup first (~{first_analysis_minutes(project)} min)",
            ),
            ("tensward serve", "serve your registered setup as it is"),
        ]
        table = [*_table(rows), ""] if rows else ["No runs yet."]
        print("\n".join([*table, *next_lines(choices)]))
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


def _result_said(state: str, changed: bool, reworded: bool, checked: bool) -> str:
    if not changed:
        return "baseline"
    if state == "kept" and reworded:
        said = "reasoning differs" if checked else "answers differ"
        return f"kept, {said} (you accepted)"
    if state == "kept":
        return "kept"
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


BATCH_ARGS = ("batch-size", "ubatch-size")


def _layers_on_card(run_dir: Path) -> str | None:
    """Where the run's layers lived: "47 of 47 layers, the experts of 16 on the host"; None
    when the run did not record it."""
    block = (_recorded(run_dir) or {}).get("placement") or {}
    placed, memory = block.get("placement") or {}, block.get("memory") or {}
    total = memory.get("total_layers")
    if not placed or not total:
        return None
    words = f"{total - int(placed.get('host_layers') or 0)} of {total} layers"
    if experts := placed.get("host_expert_layers"):
        words += f", the experts of {experts} on the host"
    return words


def _layer_line(runs: Path, run_id: str, engine_args: Sequence[str]) -> str | None:
    """A -b / -ub try's layers on the card beside its parent's: llama.cpp's --fit places layers
    by the compute buffer, so a larger micro-batch can move layers to the host."""
    if not any(arg.partition("=")[0] in BATCH_ARGS for arg in engine_args):
        return None
    if (after := _layers_on_card(runs / run_id)) is None:
        return None
    try:
        parent = json.loads((runs / run_id / "window.json").read_text("utf-8")).get("parent")
    except (OSError, ValueError):
        parent = None
    before = _layers_on_card(runs / parent) if parent else None
    if before is None:
        return f"On the card: {after}."
    if before == after:
        return f"On the card: {after}, as before."
    return f"On the card: {after} (before: {before})."
