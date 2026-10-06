"""Renders and writes the before/after comparison and the answers side by side."""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Callable, Mapping, Sequence

from pydantic import JsonValue

from ..compare import HEALTH_NAMES, NO_FLOOR, Answer, failures_rose, percent_text
from ..files import write_json, write_private
from ..inputs import PromptEntry
from ..project import ResolvedProject
from ..runs import SPEED_LABELS, Comparison, RunRecord
from ..text import table
from ..workload import ChatMessage, ImagePart

INLINE_PAIRS = 3  # prompts shown with their answers in the report
INLINE_CHARS = 400  # per answer shown in the report
FILE_CHARS = 4_000  # per answer written to answers.md
FILE_ANSWERS = 3  # distinct answers per side and prompt in answers.md


def _fenced(text: str) -> str:
    """``text`` in a code fence long enough that backticks inside it cannot close it."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{text}\n{fence}"


def _trimmed(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "… (trimmed)"


def _show(answer: Answer, limit: int) -> str:
    calls = [f"{name}({arguments})" for name, arguments in answer.tool_calls or ()]
    return _fenced(_trimmed("\n".join([answer.text, *calls]).strip() or "(empty)", limit))


def _cell(value: float | None, show: Callable[[float], str], note: str = "") -> str:
    return "—" if value is None else show(value) + note


def _number(value: float) -> str:
    return f"{value:,.0f}" if value >= 1000 else f"{value:.1f}"


def _decimal(value: float) -> str:
    return f"{value:.2f}"


def _change_text(before: float, after: float) -> str:
    """The relative change as a signed percent, or ÷N for a fall of 90% or more."""
    percent = round((after / before - 1) * 100)
    if percent <= -90 and after > 0:
        return f"÷{round(before / after)}"
    return f"{'−' if percent < 0 else '+'}{abs(percent)}%"


def _change(before: float | None, after: float | None) -> str:
    return "" if before is None or after is None or before == 0 else _change_text(before, after)


_TOOL_CALL_LABELS = {
    "produced_call": "produced a tool call",
    "valid_json": "valid JSON arguments",
    "known_tool": "known tool",
    "schema_valid": "arguments match the schema",
}


def _tool_call_rows(baseline: RunRecord, candidate: RunRecord) -> list[list[str]]:
    sides = [(r.metrics.tool_calls if r.metrics else None) or {} for r in (baseline, candidate)]

    def share(side: Mapping[str, JsonValue], key: str) -> float | None:
        value = side.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    return [
        [f"tool calls: {label}", *(_cell(share(side, key), percent_text) for side in sides)]
        for key, label in _TOOL_CALL_LABELS.items()
        if any(key in side for side in sides)
    ]


def _mean(values: Sequence[float | None]) -> float | None:
    known = [value for value in values if value is not None]
    return fmean(known) if known else None


def _baseline_name(comparison: Comparison) -> str:
    if comparison.recorded:
        return "your recorded answers"
    return f"run {comparison.baseline_id}" if comparison.between_runs else "your current setup"


def _engine_rate_note(comparison: Comparison) -> str | None:
    """The line that says which runs' output rate is the engine's, not the requests'."""
    sides = [
        name
        for name, rated in zip((_baseline_name(comparison), "this run"), comparison.engine_rated)
        if rated
    ]
    if not sides:
        return None
    return (
        f"Output tokens per second for {' and '.join(sides)} is the engine's own count over the "
        "window: the requests' token count disagreed with it (see Checks)."
    )


# Metrics where a rise is worse; for the others (throughput) a fall is worse.
_LOWER_IS_BETTER = frozenset({"ttft_p50_ms", "ttft_p95_ms", "tpot_p95_ms"})
_CHANGE_ROWS = (
    ("output", "output_throughput", " tok/s"),
    ("requests", "request_throughput", "/s"),
    ("TTFT p50", "ttft_p50_ms", " ms"),
    ("TTFT p95", "ttft_p95_ms", " ms"),
    ("TPOT p95", "tpot_p95_ms", " ms"),
)
NOISE_NOTE = "One run each: repeat both runs before trusting a difference of a few percent."


def _delta(
    name: str, before: float | None, after: float | None, unit: str, why: str | None = None
) -> str | None:
    if before is None or after is None or before == 0:
        return None
    if _number(before) == _number(after):
        return f"{_number(before)} → {_number(after)}{unit} (unchanged at this precision)"
    percent = round((after / before - 1) * 100)
    worse = percent != 0 and (percent > 0) == (name in _LOWER_IS_BETTER)
    note = ", worse" if worse else ""
    text = f"{_number(before)} → {_number(after)}{unit} ({_change_text(before, after)}{note})"
    return f"{text}; {why}" if worse and why else text


def _known(value: JsonValue | None) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def tpot_note(baseline: RunRecord, candidate: RunRecord) -> str | None:
    """Why TPOT rose when the candidate raised the concurrency cap; None otherwise."""
    before, after = (
        _known((run.settings or {}).get("max_concurrent_requests")) for run in (baseline, candidate)
    )
    old, new = baseline.metrics, candidate.metrics
    if before is None or after is None or old is None or new is None or after <= before:
        return None
    if old.tpot_p95_ms is None or new.tpot_p95_ms is None or new.tpot_p95_ms <= old.tpot_p95_ms:
        return None
    per_step = new.prompt_tokens_per_step
    batch = _known((new.ceilings or {}).get("avg_running_batch"))
    if per_step is not None and batch is not None and per_step >= 4 * batch:
        return (
            f"most of each step is prompt processing (about {per_step:.0f} prompt tokens per step)"
        )
    return "more sequences share every step"


def render_changes(
    comparison: Comparison,
    engine_args: Sequence[str],
    bottlenecks: tuple[str, str],
    caveats: Sequence[str],
    tpot: str | None = None,
) -> list[str]:
    """The short block a changed run opens with: what changed, each headline metric before and
    after (worse ones said so, a worse TPOT with ``tpot`` as the reason), the answers'
    verdict, the bottleneck before and after (when both runs were diagnosed), and why the two
    runs may not be comparable."""
    lines = [f"## What changed vs {_baseline_name(comparison)} (run {comparison.baseline_id})", ""]
    if engine_args:
        lines.append("Change: " + " ".join(f"--engine-arg {arg}" for arg in engine_args))
    for label, name, unit in _CHANGE_ROWS:
        before, after = comparison.speed.get(name, (None, None))
        why = tpot if name == "tpot_p95_ms" else None
        if (text := _delta(name, before, after, unit, why)) is not None:
            lines.append(f"- {label} {text}")
    if note := _engine_rate_note(comparison):
        lines.append(f"- {note}")
    verdict = comparison.outcome.verdict
    lines.append(
        "- answers: no noise floor to judge them against (see the comparison)"
        if verdict.startswith(NO_FLOOR)
        else f"- answers: {verdict}"
    )
    if all(bottlenecks):
        lines.append(f"- bottleneck: {bottlenecks[0]} → {bottlenecks[1]}")
    notes = [*caveats, *([NOISE_NOTE] if comparison.speed else [])]
    return lines + [part for note in notes for part in ("", note)]


def _quality_rows(
    comparison: Comparison, baseline: RunRecord, candidate: RunRecord
) -> list[list[str]]:
    outcome = comparison.outcome
    prompts = outcome.prompts
    rows = [
        [
            "word-overlap similarity",
            _cell(_mean([p.noise for p in prompts]), _decimal, " (its own repeats)"),
            _cell(
                _mean([p.agreement for p in prompts]),
                _decimal,
                f" (to {_baseline_name(comparison)})",
            ),
        ],
        ["answers with the same words", "—", _cell(_mean([p.same for p in prompts]), percent_text)],
    ]
    if any(rate is not None for rate in outcome.reference):
        rows.append(["match with the reference", *(_cell(r, _decimal) for r in outcome.reference)])
    rows += [
        [HEALTH_NAMES[name], *(_cell(rate, percent_text) for rate in rates)]
        for name, rates in outcome.health.items()
    ]
    return [*rows, *_tool_call_rows(baseline, candidate)]


_SEED_ADVICE = (
    "Sampling is random: set a seed in the configuration, run `tensward init` again, then "
    "analyse your current setup again."
)


def _prompts(count: int) -> str:
    return f"{count} prompt" if count == 1 else f"{count} prompts"


def _equality_lines(comparison: Comparison, project: ResolvedProject, advice: str) -> list[str]:
    """Whether every answer is one the baseline gave, why not, and how well the baseline
    reproduces itself, with ``advice`` on making it repeatable when it does not."""
    outcome = comparison.outcome
    reproduced, repeated = outcome.reproduction
    covered, answered = outcome.coverage
    complete = covered == answered
    if outcome.equal:
        same = f"All answers identical to {comparison.baseline_id}"
        lines = [f"{same} (text, tool calls and finish reason)."]
    elif outcome.equal is None:
        lines = ["Not identical: nothing to judge, no prompt was answered."]
    else:
        judged = [p for p in outcome.prompts if p.identical is not None]
        differing = [p for p in judged if p.identical is False]
        problems = []
        if differing:
            where = next((p for p in differing if p.first_difference is not None), None)
            at = (
                f" (first difference at character {where.first_difference} in {where.prompt_id})"
                if where
                else ""
            )
            problems.append(f"{len(differing)} of {len(judged)} prompts differ{at}")
        if unanswered := sum(p.unanswered for p in outcome.prompts):
            problems.append(f"{_prompts(unanswered)} got no answer")
        if failures_rose(outcome.health["failed"]):
            problems.append("the failure rate rose")
        if not complete:
            problems.append("the coverage is incomplete")
        lines = [
            ("Not identical: " if differing else "Equality not shown: ") + "; ".join(problems) + "."
        ]
    calls = [p.calls_identical for p in outcome.prompts if p.calls_identical is not None]
    if calls:
        lines.append(
            f"Tool calls identical for {sum(calls)} of {len(calls)} prompts that call tools."
        )
    if comparison.recorded or not complete:
        verb = (
            "The recorded answers cover"
            if comparison.recorded
            else f"{_baseline_name(comparison).capitalize()} covers"
            if comparison.between_runs
            else "The current-setup run covers"
        )
        needed = "" if complete else ", and every prompt the candidate answered must be covered"
        lines.append(f"{verb} {covered} of {answered} prompts{needed}.")
    if comparison.recorded:
        lines.append("Self-reproduction not applicable: the baseline is recorded answers.")
    elif not repeated:
        lines.append("Self-reproduction not checked: each prompt ran once.")
    else:
        lines.append(
            f"{_baseline_name(comparison).capitalize()} reproduced its own answers for "
            f"{reproduced} of {repeated} prompts."
        )
        if reproduced < repeated:
            unseeded = comparison.temperature > 0 and project.config.workload.seed is None
            lines.append(_SEED_ADVICE if unseeded else advice)
    return lines


def render_section(
    comparison: Comparison,
    candidate: RunRecord,
    baseline: RunRecord,
    project: ResolvedProject,
    *,
    advice: str,
) -> list[str]:
    """The report section that compares ``candidate`` with ``baseline``; ``advice`` says how to
    make the baseline's answers repeatable when it did not reproduce them."""
    outcome = comparison.outcome
    name = _baseline_name(comparison)
    if comparison.between_runs:
        lines = [f"## Run {comparison.candidate_id} compared with {name}", ""]
    else:
        lines = [f"## Compared with {name} (run {comparison.baseline_id})", ""]
    if comparison.differences:
        lines += [
            f"Measured on a different {', '.join(comparison.differences)} than "
            f"{name if comparison.between_runs else 'your current-setup run'}: speed and "
            "answers may differ for that reason alone.",
            "",
        ]
    lines += [f"**{outcome.verdict}**", ""]
    lines += [*_equality_lines(comparison, project, advice), ""]
    if note := _engine_rate_note(comparison):
        lines += [note, ""]
    if comparison.speed:
        lines += table(
            ["", name.removeprefix("your "), "this run", "change"],
            [
                [SPEED_LABELS[name], *(_cell(v, _number) for v in values), _change(*values)]
                for name, values in comparison.speed.items()
            ],
        )
    lines += table(
        ["", name.removeprefix("your "), "this run"], _quality_rows(comparison, baseline, candidate)
    )
    owner = f"{name}'" if name.endswith("s") else f"{name}'s"
    rule = (
        f"A prompt counts as changed when its answers are more than 0.15 less similar to {owner} "
        f"than {owner} are to each other (word overlap, at most 16 "
        "answers per prompt; a single changed number barely moves it, so read the answers "
        f"below). Temperature: {comparison.temperature:g}."
    )
    if comparison.temperature > 0 and project.config.workload.seed is not None:
        rule += (
            " With a fixed seed, repeats of a prompt sample the same answer, so the noise floor "
            "is tight and any engine change that alters sampling shows as changed answers; read "
            "the pairs."
        )
    lines += [rule, ""]
    for prompt in outcome.prompts[:INLINE_PAIRS]:
        before = baseline.answers.get(prompt.prompt_id)
        after = candidate.answers.get(prompt.prompt_id)
        lines += [f"### Prompt `{prompt.prompt_id}`", ""]
        for who, given in ((name.capitalize(), before), ("This run", after)):
            lines += [f"{who}:", "", _show(given[0], INLINE_CHARS) if given else "(no answer)", ""]
    lines.append(f"All prompts, with every distinct answer: `{_directory(comparison)}/answers.md`")
    return lines


def _directory(comparison: Comparison) -> str:
    return f"compare/{comparison.baseline_id}-vs-{comparison.candidate_id}"


def _prompt_display(entry: PromptEntry) -> str:
    """The prompt as written in the workload; an image is named by its path there."""
    if entry.messages is None:
        return entry.prompt or ""

    def text(message: ChatMessage) -> str:
        if isinstance(message.content, str):
            return message.content
        return "\n".join(
            f"[image: {part.image_url.url}]" if isinstance(part, ImagePart) else part.text
            for part in message.content
        )

    return "\n".join(f"{message.role}: {text(message)}" for message in entry.messages)


def render_answers(
    comparison: Comparison, candidate: RunRecord, baseline: RunRecord, project: ResolvedProject
) -> str:
    """Every prompt with its answers from both runs, side by side, for reading."""
    entries = {entry.id: entry for entry in project.prompts}
    name = _baseline_name(comparison)
    lines = [
        f"# Answers: {comparison.baseline_id} ({name}) and run {comparison.candidate_id}",
        "",
        comparison.outcome.verdict,
        "",
    ]
    for prompt in comparison.outcome.prompts:
        if prompt.agreement is None:
            note = "no answer" if prompt.candidate_count == 0 else f"not in {name}"
        elif prompt.noise is None:
            note = f"agreement {prompt.agreement:.2f}, not judged: it ran once"
        else:
            note = f"agreement {prompt.agreement:.2f}, the baseline's own {prompt.noise:.2f}"
        if prompt.first_difference is not None:
            note += f"; first difference at character {prompt.first_difference}"
        lines += [f"## {prompt.prompt_id}: {note}", ""]
        if prompt.prompt_id in entries:
            lines += [_fenced(_trimmed(_prompt_display(entries[prompt.prompt_id]), FILE_CHARS)), ""]
        for who, run in ((name.capitalize(), baseline), ("This run", candidate)):
            distinct = list(dict.fromkeys(run.answers.get(prompt.prompt_id, ())))[:FILE_ANSWERS]
            lines += [f"### {who}", ""]
            lines += [_show(answer, FILE_CHARS) + "\n" for answer in distinct] or [
                "(no answer)",
                "",
            ]
    return "\n".join(lines)


def write_comparison(
    project_dir: Path,
    comparison: Comparison,
    baseline: RunRecord,
    candidate: RunRecord,
    project: ResolvedProject,
) -> Path:
    """Write ``answers.md`` and ``compare.json`` privately below ``project_dir/compare`` and
    return the path of ``answers.md``."""
    directory = project_dir / _directory(comparison)
    for level in (directory.parent, directory):
        level.mkdir(mode=0o700, exist_ok=True)
    write_json(directory / "compare.json", asdict(comparison))
    answers = directory / "answers.md"
    write_private(answers, render_answers(comparison, candidate, baseline, project))
    return answers
