"""Did a change move the answers? Each prompt is compared with the baseline's own repeats of it."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations, product
from statistics import fmean
from typing import Callable, Mapping, Sequence

CHANGE_MARGIN = 0.15  # below the baseline's own similarity by more than this: changed
HEALTH_MARGIN = 0.05  # a format problem rate that rises by more than this is worse
MAX_ANSWERS_PER_PROMPT = 16  # per side; bounds the pairwise similarity only
_WORD = re.compile(r"\w+")
_Key = tuple[str, tuple[tuple[str, str], ...] | None, str | None]
Rates = tuple[float | None, float | None]  # (baseline, candidate)


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    tool_calls: tuple[tuple[str, str], ...] | None  # (name, arguments as JSON text)
    finish_reason: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptComparison:
    prompt_id: str
    agreement: float | None
    noise: float | None
    same: float | None
    changed: bool
    unanswered: bool
    baseline_count: int
    candidate_count: int
    identical: bool | None
    first_difference: int | None
    calls_identical: bool | None = None  # None: no tool call on either side to compare


@dataclass(frozen=True, slots=True)
class Outcome:
    prompts: tuple[PromptComparison, ...]
    reference: Rates
    health: dict[str, Rates]
    verdict: str
    changed: bool | None  # None: no noise floor to judge against
    equal: bool | None
    reproduction: tuple[int, int]  # prompts whose baseline answers agree, prompts with repeats
    coverage: tuple[int, int]  # prompts answered by both sides, prompts the candidate answered


def canonical_arguments(arguments: str) -> str:
    """Arguments as key-sorted compact JSON, so key order and spacing are not a change."""
    try:
        return json.dumps(json.loads(arguments), sort_keys=True, separators=(",", ":"))
    except ValueError:
        return arguments


def _calls(answer: Answer) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, canonical_arguments(arguments)) for name, arguments in answer.tool_calls or ()
    )


def _words(answer: Answer) -> Counter[str]:
    return Counter(_WORD.findall(answer.text.lower()))


def _f1(a: Counter[str], b: Counter[str]) -> float:
    total = sum(a.values()) + sum(b.values())
    return 1.0 if total == 0 else 2 * sum((a & b).values()) / total


def _pair(a: Answer, words_a: Counter[str], b: Answer, words_b: Counter[str]) -> float:
    """1 for the same tool calls or the same words, 0 for nothing in common: word-overlap F1."""
    if a.tool_calls or b.tool_calls:
        return float(_calls(a) == _calls(b))
    return _f1(words_a, words_b)


def _normal(answer: Answer) -> tuple[str, tuple[tuple[str, str], ...]]:
    return " ".join(answer.text.lower().split()), _calls(answer)


def _identity(baseline: Sequence[Answer]) -> Callable[[Answer], _Key]:
    """What makes two answers the same: the text, plus the tool calls and the finish reason when
    every baseline answer to the prompt has them (recorded answers may not)."""
    calls = all(a.tool_calls is not None for a in baseline)
    finish = all(a.finish_reason is not None for a in baseline)
    return lambda a: (a.text, _calls(a) if calls else None, a.finish_reason if finish else None)


def _render(key: _Key) -> str:
    text, calls, finish = key
    lines = [text, *(f"{name}({arguments})" for name, arguments in calls or ())]
    return "\n".join(lines if finish is None else [*lines, f"finish:{finish}"])


def _first_difference(a: str, b: str) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def _exact(
    baseline: Sequence[Answer], candidate: Sequence[Answer]
) -> tuple[bool | None, int | None]:
    """Whether every candidate answer is one of the baseline's, and where the first one that is
    not departs from the baseline's first answer. Reads every answer: no sample."""
    if not baseline or not candidate:
        return None, None
    identity = _identity(baseline)
    known = {identity(answer) for answer in baseline}
    stray = next((answer for answer in candidate if identity(answer) not in known), None)
    if stray is None:
        return True, None
    return False, _first_difference(_render(identity(stray)), _render(identity(baseline[0])))


def _calls_identical(baseline: Sequence[Answer], candidate: Sequence[Answer]) -> bool | None:
    """Whether every candidate answer makes the same tool calls (names and arguments) as one of
    the baseline's; None when no answer on either side calls a tool or either side did not
    record calls."""
    if not baseline or not candidate:
        return None
    if any(a.tool_calls is None for a in (*baseline, *candidate)):
        return None
    if not any(a.tool_calls for a in (*baseline, *candidate)):
        return None
    known = {_calls(a) for a in baseline}
    return all(_calls(a) in known for a in candidate)


def _compare_prompt(
    prompt_id: str, baseline: Sequence[Answer], candidate: Sequence[Answer]
) -> PromptComparison:
    identical, first_difference = _exact(baseline, candidate)
    calls = _calls_identical(baseline, candidate)
    if not baseline or not candidate:
        return PromptComparison(
            prompt_id=prompt_id,
            agreement=None,
            noise=None,
            same=None,
            changed=False,
            unanswered=bool(baseline),
            baseline_count=len(baseline),
            candidate_count=len(candidate),
            identical=identical,
            first_difference=first_difference,
        )
    base = [(a, _words(a)) for a in baseline[:MAX_ANSWERS_PER_PROMPT]]
    cand = [(a, _words(a)) for a in candidate[:MAX_ANSWERS_PER_PROMPT]]
    noise = fmean(_pair(*x, *y) for x, y in combinations(base, 2)) if len(base) > 1 else None
    agreement = fmean(_pair(*x, *y) for x, y in product(base, cand))
    same = fmean(_normal(x) == _normal(y) for (x, _), (y, _) in product(base, cand))
    changed = noise is not None and agreement < noise - CHANGE_MARGIN
    return PromptComparison(
        prompt_id=prompt_id,
        agreement=agreement,
        noise=noise,
        same=same,
        changed=changed,
        unanswered=False,
        baseline_count=len(baseline),
        candidate_count=len(candidate),
        identical=identical,
        first_difference=first_difference,
        calls_identical=calls,
    )


def _order(prompt: PromptComparison) -> tuple[int, float]:
    """Unanswered prompts first, then the ones whose words differ, least agreeing first."""
    if prompt.unanswered:
        return 0, 0.0
    if prompt.same is not None and prompt.same < 1:
        return 1, prompt.agreement or 0.0
    return 2, 0.0


def failures_rose(failed: Rates) -> bool:
    return (failed[1] or 0) > (failed[0] or 0)


def share(count: int, total: int) -> float | None:
    return count / total if total else None


def _health(answers: Sequence[Answer], structured: bool) -> dict[str, float | None]:
    def invalid(answer: Answer) -> bool:
        try:
            json.loads(answer.text)
        except ValueError:
            return True
        return False

    rates = {
        "cut short": sum(a.finish_reason == "length" for a in answers),
        "empty": sum(not a.text and not a.tool_calls for a in answers),
    }
    if structured:
        rates["invalid JSON"] = sum(invalid(a) for a in answers)
    return {name: share(count, len(answers)) for name, count in rates.items()}


def _reference(
    answers: Mapping[str, Sequence[Answer]], references: Mapping[str, str]
) -> float | None:
    scores = [
        _f1(_words(answer), _words(Answer(references[prompt_id], None, None)))
        for prompt_id, given in answers.items()
        if prompt_id in references
        for answer in given
    ]
    return fmean(scores) if scores else None


def percent_text(rate: float) -> str:
    return f"{rate:.0%}"


HEALTH_NAMES = {
    "failed": "failed requests",
    "cut short": "answers cut short",
    "empty": "empty answers",
    "invalid JSON": "invalid JSON",
}


NO_FLOOR = "No noise floor"


def _all_empty(health: Mapping[str, Rates]) -> list[str]:
    """The sides whose every answer is empty."""
    rates = health.get("empty", (None, None))
    return [name for name, rate in zip(("the baseline's", "this run's"), rates) if rate == 1.0]


def _verdict(
    prompts: Sequence[PromptComparison],
    has_floor: bool,
    reference: Rates,
    health: Mapping[str, Rates],
    recorded: bool,
) -> tuple[str, bool | None]:
    if empty := _all_empty(health):
        return f"Answers were empty: every one of {' and '.join(empty)} answers is empty", None
    asked = sum(p.baseline_count > 0 for p in prompts)
    unanswered = sum(p.unanswered for p in prompts)
    wrong = sum(p.changed for p in prompts) + unanswered
    worse = [
        f"{HEALTH_NAMES[name]} {percent_text(before)} → {percent_text(after)}"
        for name, (before, after) in health.items()
        if before is not None and after is not None and after - before > HEALTH_MARGIN
    ]
    if reference[0] is not None and reference[1] is not None:
        if reference[1] < reference[0] - CHANGE_MARGIN:
            worse.append(
                f"reference match {percent_text(reference[0])} → {percent_text(reference[1])}"
            )
    if wrong:
        suffix = f" ({unanswered} got no answer)" if unanswered else ""
        return f"Answers changed: {wrong} of {asked} prompts{suffix}", True
    if worse:
        return f"Answers changed: {', '.join(worse)}", True
    if not has_floor:
        if recorded:
            return (
                f"{NO_FLOOR}: recorded answers have no repeats, so only equality and "
                "agreement are reported.",
                None,
            )
        ran = (
            "The one prompt that ran ran"
            if asked == 1
            else f"Each of the {asked} prompts that ran ran"
        )
        return (
            f"{NO_FLOOR}: {ran} once in your current-setup run. To judge changes, "
            "raise request_count in the configuration to at least twice the number of prompts, "
            "run `tensward init` again, then analyse your current setup and the change.",
            None,
        )
    unjudged = sum(
        bool(p.baseline_count and p.candidate_count and p.noise is None) for p in prompts
    )
    if not unjudged:
        return "Answers unchanged within noise", False
    noun = "1 prompt ran once and was" if unjudged == 1 else f"{unjudged} prompts ran once and were"
    return f"Answers unchanged within noise ({noun} not judged)", False


def compare_answers(
    baseline: Mapping[str, Sequence[Answer]],
    candidate: Mapping[str, Sequence[Answer]],
    references: Mapping[str, str],
    *,
    structured: bool,
    failed: Rates = (None, None),
    recorded: bool = False,
) -> Outcome:
    """Compare two sets of answers by prompt id. ``failed`` is each side's share of failed
    requests; a share that is None is unknown, and counts as no failures on the candidate side.
    ``recorded`` says the baseline is recorded production answers."""
    prompts = sorted(
        (
            _compare_prompt(prompt_id, baseline.get(prompt_id, ()), candidate.get(prompt_id, ()))
            for prompt_id in dict.fromkeys([*baseline, *candidate])
        ),
        key=_order,
    )
    sides = [
        _health([a for given in side.values() for a in given], structured)
        for side in (baseline, candidate)
    ]
    health: dict[str, Rates] = {"failed": failed}
    finish_known = any(a.finish_reason for given in baseline.values() for a in given)
    health |= {
        name: (sides[0][name], sides[1][name])
        for name in sides[0]
        if name != "cut short" or finish_known
    }
    reference = (_reference(baseline, references), _reference(candidate, references))
    has_floor = any(p.noise is not None for p in prompts)

    repeated = [given for given in baseline.values() if len(given) > 1]
    reproduced = sum(len({_identity(given)(a) for a in given}) == 1 for given in repeated)
    coverage = (
        sum(p.baseline_count > 0 and p.candidate_count > 0 for p in prompts),
        sum(p.candidate_count > 0 for p in prompts),
    )
    if not any(p.baseline_count or p.candidate_count for p in prompts) or _all_empty(health):
        equal = None
    else:
        equal = (
            all(p.identical is not False and not p.unanswered for p in prompts)
            and coverage[0] == coverage[1]
            and not failures_rose(failed)
        )
    verdict, changed = _verdict(prompts, has_floor, reference, health, recorded)
    return Outcome(
        tuple(prompts),
        reference,
        health,
        verdict,
        changed,
        equal,
        (reproduced, len(repeated)),
        coverage,
    )
