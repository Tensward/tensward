"""Did a change move the answers? Each prompt is compared with the baseline's own repeats of it."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from itertools import combinations, product
from statistics import fmean, median_low
from typing import Any, Callable, Mapping, Sequence

from .jsonanswers import parses_as_json

CHANGE_MARGIN = 0.15  # below the baseline's own similarity by more than this: changed
HEALTH_MARGIN = 0.05  # a format problem rate that rises by more than this is worse
MAX_ANSWERS_PER_PROMPT = 16  # per side; bounds the pairwise similarity only
# An answer repeats itself when more than this share of its REPEAT_CHARS-character pieces (case
# and spacing folded) already appeared earlier in it, as in a "be be be" loop. Structured answers
# (JSON records, numbered and HTML lists, tool calls written as text) were checked on synthetic
# examples and stay under it at 24 characters; at 12 they crossed it.
REPEAT_SHARE = 0.5
REPEAT_CHARS = 24
_WORD = re.compile(r"\w+")
_Key = tuple[str, tuple[tuple[str, str], ...] | None, str | None]
Rates = tuple[float | None, float | None]  # (baseline, candidate)


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    tool_calls: tuple[tuple[str, str], ...] | None  # (name, arguments as JSON text)
    finish_reason: str | None
    thinking: bool = False  # no final answer: the model only produced reasoning
    reasoning: str = ""
    tokens: int | None = None  # output tokens, when the run recorded them


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
    ran_out: bool = False  # unanswered because every answer only thought and hit the output cap
    # compared on reasoning: every answer only thought and hit the output cap, so the reasoning
    # is compared up to the shortest one's length, which holds this many tokens (None: unknown)
    reasoned: bool = False
    reasoning_tokens: int | None = None


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
    thought: int = 0  # prompts left out because a side only thought


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


def repeats_itself(text: str) -> bool:
    """Whether most of the text repeats earlier text in it (REPEAT_SHARE)."""
    folded = " ".join(text.lower().split())
    pieces = len(folded) - REPEAT_CHARS + 1
    if pieces < REPEAT_CHARS:
        return False
    seen: set[str] = set()
    again = 0
    for start in range(pieces):
        piece = folded[start : start + REPEAT_CHARS]
        again += piece in seen
        seen.add(piece)
    return again > REPEAT_SHARE * pieces


def broken(answer: Answer) -> bool:
    """An answer that is not one: empty (no text, no tool call) or repeating itself. An answer
    that only thought is neither."""
    if answer.thinking:
        return False
    return (not answer.text and not answer.tool_calls) or repeats_itself(answer.text)


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


def _with_cause(prompt: PromptComparison, candidate: Sequence[Answer]) -> PromptComparison:
    """``prompt`` with ``ran_out`` set when it is unanswered because the candidate only thought
    and each of those answers was cut by the output limit."""
    cut = bool(candidate) and all(a.thinking and a.finish_reason == "length" for a in candidate)
    return replace(prompt, ran_out=True) if prompt.unanswered and cut else prompt


def _cut_thinking(answers: Sequence[Answer]) -> bool:
    return bool(answers) and all(
        a.thinking and a.finish_reason == "length" and a.reasoning for a in answers
    )


def _compare_reasoning(
    prompt_id: str, baseline: Sequence[Answer], candidate: Sequence[Answer]
) -> PromptComparison:
    """Answers that ran out of output tokens while thinking, compared on their reasoning cut to
    the shortest one's length. Only identity is judged: two greedy runs part after a rounding
    flip, and word overlap past that point says nothing about the answer, so a different
    reasoning text is neither within noise nor changed."""
    shortest = min((*baseline, *candidate), key=lambda a: len(a.reasoning))
    length = len(shortest.reasoning)

    def cut(answers: Sequence[Answer]) -> list[Answer]:
        return [Answer(a.reasoning[:length], (), None) for a in answers]

    prompt = _compare_prompt(prompt_id, cut(baseline), cut(candidate))
    return replace(
        prompt, noise=None, changed=False, reasoned=True, reasoning_tokens=shortest.tokens
    )


def answers_pass(answers: Mapping[str, Any]) -> bool:
    """Whether a step block's ``answers`` pass the answer check: some compared, none broken, no
    prompt that only thought left out, every prompt compared on its reasoning identical or found
    the same by the reasoning check for a change it keeps, and the finished prompts at least 95%
    identical or within the earlier run's own repeats."""
    total = answers["total"]
    reasoned = answers.get("reasoning") or {"prompts": 0, "identical": 0}
    finished = total - reasoned["prompts"]
    share = (answers["identical"] - reasoned["identical"]) / finished if finished else 1.0
    within_repeats = bool(answers.get("judged")) and not answers.get("changed")
    return (
        total > 0
        and answers["broken"] == 0
        and not answers.get("thought")
        and (reasoned["identical"] == reasoned["prompts"] or reasoning_same(answers))
        and (share >= 0.95 or within_repeats)
    )


def reasoning_same(answers: Mapping[str, Any] | None) -> bool:
    """Whether the reasoning check found a step's reasoning the same as the current setup's, for
    a change it keeps on its own."""
    checked = (answers or {}).get("reasoning_check") or {}
    return checked.get("verdict") == "same" and checked.get("auto_keep") is True


def reasoning_changed(answers: Mapping[str, Any]) -> bool:
    """Whether the reasoning check found a step's reasoning departs from the current setup's."""
    return (answers.get("reasoning_check") or {}).get("verdict") == "changed"


def reasoning_counts(prompts: Sequence[PromptComparison]) -> dict[str, Any] | None:
    """For the prompts compared on their reasoning: how many, how many are identical, the
    fewest and most tokens compared (None when a run did not record them), and the character
    where each of the others first diverges. None when no prompt was."""
    shown = [p for p in prompts if p.reasoned]
    if not shown:
        return None
    tokens = [p.reasoning_tokens for p in shown if p.reasoning_tokens is not None]
    return {
        "prompts": len(shown),
        "identical": sum(p.identical is True for p in shown),
        "tokens": [min(tokens), max(tokens)] if len(tokens) == len(shown) else None,
        "diverge": sorted(
            p.first_difference
            for p in shown
            if p.identical is False and p.first_difference is not None
        ),
    }


def reasoning_words(found: Mapping[str, Any]) -> str:
    """``reasoning_counts`` in words: on how much reasoning the answers were compared, how many
    are identical up to there and where the others diverge."""
    tokens = found["tokens"]
    if tokens is None:
        on = "their reasoning up to the shortest answer's length"
    elif tokens[0] == tokens[1]:
        on = f"the first {tokens[0]} reasoning tokens"
    else:
        on = f"the first {tokens[0]} to {tokens[1]} reasoning tokens"
    total, same, diverge = found["prompts"], found["identical"], found["diverge"]
    noun = "prompt" if total == 1 else "prompts"
    text = (
        f"Compared on {on}: every answer ran out of output tokens while thinking. "
        f"{same} of {total} {noun} identical up to there"
    )
    if diverge:
        at = (
            f"character {diverge[0]}"
            if len(diverge) == 1
            else f"characters {diverge[0]} to {diverge[-1]}, median {median_low(diverge)}"
        )
        text += f"; the others first diverge at {at}"
    return text


def unanswered_words(prompts: Sequence[PromptComparison]) -> list[str]:
    """How many prompts got no answer, split by cause: ran out of output tokens while thinking,
    or got none."""
    ran = sum(p.ran_out for p in prompts)
    other = sum(p.unanswered for p in prompts) - ran
    return [
        *([f"{ran} ran out of output tokens while thinking"] if ran else []),
        *([f"{other} got no answer"] if other else []),
    ]


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


def _health(
    side: Mapping[str, Sequence[Answer]], json_answers: frozenset[str], thinks: bool
) -> dict[str, float | None]:
    answers = [a for given in side.values() for a in given]
    rates = {
        "cut short": share(sum(a.finish_reason == "length" for a in answers), len(answers)),
        "empty": share(
            sum(not a.text and not a.tool_calls and not a.thinking for a in answers), len(answers)
        ),
        "repeating": share(sum(repeats_itself(a.text) for a in answers), len(answers)),
    }
    if thinks:
        rates["thinking"] = share(sum(a.thinking for a in answers), len(answers))
    if json_answers:
        judged = [a for prompt, given in side.items() if prompt in json_answers for a in given]
        rates["invalid JSON"] = share(sum(not parses_as_json(a.text) for a in judged), len(judged))
    return rates


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
    "thinking": "answers that only thought (no final answer)",
    "repeating": "answers that repeat themselves (empty ones not counted)",
    "invalid JSON": "invalid JSON",
}


NOT_CHECKED = "Not checked: the model only thought"
NO_FLOOR = "No noise floor"
YOUR_CALL = "Your call"
CHANGED = "Answers changed: "


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
        suffix = f" ({', '.join(unanswered_words(prompts))})" if unanswered else ""
        return f"{CHANGED}{wrong} of {asked} prompts{suffix}", True
    if worse:
        return f"{CHANGED}{', '.join(worse)}", True
    if unsure := sum(p.reasoned and p.identical is False for p in prompts):
        return (
            f"{YOUR_CALL}: {unsure} of {asked} prompt{'' if asked == 1 else 's'} ran out of "
            "output tokens while thinking and their reasoning differs; Tensward cannot tell "
            "from the text whether their answers changed.",
            None,
        )
    reasoned = sum(p.reasoned for p in prompts)
    if reasoned and reasoned == asked:
        return (
            "Reasoning identical: every prompt ran out of output tokens while thinking, and each "
            "reasoning text matches the baseline's up to the shorter one",
            False,
        )
    if not has_floor:
        if recorded:
            return (
                f"{NO_FLOOR}: recorded answers have no repeats, so only equality and "
                "agreement are reported.",
                None,
            )
        once = asked - reasoned
        ran = "The one prompt was" if once == 1 else f"Each of the {once} prompts was"
        return (
            f"{NO_FLOOR}: {ran} sent once in your current-setup run. To judge changes, "
            "raise request_count in the configuration to at least twice the number of prompts, "
            "run `tensward init` again, then analyse your current setup and the change.",
            None,
        )
    unjudged = sum(
        bool(p.baseline_count and p.candidate_count and p.noise is None and not p.reasoned)
        for p in prompts
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
    json_answers: frozenset[str],
    failed: Rates = (None, None),
    recorded: bool = False,
    greedy: bool = False,
) -> Outcome:
    """Compare two sets of answers by prompt id. ``failed`` is each side's share of failed
    requests; a share that is None is unknown, and counts as no failures on the candidate side.
    ``recorded`` says the baseline is recorded production answers. With ``greedy`` (both sides
    sampled at temperature 0), a prompt every answer of which ran out of output tokens while
    thinking is compared on its reasoning."""
    thought = {
        prompt_id
        for prompt_id in dict.fromkeys([*baseline, *candidate])
        if all(a.thinking for side in (baseline, candidate) for a in side.get(prompt_id, ()))
        and any(a.thinking for side in (baseline, candidate) for a in side.get(prompt_id, ()))
    }
    reasoned = {
        prompt_id
        for prompt_id in thought
        if greedy
        and _cut_thinking(baseline.get(prompt_id, ()))
        and _cut_thinking(candidate.get(prompt_id, ()))
    }
    thought -= reasoned
    prompts = sorted(
        (
            _compare_reasoning(prompt_id, baseline[prompt_id], candidate[prompt_id])
            if prompt_id in reasoned
            else _with_cause(
                _compare_prompt(
                    prompt_id,
                    [a for a in baseline.get(prompt_id, ()) if not a.thinking],
                    [a for a in candidate.get(prompt_id, ()) if not a.thinking],
                ),
                candidate.get(prompt_id, ()),
            )
            for prompt_id in dict.fromkeys([*baseline, *candidate])
            if prompt_id not in thought
        ),
        key=_order,
    )
    thinks = any(a.thinking for side in (baseline, candidate) for g in side.values() for a in g)
    sides = [_health(side, json_answers, thinks) for side in (baseline, candidate)]
    health: dict[str, Rates] = {"failed": failed}
    finish_known = any(a.finish_reason for given in baseline.values() for a in given)
    health |= {
        name: (sides[0][name], sides[1][name])
        for name in sides[0]
        if name != "cut short" or finish_known
    }
    reference = (_reference(baseline, references), _reference(candidate, references))
    has_floor = any(p.noise is not None for p in prompts)

    repeated = [
        given
        for pid, given in baseline.items()
        if len(given) > 1 and pid not in thought and pid not in reasoned
    ]
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
    if (thought or reasoned) and equal:
        equal = None
    verdict, changed = _verdict(prompts, has_floor, reference, health, recorded)
    if thought and not prompts:
        verdict, changed = NOT_CHECKED, None
    return Outcome(
        tuple(prompts),
        reference,
        health,
        verdict,
        changed,
        equal,
        (reproduced, len(repeated)),
        coverage,
        len(thought),
    )
