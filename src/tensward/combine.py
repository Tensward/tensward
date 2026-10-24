"""Two to four changes measured in one step: which pairs act on different things, when a change
may join a stack, what a stack or a single change is expected to gain on the goal, and whether a
stack delivered what its numeric members were predicted to."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, fields
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Collection, Literal, Mapping, Sequence

from .context import current as analysis
from .engines.vllm.quantization import PER_CHANNEL
from .history import command_args, recorded_env
from .playbook import GOAL_WORDS, Entry, Suggestion
from .predict import CAP_CHANGES, PRECISION, Prediction
from .rejected import as_change
from .runs import RUN_RECORD

INTERACTIONS_FILE = "data/interactions.toml"
HEADROOM = 0.15  # a partner may worsen a limited metric only this far inside the limit
MODEL_ERROR = 0.05  # the prediction error accepted on validation (5 points)
# the goal's value on a ``Prediction``
PREDICTED = {"throughput": "req_s", "tpot": "tpot", "ttft": "ttft"}
MAX_MEMBERS = 4
LOSSLESS_FIRST = 0.60  # the lossless remainder is offered at this share of the full stack's score
ANSWER_RISK = {"low": 0.10, "medium": 0.25, "high": 0.50}  # the chance the answers are not adopted
PRIOR_SHARE = 0.5  # a prior counts at half its measured gain
NUMERIC = (*PRECISION, *CAP_CHANGES)  # the changes the memory-traffic model gives a number
FP8_WEIGHTS = "fp8-weights"
_PAIR_KEYS = frozenset({"first", "second", "goals", "why", "kv_bound", "answer_pair"})
_OPTIONAL_PAIR_KEYS = frozenset({"kv_bound", "answer_pair"})


@dataclass(frozen=True, slots=True)
class _Pair:
    first: frozenset[str]
    second: frozenset[str]
    goals: frozenset[str]
    why: str
    kv_bound: bool
    answer_pair: bool


@cache
def _pairs() -> tuple[_Pair, ...]:
    text = files("tensward").joinpath(INTERACTIONS_FILE).read_text("utf-8")
    pairs = []
    for table in tomllib.loads(text)["independent"]:
        if not _PAIR_KEYS - _OPTIONAL_PAIR_KEYS <= set(table) <= _PAIR_KEYS:
            raise ValueError(f"interactions: a pair has the keys {sorted(_PAIR_KEYS)}")
        pair = _Pair(
            frozenset(table["first"]), frozenset(table["second"]), frozenset(table["goals"]),
            table["why"], table.get("kv_bound", False), table.get("answer_pair", False),
        )  # fmt: skip
        if unknown := pair.goals - GOAL_WORDS.keys():
            raise ValueError(f"interactions: no goal {sorted(unknown)[0]}")
        pairs.append(pair)
    return tuple(pairs)


def _pair(first: str, second: str, goal: str) -> _Pair | None:
    for pair in _pairs():
        if goal in pair.goals and (
            (first in pair.first and second in pair.second)
            or (second in pair.first and first in pair.second)
        ):
            return pair
    return None


def independent(first: str, second: str, goal: str) -> str | None:
    """Why the two changes act on different things for ``goal``, else None; either order. A
    knowledge record that fired for this run adds a pair or, as a "never" pair, takes one away."""
    matches = analysis().kb_matches
    if never(first, second):
        return None
    pair = _pair(first, second, goal)
    if pair is not None:
        return pair.why
    for record in () if matches is None else matches.independent:
        change = record.change
        if change is None or (change.goals and goal not in change.goals):
            continue
        if (first in change.first and second in change.second) or (
            second in change.first and first in change.second
        ):
            return change.why or record.title
    return None


def never(first: str, second: str) -> bool:
    """Whether a knowledge record that fired for this run says the two are never measured
    together."""
    matches = analysis().kb_matches
    return matches is not None and frozenset((first, second)) in matches.never


def kv_bound_pair(first: str, second: str, goal: str) -> bool:
    """Whether the pair changes answers twice and is measured together only where the pair
    route predicts it: on a KV-bound run, where the memory both free holds more requests, or
    where the bytes alone shorten the step (``answer_pair``)."""
    pair = _pair(first, second, goal)
    return pair is not None and (pair.kv_bound or pair.answer_pair)


def stackable(
    members: Sequence[Suggestion],
    candidate: Suggestion,
    *,
    goal: str,
    headroom: Mapping[str, float],
    tried: Collection[frozenset[str]],
    bundle: bool = False,
    base: str | None = None,
) -> str | None:
    """None when ``candidate`` can join ``members`` in one step for ``goal``, else the one-line
    reason it cannot: it acts on something different from every member, the stack stays within
    MAX_MEMBERS, at most one member changes the answers (the fp8 weights and fp8 KV pair
    excepted), and neither its command nor the union of the commands was measured. A ``bundle``
    holds only changes that cannot change the answers, and any two of them join unless a
    knowledge record says never: the step is kept unless it is worse, and split when it is.
    ``headroom`` is, per metric, how far inside its limit the measured run is; ``tried`` holds
    the engine arguments of the runs already measured, a KV size counted as one change whatever
    its value. An approved ``base`` stays outside MAX_MEMBERS and obeys the same pair rules."""
    name = candidate.entry.name
    if name != base and sum(m.entry.name != base for m in members) >= MAX_MEMBERS:
        return f"a step holds at most {MAX_MEMBERS} changes"
    if bundle and candidate.entry.changes_answers:
        return f"{name} can change the answers"
    if base is not None and name != base:
        if bundle and never(base, name):
            return f"{base} and {name} are never measured together"
        if not bundle and independent(base, name, goal) is None:
            return f"{base} and {name} are not known to act on different things"
    for member in members:
        if bundle and never(member.entry.name, name):
            return f"{member.entry.name} and {name} are never measured together"
        if not bundle and independent(member.entry.name, name, goal) is None:
            return f"{member.entry.name} and {name} are not known to act on different things"
    if (effect := candidate.entry.effects.get(goal)) is None or effect.direction == "worsens":
        return f"{name} would worsen the {GOAL_WORDS.get(goal, goal)}"
    for metric, other in candidate.entry.effects.items():
        if other.direction == "worsens" and headroom.get(metric, -1.0) < HEADROOM:
            return f"{name} worsens {GOAL_WORDS[metric]} with no room under its limit"
    changing = [m.entry.name for m in (*members, candidate) if m.entry.changes_answers]
    ruled = len(changing) == 2 and kv_bound_pair(changing[0], changing[-1], goal)
    if len(changing) > 1 and not ruled:
        return f"{' and '.join(changing)} can all change the answers"
    if any(m.command is None for m in (*members, candidate)):
        return f"{name} has no command"
    each = [as_change(command_args(m.command or "")) for m in (*members, candidate)]
    if any(change in tried for change in each) or frozenset().union(*each) in tried:
        return f"{name} with {', '.join(m.entry.name for m in members)} was already tried"
    return None


def goal_gain(predicted: Prediction | None, goal: str) -> float | None:
    """A prediction's relative gain on ``goal``, positive when better; client TTFT p50 stands in
    for TTFT p95. None without a figure for the goal."""
    figure = getattr(predicted, PREDICTED[goal], None) if goal in PREDICTED else None
    if figure is None:
        return None
    return float(figure) if goal == "throughput" else -float(figure)


SMALL_CHECKPOINT = 2_000_000_000  # parameters under which tool calls suffer most from precision


def checkpoint_parameters() -> int | None:
    """The parameters the checkpoint holds; None when the anatomy does not say."""
    anatomy = analysis().anatomy
    params = anatomy.params if anatomy is not None else None
    return sum(getattr(params, f.name) for f in fields(params)) if params is not None else None


def _small_checkpoint() -> bool:
    """Whether the checkpoint is known to hold under SMALL_CHECKPOINT parameters; not known is not
    small."""
    params = checkpoint_parameters()
    return params is not None and params < SMALL_CHECKPOINT


def answer_risk(entry: Entry, suggestion: Suggestion, *, tools: bool = False) -> str | None:
    """How likely ``entry``'s change, as ``suggestion`` makes it, is to change the answers enough
    not to be adopted: where the requests offer tools, a precision change is high on a checkpoint
    under SMALL_CHECKPOINT parameters and medium above; otherwise fp8 weights per channel are low,
    and a change of the answers with no risk named counts medium."""
    if not entry.changes_answers:
        return None
    if tools and entry.answer_bits is not None:
        return "high" if _small_checkpoint() else "medium"
    settings = suggestion.settings
    if entry.name == FP8_WEIGHTS and settings is not None and settings.quantization == PER_CHANNEL:
        return "low"
    return entry.answer_risk or "medium"


def stack_risk(suggestion: Suggestion, *, tools: bool = False) -> str | None:
    """The highest answer risk of the offer's changes; None when none changes the answers."""
    risks = [
        answer_risk(m, suggestion, tools=tools) for m in suggestion.members or (suggestion.entry,)
    ]
    return max((r for r in risks if r is not None), key=list(ANSWER_RISK).index, default=None)


def lossless_remainder(stack: Suggestion) -> tuple[Entry, ...]:
    """The members of ``stack`` that cannot change the answers."""
    return tuple(m for m in stack.members if not m.changes_answers)


def parent_offer(parent_dir: Path, change: frozenset[str]) -> Mapping[str, Any] | None:
    """The suggestion block of the parent's ``report.json`` whose command adds exactly
    ``change`` (``--engine-arg`` values) to the parent's engine arguments."""
    try:
        run = json.loads((parent_dir / RUN_RECORD).read_text("utf-8"))
        report = json.loads((parent_dir / "report.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None
    args = frozenset(run.get("engine_args", ())) | recorded_env(run)
    for section in report.get("sections", ()):
        for block in section.get("blocks", ()):
            command = block.get("command")
            if block.get("kind") == "suggestion" and command:
                if command_args(command) - args == change:
                    return block  # type: ignore[no-any-return]
    return None


def check(gain: float, band: float, anchor_gain: float) -> Literal["agrees", "short"]:
    """Whether a stack's ``gain`` (positive when better) reaches its numeric members' predicted
    ``anchor_gain`` less the run's noise ``band`` and the model's error."""
    return "short" if gain < anchor_gain - band - MODEL_ERROR else "agrees"
