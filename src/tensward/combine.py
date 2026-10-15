"""Two changes measured in one step: which pairs act on different things, when a pair may be
offered, and whether the combined step delivered what its anchor alone was predicted to."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Collection, Literal, Mapping

from .history import command_args
from .playbook import GOAL_WORDS, Suggestion
from .runs import RUN_RECORD

INTERACTIONS_FILE = "data/interactions.toml"
HEADROOM = 0.15  # a partner may worsen a limited metric only this far inside the limit
MODEL_ERROR = 0.05  # the prediction error accepted on validation (5 points)
# the goal's value on a ``Prediction``
PREDICTED = {"throughput": "req_s", "tpot": "tpot", "ttft": "ttft"}
_PAIR_KEYS = frozenset({"first", "second", "goals", "why", "kv_bound"})


@dataclass(frozen=True, slots=True)
class _Pair:
    first: frozenset[str]
    second: frozenset[str]
    goals: frozenset[str]
    why: str
    kv_bound: bool


@cache
def _pairs() -> tuple[_Pair, ...]:
    text = files("tensward").joinpath(INTERACTIONS_FILE).read_text("utf-8")
    pairs = []
    for table in tomllib.loads(text)["independent"]:
        if not _PAIR_KEYS - {"kv_bound"} <= set(table) <= _PAIR_KEYS:
            raise ValueError(f"interactions: a pair has the keys {sorted(_PAIR_KEYS)}")
        pair = _Pair(
            frozenset(table["first"]), frozenset(table["second"]), frozenset(table["goals"]),
            table["why"], table.get("kv_bound", False),
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
    """Why the two changes act on different things for ``goal``, else None; either order."""
    pair = _pair(first, second, goal)
    return pair.why if pair is not None else None


def kv_bound_pair(first: str, second: str, goal: str) -> bool:
    """Whether the pair is measured together only on a KV-bound run, where the memory both free
    holds more requests; it changes answers twice."""
    pair = _pair(first, second, goal)
    return pair is not None and pair.kv_bound


def pairable(
    anchor: Suggestion,
    partner: Suggestion,
    *,
    goal: str,
    headroom: Mapping[str, float],
    tried: Collection[frozenset[str]],
) -> str | None:
    """None when ``partner`` can be measured together with ``anchor`` for ``goal``, else the
    one-line reason it cannot. ``headroom`` is, per metric, how far inside its limit the
    measured run is; ``tried`` holds the engine arguments of runs already measured, compared
    with each offer's command and their union."""
    first, second = anchor.entry, partner.entry
    if independent(first.name, second.name, goal) is None:
        return f"{first.name} and {second.name} are not known to act on different things"
    value = PREDICTED.get(goal)
    if value is None or getattr(anchor.predicted, value, None) is None:
        return f"{first.name} has no predicted {GOAL_WORDS.get(goal, goal)}"
    if (effect := second.effects.get(goal)) is None or effect.direction == "worsens":
        return f"{second.name} would worsen the {GOAL_WORDS.get(goal, goal)}"
    for metric, other in second.effects.items():
        if other.direction == "worsens" and headroom.get(metric, -1.0) < HEADROOM:
            return f"{second.name} worsens {GOAL_WORDS[metric]} with no room under its limit"
    if (
        first.changes_answers
        and second.changes_answers
        and not kv_bound_pair(first.name, second.name, goal)
    ):
        return f"both {first.name} and {second.name} can change the answers"
    if anchor.command is None or partner.command is None:
        return f"{second.name} has no command"
    one, two = command_args(anchor.command), command_args(partner.command)
    if one in tried or two in tried or one | two in tried:
        return f"{first.name} with {second.name} was already tried"
    return None


def parent_offer(parent_dir: Path, change: frozenset[str]) -> Mapping[str, Any] | None:
    """The suggestion block of the parent's ``report.json`` whose command adds exactly
    ``change`` (``--engine-arg`` values) to the parent's engine arguments."""
    try:
        run = json.loads((parent_dir / RUN_RECORD).read_text("utf-8"))
        report = json.loads((parent_dir / "report.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None
    args = frozenset(run.get("engine_args", ()))
    for section in report.get("sections", ()):
        for block in section.get("blocks", ()):
            command = block.get("command")
            if block.get("kind") == "suggestion" and command:
                if command_args(command) - args == change:
                    return block  # type: ignore[no-any-return]
    return None


def check(gain: float, band: float, anchor_gain: float) -> Literal["agrees", "short"]:
    """Whether a combined step's ``gain`` (positive when better) reaches the anchor's
    predicted ``anchor_gain`` less the run's noise ``band`` and the model's error."""
    return "short" if gain < anchor_gain - band - MODEL_ERROR else "agrees"
