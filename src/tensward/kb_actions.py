"""The typed binding of a value rule: what an action reads and what it returns."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .fact_values import UNKNOWN
from .settings import Settings

if TYPE_CHECKING:
    from .engines.protocol import Engine
    from .facts import Facts
    from .playbook import Situation

Built = Settings | str  # what an action proposes, or why it makes no offer


@dataclass(frozen=True, slots=True, kw_only=True)
class Inputs:
    """What an action may read: the run's facts and situation, the prompt lengths, the run's
    answers (read on demand) and the engine's concurrency cap."""

    facts: Facts
    situation: Situation
    engine: Engine
    prompt_tokens: tuple[int, ...] = ()
    answers: Callable[[], Sequence[Sequence[str]]] = tuple
    cap: int | None = None

    def fact(self, name: str) -> Any:
        value = self.facts[name]
        return None if value is UNKNOWN else value


@dataclass(frozen=True, slots=True)
class Action:
    """The typed binding of one value rule: ``build`` reads its arguments from the inputs and
    returns the settings with its result written, or the reason it makes no offer. ``writes``
    are the levers its result sets, so a record's own ``sets`` for them only name the change."""

    writes: frozenset[str]
    build: Callable[[Inputs, Settings], Built]
