"""The one hook by which an installed package extends ``analyse`` with GPU trace analysis.

A package registers an :class:`Extender` under the ``tensward.analysis`` entry-point group.
``analyse --trace`` hands it the parsed trace (``--counters`` also the kernel counters); the
extender returns extra report lines and a result that is kept in ``metrics.json``
(``trace.analysis``) for the extender's own playbook entries, which join the engine's in
:func:`tensward.playbook.applicable`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .counters import CountersSummary
    from .engines.protocol import Settings
    from .measurement import Measurement
    from .playbook import Entry
    from .trace import Trace

ANALYSIS_GROUP = "tensward.analysis"
DISABLE_ENV = "TENSWARD_DISABLE_PLUGINS"  # set (to anything) to ignore plugins; used by tests


@dataclass(frozen=True, slots=True)
class Extension:
    sections: list[str]  # report.md lines appended after the trace headline
    result: Any = None  # a dataclass; stored as the trace's ``analysis``


@dataclass(frozen=True, slots=True)
class Extender:
    analyse: Callable[[Trace, Measurement, Settings], Extension]
    entries: tuple[Entry, ...] = ()
    # ``analyse --counters``: interpret the kernel counters (stored as ``counters.analysis``).
    counters: Callable[[CountersSummary, Measurement], Extension] | None = None


def load_extender() -> Extender | None:
    """The installed analysis plugin, if any (the first one when several are installed)."""
    if os.environ.get(DISABLE_ENV):
        return None
    return _first_extender()


@cache
def _first_extender() -> Extender | None:
    return next((entry.load() for entry in entry_points(group=ANALYSIS_GROUP)), None)
