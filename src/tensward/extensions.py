"""The hooks by which an installed package extends Tensward: GPU trace analysis in ``analyse``
and extra subcommands. Entry points are checked against ``API_VERSION``.

A package registers an :class:`Extender` under the ``tensward.analysis`` entry-point group.
``analyse --trace`` hands it the parsed trace (``--counters`` also the kernel counters); the
extender returns extra report lines and a result that is kept in ``metrics.json``
(``trace.analysis``) for the extender's own playbook entries, which join the engine's in
:func:`tensward.suggest.applicable`.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from functools import cache
from importlib.metadata import entry_points
from typing import Any, Callable, Sequence

from .calibration import CalibrationProfile
from .measurement import Measurement
from .playbook import Entry
from .profiling.counters import CountersSummary
from .profiling.trace import Trace
from .settings import Settings

API_VERSION = 1
ANALYSIS_GROUP = "tensward.analysis"
COMMANDS_GROUP = "tensward.commands"
DISABLE_ENV = "TENSWARD_DISABLE_PLUGINS"  # set (to anything) to ignore plugins; used by tests


@dataclass(frozen=True, slots=True, kw_only=True)
class Extension:
    sections: Sequence[str]  # report.md lines appended after the trace headline
    result: Any = None  # a dataclass; stored as the trace's ``analysis``


@dataclass(frozen=True, slots=True, kw_only=True)
class Extender:
    api_version: int
    analyse: Callable[[Trace, Measurement, Settings], Extension]
    entries: Sequence[Entry] = ()
    # ``analyse --counters``: interpret the kernel counters (stored as ``counters.analysis``).
    counters: Callable[[CountersSummary, Measurement], Extension] | None = None
    # Calibration profiles this extension provides; ``profile_for`` prefers one naming the GPU.
    profiles: Sequence[CalibrationProfile] = ()


def _declared_version(found: Any) -> object:
    return getattr(found, "API_VERSION", getattr(found, "api_version", None))


def _skip(message: str) -> None:
    print(message, file=sys.stderr)


def _loaded(entry: Any, *, versionless_ok: bool = False) -> Any:
    """The entry point's object, or None (with one line on stderr) when it cannot be loaded by
    this Tensward or is built for another extension API."""
    name = entry.dist.name if entry.dist else entry.value
    try:
        found = entry.load()
    except Exception as error:  # a plugin's own failure must never stop the CLI
        advice = f"; upgrade {name} or tensward" if isinstance(error, ImportError) else ""
        _skip(f"{name} could not be loaded ({type(error).__name__}: {error}){advice}")
        return None
    needed = _declared_version(found)
    if needed is None and versionless_ok:
        return found
    if needed == API_VERSION:
        return found
    if not isinstance(needed, int):
        _skip(f"{name} does not declare an extension API version; upgrade {name}")
    elif needed > API_VERSION:
        _skip(f"{name} needs extension API {needed}; this Tensward has {API_VERSION}; "
              "upgrade tensward")  # fmt: skip
    else:
        _skip(f"{name} was built for extension API {needed}; this Tensward has {API_VERSION}; "
              f"upgrade {name}")  # fmt: skip
    return None


def extend(analysis: Callable[..., Extension], *arguments: Any) -> Extension | None:
    """The plugin's ``analysis(*arguments)``, or None (with one line on stderr) when it raises."""
    try:
        return analysis(*arguments)
    except Exception as error:  # a plugin's own failure must never stop the CLI
        _skip(f"the analysis plugin failed and was skipped ({type(error).__name__}: {error})")
        return None


def load_extender() -> Extender | None:
    """The installed analysis plugin, if any (the first one when several are installed)."""
    if os.environ.get(DISABLE_ENV):
        return None
    return _first_extender()


@cache
def _first_extender() -> Extender | None:
    for entry in entry_points(group=ANALYSIS_GROUP):
        if (found := _loaded(entry)) is not None:
            return found  # type: ignore[no-any-return]
    return None


def load_commands(subcommands: argparse._SubParsersAction) -> None:
    """Let each installed command plugin add its subcommands (none when plugins are disabled)."""
    if os.environ.get(DISABLE_ENV):
        return
    for entry in entry_points(group=COMMANDS_GROUP):
        register = _loaded(entry, versionless_ok=True)
        if register is None:
            continue
        try:
            register(subcommands)
        except Exception as error:  # a plugin's own failure must never stop the CLI
            name = entry.dist.name if entry.dist else entry.value
            _skip(f"{name} could not add its commands ({type(error).__name__}: {error})")
