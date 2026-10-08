"""Which playbook entries a run calls for, and the suggestions they become."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from .engines import Engine
from .extensions import load_extender
from .playbook import (
    Entry,
    HeldBack,
    Situation,
    Suggestion,
    admitted,
    blocked_changes,
    held_back,
    ranked,
)
from .settings import Settings


def applicable(
    entries: Sequence[Entry],
    situation: Situation,
    *,
    allow_quality_changes: bool,
    near: bool = False,
    engine: Engine | None = None,
) -> tuple[list[Suggestion], list[tuple[Entry, str]]]:
    """The entries the situation calls for, as suggestions, and those of them a gate rules out
    here, each with the gate's reason. An installed analysis plugin's entries join
    ``entries``. With ``near``, an entry that declares a ``low_bar`` is also offered, as a
    "could_help" suggestion, where its signal is near its gate or a related change is
    blocked. An entry whose ``unblocks`` gives a reason is offered "try_first" for its own
    bottlenecks and those in ``unblocks_for``, whatever ``applies`` said. With ``engine``, an
    installed extension's entries are kept only where the engine realises their levers (and, when
    they name engine families, only for its family), and its hold-backs apply to them."""
    found, gated = [], []
    extender = load_extender()
    facts, settings = situation.facts, situation.settings
    extra: Sequence[Entry] = extender.entries if extender else ()
    if engine is not None:
        capabilities = engine.capabilities(None)
        extra = [
            held_back(entry, engine.hold_backs)
            for entry in extra
            if (not entry.engines or engine.family in entry.engines)
            and admitted(entry, capabilities)
        ]
    for entry in (*entries, *extra):
        if entry.quality_risk and not allow_quality_changes:
            continue
        reason = entry.applies(situation)
        if isinstance(reason, HeldBack):
            gated.append((entry, str(reason)))
            continue
        suggestion = None
        if entry.unblocks and (unblocked := entry.unblocks(situation)):
            if isinstance(unblocked, HeldBack):
                gated.append((entry, str(unblocked)))
                continue
            suggestion = Suggestion(
                entry=entry,
                reason=unblocked,
                tier="try_first",
                addresses=entry.addresses | entry.unblocks_for,
                basis=entry.basis,
            )
        elif reason:
            suggestion = Suggestion(
                entry=entry,
                reason=reason,
                tier="try_first",
                addresses=entry.addresses,
                basis=entry.basis,
            )
        elif near and entry.low_bar and (low := entry.low_bar(situation)):
            if isinstance(low, HeldBack):
                gated.append((entry, str(low)))
                continue
            suggestion = Suggestion(
                entry=entry,
                reason=low,
                tier="could_help",
                addresses=entry.addresses,
                near=entry.near,
                cost=entry.low_bar_costs,
                basis=entry.basis,
            )
        if suggestion is None:
            continue
        blocked = next((gate.reason for gate in entry.gates if gate.blocks(facts, settings)), None)
        if blocked:
            gated.append((entry, blocked))
        else:
            found.append(suggestion)
    return found, gated


@dataclass(frozen=True, slots=True, kw_only=True)
class Suggested:
    suggestions: tuple[Suggestion, ...]  # ranked, tiers final
    not_applicable: tuple[tuple[str, str], ...]  # (entry name, the gate's or the engine's reason)
    blocked: Mapping[str, str]  # per bottleneck, why its change is blocked


def suggestions(
    engine: Engine,
    situation: Situation,
    *,
    current: Settings,
    project_path: Path | None,
    version: str | None = None,
) -> Suggested:
    """The engine's playbook entries for ``situation``, each with the settings it proposes and,
    given ``project_path``, the command that measures them from ``current``. A suggestion
    stays "try_first" only when it addresses a failed gate, the primary bottleneck or a
    secondary one; the others could help."""
    settings, diagnosis = situation.settings, situation.diagnosis
    entries = engine.playbook(version=version)
    found, gated = applicable(
        entries, situation, allow_quality_changes=True, near=True, engine=engine
    )
    not_applicable = [(entry.name, why) for entry, why in gated]
    first = {gate.bottleneck for gate in diagnosis.gates if gate.state == "critical"}
    first.update([*([diagnosis.primary] if diagnosis.primary else []), *diagnosis.secondary])
    offered = []
    for suggestion in ranked(found):
        entry = suggestion.entry
        proposed, note = engine.consistent(settings, entry.apply(situation))
        if why := engine.unstartable(settings, proposed):
            not_applicable.append((entry.name, why))
            continue
        if proposed == settings:
            continue
        command = None
        if project_path is not None and engine.engine_args_between(settings, proposed):
            command = suggestion_command(engine, project_path, current, proposed)
        stays = suggestion.tier == "try_first" and bool(suggestion.addresses & first)
        offered.append(
            replace(
                suggestion,
                reason=f"{suggestion.reason}; the command {note}" if note else suggestion.reason,
                tier="try_first" if stays else "could_help",
                command=command,
                settings=proposed,
            )
        )
    return Suggested(
        suggestions=tuple(offered),
        not_applicable=tuple(not_applicable),
        blocked=blocked_changes(entries, situation),
    )


def suggestion_command(
    engine: Engine, project: Path, current: Settings, suggested: Settings
) -> str:
    """The ``tensward analyse`` command that measures ``suggested``: every change from the current
    setup once, so an override already given is replaced, not repeated."""
    words = ["tensward", "analyse", "--project", str(project)]
    for text in engine.engine_args_between(current, suggested):
        words += ["--engine-arg", text]
    return shlex.join(words)
