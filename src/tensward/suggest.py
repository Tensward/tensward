"""Which playbook entries a run calls for, and the suggestions they become."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .classify import Bottleneck
from .engines import Engine
from .estimate import ESTIMATORS, PROMOTE_AT, Estimate, RunInputs, lead_bottleneck
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
    estimates: Mapping[str, Estimate] = field(default_factory=dict)  # by entry name
    estimated: Mapping[str, Entry] = field(default_factory=dict)  # the entry each was made for
    lead: Bottleneck | None = None  # the group the promoted suggestion leads in Try first
    underloaded: float | None = None  # a load-limited run's mean requests in flight

    def estimate(self, suggestion: Suggestion) -> Estimate | None:
        """The estimate of ``suggestion``, only when its entry is the one the estimate was made
        for: an extension's entry under a built-in's name gets none."""
        if self.estimated.get(suggestion.entry.name) is not suggestion.entry:
            return None
        return self.estimates.get(suggestion.entry.name)


def suggestions(
    engine: Engine,
    situation: Situation,
    *,
    current: Settings,
    project_path: Path | None,
    version: str | None = None,
    run: RunInputs | None = None,
) -> Suggested:
    """The engine's playbook entries for ``situation``, each with the settings it proposes and,
    given ``project_path``, the command that measures them from ``current``. A suggestion
    stays "try_first" only when it addresses a failed gate, the primary bottleneck or a
    secondary one; the others could help. A load-limited run gets no server change in Try first:
    the test, not the server, held it back."""
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
        if proposed == settings and entry.levers:
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
    blocked = blocked_changes(entries, situation)
    estimates = _estimates(engine, situation, entries, offered, run)
    unmoved = Suggested(
        suggestions=tuple(offered),
        not_applicable=tuple(not_applicable),
        blocked=blocked,
        estimates=estimates,
        estimated={entry.name: entry for entry in entries if entry.name in estimates},
    )
    if diagnosis.primary == "load_limited":
        return replace(
            unmoved,
            suggestions=tuple(replace(s, tier="could_help") for s in offered),
            underloaded=situation.measurement.mean_in_flight,
        )
    promoted, lead = _promoted(_ordered(offered, unmoved.estimate), unmoved.estimate)
    return replace(unmoved, suggestions=tuple(promoted), lead=lead)


def _estimates(
    engine: Engine,
    situation: Situation,
    entries: Sequence[Entry],
    offered: Sequence[Suggestion],
    run: RunInputs | None,
) -> dict[str, Estimate]:
    """The estimates of the offered suggestions whose entry is the engine's own and has an
    estimator; an estimator that raises gives none."""
    for entry in entries:
        if entry.name in ESTIMATORS and entry.quality_risk:
            raise ValueError(f"{entry.name} may change answers, so it cannot have an estimate")
    found: dict[str, Estimate] = {}
    for suggestion in offered:
        estimator = ESTIMATORS.get(suggestion.entry.name)
        own = any(suggestion.entry is entry for entry in entries)
        if estimator is None or not own or suggestion.settings is None:
            continue
        try:
            estimate = estimator(situation, proposed=suggestion.settings, run=run, engine=engine)
        except Exception:  # an estimate is advice: a failing one is left out, never fatal
            estimate = None
        if estimate is None:
            continue
        found[suggestion.entry.name] = estimate
    return found


def _ordered(
    offered: Sequence[Suggestion], estimate: Callable[[Suggestion], Estimate | None]
) -> list[Suggestion]:
    """Within the list, and so within each group of the report: the changes with an estimate
    first, highest low end first; the others after them in playbook order. A change that may
    alter the answers, or a near offer, is never moved by its estimate, and a risky change
    never comes before a safe one."""

    def key(item: tuple[int, Suggestion]) -> tuple[bool, int, float, int]:
        position, suggestion = item
        found = estimate(suggestion)
        movable = found is not None and not suggestion.entry.quality_risk and not suggestion.near
        low = found.low if found is not None and movable else 0.0
        return (suggestion.entry.quality_risk, 0 if movable else 1, -low, position)

    return [suggestion for _, suggestion in sorted(enumerate(offered), key=key)]


def _promoted(
    offered: Sequence[Suggestion], estimate: Callable[[Suggestion], Estimate | None]
) -> tuple[list[Suggestion], Bottleneck | None]:
    """The suggestions with the one whose estimate has the highest low end of at least
    PROMOTE_AT moved to the front of Try first, under the first bottleneck it relieves, and
    that bottleneck; unchanged and None when none qualifies. A change that may alter the
    answers, or a near offer, never moves."""
    qualifying = [
        (found, suggestion)
        for suggestion in offered
        if (found := estimate(suggestion)) is not None
        and found.low >= PROMOTE_AT
        and found.relieves
        and not suggestion.entry.quality_risk
        and not suggestion.near
    ]
    if not qualifying:
        return list(offered), None
    best_estimate, best = max(qualifying, key=lambda pair: pair[0].low)
    lead = lead_bottleneck(best_estimate)
    moved = replace(best, tier="try_first", addresses=best.addresses | {lead})
    return [moved, *(suggestion for suggestion in offered if suggestion is not best)], lead


def suggestion_command(
    engine: Engine, project: Path, current: Settings, suggested: Settings
) -> str:
    """The ``tensward analyse`` command that measures ``suggested``: every change from the current
    setup once, so an override already given is replaced, not repeated."""
    words = ["tensward", "analyse", "--project", str(project)]
    for text in engine.engine_args_between(current, suggested):
        words += ["--engine-arg", text]
    return shlex.join(words)
