"""Which playbook entries a run calls for, and the suggestions they become."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .classify import Bottleneck, Diagnosis
from .context import GOALS, Goal
from .context import current as analysis
from .engines import Engine
from .estimate import ESTIMATORS, PROMOTE_AT, Estimate, RunInputs, lead_bottleneck
from .extensions import load_extender
from .history import DOUBLED_RAISE, SIZED_RAISE, Memory, guided
from .measurement import Measurement
from .playbook import (
    GOAL_WORDS,
    RELEVANT,
    Entry,
    HeldBack,
    Situation,
    Suggestion,
    admitted,
    blocked_changes,
    goal_bottleneck,
    held_back,
    ranked,
)
from .playbook_common import lowers_concurrency_for_tpot, with_cap_for_more_memory
from .settings import Settings


def applicable(
    entries: Sequence[Entry],
    situation: Situation,
    *,
    allow_quality_changes: bool,
    near: bool = False,
    engine: Engine | None = None,
    opt_in: bool = False,
) -> tuple[list[Suggestion], list[tuple[Entry, str]]]:
    """The entries the situation calls for, as suggestions, and those of them a gate rules out
    here, each with the gate's reason. An installed analysis plugin's entries join
    ``entries``. With ``near``, an entry that declares a ``low_bar`` is also offered, as a
    "could_help" suggestion, where its signal is near its gate or a related change is
    blocked. An entry whose ``unblocks`` gives a reason is offered "try_first" for its own
    bottlenecks and those in ``unblocks_for``, whatever ``applies`` said. With ``engine``, an
    installed extension's entries are kept only where the engine realises their levers (and, when
    they name engine families, only for its family), and its hold-backs apply to them. An
    ``opt_in`` entry is considered only with ``opt_in``."""
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
        if entry.opt_in and not opt_in:
            continue
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
    goal: Goal = "throughput"
    other_goals: tuple[str, ...] = ()  # the changes left out because they worsen the goal
    by_list: bool = False  # Try first's groups follow the order of the list
    memory: Memory | None = None  # what the project's earlier runs say
    stop: str | None = None  # why Try first has no change, and whether more is possible
    plan: tuple[Mapping[str, Any], ...] = ()  # predicted steps after the top change
    plan_lines: tuple[str, ...] = ()
    check: Mapping[str, Any] | None = None  # the parent run's prediction against this run
    warm_kv: tuple[str, float] | None = None  # (run id, tokens) of a warm start of these settings
    test_time: str | None = None  # what testing one change takes here
    test_times: Mapping[str, str] = field(default_factory=dict)  # cap-stretched, by entry
    trade: Mapping[str, Any] | None = None  # under a latency goal, the measured trade so far
    neutral: frozenset[str] = frozenset()  # changes that leave a latency goal alone (_neutral)

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
    secondary one, or, under a latency goal, improves that goal; the others could help. A step
    budget lowered below what vLLM accepts is not applicable. The analysis's goal orders the
    list (``ordered_for_goal``), and only the throughput goal gets estimates. A load-limited
    run gets no server change in Try first: the test, not the server, held it back."""
    settings, diagnosis = situation.settings, situation.diagnosis
    context = analysis()
    goal = context.goal
    entries = engine.playbook(version=version)
    found, gated = applicable(
        entries, situation, allow_quality_changes=True, near=True, engine=engine, opt_in=True
    )
    not_applicable = [(entry.name, why) for entry, why in gated]
    first = {gate.bottleneck for gate in diagnosis.gates if gate.state == "critical"}
    first.update([*([diagnosis.primary] if diagnosis.primary else []), *diagnosis.secondary])
    offered = []
    tpot_limit = lowers_concurrency_for_tpot(situation)
    for suggestion in ranked(found):
        entry = suggestion.entry
        applied = entry.apply(situation)
        raised = with_cap_for_more_memory(situation, applied, engine)
        if (cap := raised.max_concurrent_requests) != applied.max_concurrent_requests:
            suggestion = replace(suggestion, reason=(
                f"{suggestion.reason}; sized in KV tokens, the larger pool would hold about {cap} "
                f"requests, so the command also sets max_concurrent_requests {cap}"
            ))  # fmt: skip
        proposed, note = engine.consistent(settings, raised)
        if why := engine.unstartable(settings, proposed):
            not_applicable.append((entry.name, why))
            continue
        now = settings.prefill_batch_tokens or situation.measurement.step_budget_tokens
        budget = proposed.prefill_batch_tokens
        lowered = budget is not None and now is not None and budget < now
        if lowered and (why := budget_refusal(engine, proposed, situation.measurement, version)):
            not_applicable.append((entry.name, why))
            continue
        if proposed == settings and entry.levers:
            continue
        command = None
        if project_path is not None and _changes(engine, current, settings, proposed):
            command = suggestion_command(engine, project_path, current, proposed)
        if entry.name == "lower-concurrency" and tpot_limit is not None:
            suggestion = replace(
                suggestion,
                addresses=suggestion.addresses | {tpot_limit},
                cost="throughput falls as fewer sequences run in each step",
            )
        own_signal = suggestion.tier == "try_first"
        stays = own_signal and (
            bool(suggestion.addresses & first)
            or (goal != "throughput" and _improves(suggestion, goal))
        )
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
    estimates = _estimates(engine, situation, entries, offered, run) if goal == "throughput" else {}
    unmoved = Suggested(
        suggestions=tuple(offered),
        not_applicable=tuple(not_applicable),
        blocked=blocked,
        estimates=estimates,
        estimated={entry.name: entry for entry in entries if entry.name in estimates},
        goal=goal,
    )
    if diagnosis.primary == "load_limited":
        underloaded = replace(
            unmoved,
            suggestions=tuple(replace(s, tier="could_help") for s in offered),
            underloaded=situation.measurement.mean_in_flight,
        )
        return guided(
            underloaded, situation, engine=engine, entries=entries, project_path=project_path
        )
    ranked_offers, other_goals, lead, by_list = ordered_for_goal(
        offered, goal, situation.measurement, context.ttft_p95_limit_ms, unmoved.estimate
    )
    ordered = replace(
        unmoved,
        suggestions=tuple(ranked_offers),
        lead=lead,
        other_goals=other_goals,
        by_list=by_list,
        neutral=_neutral(offered, goal, situation.measurement, context.ttft_p95_limit_ms),
    )
    return guided(ordered, situation, engine=engine, entries=entries, project_path=project_path)


GATES = frozenset({"fit", "quality"})
STEP_LEVER = "prefill_batch_tokens"
STEP_ENTRY = {
    "throughput": "raise-prefill-batch",
    "ttft": "small-step-budget",
    "tpot": "lower-prefill-batch",
}


def by_goal(
    offered: Sequence[Suggestion], goal: Goal, measurement: Measurement, ttft_limit: float | None
) -> tuple[list[Suggestion], tuple[str, ...]]:
    """``offered`` ranked for ``goal`` by each change's effects, and the names left out because
    they worsen it. A change that improves the goal keeps its place (under a latency goal its
    other worsened metrics lead its costs as a trade-off); one with no effect on it moves to
    Could help; a change for a gate, or without effects (an extension's), is kept as it is.
    With a TTFT limit this run broke, a change that worsens TTFT moves to Could help under any
    goal; within the limit, under the throughput goal, it says it may break it. The order that
    follows from a broken limit is ``ordered_for_goal``'s."""
    kept, later, left_out = [], [], []
    p95 = measurement.ttft_p95_ms
    for suggestion in offered:
        effects = suggestion.entry.effects
        effect = effects.get(goal)
        if effect is None or suggestion.addresses & GATES:
            kept.append(suggestion)
            continue
        costs = suggestion.cost or suggestion.entry.costs
        if effect.direction == "worsens":
            left_out.append(suggestion.entry.name)
            continue
        if effect.direction == "none":
            helps = [GOAL_WORDS[m] for m in GOALS if effects[m].direction == "improves"]
            text = f"does not change {GOAL_WORDS[goal]} ({effect.reason})"
            text += f"; helps {' and '.join(helps)}" if helps else ""
            later.append(replace(suggestion, tier="could_help", cost=f"{text}; {costs}"))
            continue
        worse = [m for m in GOALS if m != goal and effects[m].direction == "worsens"]
        if ttft_limit is not None and "ttft" in worse and p95 is not None and p95 > ttft_limit:
            note = f"TTFT p95 is {p95:.0f} ms, past your limit of {ttft_limit:g} ms"
            later.append(replace(suggestion, tier="could_help",
                                 cost=f"{note}, and this change raises it; {costs}"))  # fmt: skip
            continue
        if goal != "throughput" and worse:
            trade = "; ".join(f"{GOAL_WORDS[m]} worsens ({effects[m].reason})" for m in worse)
            suggestion = replace(suggestion, cost=f"trade-off: {trade}; {costs}")
        elif ttft_limit is not None and "ttft" in worse and p95 is not None:
            note = f"TTFT p95 is {p95:.0f} ms; may push it past your limit of {ttft_limit:g} ms"
            suggestion = replace(suggestion, cost=f"{note}; {costs}")
        kept.append(suggestion)
    return [*kept, *later], tuple(left_out)


def _neutral(
    offered: Sequence[Suggestion], goal: Goal, measurement: Measurement, ttft_limit: float | None
) -> frozenset[str]:
    """Under a latency goal, the changes a crossed class called for that leave the goal's metric
    alone and improve another: ``by_goal`` puts them under Could help, and ``guided`` moves them
    to Try first when no change for the goal is left. A change that raises TTFT past a limit
    this run broke is not one."""
    p95 = measurement.ttft_p95_ms
    broken = ttft_limit is not None and p95 is not None and p95 > ttft_limit
    return frozenset(
        s.entry.name for s in offered
        if goal != "throughput" and s.tier == "try_first" and not s.addresses & GATES
        and (effect := s.entry.effects.get(goal)) is not None and effect.direction == "none"
        and any(_improves(s, metric) for metric in GOALS if metric != goal)
        and not (broken and _worsens(s, "ttft"))
    )  # fmt: skip


def _one_step_budget(ranked: list[Suggestion], goal: Goal) -> list[Suggestion]:
    """At most one change of the step budget in Try first: the goal's own direction."""
    own = STEP_ENTRY[goal]
    moving = [s for s in ranked if s.tier == "try_first" and STEP_LEVER in s.entry.levers]
    keep = next((s for s in moving if s.entry.name == own), moving[0] if moving else None)
    return [replace(s, tier="could_help") if s in moving and s is not keep else s for s in ranked]


def _one_cap_raise(ranked: list[Suggestion]) -> list[Suggestion]:
    """At most one raise of the concurrency cap in Try first: the cap sized in KV tokens over
    the doubling rule's."""
    sized = any(s.tier == "try_first" and s.entry.name == SIZED_RAISE for s in ranked)
    return [
        replace(s, tier="could_help")
        if sized and s.tier == "try_first" and s.entry.name == DOUBLED_RAISE
        else s
        for s in ranked
    ]


def _improves(suggestion: Suggestion, metric: str) -> bool:
    effect = suggestion.entry.effects.get(metric)
    return effect is not None and effect.direction == "improves"


def _worsens(suggestion: Suggestion, metric: str) -> bool:
    effect = suggestion.entry.effects.get(metric)
    return effect is not None and effect.direction == "worsens"


def ordered_for_goal(
    offered: Sequence[Suggestion],
    goal: Goal,
    measurement: Measurement,
    ttft_limit: float | None,
    estimate: Callable[[Suggestion], Estimate | None],
) -> tuple[list[Suggestion], tuple[str, ...], Bottleneck | None, bool]:
    """The suggestions in their final order, the names left out for the goal, the bottleneck a
    promoted change leads, and whether the report renders Try first in this list's order. The
    effect ranking comes first, then (throughput only) the estimate order and promotion, then
    the TTFT-limit rule: with the limit already broken, no TTFT-worsening change is promoted
    and the changes that lower TTFT move to the front."""
    ranked_offers, left_out = by_goal(offered, goal, measurement, ttft_limit)
    ranked_offers = _one_cap_raise(_one_step_budget(ranked_offers, goal))
    p95 = measurement.ttft_p95_ms
    broken = (
        goal == "throughput" and ttft_limit is not None and p95 is not None and p95 > ttft_limit
    )
    lead = None
    if goal == "throughput":
        ranked_offers = _ordered(ranked_offers, estimate)
        candidates = [s for s in ranked_offers if not broken or not _worsens(s, "ttft")]
        promoted, lead = _promoted(candidates, estimate)
        ranked_offers = [*promoted, *(s for s in ranked_offers if s not in candidates)]
    if broken:
        ranked_offers.sort(key=lambda s: not _improves(s, "ttft"))
    return ranked_offers, left_out, lead, goal != "throughput" or broken


def budget_refusal(
    engine: Engine, proposed: Settings, measurement: Measurement, version: str | None
) -> str | None:
    """Why a proposed step budget would not start: vLLM refuses a budget below the concurrency
    cap, which is the setting, else the engine's default for this version and GPU."""
    budget = proposed.prefill_batch_tokens
    if budget is None:
        return None
    gpu = measurement.ceilings.gpu if measurement.ceilings is not None else None
    cap = proposed.max_concurrent_requests or engine.default_concurrency(
        version, gpu, measurement.gpu_memory_gib
    )
    if cap is None:
        return (
            f"a step budget of {budget} tokens needs a concurrency cap at most that size, and the "
            "engine's own cap is not known here"
        )
    if budget < cap:
        return (
            f"a step budget of {budget} tokens is below the concurrency cap of {cap}: "
            "vLLM refuses it"
        )
    return None


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


def _changes(engine: Engine, current: Settings, settings: Settings, proposed: Settings) -> bool:
    """Whether a command from ``current`` can measure ``proposed`` against this run's
    ``settings``: the engine arguments between them say so, or (a flag removed, which no
    argument says) the arguments from ``current`` differ."""
    return bool(engine.engine_args_between(settings, proposed)) or (
        proposed != settings
        and set(engine.engine_args_between(current, proposed))
        != set(engine.engine_args_between(current, settings))
    )


def suggestion_command(
    engine: Engine, project: Path, current: Settings, suggested: Settings
) -> str:
    """The ``tensward analyse`` command that measures ``suggested``: every change from the current
    setup once, so an override already given is replaced, not repeated."""
    words = ["tensward", "analyse", "--project", str(project)]
    for text in engine.engine_args_between(current, suggested):
        words += ["--engine-arg", text]
    return shlex.join([*words, *analysis().flags])


def _crossed(diagnosis: Diagnosis) -> list[Bottleneck]:
    """The failed gates, then the primary and secondary bottlenecks."""
    gates = [g.bottleneck for g in diagnosis.gates if g.state == "critical"]
    return [*gates, *([diagnosis.primary] if diagnosis.primary else []), *diagnosis.secondary]


def _home(diagnosis: Diagnosis, suggested: Suggested, suggestion: Suggestion) -> str | None:
    """The group a change of a ``by_list`` Try first opens: the first crossed class it addresses;
    for one that addresses none, the latency goal it is offered for, else its first class."""
    crossed = next((b for b in _crossed(diagnosis) if b in suggestion.addresses), None)
    if crossed is not None or suggested.goal == "throughput":
        return crossed or min(suggestion.addresses, default=None)
    return suggested.goal


def group_members(diagnosis: Diagnosis, suggested: Suggested, group: str) -> list[Suggestion]:
    """The Try-first changes under ``group``: a bottleneck's are those addressing it; a latency
    goal's, those that address no crossed class."""
    grouped = [s for s in suggested.suggestions if s.tier == "try_first"]
    if group in GOAL_WORDS:
        return [s for s in grouped if _home(diagnosis, suggested, s) == group]
    return [s for s in grouped if group in s.addresses]


def try_first_order(diagnosis: Diagnosis, suggested: Suggested) -> list[str]:
    """The groups of Try first in the order the report shows them: failed gates, then, when
    ``suggested.by_list``, each group in the order of its first change in the list, then what
    holds the goal back and the other crossed classes that matter to it (so a group with no
    change still says so); otherwise the group a promoted change leads, then the primary and
    secondary bottlenecks."""
    grouped = [s for s in suggested.suggestions if s.tier == "try_first"]
    addressed = {name for s in grouped for name in s.addresses}
    order: list[str] = [
        g.bottleneck for g in diagnosis.gates if g.state == "critical" and g.bottleneck in addressed
    ]
    diagnosed = [*([diagnosis.primary] if diagnosis.primary else []), *diagnosis.secondary]
    if suggested.by_list:
        for s in grouped:
            group = _home(diagnosis, suggested, s)
            if group is not None and group not in order:
                order.append(group)
        goal = suggested.goal
        named = goal_bottleneck(diagnosis, goal)
        held = [*([named] if named else []), *(b for b in diagnosed if b in RELEVANT.get(goal, ()))]
        order += [b for b in dict.fromkeys(held) if b not in (*order, "load_limited")]
        return order
    if suggested.lead is not None and suggested.lead not in order:
        order.append(suggested.lead)
    order += [b for b in diagnosed if b not in (suggested.lead, "load_limited")]
    return order


def first_try(diagnosis: Diagnosis, suggested: Suggested) -> Suggestion | None:
    """The change the report shows first under Try first: a revert after a loss, else the first
    change of its first group; None when it shows none."""
    if suggested.memory is not None and suggested.memory.revert is not None:
        return suggested.memory.revert
    for group in try_first_order(diagnosis, suggested):
        for s in group_members(diagnosis, suggested, group):
            if not s.entry.quality_risk:
                return s
    return None
