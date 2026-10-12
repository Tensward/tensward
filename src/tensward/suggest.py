"""Which playbook entries a run calls for, and the suggestions they become."""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .answer_options import FP8_KV_FLOOR, FP8_KV_RULE
from .answers_after import AnswersAfter, before_after
from .classify import Bottleneck, Diagnosis
from .context import GOALS, Goal
from .context import current as analysis
from .engines import Engine
from .engines.vllm.predicates import SMALLER_STEP_TOKENS
from .estimate import ESTIMATORS, PROMOTE_AT, Estimate, RunInputs, lead_bottleneck
from .extensions import load_extender
from .history import DOUBLED_RAISE, SIZED_RAISE, Memory, command_args, guided
from .kept import answer_changes
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
from .prompt_lookup import TAIL, coverage
from .rejected import Rejection, refused_by
from .runs import answered_prompts
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
    answers_after: AnswersAfter | None = None
    floor: Mapping[str, Any] | None = None  # the floor's measured trade, when one is declared
    rejected: tuple[tuple[Suggestion, Rejection], ...] = ()  # offers left out: the user rejected
    carried: str | None = None  # the gate this run's own change falls under: nothing stacks on it
    left_out_why: Mapping[str, str] = field(default_factory=dict)  # of other_goals, by name

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
    run gets no server change in Try first: the test, not the server, held it back. A change
    left out because it worsens the goal is not named when it moves a lever this run already
    moved from ``current``: So far shows that change."""
    settings, diagnosis = situation.settings, situation.diagnosis
    context = analysis()
    if run is not None:
        context.prompt_blocks = run.prompt_blocks
        context.kv_block_tokens = situation.measurement.kv_block_tokens
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
        options = context.answer_options
        if entry.name == CHECKPOINT and options is not None and options.fast_prompts:
            suggestion = replace(suggestion, addresses=suggestion.addresses | {"prefill"})
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
    carried = carried_gate(entries, situation, current, engine)
    if carried is not None:
        offered = []
    offered, refused = _without_rejected(offered, frozenset(context.engine_args))
    offered, worsens = _ngram_gate(_supported(offered, not_applicable), situation, goal)
    blocked = blocked_changes(entries, situation)
    estimates = _estimates(engine, situation, entries, offered, run) if goal == "throughput" else {}
    unmoved = Suggested(
        suggestions=tuple(offered),
        not_applicable=tuple(not_applicable),
        blocked=blocked,
        estimates=estimates,
        estimated={entry.name: entry for entry in entries if entry.name in estimates},
        goal=goal,
        rejected=refused,
        carried=carried,
        other_goals=tuple(worsens),
        left_out_why=worsens,
    )
    if diagnosis.primary == "load_limited":
        underloaded = replace(
            unmoved,
            suggestions=tuple(replace(s, tier="could_help") for s in offered),
            underloaded=situation.measurement.mean_in_flight,
        )
        return _with_answers_after(
            guided(
                underloaded,
                situation,
                engine=engine,
                entries=entries,
                project_path=project_path,
                registered=current,
            ),
            project_path,
            situation,
            engine,
        )
    ranked_offers, other_goals, lead, by_list = ordered_for_goal(
        offered, goal, situation.measurement, context.ttft_p95_limit_ms, unmoved.estimate
    )
    moved = {
        lever for s in offered for lever in s.entry.levers
        if engine.lever_value(settings, lever) != engine.lever_value(current, lever)
    }  # fmt: skip
    other_goals = tuple(
        name for name in other_goals
        if not any(s.entry.name == name and set(s.entry.levers) & moved for s in offered)
    )  # fmt: skip
    ordered = replace(
        unmoved,
        suggestions=tuple(ranked_offers),
        lead=lead,
        other_goals=(*other_goals, *worsens),
        by_list=by_list,
        neutral=_neutral(offered, goal, situation.measurement, context.ttft_p95_limit_ms),
    )
    return _with_answers_after(
        guided(
            ordered,
            situation,
            engine=engine,
            entries=entries,
            project_path=project_path,
            registered=current,
        ),
        project_path,
        situation,
        engine,
    )


def carried_gate(
    entries: Sequence[Entry], situation: Situation, registered: Settings, engine: Engine
) -> str | None:
    """The reason Tensward rules out a change this run already makes against the registered
    setup (a gate, or an fp8 KV cache below its floor): the run measured what Tensward would
    never offer."""
    facts, settings = situation.facts, situation.settings
    changes = answer_changes(
        engine, registered, settings, checkpoint_quantized=facts.quantization is not None
    )
    gated = next(
        (gate.reason for entry in entries if entry.changes_answers and entry.levers & set(changes)
         for gate in entry.gates if gate.blocks(facts, settings)),
        None,
    )  # fmt: skip
    if gated is None and "kv_cache_precision" in changes:
        if engine.lever_value(settings, "kv_cache_precision") == "8bit":
            return _fp8_kv_refused()
    return gated


def _with_answers_after(
    shown: Suggested, project_path: Path | None, situation: Situation, engine: Engine
) -> Suggested:
    """``shown`` with this run's answers against its parent's, also set on the analysis for the
    CLI to write ``quality.json``."""
    after = before_after(project_path, situation, engine)
    analysis().answers_after = after
    return replace(shown, answers_after=after)


def _without_rejected(
    offered: list[Suggestion], args: frozenset[str]
) -> tuple[list[Suggestion], tuple[tuple[Suggestion, Rejection], ...]]:
    """The offers without those whose command brings back a change the user rejected, and those
    left out with their rejection."""
    rejections = analysis().rejected
    kept, refused = [], []
    for suggestion in offered:
        command = command_args(suggestion.command) if suggestion.command else frozenset()
        if command and (rejection := refused_by(rejections, command, args)) is not None:
            refused.append((suggestion, rejection))
        else:
            kept.append(suggestion)
    return kept, tuple(refused)


def _fp8_kv_refused() -> str | None:
    """Why an fp8 KV cache is not offered on this GPU; None when it is, or when neither the
    engine's support nor the GPU's compute capability is known. Below the floor it is refused
    also when the support read failed."""
    context = analysis()
    options = context.answer_options
    if options is not None:
        if "fp8" in options.kv_cache:
            return None
        capability, version = options.capability, options.engine_version
    else:
        capability, version = context.gpu_capability, None
        if capability is None or capability >= FP8_KV_FLOOR:
            return None
    version = f" {version}" if version else ""
    gpu = f" (this GPU: {capability:g})" if capability is not None else ""
    return f"with vLLM{version}, an fp8 KV cache is {FP8_KV_RULE}{gpu}"


def _supported(
    offered: list[Suggestion], not_applicable: list[tuple[str, str]]
) -> list[Suggestion]:
    """The offers without an fp8 KV cache the GPU cannot run, which go to ``not_applicable``
    with the reason."""
    why = _fp8_kv_refused()
    if why is None:
        return offered
    kept = []
    for suggestion in offered:
        if "kv_cache_precision" in suggestion.entry.levers:
            not_applicable.append((suggestion.entry.name, why))
        else:
            kept.append(suggestion)
    return kept


NGRAM = "ngram-speculation"
# drafting, verifying and async scheduling off, measured on an A10G at 32 running requests
# (Phi-3.5-mini, step 41.0 -> 45.5 ms)
STEP_OVERHEAD_MS = 4.5


def _ngram_gate(
    offered: list[Suggestion], situation: Situation, goal: Goal
) -> tuple[list[Suggestion], dict[str, str]]:
    """N-gram speculation stays where it is only when prompt lookup, replayed over this run's own
    prompts and answers, drafts more of the output than the step overhead costs; under the TPOT
    goal (a p95) the answers it helps least decide, as they are the slowest once speculating. At
    or below the break-even share it could help, except under the TPOT goal, where the replay
    predicts a worse TPOT: it is left out, with the replay as the reason. Where this run cannot
    be replayed, it says why."""
    if not any(s.entry.name == NGRAM for s in offered):
        return offered, {}
    run_dir, step_ms = analysis().this_run(), situation.measurement.step_ms
    if run_dir is None:
        return offered, {}
    found = None
    if not (run_dir / "responses.jsonl").is_file():
        why = "it kept no answers"
    elif not step_ms:
        why = "its step time is unknown"
    else:
        prompts = situation.facts.prompts
        found = coverage(
            (prompts[i], text) for i, text in answered_prompts(run_dir) if i < len(prompts)
        )
        why = "it has no answered prompt to replay"
    if found is None or not step_ms:
        note = f". Prompt lookup could not be replayed on this run: {why}"
        return [
            replace(s, reason=s.reason + note) if s.entry.name == NGRAM else s for s in offered
        ], {}
    overhead = STEP_OVERHEAD_MS / step_ms
    breakeven = overhead / (1 + overhead)
    if goal == "tpot":
        share = found.tail
        whose = (
            f"of the output of the {TAIL:.0%} of answers they help least (the slowest once "
            "speculating)"
        )
    else:
        share, whose = found.pooled, "of all output tokens"
    replay = (
        f"replaying prompt lookup on this run's answers, drafts would supply {share:.1%} "
        f"{whose}, and {breakeven:.1%} is needed to pay for the {STEP_OVERHEAD_MS:.1f} ms each "
        f"step adds (measured on an A10G at 32 running requests; this run's step is "
        f"{step_ms:.0f} ms)"
    )
    if goal == "tpot" and share <= breakeven:
        return [s for s in offered if s.entry.name != NGRAM], {
            NGRAM: f"{replay}, so TPOT p95 would rise"
        }
    out = []
    for s in offered:
        if s.entry.name != NGRAM:
            out.append(s)
        elif share <= breakeven:
            cost = f"drafts would not pay for their step overhead here; {s.cost or s.entry.costs}"
            out.append(replace(s, reason=f"{s.reason}. {replay[0].upper()}{replay[1:]}",
                               tier="could_help", cost=cost))  # fmt: skip
        else:
            out.append(replace(s, reason=f"{s.reason}. {replay[0].upper()}{replay[1:]}"))
    return out, {}


GATES = frozenset({"fit", "quality"})
CHECKPOINT = "quantized-checkpoint"  # addresses prefill where the GPU runs a W8A8 format
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
            suggestion = replace(suggestion, cost=f"{trade}; {costs}")
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


def _async_before_smaller_step(ranked: list[Suggestion]) -> list[Suggestion]:
    """Where Try first holds the 256-token step budget and turning async scheduling off, the
    latter comes first: the step budget's wait threshold after it is lower, so the ladder runs
    512, async off, 256."""
    smaller = next(
        (
            i
            for i, s in enumerate(ranked)
            if s.tier == "try_first"
            and s.entry.name == "small-step-budget"
            and s.settings is not None
            and s.settings.prefill_batch_tokens == SMALLER_STEP_TOKENS
        ),
        None,
    )
    asynchronous = next(
        (
            i
            for i, s in enumerate(ranked)
            if s.tier == "try_first" and s.entry.name == "no-async-scheduling"
        ),
        None,
    )
    if smaller is None or asynchronous is None or asynchronous < smaller:
        return ranked
    out = list(ranked)
    out.insert(smaller, out.pop(asynchronous))
    return out


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


def _safe_first(ranked: list[Suggestion]) -> list[Suggestion]:
    """Try first's changes that cannot affect the answers first, then the 8-bit ones, then the
    4-bit ones, each in its order; Could help keeps its places."""
    at = [i for i, s in enumerate(ranked) if s.tier == "try_first"]
    moved = sorted(
        (ranked[i] for i in at),
        key=lambda s: (s.entry.changes_answers, s.entry.answer_bits == 4),
    )
    out = list(ranked)
    for i, suggestion in zip(at, moved):
        out[i] = suggestion
    return out


def answer_warning(entry: Entry, *, retained: bool) -> str:
    """The note after a change that can affect the answers: stronger for a 4-bit one, and
    pointing to the cost for a change no command makes (a different checkpoint)."""
    if not retained:
        return (
            " (answers may change; this run kept no answers, so run it again without "
            "--no-retain-responses to see them before and after)"
        )
    if not entry.levers:
        return " (answers may change: compare them as below)"
    if entry.answer_bits == 4:
        return (
            " (4-bit: answers change more often than with 8-bit; the next report compares them "
            "with this run's)"
        )
    return " (answers may change: the next report compares them with this run's)"


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
    and the changes that lower TTFT move to the front; last, Try first's changes that cannot
    affect the answers come before those that can (``_safe_first``)."""
    ranked_offers, left_out = by_goal(offered, goal, measurement, ttft_limit)
    ranked_offers = _one_cap_raise(_one_step_budget(ranked_offers, goal))
    if goal == "ttft":
        ranked_offers = _async_before_smaller_step(ranked_offers)
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
    return _safe_first(ranked_offers), left_out, lead, goal != "throughput" or broken


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
    secondary bottlenecks. After the failed gates, a group whose changes all can affect the
    answers comes after the others."""
    grouped = [s for s in suggested.suggestions if s.tier == "try_first"]
    addressed = {name for s in grouped for name in s.addresses}
    order: list[str] = [
        g.bottleneck for g in diagnosis.gates if g.state == "critical" and g.bottleneck in addressed
    ]
    gates = len(order)

    def changing(group: str) -> bool:
        members = group_members(diagnosis, suggested, group)
        return bool(members) and all(s.entry.changes_answers for s in members)

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
    else:
        if suggested.lead is not None and suggested.lead not in order:
            order.append(suggested.lead)
        order += [b for b in diagnosed if b not in (suggested.lead, "load_limited")]
    return order[:gates] + sorted(order[gates:], key=changing)


def first_try(diagnosis: Diagnosis, suggested: Suggested) -> Suggestion | None:
    """The change the report shows first under Try first: a revert after a loss, else the first
    change in group order that cannot affect the answers, else the first that can; None when it
    shows none."""
    if suggested.memory is not None and suggested.memory.revert is not None:
        return suggested.memory.revert
    shown = [
        s
        for group in try_first_order(diagnosis, suggested)
        for s in group_members(diagnosis, suggested, group)
    ]
    return next((s for s in shown if not s.entry.changes_answers), shown[0] if shown else None)
