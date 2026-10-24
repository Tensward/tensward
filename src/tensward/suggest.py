"""Which playbook entries a run calls for, and the suggestions they become."""

from __future__ import annotations

import json
import math
import shlex
import tomllib
from dataclasses import dataclass, field, replace
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Mapping, Sequence

from . import __version__, kvsize, trace_facts, window
from .anatomy import freed_bytes
from .answer_options import FP8_KV_FLOOR, FP8_KV_RULE
from .answers_after import AnswersAfter, before_after
from .classify import Bottleneck, Diagnosis
from .combine import (
    FP8_WEIGHTS,
    LOSSLESS_FIRST,
    MAX_MEMBERS,
    NUMERIC,
    SMALL_CHECKPOINT,
    checkpoint_parameters,
    goal_gain,
    independent,
    lossless_remainder,
    parent_offer,
    stackable,
)
from .context import GOALS, AnalysisContext, Goal
from .context import current as analysis
from .engines import Engine
from .engines.vllm.capabilities import FP8
from .engines.vllm.predicates import PER_CHANNEL_EXTRA_GIB, SMALLER_STEP_TOKENS, with_fp8_weights
from .engines.vllm.quantization import ONLINE_FP8, PER_CHANNEL
from .engines.vllm.start_phases import CLASSES, start_class
from .engines.vllm.startup import memory_usage
from .estimate import ESTIMATORS, PROMOTE_AT, Estimate, RunInputs, lead_bottleneck
from .extensions import load_extender
from .facts import extract
from .history import (
    DOUBLED_RAISE,
    GOAL_METRIC,
    GUARDED,
    SIZED_RAISE,
    Memory,
    against,
    command_args,
    full_window,
    guided,
    is_env,
    past_runs,
    seconds_to_start,
    seconds_to_test,
    step_parent,
    stopped_runs,
)
from .kb_entries import (
    Inputs,
    RunKnowledge,
    env_offers,
    judged,
    kb_block,
    kb_entries,
    null_block,
    rules,
    whole_records,
    worth_measuring,
)
from .kept import answer_changes
from .matcher import Null, annotate_offered, match, open_layers
from .measurement import Measurement, nearest_rank
from .playbook import (
    GOAL_WORDS,
    RELEVANT,
    Effect,
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
from .playbook_common import (
    _preempting,
    kv_held_back,
    lowers_concurrency_for_tpot,
    running_at_cap,
    sampled_memory,
    text_only_helps,
    this_run,
    with_cap_for_more_memory,
    without_media_encoders,
)
from .predict import (
    KV_PRECISION,
    Prediction,
    held_steps,
    predict,
    predict_pair,
    predict_stack,
    saturated,
)
from .pressure import demand, prompt_bound, state_of
from .prompt_lookup import TAIL, coverage
from .rejected import Rejection, as_change, refused_by
from .runs import answered_prompts
from .scoring import RuleScorer
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
    test_seconds: Mapping[str, float] = field(default_factory=dict)  # launch to measured, by entry
    start_seconds: Mapping[str, float] = field(default_factory=dict)  # launch to ready, by entry
    trade: Mapping[str, Any] | None = None  # under a latency goal, the measured trade so far
    neutral: frozenset[str] = frozenset()  # changes that leave a latency goal alone (_neutral)
    answers_after: AnswersAfter | None = None
    floor: Mapping[str, Any] | None = None  # the floor's measured trade, when one is declared
    rejected: tuple[tuple[Suggestion, Rejection], ...] = ()  # offers left out: the user rejected
    carried: str | None = None  # the gate this run's own change falls under: nothing stacks on it
    left_out_why: Mapping[str, str] = field(default_factory=dict)  # of other_goals, by name
    kv_first: bool = False  # tool calls: Try first's KV precision comes before its weight precision
    tool_calls: bool = False  # the requests offer tools: precision risks the answers most
    ttft_basis: str | None = None  # the TTFT the offers were ranked on, when not the client's
    ttft_skipped: str | None = None  # why a parser hold did not move the TTFT basis
    led: Suggestion | None = (
        None  # leads Try first whatever its score: the remainder of a checked stack
    )
    kb: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)  # entry name -> its kb block
    worth_measuring: tuple[Suggestion, ...] = ()
    kb_source: Mapping[str, Any] = field(default_factory=dict)  # {"pack", "reason", "skipped"}

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
    moved from ``current``: So far shows that change. The knowledge records that hold for the
    run's facts add the offers of measured and promoted records (a measured one, or one that
    changes the engine's environment, in Could help), block or demote offers by their guards,
    tombstones and measured nulls, and list the documented ones under "Worth measuring"."""
    settings, diagnosis = situation.settings, situation.diagnosis
    context = analysis()
    if run is not None:
        context.prompt_blocks = run.prompt_blocks
        context.kv_block_tokens = situation.measurement.kv_block_tokens
    goal = context.goal
    entries = engine.playbook(version=version)
    m = situation.measurement
    default_cap = _default_cap(engine, version, m)
    known = _knowledge(engine, situation, entries, version, default_cap)
    tombstoned = known.matches.tombstoned
    kb_offers, env_matches = kb_entries(known, version)
    declared = {r.merge_key: r for r in map(known.record, known.matches.offers)}
    found, gated = applicable(
        (*(e for e in entries if e.name not in tombstoned), *kb_offers), situation,
        allow_quality_changes=True, near=True, engine=engine, opt_in=True,
    )  # fmt: skip
    if any(s.entry.name == FP8_KV_DECODE for s in found):
        found = [s for s in found if s.entry.name != FP8_KV_CACHE]
    not_applicable = [*tombstoned.items(), *((entry.name, why) for entry, why in gated)]
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
        if why := _unrunnable(engine, settings, proposed, situation.measurement, version):
            not_applicable.append((entry.name, why))
            continue
        if proposed == settings and entry.levers:
            continue
        command = predicted = None
        if project_path is not None and _changes(engine, current, settings, proposed):
            command = suggestion_command(engine, project_path, current, proposed)
            linear = context.linear_weight_bytes
            predicted = predict(
                situation, proposed, entry=entry.name, linear_bytes=linear, cap=default_cap
            )
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
        gain = _predicted_gain(replace(suggestion, predicted=predicted), goal)
        stays = own_signal and (
            bool(suggestion.addresses & first)
            or (goal != "throughput" and _improves(suggestion, goal))
            or (gain is not None and gain >= LEAD_AT)
        )
        reason = f"{suggestion.reason}; the command {note}" if note else suggestion.reason
        if predicted is not None:
            reason = f"{reason}; {predicted.text}"
        past = _past_limit(predicted, situation.measurement, context)
        offered.append(
            replace(
                suggestion,
                reason=reason,
                tier="try_first" if stays and past is None else "could_help",
                cost=f"{past}; {suggestion.cost or entry.costs}" if past else suggestion.cost,
                command=command,
                settings=proposed,
                predicted=predicted,
                past_limit=past is not None,
            )
        )
    carried = carried_gate(entries, situation, current, engine)
    if carried is not None:
        offered = []
    offered, refused = _without_rejected(offered, frozenset(context.engine_args))
    offered = _sized_offers(
        offered, situation, engine=engine, current=current, project_path=project_path,
        not_applicable=not_applicable,
    )  # fmt: skip
    offered = _closed_loop_cap(offered, situation, goal)
    offered = _without_token_pool(offered, not_applicable)
    offered, worsens = _ngram_gate(_supported(offered, not_applicable), situation, goal)

    def step_command(proposed: Settings, env: bool) -> str | None:
        if project_path is None or not (env or _changes(engine, current, settings, proposed)):
            return None
        return suggestion_command(engine, project_path, current, proposed)

    offered, nulled = judged(offered, known, declared, not_applicable)
    offered += env_offers(env_matches, known, step_command, not_applicable)
    tools = m.tool_calls is not None
    worth = worth_measuring(known, step_command, not_applicable, goal, tools=tools)
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
        return _with_kb(
            _last_pass(
                _with_answers_after(
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
                ),
                underloaded,
                situation,
                engine,
            ),
            known,
            nulled,
            worth,
        )
    held = _held(gated, offered, situation, engine=engine, current=current,
                 project_path=project_path, version=version)  # fmt: skip

    led: list[Suggestion] = []

    def stacked(ranked: list[Suggestion]) -> tuple[list[Suggestion], Suggestion | None]:
        found, first = _stacked(ranked, held, situation, engine=engine, current=current,
                                project_path=project_path, version=version)  # fmt: skip
        small = _small_levers(situation, found, ranked)
        bundle, split = _bundle(situation, engine, small, current=current,
                                project_path=project_path, version=version)  # fmt: skip
        found, first = _bundled(found, first, bundle, split)
        led.extend([first] if first is not None else [])
        batch = _held_batch(ranked, situation, engine=engine, current=current,
                            project_path=project_path, version=version,
                            not_applicable=not_applicable)  # fmt: skip
        if _small_tool_run(situation):
            found = [*found, *_withheld(ranked, situation, engine=engine, current=current,
                                        project_path=project_path, version=version)]  # fmt: skip
        if batch is None:
            return found, first
        return [batch, *(s for s in found if s.entry.name != HELD_CAP)], first

    timed, ttft_basis, ttft_skipped = (
        _ttft_basis(situation, context, default_cap)
        if goal == "ttft"
        else (situation.measurement, None, None)
    )
    ranked_offers, other_goals, lead, by_list = ordered_for_goal(
        offered, goal, timed, context.ttft_p95_limit_ms, unmoved.estimate,
        past=lambda s: _past_limit(s.predicted, situation.measurement, context) is not None,
        stack=stacked,
    )  # fmt: skip
    kv_first = m.tool_calls is not None and _holds_both_precisions(ranked_offers)
    joined = {m.name for s in ranked_offers for m in s.members}
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
        kv_first=kv_first,
        tool_calls=m.tool_calls is not None,
        ttft_basis=ttft_basis,
        ttft_skipped=ttft_skipped,
        led=led[0] if led else None,
        not_applicable=tuple((name, why) for name, why in not_applicable if name not in joined),
        neutral=_neutral(offered, goal, situation.measurement, context.ttft_p95_limit_ms),
    )
    return _with_kb(
        _last_pass(
            _with_answers_after(
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
            ),
            ordered,
            situation,
            engine,
        ),
        known,
        nulled,
        worth,
    )


def _knowledge(
    engine: Engine,
    situation: Situation,
    entries: Sequence[Entry],
    version: str | None,
    cap: int | None,
) -> RunKnowledge:
    """The run's facts and the knowledge records that hold for them, kept on the analysis
    context for the stacks, the report and facts.json."""
    context = analysis()
    facts = extract(situation, context, engine, version, anatomy=context.anatomy)
    layers = open_layers({entry.name for entry in entries})
    matches = match(
        facts, layers, engine_family=engine.family, engine_version=version,
        tensward_version=__version__, rules=rules(engine),
    )  # fmt: skip
    context.kb_facts, context.kb_matches = facts, matches

    def answers() -> list[list[str]]:
        run_dir = context.this_run()
        try:
            return [] if run_dir is None else [t.split() for _, t in answered_prompts(run_dir)]
        except (OSError, ValueError):
            return []

    inputs = Inputs(
        facts=facts, situation=situation, engine=engine, answers=answers, cap=cap,
        prompt_tokens=tuple(tokens for tokens, _ in context.prompt_blocks.values()),
    )  # fmt: skip
    records = whole_records(layers, facts, engine.family, matches)
    return RunKnowledge(facts=facts, layers=layers, matches=matches, records=records, inputs=inputs)


def _with_kb(
    shown: Suggested, known: RunKnowledge, nulled: Mapping[str, Null], worth: Sequence[Suggestion]
) -> Suggested:
    """``shown`` with what the knowledge records said: a ``kb`` block (with its score) for each
    offer a pack or local record matched or a measured null demoted, the "Worth measuring" list
    and the source. The coded records of the offered playbook entries explain them now."""
    context = analysis()
    names = {s.entry.name for s in shown.suggestions}
    matches = annotate_offered(known.matches, known.facts, known.layers, names)
    context.kb_matches = matches
    scorer = RuleScorer(shown.goal, tools=shown.tool_calls)
    blocks: dict[str, Mapping[str, Any]] = {}
    for offer in shown.suggestions:
        name = offer.entry.name
        found = matches.by_entry(name)
        null = nulled.get(name)
        block = kb_block(found, matches) if found else null_block(null, matches) if null else None
        if block is not None:
            score = scorer.score(offer, known.facts, context)
            blocks[name] = {**block, "score": {"value": score.value, **score.parts}}
    source = {**matches.source, "skipped": matches.skipped, "facts_digest": matches.facts_digest}
    return replace(shown, kb=blocks, worth_measuring=tuple(worth), kb_source=source)


FIRST_TOKEN = "first-token"
MAX_PARSER_HOLD = 50.0  # decode steps; a longer wait for the first content is queueing
QUEUE_TIME = "TTFT is queue time here"
START_BAND = 0.10  # offers whose scores differ by this share or less are ordered by start cost
HASH_FILE = "data/compile_hash.toml"
SPENT = "; measured now: nothing else in Try first can run"


@cache
def _hash_neutral() -> frozenset[str]:
    table = tomllib.loads(files("tensward").joinpath(HASH_FILE).read_text("utf-8"))
    return frozenset(entry["flag"] for entry in table["neutral"])


def _flags(situation: Situation, engine: Engine, suggestion: Suggestion) -> frozenset[str]:
    """The engine flags ``suggestion`` changes from the reported run's settings."""
    after = suggestion.settings or situation.settings
    texts = engine.engine_args_between(situation.settings, after)
    return frozenset(text.partition("=")[0] for text in texts)


def start_class_of(situation: Situation, engine: Engine, suggestion: Suggestion) -> str:
    """The start class of ``suggestion``: a change of only hash-neutral flags loads the
    compiled graph."""
    quantization = suggestion.settings.quantization if suggestion.settings else None
    caches = {"aot_hit": _flags(situation, engine, suggestion) <= _hash_neutral(),
              "empty_before": False}  # fmt: skip
    return start_class(None, caches, quantization)


def _start_cost(situation: Situation, engine: Engine, suggestion: Suggestion) -> int:
    """How costly the server start of ``suggestion`` is: the position of its start class."""
    return CLASSES.index(start_class_of(situation, engine, suggestion))


def _without_worsening(shown: Suggested) -> list[Suggestion]:
    """``shown``'s offers with the Try-first ones whose own prediction worsens the goal metric
    beyond the band a step aims for moved to Could help, with the figure."""
    goal = shown.goal
    band = window.target(GOAL_METRIC[goal])
    out = []
    for suggestion in shown.suggestions:
        gain = _predicted_gain(suggestion, goal)
        if (
            suggestion.tier == "try_first"
            and gain is not None
            and gain < -band
            and not suggestion.addresses & GATES
        ):
            why = f"predicted to worsen {GOAL_WORDS[goal]} by {-gain:.0%}"
            costs = suggestion.cost or suggestion.entry.costs
            suggestion = replace(suggestion, tier="could_help", demoted=why, cost=f"{why}; {costs}")
        out.append(suggestion)
    return out


def _by_start_cost(
    offers: list[Suggestion], shown: Suggested, situation: Situation, engine: Engine
) -> list[Suggestion]:
    """``offers`` with the runnable Try-first ones reordered among their own places: within a
    run of them that differ in score by START_BAND or less and alike in whether they can change
    the answers, the shorter test first (start class when timings are unknown), and a
    hash-neutral change right after the offer whose flags include its own. A stack's lead
    (``Suggested.led``) keeps its place."""
    slots = [
        i for i, s in enumerate(offers)
        if s.tier == "try_first" and s.command and s != shown.led
    ]  # fmt: skip
    tools = shown.tool_calls

    def scored(i: int) -> float:
        return _score(offers[i], shown.goal, tools=tools)

    def alike(i: int, anchor: int) -> bool:
        a, b = scored(i), scored(anchor)
        return (
            _changes_answers(offers[i]) == _changes_answers(offers[anchor])
            and abs(a - b) <= START_BAND * max(abs(a), abs(b))
        )  # fmt: skip

    runs: list[list[int]] = []
    for i in slots:
        if runs and alike(i, runs[-1][0]):
            runs[-1].append(i)
        else:
            runs.append([i])
    out = list(offers)
    context = analysis()
    cap = situation.settings.max_concurrent_requests
    at_cap = running_at_cap(situation.measurement, situation.settings)
    for run in runs:
        group = [offers[i] for i in run]
        priced = [
            (s, seconds_to_test(
                context, start_class=start_class_of(situation, engine, s), cap_now=cap,
                cap_new=(s.settings or situation.settings).max_concurrent_requests, at_cap=at_cap,
            ))
            for s in group
        ]  # fmt: skip
        if any(seconds is None or not math.isfinite(seconds) for _, seconds in priced):
            group.sort(key=lambda s: _start_cost(situation, engine, s))
        else:
            group = [s for s, _ in sorted(priced, key=lambda pair: pair[1] or 0.0)]
        for child in [s for s in group if _flags(situation, engine, s) <= _hash_neutral()]:
            flags = _flags(situation, engine, child)
            parent = next(
                (s for s in group if s is not child and flags < _flags(situation, engine, s)), None
            )
            if parent is not None:
                group.remove(child)
                group.insert(group.index(parent) + 1, child)
        for i, offer in zip(run, group):
            out[i] = offer
    return out


def _lossless(suggestion: Suggestion) -> bool:
    return all(
        not m.quality_risk or m.checks_answers for m in suggestion.members or (suggestion.entry,)
    )


def _spent(
    offers: list[Suggestion], shown: Suggested, before: Suggested, situation: Situation
) -> list[Suggestion]:
    """``offers`` with, when no Try-first offer has a command left to run, the Could-help ones
    that have a command, cannot change the answers, were not moved down for a cost against the
    goal or its limits (nor by the project's history) and have no replay that rules out a gain,
    moved up, highest ``score`` first."""
    if any(s.tier == "try_first" and s.command for s in offers):
        return offers
    lowered = {s.entry.name for s in before.suggestions if s.tier == "try_first"}
    candidates = [
        s for s in offers
        if s.tier == "could_help" and s.command and s.demoted is None and not s.past_limit
        and s.entry.name not in lowered and _lossless(s)
        and not (
            shown.goal == "tpot" and s.tail_coverage is not None
            and s.tail_coverage < NGRAM_MIN_COVERAGE
        )
    ]  # fmt: skip
    candidates.sort(key=lambda s: -_score(s, shown.goal, tools=shown.tool_calls))
    diagnosis = situation.diagnosis
    held = goal_bottleneck(diagnosis, shown.goal)
    moved = []
    for s in candidates:
        if held is not None and _home(diagnosis, shown, s) is None:
            s = replace(s, addresses=s.addresses | {held})
        moved.append(replace(s, tier="try_first", reason=f"{s.reason}{SPENT}"))
    return [*moved, *(s for s in offers if all(s is not c for c in candidates))]


def _last_pass(
    shown: Suggested, before: Suggested, situation: Situation, engine: Engine
) -> Suggested:
    """``shown`` after the rules that need the project's history applied: Try-first offers
    predicted to worsen the goal go to Could help; near-equal offers go cheaper start first; and
    once Try first has nothing left to run, the lossless Could-help offers are promoted (the
    next step's measurement is then not wasted). Finally, judged and long-start steps come
    first. ``before``: the offers before the history."""
    offers = _without_worsening(shown)
    promoting = shown.underloaded is None and (shown.memory is None or shown.memory.revert is None)
    spent = False
    if promoting:
        offers = _by_start_cost(offers, shown, situation, engine)
        promoted = _spent(offers, shown, before, situation)
        spent = promoted != offers
        offers = _in_start_order(promoted, situation, engine)
    changed = shown
    if offers != list(shown.suggestions):
        changed = replace(shown, suggestions=tuple(offers), by_list=shown.by_list or spent)
        if spent:
            changed = replace(changed, stop=None)
        if first_try(situation.diagnosis, changed) != first_try(situation.diagnosis, shown):
            changed = replace(changed, plan=(), plan_lines=())
    context = analysis()
    cap = situation.settings.max_concurrent_requests
    at_cap = running_at_cap(situation.measurement, situation.settings)
    revert = changed.memory.revert if changed.memory is not None else None
    test_seconds: dict[str, float] = {}
    start_seconds: dict[str, float] = {}
    for s in [*offers, *([revert] if revert is not None else [])]:
        if not s.command:
            continue
        kind = start_class_of(situation, engine, s)
        test = seconds_to_test(
            context, start_class=kind, cap_now=cap,
            cap_new=(s.settings or situation.settings).max_concurrent_requests, at_cap=at_cap,
            full=full_window(s.command),
        )  # fmt: skip
        start = seconds_to_start(context, kind)
        if test is not None:
            test_seconds[s.entry.name] = test
        if start is not None:
            start_seconds[s.entry.name] = start
    return replace(changed, test_seconds=test_seconds, start_seconds=start_seconds)


FP8_KV_CACHE, FP8_KV_DECODE = "fp8-kv-cache", "fp8-kv-decode"  # never both offered


def _unrunnable(
    engine: Engine,
    settings: Settings,
    proposed: Settings,
    measurement: Measurement,
    version: str | None,
) -> str | None:
    """Why ``proposed`` would not start from this run's ``settings``: the engine refuses it, or
    it lowers the step budget below the concurrency cap."""
    if why := engine.unstartable(settings, proposed):
        return why
    now = settings.prefill_batch_tokens or measurement.step_budget_tokens
    budget = proposed.prefill_batch_tokens
    if budget is not None and now is not None and budget < now:
        return budget_refusal(engine, proposed, measurement, version)
    return None


def _past_limit(
    predicted: Prediction | None, measurement: Measurement, context: AnalysisContext
) -> str | None:
    """Why a change is predicted past the declared floor, or to raise TPOT p95 past the declared
    limit; None when it is not."""
    if predicted is None:
        return None
    floor, limit, p95 = context.max_req_s_loss, context.tpot_p95_limit_ms, measurement.tpot_p95_ms
    if floor is not None and predicted.req_s is not None and predicted.req_s < -floor:
        return f"past your floor (predicted {predicted.req_s:+.0%})"
    if limit is None or predicted.tpot is None or predicted.tpot <= 0 or p95 is None:
        return None
    after = p95 * (1 + predicted.tpot)
    if after <= limit:
        return None
    return f"past your TPOT limit (predicted TPOT p95 {after:.0f} ms, limit {limit:g} ms)"


def _headroom(measurement: Measurement, context: AnalysisContext) -> dict[str, float]:
    """How far inside each declared limit this run is (``1 - measured / limit``), and the floor
    for throughput; a metric with no limit has unlimited room."""
    room: dict[str, float] = dict.fromkeys(GOALS, math.inf)
    if context.max_req_s_loss is not None:
        room["throughput"] = context.max_req_s_loss
    for metric, value, limit in (
        ("ttft", measurement.ttft_p95_ms, context.ttft_p95_limit_ms),
        ("tpot", measurement.tpot_p95_ms, context.tpot_p95_limit_ms),
    ):
        if limit is not None:
            room[metric] = 1 - value / limit if value is not None else -math.inf
    return room


JUDGED = "judge:"  # the start of the reason for a step whose answer check did not adopt it
NOT_KEPT = "tensward: does not recommend keeping this run"  # a step kept out as within noise
NOT_STARTED = "server did not start:"  # the start of its reason for a step that did not start
FAST_PREFILL = "kv-sharing-fast-prefill"


def _held(
    gated: Sequence[tuple[Entry, str]],
    offered: Sequence[Suggestion],
    situation: Situation,
    *,
    engine: Engine,
    current: Settings,
    project_path: Path | None,
    version: str | None,
) -> list[Suggestion]:
    """The changes whose own signal fired but a gate rules out with this run's settings, where
    another offer's settings clear that gate: they are offered only together with it, in a
    stack. Those the goal would leave under Could help are left out."""
    facts, settings = situation.facts, situation.settings
    if project_path is None:
        return []

    def clears(entry: Entry, other: Settings | None) -> bool:
        return other is not None and not any(g.blocks(facts, other) for g in entry.gates)

    held = []
    for entry, _ in gated:
        if clears(entry, settings) or not any(clears(entry, s.settings) for s in offered):
            continue
        reason = entry.applies(situation)
        if not reason or isinstance(reason, HeldBack):
            continue
        proposed, _ = engine.consistent(settings, entry.apply(situation))
        if proposed == settings or _unrunnable(
            engine, settings, proposed, situation.measurement, version
        ):
            continue
        held.append(
            Suggestion(
                entry=entry,
                reason=reason,
                tier="try_first",
                addresses=entry.addresses,
                basis=entry.basis,
                settings=proposed,
                command=suggestion_command(engine, project_path, current, proposed),
            )
        )
    context = analysis()
    kept, _ = by_goal(held, context.goal, situation.measurement, context.ttft_p95_limit_ms)
    return [s for s in kept if s.tier == "try_first"]


def _composed(
    parent: Settings, members: Sequence[Suggestion], engine: Engine, *, base: Settings | None = None
) -> Settings:
    settings = base or parent
    for member in members:
        for text in engine.engine_args_between(parent, member.settings or parent):
            settings = engine.with_engine_arg(settings, text)
    return settings


def _text_only(base: Settings, settings: Settings) -> bool:
    return settings.media_inputs is False and base.media_inputs is not False


def _needs_size(base: Settings, settings: Settings, names: Collection[str]) -> bool:
    """Whether a stack starts without a normal memory profile, so its KV size is set: the text
    model alone, or fast prefill with fp8 weights quantized at start."""
    return _text_only(base, settings) or (FAST_PREFILL in names and FP8_WEIGHTS in names)


def _kv_sizer(situation: Situation) -> Callable[[int], int | None]:
    """A stack's KV size for the bytes it frees, from the session's reference run (the newest
    served run with no engine arguments: this one when it has none), so the same stack always
    gives the same command; None where that run logged no KV size."""
    context = analysis()
    this, before = this_run()
    backed_off = kvsize.failed_sized_steps(context.project, before)
    if not all(map(is_env, context.engine_args)):
        reference = kvsize.reference_run(context.project, before)
        return lambda freed: _sized_from(reference, freed, backed_off)
    return lambda freed: _sized_from(this, freed, backed_off, situation)


def _freed(situation: Situation, *, towers: bool, weights: bool) -> int:
    """The bytes a stack frees: the media towers, and half the quantizable weights for fp8."""
    context = analysis()
    found = freed_bytes(context.anatomy) if context.anatomy is not None else None
    tower_bytes = found["towers"] if found else round(situation.facts.encoder_gib * 2**30)
    half = found["quantizable"] // 2 if found else (context.linear_weight_bytes or 0) // 2
    return (tower_bytes if towers else 0) + (half if weights else 0)


def _frees(situation: Situation, settings: Settings) -> int:
    """The bytes ``settings`` free against this run's: the media towers when they start the text
    model alone, and half the quantizable weights, less per-channel's extra, for fp8 weights."""
    base = situation.settings
    quantized = settings.quantization if settings.quantization != base.quantization else None
    weights = quantized in ("fp8", *ONLINE_FP8)
    freed = _freed(situation, towers=_text_only(base, settings), weights=weights)
    return freed - round(PER_CHANNEL_EXTRA_GIB * 2**30) if quantized == PER_CHANNEL else freed


def _stack_base(situation: Situation, engine: Engine, project_path: Path) -> Settings:
    """The first stack's text-only base, where its sized pool holds the parent's measured use.
    A composite already adopted in this parent's known lineage keeps the existing base."""
    context, base, m = analysis(), situation.settings, situation.measurement
    if engine.name != "vllm" or text_only_helps(situation.facts, base) is None:
        return base
    pool, used = m.kv_available_gib, m.peak_kv_usage
    if (
        pool is None or used is None or not math.isfinite(pool) or not math.isfinite(used)
        or pool <= 0 or not 0 <= used <= 1
    ):  # fmt: skip
        return base
    rejected = {r.run for r in context.rejected}
    runs = [r for r in past_runs(project_path, engine, before=context.reported_run)
            if r.run_id not in rejected]  # fmt: skip
    args = frozenset(context.engine_args)
    while (parent := step_parent(project_path, runs, args)) is not None:
        offer = parent_offer(project_path / "runs" / parent.run_id, args - parent.engine_args)
        if offer is not None and offer.get("members"):
            return base
        args = parent.engine_args
        runs = [r for r in runs if r.run_id < parent.run_id]
    settings = without_media_encoders(situation, engine)
    size = settings.extra_args.get(kvsize.KV_FLAG)
    if settings.media_inputs is False and isinstance(size, str) and size.isdecimal():
        if int(size) / 2**30 >= pool * used:
            return settings
    return base


def _stack(
    members: Sequence[Suggestion],
    situation: Situation,
    *,
    engine: Engine,
    current: Settings,
    project_path: Path,
    version: str | None,
    size: Callable[[int], int | None],
    base: Settings | None = None,
) -> Suggestion | None:
    """``members`` measured in one step, in the anchor's (the first member's) name; None when
    the stack needs a KV size it cannot get, a member's gate blocks it, it would not start, or
    its prediction passes a declared limit."""
    context, m = analysis(), situation.measurement
    if base is None:
        base = _stack_base(situation, engine, project_path)
        if base != situation.settings and any(
            s.entry.name != "no-media-encoders"
            and independent("no-media-encoders", s.entry.name, context.goal) is None
            for s in members
        ):
            base = situation.settings
    names = [s.entry.name for s in members]
    settings, sized = _composed(situation.settings, members, engine, base=base), ""
    if _needs_size(base, settings, names):
        if FP8_WEIGHTS in names:
            fp8 = with_fp8_weights(situation, engine, sized=True)
            for text in engine.engine_args_between(situation.settings, fp8):
                settings = engine.with_engine_arg(settings, text)
        freed = _frees(situation, settings)
        found = size(freed)
        if found is None:
            return None
        settings = kvsize.with_kv_bytes(settings, found)
        sized = (
            f". The KV cache is set to {found / 2**30:.2f} GiB, the reference run's plus the "
            f"{freed / 2**30:.2f} GiB the step frees: without a size the engine's start-up "
            "profile can leave it much smaller"
        )
    if base != situation.settings:
        try:
            pool = int(str(settings.extra_args.get(kvsize.KV_FLAG))) / 2**30
        except (ValueError, OverflowError):
            return None
        if pool <= 0 or pool < (m.kv_available_gib or 0) * (m.peak_kv_usage or 0):
            return None
        if not sized:
            sized = f". The text model alone starts with an explicit {pool:.2f} GiB KV cache"
    if any(g.blocks(situation.facts, settings) for s in members for g in s.entry.gates):
        return None
    if _unrunnable(engine, situation.settings, settings, m, version):
        return None
    numeric = [s for s in members if s.entry.name in NUMERIC]
    predicted = (
        numeric[0].predicted
        if len(numeric) == 1
        else predict_stack(
            situation,
            [(s.entry.name, s.settings or base) for s in members],
            linear_bytes=context.linear_weight_bytes,
            cap=_default_cap(engine, version, m),
        )
    )
    if _past_limit(predicted, m, context):
        return None
    anchor = members[0]
    if predicted is not None:
        measured = (
            f"Measured together: if the gain falls short of the prediction for "
            f"{' with '.join(n.entry.name for n in numeric)} alone by more than noise, they are "
            "measured apart next"
        )
        if predicted != anchor.predicted:
            measured += f". Together, {predicted.text}"
    else:
        measured = "Measured together: no joint figure is predicted; each change's prior is listed"
    return replace(
        anchor,
        reason=". ".join(
            [anchor.reason, *(f"Together with: {s.reason}" for s in members[1:]), measured]
        )
        + sized,
        tier="try_first",
        cost="; ".join(s.cost or s.entry.costs for s in members),
        command=suggestion_command(engine, project_path, current, settings),
        settings=settings,
        predicted=predicted,
        members=tuple(s.entry for s in members),
        past_limit=False,
    )


def _named(
    members: Sequence[Suggestion],
    text: str,
    situation: Situation,
    engine: Engine,
    version: str | None,
) -> Suggestion:
    """The member a start-up error names (its change or one of its engine arguments), else the
    member with the newest prior in the playbook, else the newest."""
    for member in members:
        args = engine.engine_args_between(situation.settings, member.settings or situation.settings)
        if member.entry.name in text or any(a.partition("=")[0] in text for a in args):
            return member
    order = {entry.name: i for i, entry in enumerate(engine.playbook(version=version))}
    return max(members, key=lambda s: (s.entry.prior is not None, order.get(s.entry.name, -1)))


BUNDLE_WINDOW = "full"  # "auto" plans a bundle's window as any step's
BUNDLED = (
    "Bundled: none of these changes can change the answers, so they are measured in one step "
    "and kept unless it is worse; if it is worse, each is offered alone next"
)


def _small_levers(
    situation: Situation, offered: Sequence[Suggestion], ranked: Sequence[Suggestion]
) -> list[Suggestion]:
    """The Try-first changes of ``ranked`` measured alone that cannot change the answers, that
    the memory-traffic model gives no figure and that clear no failed gate, where ``offered``
    holds them alone or in a step that cannot change the answers either."""
    gates = {g.bottleneck for g in situation.diagnosis.gates if g.state == "critical"}
    held = {m.name for s in offered if s.members and not _changes_answers(s) for m in s.members}
    return [
        s for s in ranked
        if s.tier == "try_first" and s.command and s.settings is not None and not s.members
        and not s.entry.changes_answers and s.entry.name not in NUMERIC
        and not s.addresses & gates and (s in offered or s.entry.name in held)
    ]  # fmt: skip


def _bundle(
    situation: Situation,
    engine: Engine,
    offers: Sequence[Suggestion],
    *,
    current: Settings,
    project_path: Path | None,
    version: str | None,
) -> tuple[Suggestion | None, tuple[Suggestion, ...]]:
    """One step of up to MAX_MEMBERS of ``offers``, which cannot change the answers, in their
    order, any two of which may join unless a knowledge record says never, measured over
    BUNDLE_WINDOW; and the members it gives back as single steps. A change measured alone does
    not join. Only a worse verdict gives back the members; otherwise a measured or rejected
    bundle has no command, removing its members without offering another measurement. One
    that did not start loses the member its error names, else the newest."""
    if project_path is None or len(offers) < 2:
        return None, ()
    context = analysis()
    runs = past_runs(project_path, engine, before=context.reported_run)
    tried = {as_change(run.engine_args) for run in runs}
    stopped = {as_change(args): why for _, args, why in stopped_runs(project_path)}
    headroom, size = _headroom(situation.measurement, context), _kv_sizer(situation)
    base = _stack_base(situation, engine, project_path)
    base_name = "no-media-encoders" if base != situation.settings else None

    def build(chosen: Sequence[Suggestion]) -> Suggestion | None:
        return _stack(chosen, situation, engine=engine, current=current,
                      project_path=project_path, version=version, size=size, base=base)  # fmt: skip

    members: list[Suggestion] = []
    for candidate in offers:
        if as_change(command_args(candidate.command or "")) in tried:
            continue
        why = stackable(
            members,
            candidate,
            goal=context.goal,
            headroom=headroom,
            tried=(),
            bundle=True,
            base=base_name,
        )
        if why is None and (not members or build([*members, candidate])):
            members.append(candidate)
    while len(members) >= 2 and (bundle := build(members)) is not None:
        change = as_change(command_args(bundle.command or ""))
        rejection = refused_by(context.rejected, change, frozenset(context.engine_args))
        why = stopped.get(change) or (rejection.reason or "" if rejection is not None else None)
        if change in stopped or (why is not None and why.startswith(NOT_STARTED)):
            members.remove(_named(members, why or "", situation, engine, version))
            continue
        worse = why is not None and why.startswith(("goal:", JUDGED)) and "worse" in why.split()
        measured = [run for run in runs if as_change(run.engine_args) == change]
        for run in measured:
            before = step_parent(project_path, [r for r in runs if r.run_id < run.run_id],
                                 run.engine_args)  # fmt: skip
            if before is not None:
                found = against(before.run_id, project_path / "runs" / run.run_id, runs)
                worse |= any(
                    m in found.deltas and found.deltas[m].verdict == "worse"
                    and not (m == "req_s" and found.open_loop) for m in GUARDED
                )  # fmt: skip
        if worse:
            return None, tuple(members)
        if why is not None or measured:
            return replace(bundle, command=None), ()
        window_arg = f" --window {BUNDLE_WINDOW}" if BUNDLE_WINDOW != "auto" else ""
        reason, command = f"{bundle.reason}. {BUNDLED}", f"{bundle.command}{window_arg}"
        return replace(bundle, reason=reason, command=command), ()
    return None, ()


def _bundled(
    offered: list[Suggestion],
    lead: Suggestion | None,
    bundle: Suggestion | None,
    split: Sequence[Suggestion],
) -> tuple[list[Suggestion], Suggestion | None]:
    """``offered`` with ``bundle`` in the place of its members' single steps and of a lossless
    step it holds; or with the members of ``split`` as single steps in the place of a lossless
    step they make up, else just after the step that holds them. A bundle with no command only
    removes its members. ``lead`` gives way to the runnable bundle that holds it."""
    names = {m.name for m in bundle.members} if bundle else {s.entry.name for s in split}
    gone = [
        s for s in offered
        if (bundle or s.members) and not _changes_answers(s)
        and {m.name for m in s.members or (s.entry,)} <= names
    ]  # fmt: skip
    after = next(
        (i + 1 for i, s in enumerate(offered) if names & {m.name for m in s.members}),
        len(offered),
    )
    at = offered.index(gone[0]) if gone else after
    kept = [s for s in offered if s not in gone]
    new = ([bundle] if bundle.command else []) if bundle else [s for s in split if s not in offered]
    first = bundle if bundle is not None and bundle.command else None
    return [*kept[:at], *new, *kept[at:]], first if lead in gone else lead


def _stacked(
    offered: Sequence[Suggestion],
    held: Sequence[Suggestion],
    situation: Situation,
    *,
    engine: Engine,
    current: Settings,
    project_path: Path | None,
    version: str | None,
) -> tuple[list[Suggestion], Suggestion | None]:
    """``offered`` with two to four Try-first changes measured in one step, in the place of the
    first of them. The anchor is the highest-``score`` Try-first offer with a prediction on the
    goal (else the highest-score one); the others join in score order while ``stackable``, a
    ``held`` change together with the offer that clears its gate. Where the stack's lossless
    remainder scores at least LOSSLESS_FIRST of the stack, it is offered after it. A stack whose
    answer check did not adopt it, or that Tensward would not keep as within noise, gives way to its
    lossless remainder, or, on a run whose requests offer
    tools, where it held fp8 weights with fp8 KV and leaves no remainder, to fp8 KV alone; one
    that did not start loses the member its error names, else the newest; one rejected for
    another reason is not offered again. The second value is the offer that leads whatever its
    score: that remainder, or fp8 KV alone."""
    try_first = [
        s for s in offered if s.tier == "try_first" and s.command and s.settings and not s.members
    ]
    if project_path is None or not try_first:
        return list(offered), None
    context = analysis()
    goal, base = context.goal, _stack_base(situation, engine, project_path)
    base_name = "no-media-encoders" if base != situation.settings else None
    tools = situation.measurement.tool_calls is not None

    def key(s: Suggestion) -> float:
        return _score(s, goal, tools=tools)

    anchor = max(
        [s for s in try_first if goal_gain(s.predicted, goal) is not None] or try_first, key=key
    )
    runs = past_runs(project_path, engine, before=context.reported_run)
    stopped = [(as_change(args), why) for _, args, why in stopped_runs(project_path)]
    judged = {as_change(r.engine_args) for r in context.rejected} | {a for a, _ in stopped}
    again = {r.run for r in context.retried}
    tried = {as_change(run.engine_args) for run in runs if run.run_id not in again} | judged
    headroom = _headroom(situation.measurement, context)
    size = _kv_sizer(situation)
    if base_name and stackable((), anchor, goal=goal, headroom=headroom, tried=(), base=base_name):
        base, base_name = situation.settings, None

    def joins(members: list[Suggestion], new: Sequence[Suggestion]) -> bool:
        if any(s.entry.name == HELD_CAP for s in (*members, *new)):
            return False  # the cap joins precision only in the held batch
        trial = list(members)
        for s in new:
            if stackable(trial, s, goal=goal, headroom=headroom, tried=tried - judged,
                         base=base_name):  # fmt: skip
                return False
            trial.append(s)
        changing = [s for s in trial if s.entry.changes_answers]
        if len(changing) == 2 and _kv_bound_pair(situation, engine, context, version) is None:
            return False
        settings = _composed(situation.settings, trial, engine, base=base)
        names = [s.entry.name for s in trial]
        if _needs_size(base, settings, names) and size(0) is None:
            return False
        if any(g.blocks(situation.facts, settings) for s in trial for g in s.entry.gates):
            return False
        return not _unrunnable(engine, situation.settings, settings, situation.measurement, version)

    candidates = sorted((s for s in (*try_first, *held) if s is not anchor), key=lambda s: -key(s))
    enablers = [
        s
        for s in offered
        if s.command and s.settings and not s.members and not s.entry.changes_answers
    ]
    members = [anchor]
    while sum(s.entry.name != base_name for s in members) < MAX_MEMBERS:
        unit = None
        for candidate in (c for c in candidates if c not in members):
            if joins(members, [candidate]):
                unit = [candidate]
            elif candidate in held:
                unit = next(
                    (
                        [e, candidate]
                        for e in enablers
                        if e not in members and joins(members, [e, candidate])
                    ),
                    None,
                )
            if unit:
                break
        if not unit:
            break
        members += unit

    def build(chosen: Sequence[Suggestion]) -> Suggestion | None:
        return _stack(
            chosen,
            situation,
            engine=engine,
            current=current,
            project_path=project_path,
            version=version,
            size=size,
            base=base,
        )

    stack, judged_out, fallback = None, False, None
    while len(members) >= 2 and (stack := build(members)) is not None:
        change = as_change(command_args(stack.command or ""))
        rejection = refused_by(context.rejected, change, frozenset(context.engine_args))
        why = rejection.reason or "" if rejection is not None else None
        why = next((text for args, text in stopped if args == change), why)
        if why is None:
            break
        if why.startswith((JUDGED, NOT_KEPT)):
            if tools and _both_precisions(s.entry.name for s in members):
                fallback = next(
                    (s for s in try_first if s.entry.name in KV_PRECISION
                     and as_change(command_args(s.command or "")) not in tried),
                    None,
                )  # fmt: skip
            kept = [s for s in members if not s.entry.changes_answers]
            if len(kept) == len(members):
                return list(offered), fallback
            members, judged_out = kept, True
        elif why.startswith(NOT_STARTED) or rejection is None:
            members.remove(_named(members, why, situation, engine, version))
        else:
            return list(offered), None
        stack = None
        settings = _composed(situation.settings, members, engine, base=base)
        members = [
            s
            for s in members
            if not any(g.blocks(situation.facts, settings) for g in s.entry.gates)
        ]
    if stack is None or as_change(command_args(stack.command or "")) in tried:
        return list(offered), fallback
    placed = [stack]
    lossless = [s for s in members if not s.entry.changes_answers]
    if lossless and len(lossless) < len(members):
        remainder = build(lossless) if len(lossless) > 1 else lossless[0]
        if (
            remainder is not None
            and remainder not in held
            and as_change(command_args(remainder.command or "")) not in tried
            and key(remainder) >= LOSSLESS_FIRST * key(stack)
        ):
            placed = [stack, remainder]
    first = min(i for i, s in enumerate(offered) if s in members)
    before = [s for s in offered[:first] if s not in members]
    after = [s for s in offered[first:] if s not in members]
    return [*before, *placed, *after], placed[0] if judged_out else None


SMALL_TOOL_RUN = (
    "with tool calls on a checkpoint under 2B parameters, 8-bit weights with 8-bit KV together "
    "can break the calls: fp8 KV alone is offered first, and this once it holds"
)


def _small_tool_run(situation: Situation) -> bool:
    """Whether the requests offer tools to a checkpoint known to hold under SMALL_CHECKPOINT
    parameters."""
    params = checkpoint_parameters()
    return (
        situation.measurement.tool_calls is not None
        and params is not None
        and params < SMALL_CHECKPOINT
    )


def _kv_bound_pair(
    situation: Situation,
    engine: Engine,
    context: AnalysisContext,
    version: str | None,
    *,
    small: bool = False,
) -> Prediction | None:
    """The prediction for fp8 weights with fp8 KV in one step; None off ``predict_pair``'s
    routes, where the requests offer tools and KV space held none back (one judged step for both
    pays only where the memory they free runs more requests), where they offer tools to a
    checkpoint under SMALL_CHECKPOINT parameters (unless ``small``), or where the pair is
    predicted past a declared limit."""
    measurement = situation.measurement
    if measurement.tool_calls is not None and not kv_held_back(situation):
        return None
    if _small_tool_run(situation) and not small:
        return None
    cap = _default_cap(engine, version, measurement)
    pair = predict_pair(situation, linear_bytes=context.linear_weight_bytes, cap=cap)
    return None if _past_limit(pair, measurement, context) else pair


HELD_BATCH = "held-batch"  # the name a refused held batch is listed under
HELD_CAP = "lower-concurrency"  # the change whose cap holds the batch
HELD_BATCH_MARGIN = 0.05  # the cap predictions' error on a measured cap step, in req/s
LATENCY_BOUND = 1.5  # a held batch's TTFT p99 and E2E p95, at most this many times this run's


def _held_batch(
    ranked: Sequence[Suggestion],
    situation: Situation,
    *,
    engine: Engine,
    current: Settings,
    project_path: Path | None,
    version: str | None,
    not_applicable: list[tuple[str, str]],
    small: bool = False,
) -> Suggestion | None:
    """Under the TPOT goal, on a run short of KV memory (``kv_capacity`` fired) whose KV space
    held requests back: fp8 weights, with fp8 KV where the pair is offered, and a concurrency cap
    k that holds the batch near where it ran, so the bytes they free shorten each step instead of
    running more requests. k is the smallest cap from the most requests that ran, under the
    requests in flight and the run's cap, whose prediction ``_held_figures`` passes; when every
    such cap fails, the largest one's reason is listed under HELD_BATCH. None where it is not
    offered (with tool calls on a checkpoint under SMALL_CHECKPOINT parameters unless ``small``), or
    where its command was measured, stopped or rejected already. Where it is offered, it
    replaces the single cap offer."""
    context, m = analysis(), situation.measurement
    if (
        context.goal != "tpot" or project_path is None or not situation.fired("kv_capacity")
        or not kv_held_back(situation) or (_small_tool_run(situation) and not small)
    ):  # fmt: skip
        return None
    single = {s.entry.name: s for s in ranked if s.command and s.settings and not s.members}
    weights = single.get(FP8_WEIGHTS)
    entry = next((e for e in engine.playbook(version=version) if e.name == HELD_CAP), None)
    why, wanted = independent(HELD_CAP, FP8_WEIGHTS, context.goal), demand(m)
    if weights is None or entry is None or why is None or not m.peak_running or not wanted:
        return None
    kv = next((single[n] for n in KV_PRECISION if n in single), None)
    paired = (
        kv is not None
        and _kv_bound_pair(situation, engine, context, version, small=small) is not None
    )
    precision = [weights, *([kv] if kv is not None and paired else [])]
    default_cap = _default_cap(engine, version, m)
    caps = range(
        math.ceil(m.peak_running),
        min(wanted, situation.settings.max_concurrent_requests or default_cap or wanted),
    )
    refused = None
    for k in caps:
        members = [
            *(replace(s, settings=replace(s.settings, max_concurrent_requests=k))
              for s in precision if s.settings is not None),
            Suggestion(
                entry=entry, reason=f"max_concurrent_requests {k}: {why}", tier="try_first",
                addresses=entry.addresses, basis=entry.basis,
                settings=replace(situation.settings, max_concurrent_requests=k),
            ),
        ]  # fmt: skip
        predicted = predict_stack(
            situation, [(s.entry.name, s.settings or situation.settings) for s in members],
            linear_bytes=context.linear_weight_bytes, cap=default_cap,
        )  # fmt: skip
        passes, figures = _held_figures(predicted, situation, k)
        if not passes:
            refused = figures
            continue
        stack = _stack(members, situation, engine=engine, current=current,
                       project_path=project_path, version=version,
                       size=_kv_sizer(situation))  # fmt: skip
        if stack is None:
            continue
        change = as_change(command_args(stack.command or ""))
        runs = past_runs(project_path, engine, before=context.reported_run)
        tried = {as_change(r.engine_args) for r in runs}
        tried |= {as_change(args) for _, args, _ in stopped_runs(project_path)}
        if change in tried or refused_by(context.rejected, change, frozenset(context.engine_args)):
            return None
        return replace(stack, reason=f"Held batch: {figures}. {stack.reason}")
    if refused is not None:
        not_applicable.append((HELD_BATCH, f"no concurrency cap holds the batch here: {refused}"))
    return None


def _withheld(
    ranked: Sequence[Suggestion],
    situation: Situation,
    *,
    engine: Engine,
    current: Settings,
    project_path: Path | None,
    version: str | None,
) -> list[Suggestion]:
    """With tool calls on a checkpoint under SMALL_CHECKPOINT parameters, the fp8 weights with fp8
    KV pair, and the held batch, as they would be offered, under Could help with SMALL_TOOL_RUN."""
    context = analysis()
    if (
        project_path is None
        or _kv_bound_pair(situation, engine, context, version, small=True) is None
    ):
        return []
    single = {s.entry.name: s for s in ranked if s.command and s.settings and not s.members}
    kv = next((single[n] for n in KV_PRECISION if n in single), None)
    weights = single.get(FP8_WEIGHTS)
    pair = None
    if weights is not None and kv is not None:
        pair = _stack([weights, kv], situation, engine=engine, current=current,
                      project_path=project_path, version=version,
                      size=_kv_sizer(situation))  # fmt: skip
    batch = _held_batch(ranked, situation, engine=engine, current=current,
                        project_path=project_path, version=version, not_applicable=[],
                        small=True)  # fmt: skip
    measured = {as_change(r.engine_args) for r in context.rejected}
    measured |= {as_change(args) for _, args, _ in stopped_runs(project_path)}
    measured |= {
        as_change(run.engine_args)
        for run in past_runs(project_path, engine, before=context.reported_run)
    }
    return [
        replace(s, tier="could_help", demoted=SMALL_TOOL_RUN, cost=f"{SMALL_TOOL_RUN}; {s.cost}")
        for s in (pair, batch)
        if s is not None and as_change(command_args(s.command or "")) not in measured
    ]


def _held_figures(predicted: Prediction | None, situation: Situation, k: int) -> tuple[bool, str]:
    """Whether a held batch at cap ``k`` keeps its guards, and its figures against them: req/s
    at least the parent's less the declared floor plus HELD_BATCH_MARGIN (none declared: the
    parent's plus the margin); TTFT p99 within LATENCY_BOUND of the parent's, the start-up wave
    served k at a time (wave / k x each request's decode time; the parent's p99 read as the
    larger of its TTFT p95 and its wave's); TTFT p95 within the declared limit; and E2E p95,
    which no limit declares, within LATENCY_BOUND of the parent's."""
    context, m = analysis(), situation.measurement
    wanted, wave = demand(m), m.startup_wave
    if (
        predicted is None or predicted.req_s is None or predicted.tpot is None or not wanted
        or m.ttft_p95_ms is None or m.e2e_p95_ms is None or not m.tpot_p50_ms
    ):  # fmt: skip
        return False, f"cap {k}: no prediction of its req/s, TPOT, TTFT and E2E"
    floor = HELD_BATCH_MARGIN - (context.max_req_s_loss or 0.0)
    p95 = m.ttft_p95_ms * (1 + (predicted.ttft or 0.0))
    burst = wanted / k * m.tpot_p50_ms * (1 + predicted.tpot) * situation.facts.output_tokens
    parent_p99 = max(m.ttft_p95_ms, (wave.ttft_p95_ms if wave is not None else None) or 0.0)
    e2e = p95 + (m.e2e_p95_ms - m.ttft_p95_ms) * (1 + predicted.tpot)
    limit = context.ttft_p95_limit_ms
    checks = [
        (predicted.req_s >= floor, f"req/s {predicted.req_s:+.0%} (floor {floor:+.0%})"),
        (
            max(p95, burst) <= LATENCY_BOUND * parent_p99,
            f"TTFT p99 {max(p95, burst) / 1000:.1f} s with the start-up wave (bound "
            f"{LATENCY_BOUND * parent_p99 / 1000:.1f} s)",
        ),
        (
            limit is None or p95 <= limit,
            f"TTFT p95 {p95 / 1000:.1f} s"
            + (f" (limit {limit / 1000:.1f} s)" if limit is not None else ""),
        ),
        (
            e2e <= LATENCY_BOUND * m.e2e_p95_ms,
            f"E2E p95 {e2e / 1000:.1f} s (bound {LATENCY_BOUND * m.e2e_p95_ms / 1000:.1f} s)",
        ),
    ]
    text = f"cap {k}, predicted TPOT {predicted.tpot:+.0%}, " + ", ".join(t for _, t in checks)
    return all(ok for ok, _ in checks), text


def _default_cap(engine: Engine, version: str | None, measurement: Measurement) -> int | None:
    """The engine's own concurrency cap for this version, GPU and memory."""
    gpu = measurement.ceilings.gpu if measurement.ceilings is not None else None
    return engine.default_concurrency(version, gpu, measurement.gpu_memory_gib)


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
            return _fp8_kv_refused(settings)
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


def _log(run_dir: Path | None) -> str:
    try:
        return (run_dir / "server.log").read_text(errors="replace") if run_dir is not None else ""
    except OSError:
        return ""


def _pool_bytes(situation: Situation) -> int | None:
    gib = situation.measurement.kv_available_gib
    return round(gib * 2**30) if gib is not None else None


def _kv_bytes(run_dir: Path | None) -> int | None:
    if run_dir is None:
        return None
    try:
        gib = json.loads((run_dir / "metrics.json").read_text("utf-8")).get("kv_available_gib")
    except (OSError, ValueError, AttributeError):
        return None
    return round(gib * 2**30) if isinstance(gib, (int, float)) else None


def _sized_from(
    run_dir: Path | None, freed: int, backed_off: int, situation: Situation | None = None
) -> int | None:
    """The explicit KV size ``run_dir``'s pool, its logged "fully utilize" figure and its sampled
    GPU memory allow when a change frees ``freed`` bytes. ``situation`` is given when
    ``run_dir`` is this run: ``analyse`` writes its metrics.json and window.json after the
    suggestions, so its pool and memory come from the measurement and the GPU sampler."""
    usage = memory_usage(_log(run_dir))
    if situation is None:
        kv, (peak_used, total) = _kv_bytes(run_dir), kvsize.sampled_memory(run_dir)
    else:
        kv = _pool_bytes(situation)
        peak_used, total = sampled_memory(run_dir, situation.measurement)
    return kvsize.sized(
        kv,
        freed_bytes=freed,
        full_kv_bytes=int(usage["full_kv_bytes"]) if usage else None,
        peak_used_mib=peak_used,
        total_mib=total,
        backed_off=backed_off,
    )


def _with_size(
    suggestion: Suggestion, settings: Settings, size: int, engine: Engine, current: Settings,
    project_path: Path, why: str,
) -> Suggestion:  # fmt: skip
    sized = kvsize.with_kv_bytes(settings, size)
    return replace(
        suggestion,
        reason=f"{suggestion.reason}; {why} {size / 2**30:.2f} GiB",
        settings=sized,
        command=suggestion_command(engine, project_path, current, sized),
    )


def _sized_offers(
    offered: list[Suggestion], situation: Situation, *, engine: Engine, current: Settings,
    project_path: Path | None, not_applicable: list[tuple[str, str]],
) -> list[Suggestion]:  # fmt: skip
    """``offered`` with every explicit KV size of a change that frees memory taken from the run
    with no engine arguments plus what the change frees, so the command is the same whichever
    run is reported; and a single offer that a stopped step already tried given this run's pool
    plus what it frees as its size (backed off when the stopped step carried a size), or, when
    none can be worked out, left out with the stop reason."""
    context = analysis()
    this, before = this_run()
    if project_path is None or context.project is None:
        return offered
    reference = kvsize.reference_run(context.project, before)
    stopped = stopped_runs(context.project)
    failed = kvsize.failed_sized_steps(context.project, before)
    kept = []
    for suggestion in offered:
        settings = suggestion.settings
        args = command_args(suggestion.command) if suggestion.command else frozenset()
        carried = kvsize.KV_FLAG in settings.extra_args if settings is not None else False
        tried = next(
            (why for _, run_args, why in stopped
             if args and args <= run_args <= frozenset(context.engine_args) | args),
            None,
        )  # fmt: skip
        if settings is None or not args:
            kept.append(suggestion)
        elif tried is not None:
            size = _sized_from(this, _frees(situation, settings), failed + carried, situation)
            if size is None:
                not_applicable.append(
                    (suggestion.entry.name, f"{tried}; no KV size can be worked out")
                )
            else:
                kept.append(_with_size(
                    suggestion, settings, size, engine, current, project_path,
                    "an earlier step of this change stopped at start-up, so its KV size is",
                ))  # fmt: skip
        elif carried and reference is not None and reference != this:
            freed = _frees(situation, settings)
            size = _sized_from(reference, freed, failed) if freed > 0 else None
            kept.append(
                suggestion if size is None else _with_size(
                    suggestion, settings, size, engine, current, project_path,
                    f"the KV size, from run {reference.name} (no engine arguments), is",
                )
            )  # fmt: skip
        else:
            kept.append(suggestion)
    return kept


def _closed_loop_cap(
    offered: list[Suggestion], situation: Situation, goal: Goal
) -> list[Suggestion]:
    """``lower-concurrency`` under Could help, with the wait the cap adds, where the load is a
    closed loop with more clients than the cap and a TTFT limit, a req/s floor or the
    throughput goal is declared: the clients beyond the cap queue for a turn. A preempting run
    under the throughput goal with no TTFT limit keeps it: the cap ends the recompute."""
    context = analysis()
    facts, m = situation.facts, situation.measurement
    clients = facts.max_inflight
    declared = (
        context.ttft_p95_limit_ms is not None
        or context.max_req_s_loss is not None
        or goal == "throughput"
    )
    if clients is None or not facts.load.endswith("concurrent clients") or not declared:
        return offered
    if _preempting(situation) and goal == "throughput" and context.ttft_p95_limit_ms is None:
        return offered
    out = []
    for suggestion in offered:
        cap = suggestion.settings.max_concurrent_requests if suggestion.settings else None
        if suggestion.entry.name != "lower-concurrency" or not cap or clients <= cap:
            out.append(suggestion)
            continue
        queued = f"in a closed loop, {clients - cap} of {clients} clients wait for a turn"
        if m.ttft_p50_ms is not None and m.tpot_p50_ms is not None:
            e2e = m.ttft_p50_ms + m.tpot_p50_ms * max(facts.output_tokens - 1, 0)
            queued = (
                f"in a closed loop, each request waits about ({clients} - {cap}) / {cap} x "
                f"E2E p50 = {(clients - cap) / cap * e2e:.0f} ms more for its turn"
            )
        costs = suggestion.cost or suggestion.entry.costs
        out.append(
            replace(
                suggestion,
                tier="could_help",
                cost=f"{queued}; weak in a closed loop; {costs}",
                demoted=queued,
            )
        )
    return out


TOKEN_POOL_OFFERS = frozenset(
    ("more-kv-memory", "kv-sized-concurrency", "explicit-kv-size", FP8_KV_CACHE)
)  # their gain is more tokens in the KV pool


def _without_token_pool(
    offered: list[Suggestion], not_applicable: list[tuple[str, str]]
) -> list[Suggestion]:
    """``offered`` without the changes that enlarge the KV pool where the model keeps a state
    for each sequence in it, which the pool's token count does not describe; they go to
    ``not_applicable`` with the reason."""
    anatomy = analysis().anatomy
    if anatomy is None or not anatomy.linear_attention:
        return offered
    for suggestion in offered:
        if suggestion.entry.name in TOKEN_POOL_OFFERS:
            not_applicable.append(
                (
                    suggestion.entry.name,
                    "this model keeps a state for each sequence in its cache, so the pool's "
                    "size in tokens does not say what a larger pool holds",
                )
            )
    return [s for s in offered if s.entry.name not in TOKEN_POOL_OFFERS]


FLASHINFER_FP8_KV_GPU = 8.9  # the compute capability where FlashInfer's fp8 KV cache fails
FLASHINFER_FP8_KV_HEAD_DIM = 256  # the widest attention head it starts at there


def _flashinfer_refused(settings: Settings) -> str | None:
    """Why an fp8 KV cache under a forced FlashInfer backend fails to start here: on compute
    capability 8.9 it fails at head dim 512 (measured with vLLM 0.30.0)."""
    context = analysis()
    options, anatomy = context.answer_options, context.anatomy
    capability = options.capability if options is not None else context.gpu_capability
    backend = settings.extra_args.get("--attention-backend")
    if (
        capability != FLASHINFER_FP8_KV_GPU
        or anatomy is None
        or str(backend).upper() != "FLASHINFER"
        or max((g.head_dim for g in anatomy.attention), default=0) <= FLASHINFER_FP8_KV_HEAD_DIM
    ):
        return None
    return f"fp8 KV with FlashInfer fails to start at head dim 512 on this GPU ({capability:g})"


def _fp8_kv_refused(settings: Settings | None = None) -> str | None:
    """Why an fp8 KV cache is not offered on this GPU, or with ``settings``' forced attention
    backend; None when it is, or when neither the engine's support nor the GPU's compute
    capability is known. Below the floor it is refused also when the support read failed."""
    context = analysis()
    options = context.answer_options
    flashinfer = _flashinfer_refused(settings) if settings is not None else None
    if options is not None:
        if "fp8" in options.kv_cache:
            return flashinfer
        capability, version = options.capability, options.engine_version
    else:
        capability, version = context.gpu_capability, None
        if capability is None or capability >= FP8_KV_FLOOR:
            return flashinfer
    version = f" {version}" if version else ""
    gpu = f" (this GPU: {capability:g})" if capability is not None else ""
    return f"with vLLM{version}, an fp8 KV cache is {FP8_KV_RULE}{gpu}"


def _supported(
    offered: list[Suggestion], not_applicable: list[tuple[str, str]]
) -> list[Suggestion]:
    """The offers without an fp8 KV cache the GPU cannot run, which go to ``not_applicable``
    with the reason. Another 8-bit KV dtype is not an fp8 KV cache."""
    kept = []
    for suggestion in offered:
        dtype = suggestion.settings.kv_cache_dtype if suggestion.settings else FP8
        precision = "kv_cache_precision" in suggestion.entry.levers and dtype.startswith(FP8)
        if precision and (why := _fp8_kv_refused(suggestion.settings)) is not None:
            not_applicable.append((suggestion.entry.name, why))
        else:
            kept.append(suggestion)
    return kept


NGRAM = "ngram-speculation"
NGRAM_MIN_COVERAGE = 0.06
MEASURE_IT = "; measure it: the step overhead on this model is not known"


def _ngram_gate(
    offered: list[Suggestion], situation: Situation, goal: Goal
) -> tuple[list[Suggestion], dict[str, str]]:
    """N-gram speculation, with prompt lookup replayed over this run's own prompts and answers:
    under the TPOT goal (a p95) the answers it helps least decide, as they are the slowest once
    speculating. Under every goal a replay below NGRAM_MIN_COVERAGE leaves it out. Above it, the
    step overhead on this model is not known: it could help, and no break-even claim is made. On
    a prompt-bound run, draft tokens take step budget before waiting prompts: under the TTFT goal
    it is left out. Where this run cannot be replayed, it says why."""
    if not any(s.entry.name == NGRAM for s in offered):
        return offered, {}
    m = situation.measurement
    if goal == "ttft" and prompt_bound(state_of(m, situation.settings.max_concurrent_requests)):
        return [s for s in offered if s.entry.name != NGRAM], {
            NGRAM: "on a prompt-bound run the draft tokens take step budget before waiting prompts"
        }
    run_dir, step_ms = analysis().this_run(), m.step_ms
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
            _unmeasured(replace(s, reason=s.reason + note)) if s.entry.name == NGRAM else s
            for s in offered
        ], {}
    if found.pooled < NGRAM_MIN_COVERAGE:
        return [s for s in offered if s.entry.name != NGRAM], {
            NGRAM: (
                f"replaying prompt lookup, drafts would supply {found.pooled:.1%} of the output; "
                f"below {NGRAM_MIN_COVERAGE:.0%} speculation worsens TPOT"
            )
        }
    if goal == "tpot":
        share = found.tail
        whose = (
            f"of the output of the {TAIL:.0%} of answers they help least (the slowest once "
            "speculating)"
        )
    else:
        share, whose = found.pooled, "of all output tokens"
    replay = (
        f"Replaying prompt lookup on this run's answers, drafts would supply {share:.1%} {whose}"
    )
    return [
        _unmeasured(replace(s, reason=f"{s.reason}. {replay}", tail_coverage=found.tail))
        if s.entry.name == NGRAM
        else s
        for s in offered
    ], {}


def _unmeasured(s: Suggestion) -> Suggestion:
    return replace(s, reason=s.reason + MEASURE_IT, tier="could_help")


GATES = frozenset({"fit", "quality"})
CHECKPOINT = "quantized-checkpoint"  # addresses prefill where the GPU runs a W8A8 format
STEP_LEVER = "prefill_batch_tokens"
STEP_DOWN_ENTRY = "step-budget-down"
STEP_ENTRY = {
    "throughput": "raise-prefill-batch",
    "ttft": "small-step-budget",
    "tpot": "lower-prefill-batch",
}


def _says(effects: Mapping[str, Effect], metric: str, direction: str) -> bool:
    effect = effects.get(metric)
    return effect is not None and effect.direction == direction


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
        gain = _predicted_gain(suggestion, goal)
        if effect.direction == "none" and gain is not None and gain >= LEAD_AT:
            note = (
                f"{effect.reason}, but this run's figures predict {gain:+.0%} on {GOAL_WORDS[goal]}"
            )
            kept.append(replace(suggestion, cost=f"{note}; {costs}"))
            continue
        if effect.direction == "none":
            helps = [GOAL_WORDS[m] for m in GOALS if _says(effects, m, "improves")]
            text = f"does not change {GOAL_WORDS[goal]} ({effect.reason})"
            text += f"; helps {' and '.join(helps)}" if helps else ""
            later.append(replace(suggestion, tier="could_help", cost=f"{text}; {costs}"))
            continue
        worse = [m for m in GOALS if m != goal and _says(effects, m, "worsens")]
        if ttft_limit is not None and "ttft" in worse and p95 is not None and p95 > ttft_limit:
            note = f"TTFT p95 is {p95:.0f} ms, past your limit of {ttft_limit:g} ms"
            later.append(replace(suggestion, tier="could_help", demoted=note,
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
    moving = [s for s in ranked if s.tier == "try_first" and STEP_LEVER in s.entry.levers]
    own = (
        STEP_DOWN_ENTRY
        if any(s.entry.name == STEP_DOWN_ENTRY for s in moving)
        else STEP_ENTRY[goal]
    )
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


LEAD_AT = 0.10  # a predicted gain on the goal that leads Try first, answers or not


def _predicted_gain(suggestion: Suggestion, goal: Goal) -> float | None:
    """The predicted relative gain on ``goal`` (positive when better); None without one."""
    return goal_gain(suggestion.predicted, goal)


def _leads(suggestion: Suggestion, goal: Goal, *, tools: bool = False) -> bool:
    """Whether a Try-first offer leads: a stack, or a change with a prior, whose ``score`` is at
    least LEAD_AT, or a single change predicted to gain at least LEAD_AT on ``goal``."""
    if suggestion.tier != "try_first":
        return False
    if (suggestion.members or suggestion.entry.prior_gain) and _score(
        suggestion, goal, tools=tools
    ) >= LEAD_AT:
        return True
    gain = _predicted_gain(suggestion, goal)
    return not suggestion.members and gain is not None and gain >= LEAD_AT


def _leads_shown(suggestion: Suggestion, suggested: Suggested) -> bool:
    return suggestion == suggested.led or _leads(
        suggestion, suggested.goal, tools=suggested.tool_calls
    )


def _changes_answers(suggestion: Suggestion) -> bool:
    return any(m.changes_answers for m in suggestion.members or (suggestion.entry,))


def _lossless_first(ordered: list[Suggestion]) -> list[Suggestion]:
    """Each stack's lossless remainder, where it is offered, just before the stack: it banks a
    sure gain before the step that needs its answers compared."""
    out = list(ordered)
    for stack in ordered:
        rest = lossless_remainder(stack)
        if not rest or len(rest) == len(stack.members):
            continue
        remainder = next(
            (s for s in out
             if s.members == rest or (len(rest) == 1 and not s.members and s.entry is rest[0])),
            None,
        )  # fmt: skip
        if remainder is not None and out.index(remainder) > out.index(stack):
            out.remove(remainder)
            out.insert(out.index(stack), remainder)
    return out


WEIGHT_PRECISION = ("fp8-weights", "bnb-4bit-weights")


def _is_try_first(suggestion: Suggestion, names: Collection[str]) -> bool:
    return suggestion.tier == "try_first" and suggestion.entry.name in names


def _both_precisions(names: Iterable[str]) -> bool:
    found = set(names)
    return bool(found & set(KV_PRECISION)) and bool(found & set(WEIGHT_PRECISION))


def _pair_offered(ranked: Sequence[Suggestion]) -> bool:
    """Whether Try first measures a weight precision and a KV precision in one step."""
    return any(
        s.tier == "try_first" and _both_precisions(m.name for m in s.members) for s in ranked
    )


def _holds_both_precisions(ranked: Sequence[Suggestion]) -> bool:
    return (
        not _pair_offered(ranked)
        and any(_is_try_first(s, KV_PRECISION) for s in ranked)
        and any(_is_try_first(s, WEIGHT_PRECISION) for s in ranked)
    )


def _kv_before_weights(ranked: list[Suggestion]) -> list[Suggestion]:
    """On a run that offers tools, unless Try first measures the two precisions in one step, its
    weight-precision offers that precede its first KV-precision offer move to just after it: the
    KV cache stores only K and V, while 8-bit weights also quantize every matmul's activations,
    which cost tool-call validity on this repository's own runs."""
    kv = next((i for i, s in enumerate(ranked) if _is_try_first(s, KV_PRECISION)), None)
    if kv is None or _pair_offered(ranked):
        return ranked
    early = [s for s in ranked[:kv] if _is_try_first(s, WEIGHT_PRECISION)]
    if not early:
        return ranked
    before = [s for s in ranked[:kv] if s not in early]
    return [*before, ranked[kv], *early, *ranked[kv + 1 :]]


def _score(suggestion: Suggestion, goal: str, *, tools: bool = False) -> float:
    context = analysis()
    return RuleScorer(goal, tools=tools).score(suggestion, context.kb_facts, context).value


def _ttft_basis(
    situation: Situation, context: AnalysisContext, cap: int | None
) -> tuple[Measurement, str | None, str | None]:
    """The measurement with the TTFT the TTFT offers rank on, a line saying which, and why the
    engine's figure was not used where a hold showed. Behind a tool-call parser hold of
    ``HELD_MIN`` to ``MAX_PARSER_HOLD`` decode steps, on a run that was not queueing (nothing
    waiting, no KV pressure, the clients running), the client's first content arrives long
    after the engine's first token, so it is the engine's median (the hold taken off the p95);
    ``--ttft-from first-token`` ranks on the first streamed chunk with a choice instead. The
    measured TTFT is unchanged."""
    measurement = situation.measurement
    if ("--ttft-from", FIRST_TOKEN) in zip(context.flags, context.flags[1:]):
        run_dir = context.this_run()
        rows = trace_facts.request_rows(run_dir) if run_dir is not None else []
        waits = [
            (row["first_chunk_ns"] - row["dispatch_ns"]) / 1e6
            for row in rows
            if row["outcome"] == "success" and row.get("first_chunk_ns") is not None
        ]
        if not waits:
            return measurement, None, None
        p50, p95 = nearest_rank(waits, 0.5), nearest_rank(waits, 0.95)
        return (
            replace(measurement, ttft_p50_ms=p50, ttft_p95_ms=p95),
            (
                f"TTFT offers are ranked on the first streamed chunk ({p95:.0f} ms p95), as "
                "--ttft-from first-token asks"
            ),
            None,
        )
    held = held_steps(measurement)
    server, client, tail = (
        measurement.server_ttft_p50_ms,
        measurement.ttft_p50_ms,
        measurement.ttft_p95_ms,
    )
    if held is None or server is None or client is None or tail is None:
        return measurement, None, None
    if held > MAX_PARSER_HOLD or kv_held_back(situation) or not saturated(measurement, cap):
        return measurement, None, QUEUE_TIME
    ranked = max(server, tail - (client - server))
    return (
        replace(measurement, ttft_p50_ms=server, ttft_p95_ms=ranked),
        (
            f"TTFT offers are ranked on the engine's time to first token ({ranked:.0f} ms p95, "
            f"{server:.0f} ms median): the tool-call parser holds the client's first content "
            f"for {held:.0f} decode steps"
        ),
        None,
    )


def ordered_for_goal(
    offered: Sequence[Suggestion],
    goal: Goal,
    measurement: Measurement,
    ttft_limit: float | None,
    estimate: Callable[[Suggestion], Estimate | None],
    *,
    past: Callable[[Suggestion], bool] = lambda s: False,
    stack: Callable[
        [list[Suggestion]], tuple[list[Suggestion], Suggestion | None]
    ] = lambda ranked: (ranked, None),
) -> tuple[list[Suggestion], tuple[str, ...], Bottleneck | None, bool]:
    """The suggestions in their final order, the names left out for the goal, the bottleneck a
    promoted change leads, and whether the report renders Try first in this list's order. The
    effect ranking comes first, then (throughput only) the estimate order and promotion (never
    of a change ``past`` a declared limit by its prediction), then the TTFT-limit rule: with the
    limit already broken, no TTFT-worsening change is promoted and the changes that lower TTFT
    move to the front. Then ``stack`` measures Try-first changes together and may name an offer
    that leads whatever its score. Last, the offers that lead (``_leads``) come first, highest
    ``score`` first and the lossless one on a tie, a stack's lossless remainder before it, and
    after them Try first's changes that cannot affect the answers come before those that can
    (``_safe_first``)."""
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
        candidates = [
            s for s in ranked_offers if (not broken or not _worsens(s, "ttft")) and not past(s)
        ]
        promoted, lead = _promoted(candidates, estimate)
        ranked_offers = [*promoted, *(s for s in ranked_offers if s not in candidates)]
    if broken:
        ranked_offers.sort(key=lambda s: not _improves(s, "ttft"))
    ranked_offers, first = stack(ranked_offers)
    tools = measurement.tool_calls is not None
    leading = [*([first] if first is not None else []), *sorted(
        (s for s in ranked_offers if s is not first and _leads(s, goal, tools=tools)),
        key=lambda s: (-_score(s, goal, tools=tools), _changes_answers(s)),
    )]  # fmt: skip
    rest = _safe_first([s for s in ranked_offers if s not in leading])
    ordered = _lossless_first([*leading, *rest])
    if measurement.tool_calls is not None:
        ordered = _kv_before_weights(ordered)
    return ordered, left_out, lead, goal != "throughput" or broken


def budget_refusal(
    engine: Engine, proposed: Settings, measurement: Measurement, version: str | None
) -> str | None:
    """Why a proposed step budget would not start: vLLM refuses a budget below the concurrency
    cap, which is the setting, else the engine's default for this version and GPU."""
    budget = proposed.prefill_batch_tokens
    if budget is None:
        return None
    cap = proposed.max_concurrent_requests or _default_cap(engine, version, measurement)
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


def _start_order(
    situation: Situation, engine: Engine, try_first: Sequence[Suggestion]
) -> list[Suggestion]:
    """``try_first`` with the steps that may change the answers first (each is judged, and the
    judge's wait overlaps the next steps), then the long starts, then the short starts and the
    ones that load the compiled graph; in their own order within each class."""

    def rank(suggestion: Suggestion) -> int:
        if any(m.quality_risk for m in suggestion.members or (suggestion.entry,)):
            return 0
        return 1 if _start_cost(situation, engine, suggestion) else 2

    return sorted(try_first, key=rank)


def _in_start_order(
    offers: list[Suggestion], situation: Situation, engine: Engine
) -> list[Suggestion]:
    """``offers`` with the runnable Try-first ones in ``_start_order``, among their own places."""
    slots = [i for i, s in enumerate(offers) if s.tier == "try_first" and s.command]
    out = list(offers)
    for i, offer in zip(slots, _start_order(situation, engine, [offers[i] for i in slots])):
        out[i] = offer
    return out


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
    setup once, so an override already given is replaced, not repeated; for a change offered once
    more after its answer check did not run, the run to reuse (``_once_more``)."""
    changes = engine.engine_args_between(current, suggested)
    env = sorted(filter(is_env, analysis().engine_args))
    words = ["tensward", "analyse", "--project", str(project)]
    for text in changes:
        words += ["--engine-arg", text]
    for text in env:
        words += ["--engine-env", text]
    again = _once_more(project, frozenset((*changes, *env)))
    return shlex.join([*words, *again, *analysis().flags])


def _once_more(project: Path, args: frozenset[str]) -> list[str]:
    """``--reuse-run RUN`` for a command of exactly the arguments of a run whose answer check did
    not run to a result, while that run may stand in for it from this run (``reusable_run``);
    ``--remeasure`` for another command of that change; nothing for any other command."""
    from .reuse import reusable_run

    context = analysis()
    found = next((r for r in context.retried if as_change(r.engine_args) == as_change(args)), None)
    if found is None:
        return []
    here = context.reported_run or getattr(context.this_run(), "name", None)
    if (
        here is not None
        and frozenset(found.engine_args) == args
        and reusable_run(project, found.run, here) is None
    ):
        return ["--reuse-run", found.run]
    return ["--remeasure"]


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
    secondary bottlenecks. After the failed gates, the group of a change that leads by its
    predicted gain (``_leads``) comes first; then a group whose changes all can affect the answers
    comes after the others."""
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
    rest = sorted(order[gates:], key=changing)
    kv = next((s for s in grouped if suggested.kv_first and _is_try_first(s, KV_PRECISION)), None)
    if kv is not None:
        home = _home(diagnosis, suggested, kv)
        if home in rest:
            rest.remove(home)
            rest.insert(0, home)
    if grouped and _leads_shown(grouped[0], suggested):
        home = _home(diagnosis, suggested, grouped[0])
        if home in rest:
            rest.remove(home)
            rest.insert(0, home)
    return order[:gates] + rest


def first_try(diagnosis: Diagnosis, suggested: Suggested) -> Suggestion | None:
    """The change the report shows first under Try first: a revert after a loss, else the first
    change in group order (failed gates first); None when it shows none."""
    if suggested.memory is not None and suggested.memory.revert is not None:
        return suggested.memory.revert
    shown = [
        s
        for group in try_first_order(diagnosis, suggested)
        for s in group_members(diagnosis, suggested, group)
    ]
    return shown[0] if shown else None
