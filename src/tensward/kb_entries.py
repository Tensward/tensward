"""Knowledge records in the suggestions. A measured or promoted lever record becomes a playbook
entry of its own; a documented one goes to "Worth measuring"; a guard, a tombstone or a measured
null acts on every offer, the playbook's included. A record whose value needs arithmetic names a
rule, and the action bound to that rule reads what the run measured and turns the rule's result
into settings and flags, or into the reason there is no offer."""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from .classify import NAMES
from .engines import Engine
from .engines.vllm import value_rules
from .engines.vllm.command import SPECULATIVE_CONFIG_FLAG
from .engines.vllm.flags import merge_json
from .facts import UNKNOWN, Facts
from .kb_condition import ANCHOR_KEYS
from .kb_schema import METRICS, Record, RecordHead, lever_key
from .levers import LEVERS
from .matcher import Layers, Match, Matches, Null, anchor_values
from .matcher import _values as record_values
from .playbook import Effect, Entry, Evidence, HeldBack, Situation, Suggestion, admitted, held_back
from .playbook_common import PREDICATES
from .scoring import RuleScorer
from .settings import Settings

WORTH_MEASURING = 3  # documented records listed at most
DOCUMENTED = "documented, not yet measured by Tensward: measure it before you keep it"
ENV_ONLY = "it changes the engine's environment, which a step command sets with --engine-env"
SUFFIX_ACCEPTANCE = 1.8  # tokens per step a suffix tree must expect before it is offered

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


def _flags(settings: Settings, flags: Mapping[str, Any]) -> Settings:
    return replace(settings, extra_args=merge_json(settings.extra_args, flags))


def _spec_tiers(inputs: Inputs, settings: Settings) -> Built:
    running = inputs.fact("run.running_p90")
    if running is None:
        return "the running count's 90th percentile is not known"
    speculation = inputs.engine.lever_value(settings, "speculation")
    if speculation is None or not speculation.draft_tokens:
        return "no speculative method is set to schedule: a draft model, an MTP or an EAGLE3 head"
    batch = settings.max_concurrent_requests or inputs.cap or 0
    tiers = value_rules.spec_tiers(running, speculation.draft_tokens, cut=None, max_batch=batch)
    if tiers is None:
        most = value_rules.SPEC_RUNNING_P90
        return f"{running:g} requests run at the 90th percentile, above {most}"
    name = f"{SPECULATIVE_CONFIG_FLAG}.num_speculative_tokens_per_batch_size"
    return _flags(settings, {name: tiers})


def _long_prefill_threshold(inputs: Inputs, settings: Settings) -> Built:
    if not inputs.prompt_tokens:
        return "the prompt lengths are not known"
    threshold = value_rules.long_prefill_threshold(inputs.prompt_tokens)
    if threshold is None:
        return "the prompt lengths do not fall in two classes"
    return _flags(settings, {"--long-prefill-token-threshold": str(threshold)})


def _suffix_acceptance(inputs: Inputs, settings: Settings) -> Built:
    accepted = value_rules.suffix_acceptance(inputs.answers())
    if accepted is None:
        return "this run kept no answers to replay"
    if accepted < SUFFIX_ACCEPTANCE:
        return (
            f"a suffix tree over this run's answers expects {accepted:.2f} words per step, "
            f"under {SUFFIX_ACCEPTANCE}"
        )
    kept = {k: v for k, v in settings.extra_args.items() if k != SPECULATIVE_CONFIG_FLAG}
    written = json.dumps({"method": "suffix"})
    return replace(settings, extra_args={**kept, SPECULATIVE_CONFIG_FLAG: written})


def _fp8_exclusions(inputs: Inputs, settings: Settings) -> Built:
    layers = inputs.fact("anatomy.layers")
    if layers is None:
        return "the checkpoint's layer count is not known"
    return _flags(settings, {"--quantization-config": value_rules.fp8_exclusions(layers)})


def _per_token_head_dtype(inputs: Inputs, settings: Settings) -> Built:
    dtype = value_rules.per_token_head_dtype(inputs.fact("gpu.cc"))
    if dtype is None:
        return "the GPU's compute capability is not known"
    return replace(settings, kv_cache_dtype=dtype)


def _cpu_kv_offload_size(inputs: Inputs, settings: Settings) -> Built:
    host = inputs.fact("host.memory_gib")
    if host is None:
        return "the host's memory is not known"
    possible = inputs.fact("workload.possible_prefix_hit")
    hit = inputs.fact("run.prefix_hit_rate")
    gap = None if possible is None or hit is None else possible - hit
    size = value_rules.cpu_kv_offload_size(host, gap)
    if size is None:
        if gap is None:
            return "the prefix hit rate is not known"
        if gap < value_rules.OFFLOAD_PREFIX_GAP:
            least = value_rules.OFFLOAD_PREFIX_GAP
            return (
                f"the prompts allow a prefix hit rate {gap * 100:.1f} points above the measured "
                f"one; at least {least * 100:.0f} are needed"
            )
        return f"the host has {host:.0f} GiB, too little for a CPU cache beside the rest"
    return _flags(settings, {"--kv-offloading-size": f"{size:g}"})


ACTIONS: Mapping[str, Action] = {
    "vllm:spec_tiers": Action(frozenset(), _spec_tiers),
    "vllm:long_prefill_threshold": Action(frozenset(), _long_prefill_threshold),
    "vllm:suffix_acceptance": Action(frozenset({"speculation"}), _suffix_acceptance),
    "vllm:fp8_exclusions": Action(frozenset(), _fp8_exclusions),
    "vllm:per_token_head_dtype": Action(frozenset({"kv_cache_precision"}), _per_token_head_dtype),
    "vllm:cpu_kv_offload_size": Action(frozenset(), _cpu_kv_offload_size),
}
if unbound := sorted(set(ACTIONS) ^ set(value_rules.RULES)):
    raise ImportError(f"value rules and their actions differ: {', '.join(unbound)}")


def rules(engine: Engine) -> frozenset[str]:
    """Every rule and predicate name a record may name with this Tensward and ``engine``."""
    return frozenset({*PREDICATES, *getattr(engine, "predicates", {}), *value_rules.RULES})


def guard_reason(matches: Matches, record: Record) -> str | None:
    """The reason a guard blocks ``record``'s change, looked up by the keys the record declares
    (its merge key, id and lever key), never by a key read back from the settings it builds."""
    change = record.change
    declared = () if change is None or change.lever_key is None else (change.lever_key,)
    return matches.blocked_by(record.merge_key, record.id, *declared)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunKnowledge:
    """The knowledge that held for one run, with the whole records behind its matches."""

    facts: Facts
    layers: Layers
    matches: Matches
    records: Mapping[str, Record]  # "id@rev" -> the record a match names
    inputs: Inputs

    def record(self, match: Match) -> Record:
        return self.records[match.record]

    def build(self, record: Record, settings: Settings, seen: tuple[str, ...] = ()) -> Built:
        """``settings`` with ``record``'s change and the changes it needs, or why there is no
        offer: a needed record missing or blocked by a guard, a value its action cannot compute,
        or a lever the engine cannot write."""
        change = record.change
        if change is None:
            return "the record changes nothing"
        for need in change.needs:
            needed = self._needed(need)
            if need in seen:
                return f"it needs {need}, which needs it in turn"
            if needed is None:
                return f"it needs {need}, which the installed knowledge does not hold"
            if why := guard_reason(self.matches, needed):
                return f"it needs {needed.title}, which is not applicable here: {why}"
            built = self.build(needed, settings, (*seen, record.id))
            if isinstance(built, str):
                return f"it needs {needed.title}: {built}"
            settings = built
        action = ACTIONS.get(change.rule or "")
        sets, flags = record_values(record, self.facts)
        engine = self.inputs.engine
        for lever, value in sets.items():
            if action is not None and lever in action.writes:
                continue
            if lever == "speculation" and isinstance(value, str):
                return f"speculation by {value} needs more than a method's name to write"
            try:
                settings = engine.with_lever(settings, lever, value)
            except ValueError as error:
                return str(error)
        if flags:
            settings = _flags(settings, flags)
        if action is not None:
            return action.build(self.inputs, settings)
        if change.rule is not None:
            made = {**PREDICATES, **getattr(engine, "predicates", {})}[change.rule](
                replace(self.inputs.situation, settings=settings), engine=engine
            )
            if not isinstance(made, Settings):
                return f"{change.rule} makes no change"
            settings = made
        return settings

    def _needed(self, record_id: str) -> Record | None:
        known = [r for r in self.records.values() if r.id == record_id]
        if not known:
            known = list(_all_records(self.layers, record_id))
        return max(known, key=lambda r: (r.rev, r.layer == "pack"), default=None)


def _all_records(layers: Layers, record_id: str) -> Iterable[Record]:
    source = layers.source
    if source is not None:
        found = [r for r in source.candidates(dict.fromkeys(ANCHOR_KEYS)) if r.id == record_id]
        heads = [r for r in found if isinstance(r, RecordHead)]
        yield from (r for r in found if isinstance(r, Record))
        yield from source.records(heads) if heads else ()
    yield from (r for r in layers.local if r.id == record_id)


def whole_records(layers: Layers, facts: Facts, family: str, matches: Matches) -> dict[str, Record]:
    """The whole records behind ``matches``' offers and "Worth measuring" list, by "id@rev"."""
    wanted = {m.record for m in (*matches.offers, *matches.worth_measuring)}
    found: dict[str, Record] = {f"{r.id}@{r.rev}": r for r in layers.local}
    source = layers.source
    if source is not None and wanted - set(found):
        heads = []
        for record in source.candidates(anchor_values(facts, family)):
            ref = f"{record.id}@{record.rev}"
            if ref not in wanted or ref in found:
                continue
            if isinstance(record, Record):
                found[ref] = record
            else:
                heads.append(record)
        found.update({f"{r.id}@{r.rev}": r for r in source.records(heads)} if heads else {})
    return {ref: record for ref, record in found.items() if ref in wanted}


def as_entry(record: Record, *, applies: Callable[[Situation], str | None], built: Built) -> Entry:
    """``record`` as a playbook entry that offers ``built``: named by its merge key, graded by
    its confidence, costed and risked by its costs, with its per-metric effects and its prior.
    It is opt-in, so only ``suggest.suggestions`` considers it."""
    effect, costs = record.effect, record.costs
    effects = {}
    if effect is not None:
        for metric in METRICS:
            if (found := getattr(effect, metric)) is not None:
                said = found.reason or effect.reason or record.title
                effects[metric] = Effect(found.direction, said)
    risk = None if costs is None else costs.answer_risk
    change = record.change
    levers = frozenset(name for name in (change.levers if change else ()) if name in LEVERS)
    sets = {k: v for k, v in (change.sets if change else {}).items() if isinstance(v, str)}
    return Entry(
        name=record.merge_key,
        addresses=frozenset(name for name in record.anchors.get("diag", ()) if name in NAMES),
        applies=applies,
        apply=lambda situation: built if isinstance(built, Settings) else situation.settings,
        evidence=cast(Evidence, record.confidence.grade or "weak"),
        costs=costs.text if costs else "",
        quality_risk=bool(costs and costs.changes_answers),
        levers=levers,
        engines=frozenset() if record.engine == "any" else frozenset({record.engine}),
        sets=sets,
        effects=effects,
        opt_in=True,
        changes_answers=bool(costs and costs.changes_answers),
        answer_risk=risk if risk in ("low", "medium", "high") else None,
        prior=None if effect is None else effect.prior_text,
        prior_gain={} if effect is None else dict(effect.prior_gain),
        checks_answers=bool(costs and costs.checks_answers),
    )


def kb_entries(known: RunKnowledge, version: str | None) -> tuple[list[Entry], list[Match]]:
    """The entries of the measured and promoted offers whose change the engine realises, each
    offered with its change built (or held back with the reason there is none), and the offers
    that change the engine's environment, which never join the playbook's path."""
    engine, situation = known.inputs.engine, known.inputs.situation
    capabilities = engine.capabilities(version)
    entries, env = [], []
    for match in known.matches.offers:
        record = known.record(match)
        if record.change is not None and record.change.env:
            env.append(match)
            continue
        why = guard_reason(known.matches, record)
        built = known.build(record, situation.settings)
        reason: str | None = record.title
        if why is not None or isinstance(built, str):
            reason = HeldBack(why or str(built))
        elif built == situation.settings:
            reason = None
        entry = as_entry(record, applies=_said(reason), built=built)
        if admitted(entry, capabilities):
            entries.append(held_back(entry, engine.hold_backs))
    return entries, env


def _said(reason: str | None) -> Callable[[Situation], str | None]:
    return lambda situation: reason


def change_key(
    engine: Engine, before: Settings, after: Settings, defaults: Mapping[str, Any] = {}
) -> str:
    """The lever key of moving ``before`` to ``after``: each lever the engine reads differently
    and each flag that changed. A numeric lever ``before`` leaves unset moves up or down from
    its value in ``defaults``, where the engine's own value is known."""
    sets, was = {}, {}
    for name in LEVERS:
        try:
            old, new = engine.lever_value(before, name), engine.lever_value(after, name)
        except ValueError:
            continue
        if old != new:
            sets[name], was[name] = new, defaults.get(name) if old is None else old
    flags = {f: v for f, v in after.extra_args.items() if before.extra_args.get(f) != v}
    return lever_key(sets, flags, before=was)


def _parts(keys: Iterable[str]) -> list[str]:
    return [*keys, *(part for key in keys if "+" in key for part in key.split("+"))]


def judged(
    offers: Sequence[Suggestion],
    known: RunKnowledge,
    declared: Mapping[str, Record],
    not_applicable: list[tuple[str, str]],
) -> tuple[list[Suggestion], dict[str, Null]]:
    """``offers`` after the guards and the measured nulls, and the null that acted on each by
    name. An offer a guard blocks, or a null removes, goes to ``not_applicable`` with the
    reason; one a null demotes goes to Could help with it, and so does an offer of a measured
    record (a promoted one keeps its tier). A knowledge offer (in ``declared``, by name) is
    known by the keys its record declares; a playbook offer by its name and the lever key of
    its change."""
    matches, engine = known.matches, known.inputs.engine
    if not (matches.blocked or matches.nulls or declared):
        return list(offers), {}
    before = known.inputs.situation.settings
    defaults = {
        "kv_memory_fraction": engine.default_kv_memory_fraction,
        "prefill_batch_tokens": known.inputs.situation.measurement.step_budget_tokens,
        "max_concurrent_requests": known.inputs.cap,
    }
    kept, nulled = [], {}
    for offer in offers:
        name = offer.entry.name
        record = declared.get(name)
        if record is not None:
            change = record.change
            keys = [name, record.id, *([change.lever_key] if change and change.lever_key else [])]
            why = guard_reason(matches, record)
        else:
            keys = [name, change_key(engine, before, offer.settings or before, defaults)]
            why = matches.blocked_by(*keys)
        if why is not None:
            not_applicable.append((name, why))
            continue
        null = next((matches.nulls[k] for k in _parts(keys) if k in matches.nulls), None)
        if null is not None and null.removes:
            not_applicable.append((name, null.reason))
            continue
        if null is not None:
            nulled[name] = null
            costs = offer.cost or offer.entry.costs
            offer = replace(offer, tier="could_help", demoted=null.reason,
                            cost=f"{null.reason}; {costs}")  # fmt: skip
        elif record is not None and record.status == "measured":
            offer = replace(offer, tier="could_help")
        kept.append(offer)
    return kept, nulled


def _plain(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def kb_block(match: Match, matches: Matches) -> dict[str, Any]:
    """The ``kb`` block of an offer a pack or local record matched."""
    measured = None
    if match.measured is not None:
        found = match.measured
        measured = {
            "setups": found.setups,
            "pairs": found.pairs,
            **{k: [m.low, m.high] for k, m in sorted(found.metrics.items())},
        }
    return {
        "record": match.record,
        "layer": match.layer,
        "pack": matches.source.get("pack"),
        "status": match.status,
        "confidence": match.confidence,
        "fired_on": [
            {"fact": t.fact, "op": t.op, "value": _plain(t.value), "observed": _plain(t.observed)}
            for t in match.explanation.fired_on
        ],
        "unknown": list(match.explanation.unknown),
        "measured": measured,
        "evidence": list(match.evidence),
        "direction_note": match.direction_note,
        "facts_digest": matches.facts_digest,
    }


def null_block(null: Null, matches: Matches) -> dict[str, Any]:
    """The ``kb`` block of an offer a measured null demoted."""
    return {
        "null": {
            "record": null.record, "reason": null.reason, "launches": null.launches,
            "removes": null.removes, "evidence": list(null.evidence),
        },
        "pack": matches.source.get("pack"),
        "facts_digest": matches.facts_digest,
    }  # fmt: skip


def with_env(command: str, env: Mapping[str, str]) -> str:
    """``command`` with ``env`` as its ``--engine-env`` values: a variable the run already set
    takes the record's value, once."""
    words, kept = shlex.split(command), []
    pairs = iter(range(len(words)))
    for i in pairs:
        if words[i] == "--engine-env" and i + 1 < len(words):
            if words[i + 1].partition("=")[0] not in env:
                kept += words[i : i + 2]
            next(pairs)
            continue
        kept.append(words[i])
    added = [w for name in sorted(env) for w in ("--engine-env", f"{name}={env[name]}")]
    return shlex.join([*kept, *added])


def listed(
    matches: Iterable[Match],
    known: RunKnowledge,
    command: Callable[[Settings, bool], str | None],
    not_applicable: list[tuple[str, str]],
    *,
    reason: Callable[[Record], str],
    demoted: Callable[[Record], str | None],
) -> list[Suggestion]:
    """Suggestions outside the playbook's path, each with its change built and its command
    (from ``command``, told whether the change sets environment, which it then carries as
    ``--engine-env`` values); a record whose change cannot be built goes to ``not_applicable``
    with the reason, and one that changes nothing is left out."""
    situation = known.inputs.situation
    found = []
    for match in matches:
        record = known.record(match)
        env = dict(record.change.env) if record.change is not None else {}
        built = known.build(record, situation.settings)
        if isinstance(built, str):
            not_applicable.append((record.merge_key, built))
            continue
        if built == situation.settings and not env:
            continue
        shown = command(built, bool(env))
        if shown is not None and env:
            shown = with_env(shown, env)
        entry = as_entry(record, applies=_said(record.title), built=built)
        found.append(
            Suggestion(
                entry=entry,
                reason=reason(record),
                tier="could_help",
                addresses=entry.addresses,
                command=shown,
                settings=built,
                demoted=demoted(record),
            )  # fmt: skip
        )
    return found


def _improves(record: Record) -> bool:
    effect = record.effect
    said = [] if effect is None else [e for k in METRICS if (e := getattr(effect, k)) is not None]
    return not said or any(e.direction == "improves" for e in said)


def worth_measuring(
    known: RunKnowledge,
    command: Callable[[Settings, bool], str | None],
    not_applicable: list[tuple[str, str]],
    goal: str,
    *,
    tools: bool,
) -> list[Suggestion]:
    """The documented records that fired, no guard blocks and that expect some metric to
    improve (a quality repair expects none), at most WORTH_MEASURING: by the scorer, then those
    expected to improve ``goal`` first, then by id."""
    hopeful = [m for m in known.matches.worth_measuring if _improves(known.record(m))]
    found = listed(
        hopeful, known, command, not_applicable,
        reason=lambda record: f"{record.title}; {DOCUMENTED}", demoted=lambda record: None,
    )  # fmt: skip
    scorer = RuleScorer(goal, tools=tools)

    def order(offer: Suggestion) -> tuple[float, bool]:
        value = scorer.score(offer, known.facts, None).value
        effect = offer.entry.effects.get(goal)
        return -value, effect is None or effect.direction != "improves"

    return sorted(found, key=order)[:WORTH_MEASURING]


def env_offers(
    matches: Iterable[Match],
    known: RunKnowledge,
    command: Callable[[Settings, bool], str | None],
    not_applicable: list[tuple[str, str]],
) -> list[Suggestion]:
    """The measured and promoted offers that change the engine's environment: in Could help,
    never promoted, never stacked."""
    return listed(
        matches, known, command, not_applicable,
        reason=lambda record: record.title, demoted=lambda record: ENV_ONLY,
    )  # fmt: skip
