"""The runtime matcher: which knowledge records hold for one run, and how the layers that hold
them combine. The layers are the playbook in the wheel, an installed knowledge source (the
``pack``) and the user's own records (``local``).

Guards from every layer block as a union, the most restrictive reason shown. A pack tombstone
hides the playbook entry it names. Lever records that share a merge key make one offer: the
highest-status record that fired decides it, and the effect shown follows local measured >
promoted > measured. A record naming a playbook entry never fires on its own: it explains the
entry once the pipeline has offered it (:func:`annotate_offered`).

A lever record whose effect on the run's goal is "none" (a measured null) or "worsens" (a
measured loss) makes no offer. It demotes the offer of its merge key to Could help, and removes
it once :data:`NULL_REMOVES` independent Tensward launches back it; agent screens and other
priors only demote. A launch counts once however many references name it, and only when it ran on
the run's model family, GPU architecture and engine minor and its measured change on the goal
fell where the record says (inside the noise band for a null). The measured numbers of such a
record stay in the merge: the offer or the null shows the local result, its noise and its
provenance.

:func:`match` is pure: the same facts and the same opened layers give equal ``Matches``.
"""

from __future__ import annotations

import functools
import sys
from collections import Counter
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, TypeVar, cast

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from . import kb_local
from .engines import FLAG_RULES
from .extensions import load_knowledge
from .facts import UNKNOWN, Facts
from .kb_condition import Explanation, Memo, Term, Truth, evaluate, explain
from .kb_schema import (
    METRICS,
    Change,
    EngineChange,
    Kind,
    Layer,
    Measured,
    Record,
    RecordHead,
    Status,
    engine_minor,
)
from .levers import LEVERS

if TYPE_CHECKING:
    from .context import AnalysisContext
    from .playbook import Suggestion

_PREDICATE_KEYS = ("applies", "apply", "low_bar", "blocked", "unblocks", "start")
_SCREEN = "an agent screen"
_STATUS_RANK = {"promoted": 0, "measured": 1, "documented": 2}
_OFFERED = frozenset({"promoted", "measured"})
_METRIC_LABEL = {"throughput": "throughput", "ttft": "TTFT", "tpot": "TPOT"}
_NOTHING = Explanation((), ())
NULL_REMOVES = 2  # independent Tensward launches inside the band that remove an offer


class KnowledgeSource(Protocol):
    """An opened set of records, such as an installed pack."""

    name: str
    version: str
    content_digest: str

    def candidates(self, keys: Mapping[str, Sequence[str] | None]) -> Iterator[Record | RecordHead]:
        """The records whose anchors admit the run's ``keys`` (per ``ANCHOR_KEYS`` the run's
        values; None when unknown, and that key is not narrowed), in id order: whole records,
        or heads whose whole records :meth:`records` gives."""
        ...

    def records(self, heads: Sequence[RecordHead]) -> list[Record]:
        """The whole records of ``heads``, in their order."""
        ...

    def tombstones(self) -> Mapping[str, str]:
        """Playbook entry name -> why the source retired it."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class Match:
    record: str  # "id@rev" of the record that decides the offer
    layer: Layer
    status: Status
    kind: Kind
    merge_key: str
    sets: Mapping[str, Any]  # after variants
    flags: Mapping[str, Any]
    explanation: Explanation
    measured: Measured | None  # local measured, else promoted, else measured
    evidence: tuple[str, ...]  # "<layer>:<kind>:<ref>" from every merged record
    confidence: str
    # "your run <id> measured ...": the user's own run measured the opposite direction beyond
    # noise, and the offer ranks as could_help.
    direction_note: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class Null:
    record: str  # "id@rev"
    removes: bool  # else it demotes to Could help
    reason: str
    launches: int  # independent eligible launches behind it
    measured: Measured | None  # local measured, else promoted, else measured
    evidence: tuple[str, ...]  # "<layer>:<kind>:<ref>" from every record of the merge key
    direction_note: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class Matches:
    """What held for one run. ``offers`` are measured and promoted levers merged by merge key;
    after :func:`annotate_offered` they also explain the playbook entries the pipeline offered.
    ``worth_measuring`` are documented levers that fired and no guard blocks. ``blocked`` maps a
    merge key, lever key or record id (or "*") to the guard's reason. ``pending`` holds the
    records naming a playbook entry, for :func:`annotate_offered`. ``nulls`` maps a merge key
    to the measured null that demotes or removes its offer."""

    offers: tuple[Match, ...]
    worth_measuring: tuple[Match, ...]
    blocked: Mapping[str, str]
    never: frozenset[frozenset[str]]
    independent: tuple[Record, ...]
    tombstoned: Mapping[str, str]
    unknown: tuple[tuple[str, tuple[str, ...]], ...]
    skipped: Mapping[str, int]
    source: Mapping[str, Any]
    facts_digest: str
    pending: tuple[Record, ...] = ()
    nulls: Mapping[str, Null] = field(default_factory=dict)

    def by_entry(self, name: str) -> Match | None:
        return next((offer for offer in self.offers if offer.merge_key == name), None)

    def blocked_by(self, *keys: str) -> str | None:
        """The reason a guard blocks a change known by any of ``keys`` (a combined lever key
        also by each of its parts), or None."""
        named = ("*", *keys, *(part for key in keys if "+" in key for part in key.split("+")))
        return next((self.blocked[key] for key in named if key in self.blocked), None)


@dataclass(frozen=True, slots=True, kw_only=True)
class Layers:
    playbook: frozenset[str]  # the loaded playbook's entry names
    source: KnowledgeSource | None
    source_status: Mapping[str, Any]
    local: tuple[Record, ...]
    pins: Mapping[str, tuple[int, ...]] = field(default_factory=dict)  # engine -> approved pins


class Scorer(Protocol):
    name: str

    def score(self, offer: Suggestion, facts: Facts, context: AnalysisContext) -> Score: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class Score:
    value: float
    parts: Mapping[str, float]  # e.g. {"gain": 0.18, "p_adopt": 0.9}
    scorer: str  # "rules/1"


class _Fired(NamedTuple):
    record: Record
    explanation: Explanation
    sets: Mapping[str, Any]
    flags: Mapping[str, Any]


_R = TypeVar("_R", bound="Record | RecordHead")


def _for_engine(record: _R, engine_family: str, levers: frozenset[str]) -> _R | None:
    """The view of ``record`` for a project on ``engine_family`` whose engine realises
    ``levers``: None for another engine's record. An ``any`` record becomes the engine's own
    (its ``on`` table merged into the change) when it names that engine, or names no engine and
    needs only levers the engine realises; otherwise it stays ``any`` for the matcher to skip as
    ``not_for_engine``. A whole record's per-engine ``measured`` collapses to the engine's entry,
    and its status falls to ``documented`` without one. A view is its own view."""
    if record.engine not in (engine_family, "any"):
        return None
    if isinstance(record, Record):
        record = cast(_R, _collapse(record, engine_family))
    if record.engine != "any":
        return record
    change = record.change
    table = None if change is None else change.on.get(engine_family)
    if change is not None and change.on:
        if table is None:
            return record
        return replace(
            record,
            engine=engine_family,
            engine_version=table.engine_version,
            change=_on_change(change, table),
            scope="any",
        )
    named = set() if change is None else {n for n in (*change.levers, *change.sets) if n in LEVERS}
    if named <= levers:
        return replace(record, engine=engine_family, scope="any")
    return record


def _on_change(change: Change, table: EngineChange) -> Change:
    return replace(
        change,
        flags={**change.flags, **table.flags},
        env={**change.env, **table.env},
        sets={**change.sets, **table.sets},
        blocks=(*change.blocks, *(b for b in table.blocks if b not in change.blocks)),
        lever_key=table.lever_key or change.lever_key,
        on={},
    )


def _collapse(record: Record, engine_family: str) -> Record:
    measured = record.measured
    if not isinstance(measured, Mapping):
        return record
    entry = measured.get(engine_family)
    if record.status == "retired":
        status: Status = "retired"
    else:
        status = "documented" if entry is None else entry.status or record.status
    if entry is not None:
        return replace(record, measured=entry, status=status)
    seen = tuple(item for item in record.evidence if item.kind != "agent")
    return replace(record, measured=None, status=status, evidence=seen)


class _EngineSource:
    """A knowledge source whose records come out as views for one engine and its levers."""

    def __init__(self, source: KnowledgeSource, engine_family: str, levers: frozenset[str]):
        self._source, self._family, self._levers = source, engine_family, levers
        self._raw: dict[tuple[str, int], Record | RecordHead] = {}
        self.name, self.version = source.name, source.version
        self.content_digest = source.content_digest

    def candidates(self, keys: Mapping[str, Sequence[str] | None]) -> Iterator[Record | RecordHead]:
        for record in self._source.candidates(keys):
            self._raw[record.id, record.rev] = record
            if (view := _for_engine(record, self._family, self._levers)) is not None:
                yield view

    def records(self, heads: Sequence[RecordHead]) -> list[Record]:
        raw = [self._raw.get((head.id, head.rev), head) for head in heads]
        found = self._source.records([head for head in raw if isinstance(head, RecordHead)])
        return [cast(Record, _for_engine(record, self._family, self._levers)) for record in found]

    def tombstones(self) -> Mapping[str, str]:
        return self._source.tombstones()


def open_layers(
    playbook_names: Collection[str], *, engine_family: str, levers: frozenset[str]
) -> Layers:
    """The installed knowledge source (when one loads), as views for ``engine_family`` whose
    engine realises ``levers``, and the local records."""
    source: KnowledgeSource | None = None
    status: dict[str, Any] = {"pack": None, "reason": None}
    knowledge = load_knowledge()
    if knowledge is not None:
        try:
            source, status = knowledge.open(), dict(knowledge.status())
        except Exception as error:  # a knowledge source's own failure must never stop analyse
            reason = f"the knowledge source failed ({type(error).__name__}: {error})"
            print(f"knowledge source not loaded: {reason}", file=sys.stderr)
            source, status = None, {"pack": None, "reason": reason}
    return Layers(
        playbook=frozenset(playbook_names),
        source=None if source is None else _EngineSource(source, engine_family, levers),
        source_status=status,
        pins={engine: tuple(found) for engine, found in (status.get("engine_pins") or {}).items()},
        local=tuple(
            view
            for record in kb_local.local_records()
            if (view := _for_engine(record, engine_family, levers)) is not None
        ),
    )


def anchor_values(facts: Facts, engine_family: str) -> dict[str, tuple[str, ...] | None]:
    """The run's value(s) for each index key; None where the facts do not know it."""

    def known(name: str) -> tuple[str, ...] | None:
        value = facts[name]
        return None if value is UNKNOWN else (str(value),)

    primary, fired = facts["diag.primary"], facts["diag.fired"]
    diag = None
    if primary is not UNKNOWN or fired is not UNKNOWN:
        found = {str(name) for name in fired} if isinstance(fired, frozenset) else set()
        diag = tuple(sorted(found if primary is UNKNOWN else {*found, str(primary)}))
    return {
        "engine": (engine_family, "any"),
        "family": known("model.family"),
        "arch": known("gpu.arch"),
        "diag": diag,
        "goal": known("goal"),
    }


def match(
    facts: Facts,
    layers: Layers,
    *,
    engine_family: str,
    engine_version: str | None,
    tensward_version: str,
    rules: Collection[str],
    levers: frozenset[str],
) -> Matches:
    """The records of every layer that hold for ``facts``, in id order. A record is skipped and
    counted when it needs a newer Tensward, a rule or predicate missing from ``rules``, or a
    flag the engine does not allow; it is unknown when a fact it reads is."""
    source = layers.source
    found = () if source is None else tuple(source.candidates(anchor_values(facts, engine_family)))
    records = _latest((*found, *layers.local), engine_family, levers)
    skipped: Counter[str] = Counter()
    memo: Memo = {}
    held: list[tuple[Record | RecordHead, Explanation]] = []
    unknown: list[tuple[str, tuple[str, ...]]] = []
    waiting: list[Record | RecordHead] = []
    for record in records:
        reason = _skip(record, engine_family, engine_version, tensward_version, rules)
        if reason is None and record.kind == "lever" and record.oss_entry in layers.playbook:
            waiting.append(record)
            continue
        if reason is None and record.code:
            reason = "missing_entry"  # a coded record fires only through its playbook entry
        if reason is not None:
            skipped[reason] += 1
            continue
        truth, explanation = _evaluate(record, facts, engine_version, memo)
        if truth is None:
            unknown.append((_ref(record), explanation.unknown))
        elif truth:
            held.append((record, explanation))
    whole = _whole(source, [*(record for record, _ in held), *waiting], engine_family, levers)
    fired = [
        _Fired(record, explanation, *_values(record, facts))
        for record, (_, explanation) in zip(whole[: len(held)], held, strict=True)
    ]
    goal = facts["goal"]
    nulls = [f for f in fired if _null(f.record, goal)]
    for record in whole[len(held) :]:
        if _null(record, goal):
            truth, explanation = _evaluate(record, facts, engine_version, memo)
            if truth is True:
                nulls.append(_Fired(record, explanation, *_values(record, facts)))
    adverse = _grouped(nulls)
    pending = [record for record in whole[len(held) :] if not _null(record, goal)]
    fired = [f for f in fired if not _null(f.record, goal)]
    blocked = _blocked([f for f in fired if f.record.kind == "guard"], records)
    offers, worth = [], []
    for members in _grouped(f for f in fired if f.record.kind == "lever").values():
        merged = _merged(members, facts, adverse.get(members[0].record.merge_key, ()))
        if merged.status in _OFFERED:
            offers.append(merged)
        elif not _blocks(blocked, members):
            worth.append(merged)
    interactions = [f.record for f in fired if f.record.kind == "interaction"]
    tombstones = {} if source is None else source.tombstones()
    return Matches(
        offers=_in_order(offers),
        worth_measuring=_in_order(worth),
        blocked=blocked,
        never=frozenset(
            frozenset((a, b))
            for record in interactions
            if record.change is not None and record.change.relation != "independent"
            for a in record.change.first
            for b in record.change.second
            if a != b
        ),
        independent=tuple(
            r for r in interactions if r.change and r.change.relation == "independent"
        ),
        tombstoned={
            name: reason for name, reason in sorted(tombstones.items()) if name in layers.playbook
        },
        unknown=tuple(unknown),
        skipped=dict(sorted(skipped.items())),
        source={
            "pack": None if source is None else source.version,
            "content_digest": None if source is None else source.content_digest,
            "reason": layers.source_status.get("reason"),
        },
        facts_digest=facts.digest(),
        pending=tuple(pending),
        nulls=_nulls(adverse, facts, engine_family, layers.pins.get(engine_family, ())),
    )


def annotate_offered(
    matches: Matches, facts: Facts, layers: Layers, offered: Collection[str]
) -> Matches:
    """Adds the records naming the playbook entries the pipeline offered: the explanation from
    the declared summary, unknowns listed. A coded record whose summary is false while its entry
    was offered is counted under ``skipped["summary_disagrees"]``; the offer stands as the
    playbook made it."""
    version = facts["engine.version"]
    engine_version = version if isinstance(version, str) else None
    skipped: Counter[str] = Counter(matches.skipped)
    memo: Memo = {}
    kept: list[_Fired] = []
    for record in matches.pending:
        name = record.oss_entry
        if name not in offered or name not in layers.playbook or name in matches.tombstoned:
            continue
        truth, explanation = _evaluate(record, facts, engine_version, memo)
        if truth is False:
            if record.code:
                skipped["summary_disagrees"] += 1
            continue
        kept.append(_Fired(record, explanation, *_values(record, facts)))
    added = [_merged(members, facts) for members in _grouped(kept).values()]
    return replace(
        matches,
        offers=_in_order([*matches.offers, *added]),
        skipped=dict(sorted(skipped.items())),
        pending=(),
    )


def _null(record: Record, goal: object) -> bool:
    """Whether ``record`` is a lever measured to change nothing on ``goal``, or to worsen it."""
    if record.kind != "lever" or record.effect is None or goal not in METRICS:
        return False
    effect = getattr(record.effect, str(goal))
    if effect is None or effect.direction not in ("none", "worsens") or effect.basis == "derived":
        return False
    launches = any(
        launch.outcome.get(str(goal)) == effect.direction
        for item in _measured_of(record)
        for launch in item.launches
    )
    screened = any(item.kind == "agent" for item in record.evidence)
    return launches or screened


def _nulls(
    adverse: Mapping[str, Sequence[_Fired]],
    facts: Facts,
    engine_family: str,
    pins: Sequence[int] = (),
) -> dict[str, Null]:
    """Per merge key, the null that removes its offer, else the first that demotes it. Removal
    counts the independent eligible launches over the key's records, not their references."""
    family, arch, version = facts["model.family"], facts["gpu.arch"], facts["engine.version"]
    minor = engine_minor(engine_family, version if isinstance(version, str) else None, pins)
    goal = str(facts["goal"])
    found: dict[str, Null] = {}
    for key, members in adverse.items():
        counted: set[str] = set()
        for member in members:
            want = getattr(member.record.effect, goal).direction
            counted.update(
                launch.id
                for item in _measured_of(member.record)
                for launch in item.launches
                if (launch.engine, launch.family, launch.arch, launch.engine_minor)
                == (engine_family, family, arch, minor)
                and launch.outcome.get(goal) == want
            )
        removes = len(counted) >= NULL_REMOVES
        record = min(members, key=lambda m: m.record.id).record
        effect = getattr(record.effect, goal)
        what = "null" if effect.direction == "none" else "worse"
        where = _evidence_setup(record, goal, engine_family, (family, arch, minor))
        seen = "seen" if where == _SCREEN else "measured"
        said = f"{seen} {what} on {where} ({_ref(record)})"
        records = [m.record for m in members]
        found[key] = Null(
            record=_ref(record),
            removes=removes,
            reason=f"not offered: {said}" if removes else said,
            launches=len(counted),
            measured=_shown(records),
            evidence=_evidence(records),
            direction_note=_direction_note([], members, facts),
        )
    return dict(sorted(found.items()))


def _evidence_setup(
    record: Record, goal: str, engine_family: str, cell: tuple[object, object, object]
) -> str:
    """The setup the null was seen on, read from the record's own launches (the run's cell first),
    else the agent screen."""
    want = getattr(record.effect, goal).direction
    seen = sorted(
        (
            (launch.family, launch.arch, launch.engine_minor)
            for item in _measured_of(record)
            for launch in item.launches
            if launch.outcome.get(goal) == want
        ),
        key=lambda seen_cell: (seen_cell != cell, str(seen_cell)),
    )
    if not seen:
        return _SCREEN
    family, arch, minor = seen[0]
    parts = (family, arch, None if minor is None else f"{engine_family} {minor}")
    return ", ".join(part for part in parts if part) or "its launches"


def _own(record: Record) -> Measured | None:
    """The record's measured block (a view holds only its own engine's)."""
    return record.measured if isinstance(record.measured, Measured) else None


def _measured_of(record: Record) -> list[Measured]:
    return [] if (own := _own(record)) is None else [own]


def _shown(records: Iterable[Record]) -> Measured | None:
    """The measured block that is shown: a local one, else a promoted one, else any."""
    shown = min(
        (r for r in records if _own(r) is not None),
        key=lambda r: (r.layer != "local", r.status != "promoted", r.id),
        default=None,
    )
    return None if shown is None else _own(shown)


def _evidence(records: Iterable[Record]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            f"{record.layer}:{item.kind}:{item.ref}"
            for record in records
            for item in record.evidence
        )
    )


def _whole(
    source: KnowledgeSource | None,
    found: Sequence[Record | RecordHead],
    engine_family: str,
    levers: frozenset[str],
) -> list[Record]:
    """``found`` as whole records, viewed for the engine: the heads among them read from
    ``source``."""
    heads = [record for record in found if isinstance(record, RecordHead)]
    loaded = iter(()) if not heads else iter(_read(source, heads))
    whole = [record if isinstance(record, Record) else next(loaded) for record in found]
    return [cast(Record, _for_engine(record, engine_family, levers)) for record in whole]


def _read(source: KnowledgeSource | None, heads: Sequence[RecordHead]) -> list[Record]:
    assert source is not None
    return source.records(heads)


def _ref(record: Record | RecordHead) -> str:
    return f"{record.id}@{record.rev}"


def _in_order(found: Iterable[Match]) -> tuple[Match, ...]:
    return tuple(sorted(found, key=lambda m: m.record))


def _latest(
    records: Iterable[Record | RecordHead], engine_family: str, levers: frozenset[str]
) -> tuple[Record | RecordHead, ...]:
    """One record per id, in id order, viewed for the engine: the higher rev, the pack's on a
    tie; a local record never replaces a pack guard."""
    chosen: dict[str, Record | RecordHead] = {}
    for found in records:
        record = _for_engine(found, engine_family, levers)
        if record is None:
            continue
        held = chosen.get(record.id)
        if held is None or _replaces(record, held):
            chosen[record.id] = record
    return tuple(chosen[key] for key in sorted(chosen))


def _replaces(new: Record | RecordHead, held: Record | RecordHead) -> bool:
    if held.layer == "pack" and held.kind == "guard":
        return False
    if new.layer == "pack" and new.kind == "guard":
        return True
    return new.rev > held.rev or (new.rev == held.rev and new.layer == "pack")


def _skip(
    record: Record | RecordHead,
    engine_family: str,
    engine_version: str | None,
    tensward_version: str,
    rules: Collection[str],
) -> str | None:
    """Why this Tensward cannot use ``record``, or None."""
    if record.status == "retired":
        return "retired"
    if record.engine == "any":
        return "not_for_engine"
    if record.requires_tensward and _newer(record.requires_tensward, tensward_version):
        return "requires_tensward"
    if any(name not in rules for name in _rule_names(record)):
        return "missing_rule"
    flags = {} if record.change is None else record.change.flags
    env = {} if record.change is None else record.change.env
    flag_rules = FLAG_RULES.get(engine_family)
    if flag_rules is None:
        return "flag_not_allowed" if flags else "env_not_allowed" if env else None
    if not all(flag_rules.allowed(engine_version, flag) for flag in flags):
        return "flag_not_allowed"
    if not all(flag_rules.env_allowed(engine_version, name) for name in env):
        return "env_not_allowed"
    return None


@functools.lru_cache(maxsize=256)
def _newer(required: str, running: str) -> bool:
    return Version(required) > Version(running)


def _rule_names(record: Record | RecordHead) -> list[str]:
    code = record.code
    if not code:
        return [] if record.change is None or record.change.rule is None else [record.change.rule]
    names = [code[key] for key in _PREDICATE_KEYS if isinstance(code.get(key), str)]
    gates = code.get("gates", ())
    names += [gate for gate in gates if isinstance(gate, str)] if isinstance(gates, list) else []
    if record.change is not None and record.change.rule is not None:
        names.append(record.change.rule)
    return names


@functools.lru_cache(maxsize=4096)
def _within(specifier: str, version: str | None) -> Truth:
    if version is None:
        return None
    try:
        return SpecifierSet(specifier).contains(Version(version), prereleases=True)
    except (InvalidSpecifier, InvalidVersion):
        return None


def _checks(
    record: Record | RecordHead, facts: Facts, engine_version: str | None
) -> Iterator[tuple[Term, Truth]]:
    """The engine version and validity terms, each with whether it holds."""
    if record.engine_version is not None:
        yield (
            Term("engine.version", "matches", record.engine_version, engine_version),
            _within(record.engine_version, engine_version),
        )
    validity = record.validity
    cc, model, memory = facts["gpu.cc"], facts["gpu.model"], facts["gpu.memory_gib"]
    if validity.gpu_cc is not None:
        observed = None if cc is UNKNOWN else cc
        holds = None if observed is None else _within(validity.gpu_cc, str(observed))
        yield Term("gpu.cc", "matches", validity.gpu_cc, observed), holds
    if validity.gpu_models:
        observed = None if model is UNKNOWN else model
        holds = None if observed is None else observed in validity.gpu_models
        yield Term("gpu.model", "in", list(validity.gpu_models), observed), holds
    if validity.min_gpu_memory_gib is not None:
        least = validity.min_gpu_memory_gib
        holds = None if not isinstance(memory, float) else memory >= least
        yield Term("gpu.memory_gib", ">=", least, None if holds is None else memory), holds


def _evaluate(
    record: Record | RecordHead, facts: Facts, engine_version: str | None, memo: Memo
) -> tuple[Truth, Explanation]:
    """Whether ``record`` holds: its engine version, validity and condition, three-valued."""
    condition = record.condition
    truth = True if condition is None else evaluate(condition, facts, memo)
    if truth is False:
        return False, _NOTHING
    checks = list(_checks(record, facts, engine_version))
    if any(holds is False for _, holds in checks):
        return False, _NOTHING
    said = _NOTHING if condition is None else explain(condition, facts, truth, memo)
    missing = [term.fact for term, holds in checks if holds is None]
    if missing or truth is None:
        return None, Explanation((), tuple(dict.fromkeys([*missing, *said.unknown])))
    return True, Explanation((*(term for term, _ in checks), *said.fired_on), ())


def _values(record: Record, facts: Facts) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """The values ``record`` sets (its first variant that holds over its own) and its flags."""
    change = record.change
    if change is None:
        return {}, {}
    sets = dict(change.sets)
    for when, values in change.variants:
        if evaluate(when, facts) is True:
            sets.update(values)
            break
    return sets, dict(change.flags)


def _blocked(guards: Sequence[_Fired], records: Sequence[Record | RecordHead]) -> dict[str, str]:
    """Per blocked key, the reason of the most restrictive guard: a refusal, then a guard on
    every change, then the first by id. A record id maps to the merge key of that record."""
    by_id = {record.id: record for record in records}
    found: dict[str, tuple[int, str]] = {}
    for guard in guards:
        change = guard.record.change
        if change is None:
            continue
        strength = 2 if change.refuse else 1 if "*" in change.blocks else 0
        reason = change.reason or guard.record.title
        for name in change.blocks:
            key = by_id[name].merge_key if name in by_id else name
            if key not in found or strength > found[key][0]:
                found[key] = (strength, reason)
    return {key: reason for key, (_, reason) in sorted(found.items())}


def _blocks(blocked: Mapping[str, str], members: Sequence[_Fired]) -> bool:
    keys = {"*"}
    for member in members:
        record = member.record
        keys.update((record.merge_key, record.id))
        if record.change is not None and record.change.lever_key is not None:
            keys.update((record.change.lever_key, *record.change.lever_key.split("+")))
    return not keys.isdisjoint(blocked)


def _grouped(fired: Iterable[_Fired]) -> dict[str, list[_Fired]]:
    groups: dict[str, list[_Fired]] = {}
    for item in fired:
        groups.setdefault(item.record.merge_key, []).append(item)
    return groups


def _precedence(item: _Fired) -> tuple[int, int, str]:
    return _STATUS_RANK[item.record.status], item.record.layer != "pack", item.record.id


def _merged(members: Sequence[_Fired], facts: Facts, adverse: Sequence[_Fired] = ()) -> Match:
    """One offer from the records of one merge key. ``adverse`` are the measured nulls and
    losses of the key: they decide nothing, but their numbers and evidence take part in the
    merge, so a local result against the offer stays visible."""
    decider = min(members, key=_precedence)
    record = decider.record
    everyone = [m.record for m in (*members, *adverse)]
    return Match(
        record=_ref(record),
        layer=record.layer,
        status=record.status,
        kind=record.kind,
        merge_key=record.merge_key,
        sets=decider.sets,
        flags=decider.flags,
        explanation=decider.explanation,
        measured=_shown(everyone),
        evidence=_evidence(everyone),
        confidence=record.confidence.tier,
        direction_note=_direction_note(members, adverse, facts),
    )


def _direction_note(
    members: Sequence[_Fired], adverse: Sequence[_Fired], facts: Facts
) -> str | None:
    """ "your run <id> measured <metric> <change>" when a local record measured, beyond noise,
    the opposite of what the other layers expect (or, alone, against the goal)."""
    local = [m for m in (*members, *adverse) if m.record.layer == "local"]
    if not any(_own(m.record) is not None for m in local):
        return None
    others = sorted((m for m in members if m.record.layer != "local"), key=_precedence)
    effect = others[0].record.effect if others else None
    if effect is not None:
        expected = {k: e.direction for k in METRICS if (e := getattr(effect, k)) is not None}
    else:
        goal = facts["goal"]
        expected = {goal: "improves"} if isinstance(goal, str) else {}
    for member in local:
        record = member.record
        if (own := _own(record)) is None:
            continue
        for metric in METRICS:
            seen = own.metrics.get(metric)
            want = expected.get(metric)
            if seen is None or seen.noise is None or abs(seen.median) <= seen.noise:
                continue
            better = seen.median > 0 if metric == "throughput" else seen.median < 0
            if want in ("improves", "worsens") and better != (want == "improves"):
                label = _METRIC_LABEL[metric]
                return f"your run {_run_of(record)} measured {label} {seen.median:+.0%}"
    return None


def _run_of(record: Record) -> str:
    ref = next((item.ref for item in record.evidence if item.kind == "expdb"), record.id)
    return ref.rpartition("..")[2]
