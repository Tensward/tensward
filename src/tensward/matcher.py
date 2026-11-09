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
priors only demote. A launch counts once however many references name it. A record whose
launches carry coordinates is read by region (:mod:`tensward.similarity`): a launch counts when
the run sits in its region, a null seen near it demotes without removing, and one seen in
another region is only noted. A record without coordinates counts the launches that ran on the
run's model family, GPU architecture and engine minor. The measured numbers of such a
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
    Launch,
    Layer,
    Measured,
    Record,
    RecordHead,
    Status,
    Tier,
    engine_minor,
)
from .levers import LEVERS
from .similarity import Placed, Region, Similarity, dropped_axes, place

if TYPE_CHECKING:
    from .context import AnalysisContext
    from .playbook import Suggestion

_PREDICATE_KEYS = ("applies", "apply", "low_bar", "blocked", "unblocks", "start")
_SCREEN = "an agent screen"
_STATUS_RANK = {"promoted": 0, "measured": 1, "documented": 2}
_OFFERED = frozenset({"promoted", "measured"})
_METRIC_LABEL = {"throughput": "throughput", "ttft": "TTFT", "tpot": "TPOT"}
_NOTHING = Explanation((), ())
_BETTER = frozenset({"improves"})
_NOT_BETTER = frozenset({"none", "worsens"})
_REGION_RANK = {"same": 0, "near": 1, "outside": 2}
_LINE_METRIC = {"throughput": "req/s", "ttft": "TTFT", "tpot": "TPOT"}
_STEP_DOWN: dict[Tier, Tier] = {"high": "medium", "medium": "low", "low": "low"}
SAME_SIZE = 2.0  # same-region launches differing in size by more than this carry their range
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
    # where the run sits against the nearest improving launch (None for an unprofiled record),
    # that launch, the axis that decided, and the user line that names them.
    region: Region | None = None
    nearest: str | None = None
    deciding_axis: str | None = None
    line: str | None = None
    contested: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class Null:
    record: str  # "id@rev"
    removes: bool  # else it demotes to Could help
    reason: str
    launches: int  # independent eligible launches behind it
    measured: Measured | None  # local measured, else promoted, else measured
    evidence: tuple[str, ...]  # "<layer>:<kind>:<ref>" from every record of the merge key
    direction_note: str | None
    region: Region | None = None  # of the nearest null launch; None without located launches
    note_only: bool = False  # said, but it demotes nothing


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
    adverse: Mapping[str, tuple[_Fired, ...]] = field(default_factory=dict)
    improving: Mapping[str, tuple[_Fired, ...]] = field(default_factory=dict)

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


def for_engine(record: _R, engine_family: str, levers: frozenset[str]) -> _R | None:
    """The view of ``record`` for a project on ``engine_family`` whose engine realises
    ``levers``: None for another engine's record. An ``any`` record becomes the engine's own
    (its ``on`` table merged into the change) when it names that engine, or names no engine and
    needs only levers the engine realises; otherwise it stays ``any`` for the matcher to skip as
    ``not_for_engine``. A whole record's per-engine ``measured`` collapses to the engine's entry,
    and its status falls to ``documented`` without one. Agent evidence is dropped when the
    engine has no ``measured`` entry. A view is its own view."""
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
            if (view := for_engine(record, self._family, self._levers)) is not None:
                yield view

    def records(self, heads: Sequence[RecordHead]) -> list[Record]:
        raw = [self._raw.get((head.id, head.rev), head) for head in heads]
        found = self._source.records([head for head in raw if isinstance(head, RecordHead)])
        return [cast(Record, for_engine(record, self._family, self._levers)) for record in found]

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
            if (view := for_engine(record, engine_family, levers)) is not None
        ),
    )


def located(record: Record) -> bool:
    """Whether some launch of the record's evidence carries coordinates on the similarity axes."""
    measured = record.measured
    if measured is None:
        return False
    entries = measured.values() if isinstance(measured, Mapping) else (measured,)
    return any(launch.at for entry in entries for launch in entry.launches)


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
        "kind": known("model.kind"),
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
    loose = [f for f in fired if _null(f.record, goal)]
    for record in whole[len(held) :]:
        if _null(record, goal):
            truth, explanation = _evaluate(record, facts, engine_version, memo)
            if truth is True:
                loose.append(_Fired(record, explanation, *_values(record, facts)))
    adverse = _grouped(loose)
    pending = [record for record in whole[len(held) :] if not _null(record, goal)]
    fired = [f for f in fired if not _null(f.record, goal)]
    blocked = _blocked([f for f in fired if f.record.kind == "guard"], records)
    where = here(facts, engine_family, layers.pins.get(engine_family, ()))
    improving = _grouped(f for f in fired if f.record.kind == "lever")
    nulls = _nulls(adverse, improving, facts, where)
    offers, worth = [], []
    for key, members in improving.items():
        merged = _merged(members, facts, adverse.get(key, ()), where)
        if _offered(merged, nulls.get(key)):
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
        nulls=nulls,
        adverse={key: tuple(found) for key, found in adverse.items()},
        improving={key: tuple(found) for key, found in improving.items()},
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
    where = here(facts, str(facts["engine.name"]), layers.pins.get(str(facts["engine.name"]), ()))
    grouped = _grouped(kept)
    added = [_merged(members, facts, matches.adverse.get(key, ()), where)
             for key, members in grouped.items()]  # fmt: skip
    own = {key: matches.adverse.get(key, ()) for key in grouped}
    joined = {key: [*matches.improving.get(key, ()), *members] for key, members in grouped.items()}
    nulls = {**matches.nulls, **_nulls(own, joined, facts, where)}
    return replace(
        matches,
        offers=_in_order([*matches.offers, *added]),
        nulls=dict(sorted(nulls.items())),
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


def here(facts: Facts, engine_family: str, pins: Sequence[int]) -> dict[str, object]:
    """The run as :func:`tensward.similarity.place` compares it: its facts and its engine minor."""
    version = facts["engine.version"]
    minor = engine_minor(engine_family, version if isinstance(version, str) else None, pins)
    return {**facts, "engine.minor": minor}


def _placements(
    record: Record, goal: str, where: Mapping[str, object], outcomes: frozenset[str]
) -> list[tuple[Launch, Placed]]:
    """Where ``where`` sits against each located launch of ``record`` whose outcome on ``goal``
    is in ``outcomes``; none for a record without a profile."""
    sim = record.similarity
    if sim is None:
        return []
    every = [launch for item in _measured_of(record) for launch in item.launches if launch.at]
    dropped = dropped_axes(sim, [launch.at for launch in every])
    return [
        (launch, place(sim, launch.at, where, dropped))
        for launch in every
        if launch.outcome.get(goal) in outcomes
    ]


def nearest(
    record: Record, goal: str, where: Mapping[str, object], outcomes: frozenset[str]
) -> tuple[Launch, Placed] | None:
    """The located launch of ``record`` with an outcome on ``goal`` in ``outcomes`` that sits
    best against ``where``: the best region, then the smallest distance, then the region most
    launches agree on, then the newer id."""
    placed = _placements(record, goal, where, outcomes)
    if not placed:
        return None
    agreeing = Counter(p.region for _, p in placed)

    def rank(item: tuple[Launch, Placed]) -> tuple[int, int, int]:
        p = item[1]
        return _REGION_RANK[p.region], 99 if p.distance is None else p.distance, -agreeing[p.region]

    return min(sorted(placed, key=lambda item: item[0].id, reverse=True), key=rank)


def _label(fact: str) -> str:
    return fact.rpartition(".")[2].replace("_", " ")


def _same_words(sim: Similarity, where: Mapping[str, object]) -> str:
    """The profile's first regime axis in words, for a run in the same region."""
    axis = next((a for a in sim.soft if a.regime), None)
    if axis is None:
        return "in the same region"
    if axis.fact == "run.kv_state":
        return f"same KV state ({where.get(axis.fact)})"
    return f"same {_label(axis.fact)}"


def _where_words(sim: Similarity, placed: Placed, where: Mapping[str, object]) -> str:
    if placed.region == "same":
        return _same_words(sim, where)
    if placed.region == "near":
        return "nearby"
    return f"another region ({placed.deciding})"


def _setup(launch: Launch) -> str:
    named = [part for part in (launch.model, launch.gpu_model) if part]
    return ", ".join(named or [part for part in (launch.family, launch.arch) if part]) or "a launch"


def _said(value: object) -> str:
    return "not recorded" if value is None or value == UNKNOWN else str(value)


def _change_of(launch: Launch, goal: str) -> float | None:
    found = launch.change.get(goal)
    return None if found is None else found[0]


def _moved(values: Sequence[float]) -> str:
    low, high = min(values), max(values)
    if low == high:
        return f"{low:+.1%}"
    return f"{low:+.1%} to {high:+.1%}"


@dataclass(frozen=True, slots=True)
class _Placement:
    region: Region
    launch: Launch | None
    deciding: str | None
    line: str
    contested: bool


def _placement(
    members: Sequence[_Fired], adverse: Sequence[_Fired], goal: str, where: Mapping[str, object]
) -> _Placement | None:
    """Where the offer of ``members`` stands: the best-placed improving launch of any profiled
    member, the line that names it, and whether the same region also holds a null or worse
    launch. None when the deciding record carries no located launch."""
    decider = min(members, key=_precedence).record
    if decider.similarity is None or not located(decider):
        return None
    documented = decider.status not in _OFFERED
    best: tuple[Record, Launch, Placed] | None = None
    for member in members:
        found = nearest(member.record, goal, where, _BETTER)
        if found is None:
            continue
        key = (_REGION_RANK[found[1].region], found[1].distance or 0)
        if best is None or key < (_REGION_RANK[best[2].region], best[2].distance or 0):
            best = (member.record, *found)
    if best is None:
        return _Placement(
            "outside",
            None,
            None,
            f"no measured {_METRIC_LABEL.get(goal, goal)} gain here; measure it",
            False,
        )
    record, launch, placed = best
    sim = cast(Similarity, record.similarity)
    sizes = [
        c
        for other, p in _placements(record, goal, where, _BETTER)
        if p.region == "same" and (c := _change_of(other, goal)) is not None
    ]
    number = _change_of(launch, goal)
    metric = _LINE_METRIC.get(goal, goal)
    setup = _setup(launch)
    if documented:
        shown = "" if number is None or placed.region != "same" else f"{number:+.1%} {metric} "
        measured = _own(record)
        needs = (
            "needs a second model to be offered"
            if measured is None or measured.setups < 2
            else "not offered until it is promoted"
        )
        line = f"measured {shown}on {setup}, {_where_words(sim, placed, where)}; {needs}"
    elif placed.region == "same":
        shown = ""
        if len(sizes) > 1 and max(map(abs, sizes)) > SAME_SIZE * min(map(abs, sizes)):
            shown = f": {metric} {_moved(sizes)}"
        elif number is not None:
            shown = f": {metric} {number:+.1%}"
        line = f"Measured on {setup}, {_same_words(sim, where)}{shown}"
    elif placed.region == "near":
        why = {"there": "not recorded there", "here": "not measured here"}.get(
            placed.absent or "", "differs"
        )
        name = _label(placed.deciding or "")
        line = f"Measured on {setup}, nearby ({name} {why}); direction only"
    else:
        axis = placed.deciding
        there, here_ = _said(launch.at.get(axis or "")), _said(where.get(axis or ""))
        line = f"Nearest measurement: {setup}, {axis} {there}; here {here_}: outside, measure it"
    nulls = [
        (item.id, item.outcome.get(goal))
        for member in (*members, *adverse)
        for item, p in _placements(member.record, goal, where, _NOT_BETTER)
        if p.region == "same"
    ]
    contested = placed.region == "same" and bool(nulls)
    if contested and not documented:
        other, outcome = max(nulls)
        named = next(
            _setup(item)
            for member in (*members, *adverse)
            for item, p in _placements(member.record, goal, where, _NOT_BETTER)
            if p.region == "same" and item.id == other
        )
        what = "null" if outcome == "none" else "worse"
        line = (
            f"Contested: improving on {setup} but {what} on {named}, "
            f"{_same_words(sim, where)}; measure it"
        )
    return _Placement(placed.region, launch, placed.deciding, line, contested)


def _offered(merged: Match, null: Null | None) -> bool:
    """Whether a merged lever record is offered: measured or promoted, in the region where it was
    measured or near it, and not contested. An offer outside its region stays one only when a
    null measured in the run's own region demotes it."""
    if merged.status not in _OFFERED or merged.contested:
        return False
    if merged.region != "outside":
        return True
    return null is not None and null.region == "same" and not null.note_only


class _Seen(NamedTuple):
    strength: int  # 3 demotes (same region, or a record without launches here), 2 demotes (near)
    counted: frozenset[str]
    record: Record
    text: str
    region: Region | None


def _seen_by_cell(
    record: Record, goal: str, cell: tuple[object, object, object, object], engine_family: str
) -> _Seen:
    """A null record without located launches: its launches on the run's family, GPU architecture
    and engine minor count; its setup is read from them, else it is an agent screen."""
    want = getattr(record.effect, goal).direction
    placed = record.similarity is not None and located(record)
    launches = [
        launch
        for item in _measured_of(record)
        for launch in item.launches
        if launch.outcome.get(goal) == want and not placed
    ]
    counted = frozenset(
        launch.id
        for launch in launches
        if (launch.engine, launch.family, launch.arch, launch.engine_minor) == cell
    )
    what = "null" if want == "none" else "worse"
    where = _evidence_setup(launches, (cell[1], cell[2], cell[3]), engine_family)
    text = f"{'seen' if where == _SCREEN else 'measured'} {what} on {where}"
    return _Seen(3, counted, record, text, None)


def _seen_by_region(
    record: Record, near: tuple[Launch, Placed], goal: str, where: Mapping[str, object],
    *, null_record: bool,
) -> _Seen | None:  # fmt: skip
    """What the nearest null or worse launch of ``record`` says of the run: per the carrier, a
    demotion, a note or nothing."""
    launch, placed = near
    sim = cast(Similarity, record.similarity)
    what = "null" if launch.outcome.get(goal) == "none" else "worse"
    setup = _setup(launch)
    if placed.region == "same":
        counted = frozenset(
            other.id
            for other, p in _placements(record, goal, where, _NOT_BETTER)
            if p.region == "same"
        )
        return _Seen(3, counted, record, f"measured {what} on {setup}, {_same_words(sim, where)}",
                     "same")  # fmt: skip
    if placed.region == "near":
        dropped = dropped_axes(
            sim, [item.at for part in _measured_of(record) for item in part.launches if item.at]
        )
        regime_dropped = placed.distance == 0 and any(
            a.regime and a.fact in dropped for a in sim.soft
        )
        if null_record and not regime_dropped:
            text = f"measured {what} nearby on {setup}; not measured in this region"
            return _Seen(2, frozenset(), record, text, "near")
        return _Seen(1, frozenset(), record, f"measured {what} nearby on {setup}", "near")
    if null_record:
        text = f"measured {what} on {setup}, another region ({placed.deciding})"
        return _Seen(1, frozenset(), record, text, "outside")
    return None


def _nulls(
    adverse: Mapping[str, Sequence[_Fired]],
    improving: Mapping[str, Sequence[_Fired]],
    facts: Facts,
    where: Mapping[str, object],
) -> dict[str, Null]:
    """Per merge key, the null that removes its offer, else the strongest that demotes it, else
    the note that says what was measured elsewhere. Two carriers: the null records of the key,
    and the null or worse launches of its improving records, each read with the profile of the
    record that holds it. Removal counts the independent launches in the run's region over the
    key's records, not their references."""
    family, arch = facts["model.family"], facts["gpu.arch"]
    engine = str(facts["engine.name"])
    cell = (engine, family, arch, where["engine.minor"])
    goal = str(facts["goal"])
    found: dict[str, Null] = {}
    for key in sorted({*adverse, *improving}):
        seen: list[_Seen] = []
        for carrier, members in ((True, adverse.get(key, ())), (False, improving.get(key, ()))):
            for member in members:
                record = member.record
                near = (
                    nearest(record, goal, where, _NOT_BETTER)
                    if located(record) and record.similarity is not None
                    else None
                )
                if near is not None:
                    item = _seen_by_region(record, near, goal, where, null_record=carrier)
                elif carrier:
                    item = _seen_by_cell(record, goal, cell, engine)
                else:
                    item = None
                if item is not None:
                    seen.append(item)
        if not seen:
            continue
        best = min(seen, key=lambda s: (-s.strength, s.record.id))
        counted = frozenset().union(*(s.counted for s in seen))
        removes = len(counted) >= NULL_REMOVES
        said = f"{best.text} ({_ref(best.record)})"
        records = [s.record for s in seen]
        found[key] = Null(
            record=_ref(best.record),
            removes=removes,
            reason=f"not offered: {said}" if removes else said,
            launches=len(counted),
            measured=_shown(records, where, goal, _NOT_BETTER),
            evidence=_evidence(records),
            direction_note=_direction_note([], adverse.get(key, ()), facts),
            region=best.region,
            note_only=best.strength == 1,
        )
    return dict(sorted(found.items()))


def _evidence_setup(
    launches: Sequence[Launch], cell: tuple[object, object, object], engine_family: str
) -> str:
    """The setup a null was seen on, read from its launches (the run's cell first), else the
    agent screen."""
    seen = sorted(
        ((launch.family, launch.arch, launch.engine_minor) for launch in launches),
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


def _shown(
    records: Iterable[Record],
    where: Mapping[str, object] | None = None,
    goal: str | None = None,
    outcomes: frozenset[str] = _BETTER,
) -> Measured | None:
    """The measured block that is shown: of the records with a launch in the run's region, when
    there are any, else of all; of those a local one, else a promoted one, else any."""
    pool = [r for r in records if _own(r) is not None]
    if where is not None and goal is not None:
        close = [
            r
            for r in pool
            if (found := nearest(r, goal, where, outcomes)) is not None
            and found[1].region == "same"
        ]
        pool = close or pool
    shown = min(
        pool, key=lambda r: (r.layer != "local", r.status != "promoted", r.id), default=None
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
    return [cast(Record, for_engine(record, engine_family, levers)) for record in whole]


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
        record = for_engine(found, engine_family, levers)
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


def _merged(
    members: Sequence[_Fired],
    facts: Facts,
    adverse: Sequence[_Fired] = (),
    where: Mapping[str, object] | None = None,
) -> Match:
    """One offer from the records of one merge key. ``adverse`` are the measured nulls and
    losses of the key: they decide nothing, but their numbers and evidence take part in the
    merge, so a local result against the offer stays visible. With ``where`` (the run, per
    :func:`here`) a record with located launches is placed against them: ``region``, ``nearest``
    and ``line`` say how far its evidence travels, and a run that only nears it loses a step of
    confidence."""
    decider = min(members, key=_precedence)
    record = decider.record
    everyone = [m.record for m in (*members, *adverse)]
    goal = facts["goal"]
    placed = (
        _placement(members, adverse, goal, where)
        if where is not None and isinstance(goal, str)
        else None
    )
    confidence: Tier = record.confidence.tier
    if placed is not None and placed.region == "near":
        confidence = _STEP_DOWN[confidence]
    return Match(
        record=_ref(record),
        layer=record.layer,
        status=record.status,
        kind=record.kind,
        merge_key=record.merge_key,
        sets=decider.sets,
        flags=decider.flags,
        explanation=decider.explanation,
        measured=_shown(everyone, where, goal if isinstance(goal, str) else None),
        evidence=_evidence(everyone),
        confidence=confidence,
        direction_note=_direction_note(members, adverse, facts),
        region=None if placed is None else placed.region,
        nearest=None if placed is None or placed.launch is None else placed.launch.id,
        deciding_axis=None if placed is None else placed.deciding,
        line=None if placed is None else placed.line,
        contested=placed is not None and placed.contested,
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
