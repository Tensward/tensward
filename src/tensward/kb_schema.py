"""Knowledge records: one unit of what Tensward knows about a change, in the JSON form a pack
or the local layer stores, and the ``tw-train/1`` row a measured step yields.

A record is a ``lever`` (a change to offer), a ``guard`` (a change never to offer, or a
refusal) or an ``interaction`` (how two changes combine). Its condition is an AST of
:mod:`tensward.kb_condition` over the facts of :mod:`tensward.facts`; a coded record names the
playbook's predicates in ``code`` instead, or as well. :func:`record_from_json` refuses any
record that does not have exactly this shape.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal, TypeVar, get_args

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from . import kb_condition
from .context import GOALS
from .facts import FACTS_VERSION, Facts, canonical_json
from .kb_condition import ANCHOR_KEYS, ConditionError, Node, anchors, check
from .levers import LEVERS
from .playbook import EVIDENCE_ORDER, Direction

SCHEMA = 2
ENGINES = ("vllm", "llamacpp")
SCOPES = (*ENGINES, "any")
Kind = Literal["lever", "guard", "interaction"]
Status = Literal["documented", "measured", "promoted", "retired"]
Layer = Literal["playbook", "pack", "local"]
Basis = Literal["measured", "prior", "derived"]
Relation = Literal["independent", "never", "conflicts"]
EvidenceKind = Literal["expdb", "fleet", "source", "doc", "paper", "agent", "lessons", "benchmark"]
Tier = Literal["high", "medium", "low"]
AnswerRisk = Literal["none", "low", "medium", "high"]
StartupRisk = Literal["none", "low", "high"]
METRICS = ("throughput", "ttft", "tpot")
MODELS = frozenset({"memory_traffic"})  # prediction model names this Tensward has

_ID = re.compile(r"[a-z0-9][a-z0-9_.-]*(?:/[a-z0-9][a-z0-9_.-]*){2}")
_ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]*")


class RecordError(ValueError):
    """A record that does not have the schema's shape."""


@dataclass(frozen=True, slots=True, kw_only=True)
class EngineChange:
    """What an ``any`` record's change is on one engine: merged into the change when the record
    is viewed for that engine."""

    engine_version: str | None = None
    flags: Mapping[str, str | bool | None] = field(default_factory=dict)
    env: Mapping[str, str] = field(default_factory=dict)
    sets: Mapping[str, Any] = field(default_factory=dict)
    blocks: tuple[str, ...] = ()
    lever_key: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Change:
    levers: tuple[str, ...] = ()
    sets: Mapping[str, Any] = field(default_factory=dict)
    flags: Mapping[str, str | bool | None] = field(default_factory=dict)
    variants: tuple[tuple[Node, Mapping[str, Any]], ...] = ()  # the first that holds sets
    rule: str | None = None  # a change function in the code registry
    needs: tuple[str, ...] = ()  # record ids always carried with this one
    lever_key: str | None = None
    blocks: tuple[str, ...] = ()  # a guard's lever keys or record ids; "*" for every change
    refuse: bool = False  # a guard that refuses the run
    reason: str | None = None  # a guard's line under "Not applicable here"
    first: tuple[str, ...] = ()  # an interaction's lever keys or entry names
    second: tuple[str, ...] = ()
    goals: tuple[str, ...] = ()
    kv_bound: bool = False
    answer_pair: bool = False
    why: str | None = None
    relation: Relation | None = None
    env: Mapping[str, str] = field(default_factory=dict)  # the engine's environment variables
    on: Mapping[str, EngineChange] = field(default_factory=dict)  # per engine, for ``any`` records


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricEffect:
    direction: Direction
    low: float | None = None  # relative change: -0.27 is -27%
    high: float | None = None
    basis: Basis
    reason: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class RecordEffect:
    throughput: MetricEffect | None = None
    ttft: MetricEffect | None = None
    tpot: MetricEffect | None = None
    reason: str | None = None  # a template over facts
    model: str | None = None  # a prediction model in code
    prior_text: str | None = None  # the playbook's prior wording, where it is not generated
    prior_gain: Mapping[str, float] = field(default_factory=dict)  # per goal, a fraction


@dataclass(frozen=True, slots=True, kw_only=True)
class Costs:
    text: str
    answer_risk: AnswerRisk
    changes_answers: bool
    checks_answers: bool = False
    quality_risk: bool = False
    refuses: tuple[str, ...] = ()  # request kinds rejected after the change
    startup_risk: StartupRisk = "none"


@dataclass(frozen=True, slots=True, kw_only=True)
class Validity:
    gpu_cc: str | None = None  # a specifier: ">=7.5"
    gpu_models: tuple[str, ...] = ()  # empty: any
    min_gpu_memory_gib: float | None = None
    tested_families: tuple[str, ...] = ()
    recheck_on_engine_minor: bool = False
    expires: str | None = None  # an ISO date


@dataclass(frozen=True, slots=True, kw_only=True)
class Evidence:
    kind: EvidenceKind
    ref: str
    note: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Confidence:
    tier: Tier
    grade: str | None = None  # the playbook's evidence grade


@dataclass(frozen=True, slots=True, kw_only=True)
class MeasuredMetric:
    median: float
    low: float
    high: float
    noise: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Launch:
    """One Tensward launch of the change as the compiler read it from the experiment log: the
    child launch (its noise group when it has one, else its run) with the family, GPU
    architecture and engine minor it ran on, and per metric where its change fell against the
    noise band: ``none`` inside it, else ``improves`` or ``worsens``."""

    id: str
    family: str | None = None
    arch: str | None = None
    engine_minor: str | None = None
    outcome: Mapping[str, str] = field(default_factory=dict)
    engine: str = "vllm"


@dataclass(frozen=True, slots=True, kw_only=True)
class Measured:
    """What the compiler computed from the pinned experiment log."""

    setups: int
    pairs: int
    launches: tuple[Launch, ...] = ()
    metrics: Mapping[str, MeasuredMetric] = field(default_factory=dict)
    quality_ok: bool
    expdb_version: str
    status: Status | None = None  # an ``any`` record's entry for one engine


@dataclass(frozen=True, slots=True, kw_only=True)
class Record:
    """The pack and local JSON form of a record, with the compiler's derived fields."""

    id: str
    rev: int
    kind: Kind
    title: str
    status: Status
    engine: str
    engine_version: str | None = None
    condition: Node | None = None
    condition_text: str | None = None
    code: Mapping[str, Any] = field(default_factory=dict)  # [when.code]: playbook keys
    change: Change | None = None
    effect: RecordEffect | None = None
    costs: Costs | None = None
    validity: Validity
    evidence: tuple[Evidence, ...]
    confidence: Confidence
    oss_entry: str | None = None
    distribution: Literal["pack", "internal"] = "pack"
    requires_tensward: str | None = None
    facts_version: int
    contested: bool = False
    supersedes: tuple[str, ...] = ()
    retired: Mapping[str, str] | None = None
    measured: Measured | Mapping[str, Measured] | None = None  # per engine for ``any`` records
    anchors: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    layer: Layer = "pack"
    scope: Literal["any"] | None = None  # set on the view of an ``any`` record

    @property
    def merge_key(self) -> str:
        """Records with one merge key make one offer: the playbook name, else the lever key."""
        return _merge_key(self.oss_entry, self.change, self.id)


@dataclass(frozen=True, slots=True, kw_only=True)
class RecordHead:
    """The fields of a pack record the matcher reads before it knows whether the record holds.
    The source gives the whole record of those that hold, with ``KnowledgeSource.records``."""

    id: str
    rev: int
    kind: Kind
    status: Status
    requires_tensward: str | None = None
    engine: str
    engine_version: str | None = None
    condition: Node | None = None
    code: Mapping[str, Any] = field(default_factory=dict)
    change: Change | None = None
    validity: Validity = field(default_factory=Validity)
    oss_entry: str | None = None
    layer: Layer = "pack"
    scope: Literal["any"] | None = None

    @property
    def merge_key(self) -> str:
        return _merge_key(self.oss_entry, self.change, self.id)


def _merge_key(oss_entry: str | None, change: Change | None, record_id: str) -> str:
    if oss_entry is not None:
        return oss_entry
    if change is not None and change.lever_key is not None:
        return change.lever_key
    return record_id


T = TypeVar("T")
Convert = Callable[[Any, str], Any]


def _fail(where: str, why: str) -> RecordError:
    return RecordError(f"{where}: {why}")


def _str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(where, "not a non-empty string")
    return value


def _bool(value: Any, where: str) -> bool:
    if type(value) is not bool:
        raise _fail(where, "not true or false")
    return value


def _int(value: Any, where: str) -> int:
    if type(value) is not int or value < 0:
        raise _fail(where, "not a whole number")
    return value


def _float(value: Any, where: str) -> float:
    if not isinstance(value, (int, float)) or type(value) is bool or not math.isfinite(value):
        raise _fail(where, "not a finite number")
    return float(value)


def _strs(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise _fail(where, "not a list")
    return tuple(_str(item, f"{where}[{i}]") for i, item in enumerate(value))


def _choice(*values: str) -> Convert:
    def convert(value: Any, where: str) -> str:
        if value not in values:
            raise _fail(where, f"{value!r} is not one of {', '.join(values)}")
        return str(value)

    return convert


def _optional(convert: Convert) -> Convert:
    return lambda value, where: None if value is None else convert(value, where)


def _table(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise _fail(where, "not a table")
    return dict(value)


def _build(cls: type[T], converters: Mapping[str, Convert]) -> Convert:
    """A converter from a JSON table to ``cls``: every key known, every required field given."""
    required = {
        f.name
        for f in dataclasses.fields(cls)  # type: ignore[arg-type]
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    }

    def convert(value: Any, where: str) -> T:
        table = _table(value, where)
        unknown, missing = sorted(set(table) - set(converters)), sorted(required - set(table))
        if unknown:
            raise _fail(where, f"unknown keys {', '.join(unknown)} (a newer schema?)")
        if missing:
            raise _fail(where, f"missing {', '.join(missing)}")
        return cls(**{key: converters[key](item, f"{where}.{key}") for key, item in table.items()})

    return convert


def _specifier(value: Any, where: str) -> str:
    try:
        SpecifierSet(_str(value, where))
    except InvalidSpecifier:
        raise _fail(where, f"{value!r} is not a version specifier") from None
    return str(value)


def _version(value: Any, where: str) -> str:
    try:
        Version(_str(value, where))
    except InvalidVersion:
        raise _fail(where, f"{value!r} is not a version") from None
    return str(value)


def _iso_date(value: Any, where: str) -> str:
    try:
        date.fromisoformat(_str(value, where))
    except ValueError:
        raise _fail(where, f"{value!r} is not an ISO date") from None
    return str(value)


def _condition(value: Any, where: str) -> Node:
    try:
        node = kb_condition.from_json(value)
    except ConditionError as error:
        raise _fail(where, str(error)) from None
    if problems := check(node):
        raise _fail(where, "; ".join(problems))
    return node


def _flags(value: Any, where: str) -> Mapping[str, str | bool | None]:
    table = _table(value, where)
    for flag, setting in table.items():
        if not flag.startswith("-") or not isinstance(setting, (str, bool, type(None))):
            raise _fail(where, f"{flag} = {setting!r} is not an engine flag and its value")
    return table


def _env(value: Any, where: str) -> Mapping[str, str]:
    table = _table(value, where)
    for name, setting in table.items():
        if not _ENV_NAME.fullmatch(name) or not isinstance(setting, str):
            raise _fail(where, f"{name} = {setting!r} is not an environment variable and its value")
    return table


def _variants(value: Any, where: str) -> tuple[tuple[Node, Mapping[str, Any]], ...]:
    if not isinstance(value, (list, tuple)):
        raise _fail(where, "not a list")
    found = []
    for i, variant in enumerate(value):
        table = _table(variant, f"{where}[{i}]")
        if set(table) != {"when", "sets"}:
            raise _fail(f"{where}[{i}]", "a variant has exactly when and sets")
        found.append(
            (
                _condition(table["when"], f"{where}[{i}].when"),
                _table(table["sets"], f"{where}[{i}].sets"),
            )
        )
    return tuple(found)


def _per_goal(value: Any, where: str) -> Mapping[str, float]:
    table = _table(value, where)
    if unknown := sorted(set(table) - set(GOALS)):
        raise _fail(where, f"not goals: {', '.join(unknown)}")
    return {goal: _float(gain, f"{where}.{goal}") for goal, gain in table.items()}


def _retired(value: Any, where: str) -> Mapping[str, str]:
    table = _table(value, where)
    if set(table) != {"date", "reason", "by"}:
        raise _fail(where, "a retirement has exactly date, reason and by")
    return {key: _str(item, f"{where}.{key}") for key, item in table.items()}


def _anchors(value: Any, where: str) -> Mapping[str, tuple[str, ...]]:
    table = _table(value, where)
    if set(table) != set(ANCHOR_KEYS):
        raise _fail(where, f"anchors have exactly {', '.join(ANCHOR_KEYS)}")
    return {key: _strs(table[key], f"{where}.{key}") for key in ANCHOR_KEYS}


def _list_of(convert: Convert) -> Convert:
    def converted(value: Any, where: str) -> tuple[Any, ...]:
        if not isinstance(value, (list, tuple)):
            raise _fail(where, "not a list")
        return tuple(convert(item, f"{where}[{i}]") for i, item in enumerate(value))

    return converted


def _metric_keys(convert: Convert) -> Convert:
    def converted(value: Any, where: str) -> Mapping[str, Any]:
        table = _table(value, where)
        if unknown := sorted(set(table) - set(METRICS)):
            raise _fail(where, f"not metrics: {', '.join(unknown)}")
        return {key: convert(item, f"{where}.{key}") for key, item in table.items()}

    return converted


_METRIC_EFFECT = _build(
    MetricEffect,
    {
        "direction": _choice(*get_args(Direction)),
        "low": _optional(_float),
        "high": _optional(_float),
        "basis": _choice(*get_args(Basis)),
        "reason": _optional(_str),
    },
)


def _metric_effect(value: Any, where: str) -> MetricEffect:
    effect: MetricEffect = _METRIC_EFFECT(value, where)
    if effect.low is not None and effect.high is not None and effect.low > effect.high:
        raise _fail(where, "low is above high")
    return effect


_ON_CHANGE = _build(
    EngineChange,
    {
        "engine_version": _optional(_specifier),
        "flags": _flags,
        "env": _env,
        "sets": _table,
        "blocks": _strs,
        "lever_key": _optional(_str),
    },
)


def _on(value: Any, where: str) -> Mapping[str, EngineChange]:
    table = _table(value, where)
    if unknown := sorted(set(table) - set(ENGINES)):
        raise _fail(where, f"not engines: {', '.join(unknown)}")
    return {engine: _ON_CHANGE(item, f"{where}.{engine}") for engine, item in table.items()}


_CHANGE = _build(
    Change,
    {
        "levers": _strs,
        "sets": _table,
        "flags": _flags,
        "variants": _variants,
        "rule": _optional(_str),
        "needs": _strs,
        "lever_key": _optional(_str),
        "blocks": _strs,
        "refuse": _bool,
        "reason": _optional(_str),
        "first": _strs,
        "second": _strs,
        "goals": _list_of(_choice(*GOALS)),
        "kv_bound": _bool,
        "answer_pair": _bool,
        "why": _optional(_str),
        "relation": _optional(_choice(*get_args(Relation))),
        "env": _env,
        "on": _on,
    },
)
_EFFECT = _build(
    RecordEffect,
    {
        **{metric: _optional(_metric_effect) for metric in METRICS},
        "reason": _optional(_str),
        "model": _optional(_str),
        "prior_text": _optional(_str),
        "prior_gain": _per_goal,
    },
)
_COSTS = _build(
    Costs,
    {
        "text": _str,
        "answer_risk": _choice(*get_args(AnswerRisk)),
        "changes_answers": _bool,
        "checks_answers": _bool,
        "quality_risk": _bool,
        "refuses": _strs,
        "startup_risk": _choice(*get_args(StartupRisk)),
    },
)
_VALIDITY = _build(
    Validity,
    {
        "gpu_cc": _optional(_specifier),
        "gpu_models": _strs,
        "min_gpu_memory_gib": _optional(_float),
        "tested_families": _strs,
        "recheck_on_engine_minor": _bool,
        "expires": _optional(_iso_date),
    },
)
_EVIDENCE = _build(
    Evidence,
    {
        "kind": _choice(*get_args(EvidenceKind)),
        "ref": _str,
        "note": _optional(_str),
    },
)
_CONFIDENCE = _build(
    Confidence,
    {
        "tier": _choice(*get_args(Tier)),
        "grade": _optional(_choice(*EVIDENCE_ORDER)),
    },
)
_LAUNCH = _build(
    Launch,
    {
        "id": _str,
        "family": _optional(_str),
        "arch": _optional(_str),
        "engine_minor": _optional(_str),
        "outcome": _metric_keys(_choice("none", "improves", "worsens")),
        "engine": _choice(*ENGINES),
    },
)
_MEASURED = _build(
    Measured,
    {
        "setups": _int,
        "pairs": _int,
        "launches": _list_of(_LAUNCH),
        "quality_ok": _bool,
        "expdb_version": _str,
        "status": _optional(_choice(*get_args(Status))),
        "metrics": _metric_keys(
            _build(
                MeasuredMetric,
                {
                    "median": _float,
                    "low": _float,
                    "high": _float,
                    "noise": _optional(_float),
                },
            )
        ),
    },
)


def _measured(value: Any, where: str) -> Measured | Mapping[str, Measured]:
    """One measured block, or for an ``any`` record one per engine."""
    table = _table(value, where)
    if "setups" in table:
        measured: Measured = _MEASURED(table, where)
        return measured
    if unknown := sorted(set(table) - set(ENGINES)):
        raise _fail(where, f"not engines: {', '.join(unknown)}")
    return {engine: _MEASURED(item, f"{where}.{engine}") for engine, item in table.items()}


def _record_id(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise _fail(where, f"{value!r} is not <engine or any>/<area>/<slug>")
    return value


def _rev(value: Any, where: str) -> int:
    if _int(value, where) < 1:
        raise _fail(where, "revisions start at 1")
    return int(value)


def _facts_version(value: Any, where: str) -> int:
    if _int(value, where) > FACTS_VERSION:
        raise _fail(where, f"needs facts version {value}; this Tensward has {FACTS_VERSION}")
    return int(value)


_RECORD = _build(
    Record,
    {
        "id": _record_id,
        "rev": _rev,
        "kind": _choice(*get_args(Kind)),
        "title": _str,
        "status": _choice(*get_args(Status)),
        "engine": _choice(*SCOPES),
        "engine_version": _optional(_specifier),
        "condition": _optional(_condition),
        "condition_text": _optional(_str),
        "code": _table,
        "change": _optional(_CHANGE),
        "effect": _optional(_EFFECT),
        "costs": _optional(_COSTS),
        "validity": _VALIDITY,
        "evidence": _list_of(_EVIDENCE),
        "confidence": _CONFIDENCE,
        "oss_entry": _optional(_str),
        "distribution": _choice("pack", "internal"),
        "requires_tensward": _optional(_version),
        "facts_version": _facts_version,
        "contested": _bool,
        "supersedes": _strs,
        "retired": _optional(_retired),
        "measured": _optional(_measured),
        "anchors": _anchors,
        "layer": _choice(*get_args(Layer)),
        "scope": _optional(_choice("any")),
    },
)


def record_from_json(data: Mapping[str, Any], *, layer: Layer) -> Record:
    """The record ``data`` holds, read into ``layer``; RecordError on any violation."""
    where = str(data.get("id", "record")) if isinstance(data, Mapping) else "record"
    if isinstance(data, Mapping) and data.get("layer", layer) != layer:
        raise _fail(where, f"a {data['layer']} record read as {layer}")
    record: Record = _RECORD({**_table(data, where), "layer": layer}, where)
    change = record.change
    if record.kind != "interaction" and record.condition is None and not record.code:
        raise _fail(where, "a lever or guard needs a condition or code")
    if record.kind == "lever" and (change is None or record.effect is None or record.costs is None):
        raise _fail(where, "a lever needs change, effect and costs")
    if record.kind == "guard" and (change is None or not change.blocks):
        raise _fail(where, "a guard names what it blocks")
    if record.kind == "interaction" and (
        change is None or not change.first or not change.second or change.relation is None
    ):
        raise _fail(where, "an interaction needs first, second and relation")
    if change is not None and change.on and record.engine != "any":
        raise _fail(where, "only an any record has change.on")
    if layer == "local" and record.engine == "any":
        raise _fail(where, "a local record names its engine")
    measured = record.measured
    if isinstance(measured, Mapping):
        if record.engine != "any":
            raise _fail(where, "only an any record has measured per engine")
        if any(entry.status is None for entry in measured.values()):
            raise _fail(where, "a measured entry per engine carries its status")
    elif measured is not None and record.engine == "any":
        raise _fail(where, "an any record has measured per engine")
    if not record.anchors:
        found = anchors(record.condition, engine=record.engine, gpu_cc=record.validity.gpu_cc)
        record = dataclasses.replace(record, anchors=found)
    return record


def record_to_json(record: Record) -> dict[str, Any]:
    """The JSON form :func:`record_from_json` reads back to an equal record."""
    data: dict[str, Any] = _plain(record)
    return data


def record_to_compact(record: Record) -> dict[str, Any]:
    """The JSON form without the fields at their defaults, the layer and the anchors (a pack's
    index holds them): what a pack stores, read back by :func:`trusted_record`."""
    data: dict[str, Any] = _compact(record)
    for derived in ("anchors", "layer"):
        data.pop(derived, None)
    return data


def _compact(value: Any) -> Any:
    if isinstance(value, tuple) and any(dataclasses.is_dataclass(item) for item in value):
        return [_compact(item) for item in value]
    if isinstance(value, Node) or not dataclasses.is_dataclass(value):
        return _plain(value)
    found = {}
    for f in dataclasses.fields(value):
        item = getattr(value, f.name)
        if f.default is not dataclasses.MISSING and item == f.default:
            continue
        if f.default_factory is not dataclasses.MISSING and item == f.default_factory():
            continue
        found[f.name] = (
            [{"when": kb_condition.to_json(w), "sets": dict(s)} for w, s in item]
            if f.name == "variants"
            else _compact(item)
        )
    return found


def _as_tuples(data: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    for name in names:
        if name in data:
            data[name] = tuple(data[name])
    return data


def _trusted_change(data: dict[str, Any]) -> Change:
    _as_tuples(data, ("levers", "needs", "blocks", "first", "second", "goals"))
    if "variants" in data:
        data["variants"] = tuple(
            (kb_condition.from_json(v["when"]), v["sets"]) for v in data["variants"]
        )
    if "on" in data:
        data["on"] = {
            engine: EngineChange(**_as_tuples(item, ("blocks",)))
            for engine, item in data["on"].items()
        }
    return Change(**data)


def _trusted_effect(data: dict[str, Any]) -> RecordEffect:
    for metric in METRICS:
        if data.get(metric) is not None:
            data[metric] = MetricEffect(**data[metric])
    return RecordEffect(**data)


def _trusted_measured(data: dict[str, Any]) -> Measured | dict[str, Measured]:
    if "setups" not in data:
        return {engine: _trusted_measured_one(item) for engine, item in data.items()}
    return _trusted_measured_one(data)


def _trusted_measured_one(data: dict[str, Any]) -> Measured:
    data["metrics"] = {k: MeasuredMetric(**v) for k, v in data.get("metrics", {}).items()}
    if "launches" in data:
        data["launches"] = tuple(Launch(**launch) for launch in data["launches"])
    return Measured(**data)


def trusted_head(
    columns: Mapping[str, Any],
    condition: Node | None,
    parsed: dict[str, Any],
    *,
    layer: Layer = "pack",
) -> RecordHead:
    """The head of a record a signed pack stores: ``columns`` hold its scalar fields and the
    JSON of ``code``, ``change`` and ``validity`` (of :func:`record_to_compact`; None at their
    defaults). ``parsed`` maps a ``change`` or ``validity`` JSON already read to its value; the
    caller keeps it as long as it likes. Nothing is checked again: the builder checked it, and
    the pack's signature covers the bytes. Any other source goes through
    :func:`record_from_json`."""
    code, change, validity = columns["code"], columns["change"], columns["validity"]
    if change is not None and change not in parsed:
        parsed[change] = _trusted_change(json.loads(change))
    if validity is not None and validity not in parsed:
        parsed[validity] = Validity(
            **_as_tuples(json.loads(validity), ("gpu_models", "tested_families"))
        )
    return RecordHead(
        id=columns["id"],
        rev=columns["rev"],
        kind=columns["kind"],
        status=columns["status"],
        requires_tensward=columns["requires_tensward"],
        engine=columns["engine"],
        engine_version=columns["engine_version"],
        condition=condition,
        code={} if code is None else json.loads(code),
        change=None if change is None else parsed[change],
        validity=_NO_LIMITS if validity is None else parsed[validity],
        oss_entry=columns["oss_entry"],
        layer=layer,
    )


_NO_LIMITS = Validity()


def trusted_record(head: RecordHead, rest: dict[str, Any]) -> Record:
    """The whole record of ``head``, from the other fields of its compact form; ``rest`` is
    consumed and, as for :func:`trusted_head`, not checked again."""
    get = rest.get
    if (effect := get("effect")) is not None:
        rest["effect"] = _trusted_effect(effect)
    if (costs := get("costs")) is not None:
        rest["costs"] = Costs(**_as_tuples(costs, ("refuses",)))
    if (measured := get("measured")) is not None:
        rest["measured"] = _trusted_measured(measured)
    rest["evidence"] = tuple(Evidence(**item) for item in rest["evidence"])
    rest["confidence"] = Confidence(**rest["confidence"])
    _as_tuples(rest, ("supersedes",))
    return Record(
        **rest,
        id=head.id,
        rev=head.rev,
        kind=head.kind,
        status=head.status,
        requires_tensward=head.requires_tensward,
        engine=head.engine,
        engine_version=head.engine_version,
        condition=head.condition,
        code=head.code,
        change=head.change,
        validity=head.validity,
        oss_entry=head.oss_entry,
        layer=head.layer,
        scope=head.scope,
    )


def _plain(value: Any) -> Any:
    if isinstance(value, Node):
        return kb_condition.to_json(value)
    if isinstance(value, Change):
        plain = {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
        plain["variants"] = [
            {"when": kb_condition.to_json(when), "sets": dict(sets)}
            for when, sets in value.variants
        ]
        return plain
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def engine_minor(engine: str, version: str | None, pins: Sequence[int] = ()) -> str | None:
    """The bucket ``version`` falls in for the engine's "recheck on minor" and for nulls: vLLM's
    ``major.minor``; llama.cpp's largest approved pin (a build number in ``pins``) not above the
    build. None when the version is unknown or below every pin."""
    if version is None:
        return None
    if engine == "llamacpp":
        found = re.match(r"b?(\d+)", version)
        build = None if found is None else int(found[1])
        below = [pin for pin in pins if build is not None and pin <= build]
        return str(max(below)) if below else None
    return ".".join(version.split(".")[:2])


def _flag_lever(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")


def _value_class(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "on" if value else "off"
    method = getattr(value, "method", None)
    return str(method) if isinstance(method, str) else str(value)


def lever_key(
    sets: Mapping[str, Any],
    flags: Mapping[str, Any],
    before: Mapping[str, Any] | None = None,
    *,
    canonical: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> str:
    """``<lever>=<value class>`` joined by "+" in sorted order: an enum or bool lever or a flag
    by its value (a JSON flag by each leaf, however it was spelled); a numeric lever by
    "up"/"down" against ``before`` (or "set" without it). ``canonical`` is the engine's flag
    canonicaliser."""
    flags = canonical(flags)
    parts = [f"{_flag_lever(flag)}={_value_class(value)}" for flag, value in flags.items()]
    for name, value in sets.items():
        lever = LEVERS.get(name)
        if lever is None or lever.value not in ("int", "float") or value is None:
            parts.append(f"{name}={_value_class(value)}")
            continue
        old = None if before is None else before.get(name)
        moved = "set" if old is None or old == value else "up" if value > old else "down"
        parts.append(f"{name}={moved}")
    return "+".join(sorted(parts))


TRAIN_SCHEMA = "tw-train/1"
TRAIN_SOURCES = ("release_session", "fleet", "research_probe", "local", "benchmark")
TRAIN_METRICS = ("req_s", "ttft_p95_ms", "tpot_p95_ms", "output_tok_s", "e2e_p95_ms")
_TRAIN_KEYS: Mapping[str, tuple[frozenset[str], frozenset[str]]] = {  # (required, allowed)
    "action": (
        frozenset({"lever_keys"}),
        frozenset(
            {
                "lever_keys",
                "settings_delta",
                "flags_delta",
                "records",
                "offered_rank",
                "score",
                "tier",
                "stack_size",
                "scorer",
            }
        ),
    ),
    "outcome": (
        frozenset({"started"}),
        frozenset(
            {
                "started",
                "error_class",
                "change",
                "noise_band",
                "failed_share",
                "broken_answer_share_delta",
                "judge",
                "kept",
            }
        ),
    ),
    "labels": (frozenset(), frozenset({"beyond_noise", "direction", "quality_ok"})),
    "provenance": (
        frozenset({"runs", "tensward"}),
        frozenset(
            {
                "runs",
                "tensward",
                "pack",
                "content_digest",
            }
        ),
    ),
}
_PER_METRIC = (
    ("outcome", "change"),
    ("outcome", "noise_band"),
    ("labels", "beyond_noise"),
    ("labels", "direction"),
)


def train_row(
    *,
    row_id: str,
    source: str,
    features: Facts,
    action: Mapping[str, Any],
    outcome: Mapping[str, Any],
    labels: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """One ``tw-train/1`` row: a parent run's facts, the change measured from it, and what it
    did. ValueError when a block has a key outside the contract or lacks a required one."""
    if source not in TRAIN_SOURCES:
        raise ValueError(f"{source!r} is not a training-row source")
    blocks = {"action": action, "outcome": outcome, "labels": labels, "provenance": provenance}
    for name, block in blocks.items():
        required, allowed = _TRAIN_KEYS[name]
        if unknown := sorted(set(block) - allowed):
            raise ValueError(f"{name} has keys outside {TRAIN_SCHEMA}: {', '.join(unknown)}")
        if missing := sorted(required - set(block)):
            raise ValueError(f"{name} lacks {', '.join(missing)}")
    for name, key in _PER_METRIC:
        metrics = blocks[name].get(key, {})
        if unknown := sorted(set(metrics) - set(TRAIN_METRICS)):
            raise ValueError(f"{name}.{key} names metrics outside {TRAIN_SCHEMA}: {unknown}")
    row = {
        "row": {
            "schema": TRAIN_SCHEMA,
            "row_id": row_id,
            "source": source,
            "exclude_from_tuning": source == "benchmark",
        },
        "features": {"facts_version": FACTS_VERSION, "facts": features.to_json()},
        "action": {**action, "lever_keys": sorted(action["lever_keys"])},
        "outcome": dict(outcome),
        "labels": dict(labels),
        "provenance": {**provenance, "facts_digest": features.digest()},
    }
    canonical_json(row)
    return row
