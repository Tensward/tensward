"""Similarity profiles: which facts must match for a launch's evidence to travel, and how far
apart two situations read on the facts that only shade it.

A profile names hard axes (compared for equality) and soft axes (compared by band). A regime
axis decides whether the mechanism can act; a magnitude axis decides only whether the number
is shown. :func:`place` reads a launch's coordinates against the situation at hand and says
``same``, ``near`` or ``outside``. This module is imported by the record schema, so it reads
neither."""

from __future__ import annotations

import tomllib
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Literal

from .fact_values import UNKNOWN
from .facts import VOCABULARY, Facts

Region = Literal["same", "near", "outside"]

SIMILARITY_FILE = "data/similarity.toml"
KERNEL_AXES = frozenset({"run.attn_backend", "run.linear_kernel", "run.moe_backend"})
RUN_AXES = frozenset({"engine.minor"})
MAX_REGIME_AXES = 2
ARCH = "gpu.arch"


@dataclass(frozen=True, slots=True, kw_only=True)
class Axis:
    fact: str
    cuts: tuple[float, ...] = ()
    order: tuple[str, ...] = ()
    regime: bool = False
    unit: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Similarity:
    profile: str
    why: str
    hard: tuple[str, ...]
    soft: tuple[Axis, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class Placed:
    region: Region
    distance: int | None
    deciding: str | None
    unknown: tuple[str, ...]
    absent: Literal["there", "here"] | None = None


def _known(value: object) -> bool:
    return value is not None and value is not UNKNOWN


def _fact(facts: Mapping[str, object], name: str) -> object:
    value = facts.get(name, UNKNOWN)
    if not _known(value):
        return UNKNOWN
    spec = VOCABULARY.get(name)
    if (
        spec is not None
        and spec.type == "set"
        and isinstance(value, list | tuple | set | frozenset)
    ):
        return frozenset(value)
    return value


def _check_fact(name: object, where: str) -> str:
    if not isinstance(name, str) or (name not in VOCABULARY and name not in RUN_AXES):
        raise ValueError(f"{where}: {name!r} is not a fact")
    return name


def _axis(raw: Mapping[str, Any], where: str) -> Axis:
    unknown = sorted(set(raw) - {"fact", "cuts", "order", "regime", "unit"})
    if unknown:
        raise ValueError(f"{where}: unknown axis keys {unknown}")
    fact = _check_fact(raw.get("fact"), where)
    cuts = tuple(float(c) for c in raw.get("cuts", ()))
    order = tuple(str(v) for v in raw.get("order", ()))
    if cuts and order:
        raise ValueError(f"{where}: {fact} has both cuts and order")
    if any(b <= a for a, b in zip(cuts, cuts[1:])):
        raise ValueError(f"{where}: cuts of {fact} must increase")
    if len(set(order)) != len(order):
        raise ValueError(f"{where}: order of {fact} repeats a value")
    spec = VOCABULARY.get(fact)
    if spec is not None and spec.type == "enum" and not order:
        raise ValueError(f"{where}: enum axis {fact} needs an order")
    if spec is not None and spec.type in ("float", "int") and not cuts:
        raise ValueError(f"{where}: numeric axis {fact} needs cuts")
    return Axis(
        fact=fact,
        cuts=cuts,
        order=order,
        regime=bool(raw.get("regime", False)),
        unit=raw.get("unit"),
    )


def _hard(raw: object, where: str) -> tuple[str, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        raise ValueError(f"{where}: hard must be a list of facts")
    out: list[str] = []
    for name in raw:
        fact = _check_fact(name, where)
        if fact not in out:
            out.append(fact)
    return tuple(out)


def _assemble(
    profile: str, why: str, hard: tuple[str, ...], soft: Sequence[Axis], where: str
) -> Similarity:
    facts = [a.fact for a in soft]
    if len(set(facts)) != len(facts):
        raise ValueError(f"{where}: an axis is listed twice")
    clash = sorted(set(facts) & set(hard))
    if clash:
        raise ValueError(f"{where}: {clash} are both hard and soft")
    if sum(a.regime for a in soft) > MAX_REGIME_AXES:
        raise ValueError(f"{where}: more than {MAX_REGIME_AXES} regime axes")
    ordered = sorted(soft, key=lambda a: not a.regime)
    return Similarity(profile=profile, why=why, hard=hard, soft=tuple(ordered))


def _load() -> dict[str, Similarity]:
    tables = tomllib.loads(files("tensward").joinpath(SIMILARITY_FILE).read_text("utf-8"))[
        "profile"
    ]
    out: dict[str, Similarity] = {}
    for name, table in tables.items():
        where = f"{SIMILARITY_FILE} [profile.{name}]"
        why = table.get("why")
        if not isinstance(why, str) or not why:
            raise ValueError(f"{where}: why is required")
        soft = [_axis(a, where) for a in table.get("soft", ())]
        out[name] = _assemble(name, why, _hard(table.get("hard", ()), where), soft, where)
    return out


PROFILES: Mapping[str, Similarity] = _load()


def resolve(table: Mapping[str, Any], where: str) -> Similarity:
    """The named profile with the record's ``add`` axes, ``drop`` facts and extra ``hard`` facts."""
    unknown = sorted(set(table) - {"profile", "add", "drop", "hard"})
    if unknown:
        raise ValueError(f"{where}: unknown similarity keys {unknown}")
    name = table.get("profile")
    base = PROFILES.get(name) if isinstance(name, str) else None
    if base is None:
        raise ValueError(f"{where}: unknown profile {name!r}")
    drop = table.get("drop", ())
    if not isinstance(drop, Sequence) or isinstance(drop, str):
        raise ValueError(f"{where}: drop must be a list of facts")
    missing = sorted(str(d) for d in drop if d not in {a.fact for a in base.soft})
    if missing:
        raise ValueError(f"{where}: drop names {missing}, which profile {name} does not carry")
    added = [_axis(a, where) for a in table.get("add", ())]
    replaced = {a.fact for a in added}
    soft = [a for a in base.soft if a.fact not in set(drop) | replaced] + added
    hard = tuple(dict.fromkeys(base.hard + _hard(table.get("hard", ()), where)))
    return _assemble(base.profile, base.why, hard, soft, where)


def band(axis: Axis, value: object) -> int | None:
    """The index of the interval ``value`` falls in (a value equal to a cut is in the upper band);
    None when it is unknown or off the axis."""
    if not _known(value):
        return None
    if axis.order:
        return axis.order.index(value) if value in axis.order else None
    if isinstance(value, bool):
        return int(value)
    if axis.cuts and isinstance(value, int | float):
        return bisect_right(axis.cuts, value)
    return None


def _hard_gate(
    similarity: Similarity, at: Mapping[str, object], here: Mapping[str, object]
) -> str | None:
    """The hard axis that differs or is unknown, or None when every one agrees."""
    arch_checked = False
    for name in similarity.hard:
        a, h = _fact(at, name), _fact(here, name)
        if name in KERNEL_AXES and (a is UNKNOWN or h is UNKNOWN):
            if arch_checked:
                continue
            arch_checked = True
            name, a, h = ARCH, _fact(at, ARCH), _fact(here, ARCH)
        if a is UNKNOWN or h is UNKNOWN or a != h:
            return name
    return None


def _gap(
    axis: Axis, at: Mapping[str, object], here: Mapping[str, object]
) -> tuple[int, Literal["there", "here"] | None]:
    a, h = band(axis, _fact(at, axis.fact)), band(axis, _fact(here, axis.fact))
    if a is None:
        return 1, "there"
    if h is None:
        return 1, "here"
    return abs(a - h), None


def place(
    similarity: Similarity,
    at: Mapping[str, object],
    here: Facts | Mapping[str, object],
    dropped: frozenset[str] = frozenset(),
) -> Placed:
    """Where the situation ``here`` sits against one launch's coordinates ``at``."""
    failed = _hard_gate(similarity, at, here)
    if failed is not None:
        return Placed(region="outside", distance=None, deciding=failed, unknown=())
    d_regime = d_mag = 0
    deciding_regime = deciding_mag = None
    unknown: list[str] = []
    sides: dict[str, Literal["there", "here"]] = {}
    for axis in similarity.soft:
        if axis.fact in dropped:
            continue
        gap, missing = _gap(axis, at, here)
        if missing:
            unknown.append(axis.fact)
            sides[axis.fact] = missing
        if axis.regime and gap > d_regime:
            d_regime, deciding_regime = gap, axis.fact
        elif not axis.regime and gap > d_mag:
            d_mag, deciding_mag = gap, axis.fact
    capped = next((a.fact for a in similarity.soft if a.regime and a.fact in dropped), None)
    deciding = deciding_regime or deciding_mag or capped
    if d_regime == 0 and d_mag == 0 and capped is None:
        region: Region = "same"
    elif d_regime <= 1:
        region = "near"
    else:
        region = "outside"
    absent = sides.get(deciding or "") or ("there" if deciding == capped else None)
    return Placed(
        region=region, distance=d_regime, deciding=deciding, unknown=tuple(unknown), absent=absent
    )


def dropped_axes(
    similarity: Similarity, launches_at: Sequence[Mapping[str, object]]
) -> frozenset[str]:
    """The soft axes no launch of a record recorded."""
    if not launches_at:
        return frozenset()
    return frozenset(
        a.fact
        for a in similarity.soft
        if all(not _known(at.get(a.fact, UNKNOWN)) for at in launches_at)
    )
