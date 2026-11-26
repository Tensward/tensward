"""The value a fact reads when Tensward did not measure it, and the description of a fact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal


class _Unknown:
    __slots__ = ()

    def __repr__(self) -> str:
        return "UNKNOWN"


UNKNOWN: Final[object] = _Unknown()


FactType = Literal["int", "float", "bool", "str", "enum", "set", "version"]


@dataclass(frozen=True, slots=True, kw_only=True)
class FactSpec:
    """One fact. ``values`` lists an enum's values, or the element values of a set of names;
    a set without ``values`` holds integers."""

    name: str
    type: FactType
    unit: str | None  # "GiB", "ms", "tokens", "share", None
    values: tuple[str, ...] | None
    source: str  # "situation", "context", "anatomy", "engine", "measurement"
    description: str
    engines: tuple[str, ...] = ()  # the engines that define it with this meaning; empty: all
