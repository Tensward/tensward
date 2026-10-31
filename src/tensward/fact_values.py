"""The value a fact reads when Tensward did not measure it."""

from __future__ import annotations

from typing import Final


class _Unknown:
    __slots__ = ()

    def __repr__(self) -> str:
        return "UNKNOWN"


UNKNOWN: Final[object] = _Unknown()
