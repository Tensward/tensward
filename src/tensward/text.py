"""Small formatting helpers shared by every report."""

from __future__ import annotations

import json
from typing import Any, Sequence


def figure(value: float | None, unit: str = "", digits: int = 1, *, grouped: bool = False) -> str:
    """``value`` with ``digits`` decimals and ``unit``, or "not measured". ``grouped`` separates
    thousands with commas."""
    if value is None:
        return "not measured"
    return f"{value:{',' if grouped else ''}.{digits}f}{unit}"


def percent(fraction: float | None) -> float | None:
    """A fraction as a percentage."""
    return None if fraction is None else fraction * 100


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """A markdown table, with a blank line after it."""
    return [
        f"| {' | '.join(header)} |",
        f"|{'---|' * len(header)}",
        *(f"| {' | '.join(row)} |" for row in rows),
        "",
    ]


def canonical_json(payload: Any) -> bytes:
    """Compact, key-sorted JSON: the same payload always hashes the same."""
    text = json.dumps(
        payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return text.encode()
