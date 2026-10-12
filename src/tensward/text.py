"""Small formatting helpers shared by every report."""

from __future__ import annotations

import json
import re
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


# Terminal escape sequences (CSI, OSC, and two-character ones), then any other control character
# but newline and tab.
_ESCAPES = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?|\x1b[@-_]?")
_CONTROLS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def printable(text: str) -> str:
    """``text`` (a model's answer) without escape sequences or control characters, safe to print
    to a terminal."""
    return _CONTROLS.sub("", _ESCAPES.sub("", text))


def code_span(text: str) -> str:
    """``text`` as a markdown code span, so markup inside it (HTML, images, links) shows as
    written."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    ticks = "`" * (longest + 1)
    padded = f" {text} " if text.startswith("`") or text.endswith("`") else text
    return f"{ticks}{padded}{ticks}"


def canonical_json(payload: Any) -> bytes:
    """Compact, key-sorted JSON: the same payload always hashes the same."""
    text = json.dumps(
        payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return text.encode()
