"""Small formatting helpers shared by every report."""

from __future__ import annotations

import json
import re
import sys
import textwrap
from typing import Any, Sequence

GAP = 2
NARROWEST = 6


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


def plain_rule() -> bool:
    """Whether standard output cannot print "─", so a rule is drawn with "-"."""
    try:
        "─".encode(sys.stdout.encoding or "ascii")
    except (UnicodeEncodeError, LookupError):
        return True
    return False


def columns(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    width: int = 80,
    plain: bool | None = None,
) -> list[str]:
    """``rows`` as aligned columns under ``header`` and a rule, at most ``width`` wide: while
    too wide, the widest value column narrows (the first, of labels, last) and its cells wrap
    inside it. ``plain`` draws the rule
    with "-" (by default when standard output cannot print "─")."""
    count = len(header)
    table = [list(header), *([*row, *[""] * (count - len(row))] for row in rows)]
    widths = [max(len(row[i]) for row in table) for i in range(count)]
    while sum(widths) + GAP * (count - 1) > width:
        values = [i for i in range(1, count) if widths[i] > NARROWEST]
        narrowing = values or [i for i in range(count) if widths[i] > NARROWEST]
        if not narrowing:
            break
        widths[max(narrowing, key=lambda i: widths[i])] -= 1
    rule = "-" if (plain_rule() if plain is None else plain) else "─"
    lines = []
    for index, row in enumerate(table):
        cells = [textwrap.wrap(cell, widths[i]) or [""] for i, cell in enumerate(row)]
        for depth in range(max(map(len, cells))):
            parts = [
                (cell[depth] if depth < len(cell) else "").ljust(widths[i])
                for i, cell in enumerate(cells)
            ]
            lines.append((" " * GAP).join(parts).rstrip())
        if index == 0:
            lines.append(rule * (sum(widths) + GAP * (count - 1)))
    return lines


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
