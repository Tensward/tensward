"""What ``tensward analyse`` prints: a report's terminal sections."""

from __future__ import annotations

import re

from .markdown import block_lines, numbered, shown
from .model import Report, SectionId, Text, phase_two_first

TERMINAL_SECTIONS: tuple[SectionId, ...] = (
    "header", "first_screen", "what_changed", "stands", "diagnosis", "checks", "next_steps",
    "phase_two", "worth_measuring", "placement", "advice", "beyond", "answers", "answers_after",
    "keep",
)  # fmt: skip
ATTACHED: tuple[SectionId, ...] = ("checks", "answers")  # printed under the section before
MOVE = r"(?:[+−-]?\d+%|÷\d+)"
BAND = r" ±\d+(?:\.\d+)?%"
# How report.md words a move against its noise band, and how the terminal says it instead: a
# move within the noise is "no significant change", and the band itself is never printed.
PLAIN = (
    (re.compile(rf"\({MOVE}{BAND}, within noise\)"), "(no significant change)"),
    (re.compile(rf"\(({MOVE}){BAND}, (better|worse)\)"), r"(\1, \2)"),
    (re.compile(rf" {MOVE}{BAND} \(within noise\)"), ": no significant change"),
    (re.compile(rf"within noise: (.+?) {MOVE}{BAND}"), r"no significant change in \1"),
    (re.compile(r"; ± is [^\n]*"), ""),
    (re.compile(BAND), ""),
)


def to_terminal(report: Report) -> str:
    """The terminal sections in their terminal order, a blank line between them."""
    by_id = {section.id: section for section in numbered(report.sections)}
    lines: list[str] = []
    names = list(TERMINAL_SECTIONS)
    if phase_two_first(report):
        names.remove("phase_two")
        names.insert(names.index("next_steps"), "phase_two")
    for name in names:
        if (section := by_id.get(name)) is None:
            continue
        own = []
        if section.title and shown(section.title_audience, "terminal"):
            own += [f"## {section.title}", ""]
        for block in section.blocks:
            if not shown(block.audience, "terminal"):
                continue
            if name == "checks" and isinstance(block, Text):
                own.append(f"  check: {block.text}")
            else:
                own += block_lines(block, terminal=True)
        if own and lines and name not in ATTACHED:
            lines.append("")
        lines += own
    return without_bands("\n".join(lines)) + "\n"


def without_bands(text: str) -> str:
    """``text`` with every noise band taken out and a move within it said in words; report.md
    keeps the bands and how they are computed."""
    for pattern, said in PLAIN:
        text = pattern.sub(said, text)
    return text
