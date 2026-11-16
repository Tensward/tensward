"""What ``tensward analyse`` prints: a report's terminal sections."""

from __future__ import annotations

from ..wording import plain
from .markdown import block_lines, numbered, shown
from .model import Report, SectionId, Text, phase_two_first

TERMINAL_SECTIONS: tuple[SectionId, ...] = (
    "header", "first_screen", "what_changed", "stands", "diagnosis", "checks", "next_steps",
    "phase_two", "worth_measuring", "placement", "advice", "beyond", "answers", "answers_after",
    "keep",
)  # fmt: skip
ATTACHED: tuple[SectionId, ...] = ("checks", "answers")  # printed under the section before


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
                own.append(f"  check: {plain(block.text)}")
            else:
                own += block_lines(block, terminal=True)
        if own and lines and name not in ATTACHED:
            lines.append("")
        lines += own
    return "\n".join(lines) + "\n"
