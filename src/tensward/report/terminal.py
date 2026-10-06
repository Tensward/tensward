"""What ``tensward analyse`` prints: a report's terminal sections."""

from __future__ import annotations

from .markdown import block_lines, shown
from .model import Report, SectionId, Text

TERMINAL_SECTIONS: tuple[SectionId, ...] = (
    "header", "what_changed", "diagnosis", "checks", "next_steps", "answers"
)  # fmt: skip
ATTACHED: tuple[SectionId, ...] = ("checks", "answers")  # printed under the section before


def to_terminal(report: Report) -> str:
    """The terminal sections in their terminal order, a blank line between them."""
    by_id = {section.id: section for section in report.sections}
    lines: list[str] = []
    for name in TERMINAL_SECTIONS:
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
                own += block_lines(block)
        if own and lines and name not in ATTACHED:
            lines.append("")
        lines += own
    return "\n".join(lines) + "\n"
