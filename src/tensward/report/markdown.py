"""``report.md``: a report's sections laid out as markdown."""

from __future__ import annotations

from typing import Sequence

from ..text import figure, table
from .model import Audience, Block, Figure, Report, Section, SuggestionBlock, Table, Text


def shown(audience: Audience, target: Audience) -> bool:
    return audience in ("all", target)


def block_lines(block: Block) -> list[str]:
    """One block's lines, the same in every renderer."""
    if isinstance(block, Text):
        text = f"**{block.text}**" if block.tone == "strong" else block.text
        return [f"- {text}" if block.item else text]
    if isinstance(block, Figure):
        value = figure(block.value, block.unit, block.digits, grouped=block.grouped)
        return [f"- {block.label}: {value}{block.note}"]
    if isinstance(block, Table):
        return table(block.header, block.rows)[:-1]
    suggestion = block.suggestion
    entry = suggestion.entry
    basis = f" (from {suggestion.basis})" if suggestion.basis else ""
    lines = [
        f"- `{entry.name}`{block.risk}: {suggestion.reason}{basis}",
        f"  evidence: {entry.evidence}; may cost: {suggestion.cost or entry.costs}",
    ]
    if suggestion.command:
        lines.append(f"  Try: `{suggestion.command}`")
    return lines


def _listed(block: Block) -> bool:
    return isinstance(block, (Figure, SuggestionBlock)) or (isinstance(block, Text) and block.item)


def _section_lines(section: Section) -> list[str]:
    """A section's title and blocks. A blank line ends a list or a table before what follows."""
    titled = section.title and shown(section.title_audience, "markdown")
    lines = [f"## {section.title}", ""] if titled else []
    previous: Block | None = None
    for block in section.blocks:
        if not shown(block.audience, "markdown"):
            continue
        ends = isinstance(previous, Table) or (
            previous is not None and _listed(previous) and not _listed(block)
        )
        rendered = block_lines(block)
        if ends and lines and lines[-1] and rendered != [""]:
            lines.append("")
        lines += rendered
        previous = block
    return lines


def sections_markdown(sections: Sequence[Section]) -> str:
    """The sections that have anything for markdown, a blank line between them."""
    parts = [lines for section in sections if (lines := _section_lines(section))]
    return "\n\n".join("\n".join(lines) for lines in parts) + "\n"


def to_markdown(report: Report) -> str:
    return sections_markdown(report.sections)
