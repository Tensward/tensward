"""``report.md``: a report's sections laid out as markdown."""

from __future__ import annotations

import re
import textwrap
from dataclasses import replace
from typing import Sequence

from ..combine import NUMERIC
from ..estimate import expected_text
from ..text import figure, table
from .kb import why_line
from .model import Audience, Block, Data, Figure, Report, Section, SuggestionBlock, Table, Text

TERMINAL_WIDTH = 80
BACKING = {
    "strong": "well backed by measurements",
    "moderate": "some measurements back it",
    "weak": "little measured backing",
    "ours": "backed by Tensward's own runs",
}


def shown(audience: Audience, target: Audience) -> bool:
    return audience in ("all", target)


def block_lines(block: Block, *, terminal: bool = False) -> list[str]:
    """One block's lines. On the ``terminal`` an offer says what it does and its risk; why
    Tensward thinks so (priors, backing, costs, knowledge records) stays in report.md."""
    if isinstance(block, Text):
        text = f"**{block.text}**" if block.tone == "strong" else block.text
        return [f"- {text}" if block.item else text]
    if isinstance(block, Figure):
        value = figure(block.value, block.unit, block.digits, grouped=block.grouped)
        return [f"- {block.label}: {value}{block.note}"]
    if isinstance(block, Table):
        return table(block.header, block.rows)[:-1]
    if isinstance(block, Data):
        return []
    if terminal:
        return _terminal_offer(block)
    suggestion = block.suggestion
    entry = suggestion.entry
    basis = f" (from {suggestion.basis})" if suggestion.basis else ""
    members = suggestion.members or (entry,)
    name = f"`{block.change}`" if block.change else " with ".join(f"`{m.name}`" for m in members)
    lines = [
        f"- {name}{block.risk}: {suggestion.reason}{basis}",
    ]
    if block.expected is not None:
        lines.append(f"  expected: {expected_text(block.expected)}")
    if not terminal:
        lines += _why_lines(block)
    if suggestion.command and not suggestion.command.startswith("tensward analyse"):
        lines.append(f"  Try: `{suggestion.command}`")
    elif suggestion.command and block.offer is not None:
        lines.append(f"  Test it: `{try_command(block.offer)}`")
    if block.test:
        lines.append(f"  testing it: {block.test}")
    lines += [f"  {line}" for line in block.plan]
    return lines


def _terminal_offer(block: SuggestionBlock) -> list[str]:
    """An offer as the terminal shows it: the change on a bullet, one short line of why, and
    the command that tests it."""
    suggestion = block.suggestion
    members = suggestion.members or (suggestion.entry,)
    name = block.change or " with ".join(member.name for member in members)
    lines = [f"- {name}{'  (answers may change)' if block.risk else ''}"]
    lines += textwrap.wrap(
        short_reason(suggestion.reason),
        TERMINAL_WIDTH,
        initial_indent="    why: ",
        subsequent_indent="         ",
        break_on_hyphens=False,
    )
    command = suggestion.command
    if command and not command.startswith("tensward analyse"):
        lines.append(f"    Try: {command}")
    elif command and block.offer is not None:
        took = f"   ({block.test.split(' (')[0]})" if block.test else ""
        lines.append(f"    Test it: {try_command(block.offer)}{took}")
    return lines


def short_reason(reason: str) -> str:
    """The first clause of an offer's reason: up to its first semicolon or full stop."""
    found = re.search(r"; |\. ", reason)
    return reason[: found.start()] if found else reason


def _why_lines(block: SuggestionBlock) -> list[str]:
    """Why Tensward expects an offer to help, and what backs it."""
    suggestion = block.suggestion
    entry = suggestion.entry
    members = suggestion.members or (entry,)
    lines = [
        f"  `{member.name}`: {member.prior}"
        for member in suggestion.members
        if member.prior and (suggestion.predicted is None or member.name not in NUMERIC)
    ]
    backing = "; ".join(dict.fromkeys(BACKING.get(m.evidence, m.evidence) for m in members))
    if block.direction:
        lines.append(f"  expected direction: {block.direction}")
    lines.append(f"  how well backed: {backing}; may cost: {suggestion.cost or entry.costs}")
    if block.kb:
        lines.append(f"  {why_line(block.kb)}")
    return lines


def try_command(offer: int) -> str:
    """The command that tests offer ``offer``."""
    return "tensward try" if offer == 1 else f"tensward try {offer}"


def numbered(sections: Sequence[Section]) -> list[Section]:
    """``sections`` with each offered change (one with a command) given its number, in order:
    the ``N`` of ``tensward try N``."""
    count = 0
    out = []
    for section in sections:
        blocks = []
        for block in section.blocks:
            if isinstance(block, SuggestionBlock) and block.suggestion.command:
                count += 1
                block = replace(block, offer=count)
            blocks.append(block)
        out.append(replace(section, blocks=tuple(blocks)))
    return out


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
    parts = [lines for section in numbered(sections) if (lines := _section_lines(section))]
    return "\n\n".join("\n".join(lines) for lines in parts) + "\n"


def to_markdown(report: Report) -> str:
    return sections_markdown(report.sections)
