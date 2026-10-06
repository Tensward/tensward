"""A report as data: sections of blocks that a renderer only lays out."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from ..playbook import Suggestion

SectionId = Literal["header", "what_changed", "diagnosis", "next_steps", "setup", "requests",
                    "performance", "engine", "defaults", "ceilings", "quantization",
                    "tool_calls", "context", "checks", "trace", "counters", "extension",
                    "answers", "feedback"]  # fmt: skip
Audience = Literal["all", "markdown", "terminal"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Figure:
    key: str
    label: str
    value: float | None
    unit: str = ""
    digits: int = 1
    grouped: bool = False  # thousands separated with commas
    note: str = ""
    audience: Audience = "all"


@dataclass(frozen=True, slots=True, kw_only=True)
class Text:
    key: str
    text: str
    tone: Literal["plain", "strong", "warning"] = "plain"
    item: bool = False  # a list item ("- " in markdown)
    audience: Audience = "all"


def item(key: str, text: str, tone: Literal["plain", "strong", "warning"] = "plain") -> Text:
    """A list item of a section."""
    return Text(key=key, text=text, tone=tone, item=True)


@dataclass(frozen=True, slots=True, kw_only=True)
class Table:
    key: str
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    audience: Audience = "all"


@dataclass(frozen=True, slots=True, kw_only=True)
class SuggestionBlock:
    suggestion: Suggestion  # tier, reason, costs, evidence, command
    risk: str = ""  # what its quality risk means for this run, when the entry has one
    audience: Audience = "all"


Block = Figure | Text | Table | SuggestionBlock


@dataclass(frozen=True, slots=True, kw_only=True)
class Section:
    id: SectionId
    title: str | None
    blocks: tuple[Block, ...]
    title_audience: Audience = "all"


@dataclass(frozen=True, slots=True, kw_only=True)
class Report:
    schema_version: Literal["1"] = "1"
    run_id: str
    ran: str
    sections: tuple[Section, ...]


def _block(block: Block) -> dict[str, Any]:
    if isinstance(block, SuggestionBlock):
        suggestion = block.suggestion
        entry = suggestion.entry
        return {
            "kind": "suggestion",
            "key": entry.name,
            "name": entry.name,
            "tier": suggestion.tier,
            "reason": suggestion.reason,
            "near": suggestion.near,
            "basis": suggestion.basis,
            "evidence": entry.evidence,
            "costs": suggestion.cost or entry.costs,
            "quality_risk": entry.quality_risk,
            "command": suggestion.command,
            "audience": block.audience,
        }
    kind = {Figure: "figure", Text: "text", Table: "table"}[type(block)]
    return {"kind": kind, **asdict(block)}


def to_json(report: Report) -> dict[str, Any]:
    """``report.json``: every block as a dict with its ``kind``; a suggestion as its entry's
    name, tier, reason, near, basis, evidence, costs, quality_risk and command."""
    return {
        "schema_version": report.schema_version,
        "run_id": report.run_id,
        "ran": report.ran,
        "sections": [
            {
                "id": section.id,
                "title": section.title,
                "title_audience": section.title_audience,
                "blocks": [_block(block) for block in section.blocks],
            }
            for section in report.sections
        ],
    }
