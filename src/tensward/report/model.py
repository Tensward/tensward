"""A report as data: sections of blocks that a renderer only lays out."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, get_args

from ..combine import stack_risk
from ..estimate import Estimate
from ..playbook import Suggestion

# in the order a report shows them
SectionId = Literal["header", "what_changed", "diagnosis", "next_steps", "worth_measuring",
                    "setup", "requests", "performance", "engine", "defaults", "ceilings",
                    "quantization", "tool_calls", "structured_answers", "context", "checks",
                    "trace", "counters", "extension", "answers", "answers_after", "keep",
                    "feedback"]  # fmt: skip
SECTION_ORDER: tuple[SectionId, ...] = get_args(SectionId)
Audience = Literal["all", "markdown", "terminal", "json"]  # "json": report.json only


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
    expected: Estimate | None = None  # the change's expected effect, when it has an estimate
    test: str | None = None  # what testing it takes here, when it has a command
    plan: tuple[str, ...] = ()  # the predicted steps after it, for the top change
    tools: bool = False  # the requests offer tools: a precision change risks the answers most
    kb: Mapping[str, Any] | None = None  # the knowledge records behind the offer (report.json)
    direction: str = ""  # what a documented record expects of the change, in words
    test_seconds: float | None = None  # launch to the end of its measurement, when known
    start_seconds: float | None = None  # launch to ready, for its start class, when known


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
    goal: Mapping[str, Any] | None = None  # the goal and the declared limits it was written for
    kb: Mapping[str, Any] = field(default_factory=lambda: {"pack": None})  # the knowledge source


def _block(block: Block) -> dict[str, Any]:
    if isinstance(block, SuggestionBlock):
        suggestion = block.suggestion
        entry = suggestion.entry
        names = [member.name for member in suggestion.members]
        members = suggestion.members or (entry,)
        predicted = suggestion.predicted
        risk = stack_risk(suggestion, tools=block.tools)
        return {
            "kind": "suggestion",
            "key": "+".join(names) if names else entry.name,
            "name": entry.name,
            "tier": suggestion.tier,
            "reason": suggestion.reason,
            "near": suggestion.near,
            "basis": suggestion.basis,
            "evidence": entry.evidence,
            "costs": suggestion.cost or entry.costs,
            "quality_risk": any(member.quality_risk for member in members),
            "changes_answers": any(member.changes_answers for member in members),
            **({"answer_risk": risk} if risk else {}),
            **({"prior": entry.prior} if entry.prior else {}),
            **({"prior_gain": dict(entry.prior_gain)} if entry.prior_gain else {}),
            **({"checks_answers": True} if any(m.checks_answers for m in members) else {}),
            "command": suggestion.command,
            "predicted": (
                {
                    "tpot": predicted.tpot,
                    "req_s": predicted.req_s,
                    "ttft": predicted.ttft,
                    "text": predicted.text,
                }
                if predicted
                else None
            ),
            **({"members": names} if names else {}),
            **({"kb": dict(block.kb)} if block.kb else {}),
            **({"direction": block.direction} if block.direction else {}),
            "audience": block.audience,
            **({"expected": _expected(block.expected)} if block.expected else {}),
            **({"test": block.test} if block.test else {}),
            **({"plan": list(block.plan)} if block.plan else {}),
            "test_seconds": block.test_seconds,
            "start_seconds": block.start_seconds,
        }
    kind = {Figure: "figure", Text: "text", Table: "table"}[type(block)]
    return {"kind": kind, **asdict(block)}


def _expected(estimate: Estimate) -> dict[str, Any]:
    return {
        "metric": "throughput",
        "low": math.floor(round(estimate.low * 100, 6)) / 100,  # down: "at least" stays true
        "high": round(estimate.high, 2),
        "inputs": [
            {"key": given.key, "value": round(given.value, 3), "text": given.text}
            for given in estimate.inputs
        ],
        "note": estimate.note,
        "binding": estimate.binding,
    }


def to_json(report: Report) -> dict[str, Any]:
    """``report.json``: every block as a dict with its ``kind``; a suggestion as its entry's
    name, tier, reason, near, basis, evidence, costs, quality_risk, changes_answers and command,
    the answer_risk, prior, prior_gain and checks_answers it has, ``expected`` when set and the
    ``kb`` block of the records behind it; ``goal`` when the report has one; ``kb``, the
    knowledge source."""
    return {
        "schema_version": report.schema_version,
        "run_id": report.run_id,
        "ran": report.ran,
        **({"goal": dict(report.goal)} if report.goal is not None else {}),
        "kb": dict(report.kb),
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
