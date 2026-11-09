"""A report as data: sections of blocks that a renderer only lays out."""

from __future__ import annotations

import math
import shlex
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, get_args

from ..combine import stack_risk
from ..estimate import Estimate
from ..playbook import Suggestion

# in the order a report shows them
SectionId = Literal["header", "first_screen", "what_changed", "stands", "diagnosis", "next_steps",
                    "phase_two", "worth_measuring", "beyond", "placement", "advice", "setup",
                    "requests", "performance", "engine", "defaults", "ceilings",
                    "quantization", "tool_calls", "structured_answers", "context", "checks",
                    "trace", "counters", "extension", "answers", "answers_after", "keep",
                    "feedback"]  # fmt: skip
SECTION_ORDER: tuple[SectionId, ...] = get_args(SectionId)


def ordered(sections: Sequence[Section], *, first: bool = False) -> tuple[Section, ...]:
    """``sections`` in report order; ``phase_two`` comes before ``next_steps`` when ``first``
    (the project chain has no change left for the goal)."""
    rank: dict[str, float] = {name: at for at, name in enumerate(SECTION_ORDER)}
    if first:
        rank["phase_two"] = rank["next_steps"] - 0.5
    return tuple(sorted(sections, key=lambda s: rank[s.id]))


def phase_two_first(report: Report) -> bool:
    """Whether ``report`` lists ``phase_two`` before ``next_steps``."""
    ids = [s.id for s in report.sections]
    return (
        "phase_two" in ids
        and "next_steps" in ids
        and ids.index("phase_two") < ids.index("next_steps")
    )


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
    change: str = ""  # what it changes, as flags (``--threads 16``), when it has a command
    offer: int | None = None  # its number in ``tensward try N``, when it has a command


@dataclass(frozen=True, slots=True, kw_only=True)
class Data:
    """Values for report.json alone: they are written beside ``key`` and never laid out."""

    key: str
    data: Mapping[str, Any]
    audience: Audience = "json"


Block = Figure | Text | Table | SuggestionBlock | Data


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
    versions: Mapping[str, Any] = field(
        default_factory=dict
    )  # Tensward, engine, hardware, baseline


def _offered_arguments(command: str | None, flag: str) -> list[str]:
    """The values a ``tensward analyse`` command gives ``flag``, in order."""
    words = shlex.split(command or "")
    return [words[at + 1] for at, word in enumerate(words[:-1]) if word == flag]


def change_words(command: str | None, held: Collection[str] = ()) -> str:
    """What a ``tensward analyse`` command changes, as flags (``--threads 16``): its engine
    arguments and variables except ``held``, the ones the setup already has."""
    shown = []
    for flag in ("--engine-arg", "--engine-env"):
        for arg in _offered_arguments(command, flag):
            if arg in held:
                continue
            name, given, value = arg.partition("=")
            if flag == "--engine-env":
                shown.append(arg)
            else:
                shown.append(f"--{name} {value}" if given else f"--{name}")
    return " ".join(shown)


def _block(block: Block) -> dict[str, Any]:
    if isinstance(block, Data):
        return {"kind": "data", "key": block.key, **block.data, "audience": block.audience}
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
            **({"side": suggestion.side} if suggestion.side else {}),
            "command": suggestion.command,
            "engine_args": _offered_arguments(suggestion.command, "--engine-arg"),
            "engine_env": _offered_arguments(suggestion.command, "--engine-env"),
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
            "change": block.change,
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
    name, tier, reason, near, basis, evidence, costs, quality_risk, changes_answers, command and
    the engine_args and engine_env of that command,
    the answer_risk, prior, prior_gain and checks_answers it has, ``expected`` when set and the
    ``kb`` block of the records behind it; ``goal`` when the report has one; ``kb``, the
    knowledge source."""
    return {
        "schema_version": report.schema_version,
        "run_id": report.run_id,
        "ran": report.ran,
        **({"goal": dict(report.goal)} if report.goal is not None else {}),
        "kb": dict(report.kb),
        **({"versions": dict(report.versions)} if report.versions else {}),
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
