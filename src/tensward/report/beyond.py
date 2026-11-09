"""The "Beyond flags" section: measured alternatives to the user's file or engine, each with the
commands that measure it on their machine as a new project."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .. import alternatives
from ..alternatives import (
    DECIDES,
    DECIDES_ON,
    METRIC_WORDS,
    SECRET_NOTE,
    evidence_lines,
    heading,
    matching,
    model_identity,
    percent,
    short_name,
    steps,
    table_row,
    wrapped,
)
from ..classify import Diagnosis
from ..errors import PreflightError
from ..project import registered_model, registered_setup
from ..text import code_span, columns, printable
from .model import Data, Section, Text

POWER_CAP_LEAD = (
    "No server flag lifts the GPU's power cap; a file or an engine that does less work per "
    "token can."
)
ABOVE = "steps above, under Beyond flags"
IN_REPORT = "steps in report.md, under Beyond flags"
MENU_TEXT = "measure {label} as a new project (" + ABOVE + ")"
SUMMARY = "Beyond flags: {count} ({options}); see the first report or report.md"
OTHERS = 'Steps for {which}: report.md, section "Beyond flags".'
SHORT_DECIDES = "Answers differ with another file or engine; the new project's check decides."


def beyond_section(
    facts: Mapping[str, object] | None, goal: str, diagnosis: Diagnosis, project: Path | None
) -> Section | None:
    """None when no measured alternative passes every check for this setup and ``goal``. The
    terminal shows a table of them and the numbered steps of the first, the largest gain, whose
    first step is the ``Next:`` menu's one line; report.md has every option's steps."""
    if facts is None or project is None or not alternatives.table():
        return None
    try:
        model, setup = registered_model(project), registered_setup(project)
    except PreflightError:
        return None
    if model is None:
        return None
    facts = {**facts, "model.identity": model_identity(model)}
    found = matching(facts, goal)
    if not found:
        return None
    current = setup.text if setup is not None else None
    capped = "power_cap" in (diagnosis.primary, *diagnosis.secondary)
    lead = [*textwrap.wrap(POWER_CAP_LEAD, 80), ""] if capped else []
    header = ("#", "Option", METRIC_WORDS[DECIDES_ON[goal]].capitalize(), "Answers")
    rows = [table_row(at, alternative, goal) for at, alternative in enumerate(found, 1)]
    terminal = [
        *lead,
        heading(found[0]),
        "",
        *columns(header, rows, width=80),
    ]
    markdown = [POWER_CAP_LEAD] if capped else []
    for at, alternative in enumerate(found, 1):
        choices, screen = steps(alternative, facts, current, model, project)
        if at == 1:
            menu = [(_first_step(choices), MENU_TEXT.format(label=short_name(alternative)))]
            terminal += ["", "  To try option 1:"]
            for number, line in enumerate(screen, 1):
                terminal += wrapped(line, f"    {number}. ")
            if "***" in screen[-1]:
                terminal.append(f"       {SECRET_NOTE}")
        markdown += evidence_lines(alternative)
        markdown += [f"  {number}. {code_span(line)}" for number, line in enumerate(screen, 1)]
        if "***" in screen[-1]:
            markdown.append(f"     {SECRET_NOTE}")
    if len(found) > 1:
        rest = [str(at) for at in range(2, len(found) + 1)]
        which = f"option {rest[0]}" if len(rest) == 1 else f"options {', '.join(rest)}"
        terminal.append(f"  {OTHERS.format(which=which)}")
    terminal.append(f"  {SHORT_DECIDES}")
    markdown.append(DECIDES)
    options = ", ".join(f"{short_name(a)} {percent(a.changes[DECIDES_ON[goal]])}" for a in found)
    count = "1 measured option" if len(found) == 1 else f"{len(found)} measured options"
    data: dict[str, Any] = {
        "next": [list(choice) for choice in menu],
        "summary": SUMMARY.format(count=count, options=options),
    }
    return Section(
        id="beyond",
        title="Beyond flags",
        title_audience="markdown",
        blocks=(
            Text(key="beyond", text="\n".join(terminal), audience="terminal"),
            Text(key="beyond", text="\n".join(markdown), audience="markdown"),
            Data(key="beyond", data=data),
        ),
    )


def _first_step(choices: list[tuple[str, str]]) -> str:
    """The first command to run: an install or a download, else the new folder and its init
    on one line."""
    first = choices[0][0]
    return f"{first} && {choices[1][0]}" if first.startswith("mkdir ") else first


def beyond_summary(report: Mapping[str, Any]) -> str | None:
    """The one line ``try`` prints of a report.json's Beyond flags section."""
    return next(
        (
            printable(str(block["summary"]))
            for section in report.get("sections", ())
            for block in section.get("blocks", ())
            if block.get("kind") == "data" and block.get("key") == "beyond"
        ),
        None,
    )


def beyond_choices(report: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The ``Next:`` choices a report.json's Beyond flags section holds."""
    return [
        (printable(str(command)), printable(str(text)))
        for section in report.get("sections", ())
        for block in section.get("blocks", ())
        if block.get("kind") == "data" and block.get("key") == "beyond"
        for command, text in block.get("next", ())
    ]
