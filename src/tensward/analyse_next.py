"""What ``tensward analyse`` ends with: a one-line result and the menu of what to run next,
built from the offers ``tensward try`` itself would make, so the two cannot disagree."""

from __future__ import annotations

import math
from pathlib import Path

from .report.beyond import beyond_choices
from .report.model import Report, SuggestionBlock, Text, to_json
from .report.summary import SIDE_GAINS

SERVE_TEXT = "serve your current setup as it is"


def result_of(report: Report) -> str | None:
    """The report's one-line result, when it has one."""
    for section in report.sections:
        for block in section.blocks:
            if isinstance(block, Text) and block.key == "result":
                return block.text
    return None


def _changes(report: Report) -> dict[str, str]:
    return {
        block.suggestion.entry.name: block.change
        for section in report.sections
        for block in section.blocks
        if isinstance(block, SuggestionBlock) and block.change
    }


def menu(project: Path, report: Report | None) -> list[tuple[str, str]]:
    """(command, what it does) for each next step: testing the first offer when ``try`` has
    one, the steps that measure the first Beyond flags alternative as a new project (first
    when nothing is offered for the goal), and serving what is set now."""
    from .cli import first_analysis_minutes
    from .cli_try import UNTIMED, NothingToTry, offers, untimed

    choices: list[tuple[str, str]] = []
    try:
        first = offers(project)[0]
    except NothingToTry:
        first = None
    beyond = beyond_choices(to_json(report)) if report is not None else []
    if first is not None:
        seconds = first.test_seconds
        if untimed(project):
            took = UNTIMED
        else:
            took = f"~{math.ceil(seconds / 60) if seconds else first_analysis_minutes(project)} min"
        change = (_changes(report) if report else {}).get(first.name) or first.name
        if first.side in SIDE_GAINS:
            text = f"test {change} for {SIDE_GAINS[first.side]} ({took})"
        else:
            text = f"test the first suggested change, {change} ({took})"
        choices.append(("tensward try", text))
    main = first is None or first.side in SIDE_GAINS
    choices = [*beyond, *choices] if main else [*choices, *beyond]
    choices.append(("tensward serve", SERVE_TEXT))
    return choices


def lines(project: Path, report: Report) -> list[str]:
    """The result line and the ``Next:`` menu, ready to print."""
    from .cli_try import next_lines

    result = result_of(report)
    return [
        *([result, ""] if result else []),
        *next_lines(menu(project, report)),
    ]


def fix_lines(project: Path) -> list[str]:
    """The ``Next:`` menu for a setup that cannot start: testing the offered fix. Serving would
    fail the same way, so it is not offered."""
    from .cli_try import next_lines

    choices = fix_choices(project)
    return next_lines(choices) if choices else []


def fix_choices(project: Path) -> list[tuple[str, str]]:
    """``tensward try`` to test the offered fix, with its minutes; empty without an offer."""
    from .cli import first_analysis_minutes
    from .cli_try import NothingToTry, offers

    try:
        first = offers(project)[0]
    except NothingToTry:
        return []
    minutes = (
        math.ceil(first.test_seconds / 60)
        if first.test_seconds
        else first_analysis_minutes(project)
    )
    return [("tensward try", f"test it (~{minutes} min)")]
