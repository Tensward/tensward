"""Names for the runs of a project: ``last``, ``baseline`` or a run's number."""

from __future__ import annotations

import json
from pathlib import Path

from .errors import PROJECT_INPUTS_INVALID, PreflightError
from .files import NUMBERED_RUN_ID

LAST = "last"
BASELINE = "baseline"


def _complete_runs(runs: Path) -> list[Path]:
    found = [
        path
        for path in (runs.iterdir() if runs.is_dir() else ())
        if (path / "run.json").is_file() and (path / "metrics.json").is_file()
    ]
    return sorted(found, key=lambda path: path.name)


def _is_baseline(run_dir: Path) -> bool:
    try:
        record = json.loads((run_dir / "run.json").read_text("utf-8"))
    except (OSError, ValueError):
        return False
    return not record.get("engine_args") and not record.get("inherited_env")


def resolve_run(project: Path, text: str) -> str:
    """The id of the run ``text`` names in the project: ``last`` is the newest complete run,
    ``baseline`` the newest one with no engine arguments and no inherited environment, ``3`` or
    ``003`` the run numbered 3; anything else is returned as given."""
    runs = project.expanduser() / "runs"
    found: Path | None = None
    if text == LAST:
        found = next(iter(reversed(_complete_runs(runs))), None)
    elif text == BASELINE:
        found = next((p for p in reversed(_complete_runs(runs)) if _is_baseline(p)), None)
    elif text.isdecimal():
        wanted = int(text)
        found = next(
            (
                path
                for path in (runs.iterdir() if runs.is_dir() else ())
                if (named := NUMBERED_RUN_ID.fullmatch(path.name)) and int(named.group(1)) == wanted
            ),
            None,
        )
    else:
        return text
    if found is None:
        raise PreflightError(PROJECT_INPUTS_INVALID, f"no run {text} in {runs}")
    return found.name
