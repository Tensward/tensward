"""Where a command finds its project: ``.tensward`` in this folder or a parent, like git."""

from __future__ import annotations

from pathlib import Path

from .errors import PROJECT_NOT_REGISTERED, PreflightError

PROJECT_DIR = ".tensward"
PROJECT_DOCUMENT = "project.json"


def find_project(start: Path) -> Path | None:
    """The ``.tensward`` folder of ``start`` or its nearest parent that holds a project."""
    for folder in (start, *start.parents):
        if (folder / PROJECT_DIR / PROJECT_DOCUMENT).is_file():
            return folder / PROJECT_DIR
    return None


def project_for(command: str, given: Path | None) -> Path:
    """``given`` wins; ``init`` makes ``./.tensward``; any other command needs a found project."""
    if given is not None:
        return given
    if command == "init":
        return Path.cwd() / PROJECT_DIR
    found = find_project(Path.cwd())
    if found is None:
        raise PreflightError(
            PROJECT_NOT_REGISTERED, "no Tensward project in this folder or its parents"
        )
    return found
