"""The changes the user rejected with ``tensward reject``: a run's change (its engine arguments
beyond those of the run it was made from) that ``analyse`` and ``report`` no longer offer."""

from __future__ import annotations

import json
import shlex
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .errors import PROJECT_RECORD_INVALID, PreflightError
from .files import write_private
from .kept import LAST, OTHER_REGISTRATION, REGISTERED, newest_run, read_kept, recorded_run

REJECTED_FILE = "rejected.json"


@dataclass(frozen=True, slots=True)
class Rejection:
    run: str  # the run whose change was rejected
    engine_args: tuple[str, ...]  # its --engine-arg values, as run.json recorded them
    change: tuple[str, ...]  # those beyond the engine arguments of the run it was made from
    made_from: str | None  # that run; None when no earlier run is one
    reason: str | None
    at: str

    @property
    def words(self) -> str:
        return " ".join(self.change)

    @property
    def because(self) -> str:
        return f"rejected by you ({self.reason})" if self.reason else "rejected by you"


class NotRejected(Exception):
    """A run whose change cannot be rejected; ``str(error)`` is the one line shown to the user."""


def read_rejected(project: Path) -> tuple[Rejection, ...]:
    """The project's rejections, oldest first; none when nothing was rejected."""
    path = project.expanduser() / REJECTED_FILE
    try:
        rows = json.loads(path.read_text("utf-8"))["rejected"]
        return tuple(
            Rejection(
                run=row["run"],
                engine_args=tuple(row["engine_args"]),
                change=tuple(row["change"]),
                made_from=row["made_from"],
                reason=row["reason"],
                at=row["at"],
            )
            for row in rows
        )
    except FileNotFoundError:
        return ()
    except (OSError, ValueError, KeyError, TypeError):
        raise PreflightError(
            PROJECT_RECORD_INVALID,
            f"{path} is not readable; remove it to offer every change again",
        ) from None


def _write(project: Path, rejections: Sequence[Rejection]) -> None:
    rows = [asdict(rejection) for rejection in rejections]
    write_private(project / REJECTED_FILE, json.dumps({"rejected": rows}, indent=2) + "\n")


def refused_by(
    rejections: Sequence[Rejection], command: frozenset[str], args: frozenset[str]
) -> Rejection | None:
    """The rejection a command's engine arguments ``command`` would bring back to a run with
    ``args``: its change is in the command and not in the run."""
    return next(
        (r for r in rejections if set(r.change) <= command and not set(r.change) <= args), None
    )


def carried(rejections: Sequence[Rejection], args: frozenset[str]) -> Rejection | None:
    """The newest rejection whose change a run with ``args`` carries."""
    return next((r for r in reversed(rejections) if set(r.change) <= args), None)


def reject(project: Path, run: str, reason: str | None) -> Rejection:
    """Reject ``run``'s change (``LAST``: the newest run's); raises NotRejected for a run whose
    change cannot be rejected."""
    from .history import parent, past_runs
    from .project import load_project, project_engine
    from .runs import RunFile

    project = project.expanduser()
    resolved = load_project(project)
    run = newest_run(project, NotRejected) if run == LAST else run
    run_file = RunFile.read(recorded_run(project, run, NotRejected))
    if run_file.snapshot_id != resolved.record.snapshot_id:
        raise NotRejected(f"{OTHER_REGISTRATION.format(run=run)}, so its change is not this "
                          "project's to reject")  # fmt: skip
    args = frozenset(run_file.engine_args)
    if not args:
        raise NotRejected(
            f"run {run} measured the registered setup (no engine arguments): it has no change "
            "to reject"
        )
    others = [r for r in read_rejected(project) if r.run != run]
    engine = project_engine(resolved.record.current_setup)
    runs = past_runs(project, engine, before=run)
    before = parent(args, [r for r in runs if carried(others, r.engine_args) is None])
    change = args - before.engine_args if before else args
    kept = read_kept(project)
    if kept is not None and change <= frozenset(kept.engine_args):
        named = f"measured the kept setup (run {kept.run})"
        if run == kept.run:
            named = "is the kept setup"
        elif args != frozenset(kept.engine_args):
            words = " ".join(sorted(change))
            named = f"made a change ({words}) that is part of the kept setup (run {kept.run})"
        go_back = shlex.join(["tensward", "set", REGISTERED, "--project", str(project)])
        raise NotRejected(
            f"run {run} {named}: go back first with `{go_back}` (or keep another run with "
            "`tensward set RUN`), then reject it"
        )
    rejection = Rejection(
        run=run,
        engine_args=run_file.engine_args,
        change=tuple(sorted(change)),
        made_from=before.run_id if before else None,
        reason=reason,
        at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    _write(project, [*others, rejection])
    return rejection


def clear(project: Path, run: str) -> Rejection | None:
    """Take back the rejection of ``run``'s change; None when it was not rejected."""
    project = project.expanduser()
    rejections = read_rejected(project)
    found = next((r for r in rejections if r.run == run), None)
    if found is not None:
        _write(project, [r for r in rejections if r is not found])
    return found


def _command(*words: str) -> str:
    return shlex.join(["tensward", *words])


def reject_lines(project: Path, rejection: Rejection, *, latest: bool) -> list[str]:
    """What ``tensward reject`` prints: the change, why, the next step and the undo."""
    project = project.expanduser()
    named = f"run {rejection.run} (the latest run)" if latest else f"run {rejection.run}"
    source = f"run {rejection.made_from}" if rejection.made_from else "the registered setup"
    lines = [
        f"rejected the change of {named}: {rejection.words} (made from {source})",
        f"reason: {rejection.reason or 'none given'}",
        "tensward analyse and tensward report no longer offer it for this project",
    ]
    if rejection.made_from is not None:
        report = _command("report", rejection.made_from, "--project", str(project))
        lines.append(f"next:   {report} (the next change from run {rejection.made_from})")
    clear_it = _command("reject", "--clear", rejection.run, "--project", str(project))
    lines.append(f"undo:   {clear_it}")
    return lines


def list_lines(rejections: Sequence[Rejection]) -> list[str]:
    """What ``tensward reject --list`` prints: one line per rejected change, oldest first."""
    if not rejections:
        return ["nothing is rejected in this project"]
    return [f"run {r.run}: {r.words}, {r.because}, at {r.at}" for r in rejections]


def clear_lines(run: str, cleared: Rejection | None) -> list[str]:
    """What ``tensward reject --clear`` prints."""
    if cleared is None:
        return [f"run {run}'s change was not rejected; nothing to clear"]
    return [f"cleared: run {run}'s change ({cleared.words}) can be offered again"]
