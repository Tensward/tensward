"""The analysis running in a project: which process runs it, since when, for about how long,
and how ``tensward stop`` ends it."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .files import write_json
from .runtime import (
    container_running,
    process_start_time,
    stop_container,
    stop_local_process,
)

RUNNING_FILE = "analysis.running.json"
LEFTOVER_RECORDS = 3  # an analysis killed outright leaves the server of its newest runs
STOP_WAIT_S = 90.0  # the analysis stops its server (and a census start) before it exits
_HELD: dict[Path, int] = {}  # how deep this process is in recording() for each project


@dataclass(frozen=True, slots=True)
class Running:
    pid: int
    start_time: str
    started_at: float  # epoch seconds
    minutes: int | None  # what the screen said before it started
    command: str = "analyse"  # or "try"

    @property
    def noun(self) -> str:
        """``analysis`` or ``try``, as the user started it."""
        return "try" if self.command == "try" else "analysis"

    def started(self) -> str:
        return time.strftime("%H:%M", time.localtime(self.started_at))

    def minutes_left(self, now: float | None = None) -> int | None:
        if self.minutes is None:
            return None
        elapsed = ((time.time() if now is None else now) - self.started_at) / 60
        return max(round(self.minutes - elapsed), 0)

    def said(self) -> str:
        """``pid 8339, started 21:05, about 20 min left``."""
        left = self.minutes_left()
        tail = "" if left is None else ", nearly done" if left == 0 else f", about {left} min left"
        return f"pid {self.pid}, started {self.started()}{tail}"


def _path(project: Path) -> Path:
    return Path(project).expanduser() / RUNNING_FILE


def read_running(project: Path) -> Running | None:
    """The analysis running in ``project``, None when none is (a record left by a process that
    is gone counts as none)."""
    try:
        document = json.loads(_path(project).read_text("utf-8"))
        running = Running(
            int(document["pid"]),
            str(document["start_time"]),
            float(document["started_at"]),
            None if document.get("minutes") is None else int(document["minutes"]),
            str(document.get("command", "analyse")),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return running if _alive(running) else None


def _alive(running: Running) -> bool:
    return process_start_time(running.pid) == running.start_time


@contextlib.contextmanager
def recording(project: Path, minutes: int | None, command: str = "analyse") -> Iterator[None]:
    """Record this process as the project's running analysis (``command`` is ``analyse`` or
    ``try``) for the duration of the block. A nested block (each step of a ``try``) updates the
    estimate; the outermost removes it. The record of another process that is still running is
    left as it is."""
    path = _path(project)
    key = path.resolve()
    pid = os.getpid()
    start_time = process_start_time(pid)
    other = read_running(project)
    if start_time is not None and (other is None or other.pid == pid):
        with contextlib.suppress(OSError):
            write_json(
                path,
                {"pid": pid, "start_time": start_time, "started_at": time.time(),
                 "minutes": minutes, "command": other.command if other else command},
            )  # fmt: skip
    _HELD[key] = _HELD.get(key, 0) + 1
    try:
        yield
    finally:
        _HELD[key] -= 1
        if not _HELD[key]:
            del _HELD[key]
            with contextlib.suppress(OSError):
                held = json.loads(path.read_text("utf-8"))
                if held.get("pid") == pid:
                    path.unlink()


def stop_running(project: Path, wait_s: float = STOP_WAIT_S) -> tuple[Running, bool] | None:
    """Stop the project's running analysis the way Ctrl-C does (it stops its server, then
    exits); what it leaves behind after ``wait_s`` is killed. The analysis and whether it is
    gone; None when nothing was running."""
    from .serve import process_tree

    running = read_running(project)
    if running is None:
        return None
    tree = {pid: process_start_time(pid) for pid in process_tree(running.pid) - {os.getpid()}}
    own_group = os.getpgrp()
    with contextlib.suppress(ProcessLookupError):
        os.kill(running.pid, signal.SIGTERM)
    deadline = time.monotonic() + wait_s
    while _alive(running) and time.monotonic() < deadline:
        time.sleep(0.2)
    for pid, started in tree.items():
        if started is None or process_start_time(pid) != started:
            continue
        with contextlib.suppress(OSError):
            if pid not in (running.pid, own_group) and os.getsid(pid) == pid:  # a server
                os.killpg(pid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
    time.sleep(0.2)
    gone = not _alive(running)
    if gone:
        _path(project).unlink(missing_ok=True)
    return running, gone


def stop_leftovers(project: Path) -> list[str]:
    """Stop the servers that an analysis started and did not stop (it was killed): the newest
    run records whose server is still the same process or container, and drop the record of
    the analysis that is gone. What each one was."""
    if read_running(project) is None:
        _path(project).unlink(missing_ok=True)
    records = sorted(
        Path(project).expanduser().glob("runs/**/serve.json"),
        key=lambda path: path.stat().st_mtime,
    )[-LEFTOVER_RECORDS:]
    stopped = []
    for record in records:
        try:
            identity = json.loads(record.read_text("utf-8"))["identity"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        run = record.parent.relative_to(Path(project).expanduser() / "runs")
        pid, start_time = identity.get("pid"), identity.get("start_time")
        name = identity.get("container_name")
        if isinstance(pid, int) and start_time and process_start_time(pid) == start_time:
            stop_local_process(pid, start_time)
            stopped.append(f"run {run}, pid {pid}")
        elif identity.get("runtime") == "docker" and name and container_running(name):
            stop_container(name)
            stopped.append(f"run {run}, container {name}")
    return stopped
