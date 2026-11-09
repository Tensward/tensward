"""What a long command is doing, as events. The default sink prints them to standard error
with the seconds since the command began, so the command's own output stays clean for pipes;
an app sets its own sink for a block of work. Inside :func:`live` a wait also shows as a status
line: a spinner rewritten in place on a terminal, a plain line every 30 s otherwise."""

from __future__ import annotations

import atexit
import math
import os
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, TextIO, cast

_START = time.monotonic()

Phase = Literal["preflight", "launch", "warmup", "measure", "trace", "counters", "compare",
                "diagnose", "write", "stop"]  # fmt: skip


def elapsed() -> float:
    return time.monotonic() - _START


@dataclass(frozen=True, slots=True, kw_only=True)
class ProgressEvent:
    at_s: float = field(default_factory=elapsed)  # seconds since the command began


@dataclass(frozen=True, slots=True, kw_only=True)
class PhaseStarted(ProgressEvent):
    phase: Phase
    detail: str = ""  # the terminal line; empty prints nothing


@dataclass(frozen=True, slots=True, kw_only=True)
class ServerLoading(ProgressEvent):
    phase: Phase
    waited_s: float  # 0 when loading begins, then every STILL_LOADING_EVERY_S
    expected_s: float | None = None  # how long the load is expected to take, when it is known
    engine: str = ""  # the engine's name in plain words
    model_bytes: int | None = None  # the weights' size, when it is known


@dataclass(frozen=True, slots=True, kw_only=True)
class ServerReady(ProgressEvent):
    phase: Phase
    load_s: float


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestsDone(ProgressEvent):
    phase: Phase
    done: int
    total: int


@dataclass(frozen=True, slots=True, kw_only=True)
class WindowOpened(ProgressEvent):
    lead: int  # requests in the start-up wave before the window


@dataclass(frozen=True, slots=True, kw_only=True)
class Note(ProgressEvent):
    text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RunWritten(ProgressEvent):
    run_dir: Path


Sink = Callable[[ProgressEvent], None]
LABELS: dict[Phase, str] = {"measure": "measuring"}  # the word RequestsDone lines start with


def _line(event: ProgressEvent) -> str | None:
    if isinstance(event, PhaseStarted):
        return event.detail or None
    if isinstance(event, ServerLoading):
        if not event.waited_s:
            return _loading(event) + (f", {about(event.expected_s)}" if event.expected_s else "")
        return f"still loading... {event.waited_s:.0f} s"
    if isinstance(event, ServerReady):
        return f"ready after {event.load_s:.0f} s"
    if isinstance(event, RequestsDone):
        marks = {event.total * quarter // 4: quarter * 25 for quarter in (1, 2, 3, 4)}
        if event.done not in marks:
            return None
        label = LABELS.get(event.phase, event.phase)
        return f"{label}: {event.done}/{event.total} requests ({marks[event.done]}%)"
    if isinstance(event, Note):
        return event.text
    if isinstance(event, RunWritten):
        return f"done: {event.run_dir}"
    return None


def clock(seconds: float) -> str:
    """``0:42``, ``12:05``, ``1:02:03``."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def about(seconds: float) -> str:
    return f"about {seconds:.0f} s" if seconds < 60 else f"about {math.ceil(seconds / 60)} min"


def _loading(event: ServerLoading) -> str:
    size = f" ({event.model_bytes / 1e9:.1f} GB)" if event.model_bytes else ""
    starting = f"Starting {event.engine} and loading" if event.engine else "Loading"
    return f"{starting} the model{size}"


PHASE_WORDS: dict[Phase, str] = {
    "preflight": "Checking the machine and the model",
    "compare": "Comparing the answers",
    "diagnose": "Working out what limits the speed",
    "write": "Writing the report",
}
SHOW_AFTER_S = 2.0  # a wait shorter than this shows nothing
QUIET_LINE_EVERY_S = 30.0  # without a terminal: a line when nothing was printed for this long
TICK_S = 0.1
BAR_CELLS = 16
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
ASCII_SPINNER = "|/-\\"
HIDE_CURSOR, SHOW_CURSOR, CLEAR_LINE = "\x1b[?25l", "\x1b[?25h", "\r\x1b[2K"


@dataclass(slots=True)
class Wait:
    text: str
    began: float = field(default_factory=time.monotonic)
    expected_s: float | None = None
    done: int | None = None
    total: int | None = None


def wait_for(event: ProgressEvent, now: Wait | None) -> Wait | None:
    """The wait ``event`` leaves the command in: a new one, ``now`` updated, or None."""
    if isinstance(event, PhaseStarted):
        text = event.detail or PHASE_WORDS.get(event.phase)
        return Wait(text[:1].upper() + text[1:]) if text else None
    if isinstance(event, ServerLoading):
        if event.waited_s and now is not None:
            return now
        return Wait(_loading(event), expected_s=event.expected_s)
    if isinstance(event, RequestsDone):
        if event.done >= event.total:
            return None
        if now is None or now.total is None:
            now = Wait("Sending requests", began=now.began if now else time.monotonic())
        now.done, now.total = event.done, event.total
        return now
    if isinstance(event, ServerReady):
        return Wait("Getting ready to send requests")
    if isinstance(event, RunWritten):
        return None
    return now


def _bar(done: int, total: int, unicode: bool) -> str:
    filled = round(BAR_CELLS * done / max(total, 1))
    if unicode:
        return f"▕{'█' * filled}{'░' * (BAR_CELLS - filled)}▏ {done}/{total}"
    return f"[{'#' * filled}{'-' * (BAR_CELLS - filled)}] {done}/{total}"


def status_line(wait: Wait, now: float, frame: int, *, unicode: bool, color: bool,
                width: int) -> str:  # fmt: skip
    """One terminal line for ``wait``: spinner, words, bar, clock; never wider than ``width``."""
    spinner = SPINNER if unicode else ASCII_SPINNER
    timing = clock(now - wait.began)
    if wait.expected_s:
        timing += f" of {about(wait.expected_s)}"
    bar = f" {_bar(wait.done or 0, wait.total, unicode)}" if wait.total is not None else ""
    ellipsis = "…" if unicode else "..."
    room = width - 1 - len(bar) - len(timing) - 4 - len(ellipsis)
    text = wait.text if len(wait.text) <= room else wait.text[: max(room - 1, 0)].rstrip() + "…"
    shown = f"\x1b[2m{timing}\x1b[22m" if color else timing
    return f"{spinner[frame % len(spinner)]} {text}{ellipsis}{bar}  {shown}"


class Status:
    """The one live status of this process. On a terminal a daemon thread redraws the wait at
    10 frames a second; otherwise it prints a plain line when nothing was printed for 30 s."""

    def __init__(self, stream: TextIO, environ: Any = os.environ) -> None:
        self.stream = stream
        term = environ.get("TERM", "")
        self.spin = stream.isatty() and term != "dumb"
        self.color = self.spin and "NO_COLOR" not in environ
        self.unicode = (getattr(stream, "encoding", None) or "").lower().replace("-", "") == "utf8"
        self.wait: Wait | None = None
        self.lock = threading.RLock()
        self.drawn = False
        self.printed = time.monotonic()
        self.frame = 0
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, name="tensward-status", daemon=True)

    def take(self, event: ProgressEvent) -> None:
        with self.lock:
            text = None if isinstance(event, ServerLoading) and event.waited_s else _line(event)
            if text is not None and not self.spin:
                self.say(f"[{event.at_s:5.0f}s] {text}")
            elif text is not None and not isinstance(event, (ServerLoading, RequestsDone)):
                self.say(text[:1].upper() + text[1:])
            self.wait = wait_for(event, self.wait)
            if self.wait is None:
                self.clear()

    def say(self, text: str) -> None:
        with self.lock:
            self.clear()
            self.stream.write(text + "\n")
            self.stream.flush()
            self.printed = time.monotonic()

    def clear(self) -> None:
        with self.lock:
            if self.drawn:
                self.stream.write(CLEAR_LINE + SHOW_CURSOR)
                self.stream.flush()
                self.drawn = False

    def hush(self) -> None:
        """Stop showing the wait: a prompt or the command's own output follows."""
        with self.lock:
            self.clear()
            self.wait = None

    def tick(self, now: float) -> None:
        with self.lock:
            wait = self.wait
            if wait is None or now - wait.began < SHOW_AFTER_S:
                return
            if not self.spin:
                if now - self.printed >= QUIET_LINE_EVERY_S:
                    timing = clock(now - wait.began)
                    if wait.expected_s:
                        timing += f" of {about(wait.expected_s)}"
                    counts = f" ({wait.done} of {wait.total})" if wait.total is not None else ""
                    self.say(f"[{elapsed():5.0f}s] {wait.text}{counts}... {timing}")
                return
            width = shutil.get_terminal_size((80, 24)).columns
            line = status_line(wait, now, self.frame, unicode=self.unicode, color=self.color,
                               width=width)  # fmt: skip
            self.frame += 1
            self.stream.write(("" if self.drawn else HIDE_CURSOR) + CLEAR_LINE + line)
            self.stream.flush()
            self.drawn = True

    def _run(self) -> None:
        while not self.stopped.wait(TICK_S):
            try:
                self.tick(time.monotonic())
            except (OSError, ValueError):
                return

    def close(self) -> None:
        self.stopped.set()
        if self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(timeout=1)
        try:
            self.clear()
        except (OSError, ValueError):
            pass


class _Guarded:
    """A stream whose writes first take the status line away: the command's own output or a
    question follows, and the next event shows the next wait."""

    def __init__(self, stream: TextIO, status: Status) -> None:
        self._stream, self._status = stream, status

    def write(self, text: str) -> int:
        with self._status.lock:
            if text:
                self._status.hush()
            return self._stream.write(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


_LIVE: Status | None = None


@contextmanager
def live() -> Iterator[None]:
    """Show waits on standard error for this block (one block at a time)."""
    global _LIVE
    if _LIVE is not None:
        yield
        return
    status = Status(sys.stderr)
    out, err = sys.stdout, sys.stderr
    if status.spin:
        sys.stdout = cast(TextIO, _Guarded(out, status))
        sys.stderr = cast(TextIO, _Guarded(err, status))
    _LIVE = status
    atexit.register(status.close)
    status.thread.start()
    try:
        yield
    finally:
        status.close()
        atexit.unregister(status.close)
        sys.stdout, sys.stderr, _LIVE = out, err, None


def hush() -> None:
    """Take the status line away before a question."""
    if _LIVE is not None:
        _LIVE.hush()


def terminal_sink(event: ProgressEvent) -> None:
    """The terminal lines: one per event that has one, request counts at each quarter; inside
    :func:`live`, the wait each event starts as well."""
    if _LIVE is not None:
        _LIVE.take(event)
    elif (text := _line(event)) is not None:
        print(f"[{event.at_s:5.0f}s] {text}", file=sys.stderr, flush=True)


_SINK: ContextVar[Sink] = ContextVar("progress_sink", default=terminal_sink)


def emit(event: ProgressEvent) -> None:
    """Send ``event`` to the current sink."""
    _SINK.get()(event)


def current_sink() -> Sink:
    """Where events go now, so a caller can pass them on."""
    return _SINK.get()


@contextmanager
def sink(target: Sink) -> Iterator[None]:
    """Send this block's events to ``target`` (threads started with ``asyncio.to_thread``
    inherit it)."""
    token = _SINK.set(target)
    try:
        yield
    finally:
        _SINK.reset(token)
