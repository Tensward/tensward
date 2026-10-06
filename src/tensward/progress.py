"""What a long command is doing, as events. The default sink prints them to standard error
with the seconds since the command began, so the command's own output stays clean for pipes;
an app sets its own sink for a block of work."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Literal

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
            return "waiting for the model to load"
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


def terminal_sink(event: ProgressEvent) -> None:
    """The terminal lines: one per event that has one, request counts at each quarter."""
    if (text := _line(event)) is not None:
        print(f"[{event.at_s:5.0f}s] {text}", file=sys.stderr, flush=True)


_SINK: ContextVar[Sink] = ContextVar("progress_sink", default=terminal_sink)


def emit(event: ProgressEvent) -> None:
    """Send ``event`` to the current sink."""
    _SINK.get()(event)


@contextmanager
def sink(target: Sink) -> Iterator[None]:
    """Send this block's events to ``target`` (threads started with ``asyncio.to_thread``
    inherit it)."""
    token = _SINK.set(target)
    try:
        yield
    finally:
        _SINK.reset(token)
