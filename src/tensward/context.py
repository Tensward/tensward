"""What one analysis is for and what it saw along the way: the goal and the declared TTFT
limit, the GPU sampler and the launch's timings. The analysis functions keep
their signatures, so the command sets this for the block of work it runs, as
:func:`tensward.progress.sink` does for progress events."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Literal

from . import progress
from .gpu_state import GpuSampler, GpuState
from .progress import PhaseStarted, ProgressEvent, RequestsDone, ServerReady

Goal = Literal["throughput", "ttft", "tpot"]
GOALS: tuple[Goal, ...] = ("throughput", "ttft", "tpot")


@dataclass(slots=True, kw_only=True)
class AnalysisContext:
    goal: Goal = "throughput"
    ttft_p95_limit_ms: float | None = None  # declared
    flags: tuple[str, ...] = ()  # the goal and limit options a suggested command repeats
    engine_args: tuple[str, ...] = ()  # the run's ``--engine-arg`` values
    cold_kv_tokens: float | None = None  # smallest cold-start KV pool of these settings
    sampler: GpuSampler | None = None
    launched_s: float | None = None  # progress seconds when the measurement launch began
    load_s: float | None = None  # how long that server took to load
    measured_s: float | None = None  # progress seconds when its last measured request ended

    def observe(self, event: ProgressEvent) -> None:
        if isinstance(event, PhaseStarted) and event.phase == "launch" and self.launched_s is None:
            self.launched_s = event.at_s
        elif isinstance(event, ServerReady) and self.load_s is None:
            self.load_s = event.load_s
        elif (
            isinstance(event, RequestsDone)
            and event.phase == "measure"
            and event.done == event.total
        ):
            self.measured_s = event.at_s

    @property
    def test_s(self) -> float | None:
        """Launch to the end of the measurement: what testing one change takes here."""
        if self.launched_s is None or self.measured_s is None:
            return None
        return self.measured_s - self.launched_s

    def gpu_state(self, start_ns: int | None, end_ns: int | None) -> GpuState | None:
        return None if self.sampler is None else self.sampler.state(start_ns, end_ns)


_CURRENT: ContextVar[AnalysisContext | None] = ContextVar("analysis", default=None)


def current() -> AnalysisContext:
    """The active context; outside ``active``, a fresh default (never a shared instance)."""
    return _CURRENT.get() or AnalysisContext()


@contextmanager
def active(context: AnalysisContext) -> Iterator[AnalysisContext]:
    """Run a block of work under ``context``: its sampler runs throughout, and its progress
    events are observed on their way to the sink that was current."""
    outer = progress.current_sink()

    def observe(event: ProgressEvent) -> None:
        context.observe(event)
        outer(event)

    token = _CURRENT.set(context)
    try:
        if context.sampler is not None:
            context.sampler.start()
        with progress.sink(observe):
            yield context
    finally:
        if context.sampler is not None:
            context.sampler.stop()
        _CURRENT.reset(token)
