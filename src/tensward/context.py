"""What one analysis is for and what it saw along the way: the goal and the declared TTFT
and TPOT limits, the GPU sampler and the launch's timings. The analysis functions keep
their signatures, so the command sets this for the block of work it runs, as
:func:`tensward.progress.sink` does for progress events."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal

from . import progress
from .gpu_state import GpuSampler, GpuState
from .kept import Kept
from .progress import PhaseStarted, ProgressEvent, RequestsDone, ServerReady
from .rejected import Rejection

if TYPE_CHECKING:
    from .anatomy import ModelAnatomy
    from .answer_options import AnswerOptions
    from .answers_after import AnswersAfter
    from .facts import Facts
    from .matcher import Matches

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
    project: Path | None = None  # the project directory (None outside the CLI)
    runs_before: frozenset[str] | None = None  # run directory names present before analyse()
    new_run: Path | None = None  # this run's directory when known up front (replays)
    kept: Kept | None = None
    answer_options: AnswerOptions | None = None  # None: not read, or the read failed
    json_answers: frozenset[str] = frozenset()  # prompt ids whose answers must be JSON
    # (step budget, tokens) of matching cold starts, oldest first
    cold_pools: tuple[tuple[int | None, float], ...] = ()
    max_req_s_loss: float | None = None  # the declared floor, as a share (0.2 for 20%)
    answers_after: AnswersAfter | None = None  # set while suggesting; the CLI writes quality.json
    # reads answer_options at the first launch, so every refusal comes before it
    read_answer_options: (
        Callable[[], tuple[AnswerOptions | None, str | None, float | None]] | None
    ) = None
    answer_options_unknown: str | None = None  # why answer_options is None after the read
    gpu_capability: float | None = None  # read with answer_options, also when that read fails
    # a recorded run whose report is written again: its history is the runs before it
    reported_run: str | None = None
    rejected: tuple[Rejection, ...] = ()  # the changes the user rejected (``tensward reject``)
    # rejections for a transport failure of the answer check: their change is offered once more
    retried: tuple[Rejection, ...] = ()
    # set while suggesting: by prompt index, its token count and block hashes, and the block size
    prompt_blocks: Mapping[int, tuple[int, tuple[str, ...]]] = field(default_factory=dict)
    kv_block_tokens: float | None = None
    window: str = "auto"  # --window: "auto", "full" or seconds as text
    # window.Plan as a dict, set by the CLI before analyse()
    window_plan: Mapping[str, Any] | None = None
    # what the report computed, for window.json
    window_record: dict[str, Any] = field(default_factory=dict)
    # the checkpoint's quantizable linear weights, from its anatomy
    linear_weight_bytes: int | None = None
    adopt_within_noise: bool = False  # --adopt-within-noise, or TENSWARD_ADOPT_WITHIN_NOISE=1
    tpot_p95_limit_ms: float | None = None  # declared
    anatomy: ModelAnatomy | None = None  # the checkpoint's anatomy, set beside linear_weight_bytes
    startup_guard: Mapping[str, Any] | None = None  # guard.Armed as a dict, set before analyse()
    window_extra: dict[str, Any] = field(default_factory=dict)  # keys merged into window.json
    kb_facts: Facts | None = None  # set by suggest.suggestions; written as facts.json by the CLI
    # set by suggest.suggestions; read by the report and the row writer
    kb_matches: Matches | None = None
    # called by observe() after its own handling
    observers: list[Callable[[ProgressEvent], None]] = field(default_factory=list)

    def observe(self, event: ProgressEvent) -> None:
        if isinstance(event, PhaseStarted) and event.phase == "launch" and self.launched_s is None:
            began = progress.elapsed()
            if self.read_answer_options is not None:
                read = self.read_answer_options()
                self.answer_options, self.answer_options_unknown, self.gpu_capability = read
            self.launched_s = event.at_s + progress.elapsed() - began
        elif isinstance(event, ServerReady) and self.load_s is None:
            self.load_s = event.load_s
        elif (
            isinstance(event, RequestsDone)
            and event.phase == "measure"
            and event.done == event.total
        ):
            self.measured_s = event.at_s
        for observer in self.observers:
            observer(event)

    def this_run(self) -> Path | None:
        """This analysis's run directory: ``new_run``, else the one directory under the project's
        runs/ that was not there before it began; None when that is not exactly one."""
        if self.new_run is not None:
            return self.new_run
        runs = self.project / "runs" if self.project is not None else None
        if runs is None or self.runs_before is None or not runs.is_dir():
            return None
        found = [p for p in runs.iterdir() if p.is_dir() and p.name not in self.runs_before]
        return found[0] if len(found) == 1 else None

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
