"""What held a run's throughput back, read from its time-averaged engine state: the measures
the diagnosis, the expected-effect estimates and their scoring share, the levels that judge
them, and the one place they are assembled for a run. Engine-neutral."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Sequence

if TYPE_CHECKING:
    from .measurement import Measurement, Window

Limit = Literal["demand", "cap", "kv", "step_budget", "prompt"]
LIMITS: tuple[Limit, ...] = ("demand", "cap", "kv", "step_budget", "prompt")
PROMPT_SHARE = 0.5  # prompt work at least this share of the engine's capacity names prompt work
FULL = 0.9  # a limit used this much, with requests waiting, binds
NEAR = 0.8  # below this share of its limit a resource is not near it
CAP_FULL = 0.9  # running at this share of the cap: the cap binds too (queueing not a symptom)
WAITING = 1.0  # time-averaged waiting requests from which requests queue
TOOL_SHARE = 0.2  # answers with a tool call from which the tool parser's hold shows
TTFT_GAP_MS = 500.0  # client over server mean TTFT that the tool parser's hold explains
NEAR_MARGIN = 0.05  # a near offer's least distance under its threshold


class Averages:
    """The mean of the values seen, None before the first."""

    __slots__ = ("total", "count")

    def __init__(self) -> None:
        self.total, self.count = 0.0, 0

    def observe(self, value: float | None) -> None:
        if value is not None:
            self.total += value
            self.count += 1

    @property
    def mean(self) -> float | None:
        return self.total / self.count if self.count else None


def mean_in_flight(spans: Sequence[tuple[int, int]], start_ns: int, end_ns: int) -> float | None:
    """The client's requests in flight, averaged over [start_ns, end_ns]."""
    if end_ns <= start_ns:
        return None
    inside = sum(max(0, min(end, end_ns) - max(begin, start_ns)) for begin, end in spans)
    return inside / (end_ns - start_ns)


def measures(
    *,
    seconds: float,
    iterations: float | None,
    prompt_computed: float | None,
    generated: float | None,
    capacity: float | None,
    spans: Sequence[tuple[int, int]],
    window: Window,
) -> dict[str, Any]:
    """The measures a run's engine counters give over the engine's own span: prompt-work share
    against the probed capacity (spec 2.1; 0.0 when no prompt token was computed), scheduled
    tokens and seconds per step, and the client's mean in flight."""
    rate = prompt_computed / seconds if prompt_computed is not None and seconds > 0 else None
    share = None
    if rate is not None and capacity:
        share = min(rate / capacity, 1.0)
    steps = iterations or None
    return {
        "prompt_work_share": share,
        "scheduled_tokens_per_step": (prompt_computed + generated) / steps
        if steps and prompt_computed is not None and generated is not None
        else None,
        "step_ms": seconds * 1e3 / steps if steps and seconds > 0 else None,
        "mean_in_flight": mean_in_flight(spans, window.start_ns, window.end_ns),
    }


def demand(m: Measurement) -> int | None:
    """The client's in-flight peak when known, else running plus waiting at their peaks."""
    if m.peak_in_flight is not None:
        return m.peak_in_flight
    if m.peak_running is None:
        return None
    return round(m.peak_running) + round(m.peak_waiting or 0)


@dataclass(frozen=True, slots=True, kw_only=True)
class State:
    """A run's time-averaged state against its limits. None: not measured."""

    running: float | None
    waiting: float | None
    kv: float | None
    in_flight: float | None
    cap: int | None
    budget: int | None
    scheduled: float | None
    prompt_share: float | None


def state_of(m: Measurement, cap: int | None) -> State:
    return State(
        running=m.mean_running, waiting=m.mean_waiting, kv=m.mean_kv_usage,
        in_flight=m.mean_in_flight, cap=cap, budget=m.step_budget_tokens,
        scheduled=m.scheduled_tokens_per_step, prompt_share=m.prompt_work_share,
    )  # fmt: skip


def prompt_bound(s: State) -> bool | None:
    """Prompt work took at least PROMPT_SHARE of the engine's capacity (spec 2.1)."""
    return None if s.prompt_share is None else s.prompt_share >= PROMPT_SHARE


def step_full(s: State) -> bool | None:
    if s.budget is None or s.scheduled is None or s.waiting is None:
        return None
    return s.waiting >= WAITING and s.scheduled >= FULL * s.budget


def cap_full(s: State) -> bool | None:
    """Running at CAP_FULL of the cap with requests waiting (spec 2.7)."""
    if s.cap is None or s.running is None or s.waiting is None:
        return None
    return s.waiting >= WAITING and s.running >= CAP_FULL * s.cap


def load_limited(s: State) -> bool | None:
    if (
        s.running is None or s.in_flight is None or s.waiting is None or s.kv is None
        or s.prompt_share is None or s.scheduled is None or s.budget is None
    ):  # fmt: skip
        return None
    return (
        s.in_flight > 0
        and s.running >= FULL * s.in_flight
        and s.waiting < WAITING
        and s.kv < NEAR
        and s.scheduled < NEAR * s.budget
        and s.prompt_share < PROMPT_SHARE
    )


def observed(s: State) -> frozenset[Limit]:
    """The limits holding the run back by its time-averaged state (spec 5.3); the diagnosis
    rules call the same functions (review concern 2)."""
    found: set[Limit] = set()
    if s.waiting is not None and s.waiting >= WAITING and s.kv is not None and s.kv >= FULL:
        found.add("kv")
    if step_full(s):
        found.add("step_budget")
    if prompt_bound(s):
        found.add("prompt")
    if cap_full(s):
        found.add("cap")
    if load_limited(s):
        found.add("demand")
    return frozenset(found)
