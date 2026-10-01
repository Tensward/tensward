"""What a run measured: per-request client timings and polled engine signals, reduced to one
:class:`Measurement`. ``None`` means not measured, never a guess."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

from .ceilings import Ceilings
from .client import RequestRecord
from .counters import CountersSummary
from .engines.protocol import EngineSignals, QuantKernel
from .slo import Slo
from .toolcalls import ToolCallStats
from .trace import TraceSummary


@dataclass(frozen=True, slots=True)
class Measurement:
    """What one server launch measured. ``None`` means not measured, never a guess."""

    succeeded: int
    failed: int
    request_throughput: float | None = None  # requests per second
    output_throughput: float | None = None  # generated tokens per second
    total_throughput: float | None = None  # prompt plus generated tokens per second
    goodput: float | None = None  # requests per second that succeeded and met the SLO
    slo_attainment: float | None = None  # fraction of all requests that succeeded and met it
    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None
    tpot_p50_ms: float | None = None
    tpot_p95_ms: float | None = None
    e2e_p95_ms: float | None = None
    peak_kv_usage: float | None = None  # engine signals, polled from its metrics
    peak_running: float | None = None
    peak_waiting: float | None = None
    preemptions: float | None = None
    prefix_cache_hits: float | None = None
    prefix_cache_hit_rate: float | None = None  # hits over lookups, in prompt tokens
    max_prompt_tokens: int | None = None  # longest prompt, as tokenized by the server
    max_context_len: int | None = None
    too_long: tuple[str, ...] = ()  # ids of prompts whose tokens plus output exceed max_context_len
    quant_kernels: tuple[QuantKernel, ...] = ()  # from the server log; empty means not detected
    tool_calls: ToolCallStats | None = None  # None when no request offered tools
    seconds: float | None = None  # wall time of the workload run
    mean_running: float | None = None  # average of the polled running-requests gauge
    ceilings: Ceilings | None = None  # Level 0: theoretical upper bounds for this GPU
    trace: TraceSummary | None = None  # Level 1: only with --trace, from a separate launch
    counters: CountersSummary | None = None  # Level 2: only with --counters, more launches
    image: str | None = None  # the container image that ran, if it was a container


@dataclass(slots=True)
class Peaks:
    """The highest engine signals seen while the workload ran."""

    kv_usage: float | None = None
    running: float | None = None
    waiting: float | None = None
    running_total: float = 0.0
    running_samples: int = 0

    def observe(self, signals: EngineSignals) -> None:
        self.kv_usage = _larger(self.kv_usage, signals.kv_usage)
        self.running = _larger(self.running, signals.running)
        self.waiting = _larger(self.waiting, signals.waiting)
        if signals.running is not None:
            self.running_total += signals.running
            self.running_samples += 1

    @property
    def mean_running(self) -> float | None:
        return self.running_total / self.running_samples if self.running_samples else None


def _larger(current: float | None, seen: float | None) -> float | None:
    if seen is None:
        return current
    return seen if current is None else max(current, seen)


def nearest_rank(values: Sequence[float], q: float) -> float:
    """The ``ceil(q * n)``-th smallest of ``values``, with the rank computed exactly.

    ``values`` must be non-empty and ``q`` inside ``(0, 1]``.
    """
    rank = -(-Fraction(repr(q)) * len(values) // 1)
    return sorted(values)[rank - 1]


def _quantile(values: Sequence[float], q: float) -> float | None:
    return nearest_rank(values, q) if values else None


def _ttft_ms(record: RequestRecord) -> float | None:
    if record.first_content_ns is None or record.dispatch_ns is None:
        return None
    return (record.first_content_ns - record.dispatch_ns) / 1e6


def _tpot_ms(record: RequestRecord) -> float | None:
    """Milliseconds per output token after the first; None for a one-token answer."""
    tokens = record.committed_output_tokens
    if record.first_content_ns is None or record.terminal_ns is None or not tokens or tokens < 2:
        return None
    return (record.terminal_ns - record.first_content_ns) / 1e6 / (tokens - 1)


def summarize(
    records: Sequence[RequestRecord],
    *,
    seconds: float,
    slo: Slo,
    request_prompt_tokens: Mapping[str, int | None],
    peaks: Peaks,
    preemptions: float | None,
    prefix_cache_hits: float | None,
    prefix_cache_queries: float | None,
    prompt_tokens: Sequence[int],
    max_context_len: int,
    too_long: tuple[str, ...],
    quant_kernels: tuple[QuantKernel, ...],
    tool_calls: ToolCallStats | None,
    mean_running: float | None,
) -> Measurement:
    """Client-side timings of the successful requests plus the polled engine signals."""
    successes = [record for record in records if record.outcome == "success"]
    ttft = [value for record in successes if (value := _ttft_ms(record)) is not None]
    tpot = [value for record in successes if (value := _tpot_ms(record)) is not None]
    e2e = [
        (record.terminal_ns - record.dispatch_ns) / 1e6
        for record in successes
        if record.dispatch_ns is not None and record.terminal_ns is not None
    ]
    tokens = [
        record.committed_output_tokens
        for record in successes
        if record.committed_output_tokens is not None
    ]
    prompts = [request_prompt_tokens.get(record.request_id) for record in successes]
    total = None if None in prompts or not tokens else sum(tokens) + sum(prompts)  # type: ignore[arg-type]
    good = sum(1 for record in successes if slo.met(_ttft_ms(record), _tpot_ms(record)))
    measured = seconds > 0
    return Measurement(
        succeeded=len(successes),
        failed=len(records) - len(successes),
        request_throughput=len(successes) / seconds if successes and measured else None,
        output_throughput=sum(tokens) / seconds if tokens and measured else None,
        total_throughput=total / seconds if total and measured else None,
        goodput=good / seconds if measured else None,
        slo_attainment=good / len(records) if records else None,
        ttft_p50_ms=_quantile(ttft, 0.5),
        ttft_p95_ms=_quantile(ttft, 0.95),
        tpot_p50_ms=_quantile(tpot, 0.5),
        tpot_p95_ms=_quantile(tpot, 0.95),
        e2e_p95_ms=_quantile(e2e, 0.95),
        peak_kv_usage=peaks.kv_usage,
        peak_running=peaks.running,
        peak_waiting=peaks.waiting,
        preemptions=preemptions,
        max_prompt_tokens=max(prompt_tokens, default=None),
        prefix_cache_hits=prefix_cache_hits,
        prefix_cache_hit_rate=(
            prefix_cache_hits / prefix_cache_queries
            if prefix_cache_hits is not None and prefix_cache_queries
            else None
        ),
        max_context_len=max_context_len,
        too_long=too_long,
        quant_kernels=quant_kernels,
        tool_calls=tool_calls,
        seconds=seconds,
        mean_running=mean_running,
    )
