"""What a run measured: per-request client timings and polled engine signals, reduced to one
:class:`Measurement`. ``None`` means not measured, never a guess."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Mapping, Sequence

from .ceilings import Ceilings
from .client import RequestRecord
from .counters import CountersSummary
from .engines.protocol import EngineSignals, QuantKernel
from .slo import Slo
from .toolcalls import ToolCallStats
from .trace import TraceSummary


@dataclass(frozen=True, slots=True)
class Window:
    """The span the throughput figures cover, in client monotonic nanoseconds. ``steady`` runs
    from the end of a closed loop's start-up wave to its last dispatch; ``whole_run`` runs from
    the first request to the end of the run."""

    kind: Literal["steady", "whole_run"]
    start_ns: int
    end_ns: int
    seconds: float


@dataclass(frozen=True, slots=True)
class GroupStats:
    """The client-side figures of one group of requests."""

    requests: int
    ttft_p50_ms: float | None
    ttft_p95_ms: float | None
    tpot_p95_ms: float | None
    prompt_tokens_mean: float | None  # as the server counted them, images included


@dataclass(frozen=True, slots=True)
class ImageSplit:
    """The requests whose prompt carries images next to those whose prompt does not. A group
    with no request is None."""

    with_images: GroupStats | None
    without_images: GroupStats | None
    images_per_request: float  # over the requests that carry images


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
    kv_capacity_tokens: float | None = None  # tokens the engine says its KV cache holds
    kv_max_concurrency: float | None = None  # full-length requests it says fit at once
    kv_capacity_estimate_tokens: float | None = None  # what Tensward estimated before the run
    max_prompt_tokens: int | None = None  # longest prompt, as tokenized by the server
    max_context_len: int | None = None
    too_long: tuple[str, ...] = ()  # ids of prompts whose tokens plus output exceed max_context_len
    quant_kernels: tuple[QuantKernel, ...] = ()  # from the server log; empty means not detected
    tool_calls: ToolCallStats | None = None  # None when no request offered tools
    seconds: float | None = None  # the measurement window's length, see ``window``
    mean_running: float | None = None  # average of the polled running-requests gauge
    ceilings: Ceilings | None = None  # Level 0: theoretical upper bounds for this GPU
    trace: TraceSummary | None = None  # Level 1: only with --trace, from a separate launch
    counters: CountersSummary | None = None  # Level 2: only with --counters, more launches
    image_split: ImageSplit | None = None  # None when no prompt has images
    image: str | None = None  # the container image that ran, if it was a container
    served_as_float16: bool = False  # the engine cast a bfloat16 checkpoint to float16
    checks: tuple[str, ...] = ()  # problems with the window or the engine's signals
    queue_share: float | None = None  # share of server-side TTFT spent queued, not computing
    spec_acceptance_length: float | None = None  # 1 + accepted tokens per draft
    spec_coverage: float | None = None  # accepted draft tokens over generated tokens
    peak_in_flight: int | None = None  # most requests the client had in flight at once
    startup_wave: GroupStats | None = None  # the closed-loop start-up wave, outside the window
    startup_wave_included: bool = False  # too few requests to measure the wave separately
    window: Window | None = None


@dataclass(slots=True)
class Peaks:
    """The highest engine signals seen while the workload ran."""

    kv_usage: float | None = None
    running: float | None = None
    waiting: float | None = None
    running_total: float = 0.0
    running_samples: int = 0
    generation: int = 0  # counts resets, so a scrape begun before one can be told apart

    def reset(self) -> None:
        self.kv_usage = self.running = self.waiting = None
        self.running_total = 0.0
        self.running_samples = 0
        self.generation += 1

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


def _timings(successes: Sequence[RequestRecord]) -> tuple[list[float], list[float]]:
    """The TTFT and TPOT, in milliseconds, of the requests that have them."""
    ttft = [value for record in successes if (value := _ttft_ms(record)) is not None]
    tpot = [value for record in successes if (value := _tpot_ms(record)) is not None]
    return ttft, tpot


def group_stats(
    records: Sequence[RequestRecord], request_prompt_tokens: Mapping[str, int | None]
) -> GroupStats | None:
    if not records:
        return None
    successes = [record for record in records if record.outcome == "success"]
    ttft, tpot = _timings(successes)
    prompts = [
        tokens
        for record in successes
        if (tokens := request_prompt_tokens.get(record.request_id)) is not None
    ]
    return GroupStats(
        requests=len(records),
        ttft_p50_ms=_quantile(ttft, 0.5),
        ttft_p95_ms=_quantile(ttft, 0.95),
        tpot_p95_ms=_quantile(tpot, 0.95),
        prompt_tokens_mean=sum(prompts) / len(prompts) if prompts else None,
    )


def _image_split(
    records: Sequence[RequestRecord],
    request_images: Mapping[str, int],
    request_prompt_tokens: Mapping[str, int | None],
) -> ImageSplit | None:
    carrying = [record for record in records if request_images.get(record.request_id)]
    if not carrying:
        return None
    others = [record for record in records if not request_images.get(record.request_id)]
    return ImageSplit(
        with_images=group_stats(carrying, request_prompt_tokens),
        without_images=group_stats(others, request_prompt_tokens),
        images_per_request=sum(request_images[record.request_id] for record in carrying)
        / len(carrying),
    )


def measurement_window(
    *, closed_ns: int | None, window_ns: int, first_dispatch_ns: int, end_ns: int
) -> Window:
    """The window a run is measured over. ``closed_ns`` is the last declared dispatch when the
    start-up wave had finished by then; it is None for a run with no wave and for one whose wave
    outlasted its dispatches, which are measured whole from their first request."""
    if closed_ns is not None:
        return Window("steady", window_ns, closed_ns, (closed_ns - window_ns) / 1e9)
    return Window("whole_run", first_dispatch_ns, end_ns, (end_ns - first_dispatch_ns) / 1e9)


def window_share(record: RequestRecord, window: Window) -> float:
    """How much of a request falls inside ``window``: the share of its streaming span from
    first content to the end that lies there. A request with no streaming span (an immediate
    end) counts whole when it ended inside. This assumes the request decoded at a constant rate,
    so its tokens are spread evenly over the span."""
    first, end = record.first_content_ns, record.terminal_ns
    if first is None or end <= first:
        return 1.0 if window.start_ns <= end <= window.end_ns else 0.0
    inside = min(end, window.end_ns) - max(first, window.start_ns)
    return max(0, inside) / (end - first)


def _prompt_share(record: RequestRecord, window: Window) -> float:
    """1 when the request's prompt was prefilled inside ``window``, else 0."""
    at = record.terminal_ns if record.first_content_ns is None else record.first_content_ns
    return 1.0 if window.start_ns <= at <= window.end_ns else 0.0


def summarize(
    records: Sequence[RequestRecord],
    *,
    window: Window,
    slo: Slo,
    request_prompt_tokens: Mapping[str, int | None],
    request_images: Mapping[str, int],
    peaks: Peaks,
    preemptions: float | None,
    prefix_cache_hits: float | None,
    prefix_cache_queries: float | None,
    prompt_tokens: Sequence[int],
    max_context_len: int,
    too_long: tuple[str, ...],
    quant_kernels: tuple[QuantKernel, ...],
    tool_calls: ToolCallStats | None,
) -> Measurement:
    """Client-side timings of the successful requests plus the polled engine signals.

    Latencies cover every request. Rates are over ``window``: each request counts in
    proportion to ``window_share``, and its prompt tokens at the moment it began streaming. A
    window that no request streamed in has no rates."""
    successes = [record for record in records if record.outcome == "success"]
    ttft, tpot = _timings(successes)
    e2e = [
        (record.terminal_ns - record.dispatch_ns) / 1e6
        for record in successes
        if record.dispatch_ns is not None and record.terminal_ns is not None
    ]
    shares = {record.request_id: window_share(record, window) for record in successes}
    counted = [
        (record, record.committed_output_tokens)
        for record in successes
        if record.committed_output_tokens is not None
    ]
    tokens = sum(count * shares[record.request_id] for record, count in counted)
    total = None
    prompts = [request_prompt_tokens.get(record.request_id) for record in successes]
    if counted and None not in prompts:
        total = tokens + sum(
            prompt * _prompt_share(record, window)
            for record, prompt in zip(successes, prompts, strict=True)
            if prompt is not None
        )
    met = [record for record in successes if slo.met(_ttft_ms(record), _tpot_ms(record))]
    good = sum(shares[record.request_id] for record in met)
    seconds = window.seconds
    rated = seconds > 0 and (not successes or any(shares.values()))
    return Measurement(
        succeeded=len(successes),
        failed=len(records) - len(successes),
        request_throughput=sum(shares.values()) / seconds if successes and rated else None,
        output_throughput=tokens / seconds if counted and rated else None,
        total_throughput=total / seconds if total and rated else None,
        goodput=good / seconds if rated else None,
        slo_attainment=len(met) / len(records) if records else None,
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
        window=window,
        mean_running=peaks.mean_running,
        image_split=_image_split(records, request_images, request_prompt_tokens),
        peak_in_flight=_peak_in_flight(records),
    )


def _peak_in_flight(records: Sequence[RequestRecord]) -> int | None:
    """The most requests in flight at once, from when each was offered to when it ended."""
    events = sorted(
        [(record.dispatch_ns, 1) for record in records]
        + [(record.terminal_ns, -1) for record in records]
    )  # at equal times an end (-1) sorts before a start, so touching requests do not overlap
    peak = current = 0
    for _, step in events:
        current += step
        peak = max(peak, current)
    return peak or None
