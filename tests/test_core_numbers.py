"""The numbers outsiders check first, table-driven: percentiles, TTFT/TPOT, goodput, throughput
over a known window, and the trace's busy time and gaps."""

from __future__ import annotations

import pytest

from tensward.client import RequestRecord
from tensward.measure import frontend_cpu_cores
from tensward.measurement import (
    Peaks,
    Window,
    _tpot_ms,
    _ttft_ms,
    avg_context_tokens,
    client_batch,
    measurement_window,
    nearest_rank,
    summarize,
)
from tensward.profiling.trace import Event, _busy_time, _find_gaps
from tensward.report.compare import _change, _delta
from tensward.slo import Slo

MS = 1_000_000


@pytest.mark.parametrize(
    ("values", "q", "expected"),
    [
        ([7], 0.5, 7),
        ([7], 0.95, 7),
        ([2, 1], 0.5, 1),  # ceil(0.5 * 2) = 1st smallest
        ([2, 1], 0.95, 2),
        ([3, 1, 2], 0.5, 2),
        ([4, 3, 2, 1], 0.5, 2),
        ([4, 3, 2, 1], 0.75, 3),
        (list(range(1, 21)), 0.95, 19),  # ceil(19.0) = 19, not 20: no float round-up
        (list(range(1, 101)), 0.95, 95),
        (list(range(1, 101)), 0.99, 99),
        (list(range(1, 101)), 1.0, 100),
        (list(range(1, 11)), 0.1, 1),
        ([5, 5, 5], 0.95, 5),
    ],
)
def test_nearest_rank_known_vectors(values: list[int], q: float, expected: int) -> None:
    assert nearest_rank(values, q) == expected


def record(
    *,
    dispatch: int = 0,
    first: int | None = 100,
    terminal: int = 1100,
    tokens: int | None = 11,
    outcome: str = "success",
    name: str = "r:0",
) -> RequestRecord:
    return RequestRecord(
        request_id=name,
        scheduled_ns=dispatch * MS,
        dispatch_ns=dispatch * MS,
        first_content_ns=None if first is None else first * MS,
        terminal_ns=terminal * MS,
        outcome=outcome,  # type: ignore[arg-type]
        committed_output_tokens=tokens,
        prompt_index=0,
    )


@pytest.mark.parametrize(
    ("rec", "ttft", "tpot"),
    [
        (record(), 100.0, 100.0),  # (1100 - 100) ms over 10 inter-token gaps
        (record(dispatch=50, first=70, terminal=90, tokens=3), 20.0, 10.0),
        (record(tokens=1), 100.0, None),  # one token: there is no inter-token time
        (record(tokens=None), 100.0, None),
        (record(first=None, tokens=0), None, None),  # never produced content
    ],
)
def test_ttft_and_tpot_per_request(
    rec: RequestRecord, ttft: float | None, tpot: float | None
) -> None:
    assert _ttft_ms(rec) == ttft
    assert _tpot_ms(rec) == tpot


def whole(seconds: float) -> Window:
    return Window("whole_run", 0, round(seconds * 1e9), seconds)


def run(
    records: list[RequestRecord],
    seconds: float = 10.0,
    slo: Slo = Slo(200.0, 50.0),
    window: Window | None = None,
):
    return summarize(
        records,
        window=window or whole(seconds),
        slo=slo,
        request_prompt_tokens={r.request_id: 20 for r in records},
        request_images={},
        peaks=Peaks(),
        preemptions=None,
        prefix_cache_hits=None,
        prefix_cache_queries=None,
        prompt_tokens=[],
        max_context_len=2048,
        too_long=(),
        quant_kernels=(),
        tool_calls=None,
        structured_answers=None,
    )


def test_throughput_totals_over_a_known_window() -> None:
    records = [record(name=f"r:{i}", tokens=10) for i in range(4)]
    records.append(record(name="r:4", outcome="error", tokens=None, first=None))

    summary = run(records, seconds=2.0)

    assert (summary.succeeded, summary.failed) == (4, 1)
    assert summary.request_throughput == 2.0  # 4 successes in 2 s
    assert summary.output_throughput == 20.0  # 40 generated tokens
    assert summary.total_throughput == 60.0  # plus 4 x 20 prompt tokens
    assert run(records, seconds=0).request_throughput is None  # no window, no rate


@pytest.mark.parametrize(
    ("window", "requests", "tokens", "total"),
    [
        (Window("whole_run", 0, 2100 * MS, 2.1), 4.0, 22.0, 102.0),  # every request counts whole
        (Window("steady", 600 * MS, 1600 * MS, 1.0), 3.0, 12.0, 72.0),  # half of A and B in
        (Window("steady", 1600 * MS, 2100 * MS, 0.5), 0.5, 5.0, 5.0),  # half of B; C, D ended out
        (Window("steady", 3000 * MS, 4000 * MS, 1.0), None, None, None),  # nothing streams
    ],
)
def test_a_window_credits_each_request_by_the_share_of_its_streaming_span_inside(
    window: Window, requests: float | None, tokens: float | None, total: float | None
) -> None:
    records = [
        record(name="r:0", dispatch=0, first=100, terminal=1100, tokens=10),
        record(name="r:1", dispatch=1000, first=1100, terminal=2100, tokens=10),
        record(name="r:2", dispatch=1400, first=None, terminal=1500, tokens=1),  # immediate end
        record(name="r:3", dispatch=1100, first=1200, terminal=1200, tokens=1),  # zero-length span
    ]

    summary = run(records, window=window)

    def rate(count: float | None) -> float | None:
        return None if count is None else pytest.approx(count / window.seconds)

    assert summary.request_throughput == rate(requests)
    assert summary.output_throughput == rate(tokens)
    assert summary.total_throughput == rate(total)
    assert summary.ttft_p95_ms == 100.0  # latencies cover every request, whatever the window


def test_average_context_weights_each_request_by_its_share_of_the_window() -> None:
    window = Window("steady", 0, 1000, 1e-6)
    whole = RequestRecord("a", 0, 0, 100, 900, "success", 50, 0)  # inside: share 1
    half = RequestRecord("b", 0, 0, 500, 1500, "success", 100, 0)  # half inside
    failed = RequestRecord("c", 0, 0, None, 800, "error", None, 0)
    unknown = RequestRecord("d", 0, 0, 100, 900, "success", 10, 0)  # no prompt count
    prompts = {"a": 100, "b": 300, "c": 999, "d": None}
    # (100 + 50/2) * 1 + (300 + 100/2) * 0.5, over 1.5
    assert avg_context_tokens([whole, half, failed, unknown], window, prompts) == 200.0
    assert avg_context_tokens([failed], window, prompts) is None


def test_client_batch_is_throughput_times_token_weighted_tpot() -> None:
    window = Window("steady", 0, 10 * 10**9, 10.0)
    a = RequestRecord("a", 0, 0, 1 * 10**9, 9 * 10**9, "success", 81, 0)  # 8 s, 80 tokens
    b = RequestRecord("b", 0, 0, 2 * 10**9, 6 * 10**9, "success", 41, 0)  # 4 s, 40 tokens
    one = RequestRecord("c", 0, 0, 2 * 10**9, 3 * 10**9, "success", 1, 0)  # one token: skipped
    assert client_batch([a, b, one], window, 30.0) == pytest.approx(3.0)
    assert client_batch([a], window, None) is None


@pytest.mark.parametrize(
    ("closed_ns", "first_dispatch_ns", "expected"),
    [
        (900, 300, Window("steady", 200, 900, 7e-7)),  # the wave was over at the last dispatch
        (None, 300, Window("whole_run", 300, 1000, 7e-7)),  # it outlasted the last dispatch
        (None, 200, Window("whole_run", 200, 1000, 8e-7)),  # no wave: request_count <= concurrency
    ],
)
def test_the_window_kind_follows_the_wave_not_the_load(
    closed_ns: int | None, first_dispatch_ns: int, expected: Window
) -> None:
    window = measurement_window(
        closed_ns=closed_ns, window_ns=200, first_dispatch_ns=first_dispatch_ns, end_ns=1000
    )

    assert window == expected


def test_goodput_and_slo_attainment_count_failures_and_slow_requests_against_the_run() -> None:
    fast = dict(first=100, terminal=100 + 9 * 40, tokens=10)  # ttft 100 ms, tpot 40 ms
    records = [
        record(name="r:0", **fast),
        record(name="r:1", **fast),
        record(name="r:2", first=300, terminal=300 + 9 * 40, tokens=10),  # ttft 300 > 200
        record(name="r:3", first=100, terminal=100 + 9 * 60, tokens=10),  # tpot 60 > 50
        record(name="r:4", first=100, terminal=100, tokens=1),  # one token: only ttft counts
        record(name="r:5", outcome="timeout", tokens=None, first=None),
    ]

    summary = run(records, seconds=2.0)

    assert summary.goodput == 1.5  # r:0, r:1, r:4 in 2 s
    assert summary.slo_attainment == 0.5  # 3 of all 6 requests, the timeout included
    assert summary.ttft_p95_ms == 300.0 and summary.tpot_p95_ms == 60.0


def test_a_run_with_no_successes_reports_nothing_rather_than_zero_latency() -> None:
    summary = run([record(outcome="error", first=None, tokens=None)])

    assert summary.ttft_p50_ms is None and summary.output_throughput is None
    assert summary.goodput == 0.0 and summary.slo_attainment == 0.0


def event(start: float, end: float) -> Event:
    return Event(start, end, "k", "kernel", 0, 0, None)


@pytest.mark.parametrize(
    ("intervals", "busy"),
    [
        ([(0, 10)], 10),
        ([(0, 10), (20, 30)], 20),  # disjoint
        ([(0, 10), (5, 15)], 15),  # overlap
        ([(0, 100), (10, 20), (30, 40)], 100),  # nested
        ([(0, 10), (10, 20)], 20),  # touching
        ([(20, 30), (0, 10), (5, 25)], 30),  # unsorted, chained overlaps
        ([(0, 10), (0, 10)], 10),  # identical
        ([], 0),
    ],
)
def test_busy_time_is_the_union_of_the_intervals(
    intervals: list[tuple[float, float]], busy: float
) -> None:
    assert _busy_time(event(*i) for i in intervals) == busy


@pytest.mark.parametrize(
    ("intervals", "gaps"),
    [
        ([(0, 10), (100, 110)], [(10, 100)]),
        ([(0, 10), (60, 70)], [(10, 60)]),  # exactly the 50 us minimum counts
        ([(0, 10), (59.9, 70)], []),  # just under it does not
        ([(0, 10), (5, 20)], []),  # overlap
        (
            [(0, 100), (10, 20), (200, 210)],
            [(100, 200)],
        ),  # a nested event does not end the busy stretch
        ([(0, 10), (10, 20)], []),  # touching
        ([(200, 210), (0, 10), (5, 100)], [(100, 200)]),  # unsorted input
        ([(0, 10)], []),
    ],
)
def test_gaps_are_idle_stretches_of_at_least_fifty_microseconds(
    intervals: list[tuple[float, float]], gaps: list[tuple[float, float]]
) -> None:
    found = _find_gaps([event(*i) for i in intervals])

    assert [(g.start, g.end) for g in found] == gaps


@pytest.mark.parametrize(
    ("name", "before", "after", "expected"),
    [
        ("output_throughput", 100.0, 242.0, "100.0 → 242.0 tok/s (+142%)"),
        ("output_throughput", 100.0, 93.3, "100.0 → 93.3 tok/s (−7%, worse)"),
        ("output_throughput", 1400.0, 20.0, "1,400 → 20.0 tok/s (÷70, worse)"),
        ("ttft_p95_ms", 7000.0, 100.0, "7,000 → 100.0 ms (÷70)"),
    ],
)
def test_delta_shows_percent_and_large_drops_as_a_factor(
    name: str, before: float, after: float, expected: str
) -> None:
    unit = " tok/s" if name == "output_throughput" else " ms"
    assert _delta(name, before, after, unit) == expected


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [(100.0, 242.0, "+142%"), (100.0, 93.0, "−7%"), (16925.0, 70.2, "÷241"), (None, 5.0, "")],
)
def test_table_change_uses_the_same_factor_rule(
    before: float | None, after: float, expected: str
) -> None:
    assert _change(before, after) == expected


@pytest.mark.parametrize(
    ("cpu_seconds", "window_s", "servers", "cores"),
    [
        (10.0, 10.0, None, 1.0),
        (None, 10.0, None, None),
        (10.0, 10.0, 2, None),  # several API servers
        (10.0, 0.0, None, None),
    ],
)
def test_frontend_cpu_is_cores_over_the_window(
    cpu_seconds: float | None, window_s: float, servers: int | None, cores: float | None
) -> None:
    assert frontend_cpu_cores(cpu_seconds, window_s, servers) == pytest.approx(cores)
