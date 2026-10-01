"""The numbers outsiders check first, table-driven: percentiles, TTFT/TPOT, goodput, throughput
over a known window, and the trace's busy time and gaps."""

from __future__ import annotations

import pytest

from tensward.client import RequestRecord
from tensward.measurement import Peaks, _tpot_ms, _ttft_ms, nearest_rank, summarize
from tensward.slo import Slo
from tensward.trace import Event, _busy_time, _find_gaps

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


def run(records: list[RequestRecord], seconds: float = 10.0, slo: Slo = Slo(200.0, 50.0)):
    return summarize(
        records,
        seconds=seconds,
        slo=slo,
        request_prompt_tokens={r.request_id: 20 for r in records},
        peaks=Peaks(),
        preemptions=None,
        prefix_cache_hits=None,
        prefix_cache_queries=None,
        prompt_tokens=[],
        max_context_len=2048,
        too_long=(),
        quant_kernels=(),
        tool_calls=None,
        mean_running=None,
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
