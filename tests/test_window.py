"""The window plan and the fallbacks of the comparison with a step's parent."""

from __future__ import annotations

import dataclasses
import json
import shutil
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from tensward import window
from tensward.history import planned_window
from tensward.workload import ArrivalSpec

L4_RUN = Path(__file__).parent / "replays" / "l4-steps" / "20261014T164058Z-8255"


@pytest.mark.parametrize(
    ("rate", "clients", "configured", "predicted", "only", "requests", "seconds"),
    [
        (None, 16, 1032, None, None, 1032, None),  # no served run to take the rate from
        (3.37, 16, 200, None, None, 200, None),  # configured below the 60 s count
        (1.0, 64, 1032, None, None, 308, 180.0),  # 4 per client
        (3.37, 0, 1032, None, None, 203, 60.0),  # open loop: no wave
        (1.0, 16, 1032, None, 90.0, 130, 90.0),  # --window 90 although the rule wants longer
        (1.0, 16, 1032, None, None, 212, 180.0),  # a slow cell gets a longer window
        (0.2, 0, 1032, None, None, 130, 300.0),  # at least 130 requests
        (1.0, 16, 1032, 0.01, None, 212, 180.0),  # a 1% prediction buys no longer window
    ],
)
def test_the_plan(
    rate: float | None,
    clients: int,
    configured: int,
    predicted: float | None,
    only: float | None,
    requests: int,
    seconds: float | None,
) -> None:
    plan = window.plan(
        rate,
        clients=clients,
        configured=configured,
        metric="tpot_p95_ms",
        predicted=predicted,
        only=only,
    )
    assert (plan.requests, plan.seconds) == (requests, seconds) and plan.reason


def test_a_quality_risk_step_plans_enough_prompts_for_one_judge_call() -> None:
    kept = {"clients": 0, "configured": 1032, "metric": "tpot_p95_ms", "predicted": None}
    lossless = window.plan(0.2, **kept)
    risky = window.plan(0.2, **kept, quality_risk=True)
    assert (lossless.requests, risky.requests) == (130, 150)


def test_a_parent_with_failures_is_not_matched_and_the_fallback_says_so(tmp_path: Path) -> None:
    rows = [json.loads(line) for line in (L4_RUN / "requests.jsonl").read_text().splitlines()]
    for row in rows[::40]:
        row["outcome"] = "error"
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert window.compare_matched(L4_RUN, tmp_path, clients=16) == (
        "the parent had 2.5% failed requests"
    )
    step = window.figures(window.read_rows(L4_RUN) or [], clients=16, upto=236)
    parent = window.figures(window.read_rows(tmp_path) or [], clients=16)
    delta = window.compare_unmatched(step, parent, "tpot_p95_ms")
    assert delta is not None and not delta.matched


CHAT = Path(__file__).parent / "replays" / "l4-gemma-chat"
CHAT_FULL = {  # the full default run of each track
    "throughput": CHAT / "20261016T172654Z-ed76",
    "ttft": CHAT / "20261016T172703Z-bb04",
    "tpot": CHAT / "20261016T174845Z-d0a1",
}
CHAT_LOOP = SimpleNamespace(
    request_count=1013, arrival=ArrivalSpec(kind="closed_loop", concurrency=96)
)


def _rewritten(
    source: Path, into: Path, move: Callable[[int, int], int], upto: int | None = None
) -> Path:
    """A copy of the KV-bound run in ``source`` keeping its first ``upto`` requests, each time
    of a request moved by ``move(time, its dispatch)``."""
    into.mkdir(parents=True)
    shutil.copy(source / "metrics.json", into / "metrics.json")
    rows = [json.loads(line) for line in (source / "requests.jsonl").read_text().splitlines()]
    times = ("dispatch_ns", "first_content_ns", "terminal_ns")
    moved = [{**row, **{key: move(row[key], row["dispatch_ns"]) for key in times}}
             for row in rows[:upto]]  # fmt: skip
    (into / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in moved))
    return into


def test_a_kv_bound_run_plans_a_steady_window_under_the_throughput_goal_only(
    tmp_path: Path,
) -> None:
    plan = planned_window(CHAT_FULL["throughput"], CHAT_LOOP, goal="throughput")
    assert plan.steady is not None and plan.seconds is not None
    assert 75 < plan.steady["ramp_s"] < 80 and 150 < plan.steady["steady_s"] < 160
    assert 35 < plan.steady["tail_s"] < 42 and 265 < plan.seconds < 285
    assert 720 < plan.requests < 750
    rows = window.read_rows(CHAT_FULL["throughput"]) or []
    sent = [r for r in rows if r.index < plan.requests]
    start, last = min(r.dispatch_ns for r in sent), max(r.dispatch_ns for r in sent)
    selected = [
        r for r in sent
        if r.dispatch_ns >= start + plan.steady["ramp_s"] * 1e9 and (r.terminal_ns or last) <= last
    ]  # fmt: skip
    assert len(selected) >= 4 * 96
    for goal in ("ttft", "tpot"):  # KV at 92% with every client running and none waiting
        short = planned_window(CHAT_FULL[goal], CHAT_LOOP, goal=goal)
        assert short.seconds is not None and short.requests < 1013
        waited = tmp_path / goal
        shutil.copytree(CHAT_FULL[goal], waited)
        metrics = json.loads((waited / "metrics.json").read_text())
        (waited / "metrics.json").write_text(json.dumps({**metrics, "mean_waiting": 5.0}))
        full = planned_window(waited, CHAT_LOOP, goal=goal)
        assert (full.requests, full.seconds) == (1013, None) and f"the {goal} goal" in full.reason
    open_loop = SimpleNamespace(request_count=1013, arrival=ArrivalSpec(kind="open_loop",
                                                                       rate_rps=2.7))  # fmt: skip
    assert planned_window(CHAT_FULL["throughput"], open_loop, goal="throughput").seconds is None


@pytest.mark.parametrize("run", list(CHAT_FULL.values()), ids=list(CHAT_FULL))
def test_the_steady_part_of_a_planned_prefix_reads_like_its_full_run(run: Path) -> None:
    rows = window.read_rows(run) or []
    found = window.steady(rows, clients=96)
    assert found is not None
    deltas = window.bias(rows, found, clients=96)
    assert deltas["req_s"].verdict == deltas["tpot_p95_ms"].verdict == "within_noise"
    assert deltas["req_s"].band < 0.035 and abs(deltas["req_s"].change) < 0.02
    planned = planned_window(run, CHAT_LOOP, goal="throughput").requests
    whole = window.figures(rows, clients=96).values["req_s"] or 0.0
    prefix = window.figures(rows, clients=96, upto=planned).values["req_s"] or 0.0
    assert abs(prefix / whole - 1) < 0.03
    start = min(r.dispatch_ns for r in rows)
    first_minute = [r for r in rows if r.dispatch_ns <= start + 60e9]
    early = window.figures(first_minute, clients=96).values["req_s"] or 0.0
    assert early / whole - 1 < -deltas["req_s"].band


def test_a_parent_whose_rate_rises_late_plans_the_full_window(tmp_path: Path) -> None:
    rows = window.read_rows(CHAT_FULL["throughput"]) or []
    start, last = min(r.dispatch_ns for r in rows), max(r.dispatch_ns for r in rows)
    cut = start + (last - start) * 2 // 3
    rises = _rewritten(
        CHAT_FULL["throughput"], tmp_path / "rises", lambda at, sent: at - max(0, sent - cut) // 2
    )
    plan = planned_window(rises, CHAT_LOOP, goal="throughput")
    assert (plan.requests, plan.seconds) == (1013, None)
    assert plan.reason == (
        "run rises's steady part differs from its full window: requests per second −11% ±3%"
    )


def test_two_steady_steps_after_a_30_percent_gain_plan_from_their_own_parent(
    tmp_path: Path,
) -> None:
    source = CHAT_FULL["throughput"]
    first = planned_window(source, CHAT_LOOP, goal="throughput")
    assert first.steady is not None
    start = min(r.dispatch_ns for r in window.read_rows(source) or [])
    faster = _rewritten(
        source, tmp_path / "faster", lambda at, _: start + (at - start) * 10 // 13, first.requests
    )
    (faster / "window.json").write_text(json.dumps({"plan": dataclasses.asdict(first)}))
    gain = window.compare_steady(
        faster, source, clients=96, parent_steady=window.Steady(
            first.steady["ramp_s"], first.steady["steady_s"], first.steady["tail_s"]
        ),
    )  # fmt: skip
    assert not isinstance(gain, str)
    assert gain["req_s"].verdict == "better" and 0.25 < gain["req_s"].change < 0.35
    assert gain["tpot_p95_ms"].verdict == "better"
    second = planned_window(faster, CHAT_LOOP, goal="throughput")
    assert second.steady is not None and second.seconds is not None and first.seconds is not None
    assert second.steady["source"] == "faster" and second.steady["bias"] == first.steady["bias"]
    assert second.seconds < first.seconds and second.steady["steady_s"] == window.STEADY_MIN_S
    assert second.requests == pytest.approx(first.requests, rel=0.05)
    metrics = json.loads((faster / "metrics.json").read_text())
    calm = {**metrics, "diagnosis": {"findings": []}, "peak_kv_usage": 0.5, "preemptions": 0}
    (faster / "metrics.json").write_text(json.dumps(calm))
    after = planned_window(faster, CHAT_LOOP, goal="throughput")
    assert after.steady is None and after.seconds is not None and after.requests < 1013


RERUN = Path(__file__).parent / "replays" / "l4-gemma-rerun"
DEFAULT_TPOT, FP8_TPOT = RERUN / "20261017T005701Z-42ba", RERUN / "20261017T012916Z-f412"
RERUN_LOOP = SimpleNamespace(
    request_count=1010, arrival=ArrivalSpec(kind="closed_loop", concurrency=96)
)


def _first(source: Path, into: Path, upto: int) -> Path:
    """A copy of the run in ``source`` that sent only its first ``upto`` requests."""
    return _rewritten(source, into, lambda at, _: at, upto)


@pytest.mark.parametrize("goal", ["tpot", "ttft", "throughput"])
def test_a_first_run_sends_its_steps_count_under_the_latency_goals_and_its_children_at_least_that(
    goal: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    latency = goal != "throughput"
    count = {"tpot": 384, "ttft": 509, "throughput": 1013}[goal]  # ttft: 4 + 1.3 per client
    assert window.first(96, 1013, matched=latency, ramped=goal == "ttft").requests == 1013
    assert window.COUNT_MATCHED_FIRST is False
    released = planned_window(CHAT_FULL["throughput"], CHAT_LOOP, goal=goal, predicted=0.5)
    assert (released.requests, released.seconds) == {
        "tpot": (406, 78),
        "ttft": (406, 78),
        "throughput": (733, 273.0),
    }[goal]
    monkeypatch.setattr(window, "COUNT_MATCHED_FIRST", True)
    first = window.first(96, 1013, matched=latency, ramped=goal == "ttft")
    assert first.requests == count and first.seconds is None
    on = planned_window(CHAT_FULL["throughput"], CHAT_LOOP, goal=goal, predicted=0.5)
    monkeypatch.setattr(window, "COUNT_MATCHED_FIRST", False)
    own = planned_window(CHAT_FULL["throughput"], CHAT_LOOP, goal=goal, predicted=0.5)
    assert own == released
    assert on.requests == (max(count, own.requests) if latency else own.requests)
    assert window.first(96, 1013, matched=latency).requests == 1013


def test_a_slow_long_answer_cell_measures_150_requests_above_its_configured_104() -> None:
    why = "150 requests, above the configured 104: the answer check needs about 100 answered pairs"
    assert window.first(16, 104, matched=True) == window.Plan(150, None, why)
    shape = {"clients": 16, "configured": 104, "metric": "tpot_p95_ms", "predicted": None}
    assert window.plan(0.24, **shape, quality_risk=True) == window.Plan(150, None, why)
    assert window.plan(0.24, **shape).requests == 104


def test_a_384_request_reference_gets_its_band_from_its_own_blocks_wider_than_the_table(
    tmp_path: Path,
) -> None:
    reference = _first(DEFAULT_TPOT, tmp_path / "reference", 384)
    step = _first(FP8_TPOT, tmp_path / "step", 384)
    deltas = window.compare_matched(step, reference, clients=96)
    assert not isinstance(deltas, str)
    for metric in ("tpot_p95_ms", "ttft_p95_ms", "req_s"):
        assert deltas[metric].band > window._projected(metric, 384), metric
    found = window.waves(window.read_rows(reference) or [], clients=96)
    assert found is not None and found.blocks < window.MIN_WAVES


def test_a_longer_child_is_compared_on_the_prompts_its_reference_sent(tmp_path: Path) -> None:
    from tensward.context import AnalysisContext, active
    from tensward.history import against

    runs = tmp_path / "project" / "runs"
    _first(DEFAULT_TPOT, runs / "reference", 384)
    longer = _first(FP8_TPOT, runs / "step", 474)
    with active(AnalysisContext(project=tmp_path / "project", new_run=longer)):
        found = against("reference", longer)
    assert found.own is not None and found.own.requests == 474
    assert found.matched_words == "on the 384 prompts both runs sent"


def test_the_wave_period_of_a_closed_loop_is_about_one_median_request() -> None:
    rows = window.read_rows(DEFAULT_TPOT) or []
    median = statistics.median(e for r in rows if (e := window._e2e(r)) is not None) / 1e3
    period = window.wave_period(rows)
    assert period is not None and abs(period / median - 1) < 0.2


@pytest.mark.parametrize(
    ("goal", "counts"),
    [("tpot", {"42ba": (407, 79), "f412": (416, 65)}),
     ("ttft", {"42ba": (519, 120), "f412": (502, 90)})],
)  # fmt: skip
def test_the_children_of_full_tpot_runs_plan_on_the_table_over_two_waves_and_keep_their_bands(
    goal: str, counts: dict[str, tuple[int, int]]
) -> None:
    planned = {
        run.name[-4:]: planned_window(run, RERUN_LOOP, goal=goal)
        for run in (DEFAULT_TPOT, FP8_TPOT)
    }
    print({name: (plan.requests, plan.seconds) for name, plan in planned.items()})
    assert {name: (p.requests, p.seconds) for name, p in planned.items()} == counts
    for plan in planned.values():
        assert plan.noise_source == "table" and plan.wave_period_s is not None
        assert plan.seconds is not None and plan.seconds >= 2 * plan.wave_period_s
        assert plan.waves_band is not None and plan.waves_band["tpot_p95_ms"] is not None


def test_a_large_prompt_file_lifts_a_slow_cell_to_150_requests() -> None:
    shape = {"clients": 16, "configured": 104, "metric": "tpot_p95_ms", "predicted": None}
    assert window.first(16, 104, matched=True, prompts=10_001).requests == 150
    assert window.plan(0.24, **shape, quality_risk=True, prompts=10_001).requests == 150
    assert window.plan(0.24, **shape, quality_risk=True, prompts=120).requests == 120


def test_an_explicit_window_keeps_its_seconds_and_the_requests_that_fill_them() -> None:
    plan = planned_window(DEFAULT_TPOT, RERUN_LOOP, goal="tpot", only=60.0)
    assert (plan.requests, plan.seconds) == (356, 60)


def test_a_count_per_client_above_the_window_plans_the_seconds_it_takes() -> None:
    shape = {"clients": 96, "configured": 1000, "metric": "tpot_p95_ms", "predicted": None}
    plan = window.plan(1.0, **shape)
    assert (plan.requests, plan.seconds) == (384, 192)
    assert window.plan(1.0, **shape, only=60.0).seconds == 60


def test_runs_that_sent_different_prompts_are_compared_on_the_prompts_both_sent(
    tmp_path: Path,
) -> None:
    from tensward.context import AnalysisContext, active
    from tensward.runs import shared_prompts

    rows = L4_RUN.joinpath("requests.jsonl").read_text().splitlines()
    for name, count in (("short", 100), ("long", 150)):
        run = tmp_path / name
        run.mkdir()
        (run / "requests.jsonl").write_text("\n".join(rows[:count]) + "\n")
    ids = {json.loads(line)["prompt_id"] for line in rows[:100]}
    with active(AnalysisContext()):
        assert shared_prompts(tmp_path / "short", tmp_path / "long") == ids
        assert shared_prompts(tmp_path / "short", tmp_path / "short") is None


@pytest.mark.parametrize(
    ("count", "basis", "note"),
    [
        (384, "full_window", True),  # a goal that does not plan for the ramp
        (window.post_ramp_count(96), "post_ramp", False),  # 509: the ttft first run's count
    ],
)
def test_a_96_client_reference_reads_ttft_after_the_ramp_only_with_enough_requests_for_it(
    count: int, basis: str, note: bool, tmp_path: Path
) -> None:
    from registration_fixtures import REGISTRATION_CONFIG, write_config

    from tensward.context import AnalysisContext, active
    from tensward.history import against, window_record

    assert window.post_ramp_count(96) == 509
    workload = {**REGISTRATION_CONFIG["workload"], "request_count": 1010,
                "arrival": {"kind": "closed_loop", "concurrency": 96}}  # fmt: skip
    project = tmp_path / "project"
    project.mkdir()
    config = write_config(tmp_path / "serving.json", {**REGISTRATION_CONFIG, "workload": workload})
    (project / "project.json").write_text(json.dumps({"config_path": str(config)}))
    runs = project / "runs"
    for source in (DEFAULT_TPOT, FP8_TPOT):
        reference = _first(source, runs / f"reference-{source.name[-4:]}", count)
        child = _first(FP8_TPOT, runs / f"step-{source.name[-4:]}", count + 90)
        rows = window.read_rows(reference) or []
        ramp = window.steady(rows, clients=96)
        assert ramp is not None
        eligible = len(window._after_ramp(rows, ramp.ramp_s))
        assert (eligible >= 192) is (basis == "post_ramp"), (source.name, eligible)

        with active(AnalysisContext(project=project, new_run=child)):
            found = against(reference.name, child)

        assert found.why is None and found.upto == count and found.ttft_basis == basis
        assert ("full matched window" in found.matched_words) is note
        assert ("after the ramp" in found.shown("ttft_p95_ms")) is not note
        assert window_record(found, found.own)["matched"]["ttft_basis"] == basis
