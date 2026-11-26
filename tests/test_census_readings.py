import json
from dataclasses import replace
from pathlib import Path

import pytest

from tensward.census.readings import READINGS, ReadingContext, compute_readings
from tensward.census.record import CensusRecord, read_record, write_record
from tensward.census.snapshot import parse_snapshot, read_timeline, window
from tensward.facts import VOCABULARY, census_facts

FIXTURES = Path(__file__).parent / "census_snapshots"


def cut(name: str):
    folder = FIXTURES / name
    meta = json.loads((folder / "meta.json").read_text())
    before, after = (
        {
            int(pid): parse_snapshot(entry)
            for pid, entry in json.loads((folder / f"{tag}.json").read_text()).items()
        }
        for tag in ("before", "after")
    )
    timelines = {
        meta["pid"]: read_timeline(
            folder / "timeline.jsonl", meta["window_start_ns"], meta["window_end_ns"]
        )
    }
    win = window(
        before,
        after,
        requests=meta["requests"],
        start_ns=meta["window_start_ns"],
        end_ns=meta["window_end_ns"],
        timelines=timelines,
    )
    engine = "vllm" if meta["engine"] == "vllm" else "llamacpp"
    return meta, win, ReadingContext(engine, meta["gpu"], None, engine == "llamacpp")


@pytest.mark.parametrize(
    "name",
    [
        "llamacpp-b12246-h100-qwen3-8b-c12",
        "llamacpp-b12246-l4-qwen3-8b",
        "vllm-0.30.0-h100-qwen3-8b-c8",
        "multi-pid-6",
    ],
)
def test_busy_share_from_the_timeline_matches_the_research_harness(name):
    meta, win, ctx = cut(name)
    got = compute_readings(win, ctx)["census.busy_share"]
    assert abs(got - meta["research"]["busy_share"]) < 0.001


@pytest.mark.parametrize(
    ("name", "fact"),
    [
        ("llamacpp-b12246-h100-qwen3-8b-c12", "census.fullness"),
        ("vllm-0.30.0-h100-qwen3-8b-c8", "census.fullness_at_most"),
    ],
)
def test_fullness_matches_the_research_estimate_on_the_recorded_h100_arms(name, fact):
    meta, win, ctx = cut(name)
    got = compute_readings(win, ctx)[fact]
    assert abs(got * 100 - meta["research"]["union_pct"]) < 0.05


def test_readings_outside_their_validated_situation_stay_unknown():
    _, win, ctx = cut("vllm-0.30.0-h100-qwen3-8b-c8")
    assert {"census.fullness_at_most", "census.busy_share"} <= set(compute_readings(win, ctx))
    for other in (
        replace(ctx, gpu_model="NVIDIA L4"),
        replace(ctx, gpu_model="NVIDIA H100 PCIe"),
        replace(ctx, sm_count=114),
    ):
        assert "census.fullness_at_most" not in compute_readings(win, other)
    assert "census.kernel_ms_per_request" not in compute_readings(win, ctx)
    graphs_off = replace(ctx, graphs_off=True)
    assert "census.kernel_ms_per_request" in compute_readings(win, graphs_off)
    assert "census.fullness_at_most" in compute_readings(win, replace(ctx, sm_count=132))
    assert "census.fullness" not in compute_readings(win, ctx)

    _, one, ctx = cut("llamacpp-b12246-h100-qwen3-8b-c12")
    ctx = replace(ctx, graphs_off=True)
    alone = compute_readings(one, ctx)
    wanted = {"census.busy_share", "census.kernel_ms_per_request", "census.fullness"}
    assert wanted <= set(alone) and one.timeline
    both = replace(one, pids=(one.pids[0], one.pids[0] + 1))
    found = compute_readings(both, ctx)
    assert found["census.recording_processes"] == 2
    assert not wanted & set(found)


def test_stored_census_readings_are_run_facts(tmp_path):
    _, win, ctx = cut("llamacpp-b12246-h100-qwen3-8b-c12")
    readings = compute_readings(win, ctx)
    write_record(tmp_path, CensusRecord("ok", "", readings, requests=win.requests, wall_s=70.0))
    stored = census_facts(tmp_path)
    assert stored == readings and set(stored) <= set(VOCABULARY)
    assert stored["census.graphs_off"] is True
    write_record(tmp_path, CensusRecord("failed", "the start overran", readings))
    assert census_facts(tmp_path) == {}
    assert read_record(tmp_path).reason == "the start overran"
    assert census_facts(tmp_path / "nothing") == {} and read_record(tmp_path / "nothing") is None
    assert {r.spec.name for r in READINGS} <= set(VOCABULARY)
