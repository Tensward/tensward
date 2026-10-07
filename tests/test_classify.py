"""The bottleneck rules: logic cases on hand-built measurements, hard cases on recorded runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensward.classify import classify, measurement_from_metrics
from tensward.engines.vllm import VLLM
from tensward.measure import acceptance_length, coverage, queue_share, signal_increase
from tensward.measurement import Measurement
from tensward.playbook import Situation, WorkloadFacts
from tensward.report.build import diagnosis_section
from tensward.report.markdown import sections_markdown
from tensward.settings import EngineSignals, Settings
from tensward.suggest import applicable

CALM = dict(queue_share=0.02, peak_kv_usage=0.3, preemptions=0.0, peak_running=8.0,
            peak_waiting=0.0, ttft_p50_ms=100.0, tpot_p50_ms=20.0, tpot_p95_ms=24.0)  # fmt: skip
QUEUED = dict(queue_share=0.6, peak_waiting=4.0, peak_in_flight=32)
SNAPSHOTS = json.loads((Path(__file__).parent / "calibration_snapshots.json").read_text())


def run(succeeded: int = 100, **signals: object) -> Measurement:
    return Measurement(succeeded, 0, **{**CALM, **signals})  # type: ignore[arg-type]


@pytest.mark.parametrize("row", SNAPSHOTS, ids=[row["id"] for row in SNAPSHOTS])
def test_a_recorded_run_is_named_as_it_was_when_calibrated(row: dict) -> None:
    measurement = measurement_from_metrics(row["metrics"])
    settings = Settings(max_concurrent_requests=row["cap"])
    assert classify(measurement, settings, None).primary == row["named"]


@pytest.mark.parametrize(
    ("measurement", "primary", "confidence"),
    [
        (run(), None, None),
        (run(**QUEUED), "queueing", "high"),
        (run(**{**QUEUED, "queue_share": 0.3}), "queueing", "likely"),
        (run(**QUEUED, preemptions=3.0, peak_kv_usage=0.99), "kv_capacity", "likely"),
        (run(preemptions=1.0), None, None),  # one preemption is not KV pressure
        (
            run(**{**QUEUED, "queue_share": 0.3}, ttft_p50_ms=700.0),
            "prefill",
            "possible",
        ),  # queueing warning
        (
            run(
                **{**QUEUED, "queue_share": 0.994},
                preemptions=3.0,
                peak_kv_usage=0.99,
                ttft_p50_ms=20000.0,
                tpot_p50_ms=22.6,
            ),
            "kv_capacity",
            "likely",
        ),  # TTFT is queue time the KV cache caused, not prefill
        (run(spec_acceptance_length=1.05), "speculation", "likely"),
        (run(spec_acceptance_length=2.8, spec_coverage=0.06), "speculation", "likely"),
        (run(spec_acceptance_length=1.0, spec_coverage=0.0), "speculation", "likely"),
        (
            run(queue_share=None),
            None,
            None,
        ),  # no time split: queueing cannot tell, prefill still runs
        (run(10, **QUEUED), "queueing", "possible"),  # 10 requests are too few for more
        (Measurement(0, 10, preemptions=5.0, peak_kv_usage=0.99), None, None),  # none succeeded
    ],
)
def test_each_class_is_named_only_past_its_threshold(
    measurement: Measurement, primary: str | None, confidence: str | None
) -> None:
    diagnosis = classify(measurement, Settings(), None)
    assert (diagnosis.primary, diagnosis.confidence) == (primary, confidence)
    assert ("- evidence:" in sections_markdown([diagnosis_section(diagnosis, None)])) == (
        primary is not None
    )


def test_high_needs_a_calibrated_critical_crossing_and_queueing_says_where_it_queued() -> None:
    assert classify(run(**QUEUED), Settings(), None).confidence == "high"
    assert classify(run(**{**QUEUED, "queue_share": 0.3}), Settings(), None).confidence == "likely"
    at_cap = classify(run(**QUEUED), Settings(max_concurrent_requests=16), None)
    evidence = at_cap.findings[0].evidence
    assert "32 requests were in flight against a concurrency cap of 16" in evidence
    within = classify(run(**QUEUED), Settings(max_concurrent_requests=64), None)
    assert within.primary == "queueing"  # still queueing, but the cap did not cause it
    assert "within the cap of 64, so they waited to be admitted" in within.findings[0].evidence

    diagnosis = classify(run(**QUEUED, tpot_p95_ms=70.0), Settings(), "differ")
    assert diagnosis.secondary == ("prefill_stalls_decode",)
    unknown = {f.bottleneck: f.evidence for f in diagnosis.findings if f.state == "cant_tell"}
    assert "--counters" in unknown["gpu_compute"] and "--trace" in unknown["host_overhead"]
    assert {gate.bottleneck: gate.state for gate in diagnosis.gates} == {
        "quality": "critical",
        "fit": "clear",
    }


def test_answers_within_the_current_setups_noise_do_not_trip_the_quality_gate() -> None:
    gates = classify(run(**QUEUED), Settings(), "within_noise").gates
    assert {gate.bottleneck: gate.state for gate in gates}["quality"] == "clear"


def test_a_recorded_run_replays_whether_or_not_it_was_traced() -> None:
    base = {**CALM, "succeeded": 100, "failed": 0, "queue_share": 0.6, "peak_waiting": 4.0,
            "peak_in_flight": 32, "trace": None, "startup_wave": None, "ceilings": None,
            "too_long": [], "quant_kernels": [], "image": None}  # fmt: skip
    traced = {**base, "trace": {"status": "ok", "idle_share": 0.3, "analysis": None}}
    assert classify(measurement_from_metrics(base), Settings(), None).primary == "queueing"
    assert classify(measurement_from_metrics(traced), Settings(), None).secondary == (
        "host_overhead",
    )


def test_a_healthy_run_lists_calibrated_classes_apart_from_uncalibrated_ones() -> None:
    text = sections_markdown([diagnosis_section(classify(run(), Settings(), None), None)])
    crossed = next(line for line in text.splitlines() if line.startswith("- not crossed:"))
    uncalibrated = next(line for line in text.splitlines() if "(uncalibrated" in line)
    assert "queueing before scheduling" in crossed
    assert "queueing before scheduling" not in uncalibrated


def test_an_engine_that_reports_nothing_never_crosses_and_never_raises() -> None:
    increase = signal_increase(EngineSignals(), EngineSignals())
    blind = Measurement(
        100, 0, ttft_p50_ms=50.0, ttft_p95_ms=80.0, tpot_p50_ms=10.0, tpot_p95_ms=12.0,
        preemptions=increase.preemptions, queue_share=queue_share(increase),
        spec_acceptance_length=acceptance_length(increase), spec_coverage=coverage(increase),
    )  # fmt: skip
    diagnosis = classify(blind, Settings(), None)
    assert {f.state for f in diagnosis.findings} <= {"cant_tell", "clear"}
    situation = Situation(
        measurement=blind,
        facts=WorkloadFacts(prompts=("hello",), output_tokens=64, context_limit=4096),
        settings=Settings(),
        diagnosis=diagnosis,
    )
    applicable(VLLM.playbook(), situation, allow_quality_changes=True, near=True)


@pytest.mark.parametrize(
    ("changes", "primary"),
    [
        ({}, "kv_capacity"),  # v034a: 1 free block of 9, each request held 4
        ({"hybrid_cache": False}, "queueing"),  # the same numbers on a plain cache: as in 0.3.5
        ({"peak_kv_usage": 0.45}, "queueing"),  # 5 free blocks hold another request
        ({"peak_waiting": 0.0}, "queueing"),  # no waiting request sampled: the rule needs one
        ({"kv_blocks": None}, "queueing"),  # an engine that does not report its blocks
    ],
)
def test_a_hybrid_pool_that_cannot_admit_a_request_is_full(changes: dict, primary: str) -> None:
    fields = dict(
        hybrid_cache=True, kv_blocks=10.0, peak_kv_usage=8 / 9, peak_running=2.0,
        peak_waiting=30.0, queue_share=0.99, peak_in_flight=32, ttft_p50_ms=60000.0,
        tpot_p50_ms=90.0, tpot_p95_ms=94.0,
    )  # fmt: skip
    diagnosis = classify(
        run(160, **{**fields, **changes}), Settings(max_concurrent_requests=16), None
    )
    assert diagnosis.primary == primary
