"""The bottleneck rules, one row per class, on hand-built measurements."""

from __future__ import annotations

import dataclasses

import pytest

from tensward import classify as classify_module
from tensward.classify import classify
from tensward.engines.protocol import Settings
from tensward.measurement import Measurement
from tensward.report import render_diagnosis
from tensward.trace import TraceSummary

CALM = dict(queue_share=0.02, peak_kv_usage=0.3, preemptions=0.0, peak_running=8.0,
            peak_waiting=0.0, ttft_p50_ms=100.0, tpot_p50_ms=20.0, tpot_p95_ms=24.0)  # fmt: skip
QUEUED = dict(queue_share=0.6, peak_waiting=4.0, peak_in_flight=32)


def run(succeeded: int = 100, **signals: object) -> Measurement:
    return Measurement(succeeded, 0, **{**CALM, **signals})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("measurement", "primary", "confidence"),
    [
        (run(), None, None),
        (run(**QUEUED), "queueing", "likely"),  # critical, but no threshold is calibrated yet
        (run(**{**QUEUED, "queue_share": 0.3}), "queueing", "possible"),
        (run(**QUEUED, preemptions=3.0, peak_kv_usage=0.99), "kv_capacity", "likely"),
        (run(preemptions=1.0), None, None),  # one preemption is not KV pressure
        (run(ttft_p50_ms=600.0), "prefill", "possible"),  # 30x TPOT
        (run(**QUEUED, ttft_p50_ms=5000.0), "queueing", "likely"),  # queueing, not prefill
        (
            run(**{**QUEUED, "queue_share": 0.3}, ttft_p50_ms=700.0),
            "prefill",
            "possible",
        ),  # queueing warning
        (run(tpot_p95_ms=70.0), "prefill_stalls_decode", "likely"),  # 3.5x p50
        (run(trace=TraceSummary("ok", idle_share=0.3)), "host_overhead", "likely"),
        (run(spec_acceptance_length=1.05), "speculation", "likely"),
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
    assert ("- evidence:" in render_diagnosis(diagnosis)) == (primary is not None)


def test_high_needs_a_calibrated_critical_crossing_and_queueing_says_where_it_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibrated = dataclasses.replace(classify_module.QUEUE_SHARE, calibrated=True)
    monkeypatch.setattr(classify_module, "QUEUE_SHARE", calibrated)
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
