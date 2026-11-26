"""The wording of the knowledge records behind an offer."""

from tensward.report.kb import why_line

PACK = {
    "record": "vllm/prefill/kv-sharing-fast-prefill@3",
    "layer": "pack",
    "pack": "2026.10.20.1",
    "fired_on": [
        {"fact": "anatomy.kv_shared_layers", "op": ">", "value": 0, "observed": 18},
        {"fact": "goal", "op": "==", "value": "ttft", "observed": "ttft"},
    ],
    "measured": {"setups": 1, "pairs": 2, "ttft": [-0.31, -0.22]},
    "direction_note": None,
}


def test_the_why_line_names_the_record_its_terms_and_what_was_measured() -> None:
    assert why_line(PACK) == (
        "Why: KB record kv-sharing-fast-prefill (pack 2026.10.20.1) fired because "
        "anatomy.kv_shared_layers = 18 > 0 and goal = ttft. "
        "Measured on 1 setup: TTFT \N{MINUS SIGN}22 to \N{MINUS SIGN}31%."
    )
    null = {"null": {"record": "vllm/x/y@1", "reason": "two launches inside the band"},
            "pack": "2026.10.20.1"}  # fmt: skip
    assert why_line(null) == (
        "Why: KB record y (pack 2026.10.20.1) measured no gain from this change: "
        "two launches inside the band."
    )


def test_a_busy_gpu_in_small_steps_is_named_not_called_unexplained():
    """VH's smoke baseline (vLLM 0.34.0, one H100, Qwen3.8-27B INT4, 16 agent clients)."""
    from types import SimpleNamespace

    from tensward.context import AnalysisContext, active
    from tensward.history import _unexplained
    from tensward.measurement import Measurement

    measurement = Measurement(
        103, 0, peak_in_flight=16, peak_running=16.0, peak_waiting=0.0, step_ms=21.398
    )
    situation = SimpleNamespace(
        measurement=measurement,
        diagnosis=SimpleNamespace(primary=None, secondary=()),
        fired=lambda name: False,
    )
    facts = {"census.busy_share": 0.9338, "census.fullness_at_most": 0.1918}
    with active(AnalysisContext(engine="vllm", kb_facts=facts)):
        said = _unexplained(situation, "throughput", True)
    assert said.startswith(
        "the GPU was busy 93% of the time with all 16 requests running and none waiting, in "
        "small steps (kernels at most 19% full, about 21 ms a step)"
    )
    assert said.endswith("or a smaller or lower-precision weights path")
    with active(AnalysisContext(engine="vllm", kb_facts={**facts, "census.busy_share": 0.6})):
        assert _unexplained(situation, "throughput", True).startswith("nothing measured says")
