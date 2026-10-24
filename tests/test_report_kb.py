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
