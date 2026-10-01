"""Device-derived bandwidth: the formula on the L4, and an unlisted GPU (no GPU needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensward.artifacts import ArtifactVariant
from tensward.ceilings import (
    Ceilings,
    Measured,
    compute_ceilings,
    derived_bandwidth_gbs,
    render_ceilings,
)


def test_l4_formula_matches_datasheet() -> None:
    # Nsight Compute on the L4: 6,251,000 kHz memory clock, 192-bit bus; datasheet 300 GB/s.
    assert derived_bandwidth_gbs(6_251_000, 192) == pytest.approx(300.0, rel=1e-3)


def test_unlisted_gpu_gets_decode_but_not_prefill(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 128,
                "vocab_size": 100,
            }
        )
    )
    ceilings = compute_ceilings(
        model_dir=tmp_path,
        payload_bytes=10**9,
        variant=ArtifactVariant(
            weight_precision="bf16", activation_dtype="bfloat16", quantization_method="none"
        ),
        kv_cache_dtype="auto",
        measured=Measured(10.0, 5, 1000, 1000, 500, 2.0),
        gpus=(("NVIDIA A10G", 23028),),
        device_gbs=600.0,
    )
    assert ceilings.unavailable is None and ceilings.gpu == "NVIDIA A10G"
    assert ceilings.decode_ceiling_batch1_tok_s and ceilings.decode_ceiling_tok_s
    assert ceilings.prefill_ceiling_tok_s is None and ceilings.tensor_tflops is None
    text = "\n".join(render_ceilings(ceilings))
    assert "bandwidth: derived from device (memory clock x bus width)" in text
    assert "no published dense tensor rate for NVIDIA A10G" in text


def test_current_setup_line_falls_back_to_the_decode_share() -> None:
    from tensward.report import _ceiling_metric

    both = Ceilings(decode_pct_of_ceiling=27.8, ceiling_time_pct_of_window=57.0)
    assert _ceiling_metric(both) == "hardware ceiling reached 57%"
    assert _ceiling_metric(Ceilings(decode_pct_of_ceiling=27.8)) == "decode at 27.8% of its ceiling"
    assert _ceiling_metric(None) == "hardware ceiling reached not measured"


def _l4_ceilings(tmp_path: Path, measured: Measured) -> Ceilings:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 128,
                "vocab_size": 100,
            }
        )
    )
    return compute_ceilings(
        model_dir=tmp_path,
        payload_bytes=10**6,
        variant=ArtifactVariant(
            weight_precision="bf16", activation_dtype="bfloat16", quantization_method="none"
        ),
        kv_cache_dtype="auto",
        measured=measured,
        gpus=(("NVIDIA L4", 23034),),
    )


def test_prefill_rate_counts_only_computed_tokens_but_context_counts_cached_ones(
    tmp_path: Path,
) -> None:
    # 10 s window, 5 requests; 8,000 of the 10,000 prompt tokens came from the prefix cache.
    ceilings = _l4_ceilings(tmp_path, Measured(10.0, 5, 10_000, 2_000, 500, 2.0))

    assert ceilings.measured_prefill_tok_s == pytest.approx(200.0)
    assert ceilings.avg_context_tokens == pytest.approx(10_000 / 5 + 500 / 5 / 2)


def test_a_share_above_one_hundred_percent_is_withheld_and_named(tmp_path: Path) -> None:
    # The L4's prefill ceiling is a few hundred thousand tok/s on this tiny model; claim far more.
    ceilings = _l4_ceilings(tmp_path, Measured(1.0, 5, 10**9, 10**9, 500, 2.0))

    assert ceilings.prefill_pct_of_ceiling is None and "prefill" in ceilings.exceeds_bound
    assert ceilings.ceiling_time_pct_of_window is None and "window" in ceilings.exceeds_bound
    text = "\n".join(render_ceilings(ceilings))
    assert (
        "exceeds the theoretical bound" in text
        and "prefill, share of its ceiling: not measured" in text
    )

    sane = _l4_ceilings(tmp_path, Measured(10.0, 5, 1000, 1000, 500, 2.0))
    assert sane.exceeds_bound == () and sane.prefill_pct_of_ceiling is not None
    assert "exceeds" not in "\n".join(render_ceilings(sane))


@pytest.mark.parametrize(
    ("prompt", "hits", "computed"),
    [(1000, 400, 600), (1000, None, 1000), (1000, 0, 1000), (1000, 1500, 0), (None, 5, None)],
)
def test_computed_prompt_tokens_subtract_prefix_cache_hits(prompt, hits, computed) -> None:
    from tensward.analyse import _computed_prompt_tokens
    from tensward.engines.protocol import EngineSignals

    counters = EngineSignals(prompt_tokens=prompt, prefix_cache_hits=hits)
    assert _computed_prompt_tokens(counters) == computed


def _ceilings_on(tmp_path: Path, selected: tuple[str, ...] | None):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 128,
                "vocab_size": 100,
            }
        )
    )
    return compute_ceilings(
        model_dir=tmp_path,
        payload_bytes=10**9,
        variant=ArtifactVariant(
            weight_precision="bf16", activation_dtype="bfloat16", quantization_method="none"
        ),
        kv_cache_dtype="auto",
        measured=Measured(10.0, 5, 1000, 1000, 500, 2.0),
        selected=selected,
        gpus=(("NVIDIA A10G", 23028), ("NVIDIA L4", 23034)),
        device_gbs=600.0,
    )


def test_ceilings_use_the_selected_device_of_a_multi_gpu_host(tmp_path: Path) -> None:
    assert _ceilings_on(tmp_path, None).gpu == "NVIDIA A10G"  # no selection: device 0
    assert _ceilings_on(tmp_path, ("1",)).gpu == "L4"
    both = _ceilings_on(tmp_path, ("0", "1"))
    assert "multi-GPU not modelled" in (both.unavailable or "")
    assert "GPU 5 selected, but 2 detected" in (_ceilings_on(tmp_path, ("5",)).unavailable or "")
