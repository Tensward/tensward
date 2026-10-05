"""Device-derived bandwidth: the formula on the L4, and an unlisted GPU (no GPU needed)."""

from __future__ import annotations

import pytest

from tensward.anatomy import build_anatomy
from tensward.artifacts import ArtifactVariant
from tensward.ceilings import (
    Ceilings,
    Measured,
    compute_ceilings,
    render_ceilings,
)
from tensward.platforms.nvidia import derived_bandwidth_gbs

BF16 = ArtifactVariant(
    weight_precision="bf16", activation_dtype="bfloat16", quantization_method="none"
)
INT4 = ArtifactVariant(
    weight_precision="int4",
    activation_dtype="bfloat16",
    quantization_method="compressed-tensors",
    weight_bits=4,
    group_size=32,
)
DENSE_CONFIG = {
    "model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2,
    "num_attention_heads": 4, "intermediate_size": 128, "vocab_size": 100,
}  # fmt: skip


def dense_anatomy(scale: int = 1):
    """A two-layer dense model whose tensors are ``scale`` times the config's sizes."""
    tensors = {"model.embed_tokens.weight": ("BF16", (100, 64), 12800 * scale),
               "lm_head.weight": ("BF16", (100, 64), 12800 * scale)}  # fmt: skip
    for layer in range(2):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tensors[f"model.layers.{layer}.self_attn.{proj}.weight"] = (
                "BF16", (64, 64), 2 * 64 * 64 * scale)  # fmt: skip
        for proj in ("gate_proj", "up_proj", "down_proj"):
            tensors[f"model.layers.{layer}.mlp.{proj}.weight"] = (
                "BF16", (128, 64), 2 * 128 * 64 * scale)  # fmt: skip
    return build_anatomy(DENSE_CONFIG, None, tensors)


def test_l4_formula_matches_datasheet() -> None:
    # Nsight Compute on the L4: 6,251,000 kHz memory clock, 192-bit bus; datasheet 300 GB/s.
    assert derived_bandwidth_gbs(6_251_000, 192) == pytest.approx(300.0, rel=1e-3)


def test_unlisted_gpu_gets_decode_but_not_prefill() -> None:
    ceilings = compute_ceilings(
        anatomy=dense_anatomy(scale=10**9 // 200_000),
        variant=BF16,
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
    exceeded = Ceilings(exceeds_bound=("decode", "window"))
    assert _ceiling_metric(exceeded).endswith("(a share exceeded its bound: decode, window)")
    assert _ceiling_metric(Ceilings(unavailable="unavailable (no GPU)")).endswith(
        ": unavailable (no GPU)"
    )


def _l4_ceilings(measured: Measured) -> Ceilings:
    return compute_ceilings(
        anatomy=dense_anatomy(),
        variant=BF16,
        kv_cache_dtype="auto",
        measured=measured,
        gpus=(("NVIDIA L4", 23034),),
    )


def test_prefill_rate_counts_only_computed_tokens_but_context_counts_cached_ones() -> None:
    # 10 s window, 5 requests; 8,000 of the 10,000 prompt tokens came from the prefix cache.
    ceilings = _l4_ceilings(Measured(10.0, 5, 10_000, 2_000, 500, 2.0))

    assert ceilings.measured_prefill_tok_s == pytest.approx(200.0)
    assert ceilings.avg_context_tokens == pytest.approx(10_000 / 5 + 500 / 5 / 2)


def test_a_share_above_one_hundred_percent_is_withheld_and_named() -> None:
    # The L4's prefill ceiling is a few hundred thousand tok/s on this tiny model; claim far more.
    ceilings = _l4_ceilings(Measured(1.0, 5, 10**9, 10**9, 500, 2.0))

    assert ceilings.prefill_pct_of_ceiling is None and "prefill" in ceilings.exceeds_bound
    assert ceilings.ceiling_time_pct_of_window is None and "window" in ceilings.exceeds_bound
    text = "\n".join(render_ceilings(ceilings))
    assert (
        "exceeds the theoretical bound" in text
        and "prefill, share of its ceiling: not measured" in text
    )

    sane = _l4_ceilings(Measured(10.0, 5, 1000, 1000, 500, 2.0))
    assert sane.exceeds_bound == () and sane.prefill_pct_of_ceiling is not None
    assert "exceeds" not in "\n".join(render_ceilings(sane))


@pytest.mark.parametrize(
    ("prompt", "hits", "by_source", "computed"),
    [
        (1000, 400, None, 600),
        (1000, None, None, 1000),
        (1000, 0, None, 1000),
        (1000, 1500, None, 0),
        (None, 5, None, None),
        (208898, 206016, 4786, 4786),
    ],
)
def test_computed_prompt_tokens_prefer_the_engine_count(prompt, hits, by_source, computed) -> None:
    from tensward.analyse import _computed_prompt_tokens
    from tensward.engines.protocol import EngineSignals

    counters = EngineSignals(
        prompt_tokens=prompt, prefix_cache_hits=hits, prompt_tokens_computed=by_source
    )
    assert _computed_prompt_tokens(counters) == computed


def _ceilings_on(selected: tuple[str, ...] | None):
    return compute_ceilings(
        anatomy=dense_anatomy(scale=10**9 // 200_000),
        variant=BF16,
        kv_cache_dtype="auto",
        measured=Measured(10.0, 5, 1000, 1000, 500, 2.0),
        selected=selected,
        gpus=(("NVIDIA A10G", 23028), ("NVIDIA L4", 23034)),
        device_gbs=600.0,
    )


def test_ceilings_use_the_selected_device_of_a_multi_gpu_host() -> None:
    assert _ceilings_on(None).gpu == "NVIDIA A10G"  # no selection: device 0
    assert _ceilings_on(("1",)).gpu == "L4"
    both = _ceilings_on(("0", "1"))
    assert "multi-GPU not modelled" in (both.unavailable or "")
    assert "GPU 5 selected, but 2 detected" in (_ceilings_on(("5",)).unavailable or "")


def test_dense_ceilings_match_the_previous_formula() -> None:
    # 0.1.2: weight bytes = payload - embedding (untied); body params = layers x (attention
    # + 3 x hidden x intermediate); decode params add hidden x vocab.
    ceilings = compute_ceilings(
        anatomy=dense_anatomy(), variant=BF16, kv_cache_dtype="auto",
        measured=Measured(10.0, 5, 1000, 1000, 500, 2.0), gpus=(("NVIDIA L4", 23034),),
    )  # fmt: skip
    attention = 64 * 64 * 2 + 64 * 64 * 2
    body = 2 * (attention + 3 * 64 * 128)
    assert ceilings.weight_bytes_per_step == 2 * body + 12800  # body + output head, no gather
    assert ceilings.prefill_ceiling_tok_s == pytest.approx(121e12 / (2 * body))


def test_moe_decode_reads_only_the_routed_experts_and_never_the_vision_tower() -> None:
    from test_anatomy import gemma

    anatomy = gemma()
    ceilings = compute_ceilings(
        anatomy=anatomy, variant=INT4, kv_cache_dtype="auto",
        measured=Measured(10.0, 32, 32_000, 32_000, 3_200, 32.0), gpus=(("NVIDIA L4", 23034),),
    )  # fmt: skip
    parts = anatomy.components
    non_expert = parts.text_dense + parts.embedding  # tied: the head is the embedding
    one_expert = anatomy.moe.expert_bytes
    assert ceilings.weight_bytes_per_step == non_expert + 30 * 8 * one_expert  # about 4.0 GB
    assert ceilings.moe_experts == (128, 8)
    assert ceilings.experts_per_step == pytest.approx(128 * (1 - (1 - 8 / 128) ** 32))
    assert ceilings.decode_ceiling_tok_s == pytest.approx(
        32 * 300e9 / (non_expert + 30 * 8 * one_expert + 32 * anatomy.kv_bytes(1050, "auto")),
        rel=1e-6,
    )  # the fewest experts a step can read, whatever the batch
    assert ceilings.decode_ceiling_batch1_tok_s == pytest.approx(
        300e9 / (non_expert + 30 * 8 * one_expert + anatomy.kv_bytes(1050, "auto")), rel=1e-6
    )
    print(
        f"GEMMA decode bytes/step b1={ceilings.weight_bytes_per_step} "
        f"L4 b1 ceiling={ceilings.decode_ceiling_batch1_tok_s:.2f}"
    )
    text = "\n".join(render_ceilings(ceilings))
    assert "8 of 128 experts per token" in text and "uniform routing" in text
