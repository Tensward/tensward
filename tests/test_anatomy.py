"""The anatomy against the real Gemma 4 26B-A4B AWQ checkpoint's config and headers, plus the
dense layout the ceilings already model."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from tensward.anatomy import AttentionGroup, build_anatomy

GEMMA = Path(__file__).parent / "gemma4_26b_a4b"


def gemma(**text_overrides: object):
    config = json.loads((GEMMA / "config.json").read_text())
    config["text_config"].update(text_overrides)
    processor = json.loads((GEMMA / "processor_config.json").read_text())
    rows = json.loads(gzip.decompress((GEMMA / "tensors.json.gz").read_bytes()))
    tensors = {name: (dtype, tuple(shape), size) for name, (dtype, shape, size) in rows.items()}
    return build_anatomy(config, processor, tensors, weight_bits=4)


def test_gemma4_components_moe_attention_and_modalities() -> None:
    anatomy = gemma()

    assert anatomy.unavailable == () and anatomy.validated
    parts = anatomy.components
    assert parts is not None and parts.unclassified == 0
    assert parts.routed_experts == pytest.approx(12.85e9, rel=0.01)
    assert parts.vision == pytest.approx(1.15e9, rel=0.02)
    assert parts.embedding == pytest.approx(1.476e9, rel=0.01) and parts.lm_head == 0
    assert anatomy.tied_embeddings
    assert parts.text_dense == pytest.approx(1.72e9, rel=0.02)  # attention + dense MLP + router
    assert anatomy.moe is not None
    assert (anatomy.moe.experts, anatomy.moe.experts_per_token, anatomy.moe.layers) == (128, 8, 30)
    assert anatomy.moe.experts_per_token * anatomy.moe.expert_bytes * 30 == pytest.approx(
        0.80e9, rel=0.02
    )
    assert set(anatomy.attention) == {
        AttentionGroup("sliding", 25, 8, 256, 1024, False),
        AttentionGroup("full", 5, 2, 512, None, True),
    }
    assert anatomy.modalities == ("image", "text", "video")  # audio_config is null
    assert anatomy.vision is not None and anatomy.vision.max_tokens_per_image == 280
    assert anatomy.precisions["vision"] == ("F16",)
    params = anatomy.params
    assert params is not None
    active = params.text_dense + params.embedding + 8 * params.routed_experts / 128
    assert active == pytest.approx(3.82e9, rel=0.01)  # "A4B": about 4B active parameters
    assert params.routed_experts == pytest.approx(22.84e9, rel=0.01)


def test_gemma4_kv_bytes_match_vllms_admission_figure() -> None:
    anatomy = gemma()
    sliding = 25 * 8 * 256 * 2 * 2  # layers x heads x dim x (K and V) x bf16
    full = 5 * 2 * 512 * 2 * 2  # vLLM stores K and V even where V is K
    assert anatomy.kv_bytes(100, "auto") == 100 * (sliding + full)
    # vLLM 0.30 on the L4: 3588 blocks x 5 layers x 128 KiB pages hold 1.2334 requests of 32768.
    per_request = anatomy.kv_bytes(32768, "auto", in_flight_tokens=2 * 2496)
    assert 3588 * 5 * 131072 / per_request == pytest.approx(1.2334, rel=0.01)


def test_declared_experts_that_are_not_on_disk_make_moe_unavailable() -> None:
    anatomy = gemma(num_experts=256)

    assert anatomy.moe is None
    assert any("expert" in reason for reason in anatomy.unavailable)
    assert anatomy.components is not None  # the byte split itself is still known


def test_a_dense_llama_layout_matches_the_ceilings_kv_formula() -> None:
    config = {
        "model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 128,
        "vocab_size": 100, "tie_word_embeddings": False,
    }  # fmt: skip
    tensors = {
        "model.embed_tokens.weight": ("BF16", (100, 64), 12800),
        "lm_head.weight": ("BF16", (100, 64), 12800),
        "model.layers.0.self_attn.v_proj.weight": ("BF16", (32, 64), 4096),
        "model.layers.1.self_attn.v_proj.weight": ("BF16", (32, 64), 4096),
        "model.norm.weight": ("BF16", (64,), 128),
    }

    anatomy = build_anatomy(config, None, tensors)

    assert anatomy.attention == (AttentionGroup("full", 2, 2, 16, None, False),)
    assert anatomy.kv_bytes(10, "auto") == 10 * 2 * 2 * 2 * 16 * 2
    assert anatomy.moe is None and anatomy.vision is None and anatomy.modalities == ("text",)
    assert anatomy.components is not None and anatomy.components.lm_head == 12800
    del tensors["lm_head.weight"], config["tie_word_embeddings"]
    assert build_anatomy(config, None, tensors).tied_embeddings  # no output head: it is tied


def test_an_unrecognised_layout_has_no_component_figures() -> None:
    anatomy = build_anatomy({"model_type": "llama"}, None, {"weight": ("BF16", (1,), 2)})

    assert anatomy.components is None
    assert any("unclassified" in reason for reason in anatomy.unavailable)
