"""The anatomy against the real Gemma 4 26B-A4B AWQ checkpoint's config and headers, Gemma 4
E4B's scalar tensors, and the dense layout the ceilings already model."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from registration_fixtures import SAFETENSORS_NAME, make_checkpoint, safetensors_from_tensors

from tensward.anatomy import AttentionGroup, build_anatomy, freed_bytes
from tensward.formats.hf_safetensors import HF_SAFETENSORS

GEMMA = Path(__file__).parent / "gemma4_26b_a4b"
GEMMA_E4B = Path(__file__).parent / "gemma4_e4b"
E4B_CHAT = Path(__file__).parent / "replays" / "l4-gemma-chat" / "model"


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
    assert parts.gathered_embedding == parts.mtp == anatomy.kv_shared_layers == 0
    assert anatomy.mixed_head_dims
    assert freed_bytes(anatomy) == {"towers": 1_145_588_832, "quantizable": 14_564_587_372}


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
    assert anatomy.components.gathered_embedding == anatomy.kv_shared_layers == 0
    del tensors["lm_head.weight"], config["tie_word_embeddings"]
    assert build_anatomy(config, None, tensors).tied_embeddings  # no output head: it is tied


def test_gemma4_e4b_per_layer_tables_are_gathered_not_unclassified() -> None:
    rows = json.loads(gzip.decompress((E4B_CHAT / "tensors.json.gz").read_bytes()))
    tensors = {name: (dtype, tuple(shape), size) for name, (dtype, shape, size) in rows.items()}
    anatomy = build_anatomy(json.loads((E4B_CHAT / "config.json").read_text()), None, tensors)

    parts = anatomy.components
    assert anatomy.unavailable == () and parts is not None
    assert parts.unclassified < 0.01 * sum(size for _, _, size in tensors.values())
    projection = 55_050_240 + 512  # per_layer_model_projection and its norm: dense text weights
    assert parts.gathered_embedding == 5_692_195_328 - projection
    assert (anatomy.kv_shared_layers, anatomy.mixed_head_dims) == (18, True)
    assert freed_bytes(anatomy) == {
        "towers": parts.vision + parts.audio,
        "quantizable": parts.text_dense,
    }


def test_a_vocabulary_sized_input_table_and_mtp_weights_get_their_own_classes() -> None:
    config = {"model_type": "llama", "hidden_size": 64, "num_hidden_layers": 1,
              "num_attention_heads": 4, "vocab_size": 100}  # fmt: skip
    tensors = {
        "model.embed_tokens.weight": ("BF16", (100, 64), 12800),
        "model.embed_extra.weight": ("BF16", (100, 32), 6400),
        "model.layers.0.self_attn.v_proj.weight": ("BF16", (64, 64), 8192),
        "mtp.layers.0.proj.weight": ("BF16", (64, 64), 8192),
    }

    parts = build_anatomy(config, None, tensors).components

    assert parts is not None and parts.unclassified == 0
    assert (parts.gathered_embedding, parts.mtp, parts.text_dense) == (6400, 8192, 8192)


def test_an_unrecognised_layout_has_no_component_figures() -> None:
    anatomy = build_anatomy({"model_type": "llama"}, None, {"weight": ("BF16", (1,), 2)})

    assert anatomy.components is None
    assert any("unclassified" in reason for reason in anatomy.unavailable)


def test_gemma4_e4b_scalar_tensors_are_read_and_counted(tmp_path: Path) -> None:
    """Gemma 4 E4B's encoders hold 928 BF16 scalars (shape ``[]``, one element each); the
    fixture keeps them and a few other tensors of the real header, by name and shape."""
    rows = json.loads(gzip.decompress((GEMMA_E4B / "tensors.json.gz").read_bytes()))
    model = make_checkpoint(tmp_path / "model")
    tensors = {name: (dtype, tuple(shape)) for name, (dtype, shape) in rows.items()}
    (model / SAFETENSORS_NAME).write_bytes(safetensors_from_tensors(tensors))

    entry, anatomy = HF_SAFETENSORS.register(
        model, engine_build="synthetic", engine_name="vllm", cache=tmp_path / "cache"
    )

    assert entry.metadata.weight_bytes == 2 * (928 + 32 + 128 + 64 + 2560 + 1 + 2560)
    assert anatomy.components is not None and anatomy.params is not None
    assert (anatomy.components.audio, anatomy.params.audio) == (2 * 640, 480 + 32 + 128)
    assert (anatomy.components.vision, anatomy.params.vision) == (2 * 512, 448 + 64)


def checkpoint_anatomy(name: str, weight_bits: int | None = None, **config_overrides: object):
    """A real checkpoint's config and safetensors header (no weights), by directory."""
    directory = Path(__file__).parent / name
    config = json.loads((directory / "config.json").read_text()) | config_overrides
    header = directory / "tensors.json.gz"
    rows = json.loads(gzip.decompress(header.read_bytes())) if header.exists() else {}
    tensors = {key: (dtype, tuple(shape), size) for key, (dtype, shape, size) in rows.items()}
    return build_anatomy(config, None, tensors, weight_bits)


def test_gpt_oss_experts_are_counted_from_their_packed_blocks() -> None:
    anatomy = checkpoint_anatomy("gpt_oss_20b", weight_bits=4)

    assert anatomy.unavailable == () and not anatomy.mla and not anatomy.linear_attention
    assert anatomy.params is not None and anatomy.components is not None
    matrices, biases = 2880 * (5760 + 2880), 5760 + 2880  # per expert; two 4-bit values a byte
    assert anatomy.params.routed_experts == 24 * 32 * (matrices + biases)
    assert anatomy.components.routed_experts < anatomy.params.routed_experts  # stored packed
    assert anatomy.moe is not None and anatomy.moe.experts == 32 and anatomy.moe.layers == 24
    assert set(anatomy.attention) == {
        AttentionGroup("sliding", 12, 8, 64, 128, False),
        AttentionGroup("full", 12, 8, 64, None, False),
    }


def test_a_latent_attention_cache_holds_one_vector_a_token_and_layer() -> None:
    anatomy = checkpoint_anatomy("glm47_flash")

    assert anatomy.mla and anatomy.unavailable == ()
    assert anatomy.kv_bytes(1, "auto") == (512 + 64) * 47 * 2
    assert anatomy.kv_bytes(1000, "fp8") == 1000 * (512 + 64) * 47


def test_a_model_vllm_decompresses_to_full_heads_is_priced_as_plain_attention() -> None:
    anatomy = checkpoint_anatomy("minicpm3_4b")

    assert not anatomy.mla
    assert anatomy.kv_bytes(1, "auto") == 2 * 40 * (64 + 32) * 62 * 2  # K and V at the key width


@pytest.mark.parametrize("extra", [{"index_topk": 2048}, {"compress_ratios": [4, 128]},
                                    {"layer_types": ["full_attention"] * 47}])  # fmt: skip
def test_a_latent_cache_with_more_than_the_one_vector_is_not_priced(extra: dict) -> None:
    anatomy = checkpoint_anatomy("glm47_flash", **extra)

    assert not anatomy.mla and anatomy.attention == () and anatomy.kv_bytes(1, "auto") is None
    assert any(reason.startswith("KV bytes not priced") for reason in anatomy.unavailable)


def test_an_int8_per_token_head_cache_is_priced_at_one_byte() -> None:
    anatomy = checkpoint_anatomy("gpt_oss_20b", weight_bits=4)

    for dtype in ("fp8_per_token_head", "int8_per_token_head"):
        assert anatomy.kv_bytes(4096, dtype) * 2 == anatomy.kv_bytes(4096, "auto")


def test_a_hybrid_keeps_a_state_a_sequence_and_predicts_no_kv_pool() -> None:
    anatomy = checkpoint_anatomy("granite4_h_tiny")

    assert anatomy.linear_attention and anatomy.attention == ()
    scan, window = 48 * 64 * 128, 3 * (48 * 64 + 2 * 1 * 128)
    assert anatomy.state_bytes_per_seq == 36 * 2 * (scan + window)  # 36 of 40 layers, bfloat16
    assert any("no KV pool" in reason for reason in anatomy.unavailable)
