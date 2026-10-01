"""Synthetic input builders for the CLI and analyse tests.

Every byte is fabricated inside a pytest temporary directory: ``make_checkpoint`` writes a
tiny parseable safetensors file, never a runnable model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_ENGINE_BUILD = "synthetic-build-1"
DEFAULT_CONTEXT_LIMIT = 4096
SAFETENSORS_NAME = "model.safetensors"
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
GEMMA4 = Path(__file__).parent / "gemma4_26b_a4b"

# The exact registration configuration document the registration path accepts, written as
# the Python equivalent of the documented JSON (false -> False).
REGISTRATION_CONFIG: dict[str, Any] = {
    "schema_version": "1",
    "engine_build": DEFAULT_ENGINE_BUILD,
    "case": {
        "weight_precision": "bf16",
        "activation_dtype": "bfloat16",
        "max_model_len": 2048,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 2048,
        "kv_cache_dtype": "auto",
        "gpu_memory_utilization": 0.8,
        "prefix_cache": False,
    },
    "workload": {
        "mode": "same_text",
        "output_tokens": 32,
        "request_count": 10,
        "request_timeout_s": 60.0,
        "arrival": {"kind": "closed_loop", "concurrency": 1},
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
    },
    "rounds": 3,
    "warmup_requests": 3,
    "objective": "highest_throughput",
}

# The two synthetic plain-text workload rows: one with a reference and a label, one without.
REGISTRATION_PROMPTS: tuple[dict[str, Any], ...] = (
    {
        "id": "first",
        "prompt": "synthetic private prompt",
        "reference": "synthetic reference answer",
        "labels": ["sample"],
    },
    {"id": "second", "prompt": "another synthetic prompt"},
)


def safetensors_bytes(
    *,
    dtype: str = "BF16",
    shape: tuple[int, ...] = (1,),
    data: bytes = b"\x00\x00",
    header_override: Mapping[str, Any] | None = None,
) -> bytes:
    """Return one tiny, well-formed safetensors file whose only tensor is ``weight``."""
    header_payload: dict[str, Any] = {
        "weight": {"dtype": dtype, "shape": list(shape), "data_offsets": [0, len(data)]}
    }
    if header_override is not None:
        header_payload = dict(header_override)
    header = json.dumps(header_payload, separators=(",", ":")).encode("utf-8")
    header += b" " * (-len(header) % 8)
    return len(header).to_bytes(8, "little") + header + data


# Real quantization_config blocks and tensor dtypes as published on Hugging Face (read from
# each model's config.json and safetensors header), with tiny shapes. Per format:
# (torch_dtype, quantization_config, {tensor name: (dtype, shape)}, expected weights summary).
_ELEMENT_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "I8": 1, "I32": 4, "I64": 8}
_PROJ = "model.layers.0.mlp.down_proj"
_CT_WEIGHTS = {"block_structure": None, "group_size": None, "num_bits": 8, "symmetric": True}
QUANTIZED_CHECKPOINTS: dict[str, tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = {
    # Qwen/Qwen2.5-7B-Instruct-AWQ
    "awq": (
        "float16",
        {
            "bits": 4,
            "group_size": 128,
            "quant_method": "awq",
            "version": "gemm",
            "zero_point": True,
        },
        {
            "model.embed_tokens.weight": ("F16", (4, 2)),
            f"{_PROJ}.qweight": ("I32", (8, 2)),
            f"{_PROJ}.qzeros": ("I32", (1, 2)),
            f"{_PROJ}.scales": ("F16", (1, 16)),
            "model.layers.0.self_attn.k_proj.bias": ("F16", (4,)),
        },
        {"weight_precision": "int4", "activation_dtype": "float16", "quantization_method": "awq"}
        | {"weight_bits": 4, "group_size": 128, "activation_scheme": None},
    ),
    # neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16 (act-order GPTQ)
    "gptq": (
        "float16",
        {"bits": 4, "desc_act": True, "group_size": 128, "quant_method": "gptq", "sym": True},
        {
            "model.embed_tokens.weight": ("F16", (4, 2)),
            f"{_PROJ}.g_idx": ("I32", (8,)),
            f"{_PROJ}.qweight": ("I32", (1, 16)),
            f"{_PROJ}.qzeros": ("I32", (1, 2)),
            f"{_PROJ}.scales": ("F16", (1, 16)),
        },
        {"weight_precision": "int4", "activation_dtype": "float16", "quantization_method": "gptq"}
        | {"weight_bits": 4, "group_size": 128, "activation_scheme": None},
    ),
    # neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 (static per-tensor W8A8)
    "compressed-tensors-fp8": (
        "bfloat16",
        {
            "config_groups": {
                "group_0": {
                    "input_activations": {
                        **_CT_WEIGHTS,
                        "dynamic": False,
                        "strategy": "tensor",
                        "type": "float",
                    },
                    "targets": ["Linear"],
                    "weights": {**_CT_WEIGHTS, "strategy": "tensor", "type": "float"},
                }
            },
            "format": "naive-quantized",
            "ignore": ["lm_head"],
            "quant_method": "compressed-tensors",
        },
        {
            "model.embed_tokens.weight": ("BF16", (4, 2)),
            f"{_PROJ}.weight": ("F8_E4M3", (4, 4)),
            f"{_PROJ}.weight_scale": ("BF16", (1,)),
            f"{_PROJ}.input_scale": ("BF16", (1,)),
        },
        {"weight_precision": "fp8", "activation_dtype": "bfloat16"}
        | {"quantization_method": "compressed-tensors", "weight_bits": 8, "group_size": None}
        | {"activation_scheme": "static"},
    ),
    # neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 (dynamic per-token int8)
    "compressed-tensors-int8": (
        "bfloat16",
        {
            "config_groups": {
                "group_0": {
                    "input_activations": {
                        **_CT_WEIGHTS,
                        "dynamic": True,
                        "strategy": "token",
                        "type": "int",
                    },
                    "targets": ["Linear"],
                    "weights": {**_CT_WEIGHTS, "strategy": "channel", "type": "int"},
                }
            },
            "format": "int-quantized",
            "quant_method": "compressed-tensors",
        },
        {
            "model.embed_tokens.weight": ("BF16", (4, 2)),
            f"{_PROJ}.weight": ("I8", (4, 4)),
            f"{_PROJ}.weight_scale": ("BF16", (4, 1)),
        },
        {"weight_precision": "int8", "activation_dtype": "bfloat16"}
        | {"quantization_method": "compressed-tensors", "weight_bits": 8, "group_size": None}
        | {"activation_scheme": "dynamic"},
    ),
    # cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit (grouped int4, packed, unquantized activations)
    "compressed-tensors-int4": (
        "bfloat16",
        {
            "config_groups": {
                "group_0": {
                    "format": "pack-quantized",
                    "input_activations": None,
                    "targets": ["Linear"],
                    "weights": {
                        **_CT_WEIGHTS,
                        "num_bits": 4,
                        "group_size": 32,
                        "strategy": "group",
                        "type": "int",
                    },
                }
            },
            "format": "pack-quantized",
            "ignore": [],
            "quant_method": "compressed-tensors",
        },
        {
            f"{_PROJ}.weight_packed": ("I32", (4, 1)),
            f"{_PROJ}.weight_scale": ("F16", (4, 1)),
            f"{_PROJ}.weight_shape": ("I64", (2,)),
            "model.embed_tokens.weight": ("F16", (4, 2)),
        },
        {"weight_precision": "int4", "activation_dtype": "bfloat16"}
        | {"quantization_method": "compressed-tensors", "weight_bits": 4, "group_size": 32}
        | {"activation_scheme": None},
    ),
    # Qwen/Qwen3-8B-FP8 (block-wise FP8 with dynamic activations)
    "fp8": (
        "bfloat16",
        {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
        {
            "model.embed_tokens.weight": ("BF16", (4, 2)),
            f"{_PROJ}.weight": ("F8_E4M3", (4, 4)),
            f"{_PROJ}.weight_scale_inv": ("BF16", (1, 1)),
        },
        {"weight_precision": "fp8", "activation_dtype": "bfloat16", "quantization_method": "fp8"}
        | {"weight_bits": 8, "group_size": None, "activation_scheme": "dynamic"},
    ),
}


def safetensors_from_tensors(tensors: Mapping[str, tuple[str, tuple[int, ...]]]) -> bytes:
    """Return a well-formed safetensors file with zeroed data for each named tensor."""
    header: dict[str, Any] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        size = _ELEMENT_BYTES[dtype] * math.prod(shape)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    return safetensors_bytes(header_override=header, data=bytes(offset))


def write_shards(
    model: Path, shards: Mapping[str, Mapping[str, tuple[str, tuple[int, ...]]]]
) -> None:
    """Replace ``model.safetensors`` with the named shards (file name -> tensors) and an index."""
    (model / SAFETENSORS_NAME).unlink()
    for shard, tensors in shards.items():
        (model / shard).write_bytes(safetensors_from_tensors(tensors))
    weight_map = {name: shard for shard, tensors in shards.items() for name in tensors}
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def make_gemma4_checkpoint(model: Path) -> None:
    """Turn the registration checkpoint at ``model`` into a two-shard, two-layer, two-expert
    image+text MoE checkpoint with Gemma 4's real config keys and tensor names."""
    config = json.loads((GEMMA4 / "config.json").read_text())
    text = config["text_config"]
    text.update(num_hidden_layers=2, num_experts=2, top_k_experts=1,
                layer_types=["sliding_attention", "full_attention"])  # fmt: skip
    config["quantization_config"]["ignore"] = []
    (model / "config.json").write_text(json.dumps(config))
    (model / "processor_config.json").write_text((GEMMA4 / "processor_config.json").read_text())
    lm = "model.language_model"
    first = {f"{lm}.embed_tokens.weight": ("F16", (4, 2))}
    for layer in range(2):
        base = f"{lm}.layers.{layer}"
        first[f"{base}.self_attn.k_proj.weight_packed"] = ("I32", (2, 2))
        if layer == 0:
            first[f"{base}.self_attn.v_proj.weight_packed"] = ("I32", (2, 2))
        for expert in range(2):
            first[f"{base}.experts.{expert}.down_proj.weight_packed"] = ("I32", (2, 2))
    second = {"model.vision_tower.encoder.layers.0.mlp.down_proj.linear.weight": ("F16", (4, 4))}
    write_shards(model, {SHARDS[0]: first, SHARDS[1]: second})


def checkpoint_documents() -> dict[str, Any]:
    """Return the synthetic JSON documents of one supported text-only checkpoint."""
    return {
        "config.json": {
            "architectures": ["SyntheticForCausalLM"],
            "max_position_embeddings": DEFAULT_CONTEXT_LIMIT,
            "torch_dtype": "bfloat16",
        },
        "generation_config.json": {},
        "tokenizer.json": {
            "version": "1.0",
            "model": {"type": "WordLevel", "vocab": {"x": 0}, "unk_token": "x"},
        },
        "tokenizer_config.json": {"chat_template": "{{ messages }}"},
    }


def write_documents(root: Path, documents: Mapping[str, Any]) -> None:
    """Write each named JSON document as UTF-8 text under ``root``."""
    for name, value in documents.items():
        (root / name).write_text(json.dumps(value), encoding="utf-8")


def make_checkpoint(root: Path, quantized: str | None = None) -> Path:
    """Write one tiny synthetic checkpoint (BF16, or a ``QUANTIZED_CHECKPOINTS`` format)."""
    root.mkdir(parents=True)
    documents = checkpoint_documents()
    weights = safetensors_bytes()
    if quantized is not None:
        torch_dtype, quantization_config, tensors, _ = QUANTIZED_CHECKPOINTS[quantized]
        documents["config.json"] |= {
            "torch_dtype": torch_dtype,
            "quantization_config": quantization_config,
        }
        weights = safetensors_from_tensors(tensors)
    write_documents(root, documents)
    (root / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    (root / SAFETENSORS_NAME).write_bytes(weights)
    return root


def write_config(path: Path, payload: Mapping[str, Any] | None = None) -> Path:
    """Write one registration configuration document and return its path."""
    path.write_text(
        json.dumps(REGISTRATION_CONFIG if payload is None else payload), encoding="utf-8"
    )
    return path


def write_prompts(path: Path, rows: Sequence[Mapping[str, Any]] | None = None) -> Path:
    """Write one plain-text JSONL workload document and return its path."""
    selected = REGISTRATION_PROMPTS if rows is None else rows
    path.write_text(
        "".join(json.dumps(dict(row)) + "\n" for row in selected),
        encoding="utf-8",
    )
    return path


def make_registration_inputs(root: Path, quantized: str | None = None) -> tuple[Path, Path, Path]:
    """Build one model checkpoint, one configuration and one JSONL workload under ``root``."""
    model = make_checkpoint(root / "model", quantized)
    config = write_config(root / "serving.json")
    if quantized is not None:
        expected = QUANTIZED_CHECKPOINTS[quantized][3]
        case = {
            **REGISTRATION_CONFIG["case"],
            "weight_precision": expected["weight_precision"],
            "activation_dtype": expected["activation_dtype"],
        }
        write_config(config, {**REGISTRATION_CONFIG, "case": case})
    return model, config, write_prompts(root / "workload.jsonl")
