"""What a checkpoint is made of, from its config and its safetensors headers.

Every tensor is classified by name (text dense, input embedding, output head, routed experts,
shared experts, vision encoder, audio encoder, unclassified). Attention shape, MoE routing and
input types come from ``config.json`` (dimensions may be nested under ``text_config``) and the
processor config. The anatomy is derived from files registration already verified and is never
stored, so it is part of no identity. What it cannot derive is ``None`` with the reason in
``unavailable``; nothing is guessed, and nothing here refuses a checkpoint.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

Tensors = Mapping[str, tuple[str, tuple[int, ...], int]]  # name -> (dtype, shape, bytes)

UNCLASSIFIED_LIMIT = 0.01  # above this share of all bytes, the component split is unavailable
VALIDATED = frozenset(("gemma4", "llama", "mistral", "qwen2", "qwen3"))

_EXPERT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")
_TOP_K_KEYS = ("num_experts_per_tok", "top_k_experts", "top_k", "moe_topk")
_MLA_KEYS = ("kv_lora_rank", "q_lora_rank")

_NOT_PARAMETERS = (".weight_scale", ".weight_shape", ".weight_zero_point", ".scales", ".qzeros",
                   ".g_idx", ".input_scale", ".weight_scale_inv")  # fmt: skip
_PACKED = (".weight_packed", ".qweight")

_CLASSES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("vision", re.compile(r"(^|\.)(vision_tower|vision_model|visual|embed_vision"
                          r"|multi_modal_projector|mm_projector)\.")),
    ("audio", re.compile(r"(^|\.)(audio_tower|audio_model|embed_audio)\.")),
    ("shared_experts", re.compile(r"\.layers\.\d+\.(?:.*\.)?shared_experts?\.")),
    ("routed_experts", re.compile(r"\.layers\.\d+\.(?:.*\.)?experts\.")),
    ("embedding", re.compile(r"(^|\.)(embed_tokens|wte|tok_embeddings)\.weight$")),
    ("lm_head", re.compile(r"(^|\.)lm_head\.weight$")),
    ("text_dense", re.compile(r"(^|\.)layers\.\d+\.|(^|\.)(norm|final_layernorm)\.weight$")),
)  # fmt: skip
_EXPERT = re.compile(r"\.layers\.(?P<layer>\d+)\.(?:.*\.)?experts\.(?P<expert>\d+)\.")
_EXPERT_LAYER = re.compile(r"\.layers\.(?P<layer>\d+)\.(?:.*\.)?experts\.")
# the text decoder only: the vision tower has its own layers.N.self_attn.v_proj
_V_PROJ = re.compile(r"(^|\.)(model|language_model)\.layers\.(?P<layer>\d+)\.self_attn\.v_proj\.")


@dataclass(frozen=True, slots=True)
class Components:
    """Bytes at stored precision. ``lm_head`` is 0 when the output head is the embedding."""

    text_dense: int  # attention, norms, dense MLPs, routers
    embedding: int
    lm_head: int
    routed_experts: int
    shared_experts: int
    vision: int
    audio: int
    unclassified: int


@dataclass(frozen=True, slots=True)
class AttentionGroup:
    kind: Literal["full", "sliding"]
    layers: int
    kv_heads: int
    head_dim: int
    window: int | None  # a sliding layer keeps at most this many tokens
    k_eq_v: bool  # the layer has no V projection: its V is its K (describes the model only:
    # vLLM 0.30 stores K and V even then, Spike S3: equal 128 KiB pages for sliding and full)


@dataclass(frozen=True, slots=True)
class Moe:
    experts: int
    experts_per_token: int
    layers: int
    expert_bytes: int  # one routed expert of one layer


@dataclass(frozen=True, slots=True)
class Vision:
    max_tokens_per_image: int | None  # an upper bound; the real count depends on the image
    encoder_bytes: int
    encoder_params: int


@dataclass(frozen=True, slots=True)
class ModelAnatomy:
    model_type: str | None
    validated: bool
    hidden: int | None
    layers: int | None
    vocab: int | None
    tied_embeddings: bool
    components: Components | None
    params: Components | None  # matmul parameters per component, packed weights unpacked
    precisions: Mapping[str, tuple[str, ...]]
    attention: tuple[AttentionGroup, ...]
    moe: Moe | None
    vision: Vision | None
    modalities: tuple[str, ...]
    unavailable: tuple[str, ...] = field(default=())

    def kv_bytes(self, tokens: int, kv_cache_dtype: str, in_flight_tokens: int = 0) -> int | None:
        """KV-cache bytes for ``tokens`` tokens, or None if not modelled.

        ``in_flight_tokens`` > 0 gives what an engine reserves per request (admission: a
        sliding layer keeps its window less one plus the tokens in flight), 0 what one
        sequence holds.
        """
        if not self.attention:
            return None
        element = 1 if kv_cache_dtype.startswith("fp8") else 2

        def held(g: AttentionGroup) -> int:
            if g.window is None:
                return tokens
            return min(tokens, g.window - 1 + in_flight_tokens if in_flight_tokens else g.window)

        return sum(
            g.layers * held(g) * g.kv_heads * g.head_dim * element * 2 for g in self.attention
        )  # the last 2: K and V

    def summary(self) -> dict[str, object]:
        """What ``init`` and ``inspect`` print."""
        parts = self.components
        return {
            "model_type": self.model_type,
            "validated": self.validated,
            "modalities": list(self.modalities),
            "components_bytes": None if parts is None else {
                name: getattr(parts, name) for name in Components.__slots__
            },
            "attention": [
                {"kind": g.kind, "layers": g.layers, "kv_heads": g.kv_heads,
                 "head_dim": g.head_dim, "window": g.window, "k_eq_v": g.k_eq_v}
                for g in self.attention
            ],
            "moe": None if self.moe is None else {
                "experts": self.moe.experts, "experts_per_token": self.moe.experts_per_token,
                "layers": self.moe.layers, "expert_bytes": self.moe.expert_bytes,
            },
            "max_tokens_per_image": self.vision.max_tokens_per_image if self.vision else None,
            "unavailable": list(self.unavailable),
        }  # fmt: skip


def build_anatomy(
    config: Mapping[str, Any],
    processor: Mapping[str, Any] | None,
    tensors: Tensors,
    weight_bits: int | None = None,
) -> ModelAnatomy:
    nested = config.get("text_config")
    text: Mapping[str, Any] = nested if isinstance(nested, dict) else config
    reasons: list[str] = []
    sizes: Counter[str] = Counter()
    dtypes: dict[str, set[str]] = {}
    params: Counter[str] = Counter()
    for name, (dtype, shape, size) in tensors.items():
        kind = next((k for k, pattern in _CLASSES if pattern.search(name)), "unclassified")
        sizes[kind] += size
        dtypes.setdefault(kind, set()).add(dtype)
        params[kind] += _parameters(name, dtype, shape, weight_bits)
    total = sum(sizes.values())
    components = counts = None
    if total and sizes["unclassified"] <= UNCLASSIFIED_LIMIT * total:
        components = Components(**{name: sizes[name] for name in Components.__slots__})
        counts = Components(**{name: params[name] for name in Components.__slots__})
    else:
        reasons.append(f"{sizes['unclassified']} of {total} tensor bytes are unclassified")
    model_type = config.get("model_type")
    return ModelAnatomy(
        model_type=model_type if isinstance(model_type, str) else None,
        validated=model_type in VALIDATED,
        hidden=_count(text.get("hidden_size")),
        layers=_count(text.get("num_hidden_layers")),
        vocab=_count(text.get("vocab_size")),
        # Families differ on the flag's default; a checkpoint without an output head ties it.
        tied_embeddings=bool(text.get("tie_word_embeddings", config.get("tie_word_embeddings")))
        or (components is not None and components.lm_head == 0 and components.embedding > 0),
        components=components,
        params=counts,
        precisions={kind: tuple(sorted(found)) for kind, found in sorted(dtypes.items())},
        attention=_attention(text, tensors, reasons),
        moe=_moe(text, tensors, sizes["routed_experts"], reasons),
        vision=_vision(config, processor, sizes["vision"], params["vision"]),
        modalities=_modalities(config),
        unavailable=tuple(reasons),
    )


def _parameters(name: str, dtype: str, shape: tuple[int, ...], weight_bits: int | None) -> int:
    """Matmul parameters a tensor holds; quantization bookkeeping holds none."""
    if name.endswith(_NOT_PARAMETERS):
        return 0
    if name.endswith(_PACKED) and dtype == "I32" and weight_bits:
        return math.prod(shape) * (32 // weight_bits)
    return math.prod(shape)


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _attention(
    text: Mapping[str, Any], tensors: Tensors, reasons: list[str]
) -> tuple[AttentionGroup, ...]:
    if any(text.get(key) is not None for key in _MLA_KEYS):
        reasons.append("MLA attention is not modelled")
        return ()
    layers, heads = _count(text.get("num_hidden_layers")), _count(text.get("num_attention_heads"))
    hidden = _count(text.get("hidden_size"))
    if layers is None or heads is None or hidden is None:
        reasons.append("config.json lacks the attention dimensions")
        return ()
    kv_heads = _count(text.get("num_key_value_heads")) or heads
    head_dim = _count(text.get("head_dim")) or hidden // heads
    types = text.get("layer_types") or ["full_attention"] * layers
    shared = _count(text.get("num_kv_shared_layers")) or 0  # the last layers reuse earlier KV
    if not isinstance(types, list) or len(types) != layers:
        reasons.append("layer_types does not list every layer")
        return ()
    with_v = {int(m["layer"]) for name in tensors if (m := _V_PROJ.search(name))}
    k_eq_v = text.get("attention_k_eq_v") is True
    groups: Counter[tuple[str, int, int, int | None, bool]] = Counter()
    for index, kind in enumerate(types[: layers - shared]):
        no_v = k_eq_v and index not in with_v
        if kind == "sliding_attention" and _count(text.get("sliding_window")):
            groups["sliding", kv_heads, head_dim, text["sliding_window"], no_v] += 1
        elif kind == "full_attention":
            full_heads = _count(text.get("num_global_key_value_heads")) or kv_heads
            full_dim = _count(text.get("global_head_dim")) or head_dim
            groups["full", full_heads, full_dim, None, no_v] += 1
        else:
            reasons.append(f"attention layer type {str(kind)[:40]!r} is not modelled")
            return ()
    return tuple(
        AttentionGroup(kind, count, heads_, dim, window, no_v)  # type: ignore[arg-type]
        for (kind, heads_, dim, window, no_v), count in groups.items()
    )


def _moe(
    text: Mapping[str, Any], tensors: Tensors, routed_bytes: int, reasons: list[str]
) -> Moe | None:
    experts = next((n for k in _EXPERT_KEYS if (n := _count(text.get(k))) and n > 1), None)
    if experts is None or text.get("enable_moe_block") is False:
        return None
    top_k = next((n for k in _TOP_K_KEYS if (n := _count(text.get(k)))), None)
    layers = {int(m["layer"]) for name in tensors if (m := _EXPERT_LAYER.search(name))}
    indexed = {(m["layer"], m["expert"]) for name in tensors if (m := _EXPERT.search(name))}
    if top_k is None or not layers or (indexed and len(indexed) != len(layers) * experts):
        reasons.append(f"the expert tensors do not match the declared {experts} experts")
        return None
    return Moe(experts, top_k, len(layers), routed_bytes // (len(layers) * experts))


def _vision(
    config: Mapping[str, Any], processor: Mapping[str, Any] | None, size: int, params: int
) -> Vision | None:
    if config.get("vision_config") is None and not size:
        return None
    image = (processor or {}).get("image_processor") or {}
    candidates = (
        config.get("vision_soft_tokens_per_image"),
        config.get("mm_tokens_per_image"),
        (processor or {}).get("image_seq_length"),
        image.get("max_soft_tokens") if isinstance(image, dict) else None,
    )
    tokens = next((n for value in candidates if (n := _count(value))), None)
    return Vision(max_tokens_per_image=tokens, encoder_bytes=size, encoder_params=params)


def _modalities(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Input types the config declares (a null config means absent). Recorded only: Tensward
    measures text and images."""
    found = {"text"}
    if config.get("vision_config") is not None:
        found.add("image")
        if config.get("video_config") is not None:
            found.add("video")
    if config.get("audio_config") is not None:
        found.add("audio")
    return tuple(sorted(found))
