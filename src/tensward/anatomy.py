"""What a checkpoint is made of, from its config and its safetensors headers.

Every tensor is classified by name (text dense, input embedding, gathered embedding tables,
output head, routed experts, shared experts, vision encoder, audio encoder, multi-token
prediction, unclassified). Attention shape, MoE routing and
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
# vLLM 0.30 caches these full-head (``minicpm3.py`` builds a plain ``Attention`` with V padded to
# the key width) though their config carries ``kv_lora_rank``
_UNLATENT_TYPES = frozenset(("minicpm3",))
# a latent cache with more than the one per-layer vector, or on some layers only
_MLA_UNPRICED = (
    ("index_topk", "an indexer cache"),
    ("index_head_dim", "an indexer cache"),
    ("compress_ratios", "a compressed cache"),
    ("layer_types", "attention that differs by layer"),
    ("layers_block_type", "attention that differs by layer"),
    ("linear_attn_config", "attention that differs by layer"),
    ("layer_group_size", "attention that differs by layer"),
)
_STATE_LAYER = re.compile(r"mamba|ssm|linear|recurrent|delta", re.IGNORECASE)
_STATE_KEYS = ("mamba_d_state", "ssm_state_size", "linear_conv_kernel_dim", "mamba_n_heads")
_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}

_NOT_PARAMETERS = (".weight_scale", ".weight_shape", ".weight_zero_point", ".scales", ".qzeros",
                   ".g_idx", ".input_scale", ".weight_scale_inv", "_scales")  # fmt: skip
_PACKED = (".weight_packed", ".qweight")
_BLOCKS = "_blocks"  # mxfp4: bytes that each hold two 4-bit values

_CLASSES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("vision", re.compile(r"(^|\.)(vision_tower|vision_model|visual|embed_vision"
                          r"|multi_modal_projector|mm_projector)\.")),
    ("audio", re.compile(r"(^|\.)(audio_tower|audio_model|embed_audio)\.")),
    ("mtp", re.compile(r"(^|\.)mtp[._]|\.mtp_")),
    ("shared_experts", re.compile(r"\.layers\.\d+\.(?:.*\.)?shared_experts?\.")),
    ("routed_experts", re.compile(r"\.layers\.\d+\.(?:.*\.)?experts\.")),
    ("gathered_embedding", re.compile(r"(^|\.)embed\w*per_layer\w*\.")),
    ("embedding", re.compile(r"(^|\.)(embed_tokens|wte|tok_embeddings)\.weight$")),
    ("lm_head", re.compile(r"(^|\.)lm_head\.weight$")),
    ("text_dense", re.compile(r"(^|\.)layers\.\d+\.|(^|\.)(norm|final_layernorm)\.weight$")),
    ("text_dense", re.compile(r"(^|\.)language_model\.(?!layers\.)\w+\.(weight|bias)$")),
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
    gathered_embedding: int = 0  # tables read by index, L x d values a token (per-layer inputs)
    mtp: int = 0  # multi-token prediction heads: loaded, never read by a decode step


@dataclass(frozen=True, slots=True)
class AttentionGroup:
    kind: Literal["full", "sliding"]
    layers: int
    kv_heads: int
    head_dim: int
    window: int | None  # a sliding layer keeps at most this many tokens
    k_eq_v: bool  # the layer has no V projection: its V is its K (describes the model only:
    # vLLM 0.30 still stores both K and V for such a layer)


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
    kv_shared_layers: int = 0  # trailing layers that reuse the KV of earlier ones
    mixed_head_dims: bool = False
    mla: bool = False  # the cache holds one latent vector a token and layer, not K and V
    linear_attention: bool = False  # some layers keep a fixed-size state a sequence, not KV
    state_bytes_per_seq: int = 0  # that state, 0 when the layers are not modelled
    architecture: str | None = None  # config.json's first architecture class
    dtype: str | None = None  # the dtype the weights are stored in, or the engine runs them in
    quantization: str | None = None  # the checkpoint's quantization method, "none" if unquantized
    weight_bits: int | None = None  # bits a stored weight takes: the quantizer's, else the dtype's
    gptq_act_order: bool = False  # a GPTQ checkpoint quantized with act-order (desc_act)
    unclassified_share: float | None = None  # of all tensor bytes, those no class claimed

    def kv_bytes(self, tokens: int, kv_cache_dtype: str, in_flight_tokens: int = 0) -> int | None:
        """KV-cache bytes for ``tokens`` tokens, or None if not modelled.

        ``in_flight_tokens`` > 0 gives what an engine reserves per request (admission: a
        sliding layer keeps its window less one plus the tokens in flight), 0 what one
        sequence holds.
        """
        if not self.attention:
            return None
        element = 1 if kv_cache_dtype.startswith(("fp8", "int8")) else 2
        streams = 1 if self.mla else 2  # K and V; the latent cache is one vector

        def held(g: AttentionGroup) -> int:
            if g.window is None:
                return tokens
            return min(tokens, g.window - 1 + in_flight_tokens if in_flight_tokens else g.window)

        return sum(
            g.layers * held(g) * g.kv_heads * g.head_dim * element * streams for g in self.attention
        )

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
            "mla": self.mla,
            "linear_attention": self.linear_attention,
            "state_bytes_per_seq": self.state_bytes_per_seq,
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
    *,
    dtype: str | None = None,
    quantization: str | None = None,
    act_order: bool | None = None,
) -> ModelAnatomy:
    nested = config.get("text_config")
    text: Mapping[str, Any] = nested if isinstance(nested, dict) else config
    reasons: list[str] = []
    sizes: Counter[str] = Counter()
    dtypes: dict[str, set[str]] = {}
    params: Counter[str] = Counter()
    vocab = _count(text.get("vocab_size"))
    for name, (tensor_dtype, shape, size) in tensors.items():
        kind = next((k for k, pattern in _CLASSES if pattern.search(name)), "unclassified")
        if kind in ("unclassified", "text_dense") and _is_vocab_table(name, shape, vocab):
            kind = "gathered_embedding"
        sizes[kind] += size
        dtypes.setdefault(kind, set()).add(tensor_dtype)
        params[kind] += _parameters(name, tensor_dtype, shape, weight_bits)
    total = sum(sizes.values())
    components = counts = None
    if total and sizes["unclassified"] <= UNCLASSIFIED_LIMIT * total:
        components = Components(**{name: sizes[name] for name in Components.__slots__})
        counts = Components(**{name: params[name] for name in Components.__slots__})
    else:
        reasons.append(f"{sizes['unclassified']} of {total} tensor bytes are unclassified")
    model_type = config.get("model_type")
    arch = text.get("model_type") or model_type
    attention = _attention(text, arch, tensors, _kv_shared_layers(config, text), reasons)
    linear = _has_state(text)
    latent = bool(attention) and _is_latent(text, arch)
    return ModelAnatomy(
        model_type=model_type if isinstance(model_type, str) else None,
        validated=model_type in VALIDATED,
        hidden=_count(text.get("hidden_size")),
        layers=_count(text.get("num_hidden_layers")),
        vocab=vocab,
        # Families differ on the flag's default; a checkpoint without an output head ties it.
        tied_embeddings=bool(text.get("tie_word_embeddings", config.get("tie_word_embeddings")))
        or (components is not None and components.lm_head == 0 and components.embedding > 0),
        components=components,
        params=counts,
        precisions={kind: tuple(sorted(found)) for kind, found in sorted(dtypes.items())},
        attention=attention,
        moe=_moe(text, tensors, sizes["routed_experts"], reasons),
        vision=_vision(config, processor, sizes["vision"], params["vision"]),
        modalities=_modalities(config, processor),
        unavailable=tuple(reasons),
        kv_shared_layers=_kv_shared_layers(config, text),
        mixed_head_dims=len({g.head_dim for g in attention}) > 1,
        mla=latent,
        linear_attention=linear,
        state_bytes_per_seq=_state_bytes(text, reasons) if linear else 0,
        architecture=_architecture(config),
        dtype=dtype,
        quantization=quantization,
        weight_bits=weight_bits or (8 * _DTYPE_BYTES[dtype] if dtype in _DTYPE_BYTES else None),
        gptq_act_order=quantization == "gptq" and act_order is True,
        unclassified_share=sizes["unclassified"] / total if total else None,
    )


def _architecture(config: Mapping[str, Any]) -> str | None:
    names = config.get("architectures")
    return names[0] if isinstance(names, list) and names and isinstance(names[0], str) else None


def freed_bytes(anatomy: ModelAnatomy) -> dict[str, int] | None:
    """What a memory-freeing change can release: ``towers`` (vision and audio encoders) and
    ``quantizable`` (the linear and expert weights an online fp8 quantization shrinks). The
    output head and the gathered tables are not counted: vLLM's fp8 config quantizes linear
    and expert layers only, and ``ParallelLMHead`` is neither. None without components."""
    parts = anatomy.components
    if parts is None:
        return None
    return {
        "towers": parts.vision + parts.audio,
        "quantizable": parts.text_dense + parts.routed_experts + parts.shared_experts,
    }


def _is_vocab_table(name: str, shape: tuple[int, ...], vocab: int | None) -> bool:
    """An input-side table with one row per token, which is not the main embedding (that one
    is classed by name before this is asked)."""
    return (
        vocab is not None
        and len(shape) == 2
        and shape[0] == vocab
        and any("embed" in part for part in name.split(".")[-2:])
    )


def _kv_shared_layers(config: Mapping[str, Any], text: Mapping[str, Any]) -> int:
    return (
        _count(text.get("num_kv_shared_layers")) or _count(config.get("num_kv_shared_layers")) or 0
    )


def _parameters(name: str, dtype: str, shape: tuple[int, ...], weight_bits: int | None) -> int:
    """Matmul parameters a tensor holds; quantization bookkeeping holds none."""
    if name.endswith(_NOT_PARAMETERS):
        return 0
    if name.endswith(_PACKED) and dtype == "I32" and weight_bits:
        return math.prod(shape) * (32 // weight_bits)
    if name.endswith(_BLOCKS) and dtype == "U8" and weight_bits:
        return math.prod(shape) * (8 // weight_bits)
    return math.prod(shape)


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _is_latent(text: Mapping[str, Any], arch: object) -> bool:
    return text.get("kv_lora_rank") is not None and arch not in _UNLATENT_TYPES


def _has_state(text: Mapping[str, Any]) -> bool:
    types = text.get("layer_types")
    return any(text.get(key) is not None for key in _STATE_KEYS) or (
        isinstance(types, list)
        and any(isinstance(t, str) and _STATE_LAYER.search(t) for t in types)
    )


def _state_bytes(text: Mapping[str, Any], reasons: list[str]) -> int:
    """The state-space layers' fixed state a sequence (the convolution window and the scan
    state, at the model dtype), for the layout the Mamba 2 hybrids use; 0 with a reason
    otherwise."""
    heads, head_dim, state = (_count(text.get(k)) for k in ("mamba_n_heads", "mamba_d_head",
                                                            "mamba_d_state"))  # fmt: skip
    kernel, groups = _count(text.get("mamba_d_conv")), _count(text.get("mamba_n_groups")) or 1
    types, size = text.get("layer_types"), _DTYPE_BYTES.get(str(text.get("dtype", "")))
    size = size or _DTYPE_BYTES.get(str(text.get("torch_dtype", "")))
    if not (heads and head_dim and state and kernel and size and isinstance(types, list)):
        reasons.append("the size of the per-sequence state is not modelled for this layout")
        return 0
    layers = sum(1 for t in types if isinstance(t, str) and _STATE_LAYER.search(t))
    width = heads * head_dim
    return layers * size * (width * state + (kernel - 1) * (width + 2 * groups * state))


def _attention(
    text: Mapping[str, Any], arch: object, tensors: Tensors, shared: int, reasons: list[str]
) -> tuple[AttentionGroup, ...]:
    if _has_state(text):
        reasons.append(
            "layers with a state for each sequence (state-space or linear attention) are not "
            "modelled, so no KV pool is predicted"
        )
        return ()
    if text.get("kv_lora_rank") is not None:
        return _latent_attention(text, arch, reasons)
    layers, heads = _count(text.get("num_hidden_layers")), _count(text.get("num_attention_heads"))
    hidden = _count(text.get("hidden_size"))
    if layers is None or heads is None or hidden is None:
        reasons.append("config.json lacks the attention dimensions")
        return ()
    kv_heads = _count(text.get("num_key_value_heads")) or heads
    head_dim = _count(text.get("head_dim")) or hidden // heads
    types = text.get("layer_types") or ["full_attention"] * layers
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


def _latent_attention(
    text: Mapping[str, Any], arch: object, reasons: list[str]
) -> tuple[AttentionGroup, ...]:
    """The cache of a latent-attention model: one vector of the compressed key-value rank and
    the rotary width for each token and layer. The models vLLM decompresses to full heads keep
    K and V at the key width instead. A cache with more than that one vector is not priced."""
    layers, rank, rope = (
        _count(text.get(k)) for k in ("num_hidden_layers", "kv_lora_rank", "qk_rope_head_dim")
    )
    unpriced = next((why for key, why in _MLA_UNPRICED if text.get(key) is not None), None)
    if unpriced is not None:
        reasons.append(f"KV bytes not priced: the attention has {unpriced}")
        return ()
    if not (layers and rank and rope):
        reasons.append("config.json lacks the latent attention dimensions")
        return ()
    if _is_latent(text, arch):
        return (AttentionGroup("full", layers, 1, rank + rope, None, False),)
    heads = _count(text.get("num_attention_heads"))
    nope = _count(text.get("qk_nope_head_dim"))
    if not (heads and nope):
        reasons.append("config.json lacks the attention dimensions")
        return ()
    kv_heads = _count(text.get("num_key_value_heads")) or heads
    return (AttentionGroup("full", layers, kv_heads, nope + rope, None, False),)


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


def _modalities(config: Mapping[str, Any], processor: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Input types the config declares (a null config means absent), and video when the
    processor has a video processor, which makes the engine reserve memory for it. Tensward
    measures text and images."""
    found = {"text"}
    if config.get("vision_config") is not None:
        found.add("image")
    if (processor or {}).get("video_processor") is not None:
        found.add("video")
    if config.get("audio_config") is not None:
        found.add("audio")
    return tuple(sorted(found))
