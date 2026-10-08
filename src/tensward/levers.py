"""Levers: the engine-neutral knobs a playbook entry turns. An identity lever is the ``Settings``
field of the same name and holds neutral values; every other lever is read and written only
through its engine (``Engine.lever_value`` and ``Engine.with_lever``), which keeps it in the
field named here or in its own flags."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True, kw_only=True)
class Lever:
    name: str
    meaning: str
    value: str  # a type, a union with None, or the allowed values
    identity: bool
    field: str | None  # the Settings field that stores it; None: the engine's own flags
    aliases: tuple[str, ...] = ()  # other engines' names for it


@dataclass(frozen=True, slots=True)
class Speculation:
    """Speculative decoding as a lever value. ``method`` is "unknown" for a configuration the
    engine cannot read."""

    method: str
    draft_tokens: int = 0
    lookup_max: int = 0
    draft_model: str | None = None


def _identity(name: str, value: str, meaning: str, *aliases: str) -> Lever:
    return Lever(
        name=name, meaning=meaning, value=value, identity=True, field=name, aliases=aliases
    )


def _engine_kept(name: str, value: str, meaning: str, field: str | None = None) -> Lever:
    return Lever(name=name, meaning=meaning, value=value, identity=False, field=field)


LEVERS: Mapping[str, Lever] = {
    lever.name: lever
    for lever in (
        _identity(
            "max_concurrent_requests", "int", "requests the engine runs at once", "max_concurrency"
        ),
        _identity("max_context_len", "int", "the longest prompt plus output one request may hold"),
        _identity("kv_memory_fraction", "float", "share of the device memory the engine may take"),
        _identity("prefill_batch_tokens", "int", "prompt tokens one engine step may compute"),
        _engine_kept(
            "kv_cache_precision",
            "default|8bit|4bit",
            "KV-cache precision: default, 8bit or 4bit",
            "kv_cache_dtype",
        ),
        _identity("prefix_caching", "bool", "reuse the KV cache of a shared prompt prefix"),
        _identity(
            "quantization", "str|None", "the weight quantization method the engine is told to use"
        ),
        _engine_kept("graph_capture", "bool", "run decode steps as captured graphs", "cuda_graphs"),
        _engine_kept("graph_capture_limit", "int", "the largest batch captured as a graph"),
        _identity(
            "async_scheduling", "bool", "schedule the next step while the device runs this one"
        ),
        _identity("api_server_count", "int", "frontend processes that tokenize and stream"),
        _identity("tool_calling", "bool", "answers may be tool calls"),
        _identity("tool_parser", "str", "the parser of the model family's tool-call format"),
        _identity("media_inputs", "bool", "serve the media encoders of an image+text checkpoint"),
        _identity("media_limits", "map", "media items per prompt, per modality"),
        _engine_kept(
            "speculation", "Speculation|None", "speculative decoding: a Speculation, or None"
        ),
        _engine_kept(
            "compact_json", "bool", "structured output allows no whitespace between JSON tokens"
        ),
        _identity("dtype", "str", "the activation dtype"),
        _engine_kept("host_layers", "int", "layers whose weights stay in host memory"),
        _engine_kept(
            "host_expert_layers", "int", "MoE layers whose routed experts stay in host memory"
        ),
    )
}
