"""What vLLM 0.30 reports and can change: every neutral signal from its Prometheus metrics, and
every neutral lever but the offload family."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from ...capabilities import Capabilities, LeverSupport, SignalSupport
from ...levers import LEVERS, Speculation
from ...settings import Settings
from .command import COMPACT_JSON_BACKENDS, SPECULATIVE_CONFIG_FLAG, STRUCTURED_OUTPUTS_FLAG
from .graphs import _capture_sizes, bounded_graphs, json_flag
from .metrics import VLLM_SIGNALS

# Levers vLLM realises with a fixed set of values: n-gram speculation is the only method Tensward
# writes, and fp8 is vLLM's only 8-bit KV cache.
_BOUNDED = {
    "speculation": LeverSupport(values=frozenset({"ngram"})),
    "kv_cache_precision": LeverSupport(values=frozenset({"default", "8bit"})),
}
_NOT_REALISED = frozenset({"host_layers", "host_expert_layers"})

VLLM_CAPABILITIES = Capabilities(
    # Every EngineSignals field, each from its Prometheus family (VLLM_SIGNALS names them all).
    signals={
        name: SignalSupport(quality="exact", source="metrics", how=family.name)
        for name, family in VLLM_SIGNALS.items()
    },
    levers={
        name: _BOUNDED.get(name, LeverSupport()) for name in LEVERS if name not in _NOT_REALISED
    },
    trace="kineto",
    counters=True,
    shared_prefix_kv=True,
    min_cacheable_prefix_tokens=16,  # one 16-token KV block, vLLM's default block size
    workload={},
)


def with_api_servers(servers: int) -> Capabilities:
    """What vLLM reports with ``servers`` API-server processes. Each records a step in
    vllm:iteration_tokens_total only when the step produced output for one of its own requests
    (v1/engine/async_llm.py output_handler, f0c44cc), so with several the count lies between
    the engine's steps and ``servers`` times them, and no step count is read."""
    if servers <= 1:
        return VLLM_CAPABILITIES
    unknown = SignalSupport(
        quality="absent",
        absent_reason=f"{servers} API servers each count only the engine steps that produced "
        "output for their own requests",
    )
    return replace(VLLM_CAPABILITIES, signals={**VLLM_CAPABILITIES.signals, "iterations": unknown})


FP8 = "fp8"  # vLLM's kv_cache_dtype spellings of an 8-bit KV cache all start with it


def speculative_config(speculation: Speculation) -> str:
    """``--speculative-config`` for a method Tensward writes: n-gram, as vLLM 0.30 names it."""
    if speculation.method != "ngram":
        raise ValueError(f"Tensward does not write vLLM speculation by {speculation.method!r}")
    return json.dumps({
        "method": "ngram",
        "num_speculative_tokens": speculation.draft_tokens,
        "prompt_lookup_max": speculation.lookup_max,
    })  # fmt: skip


def _read_speculation(settings: Settings) -> Speculation | None:
    """Any ``--speculative-config`` reads as a Speculation, one Tensward cannot parse as method
    "unknown"; None only when the flag is absent."""
    if SPECULATIVE_CONFIG_FLAG not in settings.extra_args:
        return None
    config = json_flag(settings, SPECULATIVE_CONFIG_FLAG)
    tokens, lookup, model = (
        config.get("num_speculative_tokens"), config.get("prompt_lookup_max"), config.get("model")
    )  # fmt: skip
    return Speculation(
        str(config.get("method") or "unknown"),
        tokens if type(tokens) is int and tokens > 0 else 0,
        lookup if type(lookup) is int and lookup > 0 else 0,
        model if isinstance(model, str) else None,
    )


def compact_json_config(settings: Settings) -> dict[str, Any] | None:
    """The structured-outputs config with whitespace disabled, the backend pinned to xgrammar when
    the setup leaves it to vLLM ("auto" refuses the option); None when it is already disabled or
    the backend cannot take it."""
    config = json_flag(settings, STRUCTURED_OUTPUTS_FLAG)
    if config.get("disable_any_whitespace") is True:
        return None
    if config.get("backend", "auto") == "auto":
        config["backend"] = "xgrammar"
    if config["backend"] not in COMPACT_JSON_BACKENDS:
        return None
    return {**config, "disable_any_whitespace": True}


def _stored_field(lever: str) -> str:
    """The Settings field an identity lever is stored in."""
    known = LEVERS[lever]
    if not known.identity or known.field is None:
        raise ValueError(f"vLLM does not realise the lever {lever}")
    return known.field


def read_lever(settings: Settings, lever: str) -> Any:
    """``lever``'s value in ``settings`` as vLLM stores it."""
    if lever == "speculation":
        return _read_speculation(settings)
    if lever == "compact_json":
        return json_flag(settings, STRUCTURED_OUTPUTS_FLAG).get("disable_any_whitespace") is True
    if lever == "kv_cache_precision":
        return "8bit" if settings.kv_cache_dtype.startswith(FP8) else "default"
    if lever == "graph_capture":
        return settings.cuda_graphs
    if lever == "graph_capture_limit":
        return _capture_sizes(settings)[1]
    return getattr(settings, _stored_field(lever))


def write_lever(settings: Settings, lever: str, value: Any) -> Settings:
    """``settings`` with ``lever`` set to ``value`` in vLLM's fields and flags."""
    extra = settings.extra_args
    if lever == "speculation":
        if value is None:
            kept = {k: v for k, v in extra.items() if k != SPECULATIVE_CONFIG_FLAG}
            return replace(settings, extra_args=kept)
        written = speculative_config(value)
        return replace(settings, extra_args={**extra, SPECULATIVE_CONFIG_FLAG: written})
    if lever == "compact_json":
        if value is not True:
            raise ValueError("Tensward only turns vLLM's compact JSON on")
        config = compact_json_config(settings)
        if config is None:
            return settings
        return replace(settings, extra_args={**extra, STRUCTURED_OUTPUTS_FLAG: json.dumps(config)})
    if lever == "kv_cache_precision":
        if value not in ("default", "8bit"):
            raise ValueError(f"vLLM has no {value} KV cache")
        return replace(settings, kv_cache_dtype=FP8 if value == "8bit" else "auto")
    if lever == "graph_capture":
        return replace(settings, cuda_graphs=bool(value))
    if lever == "graph_capture_limit":
        return bounded_graphs(settings, value)
    change: dict[str, Any] = {_stored_field(lever): value}
    return replace(settings, **change)
