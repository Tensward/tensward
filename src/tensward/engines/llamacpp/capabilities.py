"""What llama-server reports and can change at build 11669."""

from __future__ import annotations

from ...capabilities import Capabilities, LeverSupport, Quality, SignalSupport, Source
from .metrics import LLAMACPP_SIGNALS

_ABSENT = {
    "preemptions": "llama-server does not preempt: a request waits for a free slot",
    "kv_capacity_tokens": "the pool size is read from the server log",
    "kv_max_concurrency": "the pool size is read from the server log",
    "kv_blocks": "llama-server's KV cache has cells, not blocks",
    "kv_block_tokens": "llama-server's KV cache has cells, not blocks",
    "hybrid_cache": "llama-server does not label a hybrid cache",
    "frontend_cpu_seconds": "llama-server's metrics carry no process CPU time",
}

_FROM_TIMINGS: dict[str, tuple[Quality, Source, str]] = {
    "prompt_tokens": ("derived", "api", "timings.prompt_n + timings.cache_n"),
    "prefix_cache_hits": ("derived", "api", "timings.cache_n"),
    "prefix_cache_queries": ("derived", "api", "timings.prompt_n + timings.cache_n"),
    "prefill_seconds": ("derived", "api", "timings.prompt_ms"),
    "queue_seconds": ("derived", "client", "time to first token - timings.prompt_ms"),
}

LEVER_NAMES = (
    "max_concurrent_requests", "max_context_len", "prefill_batch_tokens", "kv_cache_precision",
    "prefix_caching", "tool_calling", "host_layers", "host_expert_layers",
)  # fmt: skip

LLAMACPP_CAPABILITIES = Capabilities(
    signals={
        **{
            name: SignalSupport(
                quality="exact",
                source="metrics",
                how=family.name,
                cadence="request_end" if name == "generation_tokens" else "step",
            )
            for name, family in LLAMACPP_SIGNALS.items()
        },
        **{
            name: SignalSupport(quality=quality, source=source, how=how)
            for name, (quality, source, how) in _FROM_TIMINGS.items()
        },
        "kv_usage": SignalSupport(
            quality="approximate",
            source="api",
            how="/slots: the fullest busy slot's n_prompt_tokens over n_ctx",
            bias="a slot's share, not the pool's: a lower bound for a unified pool",
        ),
        **{
            name: SignalSupport(quality="absent", absent_reason=reason)
            for name, reason in _ABSENT.items()
        },
    },
    levers={
        name: LeverSupport(values=frozenset({"default", "8bit"}))
        if name == "kv_cache_precision"
        else LeverSupport()
        for name in LEVER_NAMES
    },
    trace=None,
    counters=False,
    shared_prefix_kv=False,
    min_cacheable_prefix_tokens=1,
    workload={},
    resolves_at_launch=True,
)
