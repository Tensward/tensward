"""vLLM's command-line rules, as the engine-neutral :class:`FlagRules`."""

from __future__ import annotations

from typing import Any, Mapping

from . import flags
from .command import VLLM_VERSION, default_kv_memory_fraction


class _VllmFlagRules:
    engine = "vllm"
    version = VLLM_VERSION
    env_names = frozenset().union(*flags.ENV_ALLOWED.values())
    pool_shrinking: Mapping[str, str] = {
        "gpu-memory-utilization": "kv_memory_fraction",
        "max-model-len": "max_context_len",
        "max-num-seqs": "max_concurrent_requests",
    }
    pool_bytes_flag: str | None = "kv-cache-memory-bytes"
    pool_sharing: frozenset[str] = frozenset()
    canonical = staticmethod(flags.canonical_flags)
    allowed = staticmethod(flags.allowed)
    env_allowed = staticmethod(flags.env_allowed)
    env_value_allowed = staticmethod(flags.env_value_allowed)
    merge = staticmethod(flags.merge_json)
    canonical_args = staticmethod(flags.canonical_args)

    def pool_defaults(self, version: str | None) -> Mapping[str, Any]:
        return {"gpu-memory-utilization": default_kv_memory_fraction(version, {})}


VLLM_FLAG_RULES = _VllmFlagRules()
