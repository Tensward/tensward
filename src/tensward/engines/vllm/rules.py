"""vLLM's own diagnosis rules: facts of its hybrid (linear-attention) cache that the neutral
classifier does not model."""

from __future__ import annotations

from ...classify import EngineRule, Finding, RuleInputs, finding_at
from ...measurement import blocks_per_request


def _hybrid_pool_full(inputs: RuleInputs) -> Finding | None:
    """On a cache with linear-attention state, a pool whose free blocks cannot hold one more
    request counts as full."""
    m = inputs.measurement
    per_request = blocks_per_request(m)
    if not (per_request and m.peak_waiting and m.kv_blocks and m.peak_kv_usage):
        return None
    usable = m.kv_blocks - 1
    free = round((1 - m.peak_kv_usage) * usable)
    if free >= per_request:
        return None
    evidence = (
        f"highest sampled KV-cache usage {m.peak_kv_usage:.0%}: {free} of {usable:.0f} "
        f"cache blocks free, fewer than the {per_request} each running request held, so "
        "no waiting request could start"
    )
    return finding_at("kv_capacity", inputs.levels["kv_peak"], 1.0, evidence)


ENGINE_RULES = (EngineRule(bottleneck="kv_capacity", rule=_hybrid_pool_full),)
