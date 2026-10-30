"""vLLM's value rules bound to their actions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from ...kb_actions import Action, Built, Inputs
from ...settings import Settings
from . import value_rules
from .command import SPECULATIVE_CONFIG_FLAG
from .flags import merge_json

SUFFIX_ACCEPTANCE = 1.8  # tokens per step a suffix tree must expect before it is offered


def _flags(settings: Settings, flags: Mapping[str, Any]) -> Settings:
    return replace(settings, extra_args=merge_json(settings.extra_args, flags))


def _spec_tiers(inputs: Inputs, settings: Settings) -> Built:
    running = inputs.fact("run.running_p90")
    if running is None:
        return "the running count's 90th percentile is not known"
    speculation = inputs.engine.lever_value(settings, "speculation")
    if speculation is None or not speculation.draft_tokens:
        return "no speculative method is set to schedule: a draft model, an MTP or an EAGLE3 head"
    batch = settings.max_concurrent_requests or inputs.cap or 0
    tiers = value_rules.spec_tiers(running, speculation.draft_tokens, cut=None, max_batch=batch)
    if tiers is None:
        most = value_rules.SPEC_RUNNING_P90
        return f"{running:g} requests run at the 90th percentile, above {most}"
    name = f"{SPECULATIVE_CONFIG_FLAG}.num_speculative_tokens_per_batch_size"
    return _flags(settings, {name: tiers})


def _long_prefill_threshold(inputs: Inputs, settings: Settings) -> Built:
    if not inputs.prompt_tokens:
        return "the prompt lengths are not known"
    threshold = value_rules.long_prefill_threshold(inputs.prompt_tokens)
    if threshold is None:
        return "the prompt lengths do not fall in two classes"
    return _flags(settings, {"--long-prefill-token-threshold": str(threshold)})


def _suffix_acceptance(inputs: Inputs, settings: Settings) -> Built:
    accepted = value_rules.suffix_acceptance(inputs.answers())
    if accepted is None:
        return "this run kept no answers to replay"
    if accepted < SUFFIX_ACCEPTANCE:
        return (
            f"a suffix tree over this run's answers expects {accepted:.2f} words per step, "
            f"under {SUFFIX_ACCEPTANCE}"
        )
    kept = {k: v for k, v in settings.extra_args.items() if k != SPECULATIVE_CONFIG_FLAG}
    written = json.dumps({"method": "suffix"})
    return replace(settings, extra_args={**kept, SPECULATIVE_CONFIG_FLAG: written})


def _fp8_exclusions(inputs: Inputs, settings: Settings) -> Built:
    layers = inputs.fact("anatomy.layers")
    if layers is None:
        return "the checkpoint's layer count is not known"
    return _flags(settings, {"--quantization-config": value_rules.fp8_exclusions(layers)})


def _per_token_head_dtype(inputs: Inputs, settings: Settings) -> Built:
    dtype = value_rules.per_token_head_dtype(inputs.fact("gpu.cc"))
    if dtype is None:
        return "the GPU's compute capability is not known"
    return replace(settings, kv_cache_dtype=dtype)


def _cpu_kv_offload_size(inputs: Inputs, settings: Settings) -> Built:
    host = inputs.fact("host.memory_gib")
    if host is None:
        return "the host's memory is not known"
    possible = inputs.fact("workload.possible_prefix_hit")
    hit = inputs.fact("run.prefix_hit_rate")
    gap = None if possible is None or hit is None else possible - hit
    size = value_rules.cpu_kv_offload_size(host, gap)
    if size is None:
        if gap is None:
            return "the prefix hit rate is not known"
        if gap < value_rules.OFFLOAD_PREFIX_GAP:
            least = value_rules.OFFLOAD_PREFIX_GAP
            return (
                f"the prompts allow a prefix hit rate {gap * 100:.1f} points above the measured "
                f"one; at least {least * 100:.0f} are needed"
            )
        return f"the host has {host:.0f} GiB, too little for a CPU cache beside the rest"
    return _flags(settings, {"--kv-offloading-size": f"{size:g}"})


VLLM_ACTIONS: Mapping[str, Action] = {
    "vllm:spec_tiers": Action(frozenset(), _spec_tiers),
    "vllm:long_prefill_threshold": Action(frozenset(), _long_prefill_threshold),
    "vllm:suffix_acceptance": Action(frozenset({"speculation"}), _suffix_acceptance),
    "vllm:fp8_exclusions": Action(frozenset(), _fp8_exclusions),
    "vllm:per_token_head_dtype": Action(frozenset({"kv_cache_precision"}), _per_token_head_dtype),
    "vllm:cpu_kv_offload_size": Action(frozenset(), _cpu_kv_offload_size),
}
if unbound := sorted(set(VLLM_ACTIONS) ^ set(value_rules.RULES)):
    raise ImportError(f"value rules and their actions differ: {', '.join(unbound)}")
