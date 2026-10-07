"""Where vLLM exports the signals Tensward reads: its Prometheus families and info-gauge labels."""

from __future__ import annotations

from ...prometheus import Family, SignalMap

VLLM_SIGNALS: SignalMap = {
    "kv_usage": Family("vllm:kv_cache_usage_perc"),
    "running": Family("vllm:num_requests_running"),
    "waiting": Family("vllm:num_requests_waiting"),
    "preemptions": Family("vllm:num_preemptions_total"),
    "prefix_cache_hits": Family("vllm:prefix_cache_hits_total"),  # prompt TOKENS, not blocks
    "prefix_cache_queries": Family("vllm:prefix_cache_queries_total"),
    "prompt_tokens": Family("vllm:prompt_tokens_total"),
    "prompt_tokens_computed": Family(
        "vllm:prompt_tokens_by_source_total", labels={"source": "local_compute"}
    ),
    "generation_tokens": Family("vllm:generation_tokens_total"),
    "iterations": Family("vllm:iteration_tokens_total_count"),  # steps, from a histogram
    # An info gauge (value 1) whose labels are CacheConfig's fields (v0.30
    # CacheConfig.metrics_info). kv_cache_size_tokens is group-aware, so right for hybrid
    # models, where num_gpu_blocks x block_size is not.
    "kv_capacity_tokens": Family(
        "vllm:cache_config_info", kind="info_label", label="kv_cache_size_tokens"
    ),
    "kv_max_concurrency": Family(
        "vllm:cache_config_info", kind="info_label", label="kv_cache_max_concurrency"
    ),
    "kv_blocks": Family("vllm:cache_config_info", kind="info_label", label="num_gpu_blocks"),
    "kv_block_tokens": Family("vllm:cache_config_info", kind="info_label", label="block_size"),
    "hybrid_cache": Family("vllm:cache_config_info", kind="info_flag", label="mamba_block_size"),
    "queue_seconds": Family("vllm:request_queue_time_seconds_sum"),
    "prefill_seconds": Family("vllm:request_prefill_time_seconds_sum"),
    "spec_drafts": Family("vllm:spec_decode_num_drafts_total"),
    "spec_accepted_tokens": Family("vllm:spec_decode_num_accepted_tokens_total"),
    "frontend_cpu_seconds": Family("process_cpu_seconds_total"),
}
