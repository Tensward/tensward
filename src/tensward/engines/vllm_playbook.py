"""vLLM's playbook: the setting changes Tensward suggests for vLLM,
each tied to the bottlenecks it addresses."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Sequence

from ..classify import queued_at_cap, speculation_crossed
from ..playbook import Entry, Gate, fitting_max_context_len, round_up_context
from ..thresholds import (
    KV_PEAK,
    MIN_PREEMPTIONS,
    QUEUE_SHARE,
    TPOT_TAIL,
    TTFT_OVER_TPOT,
)
from .protocol import Settings
from .vllm import SPECULATIVE_CONFIG_FLAG

if TYPE_CHECKING:
    from ..measurement import Measurement
    from ..playbook import WorkloadFacts

KV_HEADROOM_USAGE = 0.8  # more sequences only help below this KV usage
KV_TARGET_USAGE = 0.85  # a raised concurrency is sized to keep projected KV usage under this
RAISED_KV_MEMORY_FRACTION = 0.95
RAISED_PREFILL_BATCH_TOKENS = 8192
ASSUMED_DEFAULT_PREFILL_BATCH_TOKENS = 2048  # vLLM 0.30 serving default below 70 GB GPUs
MIN_PREFILL_BATCH_TOKENS = 512  # smaller chunks cost more in step overhead than they save
SPECULATION_MIN_PROMPT_WORDS = 256  # median prompt length at which copying spans is plausible
NGRAM_SPECULATION_FLAG = SPECULATIVE_CONFIG_FLAG
NGRAM_SPECULATION = '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4}'

# Prefix caching pays when many requests start with the same text. A prompt counts as sharing
# when it has a common prefix with another prompt that is long enough to cache (one 16-token
# KV block) and either 256 tokens or a quarter of its own length. Tokens are counted as
# whitespace-separated words, which never overstates a prefix.
MIN_CACHEABLE_PREFIX_TOKENS = 16
MIN_SHARED_PREFIX_TOKENS = 256
MIN_SHARED_PREFIX_FRACTION = 0.25
MIN_SHARING_PROMPTS = 0.20  # share of the prompts that must share a prefix


def _above(value: float | None, threshold: float) -> bool:
    return value is not None and value > threshold


def _preemptions(count: float) -> str:
    return f"{count:.0f} preemption{'' if round(count) == 1 else 's'}"


def _kv_bound(signals: Measurement) -> str | None:
    if signals.preemptions is not None and signals.preemptions >= MIN_PREEMPTIONS:
        return _preemptions(signals.preemptions)
    if _above(signals.peak_kv_usage, KV_PEAK.warning):
        return f"highest sampled KV-cache usage {signals.peak_kv_usage:.0%}"
    return None


# --- memory --------------------------------------------------------------------------


def _more_kv_memory_applies(
    signals: Measurement, facts: WorkloadFacts, settings: Settings
) -> str | None:
    evidence = _kv_bound(signals)
    fraction = settings.kv_memory_fraction
    if evidence and (fraction is None or fraction < RAISED_KV_MEMORY_FRACTION):
        current = "the engine default" if fraction is None else f"{fraction:g}"
        return (
            f"{evidence}; kv_memory_fraction is {current}. A larger share leaves the GPU less "
            "memory headroom, so the engine may fail to start or run out of memory"
        )
    return None


def _trim_max_context_len_applies(
    signals: Measurement, facts: WorkloadFacts, settings: Settings
) -> str | None:
    evidence = _kv_bound(signals)
    if not evidence or signals.max_prompt_tokens is None:
        return None
    needed = _trimmed_max_context_len(signals, facts)
    if signals.max_context_len and needed < signals.max_context_len:
        return (
            f"{evidence}; the workload needs at most {needed} tokens, "
            f"max_context_len is {signals.max_context_len}; requests longer than that would "
            "be refused"
        )
    return None


def _trimmed_max_context_len(signals: Measurement, facts: WorkloadFacts) -> int:
    return round_up_context((signals.max_prompt_tokens or 0) + facts.output_tokens)


def _fit_context_applies(
    signals: Measurement, facts: WorkloadFacts, settings: Settings
) -> str | None:
    if fitting_max_context_len(signals, facts) is None:
        return None
    longest = (signals.max_prompt_tokens or 0) + facts.output_tokens
    return (
        f"prompts that do not fit max_context_len {signals.max_context_len}: "
        f"{len(signals.too_long)}; the longest needs {longest} tokens"
    )


def _fp8_kv_cache_applies(
    signals: Measurement, facts: WorkloadFacts, settings: Settings
) -> str | None:
    evidence = _kv_bound(signals)
    if evidence and not (settings.kv_cache_dtype or "").startswith("fp8"):
        return f"{evidence}; a fp8 KV cache holds about twice the tokens"
    return None


def _no_media_encoders_applies(
    signals: Measurement, facts: WorkloadFacts, settings: Settings
) -> str | None:
    if facts.media_encoders and not facts.sends_images and settings.media_inputs is None:
        return (
            "the checkpoint takes images and video but no prompt sends any; serving only its "
            "text model stops the engine reserving memory for the media encoders and lifts the "
            "minimum batch size they impose (requests with images would then be rejected)"
        )
    return None


# --- tool calling --------------------------------------------------------------------


def _enable_tool_calling_applies(
    signals: Measurement, facts: WorkloadFacts, settings: Settings
) -> str | None:
    if facts.offers_tools and not settings.tool_calling and settings.tool_parser is not None:
        return (
            "the workload offers tools but tool calling is off, so the server refuses them; "
            f"the checkpoint's tool parser is {settings.tool_parser}"
        )
    return None


# --- quantization ---------------------------------------------------------------------


def _fast_quant_kernel_applies(
    signals: Measurement, facts: WorkloadFacts, settings: Settings
) -> str | None:
    slow = [kernel.name for kernel in signals.quant_kernels if kernel.slow]
    if slow and settings.quantization is not None:
        return (
            f"the engine selected a slower quantization path ({', '.join(slow)}) while "
            f"quantization is forced to {settings.quantization}; the engine's own choice "
            "may be faster"
        )
    return None


# --- batching ------------------------------------------------------------------------


def _demand(signals: Measurement, current: int) -> int:
    """The client's in-flight peak when known, else running plus waiting requests at their
    peaks. Engines export these gauges as floats (vLLM reports "8.0")."""
    if signals.peak_in_flight is not None:
        return signals.peak_in_flight
    return round(signals.peak_running or current) + round(signals.peak_waiting or 0)


def _raised_concurrency(signals: Measurement, current: int) -> int:
    """The observed demand (the client's in-flight peak when known, else running plus waiting),
    at least double, within the KV headroom.

    Powers of two; KV usage is assumed to grow in proportion to the running requests.
    """
    running = round(signals.peak_running or current)
    demand = _demand(signals, current)
    target = 1 << (max(demand, 2 * current) - 1).bit_length()
    if signals.peak_kv_usage:
        fits = int(running * KV_TARGET_USAGE / signals.peak_kv_usage)
        target = min(target, 1 << (max(fits, 1).bit_length() - 1))
    return target


def _concurrency_cap(signals: Measurement, settings: Settings) -> int | None:
    """The concurrency limit; when the engine chose it, the most that ever ran at once."""
    if settings.max_concurrent_requests:
        return settings.max_concurrent_requests
    return None if signals.peak_running is None else int(signals.peak_running)


def _raise_concurrency_applies(
    signals: Measurement, facts: WorkloadFacts, settings: Settings
) -> str | None:
    if settings.max_concurrent_requests is None or queued_at_cap(signals, settings) is False:
        return None
    cap = _concurrency_cap(signals, settings)
    if (
        cap is not None
        and signals.peak_running is not None
        and signals.peak_running >= cap
        and _above(signals.peak_waiting, 0)
        and signals.peak_kv_usage is not None
        and signals.peak_kv_usage < KV_HEADROOM_USAGE
        and _raised_concurrency(signals, cap) > cap
    ):
        observed = (
            f"up to {_demand(signals, cap)} requests were in flight"
            if signals.peak_in_flight is not None
            else f"highest sampled running plus waiting was {_demand(signals, cap)}"
        )
        return (
            f"running requests hit max_concurrent_requests {cap} with "
            f"{signals.peak_waiting:.0f} waiting, and the highest sampled KV-cache usage is only "
            f"{signals.peak_kv_usage:.1%}; at your declared load of {facts.load}, "
            f"{observed}, so "
            f"{_raised_concurrency(signals, cap)} is worth trying (the load is what your "
            "configuration declares, not measured traffic). It usually cuts queueing (TTFT) but "
            "slows each token (TPOT) as more sequences share every step - measure both"
        )
    return None


def _lowered_concurrency(cap: int) -> int:
    """Half the cap, the step that frees KV cache when sequences are being preempted."""
    return max(cap // 2, 1)


def _lower_concurrency_applies(
    signals: Measurement, facts: WorkloadFacts, settings: Settings
) -> str | None:
    cap = _concurrency_cap(signals, settings) or 0
    if cap <= 1:
        return None
    if signals.preemptions is not None and signals.preemptions >= MIN_PREEMPTIONS:
        return f"{_preemptions(signals.preemptions)}; fewer concurrent sequences fit the KV cache"
    return None


def _raise_prefill_batch_applies(
    signals: Measurement, facts: WorkloadFacts, settings: Settings
) -> str | None:
    if signals.queue_share is not None and signals.queue_share >= QUEUE_SHARE.warning:
        return None
    if (
        signals.ttft_p50_ms is not None
        and signals.tpot_p50_ms
        and signals.ttft_p50_ms > TTFT_OVER_TPOT.warning * signals.tpot_p50_ms
        and _above(signals.peak_waiting, 0)
        and (settings.prefill_batch_tokens or 0) < RAISED_PREFILL_BATCH_TOKENS
    ):
        return (
            f"TTFT p50 {signals.ttft_p50_ms:.0f} ms is over {TTFT_OVER_TPOT.warning:g}x "
            f"TPOT p50 {signals.tpot_p50_ms:.1f} ms with requests waiting (prefill-bound). "
            "Larger prefill chunks usually cut TTFT but can stall running decodes (TPOT p95) "
            "- measure both"
        )
    return None


def _lowered_prefill_batch(settings: Settings) -> int:
    return (settings.prefill_batch_tokens or ASSUMED_DEFAULT_PREFILL_BATCH_TOKENS) // 2


def _lower_prefill_batch_applies(
    signals: Measurement, facts: WorkloadFacts, settings: Settings
) -> str | None:
    if (
        signals.tpot_p95_ms is not None
        and signals.tpot_p50_ms
        and signals.tpot_p95_ms > TPOT_TAIL.warning * signals.tpot_p50_ms
        and _lowered_prefill_batch(settings) >= MIN_PREFILL_BATCH_TOKENS
    ):
        return (
            f"TPOT p95 {signals.tpot_p95_ms:.1f} ms is over {TPOT_TAIL.warning:g}x "
            f"p50 {signals.tpot_p50_ms:.1f} ms (decode stalled by prefill). "
            "Smaller prefill chunks usually smooth TPOT but slow prompt processing (TTFT) - "
            "measure both"
        )
    return None


def _ngram_speculation_applies(
    signals: Measurement, facts: WorkloadFacts, settings: Settings
) -> str | None:
    lengths = sorted(len(prompt.split()) for prompt in facts.prompts)
    median = lengths[len(lengths) // 2] if lengths else 0
    if (
        NGRAM_SPECULATION_FLAG not in settings.extra_args
        and signals.tpot_p50_ms is not None
        and median >= SPECULATION_MIN_PROMPT_WORDS
    ):
        return (
            f"prompts are long (median {median} words). N-gram speculation drafts tokens by "
            "copying spans of the prompt, so it only helps when answers quote or repeat their "
            "input; otherwise it costs a little - compare TPOT with and without it"
        )
    return None


def _drop_speculation_applies(
    signals: Measurement, facts: WorkloadFacts, settings: Settings
) -> str | None:
    if NGRAM_SPECULATION_FLAG in settings.extra_args and speculation_crossed(signals):
        numbers = []
        if signals.spec_acceptance_length is not None:
            length = signals.spec_acceptance_length
            numbers.append(f"each speculative draft yielded {length:.2f} tokens")
        if signals.spec_coverage is not None:
            numbers.append(
                f"accepted drafts made {signals.spec_coverage:.0%} of the generated tokens"
            )
        return (
            f"{' and '.join(numbers)}, so the drafts cost more than they save; remove "
            "`--speculative-config` from your command and compare TPOT"
        )
    return None


# --- caching and graphs --------------------------------------------------------------


def _common_words(first: Sequence[str], second: Sequence[str]) -> int:
    count = 0
    for a, b in zip(first, second):
        if a != b:
            break
        count += 1
    return count


def _shared_prefixes(prompts: Sequence[str]) -> tuple[int, int]:
    """How many prompts share a cacheable prefix with another prompt, and the longest one.

    The longest prefix a prompt shares with any other is the one it shares with a neighbour
    once the prompts are sorted, so prompts are compared in groups, not against one global
    prefix. The prefix is measured in words.
    """
    ordered = sorted(prompt.split() for prompt in prompts)
    sharing = longest = 0
    for index, words in enumerate(ordered):
        neighbours = ordered[max(index - 1, 0) : index] + ordered[index + 1 : index + 2]
        prefix = max((_common_words(words, other) for other in neighbours), default=0)
        if prefix >= MIN_CACHEABLE_PREFIX_TOKENS and (
            prefix >= MIN_SHARED_PREFIX_TOKENS or prefix >= MIN_SHARED_PREFIX_FRACTION * len(words)
        ):
            sharing += 1
            longest = max(longest, prefix)
    return sharing, longest


def _prefix_caching_applies(
    signals: Measurement, facts: WorkloadFacts, settings: Settings
) -> str | None:
    total = len(facts.prompts)
    if settings.prefix_caching is not False or not total:
        return None
    sharing, longest = _shared_prefixes(facts.prompts)
    if sharing >= MIN_SHARING_PROMPTS * total:
        return (
            f"{sharing} of {total} prompts ({sharing / total:.0%}) share a prompt prefix of "
            f"up to {longest} words with another prompt"
        )
    return None


def _cuda_graphs_applies(
    signals: Measurement, facts: WorkloadFacts, settings: Settings
) -> str | None:
    if not settings.cuda_graphs:
        return (
            "CUDA graphs are disabled; enabling them usually speeds decoding at the cost of "
            "longer startup and some GPU memory"
        )
    return None


THROUGHPUT = frozenset({"throughput", "goodput", "req_s"})
LATENCY = frozenset({"ttft", "tpot"})

MEDIA_BUDGET = Gate(
    "an image+text engine refuses a token budget smaller than one media item; serve only the "
    "text model first with no-media-encoders",
    lambda facts, settings: facts.media_encoders and settings.media_inputs is not False,
)
QWEN2_FP8_KV = Gate(
    "Qwen2 and Qwen2.5 lose their answers with a fp8 KV cache: their outlier keys cannot be "
    "rescaled, and Tensward's own run garbled every answer",
    lambda facts, settings: facts.model_type in ("qwen2", "qwen2_vl", "qwen2_5_vl"),
)
PLAYBOOK: tuple[Entry, ...] = (
    Entry(
        "enable-tool-calling",
        frozenset({"fit"}),
        _enable_tool_calling_applies,
        lambda s, f, cur: replace(cur, tool_calling=True),
        "ours",
        "none; the server starts accepting tool requests",
    ),
    Entry(
        "fast-quant-kernel",
        frozenset({"prefill", "decode_bandwidth"}),
        _fast_quant_kernel_applies,
        lambda s, f, cur: replace(cur, quantization=None),
        "moderate",
        "none expected",
        helps=THROUGHPUT | LATENCY,
    ),
    Entry(
        "fit-context",
        frozenset({"fit"}),
        _fit_context_applies,
        lambda s, f, cur: replace(cur, max_context_len=fitting_max_context_len(s, f)),
        "strong",
        "more KV cache per request, so fewer fit at once",
    ),
    Entry(
        "more-kv-memory",
        frozenset({"kv_capacity"}),
        _more_kv_memory_applies,
        lambda s, f, cur: replace(cur, kv_memory_fraction=RAISED_KV_MEMORY_FRACTION),
        "strong",
        "less memory headroom: the engine may fail to start",
        helps=THROUGHPUT | LATENCY,
        steps_on="kv_memory_fraction",
    ),
    Entry(
        "no-media-encoders",
        frozenset({"kv_capacity"}),
        _no_media_encoders_applies,
        lambda s, f, cur: replace(cur, media_inputs=False),
        "ours",
        "requests with images are rejected",
        helps=THROUGHPUT | LATENCY,
    ),
    Entry(
        "trim-max-context-len",
        frozenset({"kv_capacity"}),
        _trim_max_context_len_applies,
        lambda s, f, cur: replace(cur, max_context_len=_trimmed_max_context_len(s, f)),
        "strong",
        "longer requests are refused",
        helps=THROUGHPUT,
        steps_on="max_context_len",
        start=lambda s, cur: s.max_context_len,
    ),
    Entry(
        "raise-concurrency",
        frozenset({"queueing"}),
        _raise_concurrency_applies,
        lambda s, f, cur: replace(
            cur, max_concurrent_requests=_raised_concurrency(s, _concurrency_cap(s, cur) or 1)
        ),
        "strong",
        "TPOT rises as more sequences share each step; more KV cache in use",
        helps=THROUGHPUT | {"ttft"},
        steps_on="max_concurrent_requests",
        start=_concurrency_cap,
    ),
    Entry(
        "lower-concurrency",
        frozenset({"kv_capacity", "gpu_compute"}),
        _lower_concurrency_applies,
        lambda s, f, cur: replace(
            cur, max_concurrent_requests=_lowered_concurrency(_concurrency_cap(s, cur) or 2)
        ),
        "strong",
        "requests queue if the load grows (TTFT up, throughput down)",
        helps=LATENCY,
    ),
    Entry(
        "raise-prefill-batch",
        frozenset({"prefill"}),
        _raise_prefill_batch_applies,
        lambda s, f, cur: replace(cur, prefill_batch_tokens=RAISED_PREFILL_BATCH_TOKENS),
        "moderate",
        "TPOT p95 rises as larger chunks stall running decodes",
        helps=frozenset({"throughput", "ttft"}),
        steps_on="prefill_batch_tokens",
        start=lambda s, cur: ASSUMED_DEFAULT_PREFILL_BATCH_TOKENS,
    ),
    Entry(
        "lower-prefill-batch",
        frozenset({"prefill_stalls_decode"}),
        _lower_prefill_batch_applies,
        lambda s, f, cur: replace(cur, prefill_batch_tokens=_lowered_prefill_batch(cur)),
        "strong",
        "TTFT rises as prompts prefill in smaller chunks",
        gates=(MEDIA_BUDGET,),
        helps=frozenset({"tpot"}),
    ),
    Entry(
        "ngram-speculation",
        frozenset({"decode_bandwidth"}),
        _ngram_speculation_applies,
        lambda s, f, cur: replace(
            cur, extra_args={**cur.extra_args, NGRAM_SPECULATION_FLAG: NGRAM_SPECULATION}
        ),
        "moderate",
        "TPOT rises when answers do not copy the prompt, and at high concurrency; a slower start",
        helps=frozenset({"tpot"}),
    ),
    Entry(
        "prefix-caching",
        frozenset({"prefill"}),
        _prefix_caching_applies,
        lambda s, f, cur: replace(cur, prefix_caching=True),
        "strong",
        "a little GPU memory for the cache",
        helps=THROUGHPUT | {"ttft"},
    ),
    Entry(
        "cuda-graphs",
        frozenset({"host_overhead"}),
        _cuda_graphs_applies,
        lambda s, f, cur: replace(cur, cuda_graphs=True),
        "strong",
        "a longer start-up and some GPU memory",
        helps=THROUGHPUT | LATENCY,
    ),
    Entry(
        "drop-speculation",
        frozenset({"speculation"}),
        _drop_speculation_applies,
        lambda s, f, cur: replace(
            cur, extra_args={k: v for k, v in cur.extra_args.items() if k != NGRAM_SPECULATION_FLAG}
        ),
        "strong",
        "TPOT rises again for prompts whose drafts were accepted",
    ),
    Entry(
        "fp8-kv-cache",
        frozenset({"kv_capacity"}),
        _fp8_kv_cache_applies,
        lambda s, f, cur: replace(cur, kv_cache_dtype="fp8"),
        "strong",
        (
            "answers may change, and Qwen2 and Qwen2.5 lose theirs; speed changed little in "
            "Tensward's runs on an L4 and an A10G"
        ),
        gates=(QWEN2_FP8_KV,),
        quality_risk=True,
        helps=THROUGHPUT,
    ),
)
