"""vLLM's playbook: the setting changes Tensward suggests for vLLM,
each tied to the bottlenecks it addresses."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

from ...playbook import Entry, Gate, HeldBack
from ...playbook_common import (
    ENABLE_TOOL_CALLING,
    FAST_QUANT_KERNEL,
    FIT_CONTEXT,
    LATENCY,
    LOWER_CONCURRENCY,
    NO_MEDIA_ENCODERS,
    PREFIX_CACHING,
    RAISE_CONCURRENCY,
    THROUGHPUT,
    TRIM_MAX_CONTEXT_LEN,
    above,
    kv_bound,
    raise_concurrency_blocked,
    shared_lengths,
    text_only_helps,
    trimmed_max_context_len,
)
from ...settings import Settings
from ...thresholds import TPOT_TAIL, TTFT_OVER_TPOT
from .command import DEFAULT_PREFILL_BATCH_TOKENS, SPECULATIVE_CONFIG_FLAG
from .graphs import bounded_graphs

if TYPE_CHECKING:
    from ...measurement import Measurement
    from ...playbook import Situation, WorkloadFacts

# Memory CUDA graph capture needs beyond one full-length request's KV cache. Measured with vLLM
# 0.30 on a hybrid 27B INT4 model on a 22 GiB A10G: the KV pool lost 0.22 GiB (capture sizes up
# to 16) and 0.26 GiB (up to 32), and the engine would not start with 0.05 GiB to spare. On a
# 0.5B model on an L4 the reserve was 0.08 GiB, so the share is a ceiling, not a fit.
GRAPH_RESERVE_MIN_GIB = 0.25
GRAPH_RESERVE_SHARE = 0.012  # of the GPU's memory, as the engine's own log reports it
DEFAULT_KV_FRACTION = 0.9  # vLLM's gpu_memory_utilization when none is set
RAISED_KV_MEMORY_FRACTION = 0.95
RAISED_PREFILL_BATCH_TOKENS = 8192
MIN_PREFILL_BATCH_TOKENS = 512  # smaller chunks cost more in step overhead than they save
SPECULATION_MIN_PROMPT_WORDS = 256  # median prompt length at which copying spans is plausible
NGRAM_CONFIG = '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4}'


# --- memory --------------------------------------------------------------------------


def _more_kv_memory_applies(situation: Situation) -> str | None:
    evidence = kv_bound(situation)
    if evidence and (current := _kv_fraction_raisable(situation.settings)):
        return (
            f"{evidence}; kv_memory_fraction is {current}. A larger share leaves the GPU less "
            "memory headroom, so the engine may fail to start or run out of memory"
        )
    return None


def _kv_fraction_raisable(settings: Settings) -> str | None:
    """The current kv_memory_fraction, in words, when it can be raised."""
    fraction = settings.kv_memory_fraction
    if fraction is None:
        return "the engine default"
    return f"{fraction:g}" if fraction < RAISED_KV_MEMORY_FRACTION else None


def _more_kv_memory_unblocks(situation: Situation) -> str | None:
    settings = situation.settings
    current = _kv_fraction_raisable(settings)
    if current is None or (blocked := raise_concurrency_blocked(situation)) is None:
        return None
    fraction = settings.kv_memory_fraction
    gain = (
        f"{RAISED_KV_MEMORY_FRACTION - fraction:.0%} more of the GPU's memory"
        if fraction is not None
        else "more of the GPU's memory"
    )
    return (
        f"{blocked}; kv_memory_fraction is {current}. Raising it to {RAISED_KV_MEMORY_FRACTION:g} "
        f"gives the engine {gain} for the KV cache, which may not be enough to raise the cap, "
        "and leaves the GPU less memory headroom, so the engine may fail to start or run out "
        "of memory"
    )


def _fp8_kv_cache_applies(situation: Situation) -> str | None:
    evidence = kv_bound(situation)
    if evidence and not (situation.settings.kv_cache_dtype or "").startswith("fp8"):
        return f"{evidence}; a fp8 KV cache holds about twice the tokens"
    return None


# --- batching ------------------------------------------------------------------------


def _prefill_batch_raisable(situation: Situation) -> bool:
    """Requests wait, neither for scheduling nor for KV-cache space, and the chunk is below the
    raised size."""
    return (
        not (situation.fired("queueing") or situation.fired("kv_capacity"))
        and above(situation.measurement.peak_waiting, 0)
        and (situation.settings.prefill_batch_tokens or 0) < RAISED_PREFILL_BATCH_TOKENS
    )


def _ratio_text(ratio: float, gate: float, digits: int) -> str:
    """``ratio`` to ``digits`` places, with more when rounding would make it read as the gate."""
    text = f"{ratio:.{digits}f}"
    return text if float(text) < gate else f"{ratio - 0.5 * 10 ** -(digits + 1):.{digits + 1}f}"


def _raise_prefill_batch_applies(situation: Situation) -> str | None:
    signals = situation.measurement
    if situation.fired("prefill") and _prefill_batch_raisable(situation):
        return (
            f"TTFT p50 {signals.ttft_p50_ms:.0f} ms is over {TTFT_OVER_TPOT.warning:g}x "
            f"TPOT p50 {signals.tpot_p50_ms:.1f} ms with requests waiting (prefill-bound). "
            "Larger prefill chunks usually cut TTFT but can stall running decodes (TPOT p95) "
            "- measure both"
        )
    return None


def _raise_prefill_batch_near(situation: Situation) -> str | None:
    signals = situation.measurement
    if not (signals.ttft_p50_ms is not None and signals.tpot_p50_ms):
        return None
    ratio = signals.ttft_p50_ms / signals.tpot_p50_ms
    if situation.finding("prefill").near and _prefill_batch_raisable(situation):
        return (
            f"TTFT p50 is {_ratio_text(ratio, TTFT_OVER_TPOT.warning, 0)}x TPOT p50 with "
            f"requests waiting; the gate is {TTFT_OVER_TPOT.warning:.0f}x. Larger prefill chunks "
            "usually cut TTFT but can stall running decodes (TPOT p95) - measure both"
        )
    return None


def _lowered_prefill_batch(settings: Settings) -> int:
    return (settings.prefill_batch_tokens or DEFAULT_PREFILL_BATCH_TOKENS) // 2


def _lower_prefill_batch_applies(situation: Situation) -> str | None:
    signals = situation.measurement
    if (
        situation.fired("prefill_stalls_decode")
        and _lowered_prefill_batch(situation.settings) >= MIN_PREFILL_BATCH_TOKENS
    ):
        return (
            f"TPOT p95 {signals.tpot_p95_ms:.1f} ms is over {TPOT_TAIL.warning:g}x "
            f"p50 {signals.tpot_p50_ms:.1f} ms (decode stalled by prefill). "
            "Smaller prefill chunks usually smooth TPOT but slow prompt processing (TTFT) - "
            "measure both"
        )
    return None


def _lower_prefill_batch_near(situation: Situation) -> str | None:
    signals = situation.measurement
    if not (signals.tpot_p95_ms is not None and signals.tpot_p50_ms):
        return None
    ratio = signals.tpot_p95_ms / signals.tpot_p50_ms
    if (
        situation.finding("prefill_stalls_decode").near
        and _lowered_prefill_batch(situation.settings) >= MIN_PREFILL_BATCH_TOKENS
    ):
        return (
            f"TPOT p95 is {_ratio_text(ratio, TPOT_TAIL.warning, 1)}x p50; the gate is "
            f"{TPOT_TAIL.warning:.1f}x. Smaller prefill chunks usually smooth TPOT but slow "
            "prompt processing (TTFT) - measure both"
        )
    return None


NGRAM_COST = (
    "TPOT rises when answers do not copy the prompt, and at high concurrency; a slower start"
)
NGRAM_STRUCTURED_COST = (
    f"{NGRAM_COST}; compare the answers with `--require-equal` before adopting it"
)


def _ngram_reason(situation: Situation) -> str | None:
    prompts = list(dict.fromkeys(situation.facts.prompts))
    lengths = sorted(
        len(prompt.split()) - shared for prompt, shared in zip(prompts, shared_lengths(prompts))
    )
    median = lengths[len(lengths) // 2] if lengths else 0
    if (
        SPECULATIVE_CONFIG_FLAG not in situation.settings.extra_args
        and situation.measurement.tpot_p50_ms is not None
        and median >= SPECULATION_MIN_PROMPT_WORDS
    ):
        return (
            f"prompts are long (median {median} words beyond what they share). N-gram "
            "speculation drafts tokens by copying spans of the prompt, so it only helps when "
            "answers quote or repeat their input; otherwise it costs a little - compare TPOT "
            "with and without it"
        )
    return None


def _ngram_speculation_applies(situation: Situation) -> str | None:
    return None if situation.facts.structured else _ngram_reason(situation)


def _ngram_under_structured_output(situation: Situation) -> str | None:
    if situation.facts.structured and (reason := _ngram_reason(situation)):
        return (
            f"{reason}. This workload declares structured output: with grammar-constrained "
            "decoding the drafts are often rejected, and in one production run n-gram "
            "speculation lost half the output throughput and cut many answers short"
        )
    return None


def _drop_speculation_applies(situation: Situation) -> str | None:
    signals = situation.measurement
    if SPECULATIVE_CONFIG_FLAG in situation.settings.extra_args and situation.fired("speculation"):
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


# --- graphs --------------------------------------------------------------------------------------


def _graph_headroom(signals: Measurement, settings: Settings) -> tuple[float, float] | None:
    """(memory left after one full-length request's KV cache, memory graph capture needs), in
    GiB, when the eager run measured enough to say and the left-over is too small."""
    kv, tokens, length = (
        signals.kv_available_gib,
        signals.kv_capacity_tokens,
        signals.max_context_len,
    )
    if settings.cuda_graphs or not (kv and tokens and length and signals.gpu_memory_gib):
        return None
    spare = kv - kv * length / tokens
    reserve = max(GRAPH_RESERVE_MIN_GIB, GRAPH_RESERVE_SHARE * signals.gpu_memory_gib)
    return (spare, reserve) if spare < reserve else None


def _memory_lever(
    signals: Measurement, facts: WorkloadFacts, settings: Settings
) -> tuple[str, Settings, str] | None:
    """The first change that frees enough memory for graph capture: its name, the settings it
    makes and what it frees, in words."""
    headroom = _graph_headroom(signals, settings)
    if headroom is None:
        return None
    spare, reserve = headroom
    if text_only_helps(facts, settings) and spare + facts.encoder_gib >= reserve:
        freed = f"about {facts.encoder_gib:.2f} GiB of encoder weights"
        return "no-media-encoders", replace(settings, media_inputs=False), freed
    gpu = signals.gpu_memory_gib or 0
    if (current := settings.kv_memory_fraction or DEFAULT_KV_FRACTION) < RAISED_KV_MEMORY_FRACTION:
        if (RAISED_KV_MEMORY_FRACTION - current) * gpu >= reserve:
            raised = replace(settings, kv_memory_fraction=RAISED_KV_MEMORY_FRACTION)
            freed = f"about {(RAISED_KV_MEMORY_FRACTION - current) * gpu:.2f} GiB"
            return "more-kv-memory", raised, freed
    length, tokens = signals.max_context_len, signals.kv_capacity_tokens
    if signals.max_prompt_tokens is not None and length and tokens:
        trimmed = trimmed_max_context_len(signals, facts)
        freed_gib = (signals.kv_available_gib or 0) * (length - trimmed) / tokens
        if trimmed < length and spare + freed_gib >= reserve:
            lighter = replace(settings, max_context_len=trimmed)
            return "trim-max-context-len", lighter, f"about {freed_gib:.2f} GiB of KV cache"
    return None


def _hybrid_sequences(signals: Measurement, settings: Settings) -> int | None:
    """For a cache with linear-attention state, how many sequences graphs can serve: vLLM gives
    every decode slot one state block from the same pool as the KV cache, and refuses
    max_num_seqs above the pool. The pool shrinks by the memory capture takes, at the eager
    run's memory per block. None when the pool is not known."""
    kv, blocks = signals.kv_available_gib, signals.kv_blocks
    if not (kv and blocks and signals.gpu_memory_gib):
        return None
    reserve = max(GRAPH_RESERVE_MIN_GIB, GRAPH_RESERVE_SHARE * signals.gpu_memory_gib)
    left = int(blocks) - math.ceil(reserve / (kv / blocks))
    return min(left, settings.max_concurrent_requests or left)


def _hybrid_blocker(signals: Measurement, settings: Settings) -> str | None:
    """Why graphs are held back on a cache with linear-attention state, else None."""
    if not signals.hybrid_cache:
        return None
    sequences = _hybrid_sequences(signals, settings)
    peak = round(signals.peak_running or 0)
    base = (
        "this model mixes attention with linear-attention layers, and vLLM's CUDA graphs need "
        "max_concurrent_requests no higher than the KV cache blocks left after capture"
    )
    if sequences is None:
        return f"CUDA graphs held back: {base}, and the run did not report them"
    if sequences < 1 or sequences < peak:
        return (
            f"CUDA graphs held back: {base}. The cache has {signals.kv_blocks:.0f} blocks, "
            f"about {max(sequences, 0)} would be left, and the run reached {peak} running requests"
        )
    return None


def _hybrid_note(signals: Measurement, settings: Settings) -> str:
    sequences = _hybrid_sequences(signals, settings)
    seen = (
        f" (the run never had more than {round(signals.peak_running)} running, so the limit "
        "does not cost throughput)"
        if signals.peak_running is not None and signals.peak_running <= (sequences or 0)
        else ""
    )
    return (
        ". This model mixes attention with linear-attention layers, so graphs also need "
        f"max_concurrent_requests no higher than the cache blocks; the Try sets it to "
        f"{sequences}{seen}"
    )


def _more_api_servers_applies(situation: Situation) -> str | None:
    if not situation.fired("frontend_cpu") or situation.settings.api_server_count not in (None, 1):
        return None
    finding = situation.finding("frontend_cpu")
    return f"{finding.evidence}; a second API-server process shares tokenization and streaming"


def _cuda_graphs_applies(situation: Situation) -> str | None:
    signals, facts, settings = situation.measurement, situation.facts, situation.settings
    if settings.cuda_graphs:
        return None
    reason = (
        "CUDA graphs are disabled; enabling them usually speeds decoding at the cost of "
        "longer startup and some GPU memory"
    )
    if blocker := _hybrid_blocker(signals, settings):
        return HeldBack(blocker)
    hybrid = _hybrid_note(signals, settings) if signals.hybrid_cache else ""
    if (headroom := _graph_headroom(signals, settings)) is None:
        return reason + hybrid
    spare, need = headroom
    if (lever := _memory_lever(signals, facts, settings)) is None:
        return HeldBack(
            f"CUDA graphs need about {need:.2f} GiB beyond one full-length request's KV cache; "
            f"this card has {spare:.2f} GiB, and no change that frees memory applies"
        )
    return (
        f"{reason}. Only {spare:.2f} GiB is left after one full-length request's KV cache, and "
        f"graph capture needs about {need:.2f} GiB, so graphs alone will likely not start. "
        f"`{lever[0]}` frees {lever[2]}, which should leave enough room (an estimate, not "
        f"measured on this card){hybrid}"
    )


def _graph_cap(facts: WorkloadFacts, settings: Settings) -> int | None:
    caps = [n for n in (facts.max_inflight, settings.max_concurrent_requests) if n is not None]
    return min(caps, default=None)


def _with_cuda_graphs(situation: Situation) -> Settings:
    """Graphs on, captured only up to the most sequences a step can hold when that is known,
    with the change that frees their memory when the card has none to spare, and the sequence
    limit a cache with linear-attention state needs."""
    signals, facts, current = situation.measurement, situation.facts, situation.settings
    lever = _memory_lever(signals, facts, current)
    enabled = replace(lever[1] if lever else current, cuda_graphs=True)
    if signals.hybrid_cache and (sequences := _hybrid_sequences(signals, current)):
        enabled = replace(enabled, max_concurrent_requests=sequences)
    cap = _graph_cap(facts, enabled)
    return bounded_graphs(enabled, cap) if cap else enabled


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
MORE_KV_MEMORY = Entry(
    name="more-kv-memory",
    addresses=frozenset({"kv_capacity"}),
    applies=_more_kv_memory_applies,
    apply=lambda situation: replace(
        situation.settings, kv_memory_fraction=RAISED_KV_MEMORY_FRACTION
    ),
    evidence="strong",
    costs="less memory headroom: the engine may fail to start",
    helps=THROUGHPUT | LATENCY,
    steps_on="kv_memory_fraction",
    low_bar=_more_kv_memory_unblocks,
)
RAISE_PREFILL_BATCH = Entry(
    name="raise-prefill-batch",
    addresses=frozenset({"prefill"}),
    applies=_raise_prefill_batch_applies,
    apply=lambda situation: replace(
        situation.settings, prefill_batch_tokens=RAISED_PREFILL_BATCH_TOKENS
    ),
    evidence="moderate",
    costs="TPOT p95 rises as larger chunks stall running decodes",
    helps=frozenset({"throughput", "ttft"}),
    steps_on="prefill_batch_tokens",
    start=lambda situation: DEFAULT_PREFILL_BATCH_TOKENS,
    low_bar=_raise_prefill_batch_near,
    near=True,
)
LOWER_PREFILL_BATCH = Entry(
    name="lower-prefill-batch",
    addresses=frozenset({"prefill_stalls_decode"}),
    applies=_lower_prefill_batch_applies,
    apply=lambda situation: replace(
        situation.settings, prefill_batch_tokens=_lowered_prefill_batch(situation.settings)
    ),
    evidence="strong",
    costs="TTFT rises as prompts prefill in smaller chunks",
    gates=(MEDIA_BUDGET,),
    helps=frozenset({"tpot"}),
    low_bar=_lower_prefill_batch_near,
    near=True,
)
NGRAM_SPECULATION = Entry(
    name="ngram-speculation",
    addresses=frozenset({"decode_bandwidth"}),
    applies=_ngram_speculation_applies,
    apply=lambda situation: replace(
        situation.settings,
        extra_args={
            **situation.settings.extra_args,
            SPECULATIVE_CONFIG_FLAG: NGRAM_CONFIG,
        },
    ),
    evidence="moderate",
    costs=NGRAM_COST,
    helps=frozenset({"tpot"}),
    low_bar=_ngram_under_structured_output,
    near=True,
    low_bar_costs=NGRAM_STRUCTURED_COST,
    basis="your prompts",
)
CUDA_GRAPHS = Entry(
    name="cuda-graphs",
    addresses=frozenset({"host_overhead"}),
    applies=_cuda_graphs_applies,
    apply=_with_cuda_graphs,
    evidence="strong",
    costs="a longer start-up and some GPU memory",
    helps=THROUGHPUT | LATENCY,
    basis="your settings",
)
DROP_SPECULATION = Entry(
    name="drop-speculation",
    addresses=frozenset({"speculation"}),
    applies=_drop_speculation_applies,
    apply=lambda situation: replace(
        situation.settings,
        extra_args={
            k: v for k, v in situation.settings.extra_args.items() if k != SPECULATIVE_CONFIG_FLAG
        },
    ),
    evidence="strong",
    costs="TPOT rises again for prompts whose drafts were accepted",
)
FP8_KV_CACHE = Entry(
    name="fp8-kv-cache",
    addresses=frozenset({"kv_capacity"}),
    applies=_fp8_kv_cache_applies,
    apply=lambda situation: replace(situation.settings, kv_cache_dtype="fp8"),
    evidence="strong",
    costs=(
        "answers may change, and Qwen2 and Qwen2.5 lose theirs; speed changed little in "
        "Tensward's runs on an L4 and an A10G"
    ),
    gates=(QWEN2_FP8_KV,),
    quality_risk=True,
    helps=THROUGHPUT,
)

MORE_API_SERVERS = Entry(
    name="more-api-servers",
    addresses=frozenset({"frontend_cpu"}),
    applies=_more_api_servers_applies,
    apply=lambda situation: replace(situation.settings, api_server_count=2),
    evidence="moderate",
    costs="one more CPU process and its memory",
    helps=THROUGHPUT | LATENCY,
)

PLAYBOOK: tuple[Entry, ...] = (
    ENABLE_TOOL_CALLING,
    FAST_QUANT_KERNEL,
    FIT_CONTEXT,
    MORE_KV_MEMORY,
    NO_MEDIA_ENCODERS,
    TRIM_MAX_CONTEXT_LEN,
    RAISE_CONCURRENCY,
    LOWER_CONCURRENCY,
    RAISE_PREFILL_BATCH,
    LOWER_PREFILL_BATCH,
    NGRAM_SPECULATION,
    PREFIX_CACHING,
    CUDA_GRAPHS,
    DROP_SPECULATION,
    FP8_KV_CACHE,
    MORE_API_SERVERS,
)
