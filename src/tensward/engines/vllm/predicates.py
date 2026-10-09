"""vLLM's playbook predicates and gates: the conditions and changes its entries in
data/playbook.toml name as "vllm:<name>"."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable

from ...levers import Speculation
from ...measurement import blocks_per_request
from ...playbook import Gate, HeldBack
from ...playbook_common import (
    above,
    kv_bound,
    raise_concurrency_blocked,
    shared_lengths,
    text_only_helps,
    trimmed_max_context_len,
)
from ...settings import Settings
from ...thresholds import PROMPT_WORK, TPOT_TAIL, TTFT_OVER_TPOT
from .command import DEFAULT_KV_MEMORY_FRACTION, DEFAULT_PREFILL_BATCH_TOKENS

if TYPE_CHECKING:
    from ...measurement import Measurement
    from ...playbook import Situation, WorkloadFacts
    from ..protocol import Engine

# Memory CUDA graph capture needs beyond one full-length request's KV cache. Measured with vLLM
# 0.30 on a hybrid 27B INT4 model on a 22 GiB A10G: the KV pool lost 0.22 GiB (capture sizes up
# to 16) and 0.26 GiB (up to 32), and the engine would not start with 0.05 GiB to spare. On a
# 0.5B model on an L4 the reserve was 0.08 GiB, so the share is a ceiling, not a fit.
GRAPH_RESERVE_MIN_GIB = 0.25
GRAPH_RESERVE_SHARE = 0.012  # of the GPU's memory, as the engine's own log reports it
RAISED_KV_MEMORY_FRACTION = 0.95
RAISED_PREFILL_BATCH_TOKENS = 8192
MAX_PREFILL_BATCH_TOKENS = 16384  # the playbook's last step for max_num_batched_tokens
MIN_PREFILL_BATCH_TOKENS = 512  # smaller chunks cost more in step overhead than they save
SPECULATION_MIN_PROMPT_WORDS = 256  # median prompt length at which copying spans is plausible
NGRAM = Speculation("ngram", draft_tokens=4, lookup_max=4)


# --- memory --------------------------------------------------------------------------


HYBRID_BLOCKS = (
    "this model's cache mixes attention with linear-attention state, and vLLM sizes every "
    "cache block to the state page"
)


def _hybrid_raised_cap(situation: Situation) -> tuple[int, int, int] | None:
    """On a cache with linear-attention state whose cap held the running requests: (the blocks
    one request held, the blocks the raised memory share gives at this run's memory per block,
    the sequences those hold) when that is more than the cap; None otherwise."""
    signals, settings = situation.measurement, situation.settings
    cap, per_request = settings.max_concurrent_requests, blocks_per_request(signals)
    kv, blocks, gpu = signals.kv_available_gib, signals.kv_blocks, signals.gpu_memory_gib
    if not (cap and per_request and kv and blocks and gpu) or (signals.peak_running or 0) < cap:
        return None
    current = settings.kv_memory_fraction or DEFAULT_KV_MEMORY_FRACTION
    raised = int((kv + (RAISED_KV_MEMORY_FRACTION - current) * gpu) / (kv / blocks))
    sequences = (raised - 1) // per_request
    return (per_request, raised, sequences) if sequences > cap else None


def more_kv_memory_applies(situation: Situation, engine: Engine) -> str | None:
    evidence = kv_bound(situation)
    if not evidence:
        return None
    if not (current := _kv_fraction_raisable(situation.settings)):
        return _hybrid_at_full_memory(situation)
    reason = (
        f"{evidence}; kv_memory_fraction is {current}. A larger share leaves the GPU less "
        "memory headroom, so the engine may fail to start or run out of memory"
    )
    if (hybrid := _hybrid_raised_cap(situation)) is not None:
        per_request, raised, sequences = hybrid
        reason += (
            f". On this hybrid cache each running request held {per_request} of "
            f"{situation.measurement.kv_blocks:.0f} blocks; at {RAISED_KV_MEMORY_FRACTION:g} the "
            f"pool grows to about {raised} blocks, enough for {sequences} sequences, so the Try "
            f"raises max_concurrent_requests from {situation.settings.max_concurrent_requests} to "
            f"{sequences} (an estimate from this run's memory per block)"
        )
    return reason


def _hybrid_at_full_memory(situation: Situation) -> str | None:
    """Why a hybrid cache already at the raised share gets no memory lever, with its numbers."""
    signals = situation.measurement
    if (per_request := blocks_per_request(signals)) is None:
        return None
    usable = (signals.kv_blocks or 0) - 1
    share = situation.settings.kv_memory_fraction
    return HeldBack(
        f"kv_memory_fraction is already {share:g}; {HYBRID_BLOCKS}, and the "
        f"{usable:.0f} usable blocks hold about {int(usable // per_request)} requests of "
        f"{per_request} blocks. A smaller linear-attention state dtype would make the pages "
        "smaller; Tensward has not measured that yet, so it is not suggested"
    )


def with_more_kv_memory(situation: Situation, engine: Engine) -> Settings:
    raised = replace(situation.settings, kv_memory_fraction=RAISED_KV_MEMORY_FRACTION)
    if (hybrid := _hybrid_raised_cap(situation)) is not None:
        raised = replace(raised, max_concurrent_requests=hybrid[2])
    return raised


def _kv_fraction_raisable(settings: Settings) -> str | None:
    """The current kv_memory_fraction, in words, when it can be raised."""
    fraction = settings.kv_memory_fraction
    if fraction is None:
        return "the engine default"
    return f"{fraction:g}" if fraction < RAISED_KV_MEMORY_FRACTION else None


def more_kv_memory_unblocks(situation: Situation, engine: Engine) -> str | None:
    settings = situation.settings
    current = _kv_fraction_raisable(settings)
    if current is None or (blocked := raise_concurrency_blocked(situation, engine)) is None:
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


def fp8_kv_cache_applies(situation: Situation, engine: Engine) -> str | None:
    signals, facts = situation.measurement, situation.facts
    evidence = kv_bound(situation)
    if not evidence or engine.lever_value(situation.settings, "kv_cache_precision") == "8bit":
        return None
    reason = f"{evidence}; a fp8 KV cache holds about twice the tokens"
    block = signals.kv_block_tokens
    if not (signals.hybrid_cache and block and signals.max_prompt_tokens is not None):
        return reason
    longest = signals.max_prompt_tokens + facts.output_tokens
    if longest <= block:
        return HeldBack(
            f"{HYBRID_BLOCKS}: a fp8 KV cache doubles the tokens per attention block instead of "
            f"the blocks, and the longest request ({longest} tokens) already fits in one "
            f"{block:.0f}-token block, so the cache holds no more requests"
        )
    return (
        f"{evidence}; on this hybrid cache a fp8 KV cache doubles the tokens per attention block, "
        f"so it frees blocks only for requests longer than {block:.0f} tokens (the longest needs "
        f"{longest}); the linear-attention state is unchanged"
    )


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


def _by_prompt_work(situation: Situation) -> bool:
    return situation.finding("prefill").threshold == PROMPT_WORK.name


def _by_ttft_ratio(situation: Situation) -> bool:
    return situation.fired("prefill") and not _by_prompt_work(situation)


def raise_prefill_batch_applies(situation: Situation, engine: Engine) -> str | None:
    signals, raised = situation.measurement, raised_prefill_batch(situation)
    if situation.fired("step_budget") and raised is not None:
        return (
            f"steps carried about {signals.scheduled_tokens_per_step:.0f} tokens against a budget "
            f"of {signals.step_budget_tokens} while requests waited; a budget of {raised} lets "
            "more requests run each step"
        )
    if _by_ttft_ratio(situation) and _prefill_batch_raisable(situation):
        return (
            f"TTFT p50 {signals.ttft_p50_ms:.0f} ms is over {TTFT_OVER_TPOT.warning:g}x "
            f"TPOT p50 {signals.tpot_p50_ms:.1f} ms with requests waiting (prefill-bound). "
            "Larger prefill chunks usually cut TTFT but can stall running decodes (TPOT p95) "
            "- measure both"
        )
    return None


def raise_prefill_batch_near(situation: Situation, engine: Engine) -> str | None:
    """Near the TTFT-ratio gate; or prompt work named the bottleneck with the step budget not
    full, where a larger budget is worth measuring but not a step to try first."""
    signals = situation.measurement
    share = signals.prompt_work_share
    if (
        situation.fired("prefill")
        and _by_prompt_work(situation)
        and share is not None
        and not situation.fired("kv_capacity")
        and above(signals.peak_waiting, 0)
    ):
        return (
            f"prompt work takes {share:.0%} of engine time with requests waiting, but steps are "
            "not full; a larger budget lets more prompt tokens into each step and can stall "
            "running decodes (TPOT p95) - measure TTFT and TPOT p95"
        )
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


def lower_prefill_batch_applies(situation: Situation, engine: Engine) -> str | None:
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


def lower_prefill_batch_near(situation: Situation, engine: Engine) -> str | None:
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


def _ngram_reason(situation: Situation, engine: Engine) -> str | None:
    prompts = list(dict.fromkeys(situation.facts.prompts))
    lengths = sorted(
        len(prompt.split()) - shared for prompt, shared in zip(prompts, shared_lengths(prompts))
    )
    median = lengths[len(lengths) // 2] if lengths else 0
    if (
        engine.lever_value(situation.settings, "speculation") is None
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


def ngram_speculation_applies(situation: Situation, engine: Engine) -> str | None:
    if situation.facts.structured or situation.fired("step_budget"):
        return None
    return _ngram_reason(situation, engine)


def ngram_under_structured_output(situation: Situation, engine: Engine) -> str | None:
    if (
        situation.facts.structured
        and not situation.fired("step_budget")
        and (reason := _ngram_reason(situation, engine))
    ):
        return (
            f"{reason}. This workload declares structured output: with grammar-constrained "
            "decoding the drafts are often rejected, and in one production run n-gram "
            "speculation lost half the output throughput and cut many answers short"
        )
    return None


def drop_speculation_applies(situation: Situation, engine: Engine) -> str | None:
    signals = situation.measurement
    speculating = engine.lever_value(situation.settings, "speculation") is not None
    if speculating and situation.fired("speculation"):
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
    """The first change that leaves enough memory for graph capture: its name, the settings it
    makes and what it does, in words."""
    headroom = _graph_headroom(signals, settings)
    if headroom is None:
        return None
    spare, reserve = headroom
    if text_only_helps(facts, settings) and spare + facts.encoder_gib >= reserve:
        freed = f"frees about {facts.encoder_gib:.2f} GiB of encoder weights"
        return "no-media-encoders", replace(settings, media_inputs=False), freed
    gpu = signals.gpu_memory_gib or 0
    if (
        current := settings.kv_memory_fraction or DEFAULT_KV_MEMORY_FRACTION
    ) < RAISED_KV_MEMORY_FRACTION:
        if (RAISED_KV_MEMORY_FRACTION - current) * gpu >= reserve:
            raised = replace(settings, kv_memory_fraction=RAISED_KV_MEMORY_FRACTION)
            freed = f"frees about {(RAISED_KV_MEMORY_FRACTION - current) * gpu:.2f} GiB"
            return "more-kv-memory", raised, freed
    length, tokens = signals.max_context_len, signals.kv_capacity_tokens
    if signals.max_prompt_tokens is not None and length and tokens:
        trimmed = trimmed_max_context_len(signals, facts)
        freed_gib = (signals.kv_available_gib or 0) * (length - trimmed) / tokens
        if trimmed < length and spare + freed_gib >= reserve:
            lighter = replace(settings, max_context_len=trimmed)
            # vLLM refuses to start unless one full-length request fits the pool left after
            # graph capture (v1/core/kv_cache_utils.py check_enough_kv_cache_memory, f0c44cc);
            # a shorter limit lowers that need, not the pool.
            lowered = (
                f"lowers the KV cache one full-length request needs by about {freed_gib:.2f} GiB"
            )
            return "trim-max-context-len", lighter, lowered
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


def more_api_servers_applies(situation: Situation, engine: Engine) -> str | None:
    if not situation.fired("frontend_cpu") or situation.settings.api_server_count not in (None, 1):
        return None
    finding = situation.finding("frontend_cpu")
    return f"{finding.evidence}; a second API-server process shares tokenization and streaming"


def cuda_graphs_applies(situation: Situation, engine: Engine) -> str | None:
    signals, facts, settings = situation.measurement, situation.facts, situation.settings
    if engine.lever_value(settings, "graph_capture"):
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
        f"`{lever[0]}` {lever[2]}, which should leave enough room (an estimate, not "
        f"measured on this card){hybrid}"
    )


def _graph_cap(facts: WorkloadFacts, settings: Settings) -> int | None:
    caps = [n for n in (facts.max_inflight, settings.max_concurrent_requests) if n is not None]
    return min(caps, default=None)


def with_cuda_graphs(situation: Situation, engine: Engine) -> Settings:
    """Graphs on, captured only up to the most sequences a step can hold when that is known,
    with the change that frees their memory when the card has none to spare, and the sequence
    limit a cache with linear-attention state needs."""
    signals, facts, current = situation.measurement, situation.facts, situation.settings
    lever = _memory_lever(signals, facts, current)
    enabled = engine.with_lever(lever[1] if lever else current, "graph_capture", True)
    if signals.hybrid_cache and (sequences := _hybrid_sequences(signals, current)):
        enabled = replace(enabled, max_concurrent_requests=sequences)
    cap = _graph_cap(facts, enabled)
    return engine.with_lever(enabled, "graph_capture_limit", cap) if cap else enabled


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
# vLLM 0.30's xgrammar and guidance backends read disable_any_whitespace from the engine's
# structured-outputs config only (backend_xgrammar.py:38, backend_guidance.py:91).
RUNAWAY_WHITESPACE_SHARE = 0.05  # of the answers that must be JSON


def compact_json_low_bar(situation: Situation, engine: Engine) -> str | None:
    stats, settings = situation.measurement.structured_answers, situation.settings
    if (
        stats is None
        or stats.runaway_whitespace < RUNAWAY_WHITESPACE_SHARE
        or engine.with_lever(settings, "compact_json", True) == settings
    ):
        return None
    return (
        f"{stats.runaway_whitespace:.0%} of the answers that must be JSON ran on whitespace until "
        f"the token limit ({stats.invalid_json:.0%} are invalid JSON). The engine's JSON grammar "
        "allows any whitespace between tokens and a model can loop on it; compact JSON leaves no "
        "room for that"
    )


def with_compact_json(situation: Situation, engine: Engine) -> Settings:
    return engine.with_lever(situation.settings, "compact_json", True)


def with_ngram_speculation(situation: Situation, engine: Engine) -> Settings:
    return engine.with_lever(situation.settings, "speculation", NGRAM)


def without_speculation(situation: Situation, engine: Engine) -> Settings:
    return engine.with_lever(situation.settings, "speculation", None)


def with_fp8_kv_cache(situation: Situation, engine: Engine) -> Settings:
    return engine.with_lever(situation.settings, "kv_cache_precision", "8bit")


def raised_prefill_batch(situation: Situation) -> int | None:
    """Twice the budget the engine applied, at most MAX_PREFILL_BATCH_TOKENS; None when the
    budget is unknown or already at the last step."""
    budget = situation.measurement.step_budget_tokens
    if budget is None or budget >= MAX_PREFILL_BATCH_TOKENS:
        return None
    return min(2 * budget, MAX_PREFILL_BATCH_TOKENS)


def with_raised_prefill_batch(situation: Situation, engine: Engine) -> Settings:
    """The next step above the current budget. With the budget unknown, RAISED_PREFILL_BATCH_TOKENS
    when it is above the configured one; at or above the last step, the settings unchanged, so
    nothing is offered."""
    budget = situation.measurement.step_budget_tokens
    raised = raised_prefill_batch(situation)
    if raised is None and budget is None:
        raised = RAISED_PREFILL_BATCH_TOKENS
    current = max(budget or 0, situation.settings.prefill_batch_tokens or 0)
    if raised is None or raised <= current:
        return situation.settings
    return replace(situation.settings, prefill_batch_tokens=raised)


def default_prefill_batch(situation: Situation, engine: Engine) -> float | None:
    return DEFAULT_PREFILL_BATCH_TOKENS


def with_lowered_prefill_batch(situation: Situation, engine: Engine) -> Settings:
    lowered = _lowered_prefill_batch(situation.settings)
    return replace(situation.settings, prefill_batch_tokens=lowered)


def with_more_api_servers(situation: Situation, engine: Engine) -> Settings:
    return replace(situation.settings, api_server_count=2)


def trim_hold_back(situation: Situation) -> str | None:
    """Why trimming max_context_len is held back on vLLM: the KV cache takes the memory left
    after the weights and activations whatever max_model_len is, and a request takes blocks as
    it grows (vLLM 0.30, Qwen2.5-7B on an A10G: max_model_len 512 and 32768 gave pools of 70,688
    and 70,656 tokens). A hybrid cache's blocks are also sized to the state page."""
    if situation.measurement.hybrid_cache:
        return (
            f"{HYBRID_BLOCKS}, so a running request holds the same blocks whatever "
            "max_context_len is; trimming it adds no room and only refuses longer requests"
        )
    return (
        "vLLM gives the KV cache the memory left after loading the model whatever "
        "max_context_len is, and a request takes blocks only as it grows; trimming it adds no "
        "room and only refuses longer requests"
    )


def data_parallel_hold_back(situation: Situation) -> str | None:
    """Why CUDA graphs are held back under data parallelism: the cache and memory readings that
    gate them are one engine's and are not read over several."""
    replicas = situation.measurement.replicas
    if replicas <= 1:
        return None
    return (
        f"{replicas} data-parallel engines; the hybrid-cache and graph-memory checks need one "
        "engine's cache and memory, which were not read"
    )


PREDICATES: dict[str, Callable[..., Any] | Gate] = {
    **{
        f"vllm:{predicate.__name__}": predicate
        for predicate in (
            more_kv_memory_applies, with_more_kv_memory, more_kv_memory_unblocks,
            fp8_kv_cache_applies, with_fp8_kv_cache, raise_prefill_batch_applies,
            with_raised_prefill_batch, default_prefill_batch, raise_prefill_batch_near,
            lower_prefill_batch_applies, with_lowered_prefill_batch, lower_prefill_batch_near,
            ngram_speculation_applies, with_ngram_speculation, ngram_under_structured_output,
            drop_speculation_applies, without_speculation, cuda_graphs_applies, with_cuda_graphs,
            compact_json_low_bar, with_compact_json, more_api_servers_applies,
            with_more_api_servers,
        )
    },
    "vllm:media_budget": MEDIA_BUDGET,
    "vllm:qwen2_fp8_kv": QWEN2_FP8_KV,
}  # fmt: skip
HOLD_BACKS: dict[str, Callable[[Situation], str | None]] = {
    "trim-max-context-len": trim_hold_back,
    "cuda-graphs": data_parallel_hold_back,
}
