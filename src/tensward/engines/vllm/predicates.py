"""vLLM's playbook predicates and gates: the conditions and changes its entries in
data/playbook.toml name as "vllm:<name>"."""

from __future__ import annotations

import math
import statistics
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable

from ... import kvsize
from ...anatomy import freed_bytes
from ...answer_options import MARLIN_FP8_ARG, NATIVE_FP8, WEIGHT_ONLY, Format
from ...ceilings import weight_bytes_read
from ...context import current as analysis
from ...levers import Speculation
from ...measurement import blocks_per_request
from ...playbook import Gate, HeldBack
from ...playbook_common import (
    above,
    kv_bound,
    kv_held_back,
    raise_concurrency_blocked,
    running_at_cap,
    sampled_memory,
    shared_lengths,
    text_only_helps,
    this_run,
    trimmed_max_context_len,
    without_media_encoders,
)
from ...settings import Settings
from ...thresholds import PROMPT_WORK, QUEUE_SHARE, TPOT_TAIL, TTFT_OVER_TPOT
from .command import DEFAULT_KV_MEMORY_FRACTION, DEFAULT_PREFILL_BATCH_TOKENS
from .phases import prefill_ms_per_request
from .quantization import PER_CHANNEL
from .startup import memory_usage

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

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
NGRAM = Speculation("ngram_gpu", draft_tokens=4, lookup_max=4)
GRAPH_CAPTURE_TOKENS = 512  # vLLM 0.30's largest default CUDA-graph capture size, in tokens
SMALLER_STEP_TOKENS = 256  # the step budget after GRAPH_CAPTURE_TOKENS under --goal ttft
FIRST_TOKEN_WAIT = 5.0  # TTFT p95 at least this many TPOT p50: first tokens wait behind steps
FIRST_TOKEN_WAIT_SYNC = 3.0  # the same with async scheduling off: a first token waits a step less
ASYNC_GAP_STEPS = 2.0  # prefill per request at least this many decode steps
STREAM_INTERVAL = 8
STREAM_INTERVAL_CONCURRENCY = 32  # streams running at once from which per-token work binds
STEP_DOWN_CLIENTS = 32  # closed-loop clients from which prompts arrive in waves
STEP_DOWN_PROMPT_SHARE = 0.25  # mean prompt over the step budget at which a wave fills steps
STEP_DOWN_WAIT_STEPS = 4.0  # TTFT p95 at least this many decode steps: first tokens queue
DEFAULT_STEP_BUDGET = 2048  # vLLM 0.30's step budget on GPUs under 70 GiB
STEP_DOWN_TOKENS = 1024
STEP_DOWN_TPOT_TOKENS = MIN_PREFILL_BATCH_TOKENS
STREAM_INTERVAL_FLAG = "--stream-interval"
FAST_PREFILL_FLAG = "--kv-sharing-fast-prefill"
FAST_PREFILL_MIN_PROMPT_TOKENS = 512  # median prompt at which prefill is what a throughput run buys
KV_READ_OVER_WEIGHTS = 0.25  # KV-cache bytes a decode step reads, against its weight bytes
KV_READ_DECODE_PCT = 60.0  # decode's share of its memory-bandwidth ceiling, in %; not under tpot
PER_CHANNEL_EXTRA_GIB = 1.0  # the top of fp8_per_channel's 0.6-1 GiB over fp8 on a 4B model
PER_CHANNEL_HEADROOM = 0.20  # the KV pool's least free share at the parent's peak use
GIB = 2**30


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
    if not evidence or not kv_held_back(situation):
        return None
    if not (current := _kv_fraction_raisable(situation.settings)):
        return _hybrid_at_full_memory(situation)
    if _hinted_gain_gib(situation) > _raised_share_gain_gib(situation):
        return None
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


def fp8_kv_decode_applies(situation: Situation, engine: Engine) -> str | None:
    """Decode reads enough KV cache for halving it to shorten a step. Under TPOT the bytes read
    are the argument; under the other goals decode must also run near its bandwidth ceiling.
    Where KV memory held requests back, outside the TPOT goal, the change is fp8-kv-cache's."""
    signals = situation.measurement
    c = signals.ceilings
    goal = analysis().goal
    if (
        c is None
        or signals.hybrid_cache
        or engine.lever_value(situation.settings, "kv_cache_precision") == "8bit"
        or (kv_held_back(situation) and goal != "tpot")
    ):
        return None
    weights = weight_bytes_read(c)
    pct = c.decode_pct_of_ceiling
    if not (weights and c.kv_bytes_per_sequence and c.avg_running_batch and pct is not None):
        return None
    kv = c.kv_bytes_per_sequence * c.avg_running_batch
    if kv < KV_READ_OVER_WEIGHTS * weights or (goal != "tpot" and pct < KV_READ_DECODE_PCT):
        return None
    ratio = f"{kv / weights:.{1 if kv >= weights else 2}f}"
    return (
        f"decode reads about {kv / 1e9:.1f} GB of KV cache per step against {weights / 1e9:.1f} "
        f"GB of weights ({ratio}x) at {pct:.0f}% of the memory-bandwidth ceiling; an 8-bit KV "
        "cache halves the KV bytes"
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
        if (
            running_at_cap(signals, situation.settings)
            and signals.scheduled_tokens_per_step is not None
            and signals.step_budget_tokens
            and signals.scheduled_tokens_per_step < 0.75 * signals.step_budget_tokens
        ):
            return None
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
            "with and without it. N-gram on the GPU keeps async scheduling on; plain n-gram turns "
            "it off in vLLM 0.30"
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


def _memory_lever(situation: Situation, engine: Engine) -> tuple[str, Settings, str] | None:
    """The first change that leaves enough memory for graph capture: its name, the settings it
    makes and what it does, in words. Serving the text model alone comes with its KV size, or
    with every media input limited to 0, never as a bare ``--language-model-only``."""
    signals, facts, settings = situation.measurement, situation.facts, situation.settings
    headroom = _graph_headroom(signals, settings)
    if headroom is None:
        return None
    spare, reserve = headroom
    if text_only_helps(facts, settings) and spare + facts.encoder_gib >= reserve:
        text_only = without_media_encoders(situation, engine)
        if text_only != settings:
            freed = f"frees about {facts.encoder_gib:.2f} GiB of encoder weights"
            return "no-media-encoders", text_only, freed
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
    signals, settings = situation.measurement, situation.settings
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
    if (lever := _memory_lever(situation, engine)) is None:
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
    lever = _memory_lever(situation, engine)
    enabled = engine.with_lever(lever[1] if lever else current, "graph_capture", True)
    if signals.hybrid_cache and (sequences := _hybrid_sequences(signals, current)):
        enabled = replace(enabled, max_concurrent_requests=sequences)
    cap = _graph_cap(facts, enabled)
    return engine.with_lever(enabled, "graph_capture_limit", cap) if cap else enabled


QWEN2_FP8_KV = Gate(
    "Qwen2 and Qwen2.5 answers break with an fp8 KV cache: a few very large values in their "
    "attention keys do not fit fp8's range (in Tensward's A10G test none of 600 answers matched "
    "bf16's)",
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


def small_step_budget_applies(situation: Situation, engine: Engine) -> str | None:
    """The TTFT direction of the step-budget lever: only under --goal ttft, and never where
    the budget or queueing crossed (raise-prefill-batch's cases). Above GRAPH_CAPTURE_TOKENS it
    offers that budget, at it SMALLER_STEP_TOKENS; either needs the longest prompt with the
    running decodes to fit a step of the budget it offers."""
    m, settings = situation.measurement, situation.settings
    if (
        analysis().goal != "ttft"
        or situation.fired("step_budget")
        or situation.fired("queueing")
        or _by_ttft_ratio(situation)
    ):
        return None
    budget = settings.prefill_batch_tokens or m.step_budget_tokens
    if budget is None or budget < GRAPH_CAPTURE_TOKENS:
        return None
    if m.max_prompt_tokens is None or m.peak_running is None or m.max_prompt_tokens > 512:
        return None
    decodes = round(m.peak_running)
    offered = GRAPH_CAPTURE_TOKENS if budget > GRAPH_CAPTURE_TOKENS else SMALLER_STEP_TOKENS
    if m.max_prompt_tokens + decodes > offered:
        return None
    wait = FIRST_TOKEN_WAIT_SYNC if settings.async_scheduling is False else FIRST_TOKEN_WAIT
    if m.ttft_p95_ms is None or not m.tpot_p50_ms or m.ttft_p95_ms < wait * m.tpot_p50_ms:
        return None
    ratio = f"TTFT p95 {m.ttft_p95_ms:.0f} ms is {m.ttft_p95_ms / m.tpot_p50_ms:.0f}x TPOT p50"
    fit = (
        f"{ratio} under a {budget}-token step budget, and the longest prompt "
        f"({m.max_prompt_tokens} tokens) with {decodes} running decodes fits a {offered}-token "
        "step"
    )
    if budget == GRAPH_CAPTURE_TOKENS:
        return (
            f"{fit}: a smaller step spreads a wave of arriving prompts over more steps, so fewer "
            "first tokens wait behind it"
        )
    return f"{fit}: smaller steps let first tokens out sooner"


def with_small_step_budget(situation: Situation, engine: Engine) -> Settings:
    """GRAPH_CAPTURE_TOKENS above it (or with the budget unknown), else SMALLER_STEP_TOKENS; at
    or below that, the settings unchanged."""
    budget = situation.settings.prefill_batch_tokens or situation.measurement.step_budget_tokens
    if budget is None or budget > GRAPH_CAPTURE_TOKENS:
        return replace(situation.settings, prefill_batch_tokens=GRAPH_CAPTURE_TOKENS)
    if budget <= SMALLER_STEP_TOKENS:
        return situation.settings
    return replace(situation.settings, prefill_batch_tokens=SMALLER_STEP_TOKENS)


def no_async_scheduling_applies(situation: Situation, engine: Engine) -> str | None:
    settings, m = situation.settings, situation.measurement
    if (
        analysis().goal != "ttft"
        or settings.async_scheduling is False
        or engine.lever_value(settings, "speculation") is not None
        or not m.step_ms
    ):
        return None
    prefill = prefill_ms_per_request(analysis().this_run())
    if prefill is None or prefill < ASYNC_GAP_STEPS * m.step_ms:
        return None
    return (
        f"prefill took {prefill:.0f} ms per request, about {prefill / m.step_ms:.1f} decode steps "
        f"of {m.step_ms:.0f} ms. With async scheduling the next step is already planned when a "
        "request arrives, so it waits one extra step; with it off, it joins the next step. "
        "Measured once on a T4 with Qwen3-4B, not a prediction for this GPU: turning it off cut "
        "prefill from 310 to 164 ms per request while each step grew from 117 to 122 ms"
    )


def without_async_scheduling(situation: Situation, engine: Engine) -> Settings:
    return replace(situation.settings, async_scheduling=False)


def _mean_prompt_tokens(m: Measurement) -> float | None:
    """Prompt tokens the engine computed per request over the window."""
    if not (m.prompt_work_share and m.prompt_capacity_tok_s and m.request_throughput):
        return None
    return m.prompt_work_share * m.prompt_capacity_tok_s / m.request_throughput


def step_budget_down_applies(situation: Situation, engine: Engine) -> str | None:
    """A closed loop of many clients on the engine's default step budget, whose prompts fill
    a good part of a step and whose first tokens wait several decode steps: a smaller budget
    spreads each wave of prompts over more, shorter steps. Not where the first tokens wait in
    the queue: a step budget acts only on the time after it."""
    m, settings, facts = situation.measurement, situation.settings, situation.facts
    clients = facts.max_inflight
    if (
        situation.fired("queueing")
        or (m.queue_share is not None and QUEUE_SHARE.crossed(m.queue_share) != "clear")
        or clients is None
        or clients < STEP_DOWN_CLIENTS
        or not facts.load.endswith("concurrent clients")
        or settings.prefill_batch_tokens is not None
        or m.step_budget_tokens != DEFAULT_STEP_BUDGET
    ):
        return None
    prompt = _mean_prompt_tokens(m)
    if prompt is None or prompt < STEP_DOWN_PROMPT_SHARE * DEFAULT_STEP_BUDGET:
        return None
    steps = m.ttft_p95_ms / m.tpot_p50_ms if m.ttft_p95_ms and m.tpot_p50_ms else None
    if steps is not None and steps >= STEP_DOWN_WAIT_STEPS:
        cause = f"TTFT p95 {m.ttft_p95_ms:.0f} ms is {steps:.0f}x TPOT p50"
    elif situation.fired("prefill_stalls_decode"):
        cause = situation.finding("prefill_stalls_decode").evidence
    else:
        return None
    return (
        f"{cause}, with {clients} clients sending prompts of about {prompt:.0f} tokens into "
        f"{DEFAULT_STEP_BUDGET}-token steps: a smaller step budget spreads each wave over more, "
        "shorter steps"
    )


def with_step_budget_down(situation: Situation, engine: Engine) -> Settings:
    tokens = STEP_DOWN_TPOT_TOKENS if analysis().goal == "tpot" else STEP_DOWN_TOKENS
    return replace(situation.settings, prefill_batch_tokens=tokens)


def stream_interval_applies(situation: Situation, engine: Engine) -> str | None:
    if STREAM_INTERVAL_FLAG in situation.settings.extra_args:
        return None
    if situation.fired("frontend_cpu"):
        return (
            f"{situation.finding('frontend_cpu').evidence}; streaming every {STREAM_INTERVAL} "
            "tokens instead of every token cuts the API server's work per token"
        )
    m = situation.measurement
    if (m.peak_running or 0) >= STREAM_INTERVAL_CONCURRENCY and situation.fired("gpu_saturated"):
        return (
            f"{m.peak_running:.0f} streams ran at once on a saturated GPU; streaming every "
            f"{STREAM_INTERVAL} tokens instead of every token cuts the API server's work per "
            "token"
        )
    return None


def with_stream_interval(situation: Situation, engine: Engine) -> Settings:
    settings = situation.settings
    extra = {**settings.extra_args, STREAM_INTERVAL_FLAG: str(STREAM_INTERVAL)}
    return replace(settings, extra_args=extra)


def _median_prompt_tokens() -> float | None:
    counts = sorted(tokens for tokens, _ in analysis().prompt_blocks.values())
    return statistics.median(counts) if counts else None


def fast_prefill_applies(situation: Situation, engine: Engine) -> str | None:
    anatomy = analysis().anatomy
    if (
        anatomy is None
        or anatomy.kv_shared_layers <= 0
        or not anatomy.layers
        or FAST_PREFILL_FLAG in situation.settings.extra_args
        or engine.lever_value(situation.settings, "speculation") is not None
    ):
        return None
    goal = analysis().goal
    median = _median_prompt_tokens()
    if goal in ("tpot", "throughput") and (
        median is None or median < FAST_PREFILL_MIN_PROMPT_TOKENS
    ):
        return None
    share = anatomy.kv_shared_layers / anatomy.layers
    return (
        f"{anatomy.kv_shared_layers} of {anatomy.layers} layers reuse earlier KV: prefill skips "
        f"about {share:.0%} of its work. Tensward compares the answers of the next run with "
        "this one's"
    )


def with_fast_prefill(situation: Situation, engine: Engine) -> Settings:
    settings = situation.settings
    return replace(settings, extra_args={**settings.extra_args, FAST_PREFILL_FLAG: True})


def async_scheduling_applies(situation: Situation, engine: Engine) -> str | None:
    settings = situation.settings
    if settings.async_scheduling is not False or engine.lever_value(settings, "speculation"):
        return None
    return (
        "async scheduling is off: with it the CPU prepares the next step while the GPU runs "
        "this one"
    )


def with_async_scheduling(situation: Situation, engine: Engine) -> Settings:
    return replace(situation.settings, async_scheduling=True)


SATURATED_PROMPT_SHARE = 0.3  # prompt work share of a saturated GPU from which fp8 maths pays


def _weights_signal(situation: Situation, *, eight_bit: bool = True) -> str | None:
    """Why smaller weights pay here: decode bandwidth binds, the KV cache is short, or, for
    ``eight_bit`` weights, a saturated GPU that computes in fp8 spends at least
    SATURATED_PROMPT_SHARE of its work on prompts, whose matrix work they halve too."""
    if situation.fired("decode_bandwidth"):
        return situation.finding("decode_bandwidth").evidence
    if evidence := kv_bound(situation):
        return evidence
    if not eight_bit:
        return None
    context, m = analysis(), situation.measurement
    options = context.answer_options
    capability = options.capability if options is not None else context.gpu_capability
    share = m.prompt_work_share
    if (
        not situation.fired("gpu_saturated") or capability is None or capability < NATIVE_FP8
        or share is None or share < SATURATED_PROMPT_SHARE
    ):  # fmt: skip
        return None
    decode = m.ceilings.decode_pct_of_ceiling if m.ceilings is not None else None
    at = f", decode at {decode:.0f}% of the memory-bandwidth ceiling" if decode is not None else ""
    return (
        f"the GPU was saturated, with prompts {share:.0%} of its work{at}; at compute capability "
        f"{capability:g} 8-bit weights also halve the prompts' matrix work"
    )


def _on_the_fly(situation: Situation, method: str) -> tuple[str, str] | None:
    """(the mode this GPU runs ``method`` in, the signal) where the engine can quantize this
    dense checkpoint at start and a weight-bound limit is crossed."""
    options = analysis().answer_options
    if (
        options is None
        or method not in options.on_the_fly
        or situation.settings.quantization
        or situation.facts.quantization is not None
    ):
        return None
    signal = _weights_signal(situation, eight_bit=method == "fp8")
    return (options.on_the_fly[method], signal) if signal else None


def _per_channel(situation: Situation, *, sized: bool = False) -> tuple[bool, str] | None:
    """(whether fp8 per channel is the method, why) where this GPU runs it; None otherwise. It is
    chosen when the KV cache is sized explicitly, or when the parent's peak KV use leaves
    PER_CHANNEL_HEADROOM of the pool that the halved quantizable weights grow, less
    per-channel's extra memory."""
    context = analysis()
    options = context.answer_options
    if options is None or PER_CHANNEL not in options.on_the_fly:
        return None
    larger = "0.6-1 GiB larger than per-tensor on a 4B model"
    if sized:
        return True, f"per-channel: {larger}; the KV cache is sized explicitly"
    m = situation.measurement
    freed = freed_bytes(context.anatomy) if context.anatomy is not None else None
    if m.peak_kv_usage is None or not m.kv_available_gib or not freed:
        unknown = "the KV cache's headroom after it is not known"
        return False, f"per-tensor: per-channel is {larger}, and {unknown}"
    pool = m.kv_available_gib + freed["quantizable"] / 2 / GIB - PER_CHANNEL_EXTRA_GIB
    headroom = max(0.0, 1 - m.peak_kv_usage * m.kv_available_gib / pool) if pool > 0 else 0.0
    if headroom >= PER_CHANNEL_HEADROOM:
        return True, f"per-channel: {larger}; the KV cache keeps {headroom:.0%} headroom"
    return False, (f"per-tensor: per-channel is {larger}, and the KV cache would keep "
                   f"{headroom:.0%} headroom, under {PER_CHANNEL_HEADROOM:.0%}")  # fmt: skip


def fp8_weights_applies(situation: Situation, engine: Engine) -> str | None:
    found = _on_the_fly(situation, "fp8")
    if found is None:
        return None
    mode, signal = found
    here = (
        f"the maths stays 16-bit, and the command adds {MARLIN_FP8_ARG}: vLLM's default fp8 "
        f"kernel does not start below compute capability {NATIVE_FP8:g}"
        if mode == WEIGHT_ONLY
        else "the maths runs in 8 bits too"
    )
    method = _per_channel(situation)
    return (f"{signal}; vLLM can store the weights in 8-bit fp8 when it starts, so each decode "
            "step reads half the weight bytes and the freed memory holds more requests (on this "
            f"GPU {here}){f'; {method[1]}' if method else ''}")  # fmt: skip


def bnb_weights_applies(situation: Situation, engine: Engine) -> str | None:
    found = _on_the_fly(situation, "bitsandbytes")
    if found is None:
        return None
    _, signal = found
    return (f"{signal}; the installed bitsandbytes plugin lets vLLM load these weights in 4 bits: "
            "a quarter of the weight bytes")  # fmt: skip


def _format_words(format: Format) -> str:
    if format.bits == 4 or format.mode in format.name:
        return format.name
    if format.mode == WEIGHT_ONLY:
        return f"{format.name} (weight-only here, with {MARLIN_FP8_ARG})"
    return f"{format.name} ({format.mode})"


def quantized_checkpoint_applies(situation: Situation, engine: Engine) -> str | None:
    options = analysis().answer_options
    if (
        options is None
        or options.capability is None
        or not options.checkpoints
        or situation.facts.quantization is not None
    ):
        return None
    fast = options.fast_prompts
    prefill = situation.fired("prefill") and bool(fast)
    signal = _weights_signal(situation) or (
        situation.finding("prefill").evidence if prefill else None
    )
    if not signal:
        return None
    eight = [_format_words(f) for f in options.checkpoints if f.bits == 8]
    four = [_format_words(f) for f in options.checkpoints if f.bits == 4]
    said = [
        f"{signal}; a pre-quantized copy of this model runs on this GPU (compute capability "
        f"{options.capability:g})",
        *([f"8-bit: {', '.join(eight)}"] if eight else []),
        *([f"4-bit, where answers change more often: {', '.join(four)}"] if four else []),
        *([f"{' and '.join(fast)} also {'run' if len(fast) > 1 else 'runs'} prompt processing "
           "in 8-bit arithmetic here"] if fast else []),
    ]  # fmt: skip
    return ". ".join(said)


def with_fp8_weights(situation: Situation, engine: Engine, *, sized: bool = False) -> Settings:
    method = _per_channel(situation, sized=sized)
    if method is not None and method[0]:
        return replace(situation.settings, quantization=PER_CHANNEL)
    weights = replace(situation.settings, quantization="fp8")
    options = analysis().answer_options
    if options is not None and options.on_the_fly.get("fp8") == WEIGHT_ONLY:
        return engine.with_engine_arg(weights, MARLIN_FP8_ARG)
    return weights


def with_bnb_weights(situation: Situation, engine: Engine) -> Settings:
    return replace(situation.settings, quantization="bitsandbytes")


PEAK_OVER_REFERENCE = 2.0  # a profiled peak this many times the reference run's over-reserves
POOL_SHARE_OF_ROOM = 0.5  # a KV-bound pool below this share of the room it could take


def _log(run_dir: Path | None) -> str:
    try:
        return (run_dir / "server.log").read_text(errors="replace") if run_dir else ""
    except OSError:
        return ""


def _over_reserved(
    situation: Situation, usage: Mapping[str, float], this: Path | None, before: str | None
) -> tuple[str, float] | None:
    """Why this run's KV pool is smaller than its memory allows, and the GiB the profiler
    reserved for activations beyond a normal peak: the reference run's, else the activations
    the GPU's sampled memory shows."""
    m, peak = situation.measurement, usage["peak_activation_gib"]
    if not m.kv_available_gib:
        return None
    reference = kvsize.reference_run(analysis().project, before)
    normal = memory_usage(_log(reference)) if reference is not None else None
    if normal is not None and peak >= PEAK_OVER_REFERENCE * normal["peak_activation_gib"]:
        usual = normal["peak_activation_gib"]
        return (
            f"the engine reserved {peak:.2f} GiB for activations at start, {peak / usual:.1f}x "
            f"the {usual:.2f} GiB of the run with no engine arguments"
        ), peak - usual
    evidence = kv_bound(situation)
    peak_used, _ = sampled_memory(this, m)
    if not evidence or not (m.peak_waiting or 0) > 0 or not m.gpu_memory_gib or peak_used is None:
        return None
    fraction = situation.settings.kv_memory_fraction or DEFAULT_KV_MEMORY_FRACTION
    fixed = usage["consumed_gib"] + usage["graphs_gib"]
    room = m.gpu_memory_gib * fraction - fixed
    used = max(peak_used / 1024 - m.kv_available_gib - fixed, 0.0)
    if m.kv_available_gib >= POOL_SHARE_OF_ROOM * room or used >= peak:
        return None
    return (
        f"{evidence}; the KV cache holds {m.kv_available_gib:.2f} GiB of the {room:.2f} GiB left "
        f"after the weights and graphs, as the engine reserved {peak:.2f} GiB for activations "
        f"where the GPU's sampled memory shows about {used:.2f} GiB"
    ), peak - used


def _waited_for_cap(situation: Situation) -> bool:
    """Whether the requests that waited waited for the concurrency cap, not for KV space: the
    running requests reached the cap and the KV-capacity finding is clear."""
    return (
        running_at_cap(situation.measurement, situation.settings)
        and situation.finding("kv_capacity").state == "clear"
    )


def _hinted_kv_size(situation: Situation) -> tuple[str, int] | None:
    """On a run KV space held back, the explicit size the engine's own "fully utilize" figure
    gives, through :func:`kvsize.sized`'s caps: at most what the parent's sampled GPU memory
    leaves free, and, with no sampled peak, the figure less the margin. The extra pool is added
    memory, not memory freed from others, so the sampled peak is raised by it before the cap
    subtracts it again."""
    m = situation.measurement
    this, before = this_run()
    usage = memory_usage(_log(this))
    if (
        usage is None
        or not m.kv_available_gib
        or not kv_held_back(situation)
        or _waited_for_cap(situation)
    ):
        return None
    if kvsize.KV_FLAG in situation.settings.extra_args:
        return None
    kv, full = round(m.kv_available_gib * 2**30), int(usage["full_kv_bytes"])
    gain = max(full - kv, 0)
    peak_used, total = sampled_memory(this, m)
    size = kvsize.sized(
        kv,
        freed_bytes=gain,
        full_kv_bytes=full,
        peak_used_mib=None if peak_used is None else peak_used + gain / kvsize.MIB,
        total_mib=total,
        backed_off=kvsize.failed_sized_steps(analysis().project, before),
    )
    if size is not None and peak_used is None:
        size = min(size, full - kvsize.MARGIN_BYTES)
    if size is None or size <= kv:
        return None
    return (
        f"KV space held requests back and the engine's start log says the GPU fits a KV cache of "
        f"{full / 2**30:.2f} GiB against the {m.kv_available_gib:.2f} GiB it used; a fixed KV "
        f"size of {size / 2**30:.2f} GiB gives the cache that memory"
    ), size


def _hinted_gain_gib(situation: Situation) -> float:
    """The GiB the explicit size from the engine's figure adds to the pool; 0 without one."""
    found = _hinted_kv_size(situation)
    return found[1] / 2**30 - (situation.measurement.kv_available_gib or 0) if found else 0.0


def _raised_share_gain_gib(situation: Situation) -> float:
    current = situation.settings.kv_memory_fraction or DEFAULT_KV_MEMORY_FRACTION
    return (RAISED_KV_MEMORY_FRACTION - current) * (situation.measurement.gpu_memory_gib or 0)


def _explicit_kv_size(situation: Situation) -> tuple[str, int] | None:
    kv_gib = situation.measurement.kv_available_gib
    if kvsize.KV_FLAG in situation.settings.extra_args or not kv_gib:
        return None
    this, before = this_run()
    usage = memory_usage(_log(this))
    if usage is None:
        return None
    found = _over_reserved(situation, usage, this, before)
    if found is None:
        return _hinted_kv_size(situation)
    why, over_gib = found
    peak_used, total = sampled_memory(this, situation.measurement)
    size = kvsize.sized(
        round(kv_gib * 2**30),
        freed_bytes=round(over_gib * 2**30),
        full_kv_bytes=int(usage["full_kv_bytes"]),
        peak_used_mib=peak_used,
        total_mib=total,
        backed_off=kvsize.failed_sized_steps(analysis().project, before),
    )
    if size is None or size <= kv_gib * 2**30:
        return None
    return f"{why}; a fixed KV size of {size / 2**30:.2f} GiB gives the cache that memory", size


def explicit_kv_size_applies(situation: Situation, engine: Engine) -> str | None:
    found = _explicit_kv_size(situation)
    return found[0] if found else None


def with_explicit_kv_size(situation: Situation, engine: Engine) -> Settings:
    found = _explicit_kv_size(situation)
    return kvsize.with_kv_bytes(situation.settings, found[1]) if found else situation.settings


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
            lower_prefill_batch_applies, with_lowered_prefill_batch,
            ngram_speculation_applies, with_ngram_speculation, ngram_under_structured_output,
            drop_speculation_applies, without_speculation, cuda_graphs_applies, with_cuda_graphs,
            compact_json_low_bar, with_compact_json, more_api_servers_applies,
            with_more_api_servers, small_step_budget_applies, with_small_step_budget,
            step_budget_down_applies, with_step_budget_down,
            no_async_scheduling_applies, without_async_scheduling, stream_interval_applies,
            with_stream_interval, async_scheduling_applies, with_async_scheduling,
            fp8_weights_applies, with_fp8_weights, bnb_weights_applies, with_bnb_weights,
            quantized_checkpoint_applies, fp8_kv_decode_applies,
            explicit_kv_size_applies, with_explicit_kv_size, fast_prefill_applies,
            with_fast_prefill,
        )
    },
    "vllm:qwen2_fp8_kv": QWEN2_FP8_KV,
}  # fmt: skip
HOLD_BACKS: dict[str, Callable[[Situation], str | None]] = {
    "trim-max-context-len": trim_hold_back,
    "cuda-graphs": data_parallel_hold_back,
}
