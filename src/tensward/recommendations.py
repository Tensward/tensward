"""Tuning recommendations: one setting change each, offered only when signals call for it.

A recipe never guesses. ``applies`` reads the baseline measurement (engine signals the engine
reported plus client-side timings); a signal that was not measured never triggers a recipe.
Recipes speak in engine-neutral :class:`Settings`; each engine maps them to its own flags.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Collection, Sequence

from .engines.protocol import Settings
from .extensions import load_extender

if TYPE_CHECKING:
    from .measurement import Measurement
    from .project import ResolvedProject

KV_PRESSURE_USAGE = 0.9  # peak KV-cache usage above this counts as KV-bound
KV_HEADROOM_USAGE = 0.8  # more sequences only help below this KV usage
KV_TARGET_USAGE = 0.85  # a raised concurrency is sized to keep projected KV usage under this
RAISED_KV_MEMORY_FRACTION = 0.95
RAISED_PREFILL_BATCH_TOKENS = 8192
ASSUMED_DEFAULT_PREFILL_BATCH_TOKENS = 2048  # vLLM 0.30 serving default below 70 GB GPUs
MIN_PREFILL_BATCH_TOKENS = 512  # smaller chunks cost more in step overhead than they save
SPECULATION_MIN_PROMPT_WORDS = 256  # median prompt length at which copying spans is plausible
NGRAM_SPECULATION_FLAG = "--speculative-config"
NGRAM_SPECULATION = '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4}'
PREFILL_BOUND_TTFT_TO_TPOT = 20.0  # TTFT p50 over TPOT p50
DECODE_STALL_TPOT_RATIO = 2.0  # TPOT p95 over TPOT p50
MAX_CONTEXT_LEN_ROUNDING = 256
MAX_RUNGS = 4  # steps a numeric setting is walked in, the evidence-based target included

# Prefix caching pays when many requests start with the same text. A prompt counts as sharing
# when it has a common prefix with another prompt that is long enough to cache (one 16-token
# KV block) and either 256 tokens or a quarter of its own length. Tokens are counted as
# whitespace-separated words, which never overstates a prefix.
MIN_CACHEABLE_PREFIX_TOKENS = 16
MIN_SHARED_PREFIX_TOKENS = 256
MIN_SHARED_PREFIX_FRACTION = 0.25
MIN_SHARING_PROMPTS = 0.20  # share of the prompts that must share a prefix


@dataclass(frozen=True, slots=True)
class WorkloadFacts:
    """What the registered workload says about itself."""

    prompts: tuple[str, ...]
    output_tokens: int
    context_limit: int  # the model's max_position_embeddings, from the checkpoint's config.json
    quantization: str | None = None  # the checkpoint's declared quantization, if any
    offers_tools: bool = False  # whether any prompt offers tools
    media_encoders: bool = False  # the checkpoint takes images; no prompt sends one in this release
    load: str = "the declared workload"  # the arrival policy the configuration declares

    @classmethod
    def of(cls, project: ResolvedProject) -> WorkloadFacts:
        metadata = project.artifact.metadata
        arrival = project.settings.workload.arrival
        load = f"{arrival.concurrency} concurrent clients"
        if arrival.kind != "closed_loop":
            load = f"{arrival.rate_rps:g} requests/s" + (
                f", at most {arrival.max_inflight} in flight" if arrival.kind == "capped" else ""
            )
        return cls(
            prompts=tuple(entry.text for entry in project.prompts),
            output_tokens=project.settings.workload.output_tokens,
            context_limit=metadata.context_limit,
            quantization=metadata.variants[0].label,
            offers_tools=any(entry.tools for entry in project.prompts),
            media_encoders="image" in project.anatomy.modalities,
            load=load,
        )


@dataclass(frozen=True, slots=True)
class Recipe:
    name: str
    applies: Callable[[Measurement, WorkloadFacts, Settings], str | None]
    apply: Callable[[Measurement, WorkloadFacts, Settings], Settings]
    quality_risk: bool = False
    # Serves extensions that search settings; plain `analyse` ignores it.
    # The directions it pushes: "throughput", "goodput", "req_s", "ttft" or "tpot" (an optimize
    # point prefers recipes that serve its objective or a metric it constrains). Empty means a
    # general fix that helps every direction.
    helps: frozenset[str] = frozenset()
    # Also for the extension: the numeric setting ``apply`` moves to the evidence-based target.
    # Optimize walks it there in steps (see :func:`rungs`) so a point can stop at the step that
    # serves it.
    steps_on: str | None = None


def _above(value: float | None, threshold: float) -> bool:
    return value is not None and value > threshold


def _kv_bound(signals: Measurement) -> str | None:
    if _above(signals.preemptions, 0):
        return f"{signals.preemptions:.0f} preemptions"
    if _above(signals.peak_kv_usage, KV_PRESSURE_USAGE):
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
    return _round_up((signals.max_prompt_tokens or 0) + facts.output_tokens)


def _round_up(tokens: int) -> int:
    return -(-tokens // MAX_CONTEXT_LEN_ROUNDING) * MAX_CONTEXT_LEN_ROUNDING


def fitting_max_context_len(signals: Measurement, facts: WorkloadFacts) -> int | None:
    """The smallest multiple of 256 that fits the longest prompt plus its output.

    Never above the model's own context limit; None when no prompt is too long or when even
    the model's limit cannot hold the longest prompt.
    """
    if not signals.too_long or signals.max_prompt_tokens is None:
        return None
    longest = signals.max_prompt_tokens + facts.output_tokens
    if longest > facts.context_limit:
        return None
    return min(_round_up(longest), facts.context_limit)


def _fit_context_applies(
    signals: Measurement, facts: WorkloadFacts, settings: Settings
) -> str | None:
    if fitting_max_context_len(signals, facts) is None:
        return None
    longest = (signals.max_prompt_tokens or 0) + facts.output_tokens
    return (
        f"{len(signals.too_long)} prompts do not fit max_context_len "
        f"{signals.max_context_len}; the longest needs {longest} tokens"
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
    if facts.media_encoders and settings.media_inputs is None:
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
    """Running plus waiting requests at their peaks. Engines export these gauges as floats
    (vLLM reports "8.0")."""
    return round(signals.peak_running or current) + round(signals.peak_waiting or 0)


def _raised_concurrency(signals: Measurement, current: int) -> int:
    """The observed demand (running plus waiting), at least double, within the KV headroom.

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
        return (
            f"running requests hit max_concurrent_requests {cap} with "
            f"{signals.peak_waiting:.0f} waiting, and the highest sampled KV-cache usage is only "
            f"{signals.peak_kv_usage:.1%}; at your declared load of {facts.load}, highest sampled "
            f"running plus waiting was {_demand(signals, cap)}, so "
            f"{_raised_concurrency(signals, cap)} is worth trying (the load is what your "
            "configuration declares, not measured traffic). It usually cuts queueing (TTFT) but "
            "slows each token (TPOT) as more sequences share every step - measure both"
        )
    return None


def _lowered_concurrency(signals: Measurement, cap: int) -> int:
    """Half the cap; without preemptions never below the observed peak, where queueing would
    start (a power of two, like the raised level)."""
    half = max(cap // 2, 1)
    if _above(signals.preemptions, 0):
        return half
    return max(half, 1 << (round(signals.peak_running or cap) - 1).bit_length())


def _lower_concurrency_applies(
    signals: Measurement, facts: WorkloadFacts, settings: Settings
) -> str | None:
    cap = _concurrency_cap(signals, settings) or 0
    if cap <= 1 or _lowered_concurrency(signals, cap) >= cap:
        return None
    if _above(signals.preemptions, 0):
        return f"{signals.preemptions:.0f} preemptions; fewer concurrent sequences fit the KV cache"
    if signals.peak_waiting == 0:
        return (
            f"nothing waited while at most {signals.peak_running:.0f} of max_concurrent_requests "
            f"{cap} ran; a smaller batch limit shortens every engine step, which lowers latency "
            "as long as the queue stays empty, but if the load grows requests queue "
            "(TTFT) and throughput falls"
        )
    return None


def _raise_prefill_batch_applies(
    signals: Measurement, facts: WorkloadFacts, settings: Settings
) -> str | None:
    if (
        signals.ttft_p50_ms is not None
        and signals.tpot_p50_ms
        and signals.ttft_p50_ms > PREFILL_BOUND_TTFT_TO_TPOT * signals.tpot_p50_ms
        and _above(signals.peak_waiting, 0)
        and (settings.prefill_batch_tokens or 0) < RAISED_PREFILL_BATCH_TOKENS
    ):
        return (
            f"TTFT p50 {signals.ttft_p50_ms:.0f} ms is over {PREFILL_BOUND_TTFT_TO_TPOT:g}x "
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
    if facts.media_encoders and settings.media_inputs is not False:
        return None  # an image+text engine refuses a batch smaller than one media item (S2)
    if (
        signals.tpot_p95_ms is not None
        and signals.tpot_p50_ms
        and signals.tpot_p95_ms > DECODE_STALL_TPOT_RATIO * signals.tpot_p50_ms
        and _lowered_prefill_batch(settings) >= MIN_PREFILL_BATCH_TOKENS
    ):
        return (
            f"TPOT p95 {signals.tpot_p95_ms:.1f} ms is over {DECODE_STALL_TPOT_RATIO:g}x "
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

RECIPES: tuple[Recipe, ...] = (
    Recipe(
        "enable-tool-calling",
        _enable_tool_calling_applies,
        lambda s, f, cur: replace(cur, tool_calling=True),
    ),
    Recipe(
        "fast-quant-kernel",
        _fast_quant_kernel_applies,
        lambda s, f, cur: replace(cur, quantization=None),
        helps=THROUGHPUT | LATENCY,
    ),
    Recipe(
        "fit-context",
        _fit_context_applies,
        lambda s, f, cur: replace(cur, max_context_len=fitting_max_context_len(s, f)),
    ),
    Recipe(
        "more-kv-memory",
        _more_kv_memory_applies,
        lambda s, f, cur: replace(cur, kv_memory_fraction=RAISED_KV_MEMORY_FRACTION),
        helps=THROUGHPUT | LATENCY,
        steps_on="kv_memory_fraction",
    ),
    Recipe(
        "no-media-encoders",
        _no_media_encoders_applies,
        lambda s, f, cur: replace(cur, media_inputs=False),
        helps=THROUGHPUT | LATENCY,
    ),
    Recipe(
        "trim-max-context-len",
        _trim_max_context_len_applies,
        lambda s, f, cur: replace(cur, max_context_len=_trimmed_max_context_len(s, f)),
        helps=THROUGHPUT,
        steps_on="max_context_len",
    ),
    Recipe(
        "raise-concurrency",
        _raise_concurrency_applies,
        lambda s, f, cur: replace(
            cur, max_concurrent_requests=_raised_concurrency(s, _concurrency_cap(s, cur) or 1)
        ),
        helps=THROUGHPUT | {"ttft"},  # it also ends the queueing that dominates TTFT
        steps_on="max_concurrent_requests",
    ),
    Recipe(
        "lower-concurrency",
        _lower_concurrency_applies,
        lambda s, f, cur: replace(
            cur, max_concurrent_requests=_lowered_concurrency(s, _concurrency_cap(s, cur) or 2)
        ),
        helps=LATENCY,
    ),
    Recipe(
        "raise-prefill-batch",
        _raise_prefill_batch_applies,
        lambda s, f, cur: replace(cur, prefill_batch_tokens=RAISED_PREFILL_BATCH_TOKENS),
        helps=frozenset({"throughput", "ttft"}),
        steps_on="prefill_batch_tokens",
    ),
    Recipe(
        "lower-prefill-batch",
        _lower_prefill_batch_applies,
        lambda s, f, cur: replace(cur, prefill_batch_tokens=_lowered_prefill_batch(cur)),
        helps=frozenset({"tpot"}),
    ),
    Recipe(
        "ngram-speculation",
        _ngram_speculation_applies,
        lambda s, f, cur: replace(
            cur, extra_args={**cur.extra_args, NGRAM_SPECULATION_FLAG: NGRAM_SPECULATION}
        ),
        helps=frozenset({"tpot"}),
    ),
    Recipe(
        "prefix-caching",
        _prefix_caching_applies,
        lambda s, f, cur: replace(cur, prefix_caching=True),
        helps=THROUGHPUT | {"ttft"},
    ),
    Recipe(
        "cuda-graphs",
        _cuda_graphs_applies,
        lambda s, f, cur: replace(cur, cuda_graphs=True),
        helps=THROUGHPUT | LATENCY,
    ),
    Recipe(
        "fp8-kv-cache",
        _fp8_kv_cache_applies,
        lambda s, f, cur: replace(cur, kv_cache_dtype="fp8"),
        quality_risk=True,
        helps=THROUGHPUT,
    ),
)


def _ladder(start: float, target: float) -> list[float]:
    """Values from ``start`` to ``target``: doubling or halving each time, or, for a fraction,
    halving the distance. Ends on ``target``; at most MAX_RUNGS, spread evenly."""
    if isinstance(target, float):
        steps = [round((start + target) / 2, 3), target]
    else:
        steps, value = [], start
        while True:
            value = value * 2 if target > start else value // 2
            if value >= target if target > start else value <= target:
                break
            steps.append(value)
        steps.append(target)
    steps = list(dict.fromkeys(step for step in steps if step != start))
    last = len(steps) - 1
    picks = sorted({round(i * last / (MAX_RUNGS - 1)) for i in range(MAX_RUNGS)})
    return [steps[i] for i in picks] if last >= MAX_RUNGS else steps


def rungs(
    recipe: Recipe, signals: Measurement, facts: WorkloadFacts, settings: Settings
) -> list[Settings]:
    """The settings ``recipe`` proposes, easiest step first and the full change last.
    (Serves extensions that search settings; plain ``analyse`` applies the full change.)

    A recipe that moves a numeric setting proposes the steps between its current value and the
    target (8 -> 16 -> 32 -> 64 for concurrency), so a jump that overshoots what one point
    can tolerate still leaves a smaller gain to adopt.
    """
    target = recipe.apply(signals, facts, settings)
    knob = recipe.steps_on
    if knob is None:
        return [target]
    fallbacks = {
        "max_concurrent_requests": _concurrency_cap(signals, settings),
        "prefill_batch_tokens": ASSUMED_DEFAULT_PREFILL_BATCH_TOKENS,
        "max_context_len": signals.max_context_len,
    }
    start = getattr(settings, knob) or fallbacks.get(knob)
    if start is None:
        return [target]
    steps: list[Settings] = []
    for value in _ladder(start, getattr(target, knob)):
        change: dict[str, Any] = {knob: value}
        steps.append(replace(target, **change))
    return steps


def suggest(
    signals: Measurement,
    facts: WorkloadFacts,
    settings: Settings,
    *,
    allow_quality_changes: bool,
    skip: Collection[str] = (),
) -> list[tuple[Recipe, str]]:
    """The recipes that apply to this measurement (those of an installed analysis plugin too),
    each with its reason, minus ``skip``."""
    found = []
    extender = load_extender()
    for recipe in (*RECIPES, *(extender.recipes if extender else ())):
        if recipe.name in skip or (recipe.quality_risk and not allow_quality_changes):
            continue
        reason = recipe.applies(signals, facts, settings)
        if reason:
            found.append((recipe, reason))
    return found
