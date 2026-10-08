"""The named predicates that suit any engine: the conditions and changes the entries of
data/playbook.toml name without an engine prefix. They read the run through the engine's
capabilities and levers, so one body serves every engine."""

from __future__ import annotations

from dataclasses import replace
from statistics import median
from typing import TYPE_CHECKING, Any, Callable, Sequence

from .classify import queued_at_cap
from .measurement import blocks_per_request
from .playbook import fitting_max_context_len, round_up_context
from .settings import Settings

if TYPE_CHECKING:
    from .engines.protocol import Engine
    from .measurement import Measurement
    from .playbook import Situation, WorkloadFacts

KV_HEADROOM_USAGE = 0.8  # more sequences only help below this KV usage
KV_TARGET_USAGE = 0.85  # a raised concurrency is sized to keep projected KV usage under this

# Prefix caching pays when many requests start with the same text. A prompt counts as sharing
# when it has a common prefix with another prompt that is long enough to cache (the engine's
# smallest cacheable prefix: one 16-token KV block on vLLM) and either 256 tokens or a quarter of
# its own length. Tokens are counted as whitespace-separated words, which never overstates a
# prefix.
MIN_SHARED_PREFIX_TOKENS = 256
MIN_SHARED_PREFIX_FRACTION = 0.25
MIN_SHARING_PROMPTS = 0.20  # share of the prompts that must share a prefix
PREFIX_SHARE_FOR_KV = 0.30  # prompts sharing this much of their words make caching a KV cure


def above(value: float | None, threshold: float) -> bool:
    return value is not None and value > threshold


def kv_bound(situation: Situation) -> str | None:
    if situation.fired("kv_capacity"):
        return situation.finding("kv_capacity").evidence
    return None


# --- memory --------------------------------------------------------------------------


def trim_max_context_len_applies(situation: Situation, engine: Engine) -> str | None:
    signals, facts = situation.measurement, situation.facts
    evidence = kv_bound(situation)
    if not evidence or signals.max_prompt_tokens is None:
        return None
    needed = trimmed_max_context_len(signals, facts)
    if signals.max_context_len and needed < signals.max_context_len:
        return (
            f"{evidence}; the workload needs at most {needed} tokens, "
            f"max_context_len is {signals.max_context_len}; requests longer than that would "
            "be refused"
        )
    return None


def trimmed_max_context_len(signals: Measurement, facts: WorkloadFacts) -> int:
    return round_up_context((signals.max_prompt_tokens or 0) + facts.output_tokens)


def fit_context_applies(situation: Situation, engine: Engine) -> str | None:
    signals, facts = situation.measurement, situation.facts
    if fitting_max_context_len(signals, facts) is None:
        return None
    longest = (signals.max_prompt_tokens or 0) + facts.output_tokens
    return (
        f"prompts that do not fit max_context_len {signals.max_context_len}: "
        f"{len(signals.too_long)}; the longest needs {longest} tokens"
    )


def no_media_encoders_applies(situation: Situation, engine: Engine) -> str | None:
    return text_only_helps(situation.facts, situation.settings)


def text_only_helps(facts: WorkloadFacts, settings: Settings) -> str | None:
    if facts.media_encoders and not facts.sends_images and settings.media_inputs is None:
        return (
            "the checkpoint takes images and video but no prompt sends any; serving only its "
            "text model stops the engine reserving memory for the media encoders and lifts the "
            "minimum batch size they impose (requests with images would then be rejected)"
        )
    return None


# --- tool calling --------------------------------------------------------------------


def enable_tool_calling_applies(situation: Situation, engine: Engine) -> str | None:
    facts, settings = situation.facts, situation.settings
    if facts.offers_tools and not settings.tool_calling and settings.tool_parser is not None:
        return (
            "the workload offers tools but tool calling is off, so the server refuses them; "
            f"the checkpoint's tool parser is {settings.tool_parser}"
        )
    return None


# --- quantization ---------------------------------------------------------------------


def fast_quant_kernel_applies(situation: Situation, engine: Engine) -> str | None:
    slow = [kernel.name for kernel in situation.measurement.quant_kernels if kernel.slow]
    if slow and (forced := situation.settings.quantization) is not None:
        return (
            f"the engine selected a slower quantization path ({', '.join(slow)}) while "
            f"quantization is forced to {forced}; the engine's own choice "
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


def _hybrid_pool(signals: Measurement) -> tuple[int, int, int] | None:
    """On a cache with linear-attention state: its usable blocks, the blocks one running request
    held, and the requests those blocks hold; None on other caches."""
    per_request = blocks_per_request(signals)
    if per_request is None or not signals.kv_blocks:
        return None
    usable = int(signals.kv_blocks) - 1
    return usable, per_request, usable // per_request


def _raised_concurrency(situation: Situation, current: int) -> int:
    """The observed demand (the client's in-flight peak when known, else running plus waiting),
    at least double, as a power of two, halved until its projected KV usage fits the target.
    On a hybrid cache, the demand up to the requests the usable blocks hold."""
    if (hybrid := _hybrid_pool(situation.measurement)) is not None:
        return min(_demand(situation.measurement, current), hybrid[2])
    target = 1 << (max(_demand(situation.measurement, current), 2 * current) - 1).bit_length()
    caching = situation.settings.prefix_caching is True
    while target > 1 and above(
        projected_kv(situation, target, caching=caching, running=current), KV_TARGET_USAGE
    ):
        target //= 2
    return target


def prefix_and_rest(facts: WorkloadFacts) -> tuple[float, float]:
    """The median words a prompt shares with another (held once with prefix caching) and the
    median rest of a sequence: its unshared words plus half the output tokens. With no
    prompts, nothing is shared: ``(0.0, output_tokens / 2)``."""
    half = facts.output_tokens / 2
    if not facts.prompts:
        return 0.0, half
    shared = shared_lengths(facts.prompts)
    unshared = [len(prompt.split()) - n for prompt, n in zip(facts.prompts, shared)]
    return float(median(shared)), median(unshared) + half


def projected_kv(
    situation: Situation, cap: int, *, caching: bool, running: float | None = None
) -> float | None:
    """KV-cache usage at ``cap`` running sequences, scaled from the highest sampled usage at
    the highest sampled running count (when the engine has no running gauge, the configured
    cap, else ``running``). A sequence holds its unshared prompt and half its output; with
    prefix caching the shared prefix is held once, not once per sequence. Prompts are
    counted in words, which run about a quarter below tokens, while the output is in tokens;
    only ratios of these amounts are used, so the bias is that prompt-heavy sequences project
    slightly low."""
    m = situation.measurement
    running = m.peak_running or situation.settings.max_concurrent_requests or running
    if m.peak_kv_usage is None or not running:
        return None
    prefix, rest = prefix_and_rest(situation.facts)
    cached_now = situation.settings.prefix_caching is True
    held_now = prefix + running * rest if cached_now else running * (prefix + rest)
    held = prefix + cap * rest if caching else cap * (prefix + rest)
    return m.peak_kv_usage * held / held_now if held_now else None


def _concurrency_cap(signals: Measurement, settings: Settings) -> int | None:
    """The concurrency limit; when the engine chose it, the most that ever ran at once."""
    if settings.max_concurrent_requests:
        return settings.max_concurrent_requests
    return None if signals.peak_running is None else int(signals.peak_running)


def _capped_and_queueing(signals: Measurement, settings: Settings) -> tuple[int, float] | None:
    """The concurrency cap and the highest KV usage, when running requests hit the cap with
    others waiting; None otherwise."""
    if settings.max_concurrent_requests is None or queued_at_cap(signals, settings) is False:
        return None
    cap = _concurrency_cap(signals, settings)
    if (
        cap is not None
        and signals.peak_running is not None
        and signals.peak_running >= cap
        and above(signals.peak_waiting, 0)
        and signals.peak_kv_usage is not None
    ):
        return cap, signals.peak_kv_usage
    return None


def _kv_allows_raise(situation: Situation, cap: int, kv: float) -> bool:
    hybrid = blocks_per_request(situation.measurement) is not None
    return (hybrid or kv < KV_HEADROOM_USAGE) and _raised_concurrency(situation, cap) > cap


def raise_concurrency_blocked(situation: Situation, engine: Engine) -> str | None:
    """Why raising the cap does not apply: queueing at the cap, but too much KV in use."""
    queued = _capped_and_queueing(situation.measurement, situation.settings)
    if queued is None or _kv_allows_raise(situation, *queued):
        return None
    cap, kv = queued
    if (hybrid := _hybrid_pool(situation.measurement)) is not None:
        usable, per_request, held = hybrid
        return (
            f"raising max_concurrent_requests above {cap} is blocked: requests queue at the cap "
            f"and the {usable} usable blocks of this hybrid cache hold only {held} requests of "
            f"{per_request} blocks"
        )
    return (
        f"raising max_concurrent_requests above {cap} is blocked: requests queue at the cap and "
        f"the highest sampled KV-cache usage is {kv:.0%}, too high for a larger cap to fit"
    )


def raise_concurrency_applies(situation: Situation, engine: Engine) -> str | None:
    signals = situation.measurement
    queued = _capped_and_queueing(signals, situation.settings)
    if queued is not None and _kv_allows_raise(situation, *queued):
        cap, kv = queued
        observed = (
            f"up to {_demand(signals, cap)} requests were in flight"
            if signals.peak_in_flight is not None
            else f"highest sampled running plus waiting was {_demand(signals, cap)}"
        )
        hybrid = _hybrid_pool(signals)
        room = (
            f"the highest sampled KV-cache usage is only {kv:.1%}"
            if hybrid is None
            else f"the highest sampled KV-cache usage is {kv:.1%}, and on this hybrid cache "
            f"the {hybrid[0]} usable blocks hold {hybrid[2]} requests of {hybrid[1]} blocks"
        )
        return (
            f"running requests hit max_concurrent_requests {cap} with "
            f"{signals.peak_waiting:.0f} waiting, and {room}; at your declared load of "
            f"{situation.facts.load}, "
            f"{observed}, so "
            f"{_raised_concurrency(situation, cap)} is worth trying (the load is what your "
            "configuration declares, not measured traffic). It usually cuts queueing (TTFT) but "
            "slows each token (TPOT) as more sequences share every step - measure both"
        )
    return None


def _lowered_concurrency(cap: int) -> int:
    """Half the cap, the step that frees KV cache when sequences are being preempted."""
    return max(cap // 2, 1)


def with_lowered_concurrency(situation: Situation, engine: Engine) -> Settings:
    cap = _concurrency_cap(situation.measurement, situation.settings) or 2
    return replace(situation.settings, max_concurrent_requests=_lowered_concurrency(cap))


def with_raised_concurrency(situation: Situation, engine: Engine) -> Settings:
    signals, settings = situation.measurement, situation.settings
    raised = _raised_concurrency(situation, _concurrency_cap(signals, settings) or 1)
    return replace(settings, max_concurrent_requests=raised)


def lower_concurrency_applies(situation: Situation, engine: Engine) -> str | None:
    cap = _concurrency_cap(situation.measurement, situation.settings) or 0
    if cap <= 1 or not situation.fired("kv_capacity"):
        return None
    finding = situation.finding("kv_capacity")
    if finding.threshold == "preemption_rate":
        return f"{finding.evidence}; fewer concurrent sequences fit the KV cache"
    return None


# --- caching -------------------------------------------------------------------------


def _common_words(first: Sequence[str], second: Sequence[str]) -> int:
    count = 0
    for a, b in zip(first, second):
        if a != b:
            break
        count += 1
    return count


def shared_lengths(prompts: Sequence[str]) -> list[int]:
    """For each prompt, in order, the words it shares as a prefix with another prompt.

    The longest prefix a prompt shares with any other is the one it shares with a neighbour
    once the prompts are sorted, so prompts are compared in groups, not against one global
    prefix.
    """
    split = [prompt.split() for prompt in prompts]
    order = sorted(range(len(split)), key=split.__getitem__)
    shared = [0] * len(split)
    for position, index in enumerate(order):
        neighbours = order[max(position - 1, 0) : position] + order[position + 1 : position + 2]
        shared[index] = max((_common_words(split[index], split[n]) for n in neighbours), default=0)
    return shared


def prefix_caching_applies(situation: Situation, engine: Engine) -> str | None:
    facts = situation.facts
    total = len(facts.prompts)
    if situation.settings.prefix_caching is not False or not total:
        return None
    cacheable = engine.capabilities(None).min_cacheable_prefix_tokens
    sharing = longest = 0
    for prompt, prefix in zip(facts.prompts, shared_lengths(facts.prompts)):
        if prefix >= cacheable and (
            prefix >= MIN_SHARED_PREFIX_TOKENS
            or prefix >= MIN_SHARED_PREFIX_FRACTION * len(prompt.split())
        ):
            sharing += 1
            longest = max(longest, prefix)
    if sharing >= MIN_SHARING_PROMPTS * total:
        return (
            f"{sharing} of {total} prompts ({sharing / total:.0%}) share a prompt prefix of "
            f"up to {longest} words with another prompt"
        )
    return None


def caching_frees_kv(situation: Situation, engine: Engine) -> str | None:
    """Why prefix caching unblocks a raised cap: KV usage blocks it, caching is off, concurrent
    requests share one cached prefix copy on this engine, and the prompts share enough of their
    words that holding the prefix once makes double the cap fit."""
    if not engine.capabilities(None).shared_prefix_kv:
        return None
    queued = _capped_and_queueing(situation.measurement, situation.settings)
    blocked = raise_concurrency_blocked(situation, engine)
    if queued is None or blocked is None or situation.settings.prefix_caching is not False:
        return None
    prefix, rest = prefix_and_rest(situation.facts)
    share = prefix / (prefix + rest - situation.facts.output_tokens / 2) if prefix else 0.0
    if share < PREFIX_SHARE_FOR_KV:
        return None
    raised = 2 * queued[0]
    projected = projected_kv(situation, raised, caching=True)
    if projected is None or projected > KV_TARGET_USAGE:
        return None
    return (
        f"{blocked}; prompts share about {prefix:.0f} words ({share:.0%} of each prompt), "
        f"which prefix caching holds once, so {raised} sequences would use about "
        f"{projected:.0%} of the KV cache"
    )


def with_tool_calling(situation: Situation, engine: Engine) -> Settings:
    return replace(situation.settings, tool_calling=True)


def with_engine_quantization(situation: Situation, engine: Engine) -> Settings:
    return replace(situation.settings, quantization=None)


def with_fitting_context(situation: Situation, engine: Engine) -> Settings:
    fitting = fitting_max_context_len(situation.measurement, situation.facts)
    return replace(situation.settings, max_context_len=fitting)


def without_media_encoders(situation: Situation, engine: Engine) -> Settings:
    return replace(situation.settings, media_inputs=False)


def with_trimmed_context(situation: Situation, engine: Engine) -> Settings:
    trimmed = trimmed_max_context_len(situation.measurement, situation.facts)
    return replace(situation.settings, max_context_len=trimmed)


def measured_context_len(situation: Situation, engine: Engine) -> float | None:
    return situation.measurement.max_context_len


def concurrency_cap(situation: Situation, engine: Engine) -> float | None:
    return _concurrency_cap(situation.measurement, situation.settings)


def with_prefix_caching(situation: Situation, engine: Engine) -> Settings:
    return replace(situation.settings, prefix_caching=True)


def low_bar_only(situation: Situation, engine: Engine) -> str | None:
    """Never on its own signal: an entry with this ``applies`` is offered only by its low bar."""
    return None


# The neutral predicates, by the names data/playbook.toml uses.
PREDICATES: dict[str, Callable[[Situation, Engine], Any]] = {
    predicate.__name__: predicate
    for predicate in (
        enable_tool_calling_applies, with_tool_calling, fast_quant_kernel_applies,
        with_engine_quantization, fit_context_applies, with_fitting_context,
        no_media_encoders_applies, without_media_encoders, trim_max_context_len_applies,
        with_trimmed_context, measured_context_len, raise_concurrency_applies,
        with_raised_concurrency, concurrency_cap, raise_concurrency_blocked,
        lower_concurrency_applies, with_lowered_concurrency, prefix_caching_applies,
        with_prefix_caching, caching_frees_kv, low_bar_only,
    )
}  # fmt: skip
