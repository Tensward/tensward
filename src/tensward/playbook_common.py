"""Playbook entries that suit any engine: the setting changes Tensward suggests whatever the
engine, each tied to the bottlenecks it addresses. An engine's playbook lists them with its own."""

from __future__ import annotations

from dataclasses import replace
from statistics import median
from typing import TYPE_CHECKING, Sequence

from .classify import queued_at_cap
from .playbook import Entry, fitting_max_context_len, round_up_context
from .settings import Settings

if TYPE_CHECKING:
    from .measurement import Measurement
    from .playbook import Situation, WorkloadFacts

KV_HEADROOM_USAGE = 0.8  # more sequences only help below this KV usage
KV_TARGET_USAGE = 0.85  # a raised concurrency is sized to keep projected KV usage under this

# Prefix caching pays when many requests start with the same text. A prompt counts as sharing
# when it has a common prefix with another prompt that is long enough to cache (one 16-token
# KV block) and either 256 tokens or a quarter of its own length. Tokens are counted as
# whitespace-separated words, which never overstates a prefix.
MIN_CACHEABLE_PREFIX_TOKENS = 16
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


def _trim_max_context_len_applies(situation: Situation) -> str | None:
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


def _fit_context_applies(situation: Situation) -> str | None:
    signals, facts = situation.measurement, situation.facts
    if fitting_max_context_len(signals, facts) is None:
        return None
    longest = (signals.max_prompt_tokens or 0) + facts.output_tokens
    return (
        f"prompts that do not fit max_context_len {signals.max_context_len}: "
        f"{len(signals.too_long)}; the longest needs {longest} tokens"
    )


def _no_media_encoders_applies(situation: Situation) -> str | None:
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


def _enable_tool_calling_applies(situation: Situation) -> str | None:
    facts, settings = situation.facts, situation.settings
    if facts.offers_tools and not settings.tool_calling and settings.tool_parser is not None:
        return (
            "the workload offers tools but tool calling is off, so the server refuses them; "
            f"the checkpoint's tool parser is {settings.tool_parser}"
        )
    return None


# --- quantization ---------------------------------------------------------------------


def _fast_quant_kernel_applies(situation: Situation) -> str | None:
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


def _raised_concurrency(situation: Situation, current: int) -> int:
    """The observed demand (the client's in-flight peak when known, else running plus waiting),
    at least double, as a power of two, halved until its projected KV usage fits the target."""
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
    return kv < KV_HEADROOM_USAGE and _raised_concurrency(situation, cap) > cap


def raise_concurrency_blocked(situation: Situation) -> str | None:
    """Why raising the cap does not apply: queueing at the cap, but too much KV in use."""
    queued = _capped_and_queueing(situation.measurement, situation.settings)
    if queued is None or _kv_allows_raise(situation, *queued):
        return None
    cap, kv = queued
    return (
        f"raising max_concurrent_requests above {cap} is blocked: requests queue at the cap and "
        f"the highest sampled KV-cache usage is {kv:.0%}, too high for a larger cap to fit"
    )


def _raise_concurrency_applies(situation: Situation) -> str | None:
    signals = situation.measurement
    queued = _capped_and_queueing(signals, situation.settings)
    if queued is not None and _kv_allows_raise(situation, *queued):
        cap, kv = queued
        observed = (
            f"up to {_demand(signals, cap)} requests were in flight"
            if signals.peak_in_flight is not None
            else f"highest sampled running plus waiting was {_demand(signals, cap)}"
        )
        return (
            f"running requests hit max_concurrent_requests {cap} with "
            f"{signals.peak_waiting:.0f} waiting, and the highest sampled KV-cache usage is only "
            f"{kv:.1%}; at your declared load of {situation.facts.load}, "
            f"{observed}, so "
            f"{_raised_concurrency(situation, cap)} is worth trying (the load is what your "
            "configuration declares, not measured traffic). It usually cuts queueing (TTFT) but "
            "slows each token (TPOT) as more sequences share every step - measure both"
        )
    return None


def _lowered_concurrency(cap: int) -> int:
    """Half the cap, the step that frees KV cache when sequences are being preempted."""
    return max(cap // 2, 1)


def _with_lowered_concurrency(situation: Situation) -> Settings:
    cap = _concurrency_cap(situation.measurement, situation.settings) or 2
    return replace(situation.settings, max_concurrent_requests=_lowered_concurrency(cap))


def _with_raised_concurrency(situation: Situation) -> Settings:
    signals, settings = situation.measurement, situation.settings
    raised = _raised_concurrency(situation, _concurrency_cap(signals, settings) or 1)
    return replace(settings, max_concurrent_requests=raised)


def _lower_concurrency_applies(situation: Situation) -> str | None:
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


def _prefix_caching_applies(situation: Situation) -> str | None:
    facts = situation.facts
    total = len(facts.prompts)
    if situation.settings.prefix_caching is not False or not total:
        return None
    sharing = longest = 0
    for prompt, prefix in zip(facts.prompts, shared_lengths(facts.prompts)):
        if prefix >= MIN_CACHEABLE_PREFIX_TOKENS and (
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


def _caching_frees_kv(situation: Situation) -> str | None:
    """Why prefix caching unblocks a raised cap: KV usage blocks it, caching is off, and the
    prompts share enough of their words that holding the prefix once makes double the cap fit."""
    queued = _capped_and_queueing(situation.measurement, situation.settings)
    blocked = raise_concurrency_blocked(situation)
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


THROUGHPUT = frozenset({"throughput", "goodput", "req_s"})
LATENCY = frozenset({"ttft", "tpot"})


ENABLE_TOOL_CALLING = Entry(
    name="enable-tool-calling",
    addresses=frozenset({"fit"}),
    applies=_enable_tool_calling_applies,
    apply=lambda situation: replace(situation.settings, tool_calling=True),
    evidence="ours",
    costs="none; the server starts accepting tool requests",
    basis="your prompts and settings",
)
FAST_QUANT_KERNEL = Entry(
    name="fast-quant-kernel",
    addresses=frozenset({"prefill", "decode_bandwidth"}),
    applies=_fast_quant_kernel_applies,
    apply=lambda situation: replace(situation.settings, quantization=None),
    evidence="moderate",
    costs="none expected",
    helps=THROUGHPUT | LATENCY,
)
FIT_CONTEXT = Entry(
    name="fit-context",
    addresses=frozenset({"fit"}),
    applies=_fit_context_applies,
    apply=lambda situation: replace(
        situation.settings,
        max_context_len=fitting_max_context_len(situation.measurement, situation.facts),
    ),
    evidence="strong",
    costs="more KV cache per request, so fewer fit at once",
)
NO_MEDIA_ENCODERS = Entry(
    name="no-media-encoders",
    addresses=frozenset({"kv_capacity"}),
    applies=_no_media_encoders_applies,
    apply=lambda situation: replace(situation.settings, media_inputs=False),
    evidence="ours",
    costs="requests with images are rejected",
    helps=THROUGHPUT | LATENCY,
    basis="your prompts",
)
TRIM_MAX_CONTEXT_LEN = Entry(
    name="trim-max-context-len",
    addresses=frozenset({"kv_capacity"}),
    applies=_trim_max_context_len_applies,
    apply=lambda situation: replace(
        situation.settings,
        max_context_len=trimmed_max_context_len(situation.measurement, situation.facts),
    ),
    evidence="strong",
    costs="longer requests are refused",
    helps=THROUGHPUT,
    steps_on="max_context_len",
    start=lambda situation: situation.measurement.max_context_len,
)
RAISE_CONCURRENCY = Entry(
    name="raise-concurrency",
    addresses=frozenset({"queueing"}),
    applies=_raise_concurrency_applies,
    apply=_with_raised_concurrency,
    evidence="strong",
    costs="TPOT rises as more sequences share each step; more KV cache in use",
    helps=THROUGHPUT | {"ttft"},
    steps_on="max_concurrent_requests",
    start=lambda situation: _concurrency_cap(situation.measurement, situation.settings),
    blocked=raise_concurrency_blocked,
)
LOWER_CONCURRENCY = Entry(
    name="lower-concurrency",
    addresses=frozenset({"kv_capacity", "gpu_compute"}),
    applies=_lower_concurrency_applies,
    apply=_with_lowered_concurrency,
    evidence="strong",
    costs="requests queue if the load grows (TTFT up, throughput down)",
    helps=LATENCY,
)
PREFIX_CACHING = Entry(
    name="prefix-caching",
    addresses=frozenset({"prefill"}),
    applies=_prefix_caching_applies,
    apply=lambda situation: replace(situation.settings, prefix_caching=True),
    evidence="strong",
    costs="a little GPU memory for the cache",
    helps=THROUGHPUT | {"ttft"},
    basis="your prompts and settings",
    unblocks=_caching_frees_kv,
    unblocks_for=frozenset({"queueing"}),
)
