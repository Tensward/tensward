"""Which bottleneck held a run back, read from what the run measured.

Each bottleneck class has one rule. A rule says ``critical`` or ``warning`` when its signal
crosses a threshold, ``clear`` when it does not, and ``cant_tell`` when a signal it needs was not
measured (naming how to measure it). The primary bottleneck is the most severe crossing; the
others that crossed are secondary. Quality and fit are gates, reported beside the
bottleneck, never instead of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, Sequence

from .calibration import UNCALIBRATED, CalibrationProfile, profile_for
from .capabilities import Capabilities, Resolved, SignalSupport
from .ceilings import Ceilings, weight_bytes_read
from .measurement import ENGINE_UNMEASURED, WAVE_OUTLASTED_CHECK, GroupStats, Measurement
from .platforms import PLATFORMS
from .pressure import PROMPT_SHARE, cap_full, load_limited, state_of, step_full
from .profiling.trace import TraceSummary
from .settings import Settings
from .thresholds import MIN_CONFIDENT_REQUESTS, MIN_PREEMPTIONS, THRESHOLDS, Threshold

if TYPE_CHECKING:
    from .engines.protocol import Engine
    from .playbook import WorkloadFacts

Bottleneck = Literal[
    "load_limited", "queueing", "kv_capacity", "prefill", "step_budget", "prefill_stalls_decode",
    "decode_bandwidth", "gpu_compute", "host_overhead", "frontend_cpu", "offload", "long_context",
    "speculation", "multi_gpu", "quality", "fit",
]  # fmt: skip
State = Literal["critical", "warning", "clear", "cant_tell"]
AnswersState = Literal["equal", "differ", "within_noise", "empty", "inconclusive"]

NAMES: dict[Bottleneck, str] = {
    "load_limited": "the load generator",
    "queueing": "queueing before scheduling",
    "kv_capacity": "KV-cache capacity",
    "prefill": "prefill compute",
    "step_budget": "the per-step token budget",
    "prefill_stalls_decode": "prefill stalling decode",
    "decode_bandwidth": "decode memory bandwidth",
    "gpu_compute": "GPU compute, tensor-bound kernels",
    "host_overhead": "host / CPU overhead",
    "frontend_cpu": "API-server CPU",
    "offload": "offload / PCIe",
    "long_context": "attention / long context",
    "speculation": "speculation that does not pay",
    "multi_gpu": "multi-GPU communication",
    "quality": "quality",
    "fit": "fit and failed requests",
}
NOT_MODELLED: dict[Bottleneck, str] = {
    "offload": "needs offload signals, which come with the llama.cpp engine",
    "multi_gpu": "Tensward measures one GPU",
}
SEVERITY = {"critical": 2, "warning": 1}
# What a class needs that runs recorded before the time-averaged measures lack: a can't-tell for
# this reason is kept in the data and left out of the printed report.
UNMEASURED: dict[Bottleneck, str] = {
    "load_limited": "the time-averaged running, waiting and in-flight requests, KV use, step "
    "tokens against the budget and the prompt-work share",
    "step_budget": "the step budget (from the start-up log) and step tokens",
}


@dataclass(frozen=True, slots=True)
class Finding:
    bottleneck: Bottleneck
    state: State
    evidence: str  # what was measured; for "cant_tell", what is missing and how to measure it
    threshold: str | None = None  # the Threshold that decided the state
    margin: float = 0.0  # the value over the warning level: 1.0 at the level
    calibrated: bool = False
    confidence: str | None = None  # set by classify for a crossing
    near: bool = False  # not crossed, but within NEAR_THRESHOLD of the warning level


@dataclass(frozen=True, slots=True)
class Diagnosis:
    findings: tuple[Finding, ...]  # one per modelled class, in NAMES order
    gates: tuple[Finding, ...]  # quality (only when answers were compared) and fit
    shape: str | None  # the prompt:output mix, in words
    primary: Bottleneck | None
    confidence: str | None
    secondary: tuple[Bottleneck, ...]
    requests: int  # successful requests the diagnosis rests on
    symptoms: tuple[Bottleneck, ...] = ()  # crossed, but caused by the primary bottleneck


def finding_at(
    bottleneck: Bottleneck, threshold: Threshold, value: float, evidence: str
) -> Finding:
    return Finding(
        bottleneck,
        threshold.crossed(value),
        evidence,
        threshold.name,
        threshold.margin(value),
        threshold.calibrated,
        near=threshold.near(value),
    )


def _missing(bottleneck: Bottleneck, what: str) -> Finding:
    return Finding(bottleneck, "cant_tell", what)


def _fired(finding: Finding) -> bool:
    return finding.state in SEVERITY


def _confidence(finding: Finding, requests: int) -> str | None:
    """None when the finding did not cross or no request succeeded."""
    if not requests or not _fired(finding):
        return None
    if requests < MIN_CONFIDENT_REQUESTS:
        return "possible"
    if finding.state == "critical":
        return "high" if finding.calibrated else "likely"
    return "likely" if finding.calibrated else "possible"


@dataclass(frozen=True, slots=True, kw_only=True)
class RuleInputs:
    """What an engine's own rule reads: the run, the effective settings, the workload's facts
    (None for a caller that passed none) and what the engine chose at launch."""

    measurement: Measurement
    settings: Settings
    facts: WorkloadFacts | None
    resolved: Resolved | None
    # The thresholds at the run's calibration profile
    levels: Mapping[str, Threshold] = field(hash=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class EngineRule:
    """A diagnosis rule only one engine has. Its finding joins the class's candidates; a
    ``cant_tell`` finding replaces the class's finding."""

    bottleneck: Bottleneck
    rule: Callable[[RuleInputs], Finding | None]


@dataclass(frozen=True, slots=True)
class Need:
    """An input a class needs: engine signals (all of them, or any one with ``any``) or a named
    check of the engine, its settings and the run."""

    signals: tuple[str, ...] = ()
    any: bool = False
    check: str | None = None


SINGLE_ENGINE = Need(check="single_engine")
ON_DEVICE = Need(check="weights_on_device")
CEILINGS = Need(check="ceilings")
NEEDS: dict[Bottleneck, tuple[Need, ...]] = {
    "load_limited": (SINGLE_ENGINE,),
    "step_budget": (SINGLE_ENGINE,),
    "queueing": (Need(("queue_seconds", "prefill_seconds", "waiting")), SINGLE_ENGINE),
    "kv_capacity": (Need(("preemptions", "kv_usage"), any=True), SINGLE_ENGINE),
    "decode_bandwidth": (CEILINGS, ON_DEVICE, SINGLE_ENGINE),
    "prefill_stalls_decode": (SINGLE_ENGINE,),
    "gpu_compute": (Need(check="counters"),),
    "host_overhead": (Need(check="trace"),),
    "frontend_cpu": (
        Need(("frontend_cpu_seconds",)),
        SINGLE_ENGINE,
        Need(check="frontend_process"),
    ),
    "long_context": (CEILINGS, ON_DEVICE, SINGLE_ENGINE),
    "speculation": (Need(check="speculation_reported"),),
}
# The engine signals each derived Measurement field the classes read rests on (client timings
# rest on none; the decode batch on the engine's step and token counters).
DERIVED_FROM: dict[str, tuple[str, ...]] = {
    "queue_share": ("queue_seconds", "prefill_seconds"),
    "peak_waiting": ("waiting",),
    "peak_kv_usage": ("kv_usage",),
    "preemptions": ("preemptions",),
    "frontend_cpu_cores": ("frontend_cpu_seconds",),
    "spec_acceptance_length": ("spec_drafts", "spec_accepted_tokens"),
    "spec_coverage": ("spec_accepted_tokens", "generation_tokens"),
    "ceilings.avg_running_batch": ("generation_tokens", "iterations"),
}
# The derived field each threshold judges. A finding decided by a threshold whose signals the
# engine reports only approximately is capped at "possible" and marked estimated.
THRESHOLD_SIGNALS: dict[str, tuple[str, ...]] = {
    "queue_share": DERIVED_FROM["queue_share"],
    "preemption_rate": DERIVED_FROM["preemptions"],
    "kv_peak": DERIVED_FROM["peak_kv_usage"],
    "ttft_over_tpot": (),
    "tpot_tail": (),
    "decode_of_ceiling": DERIVED_FROM["ceilings.avg_running_batch"],
    "tensor_busy": (),
    "gpu_idle": (),
    "kv_over_weights": DERIVED_FROM["ceilings.avg_running_batch"],
    "spec_acceptance": DERIVED_FROM["spec_acceptance_length"],
    "spec_coverage": DERIVED_FROM["spec_coverage"],
    "frontend_cpu": DERIVED_FROM["frontend_cpu_cores"],
    "prompt_work": ("prompt_tokens_computed",),
}
ABSENT = SignalSupport(quality="absent")


@dataclass(frozen=True, slots=True)
class _Run:
    engine: Engine
    capabilities: Capabilities
    measurement: Measurement
    settings: Settings
    resolved: Resolved | None
    platform: str | None

    def support(self, signal: str) -> SignalSupport:
        return self.capabilities.signals.get(signal, ABSENT)

    def estimated(self, signals: Sequence[str]) -> bool:
        return any(self.support(signal).quality == "approximate" for signal in signals)


def data_parallel_note(replicas: int) -> str:
    return f"{replicas} data-parallel engines; per-engine limits are not modelled"


def _single_engine(run: _Run) -> str | None:
    replicas = run.measurement.replicas
    return data_parallel_note(replicas) if replicas > 1 else None


def _frontend_process(run: _Run) -> str | None:
    if (run.engine.frontend_processes(run.settings) or 1) <= 1:
        return None
    return "several API servers: the engine's CPU counter covers one process"


def _weights_on_device(run: _Run) -> str | None:
    if run.engine.weights_on_device(run.settings, run.resolved) is not False:
        return None
    return (
        "weights are split between the device and host memory; the ceiling does not model offload"
    )


def _ceilings(run: _Run) -> str | None:
    """Absent only when the platform has no way at all to give a bandwidth figure; an unknown
    platform, or a GPU missing from the datasheet table, keeps the decode rule's own texts."""
    if run.platform is None:
        return None
    platform = next((p for p in PLATFORMS if p.name == run.platform), None)
    if platform is not None and platform.bandwidth_figures:
        return None
    devices = platform.label if platform is not None else run.platform
    return f"the decode ceiling: Tensward has no bandwidth figure for {devices} devices yet"


def _trace(run: _Run) -> str | None:
    if run.capabilities.trace == "kineto":
        return None
    return f"a GPU trace: {run.engine.label} does not write one Tensward reads"


def _counters(run: _Run) -> str | None:
    if run.capabilities.counters:
        return None
    return f"kernel counters: {run.engine.label} cannot gate a kernel profiler"


def _speculation_reported(run: _Run) -> str | None:
    if run.engine.lever_value(run.settings, "speculation") is None:
        return None
    reported = ("spec_drafts", "spec_accepted_tokens")
    if all(run.support(signal).quality != "absent" for signal in reported):
        return None
    return f"{run.engine.label} does not report how many speculated tokens were accepted"


CHECKS: dict[str, Callable[[_Run], str | None]] = {
    "single_engine": _single_engine,
    "frontend_process": _frontend_process,
    "weights_on_device": _weights_on_device,
    "trace": _trace,
    "counters": _counters,
    "ceilings": _ceilings,
    "speculation_reported": _speculation_reported,
}


def _absent(bottleneck: Bottleneck, run: _Run) -> str | None:
    """Why ``bottleneck`` cannot be judged on this engine, platform, settings or replica count,
    naming who lacks the input; None when it can. A supported signal this run did not yield
    still goes through the class's rule."""
    for need in NEEDS.get(bottleneck, ()):
        lacking = [signal for signal in need.signals if run.support(signal).quality == "absent"]
        if lacking and (not need.any or len(lacking) == len(need.signals)):
            reason = run.support(lacking[0]).absent_reason
            return reason or f"{run.engine.label} does not report {lacking[0]}"
        if need.check is not None and (refused := CHECKS[need.check](run)) is not None:
            return refused
    return None


def _leveled(threshold: Threshold, profile: CalibrationProfile) -> Threshold:
    """``threshold`` at the profile's levels, calibrated only where the profile was."""
    warning, critical = profile.levels.get(threshold.name, (threshold.warning, threshold.critical))
    calibrated = threshold.name in profile.calibrated
    return replace(threshold, warning=warning, critical=critical, calibrated=calibrated)


def _rests_on_estimate(finding: Finding, run: _Run) -> bool:
    return run.estimated(THRESHOLD_SIGNALS.get(finding.threshold or "", ()))


def _estimated(finding: Finding, run: _Run) -> Finding:
    """A finding whose deciding threshold rests on an approximate signal: at most "possible",
    and its evidence says so."""
    if not _rests_on_estimate(finding, run):
        return finding
    confidence = "possible" if finding.confidence else None
    return replace(finding, evidence=f"{finding.evidence} (estimated)", confidence=confidence)


def _clears(kv: Finding, run: _Run) -> bool:
    """Whether a KV finding may clear queueing and prefill stalling decode: only a crossing
    that rests on exact signals."""
    return _fired(kv) and not _rests_on_estimate(kv, run)


def _queue_share_trusted(m: Measurement, run: _Run) -> bool:
    """An estimated queue share counts only when an exact waiting signal saw a request wait."""
    if not run.estimated(DERIVED_FROM["queue_share"]):
        return True
    waiting = run.support(DERIVED_FROM["peak_waiting"][0])
    return waiting.quality in ("exact", "derived") and bool(m.peak_waiting)


def _untimed(m: Measurement) -> str | None:
    """Why the client timings cannot be judged: too few requests left once the engine's
    untimed ones are excluded."""
    timed = m.succeeded - m.untimed_requests
    if not m.untimed_requests or timed >= MIN_CONFIDENT_REQUESTS:
        return None
    return f"client timings: only {timed} of {m.succeeded} answers were timed by the engine's rules"


def _untimed_share(m: Measurement) -> str:
    if not m.untimed_requests:
        return ""
    return f"; {m.untimed_requests} of {m.succeeded} answers were left out of the timings"


def _strongest(found: list[Finding]) -> Finding:
    return max(found, key=lambda f: (SEVERITY.get(f.state, 0), f.margin))


def _judged(
    bottleneck: Bottleneck,
    candidates: Callable[[], list[Finding]],
    absent: str | None,
    ruled: list[Finding],
    missing: str = "",
) -> Finding:
    """The class's finding: an engine rule's can't-tell wins; else the strongest of the neutral
    candidates and the engine rules' crossings, neutral candidates first, so a tie keeps the
    neutral finding. A class whose needs are absent reports no crossing, its own or an
    engine rule's."""
    told = next((finding for finding in ruled if finding.state == "cant_tell"), None)
    if told is not None:
        return told
    found = [] if absent else [*candidates(), *ruled]
    return _strongest(found) if found else _missing(bottleneck, absent or missing)


def _kv_capacity(m: Measurement, levels: Mapping[str, Threshold]) -> list[Finding]:
    """Preemptions and the peak usage, each a candidate for the class."""
    found = []
    if m.preemptions is not None:
        # An outlasted wave's preemptions are counted from before it, so its requests count too.
        wave = m.startup_wave.requests if m.startup_wave and WAVE_OUTLASTED_CHECK in m.checks else 0
        requests = max(m.succeeded + m.failed + wave, 1)
        rate = m.preemptions / requests if m.preemptions >= MIN_PREEMPTIONS else 0.0
        evidence = f"preemptions: {m.preemptions:.0f} over {requests} requests"
        if wave:
            evidence += f", the start-up wave's {wave} included"
        found.append(finding_at("kv_capacity", levels["preemption_rate"], rate, evidence))
    if m.peak_kv_usage is not None:
        evidence = f"highest sampled KV-cache usage {m.peak_kv_usage:.0%}"
        found.append(finding_at("kv_capacity", levels["kv_peak"], m.peak_kv_usage, evidence))
    return found


def queued_at_cap(m: Measurement, settings: Settings) -> bool | None:
    """Whether the client had more requests in flight than the concurrency cap lets run; None
    when the cap was left to the engine or the in-flight peak is unknown."""
    cap = settings.max_concurrent_requests
    if cap is None or m.peak_in_flight is None:
        return None
    return m.peak_in_flight > cap


def _where_queued(m: Measurement, settings: Settings) -> str:
    """Why requests queued, as far as the measurement tells, in words."""
    at_cap = queued_at_cap(m, settings)
    if at_cap is None:
        if settings.max_concurrent_requests is None:
            return (
                "the engine chose the concurrency cap, so it cannot say whether the cap held them"
            )
        return (
            "the client's in-flight peak was not measured, so it cannot say whether the cap "
            "held them"
        )
    cap, in_flight = settings.max_concurrent_requests, m.peak_in_flight
    if at_cap:
        return f"{in_flight} requests were in flight against a concurrency cap of {cap}"
    return (
        f"at most {in_flight} requests were in flight, within the cap of {cap}, so they waited "
        "to be admitted to prefill, not for the cap"
    )


def _queueing(
    m: Measurement, settings: Settings, kv: Finding, levels: Mapping[str, Threshold], run: _Run
) -> Finding:
    if m.queue_share is None:
        return _missing("queueing", "the engine's per-request queue time")
    if not _queue_share_trusted(m, run):
        return _missing("queueing", "the queue time is estimated and no request was seen waiting")
    if _clears(kv, run):
        return Finding(
            "queueing", "clear", "requests waited for KV-cache space, not to be scheduled"
        )
    if m.peak_waiting is None:
        return _missing("queueing", "the engine's waiting-request gauge")
    unsampled = not m.peak_waiting
    queued = levels["queue_share"]
    if unsampled and queued.crossed(m.queue_share) == "clear":
        return finding_at("queueing", queued, m.queue_share, "no waiting request was sampled")
    return finding_at(
        "queueing",
        queued,
        m.queue_share,
        f"requests spent {m.queue_share:.0%} of their time to first token queued; "
        f"{_where_queued(m, settings)}"
        + ("; the waiting-request gauge never sampled a waiting request" if unsampled else ""),
    )


def _prompt_bound(m: Measurement, levels: Mapping[str, Threshold]) -> Finding | None:
    share = m.prompt_work_share
    if share is None or share < PROMPT_SHARE:
        return None
    evidence = f"prompt processing took {share:.0%} of the engine's measured prompt capacity"
    if m.prompt_capacity_tok_s:
        evidence += f" ({m.prompt_capacity_tok_s:.0f} prompt tokens/s)"
    return finding_at("prefill", levels["prompt_work"], share, evidence)


def _prefill(
    m: Measurement, queueing: Finding, levels: Mapping[str, Threshold], run: _Run
) -> Finding:
    if (bound := _prompt_bound(m, levels)) is not None:
        return bound
    if m.ttft_p50_ms is None or not m.tpot_p50_ms:
        return _missing("prefill", "TTFT and TPOT: no request produced two tokens")
    if (untimed := _untimed(m)) is not None:
        return _missing("prefill", untimed)
    queued = levels["queue_share"]
    if queueing.state == "critical" or (
        m.queue_share is not None
        and _queue_share_trusted(m, run)
        and queued.crossed(m.queue_share) == "critical"
    ):
        return Finding("prefill", "clear", "TTFT is mostly time spent queued")
    if ENGINE_UNMEASURED in m.checks:
        return _missing(
            "prefill",
            "TTFT may be time spent queued or waiting for KV-cache space, which this run did not "
            "measure",
        )
    ratio = m.ttft_p50_ms / m.tpot_p50_ms
    evidence = f"TTFT p50 {m.ttft_p50_ms:.0f} ms is {ratio:.0f}x TPOT p50 {m.tpot_p50_ms:.1f} ms"
    return finding_at("prefill", levels["ttft_over_tpot"], ratio, evidence + _untimed_share(m))


def _load_limited(m: Measurement, settings: Settings) -> Finding:
    state = state_of(m, settings.max_concurrent_requests)
    limited = load_limited(state)
    if limited is None:
        return _missing("load_limited", UNMEASURED["load_limited"])
    if not limited:
        return Finding("load_limited", "clear", "requests waited or a resource neared its limit")
    return Finding(
        "load_limited", "critical",
        f"the server ran {state.running:.0f} of the {state.in_flight:.0f} requests the test kept "
        f"in flight, none waited, and KV use ({state.kv:.0%}), step tokens and prompt work were "
        "well under their limits: the test did not push the server",
        margin=1.0,
    )  # fmt: skip


def _step_budget(m: Measurement, settings: Settings) -> Finding:
    state = state_of(m, settings.max_concurrent_requests)
    full = step_full(state)
    if full is None:
        return _missing("step_budget", UNMEASURED["step_budget"])
    evidence = (
        f"steps carried {state.scheduled:.0f} tokens on average against a budget of "
        f"{state.budget}, with {state.waiting:.0f} requests waiting"
    )
    return Finding(
        "step_budget", "critical" if full else "clear", evidence, margin=1.0 if full else 0.0
    )


def _interference(
    m: Measurement, kv: Finding, levels: Mapping[str, Threshold], run: _Run
) -> Finding:
    if m.tpot_p95_ms is None or not m.tpot_p50_ms:
        return _missing("prefill_stalls_decode", "TPOT percentiles: no request produced two tokens")
    if (untimed := _untimed(m)) is not None:
        return _missing("prefill_stalls_decode", untimed)
    if _clears(kv, run):
        return Finding("prefill_stalls_decode", "clear", "TPOT spikes come with KV-cache pressure")
    if ENGINE_UNMEASURED in m.checks:
        return _missing(
            "prefill_stalls_decode",
            "TPOT spikes may come from KV-cache pressure, which this run did not measure",
        )
    ratio = m.tpot_p95_ms / m.tpot_p50_ms
    evidence = f"TPOT p95 {m.tpot_p95_ms:.1f} ms is {ratio:.1f}x p50 {m.tpot_p50_ms:.1f} ms"
    return finding_at(
        "prefill_stalls_decode", levels["tpot_tail"], ratio, evidence + _untimed_share(m)
    )


def _bandwidth(m: Measurement, levels: Mapping[str, Threshold]) -> Finding:
    c = m.ceilings
    if m.spec_acceptance_length is not None:
        return _missing(
            "decode_bandwidth", "the decode ceiling does not model speculative decoding"
        )
    if c is not None and "decode" in " ".join(c.exceeds_bound):
        return _missing(
            "decode_bandwidth", "a trustworthy decode ceiling: the measured rate exceeded it"
        )
    if c is None:
        return _missing(
            "decode_bandwidth", "the decode ceiling: needs a known GPU and the model's anatomy"
        )
    if c.unavailable:
        return _missing("decode_bandwidth", f"the decode ceiling: {c.unavailable}")
    if c.decode_pct_of_ceiling is None:
        return _missing(
            "decode_bandwidth", "the decode ceiling at the measured batch: no running batch sampled"
        )
    return finding_at(
        "decode_bandwidth",
        levels["decode_of_ceiling"],
        c.decode_pct_of_ceiling,
        f"decode ran at {c.decode_pct_of_ceiling:.0f}% of the memory-bandwidth ceiling at the "
        "measured batch",
    )


def _compute(m: Measurement, levels: Mapping[str, Threshold]) -> Finding:
    if m.counters is None:
        return _missing("gpu_compute", "kernel counters: run with `--counters`")
    measured = [
        (kernel, tensor, dram)
        for kernel in m.counters.kernels
        if (tensor := kernel.values.get("tensor_pct")) is not None
        and (dram := kernel.values.get("dram_pct")) is not None
    ]
    if not measured:
        return _missing("gpu_compute", f"kernel counters: {m.counters.note or m.counters.status}")
    top, tensor, dram = max(measured, key=lambda found: found[0].share)
    if dram >= tensor:
        return Finding(
            "gpu_compute",
            "clear",
            f"the busiest kernel is held by memory: {dram:.0f}% DRAM, {tensor:.0f}% tensor",
        )
    return finding_at(
        "gpu_compute",
        levels["tensor_busy"],
        tensor,
        f"the busiest kernel, {top.name} ({top.share:.0%} of GPU time, prefill or decode), keeps "
        f"the tensor pipe {tensor:.0f}% busy and DRAM {dram:.0f}%",
    )


def _host(m: Measurement, levels: Mapping[str, Threshold]) -> Finding:
    if m.trace is None:
        return _missing("host_overhead", "a GPU trace: run with `--trace`")
    if m.trace.status != "ok" or m.trace.idle_share is None:
        return _missing("host_overhead", f"a trusted GPU trace: {m.trace.note or m.trace.status}")
    return finding_at(
        "host_overhead",
        levels["gpu_idle"],
        m.trace.idle_share,
        f"the GPU sat idle {m.trace.idle_share:.0%} of the traced window",
    )


def _frontend_cpu(m: Measurement, levels: Mapping[str, Threshold]) -> Finding:
    cores = m.frontend_cpu_cores
    if cores is None:
        return _missing("frontend_cpu", "the engine's API-server CPU time")
    return finding_at(
        "frontend_cpu",
        levels["frontend_cpu"],
        cores,
        f"the API server used {cores:.2f} CPU cores over the window",
    )


def _long_context(m: Measurement, levels: Mapping[str, Threshold]) -> Finding:
    c = m.ceilings
    weights = weight_bytes_read(c) if c is not None else None
    if c is None or not weights or c.kv_bytes_per_sequence is None or c.avg_context_tokens is None:
        return _missing(
            "long_context",
            "the KV-cache and weight bytes of a decode step: "
            "needs a known GPU and the model's anatomy",
        )
    if not c.avg_running_batch:
        return _missing(
            "long_context",
            "the KV-cache and weight bytes of a decode step: no running batch sampled",
        )
    ratio = c.kv_bytes_per_sequence * c.avg_running_batch / weights
    return finding_at(
        "long_context",
        levels["kv_over_weights"],
        ratio,
        f"a decode step reads {ratio:.2f}x as many KV-cache bytes as weight bytes (average "
        f"context {c.avg_context_tokens:.0f} tokens, batch {c.avg_running_batch:.1f})",
    )


def _speculation(m: Measurement, levels: Mapping[str, Threshold]) -> Finding:
    if m.spec_acceptance_length is None:
        if ENGINE_UNMEASURED in m.checks:
            return _missing("speculation", "the engine's speculation counters")
        return Finding("speculation", "clear", "speculation is off or not reported")
    found = [
        finding_at(
            "speculation",
            levels["spec_acceptance"],
            m.spec_acceptance_length,
            f"each speculative draft yielded {m.spec_acceptance_length:.2f} tokens",
        )
    ]
    if m.spec_coverage is not None:
        evidence = f"accepted drafts made {m.spec_coverage:.0%} of the generated tokens"
        found.append(finding_at("speculation", levels["spec_coverage"], m.spec_coverage, evidence))
    return _strongest(found)


ANSWERS: dict[AnswersState, tuple[State, str]] = {
    "equal": ("clear", "answers match the current setup's"),
    "differ": ("critical", "answers differ from the current setup's (see the comparison below)"),
    "within_noise": (
        "clear",
        "answers differ from the current setup's only as much as its own repeated answers do",
    ),
    "empty": (
        "cant_tell",
        "answers were empty: every answer of the current setup or of this run is empty, so "
        "they were not compared",
    ),
    "inconclusive": (
        "cant_tell",
        "answers could not be judged: there is no noise floor to judge them against (the "
        "baseline has no repeated answers per prompt)",
    ),
}


def _fit(m: Measurement) -> Finding:
    problems = []
    if m.too_long:
        problems.append(
            f"prompts that do not fit max_context_len {m.max_context_len}: {len(m.too_long)}"
        )
    if m.kv_max_concurrency is not None and m.kv_max_concurrency < 1:
        problems.append(f"the KV cache holds {m.kv_max_concurrency:.2f} full-length requests")
    if m.failed:
        problems.append(f"requests that failed: {m.failed} of {m.succeeded + m.failed}")
    if problems:
        return Finding("fit", "critical", "; ".join(problems))
    return Finding("fit", "clear", "every request was served")


def _shape(m: Measurement) -> str | None:
    c = m.ceilings
    if c is None or not c.measured_prefill_tok_s or not c.measured_decode_tok_s:
        return None
    ratio = c.measured_prefill_tok_s / c.measured_decode_tok_s
    return f"{ratio:.1f} prompt tokens computed per generated token"


def classify(
    measurement: Measurement,
    settings: Settings,
    answers: AnswersState | None,
    *,
    engine: Engine | None = None,
    profile: CalibrationProfile | None = None,
    facts: WorkloadFacts | None = None,
    resolved: Resolved | None = None,
    version: str | None = None,
    platform: str | None = None,
) -> Diagnosis:
    """The diagnosis of one run with ``settings``. ``answers`` is the verdict of the comparison
    with the current setup, or None when none ran. ``engine`` None diagnoses as the
    vLLM adapter, and ``profile`` None takes the engine's default profile, so a caller
    diagnosing another engine passes ``engine``. ``platform`` (Platform.name) None: unknown."""
    if engine is None:
        from .engines.vllm import VLLM

        engine = VLLM
    profile = profile or profile_for(engine, None, None, None) or UNCALIBRATED
    levels = {threshold.name: _leveled(threshold, profile) for threshold in THRESHOLDS}
    run = _Run(
        engine, engine.capabilities(version, settings), measurement, settings, resolved, platform
    )
    inputs = RuleInputs(
        measurement=measurement, settings=settings, facts=facts, resolved=resolved, levels=levels
    )
    ruled: dict[Bottleneck, list[Finding]] = {name: [] for name in NAMES}
    for engine_rule in engine.engine_rules:
        if (found := engine_rule.rule(inputs)) is not None:
            ruled[engine_rule.bottleneck].append(found)

    def judged(
        name: Bottleneck, candidates: Callable[[], list[Finding]], missing: str = ""
    ) -> Finding:
        return _judged(name, candidates, _absent(name, run), ruled[name], missing)

    m = measurement
    kv = judged(
        "kv_capacity",
        lambda: _kv_capacity(m, levels),
        "the engine's preemptions and KV-cache usage",
    )
    queueing = judged("queueing", lambda: [_queueing(m, settings, kv, levels, run)])
    measured = (
        judged("load_limited", lambda: [_load_limited(m, settings)]),
        queueing,
        kv,
        judged("prefill", lambda: [_prefill(m, queueing, levels, run)]),
        judged("step_budget", lambda: [_step_budget(m, settings)]),
        judged("prefill_stalls_decode", lambda: [_interference(m, kv, levels, run)]),
        judged("decode_bandwidth", lambda: [_bandwidth(m, levels)]),
        judged("gpu_compute", lambda: [_compute(m, levels)]),
        judged("host_overhead", lambda: [_host(m, levels)]),
        judged("frontend_cpu", lambda: [_frontend_cpu(m, levels)]),
        judged("long_context", lambda: [_long_context(m, levels)]),
        judged("speculation", lambda: [_speculation(m, levels)]),
    )
    findings = tuple(
        _estimated(replace(f, confidence=_confidence(f, m.succeeded)), run) for f in measured
    )
    gates = [_fit(m)]
    if answers is not None:
        state, evidence = ANSWERS[answers]
        gates.insert(0, Finding("quality", state, evidence))
    fired = sorted(
        (f for f in findings if f.confidence),
        key=lambda f: (SEVERITY[f.state], f.margin),
        reverse=True,
    )
    named = {f.bottleneck: f for f in fired}
    prompt_bound = "prefill" in named and named["prefill"].threshold == "prompt_work"
    if "load_limited" in named:
        fired.sort(key=lambda f: f.bottleneck != "load_limited")
    elif prompt_bound:
        fired.sort(key=lambda f: f.bottleneck != "prefill")
    symptoms: tuple[Bottleneck, ...] = ()
    caused = prompt_bound or "step_budget" in named
    below_cap = cap_full(state_of(m, settings.max_concurrent_requests)) is False
    if caused and "queueing" in named and below_cap:
        fired.remove(named["queueing"])
        symptoms = ("queueing",)
    return Diagnosis(
        findings=findings,
        gates=tuple(gates),
        shape=_shape(m),
        primary=fired[0].bottleneck if fired else None,
        confidence=fired[0].confidence if fired else None,
        secondary=tuple(f.bottleneck for f in fired[1:]),
        requests=m.succeeded,
        symptoms=symptoms,
    )


def diagnose_run(
    measurement: Measurement,
    settings: Settings,
    answers: AnswersState | None,
    *,
    engine: Engine,
    platform: str | None,
    gpu: str | None,
    version: str | None,
    facts: WorkloadFacts | None = None,
    resolved: Resolved | None = None,
) -> Diagnosis:
    """The diagnosis ``measure`` and ``analyse`` write: the engine's, at the calibration profile
    for the platform, GPU and engine version the run had; uncalibrated when none matches."""
    profile = profile_for(engine, platform, gpu, version) or UNCALIBRATED
    return classify(
        measurement, settings, answers, engine=engine, profile=profile, facts=facts,
        resolved=resolved, version=version, platform=platform,
    )  # fmt: skip


def measurement_from_metrics(data: Mapping[str, Any]) -> Measurement:
    """The parts of a recorded ``metrics.json`` the classifier reads, as a Measurement, so a
    recorded run can be diagnosed again (with other thresholds, for calibration). Fields the
    classifier does not read are left at their defaults."""

    def build(kind: type[Any], value: Any) -> Any:
        if not isinstance(value, Mapping):
            return None
        names = {f.name for f in fields(kind)}
        return kind(**{k: v for k, v in value.items() if k in names})

    explicit = {"ceilings", "trace", "startup_wave", "too_long", "engine_signals", "checks"}
    scalars = {
        f.name: data[f.name]
        for f in fields(Measurement)
        if f.name in data
        and f.name not in explicit
        and not isinstance(data[f.name], (Mapping, list))
    }
    ceilings = build(Ceilings, data.get("ceilings"))
    if ceilings is not None:
        ceilings = replace(ceilings, exceeds_bound=tuple(ceilings.exceeds_bound or ()))
    return Measurement(
        **scalars,
        ceilings=ceilings,
        trace=build(TraceSummary, data.get("trace")),
        startup_wave=build(GroupStats, data.get("startup_wave")),
        too_long=tuple(data.get("too_long") or ()),
        checks=tuple(data.get("checks") or ()),
        engine_signals=dict(data.get("engine_signals") or {}),
    )
