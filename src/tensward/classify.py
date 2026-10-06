"""Which bottleneck held a run back, read from what the run measured.

Each bottleneck class has one rule. A rule says ``critical`` or ``warning`` when its signal
crosses a threshold, ``clear`` when it does not, and ``cant_tell`` when a signal it needs was not
measured (naming how to measure it). The primary bottleneck is the most severe crossing; the
others that crossed are secondary. Quality and fit are gates, reported beside the
bottleneck, never instead of it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any, Mapping

from .thresholds import (
    DECODE_OF_CEILING,
    GPU_IDLE,
    KV_OVER_WEIGHTS,
    KV_PEAK,
    MIN_CONFIDENT_REQUESTS,
    MIN_PREEMPTIONS,
    PREEMPTION_RATE,
    QUEUE_SHARE,
    SPEC_ACCEPTANCE,
    SPEC_COVERAGE,
    TENSOR_BUSY,
    TPOT_TAIL,
    TTFT_OVER_TPOT,
    Threshold,
)

if TYPE_CHECKING:
    from .engines.protocol import Settings
    from .measurement import Measurement

NAMES = {
    "queueing": "queueing before scheduling",
    "kv_capacity": "KV-cache capacity",
    "prefill": "prefill compute",
    "prefill_stalls_decode": "prefill stalling decode",
    "decode_bandwidth": "decode memory bandwidth",
    "gpu_compute": "GPU compute, tensor-bound kernels",
    "host_overhead": "host / CPU overhead",
    "offload": "offload / PCIe",
    "long_context": "attention / long context",
    "speculation": "speculation that does not pay",
    "multi_gpu": "multi-GPU communication",
    "quality": "quality",
    "fit": "fit and failed requests",
}
NOT_MODELLED = {
    "offload": "needs offload signals, which come with the llama.cpp engine",
    "multi_gpu": "Tensward measures one GPU",
}
SEVERITY = {"critical": 2, "warning": 1}


@dataclass(frozen=True, slots=True)
class Finding:
    bottleneck: str  # a NAMES key
    state: str  # "critical", "warning", "clear" or "cant_tell"
    evidence: str  # what was measured; for "cant_tell", what is missing and how to measure it
    threshold: str | None = None  # the Threshold that decided the state
    margin: float = 0.0  # the value over the warning level: 1.0 at the level
    calibrated: bool = False
    confidence: str | None = None  # set by classify for a crossing


@dataclass(frozen=True, slots=True)
class Diagnosis:
    findings: tuple[Finding, ...]  # one per modelled class, in NAMES order
    gates: tuple[Finding, ...]  # quality (only when answers were compared) and fit
    shape: str | None  # the prompt:output mix, in words
    primary: str | None
    confidence: str | None
    secondary: tuple[str, ...]
    requests: int  # successful requests the diagnosis rests on


def _finding(bottleneck: str, threshold: Threshold, value: float, evidence: str) -> Finding:
    return Finding(
        bottleneck,
        threshold.crossed(value),
        evidence,
        threshold.name,
        threshold.margin(value),
        threshold.calibrated,
    )


def _missing(bottleneck: str, what: str) -> Finding:
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


def _kv_capacity(m: Measurement) -> Finding:
    found = []
    if m.preemptions is not None:
        requests = max(m.succeeded + m.failed, 1)
        rate = m.preemptions / requests if m.preemptions >= MIN_PREEMPTIONS else 0.0
        evidence = f"preemptions: {m.preemptions:.0f} over {requests} requests"
        found.append(_finding("kv_capacity", PREEMPTION_RATE, rate, evidence))
    if m.peak_kv_usage is not None:
        evidence = f"highest sampled KV-cache usage {m.peak_kv_usage:.0%}"
        found.append(_finding("kv_capacity", KV_PEAK, m.peak_kv_usage, evidence))
    if not found:
        return _missing("kv_capacity", "the engine's preemptions and KV-cache usage")
    return max(found, key=lambda f: (SEVERITY.get(f.state, 0), f.margin))


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


def _queueing(m: Measurement, settings: Settings, kv: Finding) -> Finding:
    if m.queue_share is None:
        return _missing("queueing", "the engine's per-request queue time")
    if _fired(kv):
        return Finding(
            "queueing", "clear", "requests waited for KV-cache space, not to be scheduled"
        )
    if m.peak_waiting is None:
        return _missing("queueing", "the engine's waiting-request gauge")
    unsampled = not m.peak_waiting
    if unsampled and QUEUE_SHARE.crossed(m.queue_share) == "clear":
        return _finding("queueing", QUEUE_SHARE, m.queue_share, "no waiting request was sampled")
    return _finding(
        "queueing",
        QUEUE_SHARE,
        m.queue_share,
        f"requests spent {m.queue_share:.0%} of their time to first token queued; "
        f"{_where_queued(m, settings)}"
        + ("; the waiting-request gauge never sampled a waiting request" if unsampled else ""),
    )


def _prefill(m: Measurement, queueing: Finding) -> Finding:
    if m.ttft_p50_ms is None or not m.tpot_p50_ms:
        return _missing("prefill", "TTFT and TPOT: no request produced two tokens")
    if queueing.state == "critical" or (
        m.queue_share is not None and QUEUE_SHARE.crossed(m.queue_share) == "critical"
    ):
        return Finding("prefill", "clear", "TTFT is mostly time spent queued")
    ratio = m.ttft_p50_ms / m.tpot_p50_ms
    return _finding(
        "prefill",
        TTFT_OVER_TPOT,
        ratio,
        f"TTFT p50 {m.ttft_p50_ms:.0f} ms is {ratio:.0f}x TPOT p50 {m.tpot_p50_ms:.1f} ms",
    )


def _interference(m: Measurement, kv: Finding) -> Finding:
    if m.tpot_p95_ms is None or not m.tpot_p50_ms:
        return _missing("prefill_stalls_decode", "TPOT percentiles: no request produced two tokens")
    if _fired(kv):
        return Finding("prefill_stalls_decode", "clear", "TPOT spikes come with KV-cache pressure")
    ratio = m.tpot_p95_ms / m.tpot_p50_ms
    return _finding(
        "prefill_stalls_decode",
        TPOT_TAIL,
        ratio,
        f"TPOT p95 {m.tpot_p95_ms:.1f} ms is {ratio:.1f}x p50 {m.tpot_p50_ms:.1f} ms",
    )


def _bandwidth(m: Measurement) -> Finding:
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
    return _finding(
        "decode_bandwidth",
        DECODE_OF_CEILING,
        c.decode_pct_of_ceiling,
        f"decode ran at {c.decode_pct_of_ceiling:.0f}% of the memory-bandwidth ceiling at the "
        "measured batch",
    )


def _compute(m: Measurement) -> Finding:
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
    return _finding(
        "gpu_compute",
        TENSOR_BUSY,
        tensor,
        f"the busiest kernel, {top.name} ({top.share:.0%} of GPU time, prefill or decode), keeps "
        f"the tensor pipe {tensor:.0f}% busy and DRAM {dram:.0f}%",
    )


def _host(m: Measurement) -> Finding:
    if m.trace is None:
        return _missing("host_overhead", "a GPU trace: run with `--trace`")
    if m.trace.status != "ok" or m.trace.idle_share is None:
        return _missing("host_overhead", f"a trusted GPU trace: {m.trace.note or m.trace.status}")
    return _finding(
        "host_overhead",
        GPU_IDLE,
        m.trace.idle_share,
        f"the GPU sat idle {m.trace.idle_share:.0%} of the traced window",
    )


def _long_context(m: Measurement) -> Finding:
    c = m.ceilings
    if (
        c is None
        or not c.weight_bytes_per_step
        or c.kv_bytes_per_sequence is None
        or c.avg_context_tokens is None
    ):
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
    ratio = c.kv_bytes_per_sequence * c.avg_running_batch / c.weight_bytes_per_step
    return _finding(
        "long_context",
        KV_OVER_WEIGHTS,
        ratio,
        f"a decode step reads {ratio:.2f}x as many KV-cache bytes as weight bytes (average "
        f"context {c.avg_context_tokens:.0f} tokens, batch {c.avg_running_batch:.1f})",
    )


def _speculation(m: Measurement) -> Finding:
    if m.spec_acceptance_length is None:
        return Finding("speculation", "clear", "speculation is off or not reported")
    found = [
        _finding(
            "speculation",
            SPEC_ACCEPTANCE,
            m.spec_acceptance_length,
            f"each speculative draft yielded {m.spec_acceptance_length:.2f} tokens",
        )
    ]
    if m.spec_coverage is not None:
        evidence = f"accepted drafts made {m.spec_coverage:.0%} of the generated tokens"
        found.append(_finding("speculation", SPEC_COVERAGE, m.spec_coverage, evidence))
    return max(found, key=lambda f: (SEVERITY.get(f.state, 0), f.margin))


def speculation_crossed(m: Measurement) -> bool:
    """Whether speculation was measured and did not pay (a speculation finding crossed)."""
    return _fired(_speculation(m))


ANSWERS = {
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


def classify(measurement: Measurement, settings: Settings, answers: str | None) -> Diagnosis:
    """The diagnosis of one run with ``settings``. ``answers`` is the comparison with the
    current setup ("equal", "differ" or "inconclusive"), or None when none ran."""
    kv = _kv_capacity(measurement)
    queueing = _queueing(measurement, settings, kv)
    measured = (
        queueing,
        kv,
        _prefill(measurement, queueing),
        _interference(measurement, kv),
        _bandwidth(measurement),
        _compute(measurement),
        _host(measurement),
        _long_context(measurement),
        _speculation(measurement),
    )
    findings = tuple(replace(f, confidence=_confidence(f, measurement.succeeded)) for f in measured)
    gates = [_fit(measurement)]
    if answers is not None:
        state, evidence = ANSWERS[answers]
        gates.insert(0, Finding("quality", state, evidence))
    fired = sorted(
        (f for f in findings if f.confidence),
        key=lambda f: (SEVERITY[f.state], f.margin),
        reverse=True,
    )
    return Diagnosis(
        findings=findings,
        gates=tuple(gates),
        shape=_shape(measurement),
        primary=fired[0].bottleneck if fired else None,
        confidence=fired[0].confidence if fired else None,
        secondary=tuple(f.bottleneck for f in fired[1:]),
        requests=measurement.succeeded,
    )


def measurement_from_metrics(data: Mapping[str, Any]) -> Measurement:
    """The parts of a recorded ``metrics.json`` the classifier reads, as a Measurement, so a
    recorded run can be diagnosed again (with other thresholds, for calibration). Fields the
    classifier does not read are left at their defaults."""
    # Imported here: engines/__init__ imports the vLLM playbook, which imports this module.
    from .ceilings import Ceilings
    from .measurement import GroupStats, Measurement
    from .trace import TraceSummary

    def build(kind: type[Any], value: Any) -> Any:
        if not isinstance(value, Mapping):
            return None
        names = {f.name for f in fields(kind)}
        return kind(**{k: v for k, v in value.items() if k in names})

    explicit = {"ceilings", "trace", "startup_wave", "too_long"}
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
    )
