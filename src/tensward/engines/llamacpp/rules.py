"""llama-server's own diagnosis rules: how it shows that a KV cache is too small, and the
classes it has no input for."""

from __future__ import annotations

from ...classify import REFUSED, Bottleneck, EngineRule, Finding, RuleInputs
from .placement import OFFLOAD_SHARE, Placement, calibrate, decode_reads, kind_of
from .probes import HOST_BANDWIDTH_FACT

NO_STEP_BUDGET = "llama.cpp has no step budget; -b/-ub have their own entries"
SHARED_PROCESS = "HTTP threads share the server process"


def kv_exhausted(inputs: RuleInputs) -> Finding | None:
    """The cache ran out: a request was refused for its size, llama-server halved its batch to
    find room, or ``--fit`` cut the context below a prompt the workload sends."""
    m = inputs.measurement
    refused = int(m.engine_signals.get("llamacpp.context_exceeded", 0))
    halved = int(m.engine_signals.get("llamacpp.kv_full_lines", 0))
    changed = inputs.resolved.facts.get("fit_changed") if inputs.resolved else None
    cut = "ctx_size" in (changed or ()) and bool(m.too_long)
    if cut:
        evidence = (
            f"--fit cut the context to {m.max_context_len} tokens per slot, below the "
            f"{len(m.too_long)} prompts that need more"
        )
    elif refused:
        evidence = f"{refused} requests were refused for exceeding the context size (HTTP 400)"
    elif halved:
        evidence = (
            f'the server log has {halved} "failed to find free space in the KV cache" lines: '
            "llama-server retried in smaller batches"
        )
    else:
        return None
    turned_away = (cut or bool(refused)) and not halved
    return Finding(
        "kv_capacity",
        "critical" if cut or refused else "warning",
        evidence,
        REFUSED if turned_away else None,
        margin=1.0,
    )


def offload(inputs: RuleInputs) -> Finding | None:
    """Whether reading weights from host memory takes a quarter or more of a decode step. The
    step is the measured TPOT; without the host probe's bandwidth, or when the step is shorter
    than the probe's bandwidth allows, the host side is what the step leaves after the device
    reads, an upper bound that assumes no other overhead. Only a calibrated model, one whose
    reads explain the step within 30 % at one request, makes it "likely"."""
    resolved = inputs.resolved
    found = resolved.facts.get("placement") if resolved else None
    if not found:
        return None
    p = Placement(**found)
    mix = resolved.facts.get("expert_mix") if resolved else None
    kind = kind_of(p)
    if kind == "kv_host":
        return Finding("offload", "cant_tell", "KV reads from host memory are not modelled")
    if kind == "full_gpu":
        return Finding("offload", "clear", "the weights are all in device memory")
    if kind == "cpu_only":
        return Finding("offload", "clear", "the model runs on the CPU; nothing is offloaded")
    m = inputs.measurement
    ceilings = m.ceilings
    step = m.tpot_p50_ms
    if not step or ceilings is None or not ceilings.bandwidth_gbs:
        return Finding(
            "offload", "cant_tell", "the decode step time and the device's memory bandwidth"
        )
    batch = max(round(ceilings.avg_running_batch or 1), 1)
    if mix:
        p = decode_reads(p, mix, batch)
    probed = resolved.facts.get(HOST_BANDWIDTH_FACT) if resolved else None
    try:
        model = calibrate(
            weight_bytes=p.device_bytes_per_token + p.host_bytes_per_token,
            bw_dev_gbs=ceilings.bandwidth_gbs, full_gpu_ms=None, cpu_ms=None, bw_host_gbs=probed,
            concurrency=batch, baseline_ms=step, baseline=p,
        )  # fmt: skip
    except ValueError as error:
        return Finding("offload", "cant_tell", str(error))
    host_ms = p.host_bytes_per_token / (model.bw_host_gbs * 1e6)
    share = host_ms / step
    unexplained = model.overhead_ms / step
    if model.calibrated:
        basis = f"; the reads explain the step to within {unexplained:.0%}"
    elif model.host_floor:
        basis = (
            f"; the host bandwidth is at least {model.bw_host_gbs:.0f} GB/s, as this step shows, "
            "so the host share is an upper estimate"
        )
    else:
        basis = (
            f"; {max(unexplained, 0):.0%} of the step is not decode reads (concurrency {batch}), "
            "so the host share is a bound"
        )
    evidence = (
        f"host memory bandwidth: {host_ms:.1f} ms of the {step:.1f} ms decode step "
        f"({share:.0%}) reads {p.host_bytes_per_token / 2**30:.1f} GiB of weights from host "
        f"memory{basis}"
    )
    if share < OFFLOAD_SHARE:
        return Finding("offload", "clear", evidence)
    return Finding(
        "offload",
        "critical" if share >= 2 * OFFLOAD_SHARE else "warning",
        evidence,
        margin=share / OFFLOAD_SHARE,
        calibrated=model.calibrated,
        confidence=None if model.calibrated else "possible",
    )


def _cannot_tell(bottleneck: Bottleneck, reason: str) -> EngineRule:
    return EngineRule(
        bottleneck=bottleneck, rule=lambda inputs: Finding(bottleneck, "cant_tell", reason)
    )


ENGINE_RULES = (
    EngineRule(bottleneck="kv_capacity", rule=kv_exhausted),
    EngineRule(bottleneck="offload", rule=offload),
    _cannot_tell("step_budget", NO_STEP_BUDGET),
    _cannot_tell("frontend_cpu", SHARED_PROCESS),
)
