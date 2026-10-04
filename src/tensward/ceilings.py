"""Level 0 diagnosis: theoretical hardware ceilings for the registered model on this GPU.

No profiler is involved. The ceilings are upper bounds from the GPU's datasheet and the model's
shape; a real engine cannot reach them, and a measurement far below one does not say why (that
is what the trace is for). Anything that cannot be established says so and gives no number.

Decode reads the text model's weights once per step (for a mixture of experts, only the fewest
experts a step can read, as many as one token is routed to) plus each sequence's KV cache, so it
is bound by memory bandwidth until the batch is large enough for compute. Prefill does about
two FLOPs per active parameter per token, so it is bound by tensor throughput. The attention
FLOPs of prefill and the memory traffic of activations are ignored, which makes the bounds a
little optimistic.

Tensor rates are dense (not sparsity) FP16/BF16 figures with FP32 accumulation, the mode vLLM's
GEMMs use; the platform's datasheet table cites its sources.
"""

from __future__ import annotations

from dataclasses import dataclass

from .anatomy import ModelAnatomy
from .artifacts import ArtifactVariant
from .platforms import MIB, HardwareSpec, derived_bandwidth, detect_devices, hardware_spec

DERIVED = "derived from device (memory clock x bus width)"


class Unavailable(Exception):
    """A ceiling that cannot be established; the message says why."""


@dataclass(frozen=True, slots=True)
class Ceilings:
    """Theoretical upper bounds next to what the engine's counters measured."""

    unavailable: str | None = None
    gpu: str | None = None
    bandwidth_gbs: float | None = None
    bandwidth_source: str | None = None
    derived_bandwidth_gbs: float | None = None  # from the device, shown beside a datasheet value
    tensor_tflops: float | None = None
    weight_bytes_per_step: int | None = None  # one decode step at batch 1, KV cache excluded
    kv_bytes_per_sequence: int | None = None  # at the average context
    expert_bytes_per_step: float | None = (
        None  # routed experts one step reads, at the measured batch
    )
    experts_per_step: float | None = None  # distinct experts per MoE layer, at the measured batch
    moe_experts: tuple[int, int] | None = None  # (experts, experts per token)
    avg_context_tokens: float | None = None
    avg_running_batch: float | None = None
    decode_ceiling_batch1_tok_s: float | None = None  # per sequence
    decode_ceiling_tok_s: float | None = None  # all sequences, at the measured average batch
    prefill_ceiling_tok_s: float | None = None
    measured_decode_tok_s: float | None = None
    measured_prefill_tok_s: float | None = None
    decode_pct_of_ceiling: float | None = None
    prefill_pct_of_ceiling: float | None = None
    # Prefill and decode share the GPU: the time the window's tokens need at both ceilings,
    # as a share of the window. Near 100% the GPU was as busy as physics allows.
    ceiling_time_pct_of_window: float | None = None
    # Shares above 100% are impossible, so they are withheld and named here instead.
    exceeds_bound: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Measured:
    """What the engine's counters and the client saw over the measurement window. ``requests``
    counts the declared requests that succeeded, including any that finished before the window
    opened, so it is an approximation of the window's own count."""

    seconds: float
    requests: int
    prompt_tokens: float | None  # every prompt token, including those served from the prefix cache
    prefill_computed_tokens: float | None  # the prompt tokens the GPU actually ran through prefill
    generation_tokens: float | None
    avg_running_batch: float | None


def detect_gpus() -> tuple[tuple[str, int], ...]:
    """(name, MiB of memory) of each visible device."""
    return tuple((device.name, device.total_bytes // MIB) for device in detect_devices())


def gpu_spec(
    gpus: tuple[tuple[str, int], ...], selected: tuple[str, ...] | None
) -> tuple[int, str, HardwareSpec | None]:
    """(index, name, datasheet row or None if no platform's table has it) of the one device
    serving: the selected device, else device 0 (where an engine without a selection runs)."""
    if not gpus:
        raise Unavailable("no supported accelerator detected")
    if selected is not None and len(selected) > 1:
        raise Unavailable(f"multi-GPU not modelled (GPUs {','.join(selected)} selected)")
    index = int(selected[0]) if selected else 0
    if index >= len(gpus):
        raise Unavailable(f"GPU {index} selected, but {len(gpus)} detected")
    name = gpus[index][0]
    return index, name, hardware_spec(name)


def _peak_tflops(spec: HardwareSpec, variant: ArtifactVariant) -> float:
    """The tensor rate the matmuls run at: activations only quantize for W8A8 checkpoints."""
    if variant.activation_scheme is None:
        return spec.fp16_tflops
    if variant.weight_precision == "int8":
        return spec.int8_tops
    return spec.fp8_tflops or spec.fp16_tflops


def experts_read(experts: int, per_token: int, batch: float) -> float:
    """Expected distinct experts one MoE layer reads in a decode step of ``batch`` sequences,
    each routed to ``per_token`` of ``experts`` uniformly. This is the most distinct experts
    routing typically spreads over: real routing overlaps and reads fewer, so it is context for
    the ceiling, which assumes the fewest (``per_token``)."""
    return float(experts * (1 - (1 - per_token / experts) ** batch))


def compute_ceilings(
    *,
    anatomy: ModelAnatomy,
    variant: ArtifactVariant,
    kv_cache_dtype: str,
    measured: Measured,
    selected: tuple[str, ...] | None = None,
    gpus: tuple[tuple[str, int], ...] | None = None,
    device_gbs: float | None = None,
) -> Ceilings:
    """The ceilings for the registered checkpoint on the ``selected`` GPU (default: device 0)
    of those detected, versus ``measured``. Decode reads the text model's non-expert weights,
    the fewest experts a step can read and the KV cache; never the vision or audio encoders,
    nor the embedding table (a gather) unless it is also the output head.

    ``device_gbs`` stands in for the device-derived bandwidth when ``gpus`` is given (tests).
    """
    parts, params, moe = anatomy.components, anatomy.params, anatomy.moe
    if parts is None or params is None or not anatomy.attention:
        return Ceilings(unavailable=f"unavailable ({'; '.join(anatomy.unavailable)})")
    try:
        index, name, spec = gpu_spec(detect_gpus() if gpus is None else gpus, selected)
    except Unavailable as error:
        return Ceilings(unavailable=f"unavailable ({error})")
    head_bytes = parts.embedding if anatomy.tied_embeddings else parts.lm_head
    head_params = params.embedding if anatomy.tied_embeddings else params.lm_head
    non_expert = parts.text_dense + parts.shared_experts + head_bytes
    active_experts = (
        0.0 if moe is None else moe.experts_per_token * params.routed_experts / moe.experts
    )
    body_params = params.text_dense + params.shared_experts + active_experts  # prefill, per token
    # Sampling logits are computed only for the last prompt token, but for every decoded token.
    decode_params = body_params + head_params

    # No routing reads fewer experts than one token is sent to, whatever the batch.
    step_bytes = non_expert
    if moe is not None:
        step_bytes += moe.layers * moe.expert_bytes * moe.experts_per_token

    if gpus is None:
        device_gbs = derived_bandwidth(index)
    if spec is not None:
        bandwidth_gbs, source, tflops = spec.bandwidth_gbs, "datasheet", _peak_tflops(spec, variant)
    elif device_gbs:
        bandwidth_gbs, source, tflops = device_gbs, DERIVED, None
    else:
        return Ceilings(unavailable=f"unavailable (unknown GPU {name}, no bandwidth from device)")
    bandwidth = bandwidth_gbs * 1e9

    requests = max(measured.requests, 1)
    context = None
    if measured.prompt_tokens is not None and measured.generation_tokens is not None:
        context = measured.prompt_tokens / requests + measured.generation_tokens / requests / 2
    batch = measured.avg_running_batch
    kv = anatomy.kv_bytes(round(context or 0), kv_cache_dtype) or 0
    per_sequence = bandwidth / (step_bytes + kv)
    compute_bound = tflops * 1e12 / (2 * decode_params) if tflops else float("inf")
    aggregate = None
    if context is not None and batch:
        aggregate = min(batch * bandwidth / (step_bytes + batch * kv), compute_bound)
    prefill = tflops * 1e12 / (2 * body_params) if tflops else None

    def rate(tokens: float | None) -> float | None:
        return tokens / measured.seconds if tokens is not None and measured.seconds > 0 else None

    decode_rate = rate(measured.generation_tokens)
    prefill_rate = rate(measured.prefill_computed_tokens)
    window = None
    if aggregate and prefill and decode_rate is not None and prefill_rate is not None:
        window = 100 * (decode_rate / aggregate + prefill_rate / prefill)
    shares = {
        "decode": _pct(decode_rate, aggregate),
        "prefill": _pct(prefill_rate, prefill),
        "window": window,
    }
    exceeds = tuple(name for name, share in shares.items() if share is not None and share > 100)
    shares = {name: None if name in exceeds else share for name, share in shares.items()}
    return Ceilings(
        gpu=spec.label if spec else name,
        bandwidth_gbs=bandwidth_gbs,
        bandwidth_source=source,
        derived_bandwidth_gbs=device_gbs if spec else None,
        tensor_tflops=tflops,
        weight_bytes_per_step=round(step_bytes),
        kv_bytes_per_sequence=kv if context is not None else None,
        expert_bytes_per_step=(
            moe.layers * moe.expert_bytes * experts_read(moe.experts, moe.experts_per_token, batch)
            if moe and batch
            else None
        ),
        experts_per_step=(
            experts_read(moe.experts, moe.experts_per_token, batch) if moe and batch else None
        ),
        moe_experts=(moe.experts, moe.experts_per_token) if moe else None,
        avg_context_tokens=context,
        avg_running_batch=batch,
        decode_ceiling_batch1_tok_s=per_sequence,
        decode_ceiling_tok_s=aggregate,
        prefill_ceiling_tok_s=prefill,
        measured_decode_tok_s=decode_rate,
        measured_prefill_tok_s=prefill_rate,
        decode_pct_of_ceiling=shares["decode"],
        prefill_pct_of_ceiling=shares["prefill"],
        ceiling_time_pct_of_window=shares["window"],
        exceeds_bound=exceeds,
    )


def _pct(measured: float | None, ceiling: float | None) -> float | None:
    return None if measured is None or not ceiling else 100 * measured / ceiling


def _moe_lines(ceilings: Ceilings) -> list[str]:
    if ceilings.moe_experts is None:
        return []
    experts, per_token = ceilings.moe_experts
    lines = [f"- mixture of experts: {per_token} of {experts} experts per token"]
    if ceilings.experts_per_step is not None and ceilings.expert_bytes_per_step is not None:
        lines.append(
            f"- at the measured batch a decode step reads about {ceilings.experts_per_step:.0f} "
            f"of {experts} experts per layer ({ceilings.expert_bytes_per_step / 1e9:.2f} GB), "
            "with uniform routing; real routing overlaps and reads fewer, so measured decode can "
            "sit between this and the ceiling, which assumes each layer reads only "
            f"{per_token} experts. Steps that mix prefill read up to all experts, so the "
            "decode ceiling holds for decode-only steps"
        )
    return lines


def render_ceilings(ceilings: Ceilings | None) -> list[str]:
    """The report.md section. Always labelled as theoretical upper bounds."""
    if ceilings is None:
        return []
    lines = ["", "## Hardware ceilings (theoretical upper bounds, not targets)", ""]
    if ceilings.unavailable:
        return [*lines, f"- ceilings: {ceilings.unavailable}"]

    def show(label: str, value: float | None, unit: str, digits: int = 0) -> str:
        return f"- {label}: " + ("not measured" if value is None else f"{value:,.{digits}f}{unit}")

    if ceilings.tensor_tflops is None:
        tensor = (
            f"- dense tensor rate: unavailable (no published dense tensor rate for {ceilings.gpu})"
        )
    else:
        tensor = f"- dense tensor rate: {ceilings.tensor_tflops:g} TFLOPS (datasheet)"
    # Set whenever ceilings are available (the unavailable case returned above).
    assert ceilings.weight_bytes_per_step is not None
    cross = ""
    if ceilings.derived_bandwidth_gbs:
        cross = f"; device-derived cross-check {ceilings.derived_bandwidth_gbs:,.1f} GB/s"
    lines += [
        f"- GPU: {ceilings.gpu}",
        f"- memory bandwidth: {ceilings.bandwidth_gbs:,.1f} GB/s "
        f"(bandwidth: {ceilings.bandwidth_source}{cross})",
        tensor,
        f"- per decode step (one sequence): {ceilings.weight_bytes_per_step / 1e9:.2f} GB of "
        "weights (the vision encoder and the embedding gather are not read)",
        show(
            "KV cache read per sequence at the average context",
            None
            if ceilings.kv_bytes_per_sequence is None
            else ceilings.kv_bytes_per_sequence / 2**20,
            " MiB",
            1,
        ),
        *_moe_lines(ceilings),
        show("average context per sequence", ceilings.avg_context_tokens, " tokens"),
        show("average running batch", ceilings.avg_running_batch, " sequences", 1),
        show("decode ceiling, one sequence", ceilings.decode_ceiling_batch1_tok_s, " tok/s"),
        show("decode ceiling at the measured batch", ceilings.decode_ceiling_tok_s, " tok/s"),
        show("decode measured", ceilings.measured_decode_tok_s, " tok/s"),
        show("prefill ceiling", ceilings.prefill_ceiling_tok_s, " tok/s"),
        show("prefill measured", ceilings.measured_prefill_tok_s, " tok/s"),
        show("decode, share of its ceiling", ceilings.decode_pct_of_ceiling, "%", 1),
        show("prefill, share of its ceiling", ceilings.prefill_pct_of_ceiling, "%", 1),
        show(
            "window time the tokens need at both ceilings",
            ceilings.ceiling_time_pct_of_window,
            "%",
            1,
        ),
        *(
            [
                f"- {', '.join(ceilings.exceeds_bound)} share exceeds the theoretical bound: the "
                "measurement or the ceiling assumption is wrong, so it is not shown"
            ]
            if ceilings.exceeds_bound
            else []
        ),
        "- prefill and decode share the GPU, so each share alone understates how busy it was; "
        "the last line combines them. Attention FLOPs and activation traffic are ignored.",
    ]
    return lines
