"""Level 0 diagnosis: theoretical hardware ceilings for the registered model on this GPU.

No profiler is involved. The ceilings are upper bounds from the GPU's datasheet and the model's
shape; a real engine cannot reach them, and a measurement far below one does not say why (that
is what the trace is for). Anything that cannot be established says so and gives no number.

Decode reads the text model's weights once per step (for a mixture of experts, only the experts
the batch is routed to) plus each sequence's KV cache, so it is bound by memory bandwidth until
the batch is large enough for compute. Prefill does about two FLOPs per active parameter per
token, so it is bound by tensor throughput. The attention FLOPs of prefill and the memory
traffic of activations are ignored, which makes the bounds a little optimistic.

Tensor rates are dense (not sparsity) FP16/BF16 figures with FP32 accumulation, the mode vLLM's
GEMMs use; see the note above the GPU table for sources.
"""

from __future__ import annotations

import ctypes
import re
import subprocess
from dataclasses import dataclass
from functools import cache
from typing import Any

from .anatomy import ModelAnatomy
from .artifacts import ArtifactVariant

NVIDIA_SMI = ("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader")


@dataclass(frozen=True, slots=True)
class GpuSpec:
    """Dense (not sparsity) datasheet numbers. ``fp8_tflops`` is None when unsupported."""

    label: str
    name_pattern: str
    bandwidth_gbs: float
    fp16_tflops: float
    int8_tops: float
    fp8_tflops: float | None
    sms: int
    vram_gb: int


# Conventions. Every figure is DENSE: NVIDIA datasheets print "dense | sparse*" pairs or a single
# number footnoted "with sparsity"; a figure is halved only where the source marks it as sparse.
# Tensor FP16/BF16 rates are for FP32 accumulation, which is what vLLM's fp16/bf16 GEMMs use.
# Data-center parts (L4, A10, A100, H100, L40S) and Turing's T4 (mixed FP16/FP32) publish one
# rate; GeForce parts publish separate FP16- and FP32-accumulate rates and take the latter.
# Rows whose figures could not be checked against an official NVIDIA page or PDF are omitted
# (for example A10G, where AWS publishes only "320 tensor cores, up to 250 TOPS, 24 GB" and no
# bandwidth or FP16 rate; and H100 PCIe, whose dense rates appear only in a rounded
# pre-release table). An unlisted GPU reports "ceilings unavailable". Sources, fetched 2026-09:
#   L4: nvidia.com/en-us/data-center/l4 (242 FP16/BF16, 485 FP8/INT8 with sparsity, 300 GB/s,
#     24 GB); Ada whitepaper Appendix D (images.nvidia.com/aem-dam/Solutions/Data-Center/l4/
#     nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf): 58 SMs, BF16 121 | 242
#   A10: nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/a10-datasheet.pdf
#     (FP16 125 | 250*, INT8 250 | 500*, 600 GB/s, 24 GB, 72 RT cores = 72 SMs)
#   T4: nvidia.com/en-us/data-center/tesla-t4 (65 FP16, 130 INT8, 16 GB, "320+" GB/s);
#     320 GB/s and 40 SMs from the Ada whitepaper Table 5
#   A100: nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-
#     nvidia-1758950-r4-web.pdf (FP16 312 | 624*, INT8 624 | 1248*; 1555 GB/s 40GB, 1935 80GB
#     PCIe, 2039 80GB SXM); 108 SMs from the Ampere GA100 whitepaper
#     (images.nvidia.com/aem-dam/en-zz/Solutions/data-center/
#     nvidia-ampere-architecture-whitepaper.pdf)
#   H100 SXM: nvidia.com/en-us/data-center/h100 (FP16 1,979*, FP8 3,958*, INT8 3,958* with
#     sparsity, 3.35 TB/s); 132 SMs from
#     developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth
#   L40S: nvidia.com/en-us/data-center/l40s (FP16/BF16 362.05 | 733*, FP8 and INT8 733 | 1,466*,
#     864 GB/s, 48 GB, 18,176 CUDA cores = 142 SMs)
#   RTX 4090: Ada whitepaper Table 2 (images.nvidia.com/aem-dam/Solutions/Data-Center/l4/
#     nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf): 128 SMs, FP16 with FP32 accumulate
#     165.2 | 330.4*, INT8 660.6 | 1321.2*, FP8 with FP32 accumulate 330.3 | 660.6*, 1008 GB/s
#   RTX 3090: Ampere GA102 whitepaper Table 9 (nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-
#     architecture-whitepaper-v2.pdf): 82 SMs, FP16 with FP32 accumulate 71 | 142*, INT8
#     284 | 568*, 936 GB/s; no FP8 on Ampere
DERIVED = "derived from device (memory clock x bus width)"

GPUS: tuple[GpuSpec, ...] = (
    GpuSpec("L4", r"\bL4\b", 300, 121, 242.5, 242.5, 58, 24),
    GpuSpec("A10", r"\bA10\b", 600, 125, 250, None, 72, 24),
    GpuSpec("T4", r"\bT4\b", 320, 65, 130, None, 40, 16),
    GpuSpec("A100-40GB", r"A100.*40GB", 1555, 312, 624, None, 108, 40),
    GpuSpec("A100-80GB SXM", r"A100-SXM.*80GB", 2039, 312, 624, None, 108, 80),
    GpuSpec("A100-80GB PCIe", r"A100 80GB PCIe|A100-PCIE-80GB", 1935, 312, 624, None, 108, 80),
    GpuSpec("H100 SXM", r"H100 80GB HBM3|H100.SXM", 3350, 989.5, 1979, 1979, 132, 80),
    GpuSpec("L40S", r"\bL40S\b", 864, 362.05, 733, 733, 142, 48),
    GpuSpec("RTX 4090", r"RTX 4090", 1008, 165.2, 660.6, 330.3, 128, 24),
    GpuSpec("RTX 3090", r"RTX 3090", 936, 71, 284, None, 82, 24),
)


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
    """What the engine's counters and the client saw over the measurement window."""

    seconds: float
    requests: int
    prompt_tokens: float | None  # every prompt token, including those served from the prefix cache
    prefill_computed_tokens: float | None  # the prompt tokens the GPU actually ran through prefill
    generation_tokens: float | None
    avg_running_batch: float | None


@cache
def detect_gpus() -> tuple[tuple[str, int], ...]:
    """(name, MiB of memory) of each visible GPU, from NVML if installed, else nvidia-smi."""
    try:
        import pynvml  # noqa: PLC0415 - optional GPU extra

        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        found = tuple(
            (str(_text(pynvml.nvmlDeviceGetName(h))), pynvml.nvmlDeviceGetMemoryInfo(h).total >> 20)
            for h in handles
        )
        if found:
            return found
    except Exception:  # noqa: BLE001 - NVML missing or refusing; nvidia-smi is the fallback
        pass
    try:
        out = subprocess.run(NVIDIA_SMI, capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return ()
    rows = (line.rsplit(",", 1) for line in out.stdout.strip().splitlines() if "," in line)
    return tuple((name.strip(), int(re.sub(r"\D", "", memory) or 0)) for name, memory in rows)


# CUDA driver attribute ids (cuda.h): CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE (kHz),
# CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH (bits) and CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
_CU_MEMORY_CLOCK_RATE, _CU_MEMORY_BUS_WIDTH, _CU_MULTIPROCESSOR_COUNT = 36, 37, 16


def derived_bandwidth_gbs(memory_clock_khz: float, bus_width_bits: float) -> float:
    """Peak DRAM bandwidth in GB/s: clock x bus width x 2 (double data rate).

    This is the formula of NVIDIA's CUDA samples (deviceQuery, bandwidthTest): the CUDA
    ``memoryClockRate`` is the clock whose double is the per-pin data rate. Checked on the L4
    (Nsight Compute: 6,251,000 kHz, 192 bits): 6.251e9 x 2 x 24 B = 300.05 GB/s, the datasheet's
    300 GB/s. CUDA is used rather than NVML because NVML's memory clock for GDDR parts is the
    raw GDDR clock, whose relation to the data rate differs by memory generation, and
    nvidia-smi --query-gpu has no bus width field.
    """
    return memory_clock_khz * 1e3 * 2 * (bus_width_bits / 8) / 1e9


@cache
def _libcuda() -> ctypes.CDLL | None:
    """The CUDA driver library, initialised once; None when it is missing or refuses to start."""
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        return None if cuda.cuInit(0) else cuda
    except OSError:
        return None


def _device_attribute(device: int, attribute: int) -> int | None:
    """One attribute of a CUDA device handle, or None when the driver does not give it."""
    cuda = _libcuda()
    value = ctypes.c_int()
    try:
        if cuda is None or cuda.cuDeviceGetAttribute(ctypes.byref(value), attribute, device):
            return None
    except AttributeError:
        return None
    return value.value


@cache
def device_bandwidth_gbs(index: int = 0) -> float | None:
    """Peak DRAM bandwidth of CUDA device ``index`` via libcuda, or None (no driver, no device)."""
    cuda = _libcuda()
    device = ctypes.c_int()
    try:
        if cuda is None or cuda.cuDeviceGet(ctypes.byref(device), index):
            return None
    except AttributeError:
        return None
    clock = _device_attribute(device.value, _CU_MEMORY_CLOCK_RATE)
    width = _device_attribute(device.value, _CU_MEMORY_BUS_WIDTH)
    if clock is None or width is None or clock <= 0 or width <= 0:
        return None
    return derived_bandwidth_gbs(clock, width)


def driver_cuda_version() -> str | None:
    """The newest CUDA the installed driver supports (not the CUDA an engine was built with),
    as ``major.minor``; None when there is no driver."""
    cuda = _libcuda()
    version = ctypes.c_int()
    try:
        if cuda is None or cuda.cuDriverGetVersion(ctypes.byref(version)):
            return None
    except AttributeError:
        return None
    return f"{version.value // 1000}.{version.value % 1000 // 10}"


def device_sm_count(bus_id: str) -> int | None:
    """Streaming multiprocessors of the GPU at PCI ``bus_id`` (nvidia-smi's spelling), or None
    when the driver is missing or the process cannot see that GPU. Looking the device up by bus
    id keeps CUDA's device order and CUDA_VISIBLE_DEVICES out of it."""
    cuda = _libcuda()
    if cuda is None:
        return None
    domain, _, rest = bus_id.partition(":")
    # nvidia-smi prints an 8-digit upper-case domain; CUDA documents a 4-digit lower-case one.
    for spelling in dict.fromkeys([f"{domain[-4:]}:{rest}".lower(), bus_id]):
        device = ctypes.c_int()
        try:
            if cuda.cuDeviceGetByPCIBusId(ctypes.byref(device), spelling.encode()) == 0:
                return _device_attribute(device.value, _CU_MULTIPROCESSOR_COUNT)
        except AttributeError:
            return None
    return None


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def gpu_spec(
    gpus: tuple[tuple[str, int], ...], selected: tuple[str, ...] | None
) -> tuple[int, str, GpuSpec | None]:
    """(index, name, table row or None if the GPU is not in the table) of the one GPU serving:
    the selected device, else device 0 (where an engine without a selection runs)."""
    if not gpus:
        raise Unavailable("no GPU detected (NVML and nvidia-smi found none)")
    if selected is not None and len(selected) > 1:
        raise Unavailable(f"multi-GPU not modelled (GPUs {','.join(selected)} selected)")
    index = int(selected[0]) if selected else 0
    if index >= len(gpus):
        raise Unavailable(f"GPU {index} selected, but {len(gpus)} detected")
    name = gpus[index][0]
    return index, name, next((s for s in GPUS if re.search(s.name_pattern, name)), None)


def _peak_tflops(spec: GpuSpec, variant: ArtifactVariant) -> float:
    """The tensor rate the matmuls run at: activations only quantize for W8A8 checkpoints."""
    if variant.activation_scheme is None:
        return spec.fp16_tflops
    if variant.weight_precision == "int8":
        return spec.int8_tops
    return spec.fp8_tflops or spec.fp16_tflops


def experts_read(experts: int, per_token: int, batch: float) -> float:
    """Expected distinct experts one MoE layer reads in a decode step of ``batch`` sequences,
    each routed to ``per_token`` of ``experts`` uniformly. Real routing is skewed and reads
    fewer, so a ceiling built on this is on the high side."""
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
    the experts its tokens are routed to and the KV cache; never the vision or audio encoders,
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

    def step_bytes(batch: float) -> float:
        if moe is None:
            return non_expert
        return non_expert + moe.layers * moe.expert_bytes * experts_read(
            moe.experts, moe.experts_per_token, batch
        )

    if gpus is None:
        device_gbs = device_bandwidth_gbs(index)
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
    per_sequence = bandwidth / (step_bytes(1) + kv)
    compute_bound = tflops * 1e12 / (2 * decode_params) if tflops else float("inf")
    aggregate = None
    if context is not None and batch:
        aggregate = min(batch * bandwidth / (step_bytes(batch) + batch * kv), compute_bound)
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
        weight_bytes_per_step=round(step_bytes(1)),
        kv_bytes_per_sequence=kv if context is not None else None,
        expert_bytes_per_step=step_bytes(batch) - non_expert if moe and batch else None,
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
            "assuming uniform routing; real routing is skewed and reads fewer. Steps that mix "
            "prefill read up to all experts, so the decode ceiling holds for decode-only steps"
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
