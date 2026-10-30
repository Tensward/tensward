"""Where a llama-server launch keeps its weights and KV cache, what a decode step costs there,
and which moves would shorten it.

A decode step reads every active weight once: ``t = Wd/BWd + Wh/BWh + o``, with the bytes in
device memory, the bytes in host memory (read by the CPU), and an overhead ``o`` the two
bandwidths do not explain. The bandwidths are the datasheet's and the host probe's; ``o`` is
what is left of one measured step.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from ...anatomy import ModelAnatomy
from ...capabilities import Resolved
from ...formats.gguf import kv_bytes_per_token, swa_kv_bytes
from ...platforms.cpu import HostFacts

GIB = 2**30
MIB = 2**20
DEVICE_MARGIN = GIB
HOST_RESERVE = 2 * GIB
CALIBRATION_LIMIT = 0.30  # more than 30 % off the baseline: the finding is only "possible"
HOST_FLOOR_SHARE = 0.05  # an overhead below minus this share of the step shows the host figure low
OFFLOAD_SHARE = 0.25  # host time as a share of a decode step from which the class fires
# A resident model takes no major faults while decoding. 1,000 is a floor above stray faults
# (shared-library and tokenizer pages), not a fitted figure; each is a disk read of a page.
MAJOR_FAULTS_TIGHT = 1000
DEFAULT_UBATCH = 512
# llama.cpp's CUDA compute buffer at -ub 512 measured 100-570 MiB; it grows linearly with -ub
COMPUTE_BUFFER_BYTES = 576 * MIB
Q8_KV_SAVING = 1 - (34 / 32) / 2  # q8_0 stores 34 bytes per 32 values, f16 two per value

PlacementKind = Literal["full_gpu", "dense_offload", "expert_offload", "kv_host", "cpu_only"]
Label = Literal["likely", "possible", "rough estimate", "lower bound", "upper estimate"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Placement:
    device_weight_bytes: int
    host_weight_bytes: int  # model buffers in host memory, the token embedding excluded
    host_expert_bytes: int
    host_layers: int
    host_expert_layers: int
    kv_on_host: bool
    device_bytes_per_token: int
    host_bytes_per_token: int  # MoE: the active experts only (top-k of E of the routed)


def _is_moe(anatomy: ModelAnatomy) -> bool:
    return any(anatomy.expert_layer_bytes)


def union_share(experts: int, used: int, concurrency: int = 1) -> float:
    """The share of a layer's experts one decode step reads: the union of what ``concurrency``
    tokens are each routed to, taking every token's top-k as independent and uniform."""
    if experts and used:
        return 1 - (1 - used / experts) ** max(concurrency, 1)
    return 1.0


def active_share(anatomy: ModelAnatomy, concurrency: int = 1) -> float:
    return union_share(anatomy.experts or 0, anatomy.experts_used or 0, concurrency)


def decode_reads(p: Placement, mix: Mapping[str, int], concurrency: int) -> Placement:
    """``p`` with the bytes a decode step at ``concurrency`` reads, from the expert bytes on
    each side (``mix``, as ``expert_mix`` records them)."""
    leave = 1 - union_share(mix["experts"], mix["used"], concurrency)
    return dataclasses.replace(
        p,
        device_bytes_per_token=p.device_weight_bytes - round(mix["device"] * leave),
        host_bytes_per_token=p.host_weight_bytes - round(mix["host"] * leave) + mix["row"],
    )


def _split(
    anatomy: ModelAnatomy, resolved: Resolved
) -> tuple[Placement, dict[str, int]] | str | None:
    """Bytes by the buffer domain the start-up log put them in, never from ``offloaded N/M``;
    the bytes a decode step at ``concurrency`` reads. ``token_embd`` in host memory is not
    offload: a token reads one row of it. ``host_layers`` counts blocks: llama.cpp offloads the
    output layer first. A string says why the log and the anatomy do not fit together."""
    facts, levers = resolved.facts, resolved.levers
    device, host = facts.get("device_weight_bytes"), facts.get("host_weight_bytes")
    if device is None or host is None:
        return None
    embedding = min(anatomy.weight_bytes_by_role.get("embedding", 0), host)
    row = embedding // anatomy.vocab if anatomy.vocab else 0
    host_layers = int(levers.get("host_layers") or 0)
    expert_layers = int(levers.get("host_expert_layers") or 0)
    host_weights = host - embedding
    # A full-GPU launch keeps only the token embedding in host memory; the log's buffer and the
    # anatomy's embedding differ by padding, which must not read as offloaded weights.
    if not (host_layers or expert_layers) and host_weights < 0.02 * (device + host):
        host_weights = 0
    experts = anatomy.expert_layer_bytes
    if not device:
        on_host = sum(experts)
    elif host_layers == 0 and expert_layers:
        # --fit and --n-cpu-moe leave only expert tensors in host memory: the log's buffers say
        # how many bytes, whatever count the log's summary gave
        on_host = host_weights
    elif host_layers and expert_layers and "tensor_overrides" in facts.get("fit_changed", ()):
        return "--fit moved experts after layers it left on the host; their blocks are not readable"
    else:
        on_host = sum(experts[: max(host_layers, expert_layers)])
    if on_host > host_weights * 1.02 and on_host > 0:
        return "the experts the start-up log says stay in host memory outweigh its host buffers"
    on_device = max(sum(experts) - on_host, 0)
    kv_host, kv_device = facts.get("kv_host_bytes"), facts.get("kv_device_bytes")
    mix = {
        "device": on_device,
        "host": on_host,
        "row": row,
        "experts": anatomy.experts or 0,
        "used": anatomy.experts_used or 0,
    }
    found = Placement(
        device_weight_bytes=device,
        host_weight_bytes=host_weights,
        host_expert_bytes=sum(experts[:expert_layers]),
        host_layers=host_layers,
        host_expert_layers=expert_layers,
        kv_on_host=bool(kv_host) and not kv_device,
        device_bytes_per_token=0,
        host_bytes_per_token=0,
    )
    return found, mix


def expert_mix(anatomy: ModelAnatomy, resolved: Resolved) -> dict[str, int] | None:
    """The expert bytes on each side and the routing, which ``decode_reads`` needs to give the
    bytes a step reads at any concurrency."""
    found = _split(anatomy, resolved)
    return None if found is None or isinstance(found, str) else found[1]


def place(
    anatomy: ModelAnatomy, resolved: Resolved, concurrency: int = 1
) -> Placement | str | None:
    found = _split(anatomy, resolved)
    if found is None or isinstance(found, str):
        return found
    read = decode_reads(*found, concurrency)
    if read.device_bytes_per_token < 0 or read.host_bytes_per_token < 0:
        return "the bytes a token reads come out negative from the start-up log's buffers"
    return read


def placement(anatomy: ModelAnatomy, resolved: Resolved, concurrency: int = 1) -> Placement | None:
    found = place(anatomy, resolved, concurrency)
    return found if isinstance(found, Placement) else None


def kind_of(p: Placement) -> PlacementKind:
    if p.device_weight_bytes == 0:
        return "cpu_only"
    if p.host_layers == 0 and p.host_expert_layers == 0 and p.host_weight_bytes == 0:
        return "kv_host" if p.kv_on_host else "full_gpu"
    experts_only = p.host_layers == 0 or p.host_expert_bytes * 2 >= p.host_weight_bytes
    return "expert_offload" if p.host_expert_layers and experts_only else "dense_offload"


@dataclass(frozen=True, slots=True)
class Regime:
    placement: PlacementKind
    compute: Literal["gpu", "cpu", "mixed"]
    host_memory_tight: bool
    numa_split: bool


def regime(p: Placement, host: HostFacts, platform: str, major_faults: int | None) -> Regime:
    kind = kind_of(p)
    compute: Literal["gpu", "cpu", "mixed"] = (
        "cpu" if kind == "cpu_only" else "gpu" if kind in ("full_gpu", "kv_host") else "mixed"
    )
    resident = p.host_weight_bytes + (p.device_weight_bytes if kind == "cpu_only" else 0)
    tight = resident + HOST_RESERVE > host.mem_available_gib * GIB or (
        (major_faults or 0) > MAJOR_FAULTS_TIGHT
    )
    return Regime(
        placement=kind,
        compute=compute,
        host_memory_tight=tight and platform != "apple",
        numa_split=host.numa_nodes > 1 and compute != "gpu",
    )


@dataclass(frozen=True, slots=True)
class DecodeModel:
    bw_dev_gbs: float
    bw_host_gbs: float
    overhead_ms: float
    concurrency: int
    calibrated: bool  # only at concurrency 1 and an overhead within 30 % of the baseline step
    host_floor: bool = False  # ``bw_host_gbs`` is what the step shows at least, not a measurement
    stall_ms_per_byte: float = 0.0  # decode time a byte of host weights costs by streaming


def _ms(size: float, gbs: float) -> float:
    return size / (gbs * 1e6)


def calibrate(
    *,
    weight_bytes: int,
    bw_dev_gbs: float,
    full_gpu_ms: float | None,
    cpu_ms: float | None,
    bw_host_gbs: float | None,
    concurrency: int,
    baseline_ms: float | None = None,
    baseline: Placement | None = None,
) -> DecodeModel:
    """The cost model of one launch. With ``baseline_ms`` and ``baseline`` (the measured step of
    the placement under analysis) the overhead is what the bandwidths leave of it; without a
    host bandwidth, or when the step is shorter than the host reads would take at the one given
    by more than 5 %, the host side is read off the same step, which assumes no overhead: a
    floor for the bandwidth, never calibrated. With ``full_gpu_ms`` and ``cpu_ms`` (steps of the
    model entirely on the device and entirely on the host) the overhead is the full-GPU step's
    and the host bandwidth the CPU step's. ``bw_dev_gbs`` is used as given. Raises ValueError
    when the figures given leave no host bandwidth."""
    if baseline_ms is not None and baseline is not None:
        device_ms = _ms(baseline.device_bytes_per_token, bw_dev_gbs)
        left = baseline_ms - device_ms
        if bw_host_gbs is None or (
            left > 0
            and left - _ms(baseline.host_bytes_per_token, bw_host_gbs)
            < -HOST_FLOOR_SHARE * baseline_ms
        ):
            # The step is shorter than the host reads would take at the probed bandwidth, so the
            # probe is low for this model; the step shows the bandwidth is at least this.
            if left <= 0 or not baseline.host_bytes_per_token:
                raise ValueError("the measured step is not longer than the device reads alone")
            host_gbs = baseline.host_bytes_per_token / (left * 1e6)
            return DecodeModel(bw_dev_gbs, host_gbs, 0.0, concurrency, False, host_floor=True)
        overhead = left - _ms(baseline.host_bytes_per_token, bw_host_gbs)
        step = baseline_ms
    else:
        if full_gpu_ms is None:
            raise ValueError("a calibration needs a measured step")
        overhead = full_gpu_ms - _ms(weight_bytes, bw_dev_gbs)
        step = full_gpu_ms
        if bw_host_gbs is None:
            if cpu_ms is None or cpu_ms <= overhead:
                raise ValueError("a calibration needs the host bandwidth or a CPU step")
            bw_host_gbs = weight_bytes / ((cpu_ms - overhead) * 1e6)
    within = 0 <= overhead <= CALIBRATION_LIMIT * step
    return DecodeModel(bw_dev_gbs, bw_host_gbs, overhead, concurrency, concurrency == 1 and within)


def predict_decode_ms(model: DecodeModel, p: Placement) -> float:
    return (
        _ms(p.device_bytes_per_token, model.bw_dev_gbs)
        + _ms(p.host_bytes_per_token, model.bw_host_gbs)
        + model.overhead_ms
    )


def model_error(model: DecodeModel, p: Placement, measured_ms: float) -> float:
    return (predict_decode_ms(model, p) - measured_ms) / measured_ms


@dataclass(frozen=True, slots=True)
class Fit2:
    device_need: int
    device_free: int
    host_need: int
    host_free: int
    fits: bool
    reason: str
    kv_bytes: int = 0  # the whole pool, before the split by layers
    compute_buffer_bytes: int = 0
    pool_tokens: int = 0
    ubatch: int = 0
    slots: int = 0
    need_tokens: int = 0  # the longest request the workload sent, prompt and output
    peak_running: int = 0


def kv_pool_bytes(anatomy: ModelAnatomy, pool_tokens: int, type_k: str, type_v: str) -> int:
    return kv_bytes_per_token(anatomy, type_k, type_v) * pool_tokens + swa_kv_bytes(
        anatomy, type_k, type_v, pool_tokens
    )


def fit_domains(
    anatomy: ModelAnatomy,
    p: Placement,
    *,
    pool_tokens: int,
    type_k: str,
    type_v: str,
    compute_buffer_bytes: int,
    device_free: int,
    host: HostFacts,
    ubatch: int = 0,
    slots: int = 0,
    need_tokens: int = 0,
    peak_running: int = 0,
    kv_logged: int = 0,
) -> Fit2:
    """Whether the launch's bytes fit the memory each domain has: the device keeps a 1 GiB
    margin, the host a 2 GiB reserve. ``compute_buffer_bytes`` is the baseline log's compute
    allocation on the device, which scales linearly with ``-ub``; ``kv_logged`` the log's KV
    buffers in both domains, which the pool formula under-counts on sliding-window models."""
    kv = kv_logged or kv_pool_bytes(anatomy, pool_tokens, type_k, type_v)
    layers = anatomy.layers or 1
    kv_device = 0 if p.kv_on_host else kv * max(layers - p.host_layers, 0) // layers
    device_need = p.device_weight_bytes + kv_device + compute_buffer_bytes + DEVICE_MARGIN
    host_need = p.host_weight_bytes + kv - kv_device
    host_free = int(host.mem_available_gib * GIB)
    reasons = []
    if device_need > device_free:
        reasons.append(f"the device needs {device_need / GIB:.1f} GiB of {device_free / GIB:.1f}")
    if host_need + HOST_RESERVE > host_free:
        reasons.append(f"the host needs {host_need / GIB:.1f} GiB of {host_free / GIB:.1f}")
    return Fit2(
        device_need, device_free, host_need, host_free, not reasons,
        "; ".join(reasons) or "both domains fit",
        kv, compute_buffer_bytes, pool_tokens, ubatch, slots, need_tokens, peak_running,
    )  # fmt: skip


def logged_kv_bytes(
    anatomy: ModelAnatomy, baseline: Resolved | None, pool_tokens: int, type_k: str, type_v: str
) -> int | None:
    """The KV bytes of a pool of ``pool_tokens`` at the cache types given, scaled from the KV
    buffers (both domains) the baseline's start-up log reported, which the pool formula
    under-counts on sliding-window models. None without such a log."""
    if baseline is None:
        return None
    facts, levers = baseline.facts, baseline.levers
    logged = (facts.get("kv_device_bytes") or 0) + (facts.get("kv_host_bytes") or 0)
    base_pool = levers.get("llamacpp.kv_pool_tokens")
    base_type = "q8_0" if levers.get("kv_cache_precision") == "8bit" else "f16"
    modelled = kv_pool_bytes(anatomy, base_pool, base_type, base_type) if base_pool else 0
    if not logged or not modelled:
        return None
    return round(logged * kv_pool_bytes(anatomy, pool_tokens, type_k, type_v) / modelled)


@dataclass(frozen=True, slots=True)
class ExpertSplit:
    """The smallest number of leading expert layers to keep on the host so that every other
    weight and the KV cache fit the device before launch. ``layers`` is None when no number
    does: ``host_short`` when the device would fit but the host cannot hold those experts."""

    layers: int | None
    device_need: int  # with every layer's experts on the device
    device_weights: int  # the same, weights only
    kv_bytes: int  # on the device
    moved_bytes: int  # the experts of ``layers`` layers
    host_short: bool


def expert_split(
    anatomy: ModelAnatomy,
    device_total: int,
    host: HostFacts,
    *,
    pool_tokens: int,
    type_k: str,
    type_v: str,
    ubatch: int = DEFAULT_UBATCH,
    kv_on_device: bool = True,
    kv_logged: int | None = None,
) -> ExpertSplit:
    """The split of a launch that puts every layer on the device (``-ngl all``), with the token
    embedding in host memory; a tied output layer loads it again on the device. The device keeps
    a 1 GiB margin, the host a 2 GiB reserve. ``kv_logged`` replaces the pool formula's KV bytes
    with a baseline log's, scaled to this pool."""
    embedding = anatomy.weight_bytes_by_role.get("embedding", 0)
    tied = anatomy.token_embedding_bytes if anatomy.tied_embeddings else 0
    weights = sum(anatomy.weight_bytes_by_role.values()) - embedding + tied
    compute = COMPUTE_BUFFER_BYTES * ubatch // DEFAULT_UBATCH
    kv = 0
    if kv_on_device:
        kv = kv_pool_bytes(anatomy, pool_tokens, type_k, type_v) if kv_logged is None else kv_logged
    need = weights + kv + compute + DEVICE_MARGIN
    host_free = int(host.mem_available_gib * GIB) - HOST_RESERVE - embedding
    experts = anatomy.expert_layer_bytes
    for layers in range(len(experts) + 1):
        moved = sum(experts[:layers])
        if need - moved <= device_total:
            fits = moved <= host_free
            return ExpertSplit(layers if fits else None, need, weights, kv, moved, not fits)
    return ExpertSplit(None, need, weights, kv, sum(experts), False)


@dataclass(frozen=True, slots=True)
class Step:
    lever: str
    value: Any
    saved_ms_per_token: float | None
    label: Label
    why: str


NKVO_SHORT_TOKENS = 4096  # context x slots up to which the CPU can read the KV cache each token
MIN_UBATCH = 64
MIN_POOL_TOKENS = 512
POOL_ROUND = 256
LONG_PROMPT_TOKENS = 2048


def _unit_move(anatomy: ModelAnatomy, p: Placement, unit: int, active: float) -> tuple[int, float]:
    """(device bytes to hold, bytes a token reads from there) of moving back the ``unit``-th
    host block, counting from the device boundary. A block whose experts stay on the host keeps
    them there."""
    index = p.host_layers - 1 - unit
    whole = anatomy.layer_bytes[index]
    experts = anatomy.expert_layer_bytes[index] if anatomy.expert_layer_bytes else 0
    if index < p.host_expert_layers:
        return whole - experts, whole - experts
    return whole, whole - experts * (1 - active)


def _saving_ms(model: DecodeModel, read: float, moved: float) -> float:
    """What moving ``read`` bytes a token reads from host to device saves; ``moved`` bytes
    no longer cross PCIe for each ubatch of a prompt."""
    return (
        _ms(read, model.bw_host_gbs) - _ms(read, model.bw_dev_gbs) + moved * model.stall_ms_per_byte
    )


def _fitting_layers(
    anatomy: ModelAnatomy, p: Placement, model: DecodeModel, room: int, kv_layer: int
) -> tuple[int, float, int]:
    """(layers, milliseconds saved, room left) of moving host layers back while each fits with
    its share of the KV cache."""
    count, saved = 0, 0.0
    active = active_share(anatomy, model.concurrency)
    for unit in range(p.host_layers):
        hold, read = _unit_move(anatomy, p, unit, active)
        need = hold + kv_layer
        if need > room:
            break
        room -= need
        saved += _saving_ms(model, read, hold)
        count += 1
    return count, saved, room


def _fitting_experts(
    anatomy: ModelAnatomy, p: Placement, model: DecodeModel, room: int
) -> tuple[int, float, int]:
    """The same for the experts of device blocks that ``--n-cpu-moe`` keeps on the host."""
    count, saved = 0, 0.0
    experts = anatomy.expert_layer_bytes
    active = active_share(anatomy, model.concurrency)
    for index in range(p.host_expert_layers - 1, p.host_layers - 1, -1):
        if experts[index] > room:
            break
        room -= experts[index]
        saved += _saving_ms(model, experts[index] * active, experts[index])
        count += 1
    return count, saved, room


def _expert_conversion(
    anatomy: ModelAnatomy, p: Placement, model: DecodeModel, room: int, kv_layer: int
) -> tuple[int, float] | None:
    """For a mixture of experts whose host layers are whole blocks (plain ``-ngl``): every layer
    on the device and the experts of the first N blocks on the host, the smallest N whose
    device bytes fit in ``room`` (measured, experts first). Returns (N, milliseconds saved)."""
    experts = anatomy.expert_layer_bytes
    blocks = p.host_layers
    dense = sum(anatomy.layer_bytes[i] - experts[i] for i in range(blocks)) + kv_layer * blocks
    active = active_share(anatomy, model.concurrency)
    total_read = p.device_bytes_per_token + p.host_bytes_per_token
    before = predict_decode_ms(model, p)
    for n in range(len(experts) + 1):
        back = sum(experts[i] for i in range(n, blocks))
        away = sum(experts[i] for i in range(blocks, n))
        if dense + back - away > room:
            continue
        host_read = sum(experts[:n]) * active
        after = _ms(host_read, model.bw_host_gbs) + _ms(total_read - host_read, model.bw_dev_gbs)
        moved = max(p.host_weight_bytes - sum(experts[:n]), 0)
        return n, before - after - model.overhead_ms + moved * model.stall_ms_per_byte
    return None


def _plan(
    anatomy: ModelAnatomy, p: Placement, model: DecodeModel, room: int, kv_layer: int, label: Label
) -> tuple[list[Step], float, int]:
    """The steps that spend ``room`` on moving experts, then layers, back to the device; the
    milliseconds they save and the room left."""
    steps: list[Step] = []
    saved_all = 0.0
    if p.host_expert_layers > p.host_layers and _is_moe(anatomy):
        count, saved, room = _fitting_experts(anatomy, p, model, room)
        if count:
            why = f"{count} layers of experts fit back on the device (--n-cpu-moe)"
            steps.append(
                Step("host_expert_layers", p.host_expert_layers - count, saved, label, why)
            )
            saved_all += saved
    count, saved, room = _fitting_layers(anatomy, p, model, room, kv_layer)
    if count:
        why = f"{count} host layers fit back on the device with their KV cache (-ngl up)"
        steps.append(Step("host_layers", p.host_layers - count, saved, label, why))
        saved_all += saved
    return steps, saved_all, room


def device_kv(p: Placement, fit: Fit2, layers: int) -> int:
    if p.kv_on_host:
        return 0
    return fit.kv_bytes * max(layers - p.host_layers, 0) // layers


def next_steps(
    anatomy: ModelAnatomy,
    p: Placement,
    model: DecodeModel,
    fit: Fit2,
    regime: Regime,
    step_ms: float | None = None,
    pcie_note: str = "",
    fit_places: bool = True,
) -> tuple[Step, ...]:
    """The moves that shorten a decode step, best first. On a mixture of experts launched with
    plain ``-ngl``, every layer on the device with the smallest ``--n-cpu-moe`` that fits comes
    first (measured to beat splitting layers) and is the only step. Otherwise experts and
    layers go back to the device. What still stays on the host after those is followed by
    the steps that free device memory, ranked by milliseconds saved per MiB freed. ``likely`` only
    for a calibrated model at concurrency 1;
    at a higher concurrency every saving with a measured host bandwidth is a lower bound,
    because the model counts the weights a step reads, not what batching and the queue add;
    with ``pcie_note`` it is an estimate, since the stall term counts one prompt per ubatch;
    a host bandwidth that is only the floor the step shows makes every saving an upper
    estimate; a saving at a higher concurrency with a stall term and a theoretical link rate is a
    rough estimate. No saving exceeds the step less the step of every weight on the device. The
    steps that free device memory save time only where ``--fit`` places the layers
    (``fit_places``): a command with its own ``-ngl`` leaves the freed memory unused."""
    if regime.placement == "cpu_only":
        return ()
    label: Label = "likely" if model.calibrated and model.concurrency == 1 else "possible"
    layers = anatomy.layers or 1
    room = fit.device_free - fit.device_need
    kv_layer = fit.kv_bytes // layers
    steps: list[Step] = []
    blocks = p.host_layers
    if _is_moe(anatomy) and p.host_expert_layers == 0 and blocks:
        converted = _expert_conversion(anatomy, p, model, room, kv_layer)
        if converted is not None and converted[1] > 0:
            n, saved = converted
            why = f"all layers on the device (-ngl all) with the experts of {n} layers on the host"
            value = {"host_layers": 0, "host_expert_layers": n}
            steps.append(Step("host_expert_layers", value, saved, label, why + " (--n-cpu-moe)"))
    if not steps:
        planned, _, left = _plan(anatomy, p, model, room, kv_layer, label)
        steps += planned
        after = {step.lever: step.value for step in planned}
        layers_left = after.get("host_layers", p.host_layers)
        experts_left = after.get("host_expert_layers", p.host_expert_layers)
        if fit_places and not (left < 0 or not (layers_left or experts_left > layers_left)):
            steps += _buy_room(anatomy, p, model, fit, room, left, kv_layer)
    return _bounded(steps, p, model, step_ms, pcie_note)


def _bounded(
    steps: list[Step], p: Placement, model: DecodeModel, step_ms: float | None, pcie_note: str
) -> tuple[Step, ...]:
    step = step_ms or predict_decode_ms(model, p)
    on_device = _ms(p.device_bytes_per_token + p.host_bytes_per_token, model.bw_dev_gbs)
    limit = step - on_device
    found = []
    theoretical = "theoretical" in pcie_note
    for item in steps:
        saved = item.saved_ms_per_token
        if saved is not None:
            saved = min(saved, limit)
            if saved <= 0:
                continue
        label: Label = item.label
        why = item.why
        if model.host_floor:
            label = "upper estimate"
            why += " (an upper estimate; host memory bandwidth: at least the decode step's own)"
        elif model.concurrency > 1 and model.stall_ms_per_byte and theoretical:
            label = "rough estimate"
            why += (
                f" (a rough estimate at concurrency {model.concurrency}: decode reads and "
                f"{pcie_note}; it can be off either way: the link's theoretical rate makes it "
                "low, counting one prompt per ubatch makes it high)"
            )
        elif model.concurrency > 1 and model.stall_ms_per_byte:
            label = "possible"
            why += (
                f" (an estimate at concurrency {model.concurrency}: decode reads and {pcie_note}; "
                "the stall term counts one prompt per ubatch, so batching can make it too high)"
            )
        elif model.concurrency > 1:
            label = "lower bound"
            why += f" (a lower bound at concurrency {model.concurrency}: decode reads only)"
        found.append(dataclasses.replace(item, saved_ms_per_token=saved, label=label, why=why))
    return tuple(found)


def _buy_room(
    anatomy: ModelAnatomy,
    p: Placement,
    model: DecodeModel,
    fit: Fit2,
    room: int,
    left: int,
    kv_layer: int,
) -> list[Step]:
    """What frees device memory for what is still on the host: a smaller context, fewer slots,
    a smaller ``-ub``, a q8_0 KV cache, the KV cache on the host. Each is ranked by the
    milliseconds the layers it admits save per MiB it frees; ``--n-cpu-ffn`` follows as
    something to measure."""
    layers = anatomy.layers or 1
    on_device_kv = device_kv(p, fit, layers)
    base = _plan(anatomy, p, model, room, kv_layer, "possible")[1]

    def gain(freed: int) -> float:
        return _plan(anatomy, p, model, room + freed, kv_layer, "possible")[1] - base

    options: list[tuple[str, Any, int, float, str]] = []
    pool, need = fit.pool_tokens, fit.need_tokens
    if pool and need and fit.slots:
        cut = max(-(-need * fit.slots // POOL_ROUND) * POOL_ROUND, MIN_POOL_TOKENS)
        if cut < pool:
            freed = on_device_kv * (pool - cut) // pool
            why = f"a context of {cut} tokens holds {fit.slots} requests of the longest size"
            options.append(("max_context_len", cut, freed, gain(freed), why))
        slots = max(fit.peak_running, 1)
        if slots < fit.slots:
            cut = max(-(-need * slots // POOL_ROUND) * POOL_ROUND, MIN_POOL_TOKENS)
            if cut < pool:
                freed = on_device_kv * (pool - cut) // pool
                why = (
                    f"the workload used {slots} of {fit.slots} slots; the context shrinks with them"
                )
                options.append(("max_concurrent_requests", slots, freed, gain(freed), why))
    if fit.ubatch // 2 >= MIN_UBATCH and fit.compute_buffer_bytes:
        freed = fit.compute_buffer_bytes // 2
        why = f"-ub {fit.ubatch // 2} halves the compute buffer; prompts run in smaller pieces"
        options.append(("ubatch", fit.ubatch // 2, freed, gain(freed), why))
    if on_device_kv:
        freed = round(on_device_kv * Q8_KV_SAVING)
        why = "a q8_0 KV cache is about half the size; check the answers"
        options.append(("kv_cache_precision", "8bit", freed, gain(freed), why))
        active = active_share(anatomy, model.concurrency)
        everything = sum(_unit_move(anatomy, p, u, active)[0] for u in range(p.host_layers))
        experts = anatomy.expert_layer_bytes[p.host_layers : p.host_expert_layers]
        everything = everything + sum(experts)
        short = fit.need_tokens * max(fit.slots, 1) <= NKVO_SHORT_TOKENS
        if room < everything <= room + on_device_kv and short:
            kv_ms = _ms(fit.kv_bytes * need // max(pool, 1), model.bw_host_gbs)
            net = gain(on_device_kv) - kv_ms
            why = "the KV cache on the host lets every weight fit on the device (-nkvo)"
            options.append(("kv_on_host", True, on_device_kv, net, why))
    ranked = sorted(options, key=lambda o: o[3] / (o[2] / MIB) if o[2] > 0 else 0.0, reverse=True)
    found = [
        Step(lever, value, saved, "possible", why)
        for lever, value, _, saved, why in ranked
        if saved > 0
    ]
    dense = not _is_moe(anatomy)
    if dense and p.host_layers and fit.need_tokens >= LONG_PROMPT_TOKENS and not found:
        blocks = p.host_layers
        attention = anatomy.weight_bytes_by_role.get("attention", 0) // layers
        if blocks and blocks * (attention + kv_layer) <= room:
            why = (
                "attention and KV stay on the device for the host layers; measure it (--n-cpu-ffn)"
            )
            found.append(Step("host_ffn_layers", blocks, None, "possible", why))
    return found


@dataclass(frozen=True, slots=True)
class Prefill:
    """The prompt work of a run: ``ubatches`` it took in all (each streams the host weights over
    PCIe) and the decode ``steps`` they were spread over."""

    ubatches: int
    steps: float


def placement_block(
    anatomy: ModelAnatomy,
    resolved: Resolved | None,
    *,
    step_ms: float | None,
    bw_dev_gbs: float | None,
    concurrency: int,
    host: HostFacts,
    platform: str,
    headroom: int | None,
    compute_buffer_bytes: int,
    kv_footprint: int,
    type_k: str,
    type_v: str,
    pool_tokens: int,
    ubatch: int = 0,
    slots: int = 0,
    need_tokens: int = 0,
    peak_running: int = 0,
    bw_host_gbs: float | None = None,
    major_faults: int | None = None,
    device_free: int | None = None,
    kv_logged: int = 0,
    prefill: Prefill | None = None,
    pcie: tuple[float, str] | None = None,
    fit_places: bool = True,
) -> dict[str, Any] | None:
    """The placement, regime, decode model, fit and next steps of one run, as JSON.
    ``headroom`` is the device memory left free while the baseline was loaded, ``device_free``
    what the log says was free when it started (it replaces ``headroom``), ``kv_logged`` the
    log's KV buffers. ``pcie`` is a rate in GB/s and where it comes from; ``fit_places`` whether
    ``--fit`` chose the layer placement."""
    if resolved is None or (found := place(anatomy, resolved)) is None:
        return None
    if isinstance(found, str):
        return {"unplaced": found}
    p = found
    read = placement(anatomy, resolved, concurrency) or p
    shape = regime(p, host, platform, major_faults)
    block: dict[str, Any] = {"placement": asdict(p), "regime": asdict(shape), "step_ms": step_ms}
    model = None
    if step_ms and bw_dev_gbs and shape.placement not in ("full_gpu", "kv_host"):
        try:
            model = calibrate(
                weight_bytes=read.device_bytes_per_token + read.host_bytes_per_token,
                bw_dev_gbs=bw_dev_gbs, full_gpu_ms=None, cpu_ms=None, bw_host_gbs=bw_host_gbs,
                concurrency=concurrency, baseline_ms=step_ms, baseline=read,
            )  # fmt: skip
        except ValueError:
            model = None
    note = ""
    if model is not None and prefill and pcie and prefill.steps > 0 and concurrency > 1:
        rate, source = pcie
        note = f"prompt streaming over PCIe at {rate:.1f} GB/s, {source}"
        per_byte = prefill.ubatches / (rate * 1e6 * prefill.steps)
        model = dataclasses.replace(model, stall_ms_per_byte=per_byte)
    block["decode_model"] = None if model is None else asdict(model)
    block["overhead_share"] = None if model is None or not step_ms else model.overhead_ms / step_ms
    fit = None
    free = device_free or (
        None
        if headroom is None
        else headroom + p.device_weight_bytes + kv_footprint + compute_buffer_bytes
    )
    if free is not None and shape.placement != "cpu_only" and pool_tokens:
        fit = fit_domains(
            anatomy, p, pool_tokens=pool_tokens, type_k=type_k, type_v=type_v,
            compute_buffer_bytes=compute_buffer_bytes, device_free=free, host=host,
            ubatch=ubatch, slots=slots, need_tokens=need_tokens, peak_running=peak_running,
            kv_logged=kv_logged,
        )  # fmt: skip
    block["fit"] = None if fit is None else asdict(fit)
    steps = (
        ()
        if model is None or fit is None
        else next_steps(anatomy, read, model, fit, shape, step_ms, note, fit_places)
    )
    block["next_steps"] = [asdict(step) for step in steps]
    if shape.placement != "cpu_only" and free is None:
        block["next_steps_cant_tell"] = (
            "the free device memory was not sampled, so what fits on the device is unknown"
        )
    return block
