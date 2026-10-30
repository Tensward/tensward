"""The fact vocabulary: every named, typed value a knowledge record's condition may read, and
the extraction of those values from one analysed run.

``VOCABULARY`` is versioned as ``FACTS_VERSION``: a new fact is a minor bump, a changed meaning
a new name. A fact Tensward did not measure reads :data:`UNKNOWN`, never a guess, so a
condition over it is unknown rather than false. Extraction is pure: the same situation,
context, engine and version always give the same facts and the same digest.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from . import trace_facts
from .anatomy import freed_bytes
from .classify import NAMES
from .context import GOALS
from .engines import ENGINES
from .engines.llamacpp.startup import CPU_FEATURE_NAMES
from .environment import describe_host
from .fact_values import UNKNOWN
from .levers import LEVERS
from .measurement import nearest_rank
from .platforms.cpu import ISA_FLAGS
from .playbook_common import kv_held_back
from .predict import saturated, step_held
from .pressure import demand

if TYPE_CHECKING:
    from .anatomy import ModelAnatomy
    from .context import AnalysisContext
    from .engines.protocol import Engine
    from .measurement import Measurement
    from .playbook import Situation
    from .settings import Settings

FACTS_VERSION = 3
START_CLASSES = tuple(
    dict.fromkeys(name for engine in ENGINES.values() for name in engine.log.start_classes)
)
FactType = Literal["int", "float", "bool", "str", "enum", "set", "version"]


@dataclass(frozen=True, slots=True, kw_only=True)
class FactSpec:
    """One fact. ``values`` lists an enum's values, or the element values of a set of names;
    a set without ``values`` holds integers."""

    name: str
    type: FactType
    unit: str | None  # "GiB", "ms", "tokens", "share", None
    values: tuple[str, ...] | None
    source: str  # "situation", "context", "anatomy", "engine", "measurement"
    description: str
    engines: tuple[str, ...] = ()  # the engines that define it with this meaning; empty: all


BOTTLENECKS = tuple(NAMES)
MODEL_FORMATS = ("hf", "gguf")
PLATFORMS = ("nvidia", "apple", "cpu")
BACKENDS = ("cuda", "hip", "vulkan", "metal", "sycl", "musa", "cpu")
COMPUTE = ("gpu", "cpu", "mixed")
PLACEMENTS = (
    "full_gpu",
    "weight_offload",
    "dense_offload",
    "expert_offload",
    "kv_host",
    "cpu_only",
)
FIT_CHANGES = ("ctx_size", "n_gpu_layers", "tensor_overrides")
ARCHITECTURES = ("turing", "ampere", "ada", "hopper", "blackwell")
_LEVER_TYPES: Mapping[str, FactType] = {
    "int": "int",
    "float": "float",
    "bool": "bool",
    "str": "str",
    "str|None": "str",
}
_LEVER_UNITS = {
    "max_context_len": "tokens",
    "prefill_batch_tokens": "tokens",
    "kv_memory_fraction": "share",
}


def _spec(
    name: str,
    type_: FactType,
    source: str,
    description: str,
    unit: str | None = None,
    values: tuple[str, ...] | None = None,
    engines: tuple[str, ...] = (),
) -> FactSpec:
    return FactSpec(
        name=name,
        type=type_,
        unit=unit,
        values=values,
        source=source,
        description=description,
        engines=engines,
    )


def _lever_specs() -> Iterator[FactSpec]:
    """``settings.<lever>`` for every lever whose value is a scalar or a fixed set of words."""
    for lever in LEVERS.values():
        if lever.value in _LEVER_TYPES:
            type_, values = _LEVER_TYPES[lever.value], None
        elif "|" in lever.value and "None" not in lever.value:
            type_, values = "enum", tuple(lever.value.split("|"))
        else:
            continue
        unit = _LEVER_UNITS.get(lever.name)
        yield _spec(f"settings.{lever.name}", type_, "engine", lever.meaning, unit, values)


_LLAMACPP = ("llamacpp",)
_PLACEMENT_FACTS: tuple[tuple[str, FactType, str | None, str], ...] = (
    ("run.slots", "int", None, "request slots"),
    ("run.slot_ctx", "int", "tokens", "the context each slot holds, as the server logged it"),
    ("run.kv_unified", "bool", None, "the slots share one KV pool"),
    ("run.kv_type_k", "str", None, "the K cache type"),
    ("run.kv_type_v", "str", None, "the V cache type"),
    ("run.flash_attn", "bool", None, "flash attention resolved on"),
    ("run.host_layers", "int", None, "layers kept in host memory"),
    ("run.host_expert_layers", "int", None, "layers whose experts stay in host memory"),
    ("run.host_weight_bytes", "int", "bytes", "weights in host memory"),
    ("run.device_weight_bytes", "int", "bytes", "weights in device memory"),
    ("run.host_bytes_per_token", "int", "bytes", "weight bytes a token reads from the host"),
    ("run.device_bytes_per_token", "int", "bytes", "weight bytes a token reads on the device"),
    ("run.threads", "int", None, "generation threads"),
    ("run.threads_batch", "int", None, "prompt-processing threads"),
    ("run.major_faults", "int", None, "page faults that read the disk, while decoding"),
    ("run.compute_buffer_mib", "float", "MiB", "the compute buffer"),
    ("run.pcie_gbs_effective", "float", "GB/s", "the rate host weights cross PCIe at"),
    ("run.host_memory_tight", "bool", None, "the host cannot keep the weights resident"),
    ("run.numa_split", "bool", None, "CPU compute spans several NUMA nodes"),
    ("run.thinking_tokens_p50", "float", "tokens", "median tokens of answers that only think"),
    ("host.physical_cores", "int", None, "physical CPU cores"),
    ("host.logical_cpus", "int", None, "logical CPUs"),
    ("host.usable_cpus", "int", None, "CPUs the server may use: cgroup, cpuset, affinity"),
    ("host.cpu_quota_cores", "float", None, "the cgroup CPU quota, in cores"),
    ("host.smt", "bool", None, "simultaneous multithreading is on"),
    ("host.mem_gib", "float", "GiB", "host memory"),
    ("host.mem_available_gib", "float", "GiB", "host memory available at the run"),
    ("host.cgroup_mem_gib", "float", "GiB", "the cgroup memory limit"),
    ("host.memlock_gib", "float", "GiB", "the locked-memory limit"),
    ("host.numa_nodes", "int", None, "NUMA nodes"),
    ("host.mem_bw_gbs", "float", "GB/s", "host memory bandwidth, measured by a launch"),
    ("gpu.pcie_gen", "int", None, "the PCIe generation the GPU's link supports"),
    ("gpu.pcie_width", "int", None, "the PCIe lanes the GPU's link supports"),
    ("gpu.pcie_gbs_theoretical", "float", "GB/s", "the link's one-direction bandwidth"),
)


def _placement_specs() -> Iterator[FactSpec]:
    """Where the run kept its weights and KV cache, the host it ran on and the link between;
    llama.cpp reads them, vLLM only ``run.compute`` and ``run.placement``."""
    yield _spec("run.compute", "enum", "measurement", "where the model computes", None, COMPUTE)
    yield _spec(
        "run.placement", "enum", "measurement", "where weights and KV live", None, PLACEMENTS
    )
    for name, type_, unit, text in _PLACEMENT_FACTS:
        yield _spec(name, type_, "measurement", text, unit, engines=_LLAMACPP)
    yield _spec(
        "run.fit_changed",
        "set",
        "measurement",
        "what --fit changed",
        None,
        FIT_CHANGES,
        engines=_LLAMACPP,
    )
    yield _spec(
        "host.isa",
        "set",
        "measurement",
        "CPU instruction sets",
        None,
        tuple(sorted(ISA_FLAGS)),
        engines=_LLAMACPP,
    )


_SPECS: tuple[FactSpec, ...] = (
    _spec("model.family", "str", "situation", "config.json's model_type"),
    _spec("model.arch", "str", "situation", "config.json's architecture class"),
    _spec("model.params_b", "float", "anatomy", "matmul parameters, in billions"),
    _spec("model.weight_gib", "float", "anatomy", "checkpoint weights at stored precision", "GiB"),
    _spec("model.quant", "str", "situation", "the checkpoint's quantization method, or none"),
    _spec("model.quant_bits", "int", "situation", "bits per quantized weight"),
    _spec("model.dtype", "str", "situation", "the checkpoint's stored dtype"),
    _spec("model.context_limit", "int", "situation", "max_position_embeddings", "tokens"),
    _spec("model.gptq_desc_act", "bool", "situation", "a GPTQ checkpoint with act-order"),
    _spec("model.has_mtp", "bool", "situation", "the checkpoint carries multi-token heads"),
    _spec("model.validated", "bool", "anatomy", "Tensward validated this model family"),
    _spec("anatomy.layers", "int", "anatomy", "decoder layers"),
    _spec("anatomy.kv_shared_layers", "int", "anatomy", "layers that reuse earlier layers' KV"),
    _spec("anatomy.per_layer_embeddings_gib", "float", "anatomy", "per-layer embeddings", "GiB"),
    _spec("anatomy.media_towers", "bool", "anatomy", "the checkpoint has image or audio towers"),
    _spec("anatomy.tower_gib", "float", "anatomy", "image and audio tower weights", "GiB"),
    _spec("anatomy.head_dims", "set", "anatomy", "the attention head dims"),
    _spec("anatomy.max_head_dim", "int", "anatomy", "the largest attention head dim"),
    _spec("anatomy.mixed_head_dims", "bool", "anatomy", "more than one attention head dim"),
    _spec("anatomy.sliding_window", "bool", "anatomy", "some layers attend in a sliding window"),
    _spec("anatomy.moe", "bool", "anatomy", "routed experts"),
    _spec("anatomy.experts", "int", "anatomy", "routed experts per MoE layer"),
    _spec("anatomy.experts_per_token", "int", "anatomy", "experts each token is routed to"),
    _spec("anatomy.mla", "bool", "anatomy", "multi-head latent attention"),
    _spec("anatomy.linear_attention", "bool", "anatomy", "linear-attention (state) layers"),
    _spec("anatomy.tied_embeddings", "bool", "anatomy", "the output head is the embedding"),
    _spec(
        "anatomy.kv_bytes_per_token",
        "int",
        "anatomy",
        "KV-cache bytes of one token, at the run's KV-cache dtype",
    ),
    _spec("anatomy.quantizable_gib", "float", "anatomy", "weights fp8 would shrink", "GiB"),
    _spec("anatomy.unclassified_share", "float", "anatomy", "tensor bytes not classified", "share"),
    _spec("model.format", "enum", "engine", "how the checkpoint is stored", None, MODEL_FORMATS),
    _spec("model.ftype", "str", "anatomy", "the GGUF file type, such as Q6_K", engines=_LLAMACPP),
    _spec(
        "model.bits_per_weight",
        "float",
        "anatomy",
        "stored bits per weight, GGUF",
        engines=_LLAMACPP,
    ),
    _spec("anatomy.routed_experts", "float", "anatomy", "routed-expert weights", "GiB"),
    _spec(
        "platform.name",
        "enum",
        "measurement",
        "the accelerator platform",
        None,
        PLATFORMS,
        engines=_LLAMACPP,
    ),
    _spec("gpu.model", "str", "measurement", "the serving GPU's name"),
    _spec("host.memory_gib", "float", "measurement", "the host's memory", "GiB"),
    _spec("gpu.cc", "float", "context", "the GPU's compute capability"),
    _spec("gpu.arch", "enum", "context", "the GPU's architecture", None, ARCHITECTURES),
    _spec("gpu.memory_gib", "float", "measurement", "the GPU's memory", "GiB"),
    _spec("gpu.count", "int", "engine", "GPUs one server spans"),
    _spec("gpu.power_cap_w", "float", "measurement", "the enforced power limit"),
    _spec("engine.name", "str", "engine", "the engine family"),
    _spec("engine.version", "version", "engine", "the engine's version"),
    _spec("engine.build", "int", "engine", "the llama.cpp build number", engines=_LLAMACPP),
    _spec(
        "engine.backend",
        "enum",
        "engine",
        "the backend that holds the weights",
        None,
        BACKENDS,
        engines=_LLAMACPP,
    ),
    _spec(
        "engine.cpu_features",
        "set",
        "engine",
        "CPU features the build uses",
        None,
        CPU_FEATURE_NAMES,
        engines=_LLAMACPP,
    ),
    *_lever_specs(),
    _spec("settings.speculation_method", "str", "engine", "the speculative method, or none"),
    _spec("workload.tools", "bool", "situation", "some prompt offers tools"),
    _spec("workload.structured", "bool", "situation", "some request is structured output"),
    _spec("workload.media", "bool", "situation", "some prompt sends images"),
    _spec("workload.prompt_tokens_p50", "int", "context", "median prompt length", "tokens"),
    _spec("workload.prompt_tokens_p95", "int", "context", "95th-percentile prompt", "tokens"),
    _spec("workload.prompt_tokens_max", "int", "measurement", "longest prompt", "tokens"),
    _spec("workload.output_tokens_limit", "int", "situation", "the output cap", "tokens"),
    _spec("workload.output_tokens_p50", "int", "measurement", "median answer length", "tokens"),
    _spec("workload.arrival", "enum", "situation", "how requests arrive", None, ("closed", "open")),
    _spec("workload.clients", "int", "situation", "closed-loop clients"),
    _spec("workload.max_inflight", "int", "situation", "the most requests kept in flight"),
    _spec(
        "workload.shared_prefix_share",
        "float",
        "measurement",
        "prompt tokens in a prefix another request shares",
        "share",
    ),
    _spec("workload.json_answers", "bool", "context", "some answers must be JSON"),
    _spec("diag.primary", "enum", "situation", "the primary bottleneck", None, BOTTLENECKS),
    _spec("diag.secondary", "set", "situation", "the secondary bottlenecks", None, BOTTLENECKS),
    _spec("diag.fired", "set", "situation", "classes at warning or critical", None, BOTTLENECKS),
    _spec("diag.near", "set", "situation", "classes near their warning level", None, BOTTLENECKS),
    _spec("diag.gates", "set", "situation", "gates at warning or critical", None, BOTTLENECKS),
    _spec("diag.confidence", "str", "situation", "the diagnosis's confidence"),
    _spec("run.kv_usage_peak", "float", "measurement", "peak KV-cache usage", "share"),
    _spec("run.waiting_peak", "float", "measurement", "peak waiting requests"),
    _spec("run.preemptions", "float", "measurement", "preempted requests"),
    _spec("run.req_s", "float", "measurement", "requests per second"),
    _spec("run.ttft_p95_ms", "float", "measurement", "95th-percentile TTFT", "ms"),
    _spec("run.tpot_p95_ms", "float", "measurement", "95th-percentile TPOT", "ms"),
    _spec(
        "run.decode_share_of_ceiling",
        "float",
        "measurement",
        "decode rate over its ceiling",
        "share",
    ),
    _spec("run.kv_pool_tokens", "float", "measurement", "tokens the KV cache holds", "tokens"),
    _spec(
        "run.compile_cache", "enum", "measurement", "torch.compile's cache", None, ("warm", "cold")
    ),
    _spec("run.requests", "int", "measurement", "requests sent, succeeded or failed"),
    _spec("run.mean_running", "float", "measurement", "requests running, averaged"),
    _spec(
        "run.running_share", "float", "measurement", "mean running over the clients or cap", "share"
    ),
    _spec("run.mean_waiting", "float", "measurement", "requests waiting, averaged"),
    _spec(
        "run.waiting_capacity",
        "float",
        "measurement",
        "waiting for capacity: a gauge read at two scrapes only",
    ),
    _spec(
        "run.waiting_deferred",
        "float",
        "measurement",
        "waiting deferred: a gauge read at two scrapes only",
    ),
    _spec("run.kv_pressure", "bool", "measurement", "KV space held requests back"),
    _spec("run.saturated", "bool", "measurement", "the clients or cap are all running, none wait"),
    _spec("run.running_p50", "float", "measurement", "median of the logged running count"),
    _spec("run.running_p90", "float", "measurement", "90th percentile of the logged running count"),
    _spec("run.server_ttft_p50_ms", "float", "measurement", "engine-timed median TTFT", "ms"),
    _spec("run.server_tpot_p50_ms", "float", "measurement", "engine-timed median TPOT", "ms"),
    _spec("run.step_held_ratio", "float", "measurement", "decode steps a prompt waits in"),
    _spec("run.queue_share", "float", "measurement", "engine TTFT spent queued", "share"),
    _spec(
        "run.mixed_step_share", "float", "measurement", "steps that computed prompt tokens", "share"
    ),
    _spec("run.sm_clock_mhz", "float", "measurement", "mean SM clock while measured", "MHz"),
    _spec(
        "run.prefix_hit_rate", "float", "measurement", "prompt tokens served from cache", "share"
    ),
    _spec("run.kv_pool_gib", "float", "measurement", "memory left for the KV cache", "GiB"),
    _spec("run.kv_blocks", "int", "measurement", "blocks in the KV cache pool"),
    _spec("run.start_class", "enum", "measurement", "how the start went", None, START_CLASSES),
    _spec("run.start_s", "float", "measurement", "seconds from launch to ready", "s"),
    _spec("run.wave_period_s", "float", "measurement", "period of the closed-loop waves", "s"),
    _spec(
        "workload.possible_prefix_hit",
        "float",
        "measurement",
        "prompt tokens a cache that never evicts would serve",
        "share",
    ),
    _spec(
        "workload.common_prefix_tokens",
        "int",
        "measurement",
        "the prefix every request shares",
        "tokens",
    ),
    *_placement_specs(),
    _spec("goal", "enum", "context", "what the analysis optimizes", None, GOALS),
    _spec("limit.ttft_p95_ms", "float", "context", "the declared TTFT limit", "ms"),
    _spec("limit.tpot_p95_ms", "float", "context", "the declared TPOT limit", "ms"),
    _spec("limit.req_s_floor", "float", "context", "the declared requests/s loss allowed", "share"),
)
_VLLM_ONLY = frozenset(
    {
        "settings.prefill_batch_tokens",
        "settings.speculation_method",
        "run.kv_usage_peak",
        "run.kv_pressure",
        "run.kv_blocks",
        "run.prefix_hit_rate",
        "run.compile_cache",
        "run.start_class",
    }
)
VOCABULARY: Mapping[str, FactSpec] = {
    spec.name: replace(spec, engines=("vllm",)) if spec.name in _VLLM_ONLY else spec
    for spec in _SPECS
}


def _checked(spec: FactSpec, value: object) -> object:
    """``value`` in its stored form; ValueError when it does not have the fact's type."""
    if spec.type == "int" and type(value) is int:
        return value
    if spec.type == "float" and isinstance(value, (int, float)) and type(value) is not bool:
        if math.isfinite(value):
            return float(value)
    if spec.type == "bool" and type(value) is bool:
        return value
    if spec.type in ("str", "version") and type(value) is str:
        return value
    if spec.type == "enum" and value in (spec.values or ()):
        return value
    if spec.type == "set" and isinstance(value, (set, frozenset, list, tuple)):
        items = frozenset(value)
        if spec.values is None and all(type(item) is int for item in items):
            return items
        if spec.values is not None and items <= set(spec.values):
            return items
    raise ValueError(f"{spec.name} is {spec.type}, not {value!r}")


class Facts(Mapping[str, object]):
    """Immutable; keys are vocabulary names; a vocabulary name not measured reads UNKNOWN;
    a name outside the vocabulary raises KeyError."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, object]) -> None:
        unknown = sorted(set(values) - set(VOCABULARY))
        if unknown:
            raise KeyError(f"not in the fact vocabulary: {', '.join(unknown)}")
        self._values = {
            name: _checked(VOCABULARY[name], value)
            for name, value in values.items()
            if value is not None and value is not UNKNOWN
        }

    def __getitem__(self, name: str) -> object:
        if name not in VOCABULARY:
            raise KeyError(name)
        return self._values.get(name, UNKNOWN)

    def __iter__(self) -> Iterator[str]:
        return iter(VOCABULARY)

    def __len__(self) -> int:
        return len(VOCABULARY)

    def __repr__(self) -> str:
        return f"Facts({self.to_json()!r})"

    def digest(self) -> str:
        return "sha256:" + hashlib.sha256(canonical_json(self.to_json())).hexdigest()

    def to_json(self) -> dict[str, Any]:
        """The measured facts only, sets as sorted lists."""
        return {
            name: sorted(value) if isinstance(value, frozenset) else value
            for name, value in sorted(self._values.items())
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Facts:
        return cls(data)


def canonical_json(value: Any) -> bytes:
    """Sorted keys, separators (",", ":"), ensure_ascii=False, floats by repr, UTF-8; NaN and
    infinities raise."""
    text = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return text.encode("utf-8")


ARCH_BY_CC: tuple[tuple[float, float, str], ...] = (
    (7.5, 8.0, "turing"),
    (8.0, 8.9, "ampere"),
    (8.9, 9.0, "ada"),
    (9.0, 10.0, "hopper"),
    (10.0, 13.0, "blackwell"),
)
_CLIENTS = re.compile(r"(\d+) concurrent clients")
_FIRED = ("warning", "critical")


def arch_of(cc: float) -> str | None:
    return next((arch for low, high, arch in ARCH_BY_CC if low <= cc < high), None)


def extract(
    situation: Situation,
    context: AnalysisContext,
    engine: Engine,
    version: str | None,
    *,
    anatomy: ModelAnatomy | None,
) -> Facts:
    """Fills every fact whose source Tensward has: the model, the GPU, the engine and its
    settings, the workload, the diagnosis, the run, the goal and limits, and the anatomy's
    fields; the rest stay UNKNOWN. The same arguments and run directory give the same facts."""
    workload, settings, measured, diagnosis = (
        situation.facts,
        situation.settings,
        situation.measurement,
        situation.diagnosis,
    )
    found: dict[str, object] = {
        "model.family": workload.model_type,
        "model.context_limit": workload.context_limit,
        "gpu.model": measured.ceilings.gpu if measured.ceilings else None,
        "gpu.cc": context.gpu_capability,
        "gpu.arch": arch_of(context.gpu_capability) if context.gpu_capability else None,
        "gpu.memory_gib": measured.gpu_memory_gib,
        "gpu.count": engine.parallel_degree(settings) or None,
        "gpu.power_cap_w": diagnosis.gpu_state.power_limit_w if diagnosis.gpu_state else None,
        "engine.name": engine.family,
        "engine.version": version,
        "model.format": "gguf" if "gguf" in engine.formats else "hf",
        "workload.tools": workload.offers_tools,
        "workload.structured": workload.structured,
        "workload.media": workload.sends_images,
        "workload.prompt_tokens_max": measured.max_prompt_tokens,
        "workload.output_tokens_limit": workload.output_tokens,
        "workload.max_inflight": workload.max_inflight,
        "workload.json_answers": bool(context.json_answers),
        "diag.primary": diagnosis.primary,
        "diag.secondary": diagnosis.secondary,
        "diag.fired": [f.bottleneck for f in diagnosis.findings if f.state in _FIRED],
        "diag.near": [f.bottleneck for f in diagnosis.findings if f.near],
        "diag.gates": [f.bottleneck for f in diagnosis.gates if f.state in _FIRED],
        "diag.confidence": diagnosis.confidence,
        "run.kv_usage_peak": measured.peak_kv_usage,
        "run.waiting_peak": measured.peak_waiting,
        "run.preemptions": measured.preemptions,
        "run.req_s": measured.request_throughput,
        "run.ttft_p95_ms": measured.ttft_p95_ms,
        "run.tpot_p95_ms": measured.tpot_p95_ms,
        "run.kv_pool_tokens": measured.kv_capacity_tokens,
        "run.compile_cache": measured.compile_cache,
        "run.requests": measured.succeeded + measured.failed,
        "goal": context.goal,
        "limit.ttft_p95_ms": context.ttft_p95_limit_ms,
        "limit.tpot_p95_ms": context.tpot_p95_limit_ms,
        "limit.req_s_floor": context.max_req_s_loss,
    }
    ceilings = measured.ceilings
    if ceilings is not None and ceilings.decode_pct_of_ceiling is not None:
        found["run.decode_share_of_ceiling"] = ceilings.decode_pct_of_ceiling / 100
    if context.prompt_blocks:
        lengths = [float(tokens) for tokens, _ in context.prompt_blocks.values()]
        found["workload.prompt_tokens_p50"] = int(nearest_rank(lengths, 0.5))
        found["workload.prompt_tokens_p95"] = int(nearest_rank(lengths, 0.95))
    if clients := _CLIENTS.fullmatch(workload.load):
        found["workload.arrival"], found["workload.clients"] = "closed", int(clients[1])
    elif "requests/s" in workload.load:
        found["workload.arrival"] = "open"
    found.update(_settings(engine, settings, measured, version, anatomy))
    found.update(_run(situation, context, engine))
    if anatomy is not None:
        found.update(_anatomy(anatomy, settings.kv_cache_dtype))
    if "cpu" in (found.get("run.compute"), found.get("engine.backend")):
        found = {name: value for name, value in found.items() if not name.startswith("gpu.")}
    return Facts(found)


def _run(situation: Situation, context: AnalysisContext, engine: Engine) -> dict[str, object]:
    """The facts of the load the run saw: the engine's pressure and pace, its histograms and
    log, the request trace, the start and the host. Those read from the run directory stay
    unknown when it, or the file, is missing."""
    measured, settings = situation.measurement, situation.settings
    wanted, cap = demand(measured), settings.max_concurrent_requests
    found: dict[str, object] = {
        "run.mean_running": measured.mean_running,
        "run.mean_waiting": measured.mean_waiting,
        "run.server_ttft_p50_ms": measured.server_ttft_p50_ms,
        "run.queue_share": measured.queue_share,
        "run.prefix_hit_rate": measured.prefix_cache_hit_rate,
        "run.kv_pool_gib": measured.kv_available_gib,
        "run.kv_blocks": round(measured.kv_blocks) if measured.kv_blocks is not None else None,
        "run.step_held_ratio": step_held(measured, cap),
        "run.start_s": context.load_s,
        "run.sm_clock_mhz": situation.diagnosis.gpu_state.sm_mhz
        if situation.diagnosis.gpu_state
        else None,
        "run.wave_period_s": (context.window_plan or {}).get("wave_period_s"),
    }
    if measured.mean_running is not None and wanted:
        share = measured.mean_running / min(wanted, cap or wanted)
        found["run.running_share"] = share
        if measured.mean_waiting is not None:
            found["run.saturated"] = saturated(measured, cap)
    if measured.preemptions or (
        measured.mean_waiting is not None and measured.peak_running is not None and wanted
    ):
        found["run.kv_pressure"] = kv_held_back(situation)
    run_dir = context.this_run()
    span = None
    if run_dir is not None:
        rows = trace_facts.request_rows(run_dir)
        if rows:
            found["workload.output_tokens_p50"] = trace_facts.output_tokens_p50(rows)
            if context.prompt_blocks and measured.kv_block_tokens:
                reuse = trace_facts.prefix_reuse(
                    rows, context.prompt_blocks, int(measured.kv_block_tokens)
                )
                if reuse is not None:
                    possible, shared, common = reuse
                    found["workload.possible_prefix_hit"] = possible
                    found["workload.shared_prefix_share"] = shared
                    found["workload.common_prefix_tokens"] = common
        span = trace_facts.load_span_s(rows)
        found["host.memory_gib"] = _host_memory_gib(run_dir, context)
    found.update(engine.log.run_facts(run_dir, context, situation, span))
    return found


def _text(path: Path) -> str | None:
    try:
        return path.read_text("utf-8", errors="replace")
    except OSError:
        return None


def _host_memory_gib(run_dir: Path, context: AnalysisContext) -> float | None:
    """The host's memory as the run recorded it; for the run being analysed, whose run.json is
    not written yet, the host Tensward is running on."""
    recorded = _text(run_dir / "run.json")
    if recorded is not None:
        ram = (json.loads(recorded).get("host") or {}).get("ram_bytes")
    elif context.reported_run is None and context.new_run is None:
        ram = describe_host().ram_bytes
    else:
        ram = None
    return ram / 2**30 if ram else None


def _settings(
    engine: Engine,
    settings: Settings,
    measured: Measurement,
    version: str | None,
    anatomy: ModelAnatomy | None,
) -> dict[str, object]:
    takes_images = (
        anatomy is not None and "image" in anatomy.modalities and settings.media_inputs is not False
    )
    found: dict[str, object] = {}
    for name in VOCABULARY:
        lever = name.removeprefix("settings.")
        if lever == name or lever not in LEVERS:
            continue
        try:
            value = engine.lever_value(settings, lever)
        except ValueError:
            continue  # the engine does not realise this lever
        if value is None:
            value = engine.resolved_default(
                lever,
                version=version,
                gpu_memory_gib=measured.gpu_memory_gib,
                gpu_model=measured.ceilings.gpu if measured.ceilings else None,
                step_budget_tokens=measured.step_budget_tokens,
                takes_images=takes_images,
                frontend_processes=engine.frontend_processes(settings) or 1,
            )
        found[name] = value
    speculation = engine.lever_value(settings, "speculation")
    found["settings.speculation_method"] = "none" if speculation is None else speculation.method
    return found


def _anatomy(anatomy: ModelAnatomy, kv_cache_dtype: str) -> dict[str, object]:
    found: dict[str, object] = {
        "model.validated": anatomy.validated,
        "model.arch": anatomy.architecture,
        "model.dtype": anatomy.dtype,
        "model.quant": anatomy.quantization,
        "model.quant_bits": anatomy.weight_bits,
        "anatomy.layers": anatomy.layers,
        "anatomy.tied_embeddings": anatomy.tied_embeddings,
        "anatomy.kv_bytes_per_token": anatomy.kv_bytes(1, kv_cache_dtype),
        "anatomy.kv_shared_layers": anatomy.kv_shared_layers,
        "anatomy.mla": anatomy.mla,
        "anatomy.linear_attention": anatomy.linear_attention,
        "anatomy.unclassified_share": anatomy.unclassified_share,
    }
    if anatomy.quantization is not None:
        found["model.gptq_desc_act"] = anatomy.gptq_act_order
    if anatomy.quantization and anatomy.quantization.startswith("gguf:"):
        found["model.ftype"] = anatomy.quantization.removeprefix("gguf:")
    found["model.bits_per_weight"] = anatomy.bits_per_weight
    parts, params = anatomy.components, anatomy.params
    if params is not None:
        found["model.params_b"] = sum(getattr(params, n) for n in params.__slots__) / 1e9
    if parts is not None:
        found["model.weight_gib"] = sum(getattr(parts, n) for n in parts.__slots__) / 2**30
        found["model.has_mtp"] = parts.mtp > 0
        found["anatomy.routed_experts"] = parts.routed_experts / 2**30
        found["anatomy.media_towers"] = parts.vision + parts.audio > 0
        found["anatomy.tower_gib"] = (parts.vision + parts.audio) / 2**30
        found["anatomy.per_layer_embeddings_gib"] = parts.gathered_embedding / 2**30
        found["anatomy.quantizable_gib"] = (freed_bytes(anatomy) or {})["quantizable"] / 2**30
    if anatomy.attention:
        dims = {group.head_dim for group in anatomy.attention}
        found["anatomy.head_dims"] = dims
        found["anatomy.max_head_dim"] = max(dims)
        found["anatomy.mixed_head_dims"] = len(dims) > 1
        found["anatomy.sliding_window"] = any(g.kind == "sliding" for g in anatomy.attention)
    if anatomy.moe is not None:
        found["anatomy.moe"] = True
        found["anatomy.experts"] = anatomy.moe.experts
        found["anatomy.experts_per_token"] = anatomy.moe.experts_per_token
    elif parts is not None and parts.routed_experts == 0:
        found["anatomy.moe"] = False
        found["anatomy.experts"] = found["anatomy.experts_per_token"] = 0
    return found
