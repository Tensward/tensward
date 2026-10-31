"""What a llama-server start-up log (``-lv 4``) says about the launch: where the weights, the KV
cache and the compute buffers went, how the context was split into slots, and what ``--fit``
changed."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from ..protocol import Allocation

LINE_PREFIX = re.compile(r"^\d+\.\d{2}\.\d{3}\.\d{3} [A-Z] ")
MIB = 2**20
_SIZE = r"([\d.]+) MiB"
_BUFFERS = (
    ("model", re.compile(rf"load_tensors:\s+(\S+) model buffer size =\s+{_SIZE}")),
    ("kv", re.compile(rf"llama_kv_cache\w*:\s+(\S+) KV buffer size =\s+{_SIZE}")),
    ("output", re.compile(rf"llama_context:\s+(\S+)\s+output buffer size =\s+{_SIZE}")),
    ("compute", re.compile(rf"sched_reserve:\s+(\S+) compute buffer size =\s+{_SIZE}")),
)
CPU_FEATURE_NAMES = (
    "SSE3", "SSSE3", "AVX", "AVX2", "F16C", "FMA", "BMI2", "AVX512", "AVX512_VBMI",
    "AVX512_VNNI", "AVX512_BF16", "AMX_INT8", "AVX_VNNI", "LLAMAFILE", "OPENMP", "REPACK",
    "NEON", "ARM_FMA", "FP16_VA", "DOTPROD", "MATMUL_INT8", "SVE", "SVE_CNT", "SME",
)  # fmt: skip
"""The names the ``system_info`` line prints for a CPU feature the build uses."""
_DEVICE_BUFFER = re.compile(r"(CUDA|HIP|MUSA|SYCL|Vulkan|MTL)\d+(_Mapped)?")
_OFFLOADED = re.compile(r"load_tensors: offloaded (\d+)/(\d+) layers to GPU")
_FIT_SUMMARY = re.compile(
    r"common_params_fit_impl:\s+- (\S+)(?: \([^)]*\))?: +(\d+) layers(?: \( *(\d+) overflowing\))?,"
)  # common/fit.cpp:726, 868 and 875: the last one is the placement fit applied
_DEVICE_FREE = re.compile(r"llama_prepare_model_devices: using device \S+ .*- (\d+) MiB free")
_N_CTX = re.compile(r"llama_context: n_ctx\s+= (\d+)")
_N_UBATCH = re.compile(r"llama_context: n_ubatch\s+= (\d+)")
_KV_UNIFIED = re.compile(r"llama_context: kv_unified\s+= (true|false)")
_FLASH_REQUEST = re.compile(r"llama_context: flash_attn\s+= (\w+)")
_FLASH_RESOLVED = re.compile(
    r"resolve_fused_ops: Flash Attention (enabled|not supported, set to disabled)"
)
_KV_CACHE = re.compile(
    r"llama_kv_cache: size =\s+[\d.]+ MiB \(\s*(\d+) cells,.*K \((\w+)\):.*V \((\w+)\):"
)
_SLOTS = re.compile(r"srv\s+load_model: initializing, n_slots = (\d+), n_ctx_slot = (\d+)")
_THREADS = re.compile(r"system_info: n_threads = (\d+) \(n_threads_batch = (\d+)\)")
_SYSTEM_INFO = re.compile(r"system_info: (.*?)\s*$")
_FIT_CONTEXT = re.compile(r"common_params_fit_impl: context size reduced from \d+ to \d+")
_FIT_LAYERS = re.compile(r"common_params_fit_impl: filling (dense|dense-only) layers back-to-front")
_FIT_OVERRIDES = re.compile(
    r"common_params_fit_impl: (filling dense-only layers|converting dense-only layers)"
)


def strip_prefix(line: str) -> str:
    """``line`` without llama-server's timestamp and level prefix, which ``--log-prefix`` and
    ``--log-timestamps`` change but the default verbosity prints."""
    return LINE_PREFIX.sub("", line, count=1)


@dataclass(frozen=True, slots=True, kw_only=True)
class Startup:
    allocations: tuple[Allocation, ...]
    offloaded_layers: int | None
    total_layers: int | None
    fit_overflow: int | None  # layers whose experts fit left on the CPU, the output layer excluded
    fit_changed: tuple[str, ...]
    device_free_bytes: int | None  # what the first device had free when the server started
    ctx_total: int | None
    kv_cells: int | None
    type_k: str | None
    type_v: str | None
    kv_unified: bool | None
    slots: int | None
    slot_ctx: int | None
    ubatch: int | None
    flash_attn: str | None
    threads: int | None
    threads_batch: int | None
    system_info: str | None

    def bytes_of(self, kind: str, domain: Literal["device", "host"]) -> int:
        return sum(a.bytes for a in self.allocations if a.kind == kind and a.domain == domain)

    @property
    def on_device(self) -> bool:
        """Whether any weights sit in a device buffer. Decided by the buffer lines: a CPU-only
        build prints ``offloaded N/N layers to GPU`` as well."""
        return self.bytes_of("model", "device") > 0

    @property
    def backend(self) -> str:
        """The backend holding the weights: the device buffers' name without its index
        (``CUDA0`` is ``cuda``), ``cpu`` when none holds any."""
        device = next(
            (a.device for a in self.allocations if a.kind == "model" and a.domain == "device"),
            None,
        )
        name = re.sub(r"\d+.*", "", device).lower() if device else "cpu"
        return {"mtl": "metal"}.get(name, name)

    @property
    def fit_expert_layers(self) -> int | None:
        """Layers whose experts ``--fit`` left on the CPU; None when it left none or the summary
        line is unreadable."""
        placed = "tensor_overrides" in self.fit_changed
        return self.fit_overflow if placed and self.fit_overflow else None

    @property
    def host_layers(self) -> int | None:
        if self.total_layers is None:
            return None
        if not self.on_device or self.offloaded_layers is None:
            return max(self.total_layers - 1, 0)
        return max(self.total_layers - self.offloaded_layers, 0)


def _allocation(kind: str, buffer: str, mib: str) -> Allocation:
    device = buffer
    if buffer.startswith("CPU"):
        device = "CPU"
    domain: Literal["device", "host"] = "device" if _DEVICE_BUFFER.fullmatch(device) else "host"
    return Allocation(domain=domain, device=device, kind=kind, bytes=int(float(mib) * MIB))


def _first(pattern: re.Pattern[str], lines: list[str]) -> re.Match[str] | None:
    return next((m for line in lines if (m := pattern.search(line))), None)


def _flash(lines: list[str]) -> str | None:
    if (resolved := _first(_FLASH_RESOLVED, lines)) is not None:
        return "on" if resolved[1] == "enabled" else "off (not supported here)"
    requested = _first(_FLASH_REQUEST, lines)
    return {
        "on": "on",
        "enabled": "on",
        "off": "off (requested)",
        "disabled": "off (requested)",
    }.get(requested[1] if requested else "")  # only "auto" is followed by the resolve line


def _fit_changed(lines: list[str]) -> tuple[str, ...]:
    found = []
    if _first(_FIT_CONTEXT, lines):
        found.append("ctx_size")
    if _first(_FIT_LAYERS, lines):
        found.append("n_gpu_layers")
    if _first(_FIT_OVERRIDES, lines):
        found.append("tensor_overrides")
    return tuple(found)


@lru_cache(maxsize=8)
def read_startup(log: str) -> Startup:
    lines = [strip_prefix(line) for line in log.splitlines()]
    allocations = tuple(
        _allocation(kind, m[1], m[2])
        for line in lines
        for kind, pattern in _BUFFERS
        if (m := pattern.search(line))
    )
    offloaded = _first(_OFFLOADED, lines)
    summary = [m for line in lines if (m := _FIT_SUMMARY.search(line))]
    summary = summary[-1:]
    overflow = None
    if summary:
        overflow = int(summary[0][3] or 0)
        if overflow:
            overflow -= 1  # the overflow is the last layers, the output layer among them (fit.cpp)
    free = _first(_DEVICE_FREE, lines)
    kv = _first(_KV_CACHE, lines)
    slots = _first(_SLOTS, lines)
    unified = _first(_KV_UNIFIED, lines)
    ubatch = _first(_N_UBATCH, lines)
    threads = _first(_THREADS, lines)
    system = _first(_SYSTEM_INFO, lines)
    context = _first(_N_CTX, lines)
    return Startup(
        allocations=allocations,
        offloaded_layers=int(offloaded[1]) if offloaded else None,
        total_layers=int(offloaded[2]) if offloaded else None,
        fit_overflow=overflow,
        fit_changed=_fit_changed(lines),
        device_free_bytes=int(free[1]) * MIB if free else None,
        ctx_total=int(context[1]) if context else None,
        kv_cells=int(kv[1]) if kv else None,
        type_k=kv[2] if kv else None,
        type_v=kv[3] if kv else None,
        kv_unified=unified[1] == "true" if unified else None,
        slots=int(slots[1]) if slots else None,
        slot_ctx=int(slots[2]) if slots else None,
        ubatch=int(ubatch[1]) if ubatch else None,
        flash_attn=_flash(lines),
        threads=int(threads[1]) if threads else None,
        threads_batch=int(threads[2]) if threads else None,
        system_info=system[1] if system else None,
    )
