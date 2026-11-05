"""llama-server's reading of its own log and scrapes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from ...platforms import PLATFORMS, darwin
from ...platforms.apple import chip_of
from ...probes import executable
from ...settings import Settings
from . import start_phases, startup
from .advice import system_info_parts
from .command import probe_build
from .flags import ADMITTED
from .startup import CPU_FEATURE_NAMES, MIB

if TYPE_CHECKING:
    from ...context import AnalysisContext
    from ...playbook import Situation
    from ...runtime import Runtime

GIB = 2**30
TOPOLOGY = {platform.name: platform.memory_topology for platform in PLATFORMS}
LOAD_MODE_FLAG = "--load-mode"


_BUILD = re.compile(r"\bbuild ([1-9]\d*) \(")


def _text(path: Path) -> str | None:
    try:
        return path.read_text("utf-8", errors="replace")
    except OSError:
        return None


def _log_facts(log: str) -> dict[str, object]:
    started = startup.read_startup(log)
    build = _BUILD.search(log)
    compute = started.bytes_of("compute", "device") or started.bytes_of("compute", "host")
    features, _ = system_info_parts(started.system_info)
    return {
        "engine.build": int(build[1]) if build else None,
        "engine.backend": started.backend,
        "engine.cpu_features": features & set(CPU_FEATURE_NAMES) if started.system_info else None,
        "run.slots": started.slots,
        "run.slot_ctx": started.slot_ctx,
        "run.kv_unified": started.kv_unified,
        "run.kv_type_k": started.type_k,
        "run.kv_type_v": started.type_v,
        "run.flash_attn": None if started.flash_attn is None else started.flash_attn[:2] == "on",
        "run.fit_changed": started.fit_changed,
        "run.threads": started.threads,
        "run.threads_batch": started.threads_batch,
        "run.compute_buffer_mib": compute / MIB if compute else None,
        "run.graph_splits": started.graph_splits,
    }


def _metal_facts(log: str | None, identity: Mapping[str, Any] | None) -> dict[str, object]:
    """The Metal device and its buffers, for a run that used the Mac's GPU: a CPU-only run
    prints the device lines as well."""
    started = startup.read_startup(log) if log is not None else None
    chip = chip_of(str(identity["name"]), identity.get("gpu_cores")) if identity else None
    found: dict[str, object] = {
        "metal.chip": chip.chip if chip else None,
        "metal.gpu_cores": chip.gpu_cores if chip else None,
    }
    if started is not None:
        working_set = started.working_set_bytes
        metal = sum(a.bytes for a in started.allocations if a.device.startswith("MTL"))
        found |= {
            "metal.gpu_family": started.gpu_family,
            "metal.has_tensor": started.has_tensor,
            "metal.residency_sets": started.residency_sets,
            "metal.working_set_gib": working_set / GIB if working_set else None,
            "run.metal_bytes": metal or None,
        }
    return found


def _window_facts(window: Mapping[str, Any] | None) -> dict[str, object]:
    """How far swap and compressed memory grew over the measured window, and the pressure
    level at its end."""
    first, last = (window or {}).get("start"), (window or {}).get("end")
    if not first or not last:
        return {}

    def grew(name: str) -> float | None:
        before, after = first.get(name), last.get(name)
        return None if before is None or after is None else after - before

    return {
        "run.swap_out_delta_mib": grew("swap_used_mib"),
        "run.compressed_delta_mib": grew("compressed_mib"),
        "run.pressure_level": last.get("pressure_level"),
    }


def _placement_facts(block: Mapping[str, Any] | None) -> dict[str, object]:
    """The run's placement, regime and host from the placement block of its report."""
    if not block or "unplaced" in block:
        return {}
    placed, shape, host = block["placement"], block["regime"], block.get("environment") or {}
    faults = block.get("major_faults")
    platform = host.get("platform")
    return {
        "run.placement": shape["placement"],
        "run.compute": shape["compute"],
        "run.host_memory_tight": shape["host_memory_tight"],
        "run.numa_split": shape["numa_split"],
        "run.host_layers": placed["host_layers"],
        "run.host_expert_layers": placed["host_expert_layers"],
        "run.host_weight_bytes": placed["host_weight_bytes"],
        "run.device_weight_bytes": placed["device_weight_bytes"],
        "run.host_bytes_per_token": placed["host_bytes_per_token"],
        "run.device_bytes_per_token": placed["device_bytes_per_token"],
        "run.major_faults": faults["end"] - faults["start"] if faults else None,
        "platform.name": platform,
        "memory.topology": TOPOLOGY.get(platform or ""),
        "os.family": host.get("os_family", "linux"),
        "host.p_cores": host.get("p_cores"),
        "host.e_cores": host.get("e_cores"),
        "host.low_power_mode": host.get("low_power_mode"),
        "host.thermal_limited": host.get("thermal_limited"),
        "metal.wired_limit_mb": host.get("wired_limit_mb") if platform == "apple" else None,
        "host.physical_cores": host.get("physical_cores"),
        "host.logical_cpus": host.get("logical_cpus"),
        "host.usable_cpus": host.get("usable_cpus"),
        "host.cpu_quota_cores": host.get("quota_cores"),
        "host.smt": host.get("smt"),
        "host.mem_gib": host.get("mem_gib"),
        "host.mem_available_gib": host.get("mem_available_gib"),
        "host.cgroup_mem_gib": host.get("cgroup_mem_gib"),
        "host.memlock_gib": host.get("memlock_gib"),
        "host.isa": host.get("isa"),
        "host.numa_nodes": host.get("numa_nodes"),
        **_window_facts(block.get("host_window")),
    }


def _probe_facts(probes: Mapping[str, Any] | None) -> dict[str, object]:
    """The host bandwidth a launch measured and the PCIe link, from the run's probes block. A
    bandwidth read passively off the run's own decode step is not a host fact."""
    if not probes:
        return {}
    host, pcie = probes.get("host") or {}, probes.get("pcie") or {}
    return {
        "host.mem_bw_gbs": host.get("mem_bw_gbs") if host.get("method") == "launch" else None,
        "run.pcie_gbs_effective": pcie.get("gbs_effective"),
        "gpu.pcie_gen": pcie.get("gen"),
        "gpu.pcie_width": pcie.get("width"),
        "gpu.pcie_gbs_theoretical": pcie.get("theoretical_gbs"),
    }


class LlamaCppLog:
    start_classes: tuple[str, ...] = start_phases.CLASSES

    def start_class(self, log: str, caches: Any, settings: Settings) -> str | None:
        return start_phases.start_class(log)

    def start_class_for(self, flags: frozenset[str], quantization: str | None) -> str:
        return "cold_dio" if LOAD_MODE_FLAG in flags else "warm"

    def compile_neutral(self, flags: frozenset[str]) -> bool:
        return True

    def start_cost(self, flags: frozenset[str], quantization: str | None) -> int:
        return start_phases.CLASSES.index(self.start_class_for(flags, quantization))

    def window_facts(
        self,
        log: str,
        compile_cache: str | None,
        caches: Any = None,
        quantization: str | None = None,
    ) -> dict[str, object]:
        loaded = start_phases.load(log)
        started = startup.read_startup(log)
        return {
            "start_class": start_phases.start_class(log),
            "load_s": loaded.seconds if loaded else None,
            "load_mb_s": loaded.mb_per_s if loaded else None,
            "kv_pool_tokens": started.ctx_total,
            "fit_changed": list(started.fit_changed),
        }

    def kv_pool(self, log: str) -> tuple[int | None, float | None]:
        started = startup.read_startup(log)
        kv = sum(a.bytes for a in started.allocations if a.kind == "kv")
        return started.ctx_total, kv / GIB if kv else None

    def slot_tokens(self, log: str) -> int | None:
        return startup.read_startup(log).slot_ctx

    def load_lines(self, log: str) -> None:
        return None

    def load_queue(self, log: str, skip: int = 0) -> None:
        return None

    def no_kv_memory(self, log: str) -> bool:
        return False

    def step_budget(self, log: str) -> int | None:
        return None

    def memory_usage(self, log: str) -> dict[str, float] | None:
        return None

    def run_facts(
        self,
        run_dir: Path | None,
        context: AnalysisContext,
        situation: Situation,
        span_s: float | None,
    ) -> dict[str, object]:
        found: dict[str, object] = {}
        log = _text(run_dir / "server.log") if run_dir is not None else None
        if log is not None:
            found.update(_log_facts(log))
        found.update(_placement_facts(context.placement))
        if ((context.placement or {}).get("environment") or {}).get("platform") == "apple":
            found.update(_metal_facts(log, context.gpu_identity))
        built = context.window_extra.get("llamacpp") or {}
        found["engine.binary_arch"] = built.get("binary_arch")
        found.update(_probe_facts(context.probes))
        signals = situation.measurement.engine_signals
        found["run.thinking_tokens_p50"] = signals.get("llamacpp.thinking_tokens_p50")
        return found

    def cache_observer(
        self, context: AnalysisContext, runtime: Runtime, env: Mapping[str, str]
    ) -> Callable[[Any], None] | None:
        """Records the build, the levers its ``--help`` lacks and, on a Mac, the binary's
        architecture in ``context.window_extra``."""
        if runtime.kind != "local":
            return None
        recorded: dict[str, Any] = {}
        if (found := probe_build(runtime.target)) is not None:
            build, manifest = found
            recorded = {
                "build": build.build,
                "commit": build.commit,
                "flags_missing_from_help": manifest.missing(sorted(ADMITTED)),
            }
        program = executable(runtime.target)
        if program is not None and (arch := darwin.mach_o_arch(Path(program))) is not None:
            recorded["binary_arch"] = arch
        if recorded:
            context.window_extra["llamacpp"] = recorded
        return None


LLAMACPP_LOG = LlamaCppLog()
