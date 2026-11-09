"""Report advice for a llama.cpp run that needs no knowledge record: what the build lacks for
this machine, how the machine is set up, which GGUF type and prompt order the workload has."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...platforms.cpu import HostFacts
from ...platforms.nvidia import compute_capability
from ...playbook import WorkloadFacts
from ...settings import Settings

GOVERNOR_FILE = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
# /proc/cpuinfo flag -> the name system_info prints, most useful first
CPU_FEATURES = {
    "avx512f": "AVX512",
    "avx512_vnni": "AVX512_VNNI",
    "avx512_bf16": "AVX512_BF16",
    "amx_int8": "AMX_INT8",
    "avx_vnni": "AVX_VNNI",
    "avx2": "AVX2",
}
METAL_DEBUG_ENV = (
    "GGML_METAL_NO_RESIDENCY",
    "GGML_METAL_SHARED_BUFFERS_DISABLE",
    "GGML_METAL_FUSION_DISABLE",
    "GGML_METAL_FUSION_DEBUG",
    "GGML_METAL_CONCURRENCY_DISABLE",
    "GGML_METAL_GRAPH_OPTIMIZE_DISABLE",
    "GGML_METAL_CAPTURE_COMPUTE",
    "GGML_METAL_GRAPH_DEBUG",
)
MS_PER_S = 1000
CACHE_UPDATE = re.compile(r"prompt cache update took ([\d.]+) ms")
AFFINITY_FLAGS = (
    "-C", "-Cr", "-Cb", "-Crb", "--cpu-mask", "--cpu-range", "--cpu-strict",
    "--cpu-mask-batch", "--cpu-range-batch", "--cpu-strict-batch",
)  # fmt: skip
UNQUANTIZED = frozenset({"F32", "F16", "BF16"})
MIN_PROMPTS = 4
MIN_SHARED_CHARS = 200
MIN_SHARED_SHARE = 0.2
DEFAULT_CACHE_RAM_MIB = 8192
MIB = 2**20
CACHE_RAM_STEP_MIB = 1024


def system_info_parts(info: str | None) -> tuple[frozenset[str], tuple[int, ...]]:
    """The CPU features the build uses (``AVX2``, ``AVX512``...) and its CUDA architectures
    (``890`` for compute capability 8.9) from the ``system_info`` log line."""
    features: set[str] = set()
    archs: tuple[int, ...] = ()
    section = ""
    for part in (info or "").split("|"):
        text = part.strip()
        if " : " in text:
            section, _, text = text.partition(" : ")
        name, _, value = text.partition(" = ")
        if section == "CPU" and value.strip() == "1":
            features.add(name.strip())
        elif section == "CUDA" and name.strip() == "ARCHS":
            archs = tuple(int(a) for a in value.split(",") if a.strip().isdigit())
    return frozenset(features), archs


def governor() -> str | None:
    try:
        return GOVERNOR_FILE.read_text("utf-8").strip() or None
    except OSError:
        return None


def environment(host: HostFacts, system_info: str | None, platform: str) -> dict[str, Any]:
    """What the build and the machine offer each other, for the report."""
    features, archs = system_info_parts(system_info)
    capability = compute_capability(None) if platform == "nvidia" else None
    found: dict[str, Any] = {
        "platform": platform or "cpu",
        "isa": sorted(host.isa),
        "cpu_features": sorted(features),
        "cuda_archs": list(archs),
        "gpu_cc": capability,
        "physical_cores": host.physical_cores,
        "logical_cpus": host.logical_cpus,
        "usable_cpus": host.usable_cpus,
        "quota_cores": host.quota_cores,
        "smt": host.smt,
        "mem_gib": host.mem_gib,
        "mem_available_gib": host.mem_available_gib,
        "cgroup_mem_gib": host.cgroup_mem_gib,
        "memlock_gib": host.memlock_gib,
        "numa_nodes": host.numa_nodes,
        "governor": governor(),
    }
    if host.os_family == "macos":
        found |= {
            "os_family": host.os_family,
            "p_cores": host.p_cores,
            "e_cores": host.e_cores,
            "low_power_mode": host.low_power_mode,
            "thermal_limited": host.thermal_limited,
            "wired_limit_mb": host.wired_limit_mb,
        }
    return found


def _shared(prompts: Sequence[str]) -> tuple[int, int]:
    """(characters every prompt starts with, characters every prompt ends with)."""
    front = len(os.path.commonprefix(list(prompts)))
    back = len(os.path.commonprefix([p[::-1] for p in prompts]))
    return front, back


def _mac_advice(env: Mapping[str, Any], settings: Settings | None) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if env.get("low_power_mode") or env.get("thermal_limited"):
        causes = [
            cause
            for cause, held in (
                ("Low Power Mode was on", env.get("low_power_mode")),
                ("the CPU speed was held below its limit", env.get("thermal_limited")),
            )
            if held
        ]
        found.append((
            "mac_power",
            f"{' and '.join(causes)} during this run, so its timings may be lower than this "
            "Mac delivers at full power; draw no conclusion from it against a run at full power",
        ))  # fmt: skip
    given = set(settings.extra_env) if settings is not None else set()
    names = sorted(name for name in METAL_DEBUG_ENV if name in given or name in os.environ)
    if names:
        found.append((
            "metal_debug_env",
            f"{', '.join(names)} is set: it turns off a Metal optimisation or adds debug work "
            "and is a likely cause of a slow run; unset it unless you are debugging",
        ))  # fmt: skip
    flags = sorted(f for f in AFFINITY_FLAGS if settings is not None and f in settings.extra_args)
    if flags:
        found.append((
            "mac_affinity",
            f"{', '.join(flags)} set CPU affinity, which has no effect on macOS, so they change "
            "nothing in this run",
        ))  # fmt: skip
    return found


def _host_cache(block: Mapping[str, Any], settings: Settings | None, log: str) -> str | None:
    """The note on llama-server's host prompt cache on a Mac that ran short of memory, when the
    launch keeps more of it than the system memory left after the run's other needs."""
    fit = block.get("fit")
    if block.get("topology") != "unified" or not fit or not block["regime"]["host_memory_tight"]:
        return None
    given = settings.extra_args.get("--cache-ram") if settings is not None else None
    try:
        kept = DEFAULT_CACHE_RAM_MIB if given is None else int(str(given))
    except ValueError:
        return None
    window = block.get("host_window") or {}
    start, end = window.get("start") or {}, window.get("end") or {}
    swapped = (end.get("swap_used_mib") or 0) - (start.get("swap_used_mib") or 0) > 0
    room = max((fit["host_free"] - fit["device_need"] - fit["host_need"]) // MIB, 0)
    target = room // CACHE_RAM_STEP_MIB * CACHE_RAM_STEP_MIB
    if 0 <= kept <= target:
        return None
    held = "unlimited" if kept < 0 else f"{kept} MiB"
    note = (
        f"host memory was tight in this run, and llama-server's host prompt cache "
        f"(--cache-ram, {held} here) shares the unified memory with the weights and is the "
        f"likely cause of the pressure; lowering the cache frees memory for other processes at "
        f"the cost of prompt reuse (in an earlier measurement on a 24 GiB M4 a cache hit cut a "
        f"4k-token prompt's first-token time from about 19 s to 0.09 s, and the pressure cost "
        f"no throughput there)"
    )
    updates = [float(ms) for ms in CACHE_UPDATE.findall(log)]
    if sum(updates) >= MS_PER_S:
        note += (
            f"; in this run, prompt-cache updates held the server for "
            f"{sum(updates) / MS_PER_S:.1f} s (longest {max(updates) / MS_PER_S:.1f} s)"
        )
    if swapped and target:
        note += (
            f"; if swap keeps growing, a cap near the room left, --cache-ram {target}, is an "
            "option that was not measured as a setting"
        )
    return note


def advice(
    block: Mapping[str, Any],
    settings: Settings | None,
    facts: WorkloadFacts | None,
    log: str = "",
) -> list[dict[str, str]]:
    """The advice lines that hold for this run, each with the key of its topic."""
    found: list[dict[str, str]] = []

    def add(key: str, text: str, terminal: str | None = None) -> None:
        found.append({"key": key, "text": text, **({"terminal": terminal} if terminal else {})})

    env = block.get("environment") or {}
    compute = (block.get("regime") or {}).get("compute")
    on_cpu = compute in ("cpu", "mixed")
    if on_cpu and env:
        built = set(env["cpu_features"])
        lacking = [
            name for flag, name in CPU_FEATURES.items() if flag in env["isa"] and name not in built
        ]
        if env["cpu_features"] and lacking:
            add(
                "rebuild_isa",
                f"rebuild your llama.cpp with {', '.join(lacking)}: this CPU has "
                f"{'it' if len(lacking) == 1 else 'them'} and the build does not use "
                f"{'it' if len(lacking) == 1 else 'them'} (-DGGML_NATIVE=ON builds for this CPU)",
            )
        quota = env["quota_cores"]
        if quota is not None and quota < env["logical_cpus"]:
            add(
                "cpu_quota",
                f"the CPU quota allows {quota:g} of the {env['logical_cpus']} logical CPUs, and "
                "llama.cpp's default thread count ignores it; set -t to the CPUs the quota allows "
                "(automatic -t ran 25.6 % worse in TPOT than -t 4 under a 4-core quota)",
                f"the CPU quota allows {quota:g} of the {env['logical_cpus']} logical CPUs; set "
                "-t to that number, as llama.cpp's default ignores the quota",
            )
        elif env["smt"] and compute == "cpu":
            add(
                "smt_threads",
                "this CPU has SMT: on Intel, -t at the logical CPU count beat -t at the physical "
                "core count (TPOT 0.57 against 0.76, relative to -t 4); on AMD Zen 4 it was "
                "neutral to 3 % worse; try both",
                "this CPU has SMT: -t at the logical CPU count can be faster than at the "
                "physical core count; try both",
            )
        if env["governor"] not in (None, "performance"):
            add(
                "cpu_governor",
                f"the CPU frequency governor is {env['governor']}; 'performance' keeps the clock "
                "steady for decode, and Tensward does not change system settings",
            )
    capability, archs = env.get("gpu_cc"), env.get("cuda_archs") or []
    if (
        capability
        and archs
        and not any(
            a <= round(capability * 100) and a // 100 == round(capability * 100) // 100
            for a in archs
        )
    ):
        add(
            "rebuild_cuda_arch",
            f"this build's CUDA architectures ({','.join(map(str, archs))}) do not include this "
            f"GPU (compute capability {capability:g}); rebuild your llama.cpp with "
            f"-DCMAKE_CUDA_ARCHITECTURES={round(capability * 10)}",
        )
    if env.get("os_family") == "macos":
        for key, text in _mac_advice(env, settings):
            add(key, text)
    if (cached := _host_cache(block, settings, log)) is not None:
        add(
            "host_prompt_cache",
            cached,
            "host memory was tight, and llama-server's host prompt cache (--cache-ram) is the "
            "likely cause; a smaller cache frees memory but reuses fewer prompts",
        )
    quantization = facts.quantization if facts else None
    kind = (quantization or "").split(" ")[0].removeprefix("gguf:")
    if quantization and quantization.startswith("gguf:") and kind not in UNQUANTIZED | {"unknown"}:
        add(
            "quant_drift",
            f"this model is {kind}. Measured on Gemma 4 E4B, not on your model: Q4_K_M drifts "
            "from BF16 by KL 0.077, Q6_K by 0.010, Q8_0 by 0.003, and Q8_0 ran 22 % slower than "
            "Q4_K_M with 4 slots. KL measures how far the token probabilities move, not whether "
            "your task suffers; compare answers on your own prompts before switching",
            f"this model is {kind}: a higher-bit file (Q6_K, Q8_0) keeps answers closer to the "
            "full model but runs slower",
        )
        if compute == "cpu":
            add(
                "cpu_quant",
                "in Tensward's CPU decode tests, Q4_0 ran fastest on AVX512; K-quants and "
                "IQ4_XS prefill faster; Q8_0 is the slow choice",
                "on a CPU, Q4_0 files decode fastest and Q8_0 slowest",
            )
    if settings is not None and settings.extra_args.get("--cache-reuse") not in (None, False, "0"):
        add(
            "cache_reuse",
            "in Tensward's tests on an agent workload, --cache-reuse gave no useful gain "
            "(+0.4 %) and changed answers on 14 of 40 prompts, so Tensward does not offer it",
            "--cache-reuse may reword answers for little or no speed; Tensward does not offer it",
        )
    prompts = facts.prompts if facts else ()
    if len(prompts) >= MIN_PROMPTS:
        front, back = _shared(prompts)
        mean = sum(map(len, prompts)) / len(prompts)
        if back >= MIN_SHARED_CHARS and back >= MIN_SHARED_SHARE * mean and front < back / 4:
            add(
                "prompt_tail",
                f"the prompts share {back} characters at their end but start differently; the "
                "prompt cache reuses only what comes before the first difference, so put the "
                "stable text (system prompt, tool list, documents) first and the changing text "
                "last",
            )
    return found
