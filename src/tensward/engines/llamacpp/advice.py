"""Report advice for a llama.cpp run that needs no knowledge record: what the build lacks for
this machine, how the machine is set up, which GGUF type and prompt order the workload has."""

from __future__ import annotations

import os
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
UNQUANTIZED = frozenset({"F32", "F16", "BF16"})
MIN_PROMPTS = 4
MIN_SHARED_CHARS = 200
MIN_SHARED_SHARE = 0.2


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
    return {
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


def _shared(prompts: Sequence[str]) -> tuple[int, int]:
    """(characters every prompt starts with, characters every prompt ends with)."""
    front = len(os.path.commonprefix(list(prompts)))
    back = len(os.path.commonprefix([p[::-1] for p in prompts]))
    return front, back


def advice(
    block: Mapping[str, Any], settings: Settings | None, facts: WorkloadFacts | None
) -> list[dict[str, str]]:
    """The advice lines that hold for this run, each with the key of its topic."""
    found: list[dict[str, str]] = []

    def add(key: str, text: str) -> None:
        found.append({"key": key, "text": text})

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
            )
        elif env["smt"] and compute == "cpu":
            add(
                "smt_threads",
                "this CPU has SMT: on Intel, -t at the logical CPU count beat -t at the physical "
                "core count (TPOT 0.57 against 0.76, relative to -t 4); on AMD Zen 4 it was "
                "neutral to 3 % worse; try both",
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
    quantization = facts.quantization if facts else None
    kind = (quantization or "").split(" ")[0].removeprefix("gguf:")
    if quantization and quantization.startswith("gguf:") and kind not in UNQUANTIZED | {"unknown"}:
        add(
            "quant_drift",
            f"this model is {kind}. Q4_K_M differs from BF16 by KL 0.077, Q6_K 0.010, Q8_0 0.003 "
            "on Gemma 4 E4B; Q8_0 ran 22 % slower (than Q4_K_M, at -np 4). KL measures drift in "
            "token probabilities, not whether your task suffers; compare answers on your prompts",
        )
        if compute == "cpu":
            add(
                "cpu_quant",
                "on CPU decode, Q4_0 ran fastest on AVX512; K-quants and IQ4_XS prefill faster; "
                "Q8_0 is the slow choice",
            )
    if settings is not None and settings.extra_args.get("--cache-reuse") not in (None, False, "0"):
        add(
            "cache_reuse",
            "--cache-reuse gave no useful gain (+0.4 %) on an agent workload and changed answers "
            "on 14 of 40 prompts, so Tensward does not offer it",
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
