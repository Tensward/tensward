"""What the engine and the GPU can run in place of the registered setup's weights and KV cache."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .files import cache_dir, write_json
from .probes import executable, run_probe
from .progress import Note, emit

if TYPE_CHECKING:
    from .engines import Engine
    from .runtime import Runtime

# vLLM 0.30.0 runs an fp8 KV cache from here on any model: v1/attention/backends/triton_attn.py
# needs SM89, fa_utils.py takes it only for FA3 (SM90) or FA4. Below it FlashInfer starts one for
# 128-wide attention heads, but on an A10G it broke the answers of the model measured
# (Qwen2.5-7B), so Tensward does not offer it there
FP8_KV_FLOOR = 8.9
FP8_KV_RULE = f"offered only from compute capability {FP8_KV_FLOOR:g}"
# fp8 weights run W8A8 from here (CUTLASS fp8 needs SM89, cutlass/scaled_mm_entry.cu)
NATIVE_FP8 = 8.9
WEIGHT_ONLY = "weight-only (W8A16)"
# Selects Marlin for fp8 weights below NATIVE_FP8 (why: FP8_RUNS_FROM, engines/vllm/quantization.py)
MARLIN_FP8_ARG = "linear-backend=marlin"
SUPPORT_CACHE = "engine-support"
VERSION_SNIPPET = (
    "import importlib.metadata as m, json\n"
    "def v(n):\n"
    "    try: return m.version(n)\n"
    "    except m.PackageNotFoundError: return None\n"
    "print(json.dumps({'vllm': v('vllm'), 'plugins': {n: v(n) for n in %r}}))"
)


@dataclass(frozen=True, slots=True)
class Format:
    name: str  # as people say it: "int8 W8A8", "AWQ int4"
    method: str  # the engine's --quantization value for it
    bits: int  # 8 or 4
    mode: str  # how this GPU runs it: "W8A8", "weight-only (W8A16)", "W4A16"


@dataclass(frozen=True, slots=True)
class AnswerOptions:
    engine_version: str | None
    capability: float | None  # the GPU's compute capability, e.g. 8.6
    on_the_fly: Mapping[str, str]  # method -> mode, from the registered checkpoint
    kv_cache: tuple[str, ...]  # 8-bit KV-cache dtypes this GPU runs: ("fp8",) or ()
    checkpoints: tuple[Format, ...]  # pre-quantized formats it runs, 8-bit first
    unjudged: tuple[str, ...]  # registered methods (aliases folded) Tensward does not judge
    plugins: Mapping[str, str | None]  # distribution -> version, None when absent

    @property
    def fast_prompts(self) -> tuple[str, ...]:
        """The pre-quantized formats this GPU runs in 8-bit arithmetic, which speeds prompt
        processing."""
        return tuple(f.name for f in self.checkpoints if f.mode == "W8A8")


def mode_words(mode: str) -> str:
    """A mode as running text: "8-bit weights, 16-bit maths" for weight-only fp8."""
    return "8-bit weights, 16-bit maths" if mode == WEIGHT_ONLY else mode


def support_key(runtime: Runtime) -> str | None:
    """What the engine's quantization support depends on, or None when any part is unknown:
    docker, the image id; local, the command's path and mtime, the engine version and the vLLM
    plugin distributions with their versions; in both, the selected GPUs."""
    from .engines.vllm.quantization import PLUGINS, support_command

    if runtime.kind == "docker":
        inspect = ["docker", "image", "inspect", "--format", "{{.Id}}", runtime.target]
        parts = [(run_probe(inspect) or "").strip()]
    else:
        command = support_command(runtime)
        path = executable(runtime.target)
        if isinstance(command, str) or path is None:
            return None
        try:
            stamp = os.stat(path).st_mtime_ns
            probe = run_probe([command[0][0], "-c", VERSION_SNIPPET % (PLUGINS,)])
            versions = json.loads(probe or "")
        except (OSError, ValueError):
            return None
        if not isinstance(versions, dict) or not versions.get("vllm"):
            return None
        parts = [os.path.realpath(path), str(stamp), json.dumps(versions, sort_keys=True)]
    if not all(parts):
        return None
    return "\n".join([runtime.kind, *parts, ",".join(runtime.gpus) if runtime.gpus else "all"])


def _cached_support(runtime: Runtime) -> dict[str, Any] | str:
    from .engines.vllm.quantization import read_support

    key = support_key(runtime)
    if key is None:
        return read_support(runtime)
    name = hashlib.sha256(key.encode()).hexdigest()[:16]
    path = cache_dir() / SUPPORT_CACHE / f"{name}.json"
    try:
        cached = json.loads(path.read_text("utf-8"))
        if isinstance(cached, dict):
            return cached
    except (OSError, ValueError):
        pass
    support = read_support(runtime)
    if isinstance(support, dict):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, support)
        except OSError:
            pass
    return support


def read_answer_options(
    engine: Engine, runtime: Runtime
) -> tuple[AnswerOptions | None, str | None, float | None]:
    """(the options, why they are unknown, the GPU's compute capability)."""
    if engine.name != "vllm":
        return None, f"{engine.label} has no quantization reader", None
    from .engines.vllm.quantization import options_from
    from .platforms.nvidia import compute_capability

    emit(Note(text="reading which quantization methods the engine supports"))
    capability = None
    try:
        capability = compute_capability(runtime.gpus)
        support = _cached_support(runtime)
        if isinstance(support, str):
            return None, support, capability
        options = options_from(support, capability)
    except Exception as error:  # noqa: BLE001 - a failed read never stops a run
        return None, f"the read failed: {error}", capability
    return (None, options, capability) if isinstance(options, str) else (options, None, capability)


def record_answer_options(
    run_dir: Path, options: AnswerOptions | None, why: str | None, capability: float | None
) -> None:
    """Record the options in the run directory."""
    from .runs import RUN_RECORD

    path = run_dir / RUN_RECORD
    recorded = (
        {"read": True, **asdict(options)}
        if options
        else {"read": False, "why": why, "capability": capability}
    )
    write_json(path, {**json.loads(path.read_text("utf-8")), "answer_options": recorded})
