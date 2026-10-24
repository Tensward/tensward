"""Which quantization methods and KV-cache dtypes the installed vLLM runs on this GPU, read
without loading a model. The tables are from vLLM 0.30.0's source; paths below are in its
repository."""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ...answer_options import FP8_KV_FLOOR, NATIVE_FP8, WEIGHT_ONLY, AnswerOptions, Format
from ...errors import CHECKPOINT_ACT_ORDER, PreflightError
from ...formats.hf_safetensors import Quantization, read_quantization
from ...probes import docker_image, executable, run_probe, script_interpreter
from .command import QUANTIZATION_FAMILIES

if TYPE_CHECKING:
    from ...runtime import Runtime

ACT_ORDER_REMOVED_IN = (0, 30)  # vLLM 0.30 dropped the act-order GPTQ kernels
SUPPORT_TIMEOUT_S = 120.0
PLUGINS = ("vllm-bnb-plugin", "vllm-gguf-plugin")
# From an fp16/bf16 checkpoint: fp8 online per tensor (quantization/fp8.py), bitsandbytes through
# the vllm-bnb-plugin (docs/features/quantization/bnb.md)
ON_THE_FLY_BITS = {"fp8": 8, "bitsandbytes": 4}
# vLLM 0.30.0 reports 7.5 as the fp8 floor, but below 10.0 it picks the CUTLASS fp8 kernel
# (scaled_mm/cutlass.py), which fails in the profile run on a T4 (cutlass_scaled_mm_sm75_epilogue)
# and on an A10G (cutlass_scaled_mm_sm80_epilogue), both measured. From 8.0 the Marlin kernel
# serves fp8 weight-only (measured on an A10G); on 7.5 it is unmeasured, so fp8 starts at 8.0
FP8_RUNS_FROM = 8.0
CHECKPOINTS = (  # (name, method, bits, floor): pre-quantized formats
    ("int8 W8A8", "compressed-tensors", 8, 7.5),  # schemes/compressed_tensors_w8a8_int8.py
    ("fp8", "fp8", 8, FP8_RUNS_FROM),  # fp8.py; Marlin weight-only below 8.9
    ("AWQ int4", "awq", 4, 7.5),  # auto_awq.py, marlin_utils.py
    ("GPTQ int4", "gptq", 4, 6.0),  # auto_gptq.py; Exllama kernel from 6.0, Marlin from 7.5
    ("W4A16", "compressed-tensors", 4, 7.5),  # schemes/compressed_tensors_wNa16.py
)
MOE_ONLY = frozenset({"experts_int8", "moe_wna16", "int8_per_channel_weight_only"})
NEWEST_GPUS_ONLY = frozenset({"nvfp4_per_token", "mxfp8"})  # Blackwell, or SM100+ for W8A8
# CLI shorthands of online fp8 (config/quantization.py)
ONLINE_FP8 = ("fp8_per_tensor", "fp8_per_block", "fp8_per_channel")
# One scale per output channel instead of one per tensor: 0.6-1 GiB larger than fp8 on a 4B model
# (measured with vLLM 0.30.0), and it fails to start below NATIVE_FP8
PER_CHANNEL = "fp8_per_channel"
JUDGED = frozenset(
    {*ON_THE_FLY_BITS, *ONLINE_FP8, *(method for _, method, _, _ in CHECKPOINTS), *MOE_ONLY,
     *NEWEST_GPUS_ONLY}
)  # fmt: skip
NOTHING_READABLE = "nothing readable: the engine's registry gave no method Tensward could judge"

SUPPORT_SNIPPET = r"""
import json
from importlib import metadata
from typing import get_args

out = {"plugins": {}}
for name in ("vllm-bnb-plugin", "vllm-gguf-plugin"):
    try:
        out["plugins"][name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        out["plugins"][name] = None
import vllm

out["version"] = getattr(vllm, "__version__", None)
if any(out["plugins"].values()):
    from vllm.plugins import load_general_plugins

    load_general_plugins()
from vllm.config.cache import CacheDType
from vllm.model_executor.layers import quantization as q

methods = {}
for name in list(q.QUANTIZATION_METHODS):
    try:
        methods[name] = q.get_quantization_config(name).get_min_capability()
    except Exception:
        methods[name] = None
out["methods"] = methods
out["builtin"] = list(get_args(q.QuantizationMethods))
out["kv_dtypes"] = list(get_args(CacheDType))
print(json.dumps(out))
"""
_PYTHON = re.compile(r"python(\d+(\.\d+)?)?$")


def support_command(runtime: Runtime) -> tuple[list[str], dict[str, str] | None] | str:
    """How to run the snippet where the engine is installed, with the GPUs the server would get;
    or why it cannot be run here. Docker never pulls and has no network."""
    if runtime.kind == "docker":
        image = docker_image(runtime.target)
        if image.labels is None:
            return {
                "docker_missing": "docker is not installed",
                "image_absent": f"the image {runtime.target} is not present locally",
            }.get(image.reason or "", "docker did not answer")
        argv = ["docker", "run", "--rm", "--pull=never", "--network", "none",
                "--name", f"tensward-support-{secrets.token_hex(4)}",
                *runtime.platform.docker_device_args(runtime.gpus),
                "--entrypoint", "python3", runtime.target, "-c", SUPPORT_SNIPPET]  # fmt: skip
        return argv, None
    found = executable(runtime.target)
    if found is None:
        return "the local server command was not found"
    interpreter = found if _PYTHON.match(Path(found).name) else script_interpreter(found)
    if interpreter is None or not _PYTHON.match(Path(interpreter).name):
        return "the local server command is not run by a Python interpreter"
    gpus = runtime.platform.select_env(runtime.gpus, os.environ)
    return [interpreter, "-c", SUPPORT_SNIPPET], {**os.environ, **gpus}


def read_support(runtime: Runtime) -> dict[str, Any] | str:
    """The snippet's JSON, or why it was not read."""
    command = support_command(runtime)
    if isinstance(command, str):
        return command
    argv, env = command
    with tempfile.TemporaryDirectory(prefix="tensward-support-") as empty:
        try:
            done = subprocess.run(  # noqa: S603 - list argv, no shell
                argv, cwd=empty, env=env, capture_output=True, text=True,
                stdin=subprocess.DEVNULL, timeout=SUPPORT_TIMEOUT_S, check=False,
            )  # fmt: skip
        except subprocess.TimeoutExpired:
            if argv[0] == "docker":
                run_probe(["docker", "rm", "-f", argv[argv.index("--name") + 1]])
            return f"the engine did not answer within {SUPPORT_TIMEOUT_S:.0f} s"
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            return f"the engine could not be asked: {error}"
    if done.returncode != 0:
        lines = done.stderr.strip().splitlines()
        return f"the engine's read failed: {lines[-1] if lines else f'exit {done.returncode}'}"
    lines = done.stdout.strip().splitlines()
    try:
        support = json.loads(lines[-1]) if lines else None
    except ValueError:
        support = None
    keys = (("methods", dict), ("kv_dtypes", list), ("plugins", dict))
    if not isinstance(support, dict) or not all(
        isinstance(support.get(key), kind) for key, kind in keys
    ):
        return "the engine's read gave no readable answer"
    return support


def _family(name: str) -> str:
    return "fp8" if name in ONLINE_FP8 else QUANTIZATION_FAMILIES.get(name, name)


def _fp8_mode(capability: float) -> str:
    return "W8A8" if capability >= NATIVE_FP8 else WEIGHT_ONLY


def options_from(support: Mapping[str, Any], capability: float | None) -> AnswerOptions | str:
    """What this GPU can run in place of the registered weights and KV cache, or why nothing in
    the read could be judged."""
    floors: dict[str, float | None] = {}
    for name, floor in support["methods"].items():
        read = floor / 10 if isinstance(floor, (int, float)) else None
        family = _family(name)
        known = [f for f in (floors.get(family), read) if f is not None]
        floors[family] = min(known) if known else None
        if name in ONLINE_FP8:
            floors[name] = read
    if not any(floors.get(name) is not None for name in JUDGED):
        return NOTHING_READABLE

    plugins = {name: support["plugins"].get(name) for name in PLUGINS}
    on_the_fly: dict[str, str] = {}
    checkpoints: list[Format] = []
    if capability is not None:

        def runs(method: str, row_floor: float = 0.0) -> bool:
            floor = floors.get(method)
            return floor is not None and capability >= max(row_floor, floor)

        if runs("fp8", FP8_RUNS_FROM):
            on_the_fly["fp8"] = _fp8_mode(capability)
        if runs(PER_CHANNEL, NATIVE_FP8):
            on_the_fly[PER_CHANNEL] = _fp8_mode(capability)
        if plugins["vllm-bnb-plugin"] is not None and runs("bitsandbytes"):
            on_the_fly["bitsandbytes"] = "W4A16"
        checkpoints = [
            Format(name, method, bits,
                   _fp8_mode(capability) if method == "fp8" else "W8A8" if bits == 8 else "W4A16")
            for name, method, bits, floor in CHECKPOINTS
            if runs(method, floor)
        ]  # fmt: skip
    kv = "fp8" in support["kv_dtypes"] and capability is not None and capability >= FP8_KV_FLOOR
    return AnswerOptions(
        engine_version=support.get("version"),
        capability=capability,
        on_the_fly=on_the_fly,
        kv_cache=("fp8",) if kv else (),
        checkpoints=tuple(sorted(checkpoints, key=lambda f: -f.bits)),
        unjudged=tuple(sorted(name for name in floors if name not in JUDGED)),
        plugins=plugins,
    )


def act_order_refusal(quantization: Quantization | None, engine_version: str | None) -> str | None:
    """Why vLLM cannot serve a GPTQ checkpoint quantized with act-order, or None."""
    if quantization is None or quantization.method != "gptq" or not quantization.act_order:
        return None
    found = re.match(r"(\d+)\.(\d+)", engine_version or "")
    if found is None or (int(found[1]), int(found[2])) < ACT_ORDER_REMOVED_IN:
        return None
    return "GPTQ checkpoints with act-order (desc_act) are not supported by vLLM 0.30 and later"


def require_servable_gptq(model_dir: Path, engine_version: str | None) -> None:
    """Refuse, before any launch, a checkpoint the installed vLLM cannot serve."""
    if reason := act_order_refusal(read_quantization(model_dir), engine_version):
        raise PreflightError(CHECKPOINT_ACT_ORDER, reason)
