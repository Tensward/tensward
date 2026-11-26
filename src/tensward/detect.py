"""What this machine has before ``init`` asks anything: the engines installed, the hardware,
the servers already running, and whether an engine and the hardware go together."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .engines import ENGINES
from .engines.llamacpp.command import list_devices, probe_build
from .engines.llamacpp.flags import PIN_BUILD
from .engines.vllm.command import _split_docker_run
from .environment import describe_host, environment_report
from .formats.gguf import _file_type, model_files, read_gguf
from .platforms import Device, detect_platform
from .secrets import shown_command

GIB = 2**30
LLAMACPP_RELEASES = "https://github.com/ggml-org/llama.cpp/releases"
LLAMACPP_BACKENDS = "https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md"
LISTED_GPU = re.compile(r"^\s+(MTL|CUDA|HIP|Vulkan|SYCL)\d+:", re.MULTILINE)
OLLAMA_BLOB = "/blobs/sha256-"
QUERY_FLAGS = frozenset({"-h", "--help", "--version", "--list-devices"})
WEIGHT_MEMORY_SHARE = 0.9  # of device memory the weights may take before the line says so
DTYPES = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}


@dataclass(frozen=True, slots=True)
class EngineFound:
    name: str
    version: str | None
    command: str
    backends: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Hardware:
    kind: Literal["nvidia", "apple", "cpu"]
    devices: tuple[tuple[str, int], ...]
    driver: str | None
    cuda: bool
    ram_bytes: int | None
    physical_cores: int | None
    logical_cores: int | None

    def line(self) -> str:
        """``1× NVIDIA L4 22.5 GiB``, ``Apple M4 Pro 48 GiB`` or ``CPU only, 32 cores, 128 GiB``."""
        if self.kind == "cpu" or not self.devices:
            cores = self.logical_cores or self.physical_cores
            parts = ["CPU only"]
            if cores:
                parts.append(f"{cores} cores")
            if self.ram_bytes:
                parts.append(f"{_gib(self.ram_bytes)} GiB")
            return ", ".join(parts)
        name, size = self.devices[0]
        same = all(device == self.devices[0] for device in self.devices)
        if self.kind == "apple":
            return f"{name} {_gib(size)} GiB"
        count = f"{len(self.devices)}× " if same else ""
        return f"{count}{name} {_gib(size)} GiB"


@dataclass(frozen=True, slots=True)
class RunningServer:
    pid: int
    engine: str
    command: str


@dataclass(frozen=True, slots=True)
class Machine:
    engines: tuple[EngineFound, ...]
    hardware: Hardware
    running: tuple[RunningServer, ...]


@dataclass(frozen=True, slots=True)
class Problem:
    stop: bool
    text: str


def _gib(size: int) -> str:
    return f"{size / GIB:.1f}".removesuffix(".0")


def detect_hardware() -> Hardware:
    host = describe_host()
    found = detect_platform()
    kind: Literal["nvidia", "apple", "cpu"] = "cpu"
    devices: tuple[Device, ...] = ()
    if found is not None:
        platform, detected = found
        kind = "nvidia" if platform.name == "nvidia" else "apple"
        devices = detected
    return Hardware(
        kind=kind,
        devices=tuple((d.name, d.total_bytes) for d in devices),
        driver=next((d.driver for d in devices if d.driver), None),
        cuda=kind == "nvidia",
        ram_bytes=host.ram_bytes,
        physical_cores=host.physical_cores,
        logical_cores=host.logical_cores,
    )


def _backends(command: str, declared: tuple[str, ...]) -> tuple[str, ...]:
    """The backends a llama-server uses: the ones its version names, else ``CPU`` when its
    device list names no accelerator."""
    if declared:
        return declared
    listed = list_devices(command)
    if listed is not None and not LISTED_GPU.search(listed):
        return ("CPU",)
    return ()


def detect_engines(
    engines: tuple[str, ...] = tuple(ENGINES), local_command: str | None = None
) -> tuple[EngineFound, ...]:
    report = environment_report(
        engines=engines, runtimes=("local",), image=None, local_command=local_command
    )
    found = []
    for row in report["engines"]:
        local = row["local"]
        if not local["available"]:
            continue
        command = local["target"]
        backends: tuple[str, ...] = ()
        if row["name"] == "llamacpp":
            built = probe_build(command)
            backends = _backends(command, built[0].backends if built else ())
        found.append(EngineFound(row["name"], local["version"], command, backends))
    return tuple(found)


def _processes() -> list[tuple[int, int, str]]:
    """Every process as (pid, parent pid, command line)."""
    proc = Path("/proc")
    rows = []
    if proc.is_dir() and (proc / "self/stat").exists():
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "cmdline").read_bytes()
                stat = (entry / "stat").read_text()
            except OSError:
                continue
            command = shlex.join(raw.rstrip(b"\0").decode(errors="replace").split("\0")).strip()
            parent = stat.rsplit(")", 1)[-1].split()[1]
            if command and parent.isdigit():
                rows.append((int(entry.name), int(parent), command))
        return rows
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    for line in out.splitlines():
        fields = line.split(None, 2)
        if len(fields) == 3 and fields[0].isdigit() and fields[1].isdigit():
            rows.append((int(fields[0]), int(fields[1]), fields[2]))
    return rows


def _program(command: str) -> str:
    try:
        return os.path.basename(shlex.split(command)[0])
    except (ValueError, IndexError):
        return ""


def ollama_processes() -> tuple[tuple[int, str], ...]:
    """The ``ollama serve`` processes running now as (pid, command line)."""
    found = []
    for pid, _, command in _processes():
        words = command.split()
        if _program(command) == "ollama" and "serve" in words[1:]:
            found.append((pid, command))
    return tuple(found)


def running_servers() -> tuple[RunningServer, ...]:
    """The vLLM and llama.cpp servers running now, bar this process and its children and the
    ``llama-server`` that Ollama starts for a loaded model."""
    rows = _processes()
    parents = {pid: parent for pid, parent, _ in rows}
    programs = {pid: _program(command) for pid, _, command in rows}
    me = os.getpid()

    def mine(pid: int) -> bool:
        seen = set()
        while pid > 1 and pid not in seen:
            if pid == me:
                return True
            seen.add(pid)
            pid = parents.get(pid, 0)
        return False

    servers = []
    for pid, _, command in rows:
        if mine(pid) or QUERY_FLAGS.intersection(command.split()):
            continue
        owned = programs.get(parents.get(pid, 0)) == "ollama"
        if programs[pid] == "llama-server" and (owned or OLLAMA_BLOB in command):
            continue
        for engine in ENGINES.values():
            try:
                recognised = engine.recognizes(command)
            except (ValueError, OSError):
                recognised = False
            if recognised:
                servers.append(RunningServer(pid, engine.name, shown_command(command)))
                break
    return tuple(servers)


_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")


def program_of(command: str | None) -> str:
    """The program a serve command runs, past any leading ``NAME=value`` assignments; "" when
    the command is empty or cannot be split."""
    try:
        words = shlex.split(command or "")
    except ValueError:
        return ""
    words = _past_sudo(words[_first_word(words) :])
    return words[0] if words else ""


def _first_word(words: list[str]) -> int:
    return next((i for i, w in enumerate(words) if not _ASSIGNMENT.match(w)), 0)


def _past_sudo(words: list[str]) -> list[str]:
    """``words`` after a leading ``sudo`` and its options."""
    if not words or os.path.basename(words[0]) != "sudo":
        return words
    rest = words[1:]
    while rest and rest[0].startswith("-"):
        rest = rest[1:]
    return rest


CONTAINER_PROGRAMS = ("docker", "podman")


def run_image(command: str | None) -> str | None:
    """The image a ``docker run`` (or ``podman run``) serve command starts; None for any other
    command."""
    try:
        words = shlex.split(command or "")
    except ValueError:
        return None
    words = _past_sudo(words[_first_word(words) :])
    if not words or os.path.basename(words[0]) not in CONTAINER_PROGRAMS:
        return None
    try:
        return _split_docker_run(words)[0]
    except ValueError:
        return None


def detect_machine(current: str | None = None) -> Machine:
    """What is here; with ``current``, a serve command whose program is given by its path, the
    engine it runs counts as installed when that program answers, though it is not on PATH, and
    a ``docker run`` command's engine counts when its image is on this machine."""
    engines = detect_engines()
    if (image := run_image(current)) is not None:
        for name, engine in ENGINES.items():
            if all(found.name != name for found in engines) and engine.recognizes(current or ""):
                found = engine.availability("docker", image)
                if found.available:
                    engines += (EngineFound(name, found.version, image, ()),)
    program = program_of(current)
    if os.sep in program and os.access(program, os.X_OK):
        named = next((n for n, e in ENGINES.items() if e.recognizes(current or "")), None)
        if named is not None and all(found.name != named for found in engines):
            engines += detect_engines((named,), program)
    return Machine(engines, detect_hardware(), running_servers())


def install_hint(hardware: Hardware) -> str:
    if hardware.kind == "nvidia":
        return f"`uv tool install vllm`, or a CUDA build of llama.cpp ({LLAMACPP_BACKENDS})"
    if hardware.kind == "apple":
        return "`brew install llama.cpp`"
    return f"download llama.cpp from {LLAMACPP_RELEASES}"


def serving_problems(
    engine: str, hardware: Hardware, backends: tuple[str, ...]
) -> tuple[Problem, ...]:
    """What stops or worries ``engine`` on ``hardware``. ``backends`` are those of the
    llama-server build, empty when unknown."""
    problems = []
    if engine == "vllm":
        if hardware.kind != "nvidia":
            problems.append(
                Problem(
                    True,
                    f"vLLM cannot serve on {hardware.line()}: it needs an NVIDIA GPU. "
                    f"llama.cpp is supported on this machine: {install_hint(hardware)}",
                )
            )
        elif not hardware.driver or not hardware.cuda:
            problems.append(
                Problem(True, "vLLM needs the NVIDIA driver and CUDA; nvidia-smi found no driver")
            )
    elif engine == "llamacpp":
        if hardware.kind == "nvidia" and "CPU" in backends and "CUDA" not in backends:
            problems.append(
                Problem(False, "this llama-server is a CPU-only build: it will not use the GPU")
            )
        if hardware.kind == "apple" and backends and "Metal" not in backends:
            problems.append(
                Problem(
                    True,
                    "this llama-server has no Metal: download "
                    f"`llama-b{PIN_BUILD}-bin-macos-arm64.tar.gz` or rebuild with Metal "
                    "(on by default)",
                )
            )
    if len(hardware.devices) > 1 and hardware.kind == "nvidia":
        problems.append(
            Problem(
                False,
                f"This machine has {len(hardware.devices)} GPUs. Tensward tunes one GPU; "
                "it continues with single-GPU advice.",
            )
        )
    return tuple(problems)


def _weights(model: Path) -> tuple[int, str]:
    """The size of the weight files and how they are stored, from the file headers only."""
    if model.is_file() or any(model.glob("*.gguf")):
        files = model_files(model)
        quant = _file_type(read_gguf(files[0]).metadata)
        return sum(f.stat().st_size for f in files), quant.lower()
    size = sum(f.stat().st_size for f in model.glob("*.safetensors"))
    config = json.loads((model / "config.json").read_text())
    quantized = (config.get("quantization_config") or {}).get("quant_method")
    dtype = config.get("dtype") or config.get("torch_dtype")
    return size, quantized or DTYPES.get(str(dtype), str(dtype))


def _fit_verdict(size: int, hardware: Hardware) -> str:
    if hardware.kind == "cpu" or not hardware.devices:
        room = hardware.ram_bytes
    else:
        room = hardware.devices[0][1]
    if room is None:
        return "fit unknown"
    where = "RAM" if hardware.kind == "cpu" or not hardware.devices else "this GPU"
    return f"fits {where}" if size <= room * WEIGHT_MEMORY_SHARE else f"may not fit {where}"


def model_line(model: Path, hardware: Hardware) -> str:
    """``/models/Qwen3-8B (bf16, 15.3 GiB, fits this GPU)``: the weights against the memory."""
    size, stored = _weights(model)
    return f"{model} ({stored}, {size / GIB:.1f} GiB, {_fit_verdict(size, hardware)})"


FIT_CONSEQUENCE = (
    "          Too big for that memory: `tensward try` can keep part of the model in CPU memory."
)


def fit_consequence(model: Path, hardware: Hardware) -> str | None:
    """What happens when the weights may not fit the memory ``model_line`` compares them with."""
    size, _ = _weights(model)
    return FIT_CONSEQUENCE if _fit_verdict(size, hardware).startswith("may not fit") else None
