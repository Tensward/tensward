"""vLLM command lines: flag tables, parsing a launch command, version probes."""

from __future__ import annotations

import json
import os
import re
import shlex
from typing import Any, Callable, Mapping, Sequence

from ...probes import run_probe, script_interpreter
from ...secrets import ASSIGNMENT
from ...settings import Settings


def _count(text: str) -> int:
    """An integer that may carry vLLM's k/m/g suffix: lower case is 1000s, upper case 1024s."""
    power = {"k": 1, "m": 2, "g": 3}.get(text[-1:].lower())
    if power is None:
        return int(text)
    return int(float(text[:-1]) * (1024 if text[-1].isupper() else 1000) ** power)


def _context_length(text: str) -> int | None:
    """--max-model-len: ``auto`` and ``-1`` both ask for the model's own length, i.e. unset."""
    return None if text in ("auto", "-1") else _count(text)


def _media_limits(text: str) -> dict[str, int]:
    """--limit-mm-per-prompt: a JSON object of modality to a count of at most that many inputs."""
    limits = json.loads(text)
    if not isinstance(limits, dict) or not all(
        isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in limits.values()
    ):
        raise ValueError("limits must be a JSON object of non-negative integers")
    return limits


def _flag_text(value: Any) -> str:
    """A setting as its flag's text: floats without trailing zeros, limits as sorted JSON."""
    if isinstance(value, float):
        return f"{value:g}"
    return json.dumps(value, sort_keys=True) if isinstance(value, Mapping) else str(value)


# Settings field -> (vLLM flag, how to read the flag's text). A None value is not passed.
_VALUE_FLAGS: dict[str, tuple[str, Callable[[str], Any]]] = {
    "dtype": ("--dtype", str),
    "max_concurrent_requests": ("--max-num-seqs", _count),
    "max_context_len": ("--max-model-len", _context_length),
    "kv_memory_fraction": ("--gpu-memory-utilization", float),
    "prefill_batch_tokens": ("--max-num-batched-tokens", _count),
    "kv_cache_dtype": ("--kv-cache-dtype", str),
    "quantization": ("--quantization", str),
    "tool_parser": ("--tool-call-parser", str),
    "api_server_count": ("--api-server-count", int),
    "media_limits": ("--limit-mm-per-prompt", _media_limits),
}
_FLAG_TO_FIELD = {flag: (name, read) for name, (flag, read) in _VALUE_FLAGS.items()}
PREFIX_CACHING_ON = "--enable-prefix-caching"
PREFIX_CACHING_OFF = "--no-enable-prefix-caching"
CUDA_GRAPHS_OFF = "--enforce-eager"
TOOL_CALLING_ON = "--enable-auto-tool-choice"
ASYNC_SCHEDULING_ON = "--async-scheduling"  # on by default in 0.30 unless incompatible
ASYNC_SCHEDULING_OFF = "--no-async-scheduling"
# vLLM 0.30 MultiModalConfig.language_model_only sets every modality limit to 0.
# `--limit-mm-per-prompt image=0` alone still reserves memory for video.
MEDIA_OFF = "--language-model-only"
MEDIA_ON = "--no-language-model-only"
# VllmConfig.max_in_flight_tokens = max_concurrent_batches (2 with async scheduling)
# x max_num_batched_tokens; 2048 is the serving default below 70 GiB GPUs.
DEFAULT_PREFILL_BATCH_TOKENS = 2048
MAX_CONCURRENT_BATCHES = 2
# With media inputs on, vLLM 0.30 raises max_num_batched_tokens to the largest media item: 2496
# for Gemma 4 (a video item, not derivable from its processor config).
MEDIA_ITEM_BATCH_TOKENS = 2496
DEFAULT_KV_MEMORY_FRACTION = 0.92  # CacheConfig.gpu_memory_utilization at v0.30.0
# Memory a launch takes beyond the weights (as the anatomy counts them) and the KV
# cache: CUDA context, activation workspace, CUDA graphs, sampler and media-encoder profiling.
# Measured on an L4 with Gemma 4 26B-A4B AWQ: 20.69 GiB usable - 16.02 GiB weights - 2.19 GiB
# KV cache = 2.48 GiB, rounded up. An estimate.
MEMORY_OVERHEAD_BYTES = int(2.5 * 2**30)
PARALLEL_FLAGS = ("--tensor-parallel-size", "--pipeline-parallel-size")
# `-dp` is read here only: it does not join _ALIASES, so parse_setup keeps what the user typed.
DATA_PARALLEL_FLAGS = ("--data-parallel-size", "-dp")
# Short spellings (from `vllm serve --help=all` at 0.30.0).
_ALIASES = {
    "-q": "--quantization",
    "-asc": "--api-server-count",
    "-tp": "--tensor-parallel-size",
    "-pp": "--pipeline-parallel-size",
    "-cc": "--compilation-config",
}
# Boolean flag -> (Settings field, its value when the flag is given as-is). vLLM spells each
# switch on and off (`--x` / `--no-x`), and `--x=false` means the opposite of `--x`.
_SWITCHES: dict[str, tuple[str, bool]] = {
    PREFIX_CACHING_ON: ("prefix_caching", True),
    PREFIX_CACHING_OFF: ("prefix_caching", False),
    TOOL_CALLING_ON: ("tool_calling", True),
    "--no-enable-auto-tool-choice": ("tool_calling", False),
    CUDA_GRAPHS_OFF: ("cuda_graphs", False),
    "--no-enforce-eager": ("cuda_graphs", True),
    ASYNC_SCHEDULING_ON: ("async_scheduling", True),
    ASYNC_SCHEDULING_OFF: ("async_scheduling", False),
    MEDIA_OFF: ("media_inputs", False),
    MEDIA_ON: ("media_inputs", True),
}
COMPILATION_FLAG = "--compilation-config"
MAX_CAPTURE_FLAG = "--max-cudagraph-capture-size"
SPECULATIVE_CONFIG_FLAG = "--speculative-config"
TRUST_REMOTE_CODE_FLAG = "--trust-remote-code"
STRUCTURED_OUTPUTS_FLAG = "--structured-outputs-config"
# vLLM 0.30 takes disable_any_whitespace only with these backends (config/structured_outputs.py)
COMPACT_JSON_BACKENDS = ("xgrammar", "guidance")
# The capture sizes tensward reads and adjusts are the ones in --compilation-config's JSON; these
# spellings carry them where it cannot, so they are refused instead of silently not adjusted.
_UNREAD_CAPTURE_FLAGS = (
    "--cudagraph-capture-sizes",
    f"{COMPILATION_FLAG}.cudagraph_capture_sizes",
    f"{COMPILATION_FLAG}.max_cudagraph_capture_size",
)
# Profiler (vllm/config/profiler.py, entrypoints/serve/profile/api_router.py at v0.30.0): the
# torch profiler must be enabled at launch, which also mounts POST /start_profile and
# /stop_profile. A worker trace and an API-server (AsyncLLM) trace are written into
# torch_profiler_dir as <host>_<pid>[.async_llm].<ts>.pt.trace.json.gz. max_iterations stops
# the worker after that many engine steps, which bounds the trace size.
PROFILER_FLAG = "--profiler-config"
TRACE_MAX_ITERATIONS = 100
TRACE_GLOB = "*.pt.trace.json*"
TRACE_STOP_TIMEOUT_S = 120.0  # /stop_profile blocks until the trace is flushed
# The checkpoint's config.json model_type -> the vLLM 0.30 tool parser for its tool-call format
# (names from vllm/tool_parsers/__init__.py and docs/features/tool_calling.md at v0.30.0). Qwen2
# and Qwen2.5 use the Hermes format in their chat template; v0.30.0 registers `gemma4`.
_TOOL_PARSERS = {
    "qwen2": "hermes",
    "mistral": "mistral",
    "gemma4": "gemma4",
    "qwen3_5": "qwen3_xml",
}
LLAMA3_VOCAB_SIZE = 128256  # model_type "llama" covers Llama 2 too; Llama 3.x has this vocabulary
# Owned by Tensward (the runtime sets them), so an operator cannot override them.
RESERVED_FLAGS = frozenset({"--host", "--port", "--served-model-name", "--api-key", "--model"})
_MULTI_VALUE_FLAGS = frozenset({"--served-model-name", "--api-key"})  # nargs="+" in vLLM
# Only environment that changes the engine's behaviour is kept from a customer's command.
ENGINE_ENV_PREFIX = "VLLM_"
CONTAINER_CLIS = ("docker", "podman")
VLLM_VERSION = "0.30.0"  # the release this adapter is validated on
OCI_VERSION_LABEL = "org.opencontainers.image.version"
_PYTHON = re.compile(r"python(\d+(\.\d+)?)?")
_TAG_VERSION = re.compile(r":v?(\d+(?:\.\d+)+)$")
# Methods that vLLM v0.30 resolves to one another (quantization/__init__.py, the override hooks)
QUANTIZATION_FAMILIES = {
    "gptq_marlin": "gptq",
    "auto_gptq": "gptq",
    "awq_marlin": "awq",
    "auto_awq": "awq",
}
_SECRET_FLAG_NOTE = (
    "{flag} carries a secret, which Tensward never stores or puts on a command line; set the "
    "matching environment variable (for a hub token, HF_TOKEN) where Tensward runs instead"
)
_NUMBER = re.compile(r"-?[0-9.]+")
_VLLM_MODULE = "vllm.entrypoints.openai.api_server"
_LAUNCHER_HELP = (
    "expected a `vllm serve ...`, `python -m vllm.entrypoints.openai.api_server ...` or "
    "`docker run ... <image> ...` command"
)
_DOCKER_SWITCH = re.compile(
    r"--(detach|interactive|tty|rm|privileged|init|read-only|publish-all|no-healthcheck"
    r"|oom-kill-disable)|-[diPqt]+"
)


def _flag_name(name: str) -> str:
    """A flag in its canonical spelling: vLLM accepts `--foo_bar` for `--foo-bar`. A dotted flag
    (`--compilation-config.cudagraph_mode`) keeps what follows the dot: vLLM converts underscores
    only before it, and the field names after it use them (utils/argparse_utils.py)."""
    head, dot, tail = name.partition(".")
    head = _ALIASES.get(head) or (
        "--" + head[2:].replace("_", "-") if head.startswith("--") else head
    )
    return head + dot + tail


def _require_readable(flag: str) -> None:
    if flag in _UNREAD_CAPTURE_FLAGS:
        raise ValueError(
            f"{flag} is not supported: use "
            f"""{COMPILATION_FLAG} '{{"cudagraph_capture_sizes": [...]}}' instead"""
        )


def _device_list(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _docker_gpu_devices(value: str) -> tuple[str, ...]:
    """The devices of docker's ``--gpus device=0,1`` (quoted or not); ``all`` or a count name no
    particular device."""
    value = value.strip("\"'")
    return _device_list(value.removeprefix("device=")) if value.startswith("device=") else ()


def _split_docker_run(tokens: list[str]) -> tuple[str, list[str], list[tuple[str, str]]]:
    """A `docker run` line as (image, the image's arguments, [(docker option, value)])."""
    rest = tokens[1:]
    if rest[:1] == ["container"]:
        rest = rest[1:]
    if rest[:1] != ["run"]:
        raise ValueError("only `docker run` is supported")
    options: list[tuple[str, str]] = []
    position = 1
    while position < len(rest) and rest[position].startswith("-"):
        flag, equals, value = rest[position].partition("=")
        position += 1
        if not equals and not _DOCKER_SWITCH.fullmatch(flag):
            if position >= len(rest):
                raise ValueError(f"docker option {flag} has no value")
            value = rest[position]
            position += 1
        options.append((flag, value))
    if position >= len(rest):
        raise ValueError("the docker run line names no image")
    return rest[position], rest[position + 1 :], options


def _installed_version(command: str) -> str | None:
    """The version of the vLLM that the ``vllm`` script, or the Python interpreter, at
    ``command`` runs. The package metadata answers in milliseconds, where ``vllm --version``
    imports torch first."""
    name = os.path.basename(command)
    if name == "vllm":
        python = script_interpreter(command)
    else:
        python = command if _PYTHON.fullmatch(name) else None
    if python is None:
        return None
    out = run_probe([python, "-c", "import importlib.metadata as m; print(m.version('vllm'))"])
    return (out or "").strip() or None


def _image_version(image: str, labels: Mapping[str, str]) -> str | None:
    """The vLLM version an image's tag names (``:v0.30.0``), else its OCI version label."""
    tag = _TAG_VERSION.search(image)
    return tag[1] if tag else labels.get(OCI_VERSION_LABEL)


def _launch_tokens(text: str) -> tuple[dict[str, str], list[str]]:
    """A command line as (the ``VAR=value`` assignments before it, its tokens after any
    ``sudo``)."""
    tokens = shlex.split(re.sub(r"\\\r?\n", " ", text))
    env: dict[str, str] = {}
    while tokens and ASSIGNMENT.match(tokens[0]):  # VAR=value vllm serve ...
        name, _, value = tokens.pop(0).partition("=")
        env[name] = value
    return env, tokens[1:] if tokens[:1] == ["sudo"] else tokens


def _is_container_run(tokens: Sequence[str]) -> bool:
    return bool(tokens) and os.path.basename(tokens[0]) in CONTAINER_CLIS


def _without_launcher(tokens: list[str], *, required: bool) -> list[str]:
    """The engine's own arguments: drop `vllm serve` or `python -m <api_server module>`."""
    if tokens[:2] == ["vllm", "serve"]:
        return tokens[2:]
    if len(tokens) > 2 and tokens[0].startswith("python") and tokens[1:3] == ["-m", _VLLM_MODULE]:
        return tokens[3:]
    if required:
        raise ValueError(_LAUNCHER_HELP)
    return tokens


def _flags(tokens: list[str]) -> tuple[list[tuple[str, str | None]], str | None]:
    """(flag, value) pairs in canonical spelling, and the positional model if there is one."""
    pairs: list[tuple[str, str | None]] = []
    model = None
    position = 0
    while position < len(tokens):
        token = tokens[position]
        position += 1
        if not token.startswith("-") or _NUMBER.fullmatch(token):
            if model is not None:
                raise ValueError(f"unexpected argument {token!r}: cannot tell which flag it is for")
            model = token
            continue
        name, equals, value = token.partition("=")
        flag = _flag_name(name)
        _require_readable(flag)
        given = value if equals else None
        if given is None and flag not in _SWITCHES and position < len(tokens):
            if not tokens[position].startswith("-") or _NUMBER.fullmatch(tokens[position]):
                given = tokens[position]
                position += 1
                while flag in _MULTI_VALUE_FLAGS and position < len(tokens):
                    if tokens[position].startswith("-"):
                        break
                    position += 1
        pairs.append((flag, given))
    return pairs, model


def torch_mismatch_warning() -> str | None:
    """A warning when the installed torch-family packages were built for different CUDA
    versions, which makes vLLM crash at start-up."""
    from ...environment import torch_cuda_builds

    builds = torch_cuda_builds()
    if len(set(builds.values())) < 2:
        return None
    found = ", ".join(f"{name} {tag}" for name, tag in builds.items())
    return (
        f"warning: torch packages are built for different CUDA versions ({found}); vLLM can "
        "crash at start-up. Fix: pip uninstall -y torchaudio, or install builds for the same CUDA"
    )


def data_parallel_size(settings: Settings) -> int:
    """The data-parallel engines the flags ask for; 1 when none, or when the value is not a
    number (vLLM then refuses to start)."""
    for flag in DATA_PARALLEL_FLAGS:
        try:
            size = int(str(settings.extra_args.get(flag) or 1))
        except ValueError:
            size = 1
        if size > 1:
            return size
    return 1
