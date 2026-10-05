"""vLLM: the first engine. Everything specific to vLLM 0.30 lives here.

That is its flag names, image, API-key variable, endpoints and Prometheus metric names.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import os
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, Sequence

import httpx

from ..platforms.nvidia import VISIBLE_DEVICES
from ..probes import executable, run_probe, script_interpreter
from .protocol import (
    Availability,
    EngineSignals,
    ParsedSetup,
    QuantKernel,
    Settings,
    docker_availability,
)

if TYPE_CHECKING:
    from ..playbook import Entry


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
# The capture sizes tensward reads and adjusts are the ones in --compilation-config's JSON; these
# spellings carry them where it cannot, so they are refused instead of silently not adjusted.
_UNREAD_CAPTURE_FLAGS = (
    "--cudagraph-capture-sizes",
    f"{COMPILATION_FLAG}.cudagraph_capture_sizes",
    f"{COMPILATION_FLAG}.max_cudagraph_capture_size",
)
# Speculative methods that vLLM v0.30 runs on its V1 model runner (config/vllm.py
# use_v2_model_runner), the only one that rounds CUDA graph sizes to multiples of 1 + k.
_V1_SPECULATIVE_METHODS = frozenset(
    {"ngram", "ngram_gpu", "draft_model", "suffix", "medusa", "mlp_speculator", "custom_class"}
)
_NO_ROUNDING_MODES = frozenset({"PIECEWISE", "NONE"})
_METHOD_NAMES = {"ngram": "n-gram"}
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
_TOOL_PARSERS = {"qwen2": "hermes", "mistral": "mistral", "gemma4": "gemma4"}
LLAMA3_VOCAB_SIZE = 128256  # model_type "llama" covers Llama 2 too; Llama 3.x has this vocabulary
# Owned by Tensward (the runtime sets them), so an operator cannot override them.
RESERVED_FLAGS = frozenset({"--host", "--port", "--served-model-name", "--api-key", "--model"})
_MULTI_VALUE_FLAGS = frozenset({"--served-model-name", "--api-key"})  # nargs="+" in vLLM
# Only environment that changes the engine's behaviour is kept from a customer's command.
ENGINE_ENV_PREFIX = "VLLM_"
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
CONTAINER_CLIS = ("docker", "podman")
VLLM_VERSION = "0.30.0"  # the release this adapter is validated on
OCI_VERSION_LABEL = "org.opencontainers.image.version"
_TAG_VERSION = re.compile(r":v?(\d+(?:\.\d+)+)$")
# A flag or variable whose name has one of these words (`--hf-token`, `HF_TOKEN`,
# `AWS_SECRET_ACCESS_KEY`) carries a secret. Whole words, so `--max-num-batched-tokens` is not one.
_SECRET_WORDS = frozenset({"key", "apikey", "secret", "password", "passwd", "auth"})
# Methods that vLLM v0.30 resolves to one another (quantization/__init__.py, the override hooks)
QUANTIZATION_FAMILIES = {
    "gptq_marlin": "gptq",
    "auto_gptq": "gptq",
    "awq_marlin": "awq",
    "auto_awq": "awq",
}
_ENV_FLAGS = ("-e", "--env")
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

# Startup log lines that name the quantized-linear kernel chosen for a layer type, e.g.
# "Using MarlinLinearKernel for AutoAWQMarlinLinearMethod" (AWQ, GPTQ, compressed-tensors
# weight-only) and "Selected CutlassFP8ScaledMMLinearKernel for Fp8LinearMethod" (FP8, int8
# W8A8). The wording is from vLLM 0.30.0's source (model_executor/kernels/linear and
# layers/quantization); it has not been checked against a running server.
_KERNEL_LINE = re.compile(r"(?:Using|Selected) (\w+Kernel) for (\w+)")
# The AWQ layer that cannot use Marlin says so once per layer.
_AWQ_FALLBACK = "Falling back to unoptimized AWQ kernels"
# The MoE layers name their backend once: "Using 'MARLIN' WNA16 MoE backend." (seen on L4 and
# A10G with Gemma 4 26B-A4B AWQ, 2026-10-01). No warning pattern: that run logged none.
_MOE_BACKEND = re.compile(r"Using '(\w+)' (\w+) MoE backend")
# Kernels vLLM 0.30.0 ranks after MarlinLinearKernel for CUDA weight-only quantization
# (_POSSIBLE_KERNELS in kernels/linear/__init__.py): Marlin is chosen first when it fits.
_SLOW_KERNELS = frozenset({"ConchLinearKernel", "ExllamaLinearKernel", "TritonW4A16LinearKernel"})

_NAME = re.compile(r"[A-Za-z_:][A-Za-z0-9_:]*")
_LABEL = re.compile(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*,?')
_ESCAPE = re.compile(r"\\(.)")
_VALUE = re.compile(r"\s+([-+]?(?:\d[\d.]*(?:[eE][-+]?\d+)?|Inf|NaN))")

PROMPT_TOKENS_FAMILY = "vllm:prompt_tokens_total"
GENERATION_TOKENS_FAMILY = "vllm:generation_tokens_total"
ITERATIONS_FAMILY = "vllm:iteration_tokens_total_count"  # steps, from a histogram
KV_USAGE_FAMILY = "vllm:kv_cache_usage_perc"
RUNNING_FAMILY = "vllm:num_requests_running"
WAITING_FAMILY = "vllm:num_requests_waiting"
PREEMPTIONS_FAMILY = "vllm:num_preemptions_total"
PROMPT_BY_SOURCE_FAMILY = "vllm:prompt_tokens_by_source_total"
PREFIX_HITS_FAMILY = "vllm:prefix_cache_hits_total"  # prompt TOKENS served from cache (not blocks)
PREFIX_QUERIES_FAMILY = "vllm:prefix_cache_queries_total"
QUEUE_TIME_FAMILY = "vllm:request_queue_time_seconds_sum"
PREFILL_TIME_FAMILY = "vllm:request_prefill_time_seconds_sum"
SPEC_DRAFTS_FAMILY = "vllm:spec_decode_num_drafts_total"
SPEC_ACCEPTED_FAMILY = "vllm:spec_decode_num_accepted_tokens_total"
# An info gauge (value 1) whose labels are CacheConfig's fields (v0.30 CacheConfig.metrics_info).
# kv_cache_size_tokens is group-aware, so right for hybrid models, where num_gpu_blocks x
# block_size is not.
CACHE_CONFIG_FAMILY = "vllm:cache_config_info"
KV_CAPACITY_LABEL = "kv_cache_size_tokens"
KV_MAX_CONCURRENCY_LABEL = "kv_cache_max_concurrency"


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
    """The version of the vLLM that the ``vllm`` script at ``command`` runs. The package
    metadata answers in milliseconds, where ``vllm --version`` imports torch first."""
    python = script_interpreter(command) if os.path.basename(command) == "vllm" else None
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
    while tokens and _ASSIGNMENT.match(tokens[0]):  # VAR=value vllm serve ...
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


def _is_secret(name: str) -> bool:
    """A name that carries a credential: ``--api-key``, ``HF_TOKEN``, ``--hf-token``. "token"
    counts only as the last word, because serving flags count tokens
    (``--long-prefill-token-threshold``)."""
    words = re.split(r"[-_]+", name.lower().strip("-"))
    return not _SECRET_WORDS.isdisjoint(words) or words[-1] == "token"


def _blank(assignment: str) -> str:
    """``NAME=value`` with the value blanked when NAME is a secret's name."""
    name, equals, _ = assignment.partition("=")
    return f"{name}=***" if equals and _is_secret(name) else assignment


def _without_secrets(tokens: list[str]) -> list[str]:
    """The command with every secret value blanked: a secret-named flag's value (`--api-key`,
    `--hf-token`) and a secret-named variable's (`NAME=v`, `-e NAME=v`, `--env=NAME=v`)."""
    shown: list[str] = []
    hiding = 0  # how many following tokens are a secret flag's values
    env_next = False  # the next token is the value of a docker -e / --env
    for token in tokens:
        if hiding and not token.startswith("-"):
            shown.append("***")
            hiding -= 1
            continue
        hiding = 0
        name, equals, value = token.partition("=")
        if env_next:
            shown.append(_blank(token))
        elif not token.startswith("-"):
            shown.append(_blank(token) if _ASSIGNMENT.match(token) else token)
        elif _flag_name(name) in _ENV_FLAGS:
            shown.append(f"{name}={_blank(value)}" if equals else token)
        elif token.startswith("-e") and not token.startswith("--") and equals:  # -eNAME=value
            shown.append("-e" + _blank(token[2:]))
        elif _is_secret(name):
            shown.append(f"{name}=***" if equals else token)
            if not equals:
                hiding = 99 if _flag_name(name) in _MULTI_VALUE_FLAGS else 1
        else:
            shown.append(token)
        env_next = token in _ENV_FLAGS
    return shown


def _json_object(settings: Settings, flag: str) -> dict[str, Any]:
    """The JSON object a flag carries; empty when it is absent or carries anything else."""
    try:
        value = json.loads(str(settings.extra_args.get(flag)))
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _int_list(value: Any) -> list[int] | None:
    if isinstance(value, list) and value and all(type(n) is int and n > 0 for n in value):
        return value
    return None


def _capture_sizes(settings: Settings) -> tuple[list[int] | None, int | None]:
    """The pinned CUDA graph capture sizes, and the pinned largest one, as ``--compilation-config``
    (``-cc``) and ``--max-cudagraph-capture-size`` give them."""
    compilation = _json_object(settings, COMPILATION_FLAG)
    ceiling = compilation.get("max_cudagraph_capture_size")
    if type(ceiling) is not int:
        try:
            ceiling = int(str(settings.extra_args[MAX_CAPTURE_FLAG]))
        except (KeyError, ValueError):
            ceiling = None
    return _int_list(compilation.get("cudagraph_capture_sizes")), ceiling


def _default_sizes(ceiling: int) -> list[int]:
    """The sizes vLLM v0.30 captures up to ``ceiling`` when none are pinned (config/vllm.py)."""
    grid = [size for size in (1, 2, 4) if size <= ceiling]
    return grid + list(range(8, min(ceiling + 1, 256), 8)) + list(range(256, ceiling + 1, 16))


def _graph_mode(settings: Settings) -> str:
    return str(_json_object(settings, COMPILATION_FLAG).get("cudagraph_mode", "")).upper()


def _speculation(settings: Settings) -> tuple[str, int]:
    """The speculative method and its tokens per step per sequence (``num_speculative_tokens``,
    which a method that picks its own count, such as MTP, may leave unset: counted as 0)."""
    config = _json_object(settings, SPECULATIVE_CONFIG_FLAG)
    tokens = config.get("num_speculative_tokens")
    return str(config.get("method", "")), tokens if type(tokens) is int and tokens > 0 else 0


def _pinned_sizes(settings: Settings) -> list[int] | None:
    """The capture sizes in effect when the settings pin them or their largest, else None
    (also None when CUDA graphs are off)."""
    if not settings.cuda_graphs or _graph_mode(settings) == "NONE":
        return None
    sizes, ceiling = _capture_sizes(settings)
    if sizes is None and ceiling is not None and ceiling > 0:
        return _default_sizes(ceiling)
    return sizes


def _captured_sequences(sizes: list[int], k: int, mode: str, *, rounded: bool) -> int | None:
    """How many sequences a decode step can batch inside a captured CUDA graph, each taking
    ``1 + k`` tokens, or None when vLLM v0.30 would refuse the sizes. With ``rounded`` (the V1
    model runner) and a graph mode that captures decode, every size is rounded up to a multiple
    of ``1 + k`` and those above the largest pinned size are dropped, keeping ``1 + k`` itself if
    nothing remains (config/compilation.py adjust_cudagraph_sizes_for_spec_decode). It ignores
    that the multiple is ``max(1 + k, tp)`` under tensor and sequence parallelism."""
    step = 1 + k
    if rounded and k and mode not in _NO_ROUNDING_MODES:
        top = max(sizes)
        sizes = sorted({r for size in sizes if (r := -(-size // step) * step) <= top})
        if not sizes and step <= top:
            sizes = [step]
    return max(sizes) // step if sizes else None


def _coverage(settings: Settings) -> tuple[bool, int | None]:
    """Whether the settings pin CUDA graph sizes, and how many sequences those graphs batch."""
    sizes = _pinned_sizes(settings)
    if sizes is None:
        return False, None
    method, k = _speculation(settings)
    rounded = method in _V1_SPECULATIVE_METHODS
    return True, _captured_sequences(sizes, k, _graph_mode(settings), rounded=rounded)


def _widen_graphs(before: Settings, after: Settings) -> tuple[Settings, str | None]:
    """``after`` with its pinned CUDA graph sizes raised to cover a larger concurrency than
    ``before`` ran, written back where they were found; the sizes of any other change stay."""
    n, was = after.max_concurrent_requests, before.max_concurrent_requests
    pinned, covered = _coverage(after)
    if not pinned or n is None or was is None or n <= was or (covered or 0) >= n:
        return after, None
    method, k = _speculation(after)
    step = 1 + k
    rounded = method in _V1_SPECULATIVE_METHODS
    mode = _graph_mode(after)
    sizes, _ = _capture_sizes(after)
    compilation = _json_object(after, COMPILATION_FLAG)
    if sizes is not None:
        wanted = {step * 2**power for power in range(n.bit_length())}
        sizes = sorted({*sizes, *wanted, step * n})
        compilation["cudagraph_capture_sizes"] = sizes
        if "max_cudagraph_capture_size" in compilation:
            compilation["max_cudagraph_capture_size"] = sizes[-1]
        widened = sizes[-1]
    else:  # vLLM builds its default grid from the pinned maximum, then truncates it to fit
        widened = next(
            grid[-1]
            for grid in (_default_sizes(top) for top in itertools.count(n))
            if (_captured_sequences(grid, k, mode, rounded=rounded) or 0) >= n
        )
        if "max_cudagraph_capture_size" in compilation:
            compilation["max_cudagraph_capture_size"] = widened
    extra = dict(after.extra_args)
    if sizes is not None or "max_cudagraph_capture_size" in compilation:
        extra[COMPILATION_FLAG] = json.dumps(compilation)
    else:
        extra[MAX_CAPTURE_FLAG] = str(widened)
    note = (
        f"raises the captured CUDA graph sizes to {n} with it, "
        "so the larger batches keep their graphs"
    )
    return dataclasses.replace(after, extra_args=extra), note


class VllmEngine:
    name = "vllm"
    label = "vLLM"
    formats = frozenset({"hf-safetensors"})
    platforms = frozenset({"nvidia"})
    launchers = (
        "`vllm serve ...`, `python -m vllm.entrypoints.openai.api_server ...` or a `docker run` "
        "of an image named vllm"
    )
    default_image = f"vllm/vllm-openai:v{VLLM_VERSION}"
    local_command: tuple[str, ...] = ("vllm", "serve")
    api_key_env = "VLLM_API_KEY"
    health_path = "/health"
    models_path = "/v1/models"
    metrics_path = "/metrics"
    tokenize_path = "/tokenize"
    # Adds "gpu_model_runner: forward/sample/..." and "schedule: ..." annotations to the trace
    # (vllm/v1/utils.py record_function_or_nullcontext).
    trace_env: Mapping[str, str] = {"VLLM_CUSTOM_SCOPES_FOR_PROFILING": "1"}
    # CUDA-profiler mode: /start_profile and /stop_profile then only open and close the window
    # that `ncu --profile-from-start off` profiles, so startup kernels are never counted.
    counters_args: Sequence[str] = (PROFILER_FLAG, json.dumps({"profiler": "cuda"}))
    trace_step_scope = "gpu_model_runner: forward"
    dtype_cast: str = r"Casting torch\.(\w+) to torch\.(\w+)"
    log_checks: Mapping[str, str] = {
        "JIT compilation during inference": (
            'vLLM compiled a kernel during the measured window ("JIT compilation during '
            'inference" in server.log): some latencies include compilation; raising '
            "warmup_requests in config.json covers it"
        ),
    }
    # Read from the v0.30.0 source: CacheConfig and SchedulerConfig defaults, and
    # EngineArgs.get_batch_defaults (the OpenAI server column) in engine/arg_utils.py.
    defaults: Mapping[str, str] = {
        "prefix_caching": "on (for generative models)",
        "prefill_batch_tokens": (
            f"{DEFAULT_PREFILL_BATCH_TOKENS} (8192 on GPUs with 70 GiB or more that are not "
            "A100, 16384 from 160 GiB), with chunked prefill on"
        ),
        "max_concurrent_requests": "256 (1024 on GPUs with 70 GiB or more that are not A100)",
        "kv_memory_fraction": str(DEFAULT_KV_MEMORY_FRACTION),
        "async_scheduling": "on, unless the setup is incompatible with it",
        "cuda_graphs": "on",
    }

    default_kv_memory_fraction = DEFAULT_KV_MEMORY_FRACTION
    memory_overhead_bytes = MEMORY_OVERHEAD_BYTES

    added_flags = (
        "--generation-config vllm (sampling follows the declared workload, not the checkpoint's "
        "generation_config.json), and the host, port, served model name and API key"
    )

    def launch_argv(
        self,
        settings: Settings,
        *,
        model: str,
        served_model_name: str,
        host: str,
        port: int,
        trace_dir: str | None = None,
    ) -> list[str]:
        argv = ["--model", model, "--served-model-name", served_model_name]
        argv += ["--host", host, "--port", str(port)]
        for name, (flag, _) in _VALUE_FLAGS.items():
            value = getattr(settings, name)
            if value is not None:
                argv += [flag, _flag_text(value)]
        if settings.serves_tools:
            argv.append(TOOL_CALLING_ON)
        if settings.prefix_caching is not None:
            argv.append(PREFIX_CACHING_ON if settings.prefix_caching else PREFIX_CACHING_OFF)
        if not settings.cuda_graphs:
            argv.append(CUDA_GRAPHS_OFF)
        if settings.async_scheduling is not None:
            argv.append(ASYNC_SCHEDULING_ON if settings.async_scheduling else ASYNC_SCHEDULING_OFF)
        if settings.media_inputs is not None:
            argv.append(MEDIA_ON if settings.media_inputs else MEDIA_OFF)
        if trace_dir is not None:
            profiler = {
                "profiler": "torch",
                "torch_profiler_dir": trace_dir,
                "max_iterations": TRACE_MAX_ITERATIONS,
                "torch_profiler_dump_cuda_time_total": False,
            }
            argv += [PROFILER_FLAG, json.dumps(profiler)]
        # The declared workload, not the checkpoint's own generation config, decides sampling.
        argv += ["--generation-config", "vllm"]
        for flag, value in settings.extra_args.items():
            if value is False:
                continue
            argv += [flag] if value is None or value is True else [flag, str(value)]
        return argv

    def engine_args_between(self, before: Settings, after: Settings) -> list[str]:
        """The ``--engine-arg`` texts that turn ``before`` into ``after``."""
        texts = []
        for name, (flag, _) in _VALUE_FLAGS.items():
            value = getattr(after, name)
            if value != getattr(before, name):
                shown = "none" if value is None else _flag_text(value)
                texts.append(f"{flag[2:]}={shown}")
        for flag, (name, given) in _SWITCHES.items():
            if getattr(after, name) == given and getattr(before, name) != given:
                texts.append(flag[2:])
        for flag, value in after.extra_args.items():
            if before.extra_args.get(flag) != value:
                texts.append(flag[2:] if value is True else f"{flag[2:]}={value}")
        return texts

    def with_engine_arg(self, settings: Settings, text: str) -> Settings:
        key, separator, value = text.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"--engine-arg {text!r} has no flag name")
        flag = _flag_name(key if key.startswith("-") else f"--{key}")
        _require_readable(flag)
        if flag in RESERVED_FLAGS:
            raise ValueError(f"{flag} is set by Tensward and cannot be overridden")
        if _is_secret(flag):
            raise ValueError(_SECRET_FLAG_NOTE.format(flag=flag))
        enabled = not separator or value.lower() != "false"  # a bare flag switches it on
        if flag in _FLAG_TO_FIELD:
            field, read = _FLAG_TO_FIELD[flag]
            try:
                if not value:
                    raise ValueError
                # `quantization=none` lets the engine pick the kernel (no --quantization flag).
                none = field == "quantization" and value.lower() == "none"
                change: dict[str, Any] = {field: None if none else read(value)}
                return dataclasses.replace(settings, **change)
            except ValueError:
                raise ValueError(
                    f"--engine-arg {text!r} needs a {read.__name__.lstrip('_')} value"
                ) from None
        if flag in _SWITCHES:
            field, given = _SWITCHES[flag]
            switch: dict[str, Any] = {field: given == enabled}
            return dataclasses.replace(settings, **switch)
        extra: bool | str = (
            value if separator and value.lower() not in ("true", "false") else enabled
        )
        return dataclasses.replace(settings, extra_args={**settings.extra_args, flag: extra})

    def parse_setup(self, text: str) -> ParsedSetup:
        if re.search(r"^\s*services\s*:", text, re.MULTILINE):
            raise ValueError("a compose file is not supported; pass the `docker run` command")
        env, tokens = _launch_tokens(text)
        notes: list[str] = []
        image, mounts = None, []
        gpus: tuple[str, ...] = ()
        if _is_container_run(tokens):
            image, tokens, options = _split_docker_run(tokens)
            ignored = []
            for flag, value in options:
                if flag in ("-e", "--env"):
                    name, _, given = value.partition("=")
                    env[name] = given
                elif flag in ("-v", "--volume"):
                    mounts.append(value.split(":")[:2])
                elif flag == "--gpus":
                    gpus = _docker_gpu_devices(value)
                else:
                    ignored.append(flag)
            if ignored:
                notes.append(f"ignored docker options (Tensward runs it): {', '.join(ignored)}")
        text = shlex.join(_without_secrets(shlex.split(re.sub(r"\\\r?\n", " ", text))))
        flags, model = _flags(_without_launcher(tokens, required=image is None))
        settings = Settings()
        for flag, argument in flags:
            if flag == "--config":
                raise ValueError("--config names a file Tensward cannot read; pass its flags")
            if flag == "--model":
                model = argument
            elif flag in RESERVED_FLAGS:
                notes.append(f"dropped {flag}: Tensward sets it")
            elif _is_secret(flag):
                notes.append("dropped " + _SECRET_FLAG_NOTE.format(flag=flag))
            else:
                settings = self.with_engine_arg(
                    settings, flag if argument is None else f"{flag}={argument}"
                )
        if not gpus and env.get(VISIBLE_DEVICES):
            gpus = _device_list(env[VISIBLE_DEVICES])
        kept = self.inherited_env(env)
        if self.api_key_env in env:
            notes.append(f"dropped {self.api_key_env}: Tensward sets it")
        if others := sorted(set(env) - set(kept) - {self.api_key_env, VISIBLE_DEVICES}):
            notes.append(
                f"ignored environment (not engine settings, or credentials): {', '.join(others)}"
            )
        if model is not None:
            for source, destination in mounts:  # a path inside the container -> the host path
                if model == destination or model.startswith(destination.rstrip("/") + "/"):
                    model = source + model[len(destination) :]
        settings = dataclasses.replace(settings, extra_env=kept)
        return ParsedSetup(settings, model, image, tuple(notes), text, gpus)

    def recognizes(self, command: str) -> bool:
        try:
            _, tokens = _launch_tokens(command)
            if _is_container_run(tokens):
                return "vllm" in _split_docker_run(tokens)[0]
            _without_launcher(tokens, required=True)
        except ValueError:
            return False
        return True

    def availability(self, runtime_kind: str, image_or_command: str) -> Availability:
        if runtime_kind == "docker":
            return docker_availability(image_or_command, _image_version)
        found = executable(image_or_command)
        if found is None:
            return Availability(
                False, None,
                f"install vLLM (`pip install vllm=={VLLM_VERSION}`), or use --runtime docker",
                "command_missing",
            )  # fmt: skip
        return Availability(True, _installed_version(found), "")

    def inherited_env(self, environ: Mapping[str, str]) -> dict[str, str]:
        return {
            n: v
            for n, v in environ.items()
            if n.startswith(ENGINE_ENV_PREFIX) and not _is_secret(n)
        }

    def quantization_family(self, name: str) -> str:
        return QUANTIZATION_FAMILIES.get(name, name)

    def max_graph_batch(self, settings: Settings) -> int | None:
        return _coverage(settings)[1]

    def consistent(self, before: Settings, after: Settings) -> tuple[Settings, str | None]:
        after, graphs = _widen_graphs(before, after)
        notes = [graphs] if graphs else []
        method, _ = _speculation(after)
        if method == "ngram" and after.async_scheduling is True:
            after = dataclasses.replace(after, async_scheduling=False)
            notes.append(
                "turns async scheduling off, which this n-gram drafter needs; "
                "compare TPOT and throughput"
            )
        return after, "; ".join(notes) or None

    def unstartable(self, current: Settings, applied: Settings) -> str | None:
        pinned, covered = _coverage(applied)
        was_pinned, was = _coverage(current)
        if not pinned:
            return None
        method, k = _speculation(applied)
        drafting = f"{_METHOD_NAMES.get(method, method)} drafting"
        if covered is None:
            if was_pinned and was is None:  # the current setup already fails this way
                return None
            top = max(_pinned_sizes(applied) or [0])
            return (
                f"the pinned CUDA graph sizes (max {top}) are smaller than the "
                f"{1 + k}-token decode steps of {drafting}"
            )
        batch = [n for n in (was, applied.max_concurrent_requests) if n is not None]
        if k and was is not None and covered < min(batch):
            return (
                f"with {drafting}, the pinned CUDA graph sizes cover only {covered} "
                f"sequences instead of {was}; larger batches would run without graphs"
            )
        return None

    def default_tool_parser(self, model_dir: Path) -> str | None:
        try:
            config = json.loads((model_dir / "config.json").read_text("utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(config, dict):
            return None
        model_type = config.get("model_type")
        if model_type == "llama" and config.get("vocab_size") == LLAMA3_VOCAB_SIZE:
            return "llama3_json"
        return _TOOL_PARSERS.get(model_type) if isinstance(model_type, str) else None

    def parallel_degree(self, settings: Settings) -> int:
        degree = 1
        for flag in PARALLEL_FLAGS:
            try:
                degree *= int(settings.extra_args.get(flag) or 1)
            except ValueError:
                pass  # not a number: vLLM refuses to start, and one GPU is the safe reading
        return degree

    def kv_in_flight_tokens(self, settings: Settings, takes_images: bool) -> int:
        batch = settings.prefill_batch_tokens or DEFAULT_PREFILL_BATCH_TOKENS
        if takes_images and settings.media_inputs is not False:
            batch = max(batch, MEDIA_ITEM_BATCH_TOKENS)
        return MAX_CONCURRENT_BATCHES * batch

    def playbook(self) -> tuple[Entry, ...]:
        from .vllm_playbook import PLAYBOOK

        return PLAYBOOK

    def parse_signals(self, metrics_text: str) -> EngineSignals:
        values = _prometheus_values(metrics_text)
        cache = _info_labels(metrics_text, CACHE_CONFIG_FAMILY)

        def read(family: str) -> float | None:
            return values.get(family)

        def label(name: str) -> float | None:
            try:
                return float(cache[name])
            except (KeyError, ValueError):
                return None  # absent, or "None"

        return EngineSignals(
            kv_usage=read(KV_USAGE_FAMILY),
            running=read(RUNNING_FAMILY),
            waiting=read(WAITING_FAMILY),
            preemptions=read(PREEMPTIONS_FAMILY),
            prefix_cache_hits=read(PREFIX_HITS_FAMILY),
            prefix_cache_queries=read(PREFIX_QUERIES_FAMILY),
            prompt_tokens=read(PROMPT_TOKENS_FAMILY),
            prompt_tokens_computed=_labelled_value(
                metrics_text, PROMPT_BY_SOURCE_FAMILY, source="local_compute"
            ),
            generation_tokens=read(GENERATION_TOKENS_FAMILY),
            iterations=read(ITERATIONS_FAMILY),
            kv_capacity_tokens=label(KV_CAPACITY_LABEL),
            kv_max_concurrency=label(KV_MAX_CONCURRENCY_LABEL),
            queue_seconds=read(QUEUE_TIME_FAMILY),
            prefill_seconds=read(PREFILL_TIME_FAMILY),
            spec_drafts=read(SPEC_DRAFTS_FAMILY),
            spec_accepted_tokens=read(SPEC_ACCEPTED_FAMILY),
        )

    async def start_trace(self, client: httpx.AsyncClient, server_url: str) -> None:
        (await client.post(f"{server_url}/start_profile")).raise_for_status()

    async def stop_trace(self, client: httpx.AsyncClient, server_url: str) -> None:
        response = await client.post(f"{server_url}/stop_profile", timeout=TRACE_STOP_TIMEOUT_S)
        response.raise_for_status()

    def collect_trace(self, trace_dir: Path) -> list[Path]:
        return sorted(trace_dir.glob(TRACE_GLOB))

    def parse_quant_kernels(self, log_text: str) -> tuple[QuantKernel, ...]:
        found = {
            (name, layer): QuantKernel(name, layer, name in _SLOW_KERNELS)
            for name, layer in _KERNEL_LINE.findall(log_text)
        }
        for backend, quantization in _MOE_BACKEND.findall(log_text):
            found[backend, f"{quantization} MoE"] = QuantKernel(
                backend, f"{quantization} MoE", slow=False
            )
        if _AWQ_FALLBACK in log_text:
            found["awq", "AutoAWQLinearMethod"] = QuantKernel(
                "unoptimized AWQ kernels", "AutoAWQLinearMethod", slow=True
            )
        return tuple(found.values())


def _unescape(match: re.Match[str]) -> str:
    return "\n" if match[1] == "n" else match[1]  # the exposition escapes \\, \" and \n


def _samples(text: str) -> Iterator[tuple[str, dict[str, str], float]]:
    """(name, labels, value) of each sample line in a Prometheus exposition. Label values are
    quoted strings that may hold any character, including braces and escaped quotes; a line
    whose label set is not closed is skipped."""
    for line in text.splitlines():
        name = _NAME.match(line)
        if not name:
            continue
        position, labels = name.end(), {}
        if line.startswith("{", position):
            position += 1
            while not line.startswith("}", position):
                label = _LABEL.match(line, position)
                if not label:
                    break
                labels[label[1]] = _ESCAPE.sub(_unescape, label[2])
                position = label.end()
            if not line.startswith("}", position):
                continue
            position += 1
        value = _VALUE.match(line, position)
        if value:
            yield name[0], labels, float(value[1])


def _prometheus_values(text: str) -> dict[str, float]:
    """Each metric in a Prometheus exposition that has exactly one finite sample, by name."""
    samples: dict[str, list[float]] = {}
    for name, _, value in _samples(text):
        if math.isfinite(value):
            samples.setdefault(name, []).append(value)
    return {name: values[0] for name, values in samples.items() if len(values) == 1}


def _labelled_value(text: str, family: str, **labels: str) -> float | None:
    """The finite value of the one sample of ``family`` carrying these labels, else None."""
    found = [
        value
        for name, have, value in _samples(text)
        if name == family
        and math.isfinite(value)
        and all(have.get(k) == v for k, v in labels.items())
    ]
    return found[0] if len(found) == 1 else None


def _info_labels(text: str, family: str) -> dict[str, str]:
    """The labels of an info gauge's one sample; empty if it is absent or there are several."""
    found = [labels for name, labels, _ in _samples(text) if name == family]
    return found[0] if len(found) == 1 else {}


VLLM = VllmEngine()
