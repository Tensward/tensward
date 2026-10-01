"""vLLM: the first engine. Everything specific to vLLM 0.30 lives here.

That is its flag names, image, API-key variable, endpoints and Prometheus metric names.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import shlex
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import httpx

from .protocol import EngineSignals, ParsedSetup, QuantKernel, Settings


def _count(text: str) -> int:
    """An integer that may carry vLLM's k/m/g suffix: lower case is 1000s, upper case 1024s."""
    power = {"k": 1, "m": 2, "g": 3}.get(text[-1:].lower())
    if power is None:
        return int(text)
    return int(float(text[:-1]) * (1024 if text[-1].isupper() else 1000) ** power)


def _context_length(text: str) -> int | None:
    """--max-model-len: ``auto`` and ``-1`` both ask for the model's own length, i.e. unset."""
    return None if text in ("auto", "-1") else _count(text)


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
}
_FLAG_TO_FIELD = {flag: (name, read) for name, (flag, read) in _VALUE_FLAGS.items()}
PREFIX_CACHING_ON = "--enable-prefix-caching"
PREFIX_CACHING_OFF = "--no-enable-prefix-caching"
CUDA_GRAPHS_OFF = "--enforce-eager"
TOOL_CALLING_ON = "--enable-auto-tool-choice"
ASYNC_SCHEDULING_ON = "--async-scheduling"  # on by default in 0.30 unless incompatible
ASYNC_SCHEDULING_OFF = "--no-async-scheduling"
# Short spellings (from `vllm serve --help=all` at 0.30.0).
_ALIASES = {"-q": "--quantization", "-asc": "--api-server-count"}
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
}
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
# and Qwen2.5 use the Hermes format in their chat template.
_TOOL_PARSERS = {"qwen2": "hermes", "mistral": "mistral"}
LLAMA3_VOCAB_SIZE = 128256  # model_type "llama" covers Llama 2 too; Llama 3.x has this vocabulary
# Owned by Tensward (the runtime sets them), so an operator cannot override them.
RESERVED_FLAGS = frozenset({"--host", "--port", "--served-model-name", "--api-key", "--model"})
_MULTI_VALUE_FLAGS = frozenset({"--served-model-name", "--api-key"})  # nargs="+" in vLLM
# Only environment that changes the engine's behaviour is kept from a customer's command.
ENGINE_ENV_PREFIX = "VLLM_"
CUDA_DEVICES_ENV = "CUDA_VISIBLE_DEVICES"  # the devices a command runs on, recorded as ``gpus``
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
# A flag or variable whose name has one of these words (`--hf-token`, `HF_TOKEN`,
# `AWS_SECRET_ACCESS_KEY`) carries a secret. Whole words, so `--max-num-batched-tokens` is not one.
_SECRET_WORDS = frozenset({"key", "apikey", "token", "secret", "password", "passwd", "auth"})
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
# Kernels vLLM 0.30.0 ranks after MarlinLinearKernel for CUDA weight-only quantization
# (_POSSIBLE_KERNELS in kernels/linear/__init__.py): Marlin is chosen first when it fits.
_SLOW_KERNELS = frozenset({"ConchLinearKernel", "ExllamaLinearKernel", "TritonW4A16LinearKernel"})

_SAMPLE_LINE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+"
    r"(?P<value>[-+]?(?:\d[\d.]*(?:[eE][-+]?\d+)?|Inf|NaN))"
)

PROMPT_TOKENS_FAMILY = "vllm:prompt_tokens_total"
GENERATION_TOKENS_FAMILY = "vllm:generation_tokens_total"
KV_USAGE_FAMILY = "vllm:kv_cache_usage_perc"
RUNNING_FAMILY = "vllm:num_requests_running"
WAITING_FAMILY = "vllm:num_requests_waiting"
PREEMPTIONS_FAMILY = "vllm:num_preemptions_total"
PREFIX_HITS_FAMILY = "vllm:prefix_cache_hits_total"  # prompt TOKENS served from cache (not blocks)
PREFIX_QUERIES_FAMILY = "vllm:prefix_cache_queries_total"


def _flag_name(name: str) -> str:
    """A flag in its canonical spelling: vLLM accepts `--foo_bar` for `--foo-bar`."""
    if name in _ALIASES:
        return _ALIASES[name]
    return "--" + name[2:].replace("_", "-") if name.startswith("--") else name


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
    return not _SECRET_WORDS.isdisjoint(re.split(r"[-_]+", name.lower()))


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


class VllmEngine:
    name = "vllm"
    default_image = "vllm/vllm-openai:v0.30.0"  # pinned: the release this adapter is validated on
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
    # Read from the v0.30.0 source: CacheConfig and SchedulerConfig defaults, and
    # EngineArgs.get_batch_defaults (the OpenAI server column) in engine/arg_utils.py.
    defaults: Mapping[str, str] = {
        "prefix_caching": "on (for generative models)",
        "prefill_batch_tokens": (
            "2048 (8192 on GPUs with 70 GiB or more that are not A100, 16384 from 160 GiB), "
            "with chunked prefill on"
        ),
        "max_concurrent_requests": "256 (1024 on GPUs with 70 GiB or more that are not A100)",
        "kv_memory_fraction": "0.92",
        "async_scheduling": "on, unless the setup is incompatible with it",
        "cuda_graphs": "on",
    }

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
                argv += [flag, f"{value:g}" if isinstance(value, float) else str(value)]
        if settings.serves_tools:
            argv.append(TOOL_CALLING_ON)
        if settings.prefix_caching is not None:
            argv.append(PREFIX_CACHING_ON if settings.prefix_caching else PREFIX_CACHING_OFF)
        if not settings.cuda_graphs:
            argv.append(CUDA_GRAPHS_OFF)
        if settings.async_scheduling is not None:
            argv.append(ASYNC_SCHEDULING_ON if settings.async_scheduling else ASYNC_SCHEDULING_OFF)
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
                shown = (
                    "none" if value is None else f"{value:g}" if isinstance(value, float) else value
                )
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
        tokens = shlex.split(re.sub(r"\\\r?\n", " ", text))
        env: dict[str, str] = {}
        while tokens and _ASSIGNMENT.match(tokens[0]):  # VAR=value vllm serve ...
            name, _, value = tokens.pop(0).partition("=")
            env[name] = value
        notes: list[str] = []
        image, mounts = None, []
        gpus: tuple[str, ...] = ()
        if tokens[:1] == ["sudo"]:
            tokens = tokens[1:]
        if tokens and os.path.basename(tokens[0]) in ("docker", "podman"):
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
        if not gpus and env.get(CUDA_DEVICES_ENV):
            gpus = _device_list(env[CUDA_DEVICES_ENV])
        kept = {
            n: v for n, v in env.items() if n.startswith(ENGINE_ENV_PREFIX) and not _is_secret(n)
        }
        if self.api_key_env in env:
            notes.append(f"dropped {self.api_key_env}: Tensward sets it")
        if others := sorted(set(env) - set(kept) - {self.api_key_env, CUDA_DEVICES_ENV}):
            notes.append(
                f"ignored environment (not engine settings, or credentials): {', '.join(others)}"
            )
        if model is not None:
            for source, destination in mounts:  # a path inside the container -> the host path
                if model == destination or model.startswith(destination.rstrip("/") + "/"):
                    model = source + model[len(destination) :]
        settings = dataclasses.replace(settings, extra_env=kept)
        return ParsedSetup(settings, model, image, tuple(notes), text, gpus)

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

    def parse_signals(self, metrics_text: str) -> EngineSignals:
        values = _prometheus_values(metrics_text)

        def read(family: str) -> float | None:
            return values.get(family)

        return EngineSignals(
            kv_usage=read(KV_USAGE_FAMILY),
            running=read(RUNNING_FAMILY),
            waiting=read(WAITING_FAMILY),
            preemptions=read(PREEMPTIONS_FAMILY),
            prefix_cache_hits=read(PREFIX_HITS_FAMILY),
            prefix_cache_queries=read(PREFIX_QUERIES_FAMILY),
            prompt_tokens=read(PROMPT_TOKENS_FAMILY),
            generation_tokens=read(GENERATION_TOKENS_FAMILY),
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
        if _AWQ_FALLBACK in log_text:
            found["awq", "AutoAWQLinearMethod"] = QuantKernel(
                "unoptimized AWQ kernels", "AutoAWQLinearMethod", slow=True
            )
        return tuple(found.values())


def _prometheus_values(text: str) -> dict[str, float]:
    """Each metric in a Prometheus exposition that has exactly one finite sample, by name."""
    samples: dict[str, list[float]] = {}
    for line in text.splitlines():
        if match := _SAMPLE_LINE.match(line):
            value = float(match["value"])
            if math.isfinite(value):
                samples.setdefault(match["name"], []).append(value)
    return {name: values[0] for name, values in samples.items() if len(values) == 1}


VLLM = VllmEngine()
