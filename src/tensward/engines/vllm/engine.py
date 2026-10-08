"""vLLM: the first engine. Everything specific to vLLM 0.30 lives in this package.

That is its flag names, image, API-key variable, endpoints and Prometheus metric names.
"""

from __future__ import annotations

import dataclasses
import json
import re
import shlex
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Sequence

import httpx

from ...capabilities import Capabilities, Resolved
from ...capture import CapturedResponse
from ...classify import EngineRule
from ...client import RequestRecord
from ...platforms.nvidia import VISIBLE_DEVICES
from ...playbook import Entry, Gate, Situation, load_entries
from ...probes import executable
from ...secrets import is_secret, without_secrets
from ...settings import Availability, ParsedSetup, QuantKernel, Settings
from ...signal_source import PrometheusSource, SignalSource
from ...workload import WorkloadSpec
from ..protocol import Failed, NotYet, Ready, ReadyProbe, Tracing, docker_availability
from .capabilities import VLLM_CAPABILITIES, read_lever, with_api_servers, write_lever
from .command import (
    _FLAG_TO_FIELD,
    _MULTI_VALUE_FLAGS,
    _SECRET_FLAG_NOTE,
    _SWITCHES,
    _TOOL_PARSERS,
    _VALUE_FLAGS,
    ASYNC_SCHEDULING_OFF,
    ASYNC_SCHEDULING_ON,
    CUDA_GRAPHS_OFF,
    DEFAULT_KV_MEMORY_FRACTION,
    DEFAULT_PREFILL_BATCH_TOKENS,
    ENGINE_ENV_PREFIX,
    LLAMA3_VOCAB_SIZE,
    MAX_CONCURRENT_BATCHES,
    MEDIA_ITEM_BATCH_TOKENS,
    MEDIA_OFF,
    MEDIA_ON,
    MEMORY_OVERHEAD_BYTES,
    PARALLEL_FLAGS,
    PREFIX_CACHING_OFF,
    PREFIX_CACHING_ON,
    PROFILER_FLAG,
    QUANTIZATION_FAMILIES,
    RESERVED_FLAGS,
    TOOL_CALLING_ON,
    TRACE_GLOB,
    TRACE_MAX_ITERATIONS,
    TRACE_STOP_TIMEOUT_S,
    TRUST_REMOTE_CODE_FLAG,
    VLLM_VERSION,
    _device_list,
    _docker_gpu_devices,
    _flag_name,
    _flag_text,
    _flags,
    _image_version,
    _installed_version,
    _is_container_run,
    _launch_tokens,
    _require_readable,
    _split_docker_run,
    _without_launcher,
    data_parallel_size,
    torch_mismatch_warning,
)
from .graphs import (
    _METHOD_NAMES,
    _coverage,
    _pinned_sizes,
    _speculation,
    _widen_graphs,
)
from .metrics import VLLM_SIGNALS
from .predicates import HOLD_BACKS, PREDICATES
from .rules import ENGINE_RULES

# Startup log lines that name the quantized-linear kernel chosen for a layer type, e.g.
# "Using MarlinLinearKernel for AutoAWQMarlinLinearMethod" (AWQ, GPTQ, compressed-tensors
# weight-only) and "Selected CutlassFP8ScaledMMLinearKernel for Fp8LinearMethod" (FP8, int8
# W8A8). The wording is from vLLM 0.30.0's source (model_executor/kernels/linear and
# layers/quantization); it has not been checked against a running server.
_KERNEL_LINE = re.compile(r"(?:Using|Selected) (\w+Kernel) for (\w+)")
# The AWQ layer that cannot use Marlin says so once per layer.
_AWQ_FALLBACK = "Falling back to unoptimized AWQ kernels"
# The MoE layers name their backend once, quoted or not, with or without a quantization:
# "Using 'MARLIN' WNA16 MoE backend.", "Using FlashInfer CUTLASS Fp8 MoE backend out of
# potential backends: [...]", "Using TritonExperts MoE backend" (vLLM f0c44cc,
# model_executor/layers/fused_moe/oracle/*.py and quantization/utils/humming/moe.py; quoted
# MARLIN seen on L4 and A10G with Gemma 4 26B-A4B AWQ, 2026-10-01). The backend is one or two
# word tokens (never a newline, since the pattern runs over the whole log; the second word is
# taken only when the quantization cannot follow the first, so "TRITON Fp8" splits) and the
# quantization words are listed longest first.
_MOE_BACKEND = re.compile(
    r"Using '?(?P<backend>\w+(?: \w+)??)'? (?:(?P<quant>W4A8 Int8|W4A8|Fp8|Int8|WNA16|Mxfp4|MxFp8"
    r"|NvFp4|Unquantized|Humming) )?MoE backend"
)
# Kernels vLLM 0.30.0 ranks after MarlinLinearKernel for CUDA weight-only quantization
# (_POSSIBLE_KERNELS in kernels/linear/__init__.py): Marlin is chosen first when it fits.
_SLOW_KERNELS = frozenset({"ConchLinearKernel", "ExllamaLinearKernel", "TritonW4A16LinearKernel"})


class VllmTracing:
    # Adds "gpu_model_runner: forward/sample/..." and "schedule: ..." annotations to the trace
    # (vllm/v1/utils.py record_function_or_nullcontext).
    env: Mapping[str, str] = {"VLLM_CUSTOM_SCOPES_FOR_PROFILING": "1"}
    # CUDA-profiler mode: /start_profile and /stop_profile then only open and close the window
    # that `ncu --profile-from-start off` profiles, so startup kernels are never counted.
    counters_args: Sequence[str] = (PROFILER_FLAG, json.dumps({"profiler": "cuda"}))
    step_scope = "gpu_model_runner: forward"
    # Both model runners annotate each step while the torch profiler runs (vLLM
    # v1/worker/gpu_worker.py, f0c44cc); Model Runner V2, the default from v0.29, writes no
    # "gpu_model_runner: forward". A step that serves no token is not counted: the
    # default runner skips it (v1/worker/gpu/model_runner.py, f0c44cc) and the V1 runner runs
    # only a dummy forward for it, under the external launcher with data parallelism
    # (gpu_model_runner.py, f0c44cc).
    step_pattern = (
        r"^execute_(?!0_|context_0\(0\)_generation_0\()"
        r"(?:\d+_)?context_\d+\(.*\)_generation_\d+\("
    )

    async def start(self, client: httpx.AsyncClient, server_url: str) -> None:
        (await client.post(f"{server_url}/start_profile")).raise_for_status()

    async def stop(self, client: httpx.AsyncClient, server_url: str) -> None:
        response = await client.post(f"{server_url}/stop_profile", timeout=TRACE_STOP_TIMEOUT_S)
        response.raise_for_status()

    def collect(self, trace_dir: Path) -> list[Path]:
        return sorted(trace_dir.glob(TRACE_GLOB))


class VllmProbe:
    """Health 200, then the served name in the model list."""

    timeout_s = 5.0

    def __init__(self, health_path: str, models_path: str) -> None:
        self.health_path, self.models_path = health_path, models_path
        self.healthy = False

    async def check(
        self, client: httpx.AsyncClient, url: str, served_name: str
    ) -> Ready | NotYet | Failed:
        try:
            if not self.healthy:
                self.healthy = (await client.get(f"{url}{self.health_path}")).status_code == 200
            if self.healthy:
                models = await client.get(f"{url}{self.models_path}")
                if models.status_code < 400:
                    served = [entry["id"] for entry in models.json()["data"]]
                    if served_name in served:
                        return Ready()
                    return Failed(f"the server did not serve {served_name!r}; it serves {served}")
                if models.status_code in (401, 403):
                    return Failed("the server rejected this run's API key")
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            self.healthy = False  # still loading, or not an answer we can read yet
        return NotYet()


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
    signals = VLLM_SIGNALS
    # Every vLLM family carries an `engine` label: one value per data-parallel engine.
    signal_source: SignalSource = PrometheusSource(metrics_path, VLLM_SIGNALS, ("engine",))
    # Quantized-linear kernel (as the log names it) -> its name in CUDA kernel symbols.
    quant_kernel_symbols: Mapping[str, str] = {"MarlinLinearKernel": "marlin"}
    # vLLM starts each log line with its process, such as "(APIServer pid=426)".
    log_prefix: re.Pattern[str] | None = re.compile(r"^\(\w+ pid=\d+\)")
    tracing: Tracing | None = VllmTracing()
    gpu_memory_log: str = r"Free memory on device \([\d.]+/([\d.]+) GiB\)"
    kv_memory_log: str = r"Available KV cache memory: ([\d.]+) GiB"
    version_log: str = r"Initializing a V1 LLM engine \(v([^)\s]+)\)"
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
    family = "vllm"
    predicates: Mapping[str, Callable[..., Any] | Gate] = PREDICATES
    hold_backs: Mapping[str, Callable[[Situation], str | None]] = HOLD_BACKS
    memory_overhead_bytes = MEMORY_OVERHEAD_BYTES

    added_flags = (
        "--generation-config vllm (sampling follows the declared workload, not the checkpoint's "
        "generation_config.json), and the host, port, served model name and API key"
    )
    default_profile: str | None = "vllm-nvidia"
    engine_rules: tuple[EngineRule, ...] = ENGINE_RULES

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
        if is_secret(flag):
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
        text = shlex.join(
            without_secrets(
                shlex.split(re.sub(r"\\\r?\n", " ", text)),
                canonical=_flag_name,
                multi_value_flags=_MULTI_VALUE_FLAGS,
            )
        )
        flags, model = _flags(_without_launcher(tokens, required=image is None))
        settings = Settings()
        for flag, argument in flags:
            if flag == "--config":
                raise ValueError("--config names a file Tensward cannot read; pass its flags")
            if flag == "--model":
                model = argument
            elif flag in RESERVED_FLAGS:
                notes.append(f"dropped {flag}: Tensward sets it")
            elif is_secret(flag):
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
            n: v for n, v in environ.items() if n.startswith(ENGINE_ENV_PREFIX) and not is_secret(n)
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

    def lever_value(self, settings: Settings, lever: str) -> Any:
        return read_lever(settings, lever)

    def with_lever(self, settings: Settings, lever: str, value: Any) -> Settings:
        return write_lever(settings, lever, value)

    def playbook(self, *, version: str | None = None) -> tuple[Entry, ...]:
        return load_entries(self, version, family=self.family)

    async def count_prompt_tokens(
        self, client: httpx.AsyncClient, server_url: str, model: str, body: Mapping[str, Any]
    ) -> int | None:
        ids = await self.prompt_token_ids(client, server_url, model, body)
        return None if ids is None else len(ids)

    async def prompt_token_ids(
        self, client: httpx.AsyncClient, server_url: str, model: str, body: Mapping[str, Any]
    ) -> tuple[int, ...] | None:
        """The prompt's token ids as the server renders it (a chat body through the model's chat
        template, tools included) and tokenizes it; None if it cannot say."""
        try:
            response = await client.post(f"{server_url}/tokenize", json={"model": model, **body})
            response.raise_for_status()
            return tuple(int(token) for token in response.json()["tokens"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return None

    def request_extras(self, workload: WorkloadSpec) -> dict[str, Any]:
        if workload.structured_output is None:
            return {}
        return {"structured_outputs": workload.structured_output}

    def tool_calling_fix(self, settings: Settings, source: str) -> str | None:
        if settings.serves_tools:
            return None
        fixed = dataclasses.replace(
            settings, tool_calling=True, tool_parser=settings.tool_parser or "<name>"
        )
        flags = " ".join(f"--engine-arg {arg}" for arg in self.engine_args_between(settings, fixed))
        if settings.tool_parser is None:
            return (
                "the workload offers tools but no tool-call parser is known for this "
                "checkpoint's model family, so the server cannot return tool calls; name the "
                f"parser with `{flags}`"
            )
        serving = f"{TOOL_CALLING_ON} --tool-call-parser {settings.tool_parser}"
        if source == "command":
            fix = (
                f"add `{serving}` to your --current command and run `tensward init` again with "
                "it as a new project (the command overrides the configuration's tool_calling)"
            )
        else:
            fix = (
                f"set `tool_calling: true` in the serving configuration's case (vLLM then runs "
                f"with `{serving}`)"
            )
        return (
            "the workload offers tools but tool calling is not enabled, so the server refuses "
            f"those requests; {fix}, or for one analyse run add `{flags}`"
        )

    def reproducibility_advice(self, source: str) -> str:
        if source == "command":
            return (
                "Put `VLLM_BATCH_INVARIANT=1` (vLLM's batch-invariant mode: beta, compute "
                "capability 8.0 or higher, slower, and it does not support prefix caching yet, "
                "vLLM issue #27433) and `--no-enable-prefix-caching` (vLLM enables prefix "
                "caching by default) in your `--current` command, for example "
                "`VLLM_BATCH_INVARIANT=1 vllm serve … --no-enable-prefix-caching`. Run "
                "`tensward init` again, because that changes the registration, then analyse "
                "your current setup again."
            )
        return (
            "Register your setup from a `--current` command that sets `VLLM_BATCH_INVARIANT=1` "
            "(vLLM's batch-invariant mode: beta, compute capability 8.0 or higher, slower, and "
            "it does not support prefix caching yet, vLLM issue #27433) and "
            "`--no-enable-prefix-caching` (vLLM enables prefix caching by default), for example "
            "`VLLM_BATCH_INVARIANT=1 vllm serve … --no-enable-prefix-caching`: the "
            "configuration cannot set environment variables. Run `tensward init` again with "
            "it, then analyse your current setup again."
        )

    def loads_encoders(self, settings: Settings, modalities: Collection[str]) -> bool:
        """Whether the engine loads the vision and audio towers. vLLM v0.30 skips a tower once
        every modality it serves has a limit of 0, and ``--language-model-only`` sets them all
        to 0."""
        if settings.media_inputs is False:
            return False
        limits = settings.media_limits or {}
        return any(limits.get(kind) != 0 for kind in modalities if kind != "text")

    def parse_quant_kernels(self, log_text: str) -> tuple[QuantKernel, ...]:
        found = {
            (name, layer): QuantKernel(name, layer, name in _SLOW_KERNELS)
            for name, layer in _KERNEL_LINE.findall(log_text)
        }
        for line in _MOE_BACKEND.finditer(log_text):
            layer = f"{line['quant']} MoE" if line["quant"] else "MoE"
            found[line["backend"], layer] = QuantKernel(line["backend"], layer, slow=False)
        if _AWQ_FALLBACK in log_text:
            found["awq", "AutoAWQLinearMethod"] = QuantKernel(
                "unoptimized AWQ kernels", "AutoAWQLinearMethod", slow=True
            )
        return tuple(found.values())

    def capabilities(self, version: str | None, settings: Settings | None = None) -> Capabilities:
        if settings is None:
            return VLLM_CAPABILITIES
        return with_api_servers(self.frontend_processes(settings) or 1)

    def runs_remote_code(self, settings: Settings) -> bool:
        return settings.extra_args.get(TRUST_REMOTE_CODE_FLAG) not in (None, False, "false")

    def frontend_processes(self, settings: Settings) -> int | None:
        # Without --api-server-count, the internal load balancer (the default) starts one API
        # server per data-parallel engine, the full --data-parallel-size (entrypoints/cli/
        # serve.py:107-129, f0c44cc). Hybrid load balancing counts the local engines only, and
        # external load balancing or one port per engine starts one server; Tensward then
        # reports can't tell, which errs safe.
        if settings.api_server_count is not None:
            return settings.api_server_count
        return data_parallel_size(settings)

    def weights_on_device(self, settings: Settings, resolved: Resolved | None) -> bool | None:
        return None

    def host_warnings(self) -> tuple[str, ...]:
        warning = torch_mismatch_warning()
        return (warning,) if warning else ()

    async def resolved(
        self, client: httpx.AsyncClient, url: str, server_log: str
    ) -> Resolved | None:
        return None

    def effective(self, settings: Settings, resolved: Resolved | None) -> Settings:
        return settings

    def untimed(self, record: RequestRecord, response: CapturedResponse | None) -> str | None:
        return None

    def probe(self) -> ReadyProbe:
        return VllmProbe(self.health_path, self.models_path)

    def request_model(self, served_name: str) -> str:
        return served_name


VLLM = VllmEngine()
