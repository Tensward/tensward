"""llama.cpp: ``llama-server`` run as a local process. Everything specific to it lives in this
package: its flags, environment, endpoints and metric names."""

from __future__ import annotations

import dataclasses
import re
import shlex
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Collection, Iterable, Mapping

import httpx

from ...anatomy import ModelAnatomy
from ...capabilities import Capabilities, Resolved
from ...capture import CapturedResponse
from ...classify import EngineRule
from ...client import RequestRecord
from ...context import current as analysis
from ...kb_actions import Action
from ...levers import LEVERS, Speculation
from ...measurement import Measurement
from ...platforms import PLATFORMS, darwin, detect_platform
from ...platforms.cpu import HostFacts, host_facts
from ...playbook import Entry, Gate, Situation, WorkloadFacts, load_entries
from ...probes import PROBE_TIMEOUT_S, executable
from ...secrets import is_secret
from ...settings import Availability, ParsedSetup, QuantKernel, Settings
from ...signal_source import SignalSource
from ...workload import WorkloadSpec
from ..protocol import (
    Allocation,
    EngineLog,
    FlagRules,
    HostWindow,
    ProbeResult,
    ProbeRun,
    ReadyProbe,
    Tracing,
    inherited_with_prefix,
)
from .advice import advice, environment
from .capabilities import DRAFT_MTP, LLAMACPP_CAPABILITIES, NGRAM_MOD
from .command import (
    DEFAULT_BATCH,
    DEFAULT_UBATCH,
    ENGINE_ENV_PREFIX,
    LLAMA_SERVER,
    OPTIONAL_VALUE,
    Q8,
    apply_flag,
    known_build,
    launch_argv,
    parse_setup,
    probe_build,
    production_command,
    read_flag,
    runs_llama_server,
    split_kv,
)
from .flags import (
    FIT_FLAGS,
    FLAGS,
    LLAMACPP_FLAG_RULES,
    LOG_FLAGS,
    NEVER,
    NO_OP,
    PINNED,
    REFUSED_SOURCES,
    REMOVED,
    SECRETS,
    dashed,
    flag_for,
)
from .logreader import LLAMACPP_LOG
from .metrics import LLAMACPP_SIGNALS, LlamaCppSource
from .placement import (
    Placement,
    Topology,
    expert_mix,
    placement,
    placement_block,
    working_set_bound,
)
from .predicates import HOLD_BACKS, PREDICATES
from .prelaunch import fit_advice
from .probe import LlamaCppProbe
from .probes import (
    HOST_BANDWIDTH_FACT,
    PCIE_RATE_FACT,
    launched_bandwidth,
    model_fingerprint,
    prefill_work,
    probe_run,
)
from .rules import ENGINE_RULES
from .startup import LINE_PREFIX, Startup, read_startup

if TYPE_CHECKING:
    from ...project import CurrentSetup
    from ...runtime import Runtime

_LEVER_FLAGS = {
    "max_concurrent_requests": "--parallel",
    "max_context_len": "--ctx-size",
    "prefill_batch_tokens": "--batch-size",
    "prefix_caching": "--cache-prompt",
    "tool_calling": "--jinja",
    "host_layers": "--n-gpu-layers",
    "host_expert_layers": "--n-cpu-moe",
}
_FIELD_FLAGS = (
    ("max_concurrent_requests", "parallel"),
    ("max_context_len", "ctx-size"),
    ("prefill_batch_tokens", "batch-size"),
)
SPEC_TYPE = "--spec-type"
SPEC_DRAFT_MAX = "--spec-draft-n-max"
_KV_TYPE_FLAGS = ("--cache-type-k", "--cache-type-v")
_SERVE_AS_LOCAL_PROCESS = "Tensward runs llama-server as a local process; use --runtime local"
_NO_METAL_IN_CONTAINERS = "; Metal is not available in containers on macOS"
_QUARANTINE = "com.apple.quarantine"


def _with_speculation(settings: Settings, value: Speculation | None) -> Settings:
    """``settings`` speculating by ``value``'s method, which needs no draft model, or not at all
    for None. ``draft-mtp`` drafts ``value.draft_tokens`` tokens a step."""
    if value is not None and value.method not in (NGRAM_MOD, DRAFT_MTP):
        raise ValueError(f"Tensward does not write llama.cpp speculation by {value.method!r}")
    if (build := known_build()) is not None and not build[1].has(SPEC_TYPE):
        raise ValueError(f"not in this build's --help: {SPEC_TYPE}")
    extra = {k: v for k, v in settings.extra_args.items() if not k.startswith("--spec-")}
    if value is not None:
        extra[SPEC_TYPE] = value.method
    if value is not None and value.method == DRAFT_MTP and value.draft_tokens:
        extra[SPEC_DRAFT_MAX] = str(value.draft_tokens)
    return dataclasses.replace(settings, extra_args=extra)


def _device_off(extra_args: Mapping[str, Any]) -> bool:
    """Whether a setup's flags turn the accelerator off (``--device none``)."""
    return extra_args.get("--device") == "none"


def quarantined(command: str) -> str | None:
    """What to do when macOS stopped the downloaded binary ``command`` runs: it cannot be run and
    carries the quarantine attribute. None elsewhere and for a binary that runs."""
    if not darwin.on_mac():
        return None
    try:
        done = subprocess.run(  # noqa: S603 - the user's own command
            [*shlex.split(command), "--version"], capture_output=True, stdin=subprocess.DEVNULL,
            timeout=PROBE_TIMEOUT_S, check=False,
        )  # fmt: skip
        blocked = done.returncode < 0
    except OSError:
        blocked = True
    except (subprocess.SubprocessError, ValueError):
        return None
    program = executable(command)
    if not blocked or program is None or darwin.RUN(["xattr", "-p", _QUARANTINE, program]) is None:
        return None
    return (
        "macOS blocked this downloaded binary; remove the quarantine attribute with "
        f"`xattr -d {_QUARANTINE} {program}`; Tensward does not change it"
    )


_EFFECTIVE_FIELDS = (("max_concurrent_requests", "max_concurrent_requests"),)
_EFFECTIVE_FLAGS = (
    ("--kv-unified", "llamacpp.kv_unified"),
    ("--n-gpu-layers", "llamacpp.n_gpu_layers"),
    ("--n-cpu-moe", "host_expert_layers"),
)
_PINNED_FIELDS = (
    ("max_concurrent_requests", "max_concurrent_requests"),
    ("max_context_len", "llamacpp.kv_pool_tokens"),
)
_PINNED_FLAGS = _EFFECTIVE_FLAGS


def _drafts_anew(engine: LlamaCppEngine, settings: Settings, current: Settings) -> bool:
    """Whether ``settings`` speculate from the model's own MTP layers and ``current`` does not."""
    spec, before = (engine.lever_value(s, "speculation") for s in (settings, current))
    mtp = isinstance(spec, Speculation) and spec.method == DRAFT_MTP
    return mtp and not (isinstance(before, Speculation) and before.method == DRAFT_MTP)


def _host_window(window: HostWindow) -> dict[str, Any]:
    """The first and last host sample of the measured window."""
    return {
        "start": None if window.start is None else asdict(window.start),
        "end": None if window.end is None else asdict(window.end),
    }


def _precision(kind: str | None) -> str | None:
    if kind is None:
        return None
    return {"f16": "default", Q8: "8bit"}.get(kind)


def _unreadable_fit(resolved: Resolved) -> bool:
    """Whether ``--fit`` moved experts to the CPU but its summary line did not say how many
    layers, or it also left layers off the device: ``--n-cpu-moe`` counts from layer 0, so
    that placement cannot be written as a command."""
    levers = resolved.levers
    return "tensor_overrides" in resolved.facts.get("fit_changed", ()) and (
        "host_expert_layers" not in levers
        or bool(levers["host_expert_layers"] and levers.get("host_layers"))
    )


_EXPERT_PLACEMENT = ("--n-cpu-moe", "--cpu-moe", "--override-tensor")


def _host_expert_bytes(layers: int | None, anatomy: ModelAnatomy | None) -> int | None:
    """Routed-expert bytes of the first ``layers`` blocks, which ``--n-cpu-moe`` keeps in host
    memory (common/arg.cpp:2771)."""
    if not layers or anatomy is None or not anatomy.expert_layer_bytes:
        return None
    return sum(anatomy.expert_layer_bytes[:layers])


def _host_experts(started: Startup, settings: Settings) -> int | None:
    """Layers whose experts run from host memory: the command's ``--n-cpu-moe`` or
    ``--cpu-moe``, else what ``--fit`` left there."""
    if settings.extra_args.get("--cpu-moe") is True and started.total_layers:
        return started.total_layers - 1
    given = int(str(settings.extra_args.get("--n-cpu-moe") or 0))
    return given or started.fit_expert_layers


def resolution(
    started: Startup, props: Mapping[str, Any], settings: Settings, anatomy: ModelAnatomy | None
) -> Resolved | None:
    """What the server chose where the command left it free: the start-up log's figures and
    ``/props``. None when neither carries any."""
    slot_ctx = started.slot_ctx or props.get("default_generation_settings", {}).get("n_ctx")
    experts = _host_experts(started, settings)
    placed = started.on_device or "n_gpu_layers" in started.fit_changed
    levers: dict[str, Any] = {
        "max_concurrent_requests": props.get("total_slots") or started.slots,
        "max_context_len": slot_ctx,
        "host_layers": started.host_layers,
        "host_expert_layers": experts,
        "kv_cache_precision": _precision(started.type_k),
        "llamacpp.kv_pool_tokens": started.ctx_total,
        "llamacpp.flash_attn": started.flash_attn.split()[0] if started.flash_attn else None,
        "llamacpp.threads": started.threads,
        "llamacpp.threads_batch": started.threads_batch,
        "llamacpp.kv_unified": started.kv_unified,
        "llamacpp.n_gpu_layers": (started.offloaded_layers or 0)
        if placed and started.allocations
        else None,
    }
    facts: dict[str, Any] = {
        "fit_changed": list(started.fit_changed),
        "host_weight_bytes": started.bytes_of("model", "host") if started.allocations else None,
        "device_weight_bytes": started.bytes_of("model", "device") if started.allocations else None,
        "host_expert_bytes": _host_expert_bytes(experts, anatomy),
        "device_free_bytes": started.device_free_bytes,
        "kv_host_bytes": started.bytes_of("kv", "host") if started.allocations else None,
        "kv_device_bytes": started.bytes_of("kv", "device") if started.allocations else None,
        "working_set_bytes": started.working_set_bytes,
        "graph_splits": started.graph_splits,
        "metal": started.metal,
        "build_info": props.get("build_info"),
        "chat_template_caps": props.get("chat_template_caps"),
        "context_per_request": slot_ctx,
    }
    levers = {k: v for k, v in levers.items() if v is not None}
    facts = {k: v for k, v in facts.items() if v is not None}
    if not levers and not facts:
        return None
    resolved = Resolved(levers=levers, facts=facts)
    where = placement(anatomy, resolved) if anatomy is not None else None
    if where is not None and anatomy is not None:
        mix = expert_mix(anatomy, resolved)
        resolved = Resolved(
            levers=levers, facts={**facts, "placement": asdict(where), "expert_mix": mix}
        )
    return resolved


_METAL_ALLOCATION = (
    "Metal could not allocate a buffer: the Metal working set is the bound; a smaller -c or -np, "
    "or a larger unified-memory share (sudo sysctl iogpu.wired_limit_mb), gives it room"
)
# ggml-metal-device.m:2170, 2229 (ggml_metal_buffer_init) and 2256 (ggml_metal_buffer_map); the
# last key is the Metal runtime's text, printed by ggml-metal-context.m:273-275.


class LlamaCppEngine:
    name = "llamacpp"
    label = "llama.cpp"
    version_prefix = "build "
    formats = frozenset({"gguf"})
    platforms = frozenset({"nvidia", "apple", "cpu"})
    launchers = "`llama-server ...`"
    default_image = ""
    default_image_size = ""
    local_command: tuple[str, ...] = (LLAMA_SERVER,)
    api_key_env = "LLAMA_API_KEY"
    health_path = "/health"
    models_path = "/v1/models"
    metrics_path = "/metrics"
    signals = LLAMACPP_SIGNALS
    signal_source: SignalSource = LlamaCppSource(metrics_path, LLAMACPP_SIGNALS)
    quant_kernel_symbols: Mapping[str, str] = {}
    log_prefix: re.Pattern[str] | None = LINE_PREFIX
    tracing: Tracing | None = None
    log_checks: Mapping[str, str] = {
        "failed to find free space in the KV cache": (
            'llama-server ran out of KV cache during the run ("failed to find free space in the '
            'KV cache" in server.log): requests were retried in smaller batches; a larger -c '
            "or a smaller -np gives the cache room"
        ),
        "out of memory": (
            'the GPU ran out of memory ("out of memory" in server.log): lower -c, -np or -ub, '
            "or keep more layers on the CPU"
        ),
        "failed to load model": "llama-server failed to load the model (see server.log)",
        "ggml_metal_buffer_init: error: failed to allocate buffer": _METAL_ALLOCATION,
        "ggml_metal_buffer_map: error: failed to allocate buffer": _METAL_ALLOCATION,
        "kIOGPUCommandBufferCallbackErrorOutOfMemory": _METAL_ALLOCATION,
    }
    gpu_memory_log = r"(?!x)x"
    kv_memory_log = r"(?!x)x"
    dtype_cast = r"(?!x)x"
    version_log = r"\bbuild ([1-9]\d*) \("
    defaults: Mapping[str, str] = {
        "max_concurrent_requests": "4 slots, unified KV (auto)",
        "max_context_len": "the model's training context, cut by --fit to what fits",
        "prefill_batch_tokens": f"{DEFAULT_BATCH} (-b), ubatch {DEFAULT_UBATCH}",
    }
    default_kv_memory_fraction: float | None = None
    default_runtime = "local"
    log: EngineLog = LLAMACPP_LOG
    family = "llamacpp"
    predicates: Mapping[str, Callable[..., Any] | Gate] = PREDICATES
    hold_backs: Mapping[str, Callable[[Situation], str | None]] = HOLD_BACKS
    memory_overhead_bytes: int | None = None
    added_flags = (
        "--host, --port, --metrics, --slots, --alias, -lv 4 and, on every step after the "
        "baseline, --fit off with the baseline's resolved --ctx-size, --parallel, "
        "--n-gpu-layers, --n-cpu-moe and --kv-unified or --no-kv-unified"
    )
    default_profile: str | None = None
    engine_rules: tuple[EngineRule, ...] = ENGINE_RULES
    flag_rules: FlagRules = LLAMACPP_FLAG_RULES
    value_actions: Mapping[str, Action] = {}

    def launch_argv(
        self,
        settings: Settings,
        *,
        model: str,
        served_model_name: str,
        host: str,
        port: int,
        trace_dir: str | None = None,
        serving: bool = False,
    ) -> list[str]:
        return launch_argv(
            settings, model=model, served_model_name=served_model_name, host=host, port=port,
            serving=serving,
        )  # fmt: skip

    def with_engine_arg(self, settings: Settings, text: str) -> Settings:
        key, separator, value = text.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"--engine-arg {text!r} has no flag name")
        spelling = dashed(key)
        if spelling.startswith("--"):
            spelling = "--" + spelling[2:].replace("_", "-")
        if spelling in REMOVED:
            raise ValueError(f"{spelling} is {REMOVED[spelling]}")
        if separator and not value:
            raise ValueError(f"--engine-arg {text!r} needs a value")
        flag, read = read_flag(spelling, value if separator else None)
        known = FLAGS.get(flag)
        if known is not None and known.takes_value and read is True and flag not in OPTIONAL_VALUE:
            raise ValueError(f"--engine-arg {text!r} needs a value")
        if flag in NO_OP:
            raise ValueError(f"{spelling} has no effect in current builds")
        if flag in PINNED or flag in FIT_FLAGS or flag == "--model":
            raise ValueError(f"{flag} is set by Tensward and cannot be overridden")
        if flag in LOG_FLAGS:
            raise ValueError(f"{flag} changes the server log Tensward reads")
        if flag in SECRETS or is_secret(flag):
            raise ValueError(
                f"{flag} carries a secret, which Tensward never stores or puts on a command line"
            )
        if flag in REFUSED_SOURCES:
            raise ValueError(f"{flag} downloads a model; Tensward serves the registered file")
        if flag in NEVER:
            raise ValueError(f"{flag} is not changed by Tensward: {NEVER[flag]}")
        return apply_flag(settings, flag, read)

    def engine_args_between(self, before: Settings, after: Settings) -> list[str]:
        texts = []
        for field, name in _FIELD_FLAGS:
            value = getattr(after, field)
            if value != getattr(before, field):
                texts.append(f"{name}={'none' if value is None else value}")
        if after.kv_cache_dtype != before.kv_cache_dtype:
            kind = "f16" if after.kv_cache_dtype == "auto" else after.kv_cache_dtype
            texts += [f"cache-type-k={kind}", f"cache-type-v={kind}"]
        if after.prefix_caching != before.prefix_caching and after.prefix_caching is not None:
            texts.append("cache-prompt" if after.prefix_caching else "no-cache-prompt")
        if after.tool_calling and not before.tool_calling:
            texts.append("jinja")
        for flag, value in after.extra_args.items():
            if flag in FIT_FLAGS or before.extra_args.get(flag) == value:
                continue
            bare = flag.removeprefix("--")
            if value is True or value is None:
                texts.append(bare)
            elif value is False:
                texts.append(f"no-{bare}")
            else:
                texts.append(f"{bare}={value}")
        return texts

    def parse_setup(self, text: str) -> ParsedSetup:
        return parse_setup(text)

    def recognizes(self, command: str) -> bool:
        return runs_llama_server(command)

    def host_only(self, current: CurrentSetup | None) -> bool:
        return current is not None and _device_off(current.settings.get("extra_args") or {})

    def availability(self, runtime_kind: str, image_or_command: str) -> Availability:
        if runtime_kind == "docker":
            reason = _SERVE_AS_LOCAL_PROCESS + (_NO_METAL_IN_CONTAINERS if darwin.on_mac() else "")
            return Availability(False, None, reason, "docker_unsupported")
        if executable(image_or_command) is None:
            return Availability(
                False, None,
                "install llama.cpp (build 11669 or newer) so `llama-server` is on PATH, or pass "
                "--local-command",
                "command_missing",
            )  # fmt: skip
        if blocked := quarantined(image_or_command):
            return Availability(False, None, blocked, "quarantined")
        found = probe_build(image_or_command)
        if found is None:
            return Availability(
                False, None,
                f"`{shlex.split(image_or_command)[0]} --version` did not name a llama.cpp build",
                "build_unreadable",
            )  # fmt: skip
        return Availability(True, str(found[0].build) if found[0].build else None, "")

    def inherited_env(self, environ: Mapping[str, str]) -> dict[str, str]:
        return inherited_with_prefix(environ, ENGINE_ENV_PREFIX)

    def quantization_family(self, name: str) -> str:
        return name

    def max_graph_batch(self, settings: Settings) -> int | None:
        return None

    def consistent(self, before: Settings, after: Settings) -> tuple[Settings, str | None]:
        notes = []
        batch = after.prefill_batch_tokens or DEFAULT_BATCH
        ubatch = int(str(after.extra_args.get("--ubatch-size") or DEFAULT_UBATCH))
        if after != before and ubatch > batch:
            batch = ubatch
            after = dataclasses.replace(after, prefill_batch_tokens=batch)
            notes.append(f"raises -b to {batch} to hold -ub {ubatch}")
        context = after.max_context_len
        if after != before and context is not None and context < batch:
            after = dataclasses.replace(after, max_context_len=batch)
            notes.append(f"raises -c to {batch} to hold -b {batch}")
        return after, "; ".join(notes) or None

    def default_concurrency(
        self, version: str | None, gpu: str | None, memory_gib: float | None
    ) -> int | None:
        return 4

    def resolved_default(
        self,
        lever: str,
        *,
        version: str | None,
        gpu_memory_gib: float | None,
        gpu_model: str | None,
        step_budget_tokens: int | None = None,
        takes_images: bool = False,
        frontend_processes: int = 1,
    ) -> Any:
        """None: effective() already carries the values llama.cpp resolved at launch."""
        return None

    def unstartable(self, current: Settings, applied: Settings) -> str | None:
        if (build := known_build()) is None:
            return None
        used = self._flags_used(applied) - self._flags_used(current)
        missing = build[1].missing(sorted(used))
        if not missing:
            return None
        return (
            f"this llama.cpp build ({build[0].build or 'unknown'}) does not list "
            f"{', '.join(missing)} in --help; rebuild or upgrade llama.cpp to change it"
        )

    @staticmethod
    def _flags_used(settings: Settings) -> set[str]:
        argv = launch_argv(settings, model="m", served_model_name="a", host="h", port=0)
        return {word for word in argv if word.startswith("-") and flag_for(word) is not None}

    def default_tool_parser(self, model_dir: Path) -> str | None:
        return None

    def kv_in_flight_tokens(self, settings: Settings, takes_images: bool) -> int | None:
        return None

    def memory_observations(self, log: str) -> list[Allocation]:
        return list(read_startup(log).allocations)

    def parallel_degree(self, settings: Settings) -> int:
        if _device_off(settings.extra_args):
            return 0
        return 1 if detect_platform() is not None else 0

    def lever_value(self, settings: Settings, lever: str) -> Any:
        if lever == "kv_cache_precision":
            return "8bit" if settings.kv_cache_dtype == Q8 else "default"
        if lever == "host_layers":
            return None
        if lever == "host_expert_layers":
            return int(str(settings.extra_args.get("--n-cpu-moe") or 0))
        if lever == "speculation":
            method = settings.extra_args.get("--spec-type")
            if not isinstance(method, str) or method == "none":
                return None
            draft = str(settings.extra_args.get(SPEC_DRAFT_MAX))
            return Speculation(method, int(draft) if draft.isdigit() else 0)
        if lever in ("compact_json", "graph_capture_limit"):
            return None if lever == "graph_capture_limit" else False
        known = LEVERS.get(lever)
        if known is None or known.field is None:
            raise ValueError(f"llama.cpp does not realise the lever {lever}")
        if lever == "prefill_batch_tokens":
            return settings.prefill_batch_tokens or DEFAULT_BATCH
        return getattr(settings, known.field)

    def with_lever(self, settings: Settings, lever: str, value: Any) -> Settings:
        if lever == "speculation":
            return _with_speculation(settings, value)
        flag = {**_LEVER_FLAGS, "kv_cache_precision": "--cache-type-k"}.get(lever)
        if flag is None:
            raise ValueError(f"llama.cpp does not realise the lever {lever}")
        if (build := known_build()) is not None and not build[1].has(flag):
            raise ValueError(f"not in this build's --help: {flag}")
        if lever == "kv_cache_precision":
            if value not in ("default", "8bit"):
                raise ValueError(f"llama.cpp has no {value} KV cache")
            for key in _KV_TYPE_FLAGS:
                settings = apply_flag(settings, key, Q8 if value == "8bit" else "f16")
            return settings
        if lever == "host_layers":
            raise ValueError(
                "llama.cpp places layers with --n-gpu-layers, which needs the model's layer count"
            )
        if lever == "host_expert_layers":
            if not isinstance(value, int) or value < 0:
                raise ValueError("host_expert_layers takes a count of layers, 0 or more")
            extra = {k: v for k, v in settings.extra_args.items() if k != flag}
            if value:
                extra[flag] = str(value)
            return dataclasses.replace(settings, extra_args=extra)
        if lever in ("prefix_caching", "tool_calling"):
            return apply_flag(settings, flag, bool(value))
        if lever == "max_context_len" and value and not self._shared_pool(settings):
            slots = settings.max_concurrent_requests or 1
            return apply_flag(settings, flag, value * slots)
        if value is None:
            unset: dict[str, Any] = {lever: None}
            return dataclasses.replace(settings, **unset)
        return apply_flag(settings, flag, value)

    @staticmethod
    def _shared_pool(settings: Settings) -> bool:
        """Whether ``-c`` is one context every slot draws from: a unified pool, or one slot."""
        return not (split_kv(settings) and (settings.max_concurrent_requests or 1) > 1)

    async def count_prompt_tokens(
        self, client: httpx.AsyncClient, server_url: str, model: str, body: Mapping[str, Any]
    ) -> int | None:
        try:
            if "messages" in body:
                rendered = {k: body[k] for k in ("messages", "tools") if k in body}
                response = await client.post(f"{server_url}/apply-template", json=rendered)
                response.raise_for_status()
                text, special = response.json()["prompt"], False
            else:
                text, special = str(body["prompt"]), True
            response = await client.post(
                f"{server_url}/tokenize",
                json={"content": text, "add_special": special, "parse_special": True},
            )
            response.raise_for_status()
            return len(response.json()["tokens"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return None

    def request_extras(self, workload: WorkloadSpec) -> dict[str, Any]:
        constraint = workload.structured_output
        if constraint is None:
            return {}
        if "json" in constraint:
            schema = {"schema": constraint["json"]}
            return {"response_format": {"type": "json_schema", "json_schema": schema}}
        if "json_object" in constraint:
            return {"response_format": {"type": "json_object"}}
        raise ValueError("llama.cpp takes a JSON schema or a JSON object as structured output")

    def tool_calling_fix(
        self, settings: Settings, source: str, resolved: Resolved | None = None
    ) -> str | None:
        if settings.extra_args.get("--jinja") is not False:
            caps = (resolved.facts.get("chat_template_caps") if resolved else None) or {}
            if caps.get("supports_tools") is False:
                return (
                    "the workload offers tools but this GGUF's chat template does not support "
                    "tools, so llama-server cannot return tool calls"
                )
            return None
        return (
            "the workload offers tools but your command runs --no-jinja, so llama-server cannot "
            "return tool calls; drop --no-jinja from your --current command and run `tensward "
            "init` again with it as a new project"
        )

    def reproducibility_advice(self, source: str, settings: Settings) -> str:
        missing = [
            flag
            for flag, held in (
                ("`-np 1`", settings.max_concurrent_requests == 1),
                ("`--no-cache-prompt`", settings.prefix_caching is False),
            )
            if not held
        ]
        if not missing:
            return "Your `--current` command already has one slot and no prompt reuse."
        return (
            f"For the answer check only, not for serving: put {' and '.join(missing)} in a copy of "
            "your `--current` command, so that with one slot and no prompt reuse every request is "
            "computed alone. Register it with `tensward init` as a separate project, then analyse "
            "that setup."
        )

    def loads_encoders(self, settings: Settings, modalities: Collection[str]) -> bool:
        return False

    def parse_quant_kernels(self, log_text: str) -> tuple[QuantKernel, ...]:
        return ()

    def capabilities(self, version: str | None, settings: Settings | None = None) -> Capabilities:
        return LLAMACPP_CAPABILITIES

    def runs_remote_code(self, settings: Settings) -> bool:
        return False

    def frontend_processes(self, settings: Settings) -> int | None:
        return 1

    def weights_on_device(self, settings: Settings, resolved: Resolved | None) -> bool | None:
        if resolved is None or not resolved.facts.get("device_weight_bytes"):
            return None
        host_experts = int(str(settings.extra_args.get("--n-cpu-moe") or 0))
        moved = (
            settings.extra_args.get("--cpu-moe") is True
            or "--override-tensor" in settings.extra_args
        )
        return not (resolved.levers.get("host_layers") or host_experts or moved)

    def host_warnings(self) -> tuple[str, ...]:
        return ()

    def _metal_ran_out(self, log: str) -> bool:
        return any(key in log for key, text in self.log_checks.items() if text == _METAL_ALLOCATION)

    def failure_note(
        self,
        log: str,
        anatomy: ModelAnatomy | None,
        platform: str | None,
        settings: Settings,
        host_before: HostFacts | None = None,
    ) -> str | None:
        if platform != "apple" or not self._metal_ran_out(log):
            return None
        lines = ["the server log shows Metal ran out of memory: the Metal working set is the bound"]
        started = read_startup(log)
        resolved = resolution(started, {}, settings, anatomy)
        where = placement(anatomy, resolved) if anatomy is not None and resolved else None
        free, pool = started.working_set_bytes, started.ctx_total
        if anatomy is not None and where is not None and free and pool:
            lines += working_set_bound(
                anatomy, where,
                pool_tokens=pool,
                type_k=started.type_k or "f16",
                type_v=started.type_v or "f16",
                compute_buffer_bytes=started.bytes_of("compute", "device"),
                device_free=free,
                host=host_before or host_facts(),
                reading="before the launch" if host_before else "now",
            )  # fmt: skip
        return "\n".join(lines)

    def placement(self, resolved: Resolved | None) -> Placement | None:
        found = resolved.facts.get("placement") if resolved else None
        return Placement(**found) if found else None

    def fit_advice(
        self,
        anatomy: ModelAnatomy,
        settings: Settings,
        device_name: str,
        device_total_bytes: int,
        host: HostFacts,
        runtime: Runtime | None,
        context_limit: int | None,
        command: Callable[..., str] | None,
        baseline: Resolved | None = None,
    ) -> tuple[str, bool] | None:
        experts = next((e for e in self.playbook() if e.name == "offload-experts"), None)
        threads = str(settings.extra_args.get("--threads"))
        gbs = None
        if runtime is not None:
            gbs = launched_bandwidth(
                host,
                runtime,
                model_fingerprint(anatomy),
                int(threads) if threads.isdigit() else None,
            )
        return fit_advice(
            anatomy, settings, device_name, device_total_bytes, host, gbs, context_limit,
            command, experts, baseline,
        )  # fmt: skip

    def placement_report(
        self,
        anatomy: ModelAnatomy,
        measurement: Measurement,
        resolved: Resolved | None,
        log: str,
        platform: str | None,
        need: int = 0,
        host: HostFacts | None = None,
        settings: Settings | None = None,
        facts: WorkloadFacts | None = None,
        window: HostWindow | None = None,
        rows: Iterable[Mapping[str, Any]] = (),
        /,
    ) -> dict[str, Any] | None:
        started = read_startup(log)
        host = host or host_facts()
        span = measurement.window
        state = analysis().gpu_state(span.start_ns, span.end_ns) if span else None
        total, peak = (state.memory_total_mib, state.memory_peak_mib) if state else (None, None)
        ceilings = measurement.ceilings
        running = ceilings.avg_running_batch if ceilings else None
        batch = round(running or 1)
        topology: Topology = next(
            (p.memory_topology for p in PLATFORMS if p.name == platform), "discrete"
        )
        faults = window.major_faults() if window else None
        block = placement_block(
            anatomy, resolved,
            step_ms=measurement.tpot_p50_ms,
            bw_dev_gbs=ceilings.bandwidth_gbs if ceilings else None,
            concurrency=max(batch, 1),
            host=host,
            topology=topology,
            headroom=None if total is None or peak is None else int((total - peak) * 2**20),
            device_free=(
                started.working_set_bytes if topology == "unified" else started.device_free_bytes
            ),
            compute_buffer_bytes=started.bytes_of("compute", "device"),
            kv_footprint=started.bytes_of("kv", "device"),
            type_k=started.type_k or "f16",
            type_v=started.type_v or "f16",
            pool_tokens=started.ctx_total or 0,
            ubatch=started.ubatch or 0,
            slots=started.slots or 0,
            need_tokens=need,
            peak_running=round(measurement.peak_running or 0),
            bw_host_gbs=resolved.facts.get(HOST_BANDWIDTH_FACT) if resolved else None,
            window=window,
            kv_logged=started.bytes_of("kv", "device") + started.bytes_of("kv", "host"),
            prefill=prefill_work(rows, started.ubatch, running),
            pcie=resolved.facts.get(PCIE_RATE_FACT) if resolved else None,
            fit_places=bool({"n_gpu_layers", "tensor_overrides"} & set(started.fit_changed)),
            ran_out=platform == "apple" and self._metal_ran_out(log),
        )  # fmt: skip
        if block is not None and faults is not None:
            block["major_faults"] = {"start": faults[0], "end": faults[1]}
        if block is not None and host.os_family == "macos" and window is not None:
            block["host_window"] = _host_window(window)
        if block is not None:
            block["environment"] = environment(host, started.system_info, platform or "")
            block["advice"] = advice(block, settings, facts, log)
        return block

    def run_probes(self, run: ProbeRun) -> ProbeResult | None:
        return probe_run(run)

    def untimed(self, record: RequestRecord, response: CapturedResponse | None) -> str | None:
        if record.outcome != "success" or response is None or response.timings:
            return None
        return "the final stream event carried no timings"

    async def resolved(
        self,
        client: httpx.AsyncClient,
        url: str,
        server_log: str,
        settings: Settings,
        anatomy: ModelAnatomy | None,
    ) -> Resolved | None:
        props: Mapping[str, Any] = {}
        try:
            response = await client.get(f"{url}/props")
            if response.status_code == 200:
                props = response.json()
        except (httpx.HTTPError, ValueError):
            pass
        return resolution(read_startup(server_log), props, settings, anatomy)

    def probe(self) -> ReadyProbe:
        return LlamaCppProbe(self.health_path, self.models_path)

    def request_model(self, served_name: str) -> str:
        return served_name

    def effective(self, settings: Settings, resolved: Resolved | None) -> Settings:
        if resolved is None:
            return settings
        levers = resolved.levers
        given: dict[str, Any] = {
            field: levers[lever]
            for field, lever in _EFFECTIVE_FIELDS
            if levers.get(lever) is not None
        }
        if levers.get("kv_cache_precision") == "8bit":
            given["kv_cache_dtype"] = Q8
        extra = dict(settings.extra_args)
        unreadable = _unreadable_fit(resolved)
        for flag, lever in _EFFECTIVE_FLAGS:
            if unreadable and flag == "--n-gpu-layers":
                continue
            if levers.get(lever) is not None and flag not in extra:
                extra[flag] = (
                    levers[lever] if isinstance(levers[lever], bool) else str(levers[lever])
                )
        return dataclasses.replace(settings, extra_args=extra, **given)

    def pinned(self, settings: Settings, current: Settings, baseline: Resolved | None) -> Settings:
        extra = dict(settings.extra_args)
        given: dict[str, Any] = {}
        unreadable = baseline is not None and _unreadable_fit(baseline)
        # A change that drops a forced -ngl hands the placement to --fit: it must run.
        fit_places = (
            extra.get("--n-gpu-layers")
            == "auto"
            != current.extra_args.get("--n-gpu-layers", "auto")
        )
        # A draft context the baseline did not have needs device memory its context took: --fit
        # sizes the context again, unless the user's command sized it.
        refits = _drafts_anew(self, settings, current) and current.max_context_len is None
        if not unreadable and not fit_places and not refits:
            extra = {k: v for k, v in extra.items() if k not in FIT_FLAGS}
            extra["--fit"] = "off"
        if baseline is None:
            return dataclasses.replace(settings, extra_args=extra)
        levers = baseline.levers
        for field, lever in _PINNED_FIELDS:
            if refits and field == "max_context_len":
                continue
            if getattr(settings, field) == getattr(current, field) and levers.get(lever):
                given[field] = levers[lever]
        moved = any(
            settings.extra_args.get(f) != current.extra_args.get(f) for f in _EXPERT_PLACEMENT
        )
        for flag, lever in _PINNED_FLAGS:
            placement = flag in ("--n-gpu-layers", "--n-cpu-moe")
            if placement and unreadable or flag == "--n-cpu-moe" and moved:
                continue
            if settings.extra_args.get(flag) == current.extra_args.get(flag):
                if (value := levers.get(lever)) is not None:
                    extra[flag] = value if isinstance(value, bool) else str(value)
        return dataclasses.replace(settings, extra_args=extra, **given)

    def served_command(self, text: str, settings: Settings) -> str | None:
        return production_command(text, settings)

    def repinned(self, settings: Settings, resolved: Resolved | None) -> Settings | None:
        if resolved is None or "--n-gpu-layers" in settings.extra_args:
            return None
        if settings.extra_args.get("--fit") in ("off", False):
            return None
        if not {"n_gpu_layers", "tensor_overrides"} & set(resolved.facts.get("fit_changed", ())):
            return None
        if _unreadable_fit(resolved):
            return None
        return self.pinned(settings, settings, resolved)

    def playbook(self, *, version: str | None = None) -> tuple[Entry, ...]:
        return load_entries(self, version, family=self.family)


LLAMACPP = LlamaCppEngine()
