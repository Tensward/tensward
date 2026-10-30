"""What Tensward needs from a serving engine, in engine-neutral terms.

``Settings`` names the knobs Tensward reads and tunes; an :class:`Engine` maps them to its own
command line and reads its own metrics back as :class:`EngineSignals`. Everything that is
specific to one engine (flag names, metric names, image, credential variable) lives behind
that engine's implementation; callers never see it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Iterable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
)

import httpx

from ..anatomy import ModelAnatomy
from ..capabilities import Capabilities, Resolved
from ..capture import CapturedResponse
from ..classify import EngineRule
from ..client import RequestRecord
from ..playbook import Entry, Gate, Situation, WorkloadFacts
from ..probes import docker_image
from ..prometheus import SignalMap
from ..secrets import is_secret
from ..settings import Availability, ParsedSetup, QuantKernel, Settings
from ..signal_source import SignalSource
from ..workload import WorkloadSpec

if TYPE_CHECKING:
    from ..context import AnalysisContext
    from ..kb_actions import Action
    from ..measurement import Measurement
    from ..platforms.cpu import HostFacts
    from ..runtime import Runtime, ServeSpec
    from .llamacpp.placement import Placement


def docker_availability(
    image: str, version_of: Callable[[str, Mapping[str, str]], str | None]
) -> Availability:
    """Whether docker has ``image``; ``version_of`` reads the engine's version from the image
    reference and its labels. Never pulls anything."""
    found = docker_image(image)
    if found.labels is not None:
        return Availability(True, version_of(image, found.labels), "")
    if found.reason == "docker_missing":
        return Availability(False, None, "install Docker", found.reason)
    if found.reason == "docker_unreachable":
        said = f" (docker said: {found.detail})" if found.detail else ""
        return Availability(
            False, None,
            "start the Docker daemon or add your user to the docker group" + said, found.reason,
        )  # fmt: skip
    return Availability(
        False, None, f"run `docker pull {image}` (Tensward never pulls images itself)", found.reason
    )


class Tracing(Protocol):
    """An engine's profiler: how to launch with it and how to drive and collect a trace."""

    env: Mapping[str, str]  # environment that makes a traced launch's trace more useful
    # extra launch arguments when a kernel profiler wraps the launch: start/stop then gate it
    counters_args: Sequence[str]
    step_scope: str  # name of the CPU annotation wrapping one model step in its trace
    step_pattern: str  # matches the per-step CPU annotation of every model runner

    async def start(self, client: httpx.AsyncClient, server_url: str) -> None:
        """Begin recording the engine's profiler trace."""
        ...

    async def stop(self, client: httpx.AsyncClient, server_url: str) -> None:
        """Stop recording; returns once the trace files are written."""
        ...

    def collect(self, trace_dir: Path) -> list[Path]:
        """The PyTorch-profiler Chrome trace files the engine wrote into ``trace_dir``."""
        ...


@dataclass(frozen=True, slots=True)
class Ready:
    version: str | None = None  # the engine version the server reports, when it reports one


@dataclass(frozen=True, slots=True)
class NotYet:
    pass


@dataclass(frozen=True, slots=True)
class Failed:
    reason: str  # why waiting longer cannot help


class ReadyProbe(Protocol):
    """One wait for a launched server: each ``check`` asks it once."""

    timeout_s: float  # the HTTP timeout of one check

    async def check(
        self, client: httpx.AsyncClient, url: str, served_name: str
    ) -> Ready | NotYet | Failed: ...


class ModelListProbe:
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


def inherited_with_prefix(environ: Mapping[str, str], prefix: str) -> dict[str, str]:
    """The variables of ``environ`` whose names start with ``prefix``, credentials left out."""
    return {n: v for n, v in environ.items() if n.startswith(prefix) and not is_secret(n)}


class FlagRules(Protocol):
    """What an engine's command line admits for Tensward to change, and how its spellings
    compare."""

    engine: str  # the knowledge scope it serves
    version: str  # the engine version its tables are validated against
    env_names: frozenset[str]  # every environment variable Tensward may set for it
    pool_shrinking: Mapping[str, str]  # flag -> the Settings field whose lower value it shrinks
    pool_defaults: Mapping[str, Any]  # flag -> the engine's default when none is set
    pool_bytes_flag: str | None  # the flag that sets the pool in bytes, when the engine has one

    def canonical(self, flags: Mapping[str, Any]) -> dict[str, Any]:
        """``flags`` with two spellings of one setting folded into one."""
        ...

    def allowed(self, version: str | None, flag: str) -> bool:
        """Whether a record may change ``flag`` on this engine ``version``."""
        ...

    def env_allowed(self, version: str | None, name: str) -> bool:
        """Whether a record may set the environment variable ``name`` on ``version``."""
        ...

    def merge(self, extra_args: Mapping[str, Any], flags: Mapping[str, Any]) -> dict[str, Any]:
        """``extra_args`` with ``flags`` written in."""
        ...

    def canonical_args(self, args: Iterable[str]) -> frozenset[str]:
        """Engine arguments as ``--flag=value`` words, two spellings of one change as one."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ProbeRun:
    """What an engine's probes may read about the baseline run that just finished: the launch
    (``spec``, with the runtime that started it), its measurement and log, its requests, the
    host as the server saw it and where the project's earlier runs are. ``launch`` is False when
    the user asked for no extra launches."""

    runtime: Runtime
    spec: ServeSpec
    anatomy: ModelAnatomy | None
    resolved: Resolved | None
    measurement: Measurement
    server_log: str
    rows: Sequence[Mapping[str, Any]]
    host: HostFacts | None
    platform: str | None
    project_dir: Path
    snapshot_id: str
    engine_args: Sequence[str]
    ready_timeout_s: float
    launch: bool = True


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """The ``probes`` block of ``run.json``; the host memory bandwidth it measured, when it did;
    and ``resolved`` with that bandwidth where the engine's rules read it."""

    block: Mapping[str, Any]
    host_gbs: float | None
    resolved: Resolved | None


@dataclass(frozen=True, slots=True)
class Allocation:
    """One memory reading from an engine's log."""

    domain: Literal["device", "host"]
    device: str  # "CUDA0", "CPU", "CUDA_Host"
    kind: str  # "model", "kv", "compute", "output", "other"
    bytes: int


class EngineLog(Protocol):
    """What an engine's server log and scrapes tell Tensward about a launch."""

    start_classes: tuple[str, ...]  # how a start can go, cheapest first

    def start_class(self, log: str, caches: Any, settings: Settings) -> str | None:
        """How the start in ``log`` went; None when the log cannot say."""
        ...

    def start_class_for(self, flags: frozenset[str], quantization: str | None) -> str:
        """The start class of a launch that changes only ``flags``, before it has run."""
        ...

    def compile_neutral(self, flags: frozenset[str]) -> bool:
        """Whether a change of only ``flags`` keeps the compiled graph the engine caches."""
        ...

    def start_cost(self, flags: frozenset[str], quantization: str | None) -> int:
        """The position of :meth:`start_class_for` in ``start_classes``."""
        ...

    def window_facts(
        self,
        log: str,
        compile_cache: str | None,
        caches: Any = None,
        quantization: str | None = None,
    ) -> dict[str, Any]:
        """The start-up figures of a run's window.json: the start times, the pool and memory
        lines, and how the start went (``caches``: the run's recorded compile caches)."""
        ...

    def kv_pool(self, log: str) -> tuple[int | None, float | None]:
        """The KV pool the engine logged: tokens and GiB."""
        ...

    def slot_tokens(self, log: str) -> int | None:
        """The context each request slot holds, when the engine logs one."""
        ...

    def load_lines(self, log: str) -> int | None:
        """How many load stats lines the log holds, so the load's own start can be told from
        the probe and warm-up before it; None: the engine has no such watch."""
        ...

    def load_queue(self, log: str, skip: int = 0) -> Any | None:
        """Requests waiting at the KV limit early in the load, which starts after the first
        ``skip`` stats lines; None: the engine has no such watch."""
        ...

    def no_kv_memory(self, log: str) -> bool:
        """Whether the engine says no memory is left for its KV cache."""
        ...

    def step_budget(self, log: str) -> int | None:
        """The per-step token budget the engine logged."""
        ...

    def memory_usage(self, log: str) -> dict[str, float] | None:
        """What the engine's memory line says it used and could give the KV cache: GiB used and
        bytes for the KV cache (``full_kv_bytes``, ``fit_kv_bytes``); None without the line."""
        ...

    def run_facts(
        self,
        run_dir: Path | None,
        context: AnalysisContext,
        situation: Situation,
        span_s: float | None,
    ) -> dict[str, object]:
        """The facts the run's scrapes, server log and settings carry; ``run_dir`` is None when
        the run has no directory."""
        ...

    def cache_observer(
        self, context: AnalysisContext, runtime: Runtime, env: Mapping[str, str]
    ) -> Callable[[Any], None] | None:
        """Records the engine's compile caches in ``context.window_extra`` over a launch."""
        ...


class Engine(Protocol):
    """One serving engine."""

    name: str
    label: str  # the engine's name as people write it
    version_prefix: str  # what precedes its version in a sentence ("build " for 11669)
    formats: frozenset[str]  # the checkpoint formats it serves (CheckpointFormat.name)
    platforms: frozenset[str]  # the platforms it runs on (Platform.name)
    launchers: str  # the commands ``recognizes`` accepts, in words
    default_image: str  # container image used when none is given
    local_command: tuple[str, ...]  # server command for the local-process runtime
    api_key_env: str  # environment variable that carries the API key to the server
    health_path: str
    models_path: str  # OpenAI-style model list, used to confirm which model is served
    metrics_path: str
    signals: SignalMap  # where its metrics export each engine-neutral signal
    signal_source: SignalSource  # collects the neutral signals (polls and the finished run)
    # quantized-linear kernel (as parse_quant_kernels names it) -> its name in CUDA kernel symbols
    quant_kernel_symbols: Mapping[str, str]
    log_prefix: re.Pattern[str] | None  # what starts each server-log line, stripped when read
    tracing: Tracing | None  # its profiler, None if it has none
    log_checks: Mapping[str, str]  # server-log text -> the report check it raises
    gpu_memory_log: (
        str  # matches the server-log line giving the GPU memory total in GiB (one group)
    )
    kv_memory_log: str  # matches the server-log line giving the KV cache memory in GiB (one group)
    version_log: str  # matches the server-log line that names the engine's version (one group)
    # matches the server-log line for a dtype cast; groups: the checkpoint's dtype, the served one
    dtype_cast: str
    # What the engine does for a neutral setting left unset (a ``Settings`` field name -> words),
    # so a report can say what really ran.
    defaults: Mapping[str, str]
    default_kv_memory_fraction: float | None  # the share of GPU memory it uses; None: unknown
    log: EngineLog
    default_runtime: str  # how it runs when --runtime is not given ("docker" or "local")
    family: str  # the playbook entries' ``engines`` name that suit it ("vllm" for VllmEngine)
    # Engine-namespaced playbook predicates and gates ("vllm:<name>") that its entries name.
    predicates: Mapping[str, Callable[..., Any] | Gate]
    # Entry or lever name -> why it is held back on a run (None: it is not), consulted only
    # when an offer path would offer the entry.
    hold_backs: Mapping[str, Callable[[Situation], str | None]]
    # Memory a launch takes beyond the weights and the KV cache (an estimate, from measurement);
    # None: unknown.
    memory_overhead_bytes: int | None
    # What a launch carries beyond the setup, in words, so a report can disclose it.
    added_flags: str
    # The calibration profile used when the platform, GPU or engine version is unknown; None
    # for an engine without a public profile.
    default_profile: str | None
    engine_rules: tuple[EngineRule, ...]  # diagnosis rules only this engine has
    flag_rules: FlagRules
    value_actions: Mapping[str, Action]  # "<family>:<rule>" -> the action that computes its value

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
        """The engine's arguments (after the executable or image) that serve ``model``.

        With ``trace_dir`` (a path as the server sees it) the engine's profiler is enabled and
        will write its trace files there. ``serving`` is a long-lived server for clients, not a
        measured launch: the engine leaves out what only Tensward's measurement reads.
        """
        ...

    def with_engine_arg(self, settings: Settings, text: str) -> Settings:
        """``settings`` with one operator ``KEY=VALUE`` engine flag applied; ValueError if bad."""
        ...

    def engine_args_between(self, before: Settings, after: Settings) -> list[str]:
        """The ``--engine-arg`` texts that turn ``before`` into ``after``."""
        ...

    def parse_setup(self, text: str) -> ParsedSetup:
        """Understand the customer's command line (or container run) as settings; ValueError if
        it is not one this engine can reproduce."""
        ...

    def recognizes(self, command: str) -> bool:
        """Whether ``command``, a ``--current`` command line, launches this engine."""
        ...

    def availability(self, runtime_kind: str, image_or_command: str) -> Availability:
        """Whether the engine is here for ``runtime_kind`` ("docker": the image is present;
        "local": the command can be run). Never pulls, downloads or raises."""
        ...

    def inherited_env(self, environ: Mapping[str, str]) -> dict[str, str]:
        """The variables of ``environ`` that change how the engine behaves, credentials left
        out."""
        ...

    def quantization_family(self, name: str) -> str:
        """The quantization method ``name`` stands for: the engine's aliases of one method (such
        as a fused-kernel variant) share a family."""
        ...

    def consistent(self, before: Settings, after: Settings) -> tuple[Settings, str | None]:
        """``after`` with the engine options that its change from ``before`` couples to it
        aligned, and a note saying what was changed (None when nothing was)."""
        ...

    def max_graph_batch(self, settings: Settings) -> int | None:
        """The most sequences one decode step can batch while still running as a captured CUDA
        graph, when the settings pin the capture sizes (or only their maximum). None when they
        pin neither, or turn the graphs off."""
        ...

    def default_concurrency(
        self, version: str | None, gpu: str | None, memory_gib: float | None
    ) -> int | None:
        """The most requests the engine runs at once when the setup sets no cap, for engine
        ``version`` on a GPU named ``gpu`` with ``memory_gib`` of memory; None when unknown."""
        ...

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
        """The value the engine runs with when the setup leaves ``lever`` unset, for the levers
        whose default depends on the model, the GPU or the parallel size; None when unknown."""
        ...

    def unstartable(self, current: Settings, applied: Settings) -> str | None:
        """Why ``applied``, a change from ``current``, cannot start or would lose CUDA graphs
        that ``current`` has for the batches it runs; None when it can be tried. A setup that
        ``current`` already fails to start is not reported for the same failure."""
        ...

    def default_tool_parser(self, model_dir: Path) -> str | None:
        """The engine's tool-call parser for the checkpoint's model family; None if unknown."""
        ...

    def kv_in_flight_tokens(self, settings: Settings, takes_images: bool) -> int | None:
        """Tokens scheduled but not yet settled, which a sliding-window layer keeps beside its
        window: the engine reserves KV for them per request. A checkpoint that takes images
        raises the batch size to its largest media item, unless media inputs are off. None when the
        engine reserves none the model can state."""
        ...

    def memory_observations(self, log: str) -> list[Allocation]:
        """Every memory reading the server log carries."""
        ...

    def parallel_degree(self, settings: Settings) -> int:
        """How many GPUs the settings spread one model over (1 when it runs on one)."""
        ...

    def lever_value(self, settings: Settings, lever: str) -> Any:
        """``lever``'s value in ``settings``; a non-identity lever as the engine stores it."""
        ...

    def with_lever(self, settings: Settings, lever: str, value: Any) -> Settings:
        """``settings`` with ``lever`` set to ``value``; ValueError for one it cannot realise."""
        ...

    def playbook(self, *, version: str | None = None) -> tuple[Entry, ...]:
        """The setting changes this engine offers for ``version`` (None: what every supported
        version can realise), each with the bottlenecks it addresses. The same objects on every
        call."""
        ...

    async def count_prompt_tokens(
        self, client: httpx.AsyncClient, server_url: str, model: str, body: Mapping[str, Any]
    ) -> int | None:
        """The prompt's token count by the server's own tokenizer; None if it cannot say.
        ``body`` is ``{"prompt": text}`` or a chat body with ``messages``."""
        ...

    def request_extras(self, workload: WorkloadSpec) -> dict[str, Any]:
        """Fields this engine needs in every request body for ``workload``."""
        ...

    def tool_calling_fix(
        self, settings: Settings, source: str, resolved: Resolved | None = None
    ) -> str | None:
        """Why a workload that offers tools cannot be served with ``settings``, with the fix for
        a setup registered from ``source`` ("command", "config" or "defaults"); None when it
        can. ``resolved`` is what the launched server chose, when a server ran."""
        ...

    def reproducibility_advice(self, source: str, settings: Settings) -> str:
        """How to make answers repeatable when the noise floor is missing, for a setup
        registered from ``source`` ("command", "config" or "defaults")."""
        ...

    def loads_encoders(self, settings: Settings, modalities: Collection[str]) -> bool:
        """Whether a launch with ``settings`` loads the media encoders of ``modalities``."""
        ...

    def parse_quant_kernels(self, log_text: str) -> tuple[QuantKernel, ...]:
        """The quantized-linear kernels the engine's log says it selected; empty if none."""
        ...

    def capabilities(self, version: str | None, settings: Settings | None = None) -> Capabilities:
        """What this engine reports and can change. ``version`` None: what every version the
        adapter supports has; ``settings``: the launch, when its flags change what is reported."""
        ...

    def runs_remote_code(self, settings: Settings) -> bool:
        """Whether ``settings`` let the engine run the checkpoint's own Python code."""
        ...

    def frontend_processes(self, settings: Settings) -> int | None:
        """How many API-server processes a launch with ``settings`` runs; None when unknown."""
        ...

    def weights_on_device(self, settings: Settings, resolved: Resolved | None) -> bool | None:
        """False when some weights stay in host memory; None when unknown (read as on device)."""
        ...

    def host_warnings(self) -> tuple[str, ...]:
        """Problems with this machine's installation that make the engine fail, in words."""
        ...

    def placement(self, resolved: Resolved | None) -> Placement | None:
        """Where the launch keeps its weights and KV cache; None for an engine that does not
        place them (everything on the device)."""
        ...

    def fit_advice(
        self,
        anatomy: ModelAnatomy,
        settings: Settings,
        device_name: str,
        device_total_bytes: int,
        host: HostFacts,
        runtime: Runtime | None,
        context_limit: int | None,
        command: Callable[[Settings], str] | None,
        baseline: Resolved | None = None,
    ) -> tuple[str, bool] | None:
        """What to tell before a launch whose placement cannot fit the device: (text, whether the
        launch certainly fails); None when the settings fit, or the engine has no model of it.
        ``runtime`` lets the engine read a stored host-bandwidth probe; ``command`` writes the
        ``tensward analyse`` command that measures given settings, None where none is offered;
        ``baseline`` is what the project's baseline run resolved."""
        ...

    def placement_report(
        self,
        anatomy: ModelAnatomy,
        measurement: Measurement,
        resolved: Resolved | None,
        server_log: str,
        platform: str | None,
        need_tokens: int = 0,
        host: HostFacts | None = None,
        settings: Settings | None = None,
        facts: WorkloadFacts | None = None,
        major_faults: tuple[int, int] | None = None,
        rows: Sequence[Mapping[str, Any]] = (),
        /,
    ) -> dict[str, Any] | None:
        """The placement, regime, decode model and next steps of a run, for ``metrics.json``;
        None when the engine has none. ``host`` is the host as the server saw it;
        ``major_faults`` the server's cumulative count at the window's start and end; ``rows``
        the run's request records."""
        ...

    def run_probes(self, run: ProbeRun) -> ProbeResult | None:
        """Measure what the baseline's log cannot say (host memory bandwidth, PCIe rate), where
        the placement makes it matter; None when the engine has no probes or the run needs none."""
        ...

    async def resolved(
        self,
        client: httpx.AsyncClient,
        url: str,
        server_log: str,
        settings: Settings,
        anatomy: ModelAnatomy | None,
    ) -> Resolved | None:
        """What the engine chose at launch where the ``settings`` left it free, read once after
        the server is ready; None when it chose nothing Tensward reads."""
        ...

    def effective(self, settings: Settings, resolved: Resolved | None) -> Settings:
        """``settings`` with the launch-resolved lever values filled in. Facts are never
        written into settings; the stored settings stay as requested."""
        ...

    def pinned(self, settings: Settings, current: Settings, baseline: Resolved | None) -> Settings:
        """``settings`` for a launch measured against a baseline run of ``current``, with what
        the baseline's launch resolved written in wherever ``settings`` left it as ``current``
        had it, so the two differ only by the change under test."""
        ...

    def repinned(self, settings: Settings, resolved: Resolved | None) -> Settings | None:
        """The settings of a second baseline launch that holds what the first launch's engine
        chose for a placement the setup left to it; None when the first launch is the baseline."""
        ...

    def served_command(self, text: str, settings: Settings) -> str | None:
        """The customer's own command ``text`` with ``settings`` written over it, the command
        that serves what was measured; None for an engine whose settings are not command-line
        flags."""
        ...

    def untimed(self, record: RequestRecord, response: CapturedResponse | None) -> str | None:
        """Why this request's client timings do not measure the engine; None when they do."""
        ...

    def probe(self) -> ReadyProbe:
        """A fresh readiness probe for one wait."""
        ...

    def request_model(self, served_name: str) -> str:
        """The ``model`` value sent in the probe and in every request."""
        ...
