"""What Tensward needs from a serving engine, in engine-neutral terms.

``Settings`` names the knobs Tensward reads and tunes; an :class:`Engine` maps them to its own
command line and reads its own metrics back as :class:`EngineSignals`. Everything that is
specific to one engine (flag names, metric names, image, credential variable) lives behind
that engine's implementation; callers never see it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Protocol, Sequence

import httpx

from ..playbook import Entry
from ..probes import docker_image
from ..prometheus import SignalMap
from ..settings import Availability, ParsedSetup, QuantKernel, Settings
from ..workload import WorkloadSpec


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

    async def start(self, client: httpx.AsyncClient, server_url: str) -> None:
        """Begin recording the engine's profiler trace."""
        ...

    async def stop(self, client: httpx.AsyncClient, server_url: str) -> None:
        """Stop recording; returns once the trace files are written."""
        ...

    def collect(self, trace_dir: Path) -> list[Path]:
        """The PyTorch-profiler Chrome trace files the engine wrote into ``trace_dir``."""
        ...


class Engine(Protocol):
    """One serving engine."""

    name: str
    label: str  # the engine's name as people write it
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
    default_kv_memory_fraction: float  # the share of GPU memory it uses when none is set
    # Memory a launch takes beyond the weights and the KV cache (an estimate, from measurement).
    memory_overhead_bytes: int
    # What a launch carries beyond the setup, in words, so a report can disclose it.
    added_flags: str

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
        """The engine's arguments (after the executable or image) that serve ``model``.

        With ``trace_dir`` (a path as the server sees it) the engine's profiler is enabled and
        will write its trace files there.
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

    def unstartable(self, current: Settings, applied: Settings) -> str | None:
        """Why ``applied``, a change from ``current``, cannot start or would lose CUDA graphs
        that ``current`` has for the batches it runs; None when it can be tried. A setup that
        ``current`` already fails to start is not reported for the same failure."""
        ...

    def default_tool_parser(self, model_dir: Path) -> str | None:
        """The engine's tool-call parser for the checkpoint's model family; None if unknown."""
        ...

    def kv_in_flight_tokens(self, settings: Settings, takes_images: bool) -> int:
        """Tokens scheduled but not yet settled, which a sliding-window layer keeps beside its
        window: the engine reserves KV for them per request. A checkpoint that takes images
        raises the batch size to its largest media item, unless media inputs are off."""
        ...

    def parallel_degree(self, settings: Settings) -> int:
        """How many GPUs the settings spread one model over (1 when it runs on one)."""
        ...

    def playbook(self) -> tuple[Entry, ...]:
        """The setting changes this engine offers, each with the bottlenecks it addresses."""
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

    def tool_calling_fix(self, settings: Settings, source: str) -> str | None:
        """Why a workload that offers tools cannot be served with ``settings``, with the fix for
        a setup registered from ``source`` ("command", "config" or "defaults"); None when it
        can."""
        ...

    def reproducibility_advice(self, source: str) -> str:
        """How to make answers repeatable when the noise floor is missing, for a setup
        registered from ``source`` ("command", "config" or "defaults")."""
        ...

    def loads_encoders(self, settings: Settings, modalities: Collection[str]) -> bool:
        """Whether a launch with ``settings`` loads the media encoders of ``modalities``."""
        ...

    def parse_quant_kernels(self, log_text: str) -> tuple[QuantKernel, ...]:
        """The quantized-linear kernels the engine's log says it selected; empty if none."""
        ...
