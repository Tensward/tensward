"""What Tensward needs from a serving engine, in engine-neutral terms.

``Settings`` names the knobs Tensward reads and tunes; an :class:`Engine` maps them to its own
command line and reads its own metrics back as :class:`EngineSignals`. Everything that is
specific to one engine (flag names, metric names, image, credential variable) lives behind
that engine's implementation; callers never see it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import httpx


@dataclass(frozen=True, slots=True)
class Settings:
    """The serving knobs Tensward reads and tunes, independent of the engine.

    ``quantization`` is an operator override of the quantization method; None lets the engine
    detect it from the checkpoint's own configuration, which is what registration records.
    ``tool_calling`` lets the model answer with tool calls, which needs ``tool_parser``: the
    engine's name for the parser of this model family's tool-call format (None when unknown).
    ``async_scheduling`` overlaps CPU scheduling with GPU execution and ``api_server_count`` is
    the number of frontend (tokenization, output) processes; None lets the engine choose.
    ``extra_args`` holds engine-specific flags that have no neutral setting; the engine appends
    them to its command line as given. ``extra_env`` is environment that changes the engine's
    behaviour. A None knob is not passed: the engine decides, which is what a customer's own
    command line without that flag gets.
    """

    dtype: str | None = None
    max_concurrent_requests: int | None = None
    max_context_len: int | None = None
    kv_memory_fraction: float | None = None
    prefill_batch_tokens: int | None = None
    kv_cache_dtype: str = "auto"
    prefix_caching: bool | None = None
    quantization: str | None = None
    cuda_graphs: bool = True
    tool_calling: bool = False
    tool_parser: str | None = None
    async_scheduling: bool | None = None
    api_server_count: int | None = None
    extra_args: Mapping[str, str | bool | None] = field(default_factory=dict)
    extra_env: Mapping[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """The settings that differ from a bare engine launch, in words."""
        given = [
            f"{f.name}={getattr(self, f.name)}"
            for f in fields(self)
            if f.name not in ("extra_args", "extra_env")
            and getattr(self, f.name) not in (None, f.default)
        ]
        given += [f"{flag} {value}" for flag, value in self.extra_args.items()]
        given += [f"{name}={value}" for name, value in self.extra_env.items()]
        return ", ".join(given) or "the engine's defaults"

    @property
    def serves_tools(self) -> bool:
        """Whether the engine will accept requests that offer tools."""
        return self.tool_calling and self.tool_parser is not None

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Settings:
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ParsedSetup:
    """A customer's engine command line, understood.

    ``model`` is what the command serves (a path or hub id, None if it named none), ``notes``
    say which parts Tensward dropped or ignored and why, and ``text`` is the command as given
    with any secrets blanked.
    """

    settings: Settings
    model: str | None
    image: str | None
    notes: tuple[str, ...]
    text: str  # the command with secrets blanked
    gpus: tuple[str, ...] = ()  # the device indices the command selects, if it names any


@dataclass(frozen=True, slots=True)
class EngineSignals:
    """One reading of an engine's metrics. ``None`` means the engine did not report it.

    The gauges are instantaneous; the others are cumulative counters.
    """

    kv_usage: float | None = None  # fraction of the KV cache in use
    running: float | None = None
    waiting: float | None = None
    preemptions: float | None = None
    prefix_cache_hits: float | None = None  # prompt tokens served from the prefix cache
    prefix_cache_queries: float | None = None  # prompt tokens looked up in the prefix cache
    prompt_tokens: float | None = None  # prefill tokens processed
    generation_tokens: float | None = None


@dataclass(frozen=True, slots=True)
class QuantKernel:
    """One quantized-linear kernel an engine reported selecting, and whether it is a slow path."""

    name: str
    layer: str  # what the engine used it for, in the engine's words
    slow: bool


class Engine(Protocol):
    """One serving engine."""

    name: str
    default_image: str  # container image used when none is given
    local_command: tuple[str, ...]  # server command for the local-process runtime
    api_key_env: str  # environment variable that carries the API key to the server
    health_path: str
    models_path: str  # OpenAI-style model list, used to confirm which model is served
    metrics_path: str
    tokenize_path: str
    trace_env: Mapping[str, str]  # environment that makes a traced launch's trace more useful
    # extra arguments for a launch wrapped by a kernel profiler: the engine starts its own CUDA
    # profiler, so start_trace/stop_trace gate the profiler (ncu --profile-from-start off)
    counters_args: Sequence[str]
    trace_step_scope: str  # name of the CPU annotation wrapping one model step in its trace
    # What the engine does for a neutral setting left unset (a ``Settings`` field name -> words),
    # so a report can say what really ran.
    defaults: Mapping[str, str]
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

    async def start_trace(self, client: httpx.AsyncClient, server_url: str) -> None:
        """Begin recording the engine's profiler trace."""
        ...

    async def stop_trace(self, client: httpx.AsyncClient, server_url: str) -> None:
        """Stop recording; returns once the trace files are written."""
        ...

    def collect_trace(self, trace_dir: Path) -> list[Path]:
        """The PyTorch-profiler Chrome trace files the engine wrote into ``trace_dir``."""
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

    def default_tool_parser(self, model_dir: Path) -> str | None:
        """The engine's tool-call parser for the checkpoint's model family; None if unknown."""
        ...

    def parse_signals(self, metrics_text: str) -> EngineSignals:
        """Read the engine's Prometheus text into engine-neutral signals."""
        ...

    def parse_quant_kernels(self, log_text: str) -> tuple[QuantKernel, ...]:
        """The quantized-linear kernels the engine's log says it selected; empty if none."""
        ...
