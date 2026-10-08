"""The serving knobs and engine readings Tensward reasons about, independent of any engine."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from .errors import PROJECT_RECORD_INVALID, PreflightError


@dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    """The serving knobs Tensward reads and tunes, independent of the engine.

    ``quantization`` is an operator override of the quantization method; None lets the engine
    detect it from the checkpoint's own configuration, which is what registration records.
    ``tool_calling`` lets the model answer with tool calls, which needs ``tool_parser``: the
    engine's name for the parser of this model family's tool-call format (None when unknown).
    ``async_scheduling`` overlaps CPU scheduling with GPU execution and ``api_server_count`` is
    the number of frontend (tokenization, output) processes; None lets the engine choose.
    ``media_inputs`` False serves only the text model of an image+text checkpoint: the engine
    reserves no memory for image or video encoders. ``media_limits`` caps the inputs of each
    modality in one prompt (``{"image": 1, "video": 0}``); None leaves the engine's own limits.
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
    media_inputs: bool | None = None
    media_limits: Mapping[str, int] | None = None
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
        unknown = sorted(set(data) - {f.name for f in fields(cls)})
        if unknown:
            raise PreflightError(
                PROJECT_RECORD_INVALID,
                f"these settings were written by a newer Tensward ({', '.join(unknown)}); "
                "upgrade Tensward to read them",
            )
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
    """One reading of an engine's metrics. An engine leaves a field ``None`` when it cannot
    report it.

    The gauges and the two ``kv_`` capacities are instantaneous; the others are cumulative
    counters. ``prompt_tokens`` counts every prompt token, cached ones too; ``iterations``
    counts engine steps.
    """

    kv_usage: float | None = None  # fraction of the KV cache in use
    running: float | None = None
    waiting: float | None = None
    preemptions: float | None = None
    prefix_cache_hits: float | None = None  # prompt tokens served from the prefix cache
    prefix_cache_queries: float | None = None  # prompt tokens looked up in the prefix cache
    prompt_tokens: float | None = None  # prefill tokens processed
    prompt_tokens_computed: float | None = None  # of those, run through prefill
    generation_tokens: float | None = None
    iterations: float | None = None  # engine steps run
    kv_capacity_tokens: float | None = None  # tokens the KV cache holds, as the engine counts them
    kv_max_concurrency: float | None = None  # full-length requests it holds at once
    kv_blocks: float | None = None  # blocks in the KV cache pool
    kv_block_tokens: float | None = None  # tokens per attention block
    hybrid_cache: bool = False  # the cache also holds linear-attention (Mamba) state pages
    queue_seconds: float | None = None  # time requests waited to be scheduled, summed
    prefill_seconds: float | None = None  # time spent computing prompts, summed over requests
    spec_drafts: float | None = None  # speculative drafts proposed
    spec_accepted_tokens: float | None = None  # draft tokens the target model accepted
    frontend_cpu_seconds: float | None = None  # CPU time of the engine's API-server process


SPECULATION_SIGNALS = frozenset({"spec_drafts", "spec_accepted_tokens"})
"""Signals an engine reports only while speculative decoding is on."""

GAUGE_SIGNALS = frozenset({
    "kv_usage", "running", "waiting", "kv_capacity_tokens", "kv_max_concurrency", "kv_blocks",
    "kv_block_tokens", "hybrid_cache",
})  # fmt: skip
"""Signals that are the state of one engine (gauges, capacities, info labels): never summed
over replicas, and never compared with a sum."""


@dataclass(frozen=True, slots=True)
class QuantKernel:
    """One quantized-linear kernel an engine reported selecting, and whether it is a slow path."""

    name: str
    layer: str  # what the engine used it for, in the engine's words
    slow: bool


@dataclass(frozen=True, slots=True)
class Availability:
    """Whether an engine can be started one way on this machine, its version when known, how to
    get it, and why not (``docker_missing``, ``docker_unreachable``, ``image_absent`` or
    ``command_missing``; None when it is available)."""

    available: bool
    version: str | None
    how_to_get: str
    reason: str | None = None
