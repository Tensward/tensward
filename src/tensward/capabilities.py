"""What an engine reports and can change, as its adapter declares it, and what it chose at
launch. Neutral: no engine's names appear here."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

Quality = Literal["exact", "derived", "approximate", "absent"]
Source = Literal["metrics", "log", "api", "client", "trace", "model", "host"]
Cadence = Literal["step", "request_end", "interval"]


@dataclass(frozen=True, slots=True, kw_only=True)
class SignalSupport:
    """How an engine reports one neutral signal. ``exact``: its own count of the neutral meaning;
    ``derived``: computed from exact counts; ``approximate``: an estimate with a known bias;
    ``absent``: not reported, ``absent_reason`` says who lacks it."""

    quality: Quality
    absent_reason: str = ""
    source: Source | None = None
    how: str = ""  # e.g. "vllm:kv_cache_usage_perc"
    bias: str = ""  # the known bias when quality is "approximate" (not shown in reports)
    cadence: Cadence = "step"
    interval_steps: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class LeverSupport:
    values: frozenset[str] | None = None  # allowed values; for speculation, the methods
    caveat: str = ""  # appended to the costs of an entry that turns this lever


@dataclass(frozen=True, slots=True, kw_only=True)
class Capabilities:
    # Every EngineSignals field, plus <engine>.* names
    signals: Mapping[str, SignalSupport] = field(hash=False)
    # Neutral and engine-only levers it can realise
    levers: Mapping[str, LeverSupport] = field(hash=False)
    trace: Literal["kineto"] | None  # the trace format it can produce
    counters: bool  # it can gate a kernel profiler to the measured window
    shared_prefix_kv: bool  # concurrent requests share one cached prefix copy
    min_cacheable_prefix_tokens: int  # smallest prefix the cache can reuse
    # Workload field -> the sentence that refuses it
    workload: Mapping[str, str] = field(hash=False)
    polls_metrics: bool = True  # False: the engine has no endpoint to poll
    resolves_at_launch: bool = False  # it chooses lever values or facts Tensward reads


@dataclass(frozen=True, slots=True, kw_only=True)
class Resolved:
    """What the engine chose at launch where the settings left it free: lever values, and facts
    that are not levers (``context_per_request``)."""

    levers: Mapping[str, Any] = field(hash=False)
    facts: Mapping[str, Any] = field(hash=False)
