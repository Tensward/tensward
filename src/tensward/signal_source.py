"""Collecting an engine's signals: one poll of its metrics, and what the finished run itself
shows (responses, the server log of the window). An engine's adapter provides the source."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

import httpx

from .capabilities import Resolved
from .capture import CapturedResponse
from .client import RequestRecord
from .prometheus import SignalMap, read_signals, replica_count, samples, summed_counters
from .settings import EngineSignals


@dataclass(frozen=True, slots=True)
class Scrape:
    """One read of the engine's metrics: the text as served and what it says, parsed once."""

    text: str
    signals: EngineSignals
    # Other endpoints the source read in the same poll: file name -> text, written beside the
    # metrics text (a source that polls several endpoints; empty for vLLM).
    extra: Mapping[str, str] = field(default_factory=dict, hash=False)
    replicas: int = 1  # engine replicas the text came from


@dataclass(frozen=True, slots=True, kw_only=True)
class RunInputs:
    records: Sequence[RequestRecord]  # dispatched inside the measured window
    responses: Mapping[str, CapturedResponse]  # by request id
    window_s: float
    server_log: str  # the log written during the window
    resolved: Resolved | None


@dataclass(frozen=True, slots=True, kw_only=True)
class RunSignals:
    neutral: EngineSignals = field(default_factory=EngineSignals)  # window values
    # <engine>.<name> -> value
    engine: Mapping[str, float] = field(default_factory=dict, hash=False)
    window_batch: float | None = None  # an exact running batch, when the engine counts one


class SignalSource(Protocol):
    # A read-only property: a frozen dataclass such as PrometheusSource satisfies it (mypy reads
    # a frozen field as read-only, which a plain protocol attribute would refuse).
    @property
    def replica_labels(self) -> tuple[str, ...]:
        """The labels that tell an engine's replicas apart."""
        ...

    async def read(self, client: httpx.AsyncClient, server_url: str) -> Scrape | None:
        """One poll; None when the engine could not be read."""
        ...

    def from_run(self, run: RunInputs) -> RunSignals:
        """What the finished run shows beyond the polls; empty when the polls say it all."""
        ...


@dataclass(frozen=True, slots=True)
class PrometheusSource:
    """An engine that exports its signals as Prometheus text at ``metrics_path``."""

    metrics_path: str
    signals: SignalMap
    replica_labels: tuple[str, ...] = ()

    async def read(self, client: httpx.AsyncClient, server_url: str) -> Scrape | None:
        try:
            response = await client.get(f"{server_url}{self.metrics_path}")
            response.raise_for_status()
        except httpx.HTTPError:
            return None
        text = response.text
        parsed = list(samples(text))
        replicas = replica_count(parsed, self.signals, self.replica_labels)
        if replicas == 1:
            return Scrape(text, read_signals(text, self.signals))
        summed = summed_counters(parsed, self.signals, self.replica_labels)
        return Scrape(text, summed, replicas=replicas)

    def from_run(self, run: RunInputs) -> RunSignals:
        return RunSignals()
