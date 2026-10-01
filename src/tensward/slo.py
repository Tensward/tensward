"""The per-request latency limits that goodput is counted against."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .measurement import Measurement


@dataclass(frozen=True, slots=True)
class Slo:
    """The per-request latency a request must meet to count towards goodput."""

    ttft_ms: float = 1000.0
    tpot_ms: float = 100.0

    def met(self, ttft_ms: float | None, tpot_ms: float | None) -> bool:
        """Whether one request met both limits. A one-token answer has no TPOT and cannot break
        that limit; a request without a first token misses the TTFT limit."""
        return (
            ttft_ms is not None
            and ttft_ms <= self.ttft_ms
            and (tpot_ms is None or tpot_ms <= self.tpot_ms)
        )

    def met_at_p95(self, measurement: Measurement) -> bool:
        """Whether the run's TTFT p95 and TPOT p95 are both within the limits."""
        return self.met(measurement.ttft_p95_ms, measurement.tpot_p95_ms)


DEFAULT_SLO = Slo()
