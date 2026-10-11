"""The per-request latency limits that goodput is counted against, as the user declared them."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from .measurement import Measurement

# The configuration's ``slos`` keys Tensward reads, and the limit each declares.
PROJECT_LIMITS = {"ttft_p95_ms": "ttft_ms", "tpot_p95_ms": "tpot_ms"}


@dataclass(frozen=True, slots=True)
class Slo:
    """The per-request latency a request must meet to count towards goodput. An infinite limit
    is one nobody declared: every request meets it."""

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

    @property
    def declared(self) -> dict[str, float]:
        """The declared limits by name (``ttft_ms``, ``tpot_ms``)."""
        limits = (("ttft_ms", self.ttft_ms), ("tpot_ms", self.tpot_ms))
        return {name: value for name, value in limits if math.isfinite(value)}

    def describe(self) -> str:
        declared = self.declared
        parts = [
            f"{label} <= {declared[name]:g} ms"
            for name, label in (("ttft_ms", "TTFT"), ("tpot_ms", "TPOT"))
            if name in declared
        ]
        return f"{' and '.join(parts)} per request" if parts else "none declared"


DEFAULT_SLO = Slo()  # the API's default, as in 0.3.9
NO_LIMITS = Slo(math.inf, math.inf)  # tensward analyse with nothing declared


def declared_slo(ttft_ms: float | None, tpot_ms: float | None, project: Mapping[str, float]) -> Slo:
    """The limits the user declared: a command-line value, else the configuration's ``slos``
    entry, else none."""
    found = {limit: project[key] for key, limit in PROJECT_LIMITS.items() if key in project}
    ttft = ttft_ms if ttft_ms is not None else found.get("ttft_ms")
    tpot = tpot_ms if tpot_ms is not None else found.get("tpot_ms")
    return Slo(math.inf if ttft is None else ttft, math.inf if tpot is None else tpot)
