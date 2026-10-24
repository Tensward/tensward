"""What a vLLM run's two metrics scrapes and its server log say beyond the gauges the
measurement keeps: histogram growth between the scrapes, and the running-request counts of the
log's periodic stats lines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta

from ...measurement import nearest_rank
from ...prometheus import samples
from .start_phases import LOAD_LINE, _when

ITERATION_TOKENS = "vllm:iteration_tokens_total"
REQUEST_TPOT = "vllm:request_time_per_output_token_seconds"
MIN_STATS_LINES = 3


@dataclass(frozen=True, slots=True)
class Histogram:
    """The growth of one histogram family between two scrapes: ``counts[i]`` observations at or
    under ``bounds[i]``, the last bound infinite, summed over label sets."""

    bounds: tuple[float, ...]
    counts: tuple[float, ...]

    def quantile(self, q: float) -> float | None:
        """The ``q``-quantile, interpolated linearly inside its bucket; the lower bound of the
        open bucket when it falls there. None without growth."""
        if not self.counts or self.counts[-1] <= 0:
            return None
        rank, low, seen = q * self.counts[-1], 0.0, 0.0
        for bound, count in zip(self.bounds, self.counts):
            if count >= rank:
                if math.isinf(bound):
                    return low
                span = count - seen
                return low + (bound - low) * ((rank - seen) / span if span else 1.0)
            low, seen = bound, count
        return None

    def share_above(self, value: float) -> float | None:
        """The share of observations above the first bucket bound that is at least ``value``."""
        if not self.counts or self.counts[-1] <= 0:
            return None
        at_or_below = next(c for b, c in zip(self.bounds, self.counts) if b >= value)
        return 1 - at_or_below / self.counts[-1]


def _buckets(scrape: str, family: str) -> dict[float, float]:
    found: dict[float, float] = {}
    for name, labels, value in samples(scrape):
        if name == f"{family}_bucket" and "le" in labels and math.isfinite(value):
            bound = math.inf if labels["le"] == "+Inf" else float(labels["le"])
            found[bound] = found.get(bound, 0.0) + value
    return found


def growth(before: str, after: str, family: str) -> Histogram | None:
    """``family``'s histogram growth from the ``before`` scrape to the ``after`` one; None when
    the later scrape has none of it."""
    last, first = _buckets(after, family), _buckets(before, family)
    if not last or math.inf not in last:
        return None
    bounds = tuple(sorted(last))
    return Histogram(bounds, tuple(last[b] - first.get(b, 0.0) for b in bounds))


def running_counts(log: str, span_s: float) -> list[int]:
    """The running-request counts of the log's stats lines that cover the last ``span_s``
    seconds of load. The load is taken to run from the first line that shows a request to the
    last, so the idle lines before and after it are left out."""
    lines = [(_when(*found.group(1, 2, 3)), int(found[4]), int(found[5]))
             for found in LOAD_LINE.finditer(log)]  # fmt: skip
    busy = [i for i, (_, running, waiting) in enumerate(lines) if running or waiting]
    if not busy:
        return []
    end = lines[busy[-1]][0]
    return [
        running
        for at, running, _ in lines[busy[0] : busy[-1] + 1]
        if (end - at) % timedelta(days=366) < timedelta(seconds=span_s)
    ]


def running_percentile(counts: list[int], q: float) -> float | None:
    """The ``q``-quantile of the counts; None from fewer than ``MIN_STATS_LINES`` lines."""
    if len(counts) < MIN_STATS_LINES:
        return None
    return float(nearest_rank([float(count) for count in counts], q))
