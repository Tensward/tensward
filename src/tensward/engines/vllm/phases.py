"""The engine's own timing of a request's phases, read from a run's two metrics scrapes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...prometheus import samples

if TYPE_CHECKING:
    from pathlib import Path

PREFILL = "vllm:request_prefill_time_seconds"


def _totals(path: Path) -> tuple[float, float] | None:
    try:
        text = path.read_text("utf-8")
    except OSError:
        return None
    found = {"sum": 0.0, "count": 0.0}
    for name, _labels, value in samples(text):
        for part in found:
            if name == f"{PREFILL}_{part}":
                found[part] += value
    return found["sum"], found["count"]


def prefill_ms_per_request(run_dir: Path | None) -> float | None:
    """The engine's mean prefill time per request over the measured window: the increase of
    its prefill-time histogram between the run's two scrapes, over the requests it counted."""
    if run_dir is None:
        return None
    before = _totals(run_dir / "metrics_before.prom")
    after = _totals(run_dir / "metrics_after.prom")
    if before is None or after is None or after[1] <= before[1]:
        return None
    return (after[0] - before[0]) / (after[1] - before[1]) * 1e3
