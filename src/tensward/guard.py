"""The start-up guard: a step whose KV pool, logged while its server starts, is under half the
memory its parent's cache held at its peak would queue most of its requests, so it is stopped
before it is ready instead of measured for minutes. Pools are compared in GiB: tokens per GiB
move with the step budget on sliding-window models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .engines.vllm.command import DEFAULT_KV_MEMORY_FRACTION
from .engines.vllm.startup import kv_available_gib, kv_pool_tokens

if TYPE_CHECKING:
    from .history import PastRun

STOP_SHARE = 0.5  # a pool under this share of the parent's used GiB stops the step
GIB = 1 << 30
# The engine flags whose lower value shrinks the KV pool, and the run.json setting each sets.
SHRINKING = {
    "gpu-memory-utilization": "kv_memory_fraction",
    "max-model-len": "max_context_len",
    "max-num-seqs": "max_concurrent_requests",
}
KV_BYTES_FLAG = "kv-cache-memory-bytes"


@dataclass(frozen=True, slots=True)
class Armed:
    parent: str  # run id
    used_tokens: int  # the parent's pool tokens x its peak KV usage, for display
    used_gib: float  # the parent's pool GiB x its peak KV usage
    shrinks_kv: bool  # the step's own change is declared to shrink KV: never stops


def _values(args: frozenset[str]) -> dict[str, str | None]:
    return {arg.partition("=")[0]: (arg.partition("=")[2] if "=" in arg else None) for arg in args}


def _number(text: Any) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _shrinks(parent: PastRun, kv_bytes: float | None, step_args: frozenset[str]) -> bool:
    """Whether the step, against its parent, lowers a flag that sizes the KV pool. A value that
    cannot be compared counts as lowered, so the guard never stops on a guess."""
    before, after = _values(parent.engine_args), _values(step_args)
    defaults: dict[str, Any] = {"gpu-memory-utilization": DEFAULT_KV_MEMORY_FRACTION}
    for flag, setting in SHRINKING.items():
        if flag not in after and flag not in before:
            continue
        if flag not in after:
            return True
        old = _number(before.get(flag) or parent.settings.get(setting) or defaults.get(flag))
        new = _number(after[flag])
        if old is None or new is None or new < old:
            return True
    if KV_BYTES_FLAG in after:
        new = _number(after[KV_BYTES_FLAG])
        return kv_bytes is None or new is None or new < kv_bytes
    return False


def arm(parent: PastRun, parent_dir: Path, step_args: frozenset[str]) -> Armed | None:
    """None without a served parent, readable pool lines in its server.log and a peak usage."""
    if not parent.served:
        return None
    try:
        metrics = json.loads((parent_dir / "metrics.json").read_text("utf-8"))
        log = (parent_dir / "server.log").read_text("utf-8", errors="replace")
    except (OSError, ValueError):
        return None
    pool, peak = kv_pool_tokens(log), _number(metrics.get("peak_kv_usage"))
    pool_gib = kv_available_gib(log)
    if metrics.get("hybrid_cache") or not pool or not pool_gib or pool_gib < 0 or not peak:
        return None
    gib = _number(metrics.get("kv_available_gib"))
    kv_bytes = gib * GIB if gib is not None else None
    return Armed(
        parent.run_id, round(pool * peak), pool_gib * peak, _shrinks(parent, kv_bytes, step_args)
    )


def verdict(armed: Armed, log: str) -> str | None:
    """The stop reason when the log's KV pool is under half the parent's used GiB, e.g.
    "stopped at start-up: KV pool 0.36 GiB against the 3.96 GiB the parent used". A pool the
    engine logs as empty or negative is its own start failure, not this stop."""
    pool = kv_available_gib(log)
    if armed.shrinks_kv or pool is None or pool <= 0 or pool >= STOP_SHARE * armed.used_gib:
        return None
    return (
        f"stopped at start-up: KV pool {pool:.2f} GiB against the {armed.used_gib:.2f} GiB the "
        f"parent used (run {armed.parent})"
    )


def stop_record(
    armed: Mapping[str, Any] | None, reason: str, log: str, stopped_at_s: float
) -> dict[str, Any]:
    """The ``startup_guard`` record of a step that did not start, for its window.json."""
    return {
        "parent": armed["parent"] if armed else None,
        "reason": reason,
        "pool_tokens": kv_pool_tokens(log),
        "used_tokens": armed["used_tokens"] if armed else None,
        "pool_gib": kv_available_gib(log),
        "used_gib": armed["used_gib"] if armed else None,
        "stopped_at_s": round(stopped_at_s, 1),
    }
