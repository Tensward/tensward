"""An explicit KV-cache size for a step that frees memory: the parent's logged KV bytes plus what
the step frees, capped so the step's real peak fits the GPU. vLLM's profiler sizes the pool from
the memory a dummy forward pass took; when that pass over-reserves (a media profile on a
text-only start, a larger activation peak), a fixed size keeps the pool the step should get."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .settings import Settings

KV_FLAG = "--kv-cache-memory-bytes"
MARGIN_BYTES = 1 << 30
BACK_OFF = 0.85  # after a sized step failed requests
FAILED_SHARE = 0.02  # a sized step that failed more than this share of its requests backs off
MIB = 1 << 20


def sized(
    parent_kv_bytes: int | None,
    *,
    freed_bytes: int,
    full_kv_bytes: int | None,
    peak_used_mib: float | None,
    total_mib: float | None,
    backed_off: int = 0,
) -> int | None:
    """``parent_kv_bytes`` plus ``freed_bytes``, at most the parent's "fully utilize" figure
    plus what the step frees, less the margin, and, with the parent's sampled peak, at most what
    leaves the margin free on the GPU; scaled down by ``BACK_OFF`` per earlier failed sized step.
    None without the parent's KV bytes, or when nothing is left."""
    if parent_kv_bytes is None:
        return None
    size = parent_kv_bytes + freed_bytes
    if full_kv_bytes is not None:
        size = min(size, full_kv_bytes + freed_bytes - MARGIN_BYTES)
    if peak_used_mib is not None and total_mib is not None:
        others = peak_used_mib * MIB - parent_kv_bytes - freed_bytes
        size = min(size, round(total_mib * MIB - others - MARGIN_BYTES))
    size = int(size * BACK_OFF**backed_off)
    return size if size > 0 else None


def with_kv_bytes(settings: Settings, size: int) -> Settings:
    return replace(settings, extra_args={**settings.extra_args, KV_FLAG: str(size)})


def _read(path: Path) -> dict | None:
    try:
        found = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) else None


def _sized_arg(args: object) -> bool:
    return isinstance(args, list) and any(str(arg).partition("=")[0] == KV_FLAG[2:] for arg in args)


def failed_sized_steps(project: Path | None, before: str | None) -> int:
    """The project's runs older than ``before`` that set a KV size and failed more than
    ``FAILED_SHARE`` of their requests."""
    runs = project / "runs" if project is not None else None
    if runs is None or not runs.is_dir():
        return 0
    count = 0
    for run_dir in runs.iterdir():
        if before is not None and run_dir.name >= before:
            continue
        record, metrics = _read(run_dir / "run.json"), _read(run_dir / "metrics.json")
        if record is None or metrics is None or not _sized_arg(record.get("engine_args")):
            continue
        failed, served = metrics.get("failed") or 0, metrics.get("succeeded") or 0
        count += failed > FAILED_SHARE * (failed + served)
    return count


def reference_run(project: Path | None, before: str | None) -> Path | None:
    """The newest run older than ``before`` that served requests with no engine arguments: the
    size of a change comes from it, so the same change always gives the same command."""
    runs = project / "runs" if project is not None else None
    if runs is None or not runs.is_dir():
        return None
    for run_dir in sorted(runs.iterdir(), reverse=True):
        if before is not None and run_dir.name >= before:
            continue
        record, metrics = _read(run_dir / "run.json"), _read(run_dir / "metrics.json")
        if record is not None and metrics is not None and not record.get("engine_args"):
            if metrics.get("succeeded"):
                return run_dir
    return None


def sampled_memory(run_dir: Path | None) -> tuple[float | None, float | None]:
    """The run's sampled GPU memory, (peak used, total) in MiB, from its window.json."""
    window = _read(run_dir / "window.json") if run_dir is not None else None
    memory = (window or {}).get("memory") or {}
    return memory.get("peak_used_mib"), memory.get("total_mib")
