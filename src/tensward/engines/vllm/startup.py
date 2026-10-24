"""How long a vLLM 0.30 server took to start, and the memory facts it logged."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from .logs import compile_cache

INIT = re.compile(r"init engine .*? took ([\d.]+) s")
COMPILE = re.compile(r"torch\.compile took ([\d.]+) s in total")
CAPTURE = re.compile(r"^.*?(\d\d)-(\d\d) (\d\d:\d\d:\d\d) .*Graph capturing finished", re.M)
WARMUP = re.compile(r"^.*?(\d\d)-(\d\d) (\d\d:\d\d:\d\d) .*JIT kernel warmup finished", re.M)
KV_POOL = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
KV_MEMORY_PREFIX = (
    r"(?:Available KV cache memory: |reserved (?=[\d.]+ GiB memory for KV Cache "
    r"as specified by kv_cache_memory_bytes config))"
)
KV_AVAILABLE = re.compile(KV_MEMORY_PREFIX + r"(-?[\d.]+) GiB")
USAGE = re.compile(
    r"Actual usage is ([\d.]+) GiB for consumed memory \(weights \+ non-torch\), "
    r"([\d.]+) GiB for peak activation, and ([\d.]+) GiB for CUDAGraph memory.*?"
    r"--kv-cache-memory=(-?\d+).*?to fit into requested memory, or `--kv-cache-memory=(-?\d+)`"
)
FORCED = re.compile(r"forcing (\w+) backend")
NO_KV_MEMORY = "No available memory for the cache blocks"


def _at(found: re.Match[str]) -> datetime:
    return datetime.strptime(f"{found[1]}-{found[2]} {found[3]}", "%m-%d %H:%M:%S")


def _seconds(log: str, pattern: re.Pattern[str]) -> float | None:
    found = pattern.search(log)
    return float(found[1]) if found else None


def startup(log: str, recorded: str | None = None) -> dict[str, float | bool | None]:
    """The start times in ``log``: the engine's init, torch.compile, the CUDA graph captures,
    the gap from the end of the kernel warmup to the end of the second capture (None with one
    capture), and whether the compile cache was warm (``recorded``, the run's own record, wins
    over the log)."""
    captures = [_at(found) for found in CAPTURE.finditer(log)]
    warmup = WARMUP.search(log)
    gap_s = None
    if warmup and len(captures) > 1:
        gap_s = ((captures[1] - _at(warmup)) % timedelta(days=366)).total_seconds()
    state = recorded or compile_cache(log)
    return {
        "init_engine_s": _seconds(log, INIT),
        "compile_s": _seconds(log, COMPILE),
        "captures": len(captures),
        "warmup_gap_s": gap_s,
        "compile_cache_warm": None if state is None else state == "warm",
    }


def kv_pool_tokens(log: str) -> int | None:
    """The KV pool the engine logged, in tokens."""
    found = KV_POOL.search(log)
    return int(found[1].replace(",", "")) if found else None


def kv_available_gib(log: str) -> float | None:
    """The KV memory the engine logged before it sized the pool, in GiB: logged at one or two
    decimals and negative when the weights and activations already exceed the share."""
    found = KV_AVAILABLE.search(log)
    return float(found[1]) if found else None


def memory_usage(log: str) -> dict[str, float] | None:
    """What the engine's memory line says it used (GiB), the KV bytes it would give for the
    requested utilization (``fit_kv_bytes``, negative when the weights, activations and graphs
    already exceed it) and for the whole GPU (``full_kv_bytes``); None without the line."""
    found = USAGE.search(log)
    if not found:
        return None
    return {
        "consumed_gib": float(found[1]),
        "peak_activation_gib": float(found[2]),
        "graphs_gib": float(found[3]),
        "fit_kv_bytes": int(found[4]),
        "full_kv_bytes": int(found[5]),
    }


def forced_backend(log: str) -> str | None:
    """The attention backend the engine forced, when it says so."""
    found = FORCED.search(log)
    return found[1] if found else None


def no_kv_memory(log: str) -> bool:
    return NO_KV_MEMORY in log
