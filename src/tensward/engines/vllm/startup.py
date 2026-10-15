"""How long a vLLM 0.30 server took to start, from its log."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from .logs import compile_cache

INIT = re.compile(r"init engine .*? took ([\d.]+) s")
COMPILE = re.compile(r"torch\.compile took ([\d.]+) s in total")
CAPTURE = re.compile(r"^.*?(\d\d)-(\d\d) (\d\d:\d\d:\d\d) .*Graph capturing finished", re.M)
WARMUP = re.compile(r"^.*?(\d\d)-(\d\d) (\d\d:\d\d:\d\d) .*JIT kernel warmup finished", re.M)


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
