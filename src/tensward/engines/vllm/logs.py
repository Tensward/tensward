"""What vLLM's start-up log says (vLLM 0.30 lines): the per-step token budget, the request
cap, and whether torch.compile ran or loaded a cached graph."""

from __future__ import annotations

import re
from typing import Literal

CHUNKED = re.compile(r"Chunked prefill is enabled with max_num_batched_tokens=(\d+)")
# The engine-config line: compile ranges end at max_num_batched_tokens.
RANGES = re.compile(r"'compile_ranges_endpoints': \[(?:\d+, )*(\d+)\]")
LOADED = "Directly load AOT compilation"
COMPILED = re.compile(r"torch\.compile took [\d.]+ s in total")


def step_budget(log: str) -> int | None:
    found = CHUNKED.search(log) or RANGES.search(log)
    return int(found.group(1)) if found else None


def compile_cache(log: str) -> Literal["warm", "cold"] | None:
    if LOADED in log:
        return "warm"
    return "cold" if COMPILED.search(log) else None


SEQS = re.compile(r"'max_num_seqs': (\d+)")


def server_cap(log: str) -> int | None:
    found = SEQS.search(log)
    return int(found.group(1)) if found else None
