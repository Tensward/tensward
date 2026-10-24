"""What a run's request trace says about its workload: the answer length, the time the load
lasted, and how much of the prompts an unbounded prefix cache would have served."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .measurement import nearest_rank

REQUESTS = "requests.jsonl"
Row = Mapping[str, Any]
Blocks = Mapping[int, tuple[int, tuple[str, ...]]]  # prompt index -> (tokens, chained block hashes)


def request_rows(run_dir: Path) -> list[Row]:
    """The run's requests in the order they were sent; empty without ``requests.jsonl``."""
    try:
        text = (run_dir / REQUESTS).read_text("utf-8")
    except OSError:
        return []
    rows = [json.loads(line) for line in text.splitlines() if line]
    sent = [row for row in rows if row.get("dispatch_ns") is not None]
    return sorted(sent, key=lambda row: (row["dispatch_ns"], row["request_id"]))


def output_tokens_p50(rows: Sequence[Row]) -> int | None:
    """The median answer length of the requests that succeeded."""
    lengths = [
        float(row["committed_output_tokens"])
        for row in rows
        if row["outcome"] == "success" and row.get("committed_output_tokens") is not None
    ]
    return int(nearest_rank(lengths, 0.5)) if lengths else None


def load_span_s(rows: Sequence[Row]) -> float | None:
    """Seconds from the first request sent to the last one ended."""
    ends = [row["terminal_ns"] for row in rows if row.get("terminal_ns") is not None]
    return (max(ends) - rows[0]["dispatch_ns"]) / 1e9 if rows and ends else None


def prefix_reuse(
    rows: Sequence[Row], blocks: Blocks, block_tokens: int
) -> tuple[float, float, int] | None:
    """``(possible_hit, shared_share, common_tokens)`` over the requests, in the order sent,
    from their prompts' chained block hashes. ``possible_hit``: the share of prompt tokens a
    cache that never evicts would have served, each request reusing the longest prefix of
    blocks an earlier request left. ``shared_share``: the share of prompt tokens in a leading
    run of blocks that another request also carries. ``common_tokens``: the tokens of the
    prefix every request starts with. None without a request whose prompt has block hashes."""
    sent = [blocks[row["prompt_index"]] for row in rows if row.get("prompt_index") in blocks]
    if not sent or block_tokens <= 0:
        return None
    carried = Counter(block for _, hashes in sent for block in set(hashes))
    seen: set[str] = set()
    tokens = hit = shared = 0
    for prompt_tokens, hashes in sent:
        tokens += prompt_tokens
        hit += block_tokens * _leading(hashes, lambda block: block in seen)
        shared += block_tokens * _leading(hashes, lambda block: carried[block] > 1)
        seen.update(hashes)
    common = 0
    while all(len(h) > common for _, h in sent) and len({h[common] for _, h in sent}) == 1:
        common += 1
    if not tokens:
        return None
    return hit / tokens, shared / tokens, common * block_tokens


def _leading(hashes: Sequence[str], test: Callable[[str], bool]) -> int:
    count = 0
    for block in hashes:
        if not test(block):
            break
        count += 1
    return count
