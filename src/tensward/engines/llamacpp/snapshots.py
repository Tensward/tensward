"""The recurrent-state snapshots of a hybrid model, from llama-server's log at ``-lv 4``.

A hybrid model resumes a conversation only from a saved copy of its state. Each slot keeps its
own copies, so when requests begin with the same system prompt and tools, a slot that holds no
copy where that shared start ends reads the start again on every request.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

_NEW_PROMPT = re.compile(r"slot\s+\S+: id\s+(\d+) \| task \d+ \| new prompt,")
_RESTORED = re.compile(
    r"slot\s+\S+: id\s+(\d+) \| task \d+ \| restored context checkpoint .*? n_past = (\d+)"
)
_SPACING = re.compile(r"context checkpoints enabled, max = (\d+), min spacing = (\d+)")
_PICK = re.compile(r"slot\s+\S+: id\s+(\d+) \| task -?\d+ \| selected slot by (LRU|LCP similarity)")
_REREAD = re.compile(r"slot\s+\S+: id\s+(\d+) \| task \d+ \| forcing full prompt re-processing")
SLACK = 4  # a request resumes a few tokens before the start when the prompt ends with it


@dataclass(frozen=True, slots=True)
class Snapshots:
    """What a log says about the snapshots at the end of the shared start."""

    shared_start_tokens: int | None  # absent when no two slots resumed at the same place
    lost: int | None  # slots that served 2+ requests and never resumed at the shared start
    slots_served: int
    min_spacing: int | None


def read(log: str) -> Snapshots:
    served: Counter[int] = Counter()
    resumed: dict[int, set[int]] = defaultdict(set)
    for line in log.splitlines():
        if found := _NEW_PROMPT.search(line):
            served[int(found[1])] += 1
        elif found := _RESTORED.search(line):
            resumed[int(found[1])].add(int(found[2]))
    reached: Counter[int] = Counter(n for places in resumed.values() for n in places)
    shared = [n for n, slots in reached.items() if slots >= 2]
    start = max(shared, key=lambda n: (reached[n], n)) if shared else None
    lost = None
    if start is not None:
        lost = sum(
            1
            for slot, count in served.items()
            if count >= 2 and not any(start - SLACK <= n <= start for n in resumed[slot])
        )
    spacing = _SPACING.search(log)
    return Snapshots(start, lost, len(served), int(spacing[2]) if spacing else None)


def similarity_rereads(log: str) -> int | None:
    """Requests whose slot was picked by similarity to its content and then read the whole prompt
    again; None when the log has no slot picks."""
    last: dict[int, str] = {}
    found = picks = 0
    for line in log.splitlines():
        if pick := _PICK.search(line):
            last[int(pick[1])] = pick[2]
            picks += 1
        elif (again := _REREAD.search(line)) and last.get(int(again[1])) == "LCP similarity":
            found += 1
    return found if picks else None
