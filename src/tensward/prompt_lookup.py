"""Replay of prompt-lookup drafting over a run's own prompts and answers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from tensward.measurement import nearest_rank

MATCH_WORDS = 3
DRAFT_WORDS = 4
TAIL = 0.05

WORD = re.compile(r" ?\w+|\s+|[^\w\s]")


@dataclass(frozen=True, slots=True)
class Coverage:
    pooled: float
    tail: float


def words(text: str) -> list[str]:
    return WORD.findall(text)


def accepted(prompt: Sequence[str], answer: Sequence[str]) -> int:
    """Answer words a prompt-lookup drafter would have proposed and greedy decoding kept: at each
    step, the latest earlier occurrence of the context's last MATCH_WORDS words proposes the
    DRAFT_WORDS after it; the words that match the answer are accepted, and the model writes one."""
    context = list(prompt)
    latest: dict[tuple[str, ...], int] = {}
    indexed = MATCH_WORDS
    position = got = 0
    while position < len(answer):
        while indexed < len(context):
            latest[tuple(context[indexed - MATCH_WORDS : indexed])] = indexed
            indexed += 1
        end = latest.get(tuple(context[-MATCH_WORDS:])) if len(context) >= MATCH_WORDS else None
        draft = context[end : end + DRAFT_WORDS] if end is not None else []
        took = 0
        while (
            took < len(draft)
            and position + took < len(answer)
            and draft[took] == answer[position + took]
        ):
            took += 1
        got += took
        step = took + 1
        context += answer[position : position + step]
        position += step
    return got


def coverage(pairs: Iterable[tuple[str, str]]) -> Coverage | None:
    """Accepted share over all answer words, and the TAIL quantile of the per-answer shares."""
    counts: list[tuple[int, int]] = []
    for prompt, answer in pairs:
        answer_words = words(answer)
        if answer_words:
            counts.append((accepted(words(prompt), answer_words), len(answer_words)))
    if not counts:
        return None
    total = sum(n for _, n in counts)
    return Coverage(
        pooled=sum(got for got, _ in counts) / total,
        tail=nearest_rank([got / n for got, n in counts], TAIL),
    )
