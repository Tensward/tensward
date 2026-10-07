"""Structured-output quality: do the answers that must be JSON parse, and why not.

Observations about the model's output, not performance. A runaway-whitespace answer is one the
engine stopped at the token limit that is mostly whitespace: a JSON grammar that allows any
whitespace between tokens lets a model loop on it (seen on vLLM 0.30 with xgrammar at
temperature 0, when the prompt asks for something the schema cannot hold)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

RUNAWAY_WHITESPACE = 0.8  # share of an answer's characters that are whitespace


@dataclass(frozen=True, slots=True)
class StructuredAnswerStats:
    """Shares of the successful ``answers`` that must be JSON."""

    answers: int
    invalid_json: float
    runaway_whitespace: float  # stopped at the token limit, mostly whitespace
    cut_short: float  # stopped at the token limit
    runaway_prompts: tuple[str, ...] = ()  # the prompts whose answers ran away, in order


def parses_as_json(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def _runaway(text: str, finish_reason: str | None) -> bool:
    spaces = sum(character.isspace() for character in text)
    return finish_reason == "length" and bool(text) and spaces >= RUNAWAY_WHITESPACE * len(text)


def structured_answer_stats(
    answers: Sequence[tuple[str, str, str | None]],
) -> StructuredAnswerStats | None:
    """Over (prompt id, text, finish reason) triples; None when there is no answer to judge."""
    if not answers:
        return None
    count = len(answers)
    runaway = [prompt for prompt, text, finish in answers if _runaway(text, finish)]
    return StructuredAnswerStats(
        answers=count,
        invalid_json=sum(not parses_as_json(text) for _, text, _ in answers) / count,
        runaway_whitespace=len(runaway) / count,
        cut_short=sum(finish == "length" for _, _, finish in answers) / count,
        runaway_prompts=tuple(dict.fromkeys(runaway)),
    )
