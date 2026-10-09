"""The engine's prompt capacity: prompt tokens per second when prompt work fills the steps
(spec 2.1). Measured by a short probe before the workload, outside its window: prompts of the
workload's median length (at most PROBE_MAX_TOKENS), each opened with a unique line and asking
for one token, enough in flight to fill the step budget. A one-token request completes when its
prompt is done, so R is the prompt tokens of the requests completing after the ramp over that
time, counted at the client. Cached in the project per setup."""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Awaitable, Callable, Sequence

from .inputs import PromptEntry
from .workload import ChatMessage, TextPart

PROBE_SECONDS = 24.0
RAMP = 1 / 8  # completions in the first eighth of the probe are left out (3 s of 24)
LONGEST = 2.5  # the probe is extended up to this many times its length (60 s) for completions
MIN_COMPLETIONS = 20  # fewer completions after the ramp leave R unknown
SHORT = 0.8  # probe prompts the engine counted under this share of the target leave R unknown
LONG = 1.2  # and over this share of it
HUNG_SECONDS = 30.0  # past the longest probe, a request still open fails the probe
PROBE_MAX_TOKENS = 4096  # longer prompts are cut to this many tokens
STEPS_IN_FLIGHT = 2  # prompt tokens in flight: this many step budgets
MIN_IN_FLIGHT = 4
FILL = 1.5  # the probe's prompts in flight must cover this many step budgets
FILE = "prompt_capacity.json"


@dataclass(frozen=True, slots=True)
class Key:
    model: str  # the checkpoint fingerprint
    gpu: str
    budget: int
    cap: int
    parallel: int  # tensor- and pipeline-parallel degree
    kv_cache_dtype: str
    prefix_caching: bool
    quantization: str  # as served ("none" when not set)
    cuda_graphs: bool
    engine_version: str
    dtype: str


@dataclass(frozen=True, slots=True)
class CapacityResult:
    tok_s: float | None
    # "probed", "cached", "cannot-fill", "not-user-turn", "too-few-completions", "short-prompts",
    # "long-prompts", "cached-prefix", "failed", "pooled" or "unknown-setup"
    source: str
    key: Key | None
    failed_requests: int = 0
    completions: int = 0
    counter_tok_s: float | None = None  # the engine's computed-prompt counter over the probe
    target_tokens: int = 0
    median_prompt_tokens: float | None = None


def width(budget: int, cap: int, prompt_tokens: float) -> int:
    lanes = math.ceil(STEPS_IN_FLIGHT * budget / max(prompt_tokens, 1.0))
    return min(max(MIN_IN_FLIGHT, lanes), cap)


def probe_length(prompt_tokens: Sequence[int]) -> int:
    """The probe prompts' length: the workload's median, at most PROBE_MAX_TOKENS."""
    return min(int(median(prompt_tokens)), PROBE_MAX_TOKENS) if prompt_tokens else 0


def _with_text(message: ChatMessage, change: Callable[[str], str]) -> ChatMessage:
    """``message`` with ``change`` applied to its text: the whole text, or its first text part,
    so image parts stay as they are."""
    if isinstance(message.content, str):
        return message.model_copy(update={"content": change(message.content)})
    parts = list(message.content)
    for i, part in enumerate(parts):
        if isinstance(part, TextPart):
            parts[i] = part.model_copy(update={"text": change(part.text)})
            return message.model_copy(update={"content": tuple(parts)})
    return message.model_copy(update={"content": (TextPart(type="text", text=change("")), *parts)})


def unique(entry: PromptEntry, i: int, extra_words: int) -> PromptEntry:
    """The whole record as the workload sends it (every message, its tools), opened by a line no
    other prompt has, with ``extra_words`` added to (or, when negative, cut from) its last
    message's text, a user turn (``probe`` refuses records that end otherwise), and one output
    token."""
    tag = f"[probe {uuid.uuid4().hex} {i}]\n"

    def resized(text: str) -> str:
        words = text.split()
        if not words or extra_words == 0:
            return text
        if extra_words < 0:
            return " ".join(words[: max(1, len(words) + extra_words)])
        filler = (words * (extra_words // len(words) + 1))[:extra_words]
        return " ".join([*words, *filler])

    update: dict[str, object] = {"id": f"probe-{i}", "max_tokens": 1}
    if entry.messages is None:
        update["prompt"] = tag + resized(entry.prompt or "")
        return entry.model_copy(update=update)
    messages = list(entry.messages)
    messages[-1] = _with_text(messages[-1], resized)
    messages[0] = _with_text(messages[0], lambda text: tag + text)
    update["messages"] = tuple(messages)
    return entry.model_copy(update=update)


def _rows(project: Path) -> list[dict[str, Any]]:
    path = project / FILE
    rows = json.loads(path.read_text()) if path.is_file() else []
    if not isinstance(rows, list):
        raise ValueError(f"{FILE} is not a list")
    return rows


def lookup(project: Path, key: Key) -> float | None:
    """The cached capacity for ``key``; None when there is none or the cache cannot be read."""
    try:
        for row in _rows(project):
            if row["key"] == asdict(key):
                return float(row["tok_s"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def store(project: Path, key: Key, tok_s: float, run_id: str) -> None:
    """Cache ``tok_s`` for ``key``; a cache that cannot be read or written is left as it is."""
    try:
        rows = [row for row in _rows(project) if row["key"] != asdict(key)]
        rows.append({"key": asdict(key), "tok_s": tok_s, "run_id": run_id})
        (project / FILE).write_text(json.dumps(rows, indent=1) + "\n")
    except (OSError, ValueError, KeyError, TypeError):
        pass


async def probe(
    *,
    project: Path,
    key: Key | None,
    entries: Sequence[PromptEntry],
    prompt_tokens: Sequence[int],
    words_per_token: float,
    send: Callable[[PromptEntry, int], Awaitable[tuple[int, int] | None]],
    computed: Callable[[], Awaitable[float | None]],
    run_id: str,
) -> CapacityResult:
    """The cached capacity, else a probe. ``prompt_tokens`` are the counted tokens of ``entries``,
    in order. ``send`` posts one request for one output token and returns the prompt tokens the
    engine counted, or None when the request failed; nothing here raises to the run."""
    if key is None:
        return CapacityResult(None, "unknown-setup", None)
    cached = lookup(project, key)
    if cached is not None:
        return CapacityResult(cached, "cached", key)
    tokens = probe_length(prompt_tokens)
    lanes = width(key.budget, key.cap, tokens)
    if not entries or lanes * tokens < FILL * key.budget:
        return CapacityResult(None, "cannot-fill", key)
    if any(entry.messages and entry.messages[-1].role != "user" for entry in entries):
        return CapacityResult(None, "not-user-turn", key)
    seconds = PROBE_SECONDS
    start, sent, failed = time.monotonic(), 0, 0
    # (seconds since start, prompt tokens computed, prompt tokens) per completion
    done: list[tuple[float, int, int]] = []

    def after_ramp() -> int:
        return sum(1 for at, _, _ in done if at >= seconds * RAMP)

    def running() -> bool:
        elapsed = time.monotonic() - start
        if elapsed < seconds:
            return True
        return after_ramp() < MIN_COMPLETIONS and elapsed < seconds * LONGEST

    async def lane() -> None:
        nonlocal sent, failed
        while running():
            i, sent = sent, sent + 1
            entry, own = entries[i % len(entries)], prompt_tokens[i % len(entries)]
            try:
                counted = await send(unique(entry, i, int((tokens - own) * words_per_token)), i)
            except Exception:  # a probe request is advice: its failure never stops the run
                counted = None
            if counted is None:
                failed += 1
            else:  # prompt tokens less those the engine said it served from its cache
                done.append((time.monotonic() - start, counted[0] - counted[1], counted[0]))

    try:
        before = await computed()
        await asyncio.wait_for(
            asyncio.gather(*(lane() for _ in range(lanes))), seconds * LONGEST + HUNG_SECONDS
        )
        after, end = await computed(), time.monotonic() - start
    except Exception:  # a failed scrape, or a request hung past the bound: no capacity, recorded
        return CapacityResult(None, "failed", key, failed, len(done))
    if (
        before is not None
        and after is not None
        and after - before < SHORT * sum(n for _, n, _ in done)
    ):
        # The engine computed less than the client counted: shared template text before the
        # tag was served from the prefix cache (review-rev6 concern 1).
        return CapacityResult(None, "cached-prefix", key, failed, len(done), (after - before) / end)
    after_ramp_done = sorted(row for row in done if row[0] >= seconds * RAMP)
    if len(after_ramp_done) < MIN_COMPLETIONS:
        return CapacityResult(None, "too-few-completions", key, failed, len(after_ramp_done))
    counted = median(gross for _, _, gross in after_ramp_done)  # gross: cache hits are not length
    # Short: the template dropped text, so R would be far too low. Long: text outside the last
    # user turn kept the prompts over the target.
    if not SHORT * tokens <= counted <= LONG * tokens:
        source = "short-prompts" if counted < SHORT * tokens else "long-prompts"
        return CapacityResult(
            None, source, key, failed, len(after_ramp_done), None, tokens, counted
        )
    first, last = after_ramp_done[0][0], after_ramp_done[-1][0]
    # From the first completion after the ramp to the last: the prompts finished in between.
    tok_s = sum(n for _, n, _ in after_ramp_done[1:]) / (last - first)
    counter = (after - before) / end if before is not None and after is not None and end else None
    store(project, key, tok_s, run_id)
    return CapacityResult(
        tok_s, "probed", key, failed, len(after_ramp_done), counter, tokens, counted
    )
