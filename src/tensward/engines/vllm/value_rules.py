"""The values of vLLM levers that need arithmetic over what a run measured: a knowledge record
names one of :data:`RULES` as its ``change.rule``."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

SPEC_RUNNING_P90 = 16  # a batch-size schedule is offered at or below this p90 running count
SPEC_CUT = 16  # the running count above which drafting stops, without a decode-bound cut
BIMODAL_RATIO = 3.0  # the gap between two prompt classes, as long over short
BIMODAL_SHARE = 0.1  # the least share of prompts in each class
SUFFIX_MATCH = 8  # the longest suffix matched against the tree
SUFFIX_DRAFT = 16  # the most tokens drafted at once
EXCLUDED_BLOCKS = 2  # the first and the last blocks kept in bf16
FP8_KV_CC = 8.9  # the least compute capability with fp8 per-token-head KV
HOST_RESERVE_GIB = 12.0  # host memory a CPU KV offload leaves to everything else


def spec_tiers(
    running_p90: float | None, k: int, *, cut: int | None, max_batch: int
) -> list[list[int]] | None:
    """``num_speculative_tokens_per_batch_size``: ``k`` drafts up to half the cut, one up to the
    cut, none above it; None when the p90 running count is above :data:`SPEC_RUNNING_P90`."""
    if running_p90 is None or running_p90 > SPEC_RUNNING_P90:
        return None
    cut = cut or SPEC_CUT
    half = max(cut // 2, 1)
    return [[1, half, k], [half + 1, cut, 1], [cut + 1, max(max_batch, cut + 1), 0]]


def long_prefill_threshold(prompt_tokens: Sequence[int]) -> int | None:
    """``--long-prefill-token-threshold``: the p90 of the short class of a bimodal prompt-length
    histogram (the widest gap, as long over short, of at least :data:`BIMODAL_RATIO` with each
    class at least :data:`BIMODAL_SHARE` of the prompts); None when it is not bimodal."""
    lengths = sorted(n for n in prompt_tokens if n > 0)
    least = math.ceil(len(lengths) * BIMODAL_SHARE)
    splits = [
        (lengths[i] / lengths[i - 1], i) for i in range(max(least, 1), len(lengths) - least + 1)
    ]
    ratio, at = max(splits, default=(0.0, 0))
    if ratio < BIMODAL_RATIO:
        return None
    short = lengths[:at]
    return short[math.ceil(0.9 * len(short)) - 1]


def suffix_acceptance(answers: Sequence[Sequence[Hashable]]) -> float | None:
    """Tokens per step a suffix tree over the earlier answers would give these answers, in
    order: at each step the longest suffix (up to :data:`SUFFIX_MATCH` tokens) of what was
    written that occurred before drafts what followed it (up to :data:`SUFFIX_DRAFT` tokens),
    and the step keeps the drafts that match plus one token. None without tokens."""
    seen: dict[tuple[Hashable, ...], tuple[int, int]] = {}  # suffix -> (answer, end) seen last
    tokens = steps = 0
    for index, answer in enumerate(answers):
        at = 0
        while at < len(answer):
            draft = _draft(seen, answers, index, at)
            accepted = 0
            while accepted < len(draft) and at + accepted < len(answer):
                if answer[at + accepted] != draft[accepted]:
                    break
                accepted += 1
            written = min(at + accepted + 1, len(answer))
            for end in range(max(at, 1), written):  # a suffix whose next token is written
                _remember(seen, answer, index, end)
            at = written
            steps += 1
        tokens += len(answer)
    return tokens / steps if steps else None


def _draft(
    seen: Mapping[tuple[Hashable, ...], tuple[int, int]],
    answers: Sequence[Sequence[Hashable]],
    index: int,
    at: int,
) -> Sequence[Hashable]:
    """What followed the longest suffix seen before; in the answer being written, only what is
    already written."""
    answer = answers[index]
    for size in range(min(SUFFIX_MATCH, at), 0, -1):
        found = seen.get(tuple(answer[at - size : at]))
        if found is not None:
            source, end = found
            stop = end + SUFFIX_DRAFT if source != index else min(end + SUFFIX_DRAFT, at)
            return answers[source][end:stop]
    return ()


def _remember(
    seen: dict[tuple[Hashable, ...], tuple[int, int]],
    answer: Sequence[Hashable],
    index: int,
    end: int,
) -> None:
    for size in range(1, min(SUFFIX_MATCH, end) + 1):
        seen[tuple(answer[end - size : end])] = (index, end)


def fp8_exclusions(layers: int) -> str:
    """``--quantization-config``: fp8 per channel on every linear layer but those of the first
    and the last :data:`EXCLUDED_BLOCKS` blocks."""
    ends = (*range(EXCLUDED_BLOCKS), *range(layers - EXCLUDED_BLOCKS, layers))
    ignore = [f"*layers.{block}.*" for block in sorted({b for b in ends if 0 <= b < layers})]
    return json.dumps({"linear": "fp8_per_channel", "ignore": ignore})


def per_token_head_dtype(cc: float | None) -> str | None:
    """``kv_cache_dtype``: fp8 per token and head from :data:`FP8_KV_CC`, int8 below it."""
    if cc is None:
        return None
    return "fp8_per_token_head" if cc >= FP8_KV_CC else "int8_per_token_head"


def cpu_kv_offload_size(host_memory_gib: float | None) -> float | None:
    """``--kv-offloading-size`` in GiB: the host's memory less :data:`HOST_RESERVE_GIB`; None
    when the host's memory is unknown or no larger."""
    if host_memory_gib is None or host_memory_gib <= HOST_RESERVE_GIB:
        return None
    return host_memory_gib - HOST_RESERVE_GIB


RULES: Mapping[str, Callable[..., Any]] = {
    f"vllm:{rule.__name__}": rule
    for rule in (
        spec_tiers,
        long_prefill_threshold,
        suffix_acceptance,
        fp8_exclusions,
        per_token_head_dtype,
        cpu_kv_offload_size,
    )
}
