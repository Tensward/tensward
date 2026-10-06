"""CUDA graph capture sizes: reading what a command pins and widening it to fit."""

from __future__ import annotations

import dataclasses
import itertools
import json
from typing import Any

from ...settings import Settings
from .command import COMPILATION_FLAG, MAX_CAPTURE_FLAG, SPECULATIVE_CONFIG_FLAG

# Speculative methods that vLLM v0.30 runs on its V1 model runner (config/vllm.py
# use_v2_model_runner), the only one that rounds CUDA graph sizes to multiples of 1 + k.
_V1_SPECULATIVE_METHODS = frozenset(
    {"ngram", "ngram_gpu", "draft_model", "suffix", "medusa", "mlp_speculator", "custom_class"}
)
_NO_ROUNDING_MODES = frozenset({"PIECEWISE", "NONE"})
_METHOD_NAMES = {"ngram": "n-gram"}


def _json_object(settings: Settings, flag: str) -> dict[str, Any]:
    """The JSON object a flag carries; empty when it is absent or carries anything else."""
    try:
        value = json.loads(str(settings.extra_args.get(flag)))
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _int_list(value: Any) -> list[int] | None:
    if isinstance(value, list) and value and all(type(n) is int and n > 0 for n in value):
        return value
    return None


def _capture_sizes(settings: Settings) -> tuple[list[int] | None, int | None]:
    """The pinned CUDA graph capture sizes, and the pinned largest one, as ``--compilation-config``
    (``-cc``) and ``--max-cudagraph-capture-size`` give them."""
    compilation = _json_object(settings, COMPILATION_FLAG)
    ceiling = compilation.get("max_cudagraph_capture_size")
    if type(ceiling) is not int:
        try:
            ceiling = int(str(settings.extra_args[MAX_CAPTURE_FLAG]))
        except (KeyError, ValueError):
            ceiling = None
    return _int_list(compilation.get("cudagraph_capture_sizes")), ceiling


def _default_sizes(ceiling: int) -> list[int]:
    """The sizes vLLM v0.30 captures up to ``ceiling`` when none are pinned (config/vllm.py)."""
    grid = [size for size in (1, 2, 4) if size <= ceiling]
    return grid + list(range(8, min(ceiling + 1, 256), 8)) + list(range(256, ceiling + 1, 16))


def _graph_mode(settings: Settings) -> str:
    return str(_json_object(settings, COMPILATION_FLAG).get("cudagraph_mode", "")).upper()


def _speculation(settings: Settings) -> tuple[str, int]:
    """The speculative method and its tokens per step per sequence (``num_speculative_tokens``,
    which a method that picks its own count, such as MTP, may leave unset: counted as 0)."""
    config = _json_object(settings, SPECULATIVE_CONFIG_FLAG)
    tokens = config.get("num_speculative_tokens")
    return str(config.get("method", "")), tokens if type(tokens) is int and tokens > 0 else 0


def _pinned_sizes(settings: Settings) -> list[int] | None:
    """The capture sizes in effect when the settings pin them or their largest, else None
    (also None when CUDA graphs are off)."""
    if not settings.cuda_graphs or _graph_mode(settings) == "NONE":
        return None
    sizes, ceiling = _capture_sizes(settings)
    if sizes is None and ceiling is not None and ceiling > 0:
        return _default_sizes(ceiling)
    return sizes


def _captured_sequences(sizes: list[int], k: int, mode: str, *, rounded: bool) -> int | None:
    """How many sequences a decode step can batch inside a captured CUDA graph, each taking
    ``1 + k`` tokens, or None when vLLM v0.30 would refuse the sizes. With ``rounded`` (the V1
    model runner) and a graph mode that captures decode, every size is rounded up to a multiple
    of ``1 + k`` and those above the largest pinned size are dropped, keeping ``1 + k`` itself if
    nothing remains (config/compilation.py adjust_cudagraph_sizes_for_spec_decode). It ignores
    that the multiple is ``max(1 + k, tp)`` under tensor and sequence parallelism."""
    step = 1 + k
    if rounded and k and mode not in _NO_ROUNDING_MODES:
        top = max(sizes)
        sizes = sorted({r for size in sizes if (r := -(-size // step) * step) <= top})
        if not sizes and step <= top:
            sizes = [step]
    return max(sizes) // step if sizes else None


def _coverage(settings: Settings) -> tuple[bool, int | None]:
    """Whether the settings pin CUDA graph sizes, and how many sequences those graphs batch."""
    sizes = _pinned_sizes(settings)
    if sizes is None:
        return False, None
    method, k = _speculation(settings)
    rounded = method in _V1_SPECULATIVE_METHODS
    return True, _captured_sequences(sizes, k, _graph_mode(settings), rounded=rounded)


def _widen_graphs(before: Settings, after: Settings) -> tuple[Settings, str | None]:
    """``after`` with its pinned CUDA graph sizes raised to cover a larger concurrency than
    ``before`` ran, or, for a pinned maximum only, to keep the sequences ``before`` covered when
    ``after`` speculates more tokens per step; written back where they were found. The sizes of
    any other change stay."""
    n, was = after.max_concurrent_requests, before.max_concurrent_requests
    pinned, covered = _coverage(after)
    sizes, _ = _capture_sizes(after)
    more_drafts = _speculation(after)[1] > _speculation(before)[1] and sizes is None
    target = n if n is not None and was is not None and n > was else None
    if target is None and more_drafts:
        target = _coverage(before)[1]
    if not pinned or not target or (covered or 0) >= target:
        return after, None
    n = target
    method, k = _speculation(after)
    step = 1 + k
    rounded = method in _V1_SPECULATIVE_METHODS
    mode = _graph_mode(after)
    sizes, _ = _capture_sizes(after)
    compilation = _json_object(after, COMPILATION_FLAG)
    if sizes is not None:
        wanted = {step * 2**power for power in range(n.bit_length())}
        sizes = sorted({*sizes, *wanted, step * n})
        compilation["cudagraph_capture_sizes"] = sizes
        if "max_cudagraph_capture_size" in compilation:
            compilation["max_cudagraph_capture_size"] = sizes[-1]
        widened = sizes[-1]
    else:  # vLLM builds its default grid from the pinned maximum, then truncates it to fit
        widened = next(
            grid[-1]
            for grid in (_default_sizes(top) for top in itertools.count(n))
            if (_captured_sequences(grid, k, mode, rounded=rounded) or 0) >= n
        )
        if "max_cudagraph_capture_size" in compilation:
            compilation["max_cudagraph_capture_size"] = widened
    extra = dict(after.extra_args)
    if sizes is not None or "max_cudagraph_capture_size" in compilation:
        extra[COMPILATION_FLAG] = json.dumps(compilation)
    else:
        extra[MAX_CAPTURE_FLAG] = str(widened)
    note = (
        f"raises the captured CUDA graph sizes to cover {n} sequences with it, "
        "so the larger or speculative batches keep their graphs"
    )
    return dataclasses.replace(after, extra_args=extra), note


DEFAULT_CAPTURE_LIMIT = 512  # the largest graph vLLM captures by default


def bounded_graphs(settings: Settings, cap: int) -> Settings:
    """``settings`` with the CUDA graph capture bound set to the smallest power of two that holds
    ``cap`` sequences (each taking ``1 + k`` tokens when speculating): decode batches never
    exceed the cap, and capturing larger graphs costs memory and start-up time. Nothing is set
    when the bound would reach vLLM's own default of 512. Sizes the settings already pin, or a
    compilation config that is not a JSON object, are left as they are."""
    sizes, ceiling = _capture_sizes(settings)
    compilation = _json_object(settings, COMPILATION_FLAG)
    if sizes or ceiling or (COMPILATION_FLAG in settings.extra_args and not compilation):
        return settings
    tokens = cap * (1 + _speculation(settings)[1])
    bound = 1 << (tokens - 1).bit_length()
    if bound >= DEFAULT_CAPTURE_LIMIT:
        return settings
    compilation["max_cudagraph_capture_size"] = bound
    extra = {**settings.extra_args, COMPILATION_FLAG: json.dumps(compilation)}
    return dataclasses.replace(settings, extra_args=extra)
