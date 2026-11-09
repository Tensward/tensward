"""A recorded change (Settings fields and canonical flags, each ``[before, after]``) read under the
lever names knowledge records key by."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ...settings import Settings
from .capabilities import read_lever
from .command import MAX_CAPTURE_FLAG, SPECULATIVE_CONFIG_FLAG, STRUCTURED_OUTPUTS_FLAG

LEVER_OF: Mapping[str, str] = {
    "--compilation-config.cudagraph_mode": "cudagraph_mode",
    "--compilation-config.pass_config.enable_qk_norm_rope_fusion": "qk_norm_rope_fusion",
    "--compilation-config.max_cudagraph_capture_size": "graph_capture_limit",
    MAX_CAPTURE_FLAG: "graph_capture_limit",
    "--kv-offloading-backend": "kv_offloading",
    "cuda_graphs": "graph_capture",
}
UNSET_VALUE: Mapping[str, str] = {"long_prefill_token_threshold": "short_p90"}
_OFFLOADING_SIZE = "--kv-offloading-size"
_SPECULATIVE = f"{SPECULATIVE_CONFIG_FLAG}."
_SCHEDULE = f"{_SPECULATIVE}num_speculative_tokens_per_batch_size"
_INTEGER_LEVERS = frozenset({"graph_capture_limit"})
_QUANT_CONFIG = "--quantization-config."
EXCLUDED = "fp8_per_channel_excluded"


def unset_value(lever: str) -> str:
    """The value class a numeric flag takes when it was left unset and a run sets it."""
    return UNSET_VALUE.get(lever, "set")


def lever_changes(changed: Mapping[str, Sequence[Any]]) -> dict[str, Sequence[Any]]:
    """``changed`` with the KV-cache dtype, the speculative-config fields and the flags in
    :data:`LEVER_OF` named as the levers they set; the offload size is the offload rule's value;
    ignore patterns in ``--quantization-config`` as the fp8 exclusions, and a structured-outputs
    config that turns whitespace off or on as compact JSON."""
    found: dict[str, Sequence[Any]] = {}
    speculation = {name: pair for name, pair in changed.items() if name.startswith(_SPECULATIVE)}
    method = _method(speculation)
    # the fp8 exclusions rule changes only --quantization-config; with a weight precision change
    # in the same step the parts stay raw, so the row pairs with neither record
    excluded = (
        bool(changed.get(f"{_QUANT_CONFIG}ignore", [None, None])[1])
        and "quantization" not in changed
    )
    if excluded:
        found["quantization"] = [None, EXCLUDED]
    for name, (before, after) in changed.items():
        if excluded and name.startswith(_QUANT_CONFIG):
            continue
        whitespace = (
            [_whitespace_off(v) for v in (before, after)]
            if name == STRUCTURED_OUTPUTS_FLAG
            else None
        )
        if whitespace and whitespace[0] != whitespace[1]:
            found["compact_json"] = whitespace
        elif name == "kv_cache_dtype":
            found.update(_dtype(before, after))
        elif name in speculation and method is not None:
            found["speculation"] = method
        elif name == _OFFLOADING_SIZE and "--kv-offloading-backend" in changed:
            continue
        elif name in LEVER_OF:
            lever = LEVER_OF[name]
            found[lever] = [_number(v) if lever in _INTEGER_LEVERS else v for v in (before, after)]
        else:
            found[name] = [before, after]
    return found


def _whitespace_off(value: Any) -> bool:
    """Whether a recorded structured-outputs config disables whitespace (the compact JSON lever)."""
    try:
        config = json.loads(value) if isinstance(value, str) else value
    except ValueError:
        return False
    return isinstance(config, dict) and config.get("disable_any_whitespace") is True


def _dtype(before: Any, after: Any) -> dict[str, Sequence[Any]]:
    """The lever a KV-cache dtype change moves: the precision, or the dtype family when the
    dtype is not a precision."""
    (lever, old), (moved, new) = _dtype_class(before), _dtype_class(after)
    return {} if (lever, old) == (moved, new) else {moved: [old if lever == moved else None, new]}


def _dtype_class(dtype: Any) -> tuple[str, str]:
    name = str(dtype or "auto")
    if name.startswith("int4") and name.endswith("per_token_head"):
        return "kv_cache_precision", "4bit"
    if name.endswith("per_token_head"):
        return "kv_cache_dtype", "per_token_head"
    if name.startswith("turboquant"):
        return "kv_cache_dtype", "turboquant"
    precision = read_lever(Settings(kv_cache_dtype=name), "kv_cache_precision")
    return "kv_cache_precision", str(precision)


def _method(fields: Mapping[str, Sequence[Any]]) -> Sequence[Any] | None:
    """``[before, after]`` of the speculation method the speculative-config fields move; None when
    they change no method."""
    if _SCHEDULE in fields and fields[_SCHEDULE][1] is not None:
        return [_read(fields, 0), "schedule"]
    if f"{_SPECULATIVE}method" not in fields:
        return None
    return [_read(fields, 0), _read(fields, 1)]


def _read(fields: Mapping[str, Sequence[Any]], at: int) -> str | None:
    config = {name.removeprefix(_SPECULATIVE): pair[at] for name, pair in fields.items()}
    if config.get("method") is None:
        return None
    settings = Settings(extra_args={SPECULATIVE_CONFIG_FLAG: json.dumps(config)})
    return str(read_lever(settings, "speculation").method)


def _number(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value
