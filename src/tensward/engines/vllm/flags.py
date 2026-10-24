"""The vLLM flags and environment variables a knowledge record may set, per vLLM minor version.
A flag or a variable outside these lists is never placed on a user's command line or in the
engine's environment, whatever a record says.

A dotted flag (``-cc.cudagraph_mode``) sets one field of a JSON-valued flag, as vLLM's own
argument parser reads it; :func:`merge_json` folds it into that flag's JSON object."""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from packaging.version import InvalidVersion, Version

from .command import COMPILATION_FLAG, SPECULATIVE_CONFIG_FLAG

QUANTIZATION_CONFIG_FLAG = "--quantization-config"
JSON_FLAGS: Mapping[str, str] = {  # how a dotted flag names its JSON flag -> that flag
    "-cc": COMPILATION_FLAG,
    COMPILATION_FLAG: COMPILATION_FLAG,
    SPECULATIVE_CONFIG_FLAG: SPECULATIVE_CONFIG_FLAG,
    QUANTIZATION_CONFIG_FLAG: QUANTIZATION_CONFIG_FLAG,
}

ALLOWED: Mapping[str, frozenset[str]] = {
    "0.30": frozenset({
        "--attention-backend",
        "--kv-cache-dtype-skip-layers",
        "--kv-cache-memory-bytes",
        "--kv-offloading-backend",  # vllm@v0.30.0:vllm/engine/arg_utils.py:1325-1327
        "--kv-offloading-size",  # vllm@v0.30.0:vllm/engine/arg_utils.py:1322-1324
        "--kv-sharing-fast-prefill",
        "--language-model-only",
        "--limit-mm-per-prompt",
        "--linear-backend",
        "--long-prefill-token-threshold",
        "--max-num-scheduled-tokens",  # vllm@v0.30.0:vllm/engine/arg_utils.py:1587-1592
        "--moe-backend",
        "--no-disable-cascade-attn",
        "--performance-mode",
        "--prefix-cache-retention-interval",  # vllm@v0.30.0:vllm/engine/arg_utils.py:1289-1292
        "--prefix-caching-hash-algo",
        "--quantization-config",  # vllm@v0.30.0:vllm/engine/arg_utils.py:901-903
        "--quantization-config.ignore",  # vllm@v0.30.0:vllm/config/quantization.py:114
        # vllm@v0.30.0:vllm/config/speculative.py:479-485
        "--speculative-config.num_speculative_tokens_per_batch_size",
        "--stream-interval",
        "--watermark",  # vllm@v0.30.0:vllm/engine/arg_utils.py:1634
        "-cc.cudagraph_mode",  # vllm@v0.30.0:vllm/config/compilation.py:614
        "-cc.pass_config.enable_qk_norm_rope_fusion",  # vllm@v0.30.0:vllm/config/compilation.py:144
    }),
}  # fmt: skip

ENV_ALLOWED: Mapping[str, frozenset[str]] = {
    "0.30": frozenset({
        "VLLM_MARLIN_INPUT_DTYPE",  # vllm@v0.30.0:vllm/envs.py:1498-1500
        "VLLM_USE_RUST_FRONTEND",  # vllm@v0.30.0:vllm/envs.py:1394-1396
        "VLLM_USE_V2_MODEL_RUNNER",  # vllm@v0.30.0:vllm/envs.py:2037-2039
    }),
}  # fmt: skip


def allowed(version: str | None, flag: str) -> bool:
    """Whether ``flag`` is allowed for vLLM ``version``; never for an unknown version."""
    minor = None if version is None else _minor(version)
    return minor is not None and flag in ALLOWED.get(minor, frozenset())


def env_allowed(version: str | None, name: str) -> bool:
    """Whether the environment variable ``name`` is allowed for vLLM ``version``; never for an
    unknown version."""
    minor = None if version is None else _minor(version)
    return minor is not None and name in ENV_ALLOWED.get(minor, frozenset())


def merge_json(
    existing: Mapping[str, str | bool | None], dotted: Mapping[str, Any]
) -> dict[str, str | bool | None]:
    """``existing`` flags with ``dotted`` added: each dotted flag is written into its JSON flag's
    object (one ``--compilation-config`` holds what was there and the new field), its value read
    as JSON where it parses, as vLLM reads it; other flags are set as given."""
    merged = dict(existing)
    for flag, value in dotted.items():
        base, _, path = flag.partition(".")
        if not path or base not in JSON_FLAGS:
            merged[flag] = value
            continue
        target = JSON_FLAGS[base]
        try:
            document = json.loads(str(merged.get(target)))
        except ValueError:
            document = None
        document = document if isinstance(document, dict) else {}
        *parents, leaf = path.split(".")
        node = document
        for key in parents:
            if not isinstance(node.get(key), dict):
                node[key] = {}
            node = node[key]
        node[leaf] = _json_value(value)
        merged[target] = json.dumps(document)
    return merged


def canonical_flags(flags: Mapping[str, Any]) -> dict[str, Any]:
    """``flags`` with each JSON flag, whole (``--compilation-config '{...}'``) or dotted
    (``-cc.cudagraph_mode``), as one ``--<flag>.<path>`` entry per JSON leaf in sorted order:
    two spellings of one setting read as one. Other flags are kept as given."""
    found: dict[str, Any] = {}
    for flag, value in flags.items():
        base, _, path = flag.partition(".")
        target = JSON_FLAGS.get(base)
        if target is None:
            found[flag] = value
            continue
        parsed = _json_value(value)
        if path:
            leaves = dict(_leaves(parsed, path))
        elif isinstance(parsed, dict):
            leaves = dict(_leaves(parsed, ""))
        else:
            found[target] = value
            continue
        found.update({f"{target}.{name}": leaf for name, leaf in leaves.items()})
    return dict(sorted(found.items()))


def canonical_args(args: Iterable[str]) -> frozenset[str]:
    """Engine-argument texts (``compilation-config={...}``) in :func:`canonical_flags` form."""
    flags: dict[str, Any] = {}
    for text in args:
        name, given, value = text.partition("=")
        flags[name if name.startswith("-") else f"--{name}"] = value if given else True
    return frozenset(
        flag.lstrip("-") if value is True else f"{flag.lstrip('-')}={_text(value)}"
        for flag, value in canonical_flags(flags).items()
    )


def _leaves(value: Any, path: str) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict) and value:
        for key in sorted(value):
            yield from _leaves(value[key], f"{path}.{key}" if path else str(key))
    else:
        yield path, value


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


@functools.lru_cache(maxsize=256)
def _minor(version: str) -> str | None:
    try:
        release = Version(version).release
    except InvalidVersion:
        return None
    return ".".join(str(part) for part in release[:2])
