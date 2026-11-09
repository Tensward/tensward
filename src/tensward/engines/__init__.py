"""Serving engines: the neutral interface and one module per engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .llamacpp import LLAMACPP
from .llamacpp.flags import key_flags as llamacpp_key_flags
from .protocol import Engine, FlagRules
from .vllm import VLLM
from .vllm.rows import lever_changes as vllm_lever_changes
from .vllm.rows import unset_value as vllm_unset_value

ENGINES: dict[str, Engine] = {VLLM.name: VLLM, LLAMACPP.name: LLAMACPP}
"""In preference order."""

FLAG_RULES: Mapping[str, FlagRules] = {name: engine.flag_rules for name, engine in ENGINES.items()}


def canonicaliser(family: str) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """The flag canonicaliser of an engine family: two spellings of one flag read as one."""
    rules = FLAG_RULES.get(family)
    return dict if rules is None else rules.canonical


def key_flags(
    family: str, before: Mapping[str, Any], after: Mapping[str, Any], sets: Mapping[str, Any]
) -> Mapping[str, Any]:
    """The flags a lever key is built from: llama.cpp's numeric flags by direction; the flags
    that changed, as they are, for vLLM."""
    if family == "llamacpp":
        return llamacpp_key_flags(before, after, sets)
    return {f: v for f, v in after.items() if before.get(f) != v}


def lever_changes(
    family: str,
) -> Callable[[Mapping[str, Sequence[Any]]], Mapping[str, Sequence[Any]]]:
    """A recorded change of an engine family (``{name: [before, after]}``) under the lever names
    knowledge records key by."""
    return vllm_lever_changes if family == "vllm" else dict


def unset_value(family: str, lever: str) -> str:
    """The value class of a numeric flag a run sets where the parent left it unset."""
    return vllm_unset_value(lever) if family == "vllm" else "set"
