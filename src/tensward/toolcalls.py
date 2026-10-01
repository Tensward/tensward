"""Tool-call quality: did the model call a tool, with well-formed arguments for that tool?

These are observations about the model's output, not performance. A request counts as good on
a signal only when every tool call it produced does. The schema check is deliberately small:
the arguments must be an object with every ``required`` key present and each declared property
of the declared basic type. Nested schemas, enums and formats are not checked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .workload import ChatRequest

_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: str  # the JSON text the model produced, as streamed


@dataclass(frozen=True, slots=True)
class ToolCallStats:
    """Shares over the successful requests that offered tools.

    ``produced_call`` is a share of all such ``requests``; the other three are shares of the
    ``calling`` requests, the ones that produced at least one tool call.
    """

    requests: int
    calling: int
    produced_call: float | None
    valid_json: float | None
    known_tool: float | None
    schema_valid: float | None


def _is_type(value: Any, name: str) -> bool:
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    expected = _TYPES.get(name)
    return expected is None or isinstance(value, expected)  # an unknown type name is not checked


def matches_schema(value: Any, parameters: Mapping[str, Any]) -> bool:
    """Whether ``value`` has all ``required`` keys and each declared property has its type."""
    if not isinstance(value, dict):
        return False
    if any(key not in value for key in parameters.get("required", ())):
        return False
    for key, spec in parameters.get("properties", {}).items():
        declared = spec.get("type") if isinstance(spec, dict) else None
        if key in value and declared is not None:
            names = declared if isinstance(declared, list) else [declared]
            if not any(_is_type(value[key], name) for name in names):
                return False
    return True


def _judge(call: ToolCall, offered: Mapping[str, Mapping[str, Any]]) -> tuple[bool, bool, bool]:
    """(arguments parse as JSON, tool was offered, arguments match that tool's schema)."""
    try:
        value = json.loads(call.arguments)
    except ValueError:
        return False, call.name in offered, False
    known = call.name in offered
    return True, known, known and matches_schema(value, offered[call.name])


def tool_call_stats(
    requests: Sequence[tuple[ChatRequest, Sequence[ToolCall]]],
) -> ToolCallStats | None:
    """Stats over (request, tool calls) pairs; None when no request offered tools."""
    judged = []
    total = 0
    for chat, calls in requests:
        if not chat.tools:
            continue
        total += 1
        offered = {tool.function.name: tool.function.parameters for tool in chat.tools}
        if calls:
            judged.append([_judge(call, offered) for call in calls])
    if not total:
        return None

    def share(index: int) -> float | None:
        return (
            sum(all(v[index] for v in calls) for calls in judged) / len(judged) if judged else None
        )

    return ToolCallStats(
        requests=total,
        calling=len(judged),
        produced_call=len(judged) / total,
        valid_json=share(0),
        known_tool=share(1),
        schema_valid=share(2),
    )
