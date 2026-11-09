"""One live tool-call request before the measured window: the model must answer a tool prompt
with a tool call the server's parser reads."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .capture import CapturingTransport
from .client import _attempt, build_request
from .errors import AnalyseFailure, ToolCallFailure
from .inputs import PromptEntry
from .presets import read_choices, tool_prompts
from .progress import Note, emit

if TYPE_CHECKING:
    from .measure import Live
    from .project import ResolvedProject
    from .runtime import ServeSpec

PROBE_MAX_TOKENS = 256


def probe_tokens(project: ResolvedProject) -> int:
    return max(PROBE_MAX_TOKENS, project.config.workload.output_tokens)


DEFAULT_HINT = (
    "Check that the tool parser matches the model's tool-call format and that its chat "
    "template accepts tools."
)
# What to try when the model answers without a tool call, by the checkpoint's model_type.
HINTS = {
    "granite": (
        "Granite's chat template drops the tool instruction when a system prompt is given; "
        "send the tools without a system message."
    ),
}


def hint_for(model_type: str | None) -> str:
    return HINTS.get(model_type or "", DEFAULT_HINT)


def probe_entry(project: ResolvedProject, project_dir: Path) -> PromptEntry | None:
    """The first tool prompt (the built-in list for the built-in tools, else one made from the
    project's own tools) with the project's tools, single-turn, ``tool_choice`` "auto"; None when
    no prompt offers tools."""
    offering = next((entry for entry in project.prompts if entry.tools), None)
    if offering is None:
        return None
    choices = read_choices(project_dir)
    if choices is not None and choices.tools == "default":
        text = Path(__file__).parent.joinpath("data", "workloads", "tool-prompts.jsonl")
        record = json.loads(text.read_text("utf-8").splitlines()[0])
    else:
        tools = [tool.model_dump(mode="json", exclude_none=True) for tool in offering.tools]
        record = tool_prompts(tools)[0]
    entry = PromptEntry.model_validate(
        {"id": record["id"], "messages": record["messages"]}, strict=False
    )
    return entry.model_copy(
        update={"tools": offering.tools, "tool_choice": "auto", "max_tokens": probe_tokens(project)}
    )


async def check_tool_call(
    project: ResolvedProject,
    spec: ServeSpec,
    live: Live,
    *,
    run_id: str,
    project_dir: Path,
) -> None:
    """Send the probe and print the call that came back; raise ``AnalyseFailure`` when the
    answer holds no tool call."""
    entry = probe_entry(project, project_dir)
    if entry is None or not spec.settings.serves_tools:
        return
    workload = project.config.workload.model_copy(
        update={"output_tokens": probe_tokens(project), "request_count": 1, "chats": (entry.chat,)}
    )
    plan = await build_request(
        workload,
        0,
        run_id=f"{run_id}-toolcheck",
        model=spec.engine.request_model(spec.served_model_name),
        images=live.images,
        extras=spec.engine.request_extras,
    )
    transport = CapturingTransport(live.http, reasoning=spec.engine.family == "llamacpp")
    record = await _attempt(plan, time.monotonic_ns(), transport)
    await transport.decode_all()
    captured = transport.responses.get(plan.request_id)
    parser = spec.settings.tool_parser
    if record.outcome != "success" or captured is None:
        raise AnalyseFailure(f"the tool check request failed ({record.outcome})")
    if not captured.tool_calls and captured.finish_reason == "length":
        raise ToolCallFailure(
            f"the model ran out of tokens ({probe_tokens(project)}) before it made a tool call. "
            "Raise the workload's output tokens or turn its thinking off."
        )
    if not captured.tool_calls:
        model_type = project.anatomy.model_type if project.anatomy else None
        raise ToolCallFailure(
            f"the model answered without a tool call (parser {parser}). {hint_for(model_type)}"
        )
    call = captured.tool_calls[0]
    emit(Note(text=f"Tool check: {call.name}({_shown(call.arguments)}) parsed by {parser}"))


def _shown(arguments: str) -> str:
    try:
        parsed: Any = json.loads(arguments)
    except ValueError:
        return arguments
    if isinstance(parsed, dict):
        return ", ".join(f"{key}={json.dumps(value)}" for key, value in parsed.items())
    return arguments
