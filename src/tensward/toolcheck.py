"""Live tool-call requests before the measured window. A plain text answer is not a failure,
not even to all of them: a model rightly declines tools that do not fit the request (L27 on
llama.cpp: 0 calls in 28 answers, none broken; Qwen3.8-27B on vLLM with qwen3_xml: no call in 3
probes, 2 parsed calls in 40 workload answers). Raw tool-call markup left in the text or a call
whose arguments are not JSON is a failure."""

from __future__ import annotations

import json
import re
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .capture import CapturedResponse, CapturingTransport
from .client import _attempt, build_request
from .errors import AnalyseFailure, ToolCallFailure
from .inputs import PromptEntry
from .progress import Note, emit
from .toolcalls import ToolCall

if TYPE_CHECKING:
    from .measure import Live
    from .project import ResolvedProject
    from .runtime import ServeSpec

PROBE_HEADER = "X-Tensward-Probe"  # marks a probe in a proxy's or server's request log
PROBE_MAX_TOKENS = 256
PROBES = 3
# Tool-call markup that a parser reading the model's format would have taken out of the text.
MARKUP = ("<tool_call>", "<|tool_call", "[TOOL_CALLS]", "<|python_tag|>", "<function=",
          "<start_function_call>", "<|tool_calls_section_begin|>")  # fmt: skip


def probe_tokens(project: ResolvedProject) -> int:
    return max(PROBE_MAX_TOKENS, project.config.workload.output_tokens)


DEFAULT_HINT = (
    "Check that the tool parser matches the model's tool-call format and that its chat "
    "template accepts tools."
)
# What to try when a tool call comes back broken or never comes, by the checkpoint's model_type.
HINTS = {
    "granite": (
        "Granite's chat template drops the tool instruction when a system prompt is given; "
        "send the tools without a system message."
    ),
}


def hint_for(model_type: str | None) -> str:
    return HINTS.get(model_type or "", DEFAULT_HINT)


def _stems(text: str) -> set[str]:
    return {word[:5] for word in re.split(r"[^a-z0-9]+", text.lower()) if len(word) >= 4}


def _asks_for_tool(entry: PromptEntry) -> int:
    """How plainly the last user message asks for one of the record's tools: its name written
    out ranks first, then the most words it shares with one tool's name."""
    users = [message for message in entry.messages or () if message.role == "user"]
    if not users:
        return 0
    text = users[-1].text
    asked = _stems(text)
    best = 0
    for tool in entry.tools:
        name = tool.function.name
        if name in text:
            return 1000
        best = max(
            best, len(asked & _stems(re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ")))
        )
    return best


def probe_entries(project: ResolvedProject) -> list[PromptEntry]:
    """Up to ``PROBES`` of the workload's own records that offer tools and end on a user message
    (a record ending on a tool result is rightly answered in text), sent as the workload sends
    them (messages, ``tool_choice``, ``chat_template_kwargs``), those asking for one of their
    tools first."""
    offering = [
        entry for entry in project.prompts
        if entry.tools and entry.tool_choice != "none" and entry.response_format is None
    ]  # fmt: skip
    asking = [entry for entry in offering if entry.messages and entry.messages[-1].role == "user"]
    chosen = sorted(asking, key=_asks_for_tool, reverse=True)[:PROBES]
    tokens = probe_tokens(project)
    return [
        entry.model_copy(update={"max_tokens": max(entry.max_tokens or 0, tokens)})
        for entry in chosen
    ]


def _well_formed(call: ToolCall) -> bool:
    if not call.arguments.strip():
        return True
    try:
        return isinstance(json.loads(call.arguments), dict)
    except ValueError:
        return False


def judge(answers: list[CapturedResponse], parser: str | None,
          model_type: str | None) -> ToolCall | None:  # fmt: skip
    """The first well-formed call; None while every answer so far is plain text. Raise
    ``ToolCallFailure`` on markup in the text or malformed arguments."""
    for answer in answers:
        if not answer.tool_calls and any(mark in answer.text for mark in MARKUP):
            raise ToolCallFailure(
                f"the model wrote a tool call into its answer text that parser {parser} did not "
                f"read. {hint_for(model_type)}"
            )
        bad = next((call for call in answer.tool_calls if not _well_formed(call)), None)
        if bad is not None:
            raise ToolCallFailure(
                f"the model's tool call {bad.name} came back with arguments that are not a JSON "
                f"object (parser {parser}). {hint_for(model_type)}"
            )
        if answer.tool_calls:
            return answer.tool_calls[0]
    return None


async def check_tool_call(
    project: ResolvedProject,
    spec: ServeSpec,
    live: Live,
    *,
    run_id: str,
) -> None:
    """Send the probes one by one until one comes back with a tool call and print that call, or
    that none did; raise ``AnalyseFailure`` when the setup is broken (see ``judge``)."""
    entries = probe_entries(project)
    parser = spec.settings.tool_parser or spec.engine.own_tool_parser
    if not entries or not spec.settings.tool_calling or parser is None:
        return
    model_type = project.anatomy.model_type if project.anatomy else None
    answers: list[CapturedResponse] = []
    for index, entry in enumerate(entries):
        workload = project.config.workload.model_copy(
            update={"output_tokens": probe_tokens(project), "request_count": 1,
                    "chats": (entry.chat,)}
        )  # fmt: skip
        plan = await build_request(
            workload,
            0,
            run_id=f"{run_id}-toolcheck{index}",
            model=spec.engine.request_model(spec.served_model_name),
            images=live.images,
            extras=spec.engine.request_extras,
        )
        plan = replace(plan, headers={PROBE_HEADER: "tool-check"})
        transport = CapturingTransport(live.http)
        record = await _attempt(plan, time.monotonic_ns(), transport)
        await transport.decode_all()
        captured = transport.responses.get(plan.request_id)
        if record.outcome != "success" or captured is None:
            raise AnalyseFailure(f"the tool check request failed ({record.outcome})")
        answers.append(captured)
        call = judge(answers, parser, model_type)
        if call is not None:
            emit(Note(text=f"Tool check: {call.name}({_shown(call.arguments)}) parsed by {parser}"))
            return
    count = len(answers)
    how = (
        f"each ran out of its {probe_tokens(project)} tokens"
        if all(answer.finish_reason == "length" for answer in answers)
        else NO_PARSER
        if spec.engine.name == "llamacpp" and _text_only_format(live.server.log_text())
        else "answered in text"
    )
    plural = "probe" if count == 1 else "probes"
    known = f" {HINTS[model_type]}" if model_type in HINTS else ""
    emit(Note(text=f"Tool check: no tool call in {count} {plural}: {how} (parser {parser}){known}"))


def _shown(arguments: str) -> str:
    try:
        parsed: Any = json.loads(arguments)
    except ValueError:
        return arguments
    if isinstance(parsed, dict):
        return ", ".join(f"{key}={json.dumps(value)}" for key, value in parsed.items())
    return arguments


NO_PARSER = "this engine build has no tool-call parser for this model: tool calls come back as text"
_CHAT_FORMAT = re.compile(r"[Cc]hat format: (.+?)\s*$", re.M)
TEXT_FORMATS = ("content-only", "none")  # llama.cpp formats that parse no tool call


def _text_only_format(log: str) -> bool:
    """Whether llama.cpp's log names a chat format that parses no tool call. Most builds log no
    format at all (none of 533 logs of builds 11669 to 13010), which says nothing either way."""
    found = _CHAT_FORMAT.findall(log)
    return bool(found) and found[-1].strip().lower() in TEXT_FORMATS
