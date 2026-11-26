from types import SimpleNamespace

import pytest

from tensward.capture import CapturedResponse
from tensward.errors import ToolCallFailure
from tensward.toolcalls import ToolCall
from tensward.toolcheck import PROBE_MAX_TOKENS, judge, probe_tokens


def test_the_probe_has_the_room_the_workload_gives_a_thinking_model():
    thinking = SimpleNamespace(config=SimpleNamespace(workload=SimpleNamespace(output_tokens=1280)))
    short = SimpleNamespace(config=SimpleNamespace(workload=SimpleNamespace(output_tokens=128)))
    assert probe_tokens(thinking) == 1280  # type: ignore[arg-type]
    assert probe_tokens(short) == PROBE_MAX_TOKENS  # type: ignore[arg-type]


def test_a_text_answer_is_not_a_broken_setup_but_markup_is():
    refusal = CapturedResponse(
        text="None of the provided functions can be used.", finish_reason="stop"
    )
    call = CapturedResponse(tool_calls=[ToolCall("get_weather", '{"city": "Madrid"}')])
    assert judge([refusal], "hermes", None) is None
    assert judge([refusal, call], "hermes", None) == call.tool_calls[0]
    assert judge([refusal] * 3, "hermes", None) is None
    with pytest.raises(ToolCallFailure, match="did not read"):
        judge([refusal, CapturedResponse(text="<function=get_weather>")], "qwen3_xml", None)
    raw = CapturedResponse(text='<tool_call>{"name": "get_weather"}</tool_call>')
    with pytest.raises(ToolCallFailure, match="did not read"):
        judge([raw], "hermes", None)
    bad = CapturedResponse(tool_calls=[ToolCall("get_weather", '{"city": ')])
    with pytest.raises(ToolCallFailure, match="not a JSON object"):
        judge([bad], "hermes", None)


def test_the_probes_are_the_workloads_own_tool_records_sent_as_the_workload_sends_them():
    from tensward.inputs import PromptEntry
    from tensward.toolcheck import probe_entries

    tool = {"type": "function", "function": {
        "name": "get_weather", "description": "Get the weather of a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}  # fmt: skip
    off = {"chat_template_kwargs": {"enable_thinking": False}, "tool_choice": "auto"}

    def record(rid: str, text: str, **extra: object) -> PromptEntry:
        messages = [{"role": "user", "content": text}]
        return PromptEntry.model_validate({"id": rid, "messages": messages, **extra}, strict=False)

    prompts = [
        record("plain", "Write a haiku."),
        record("vague", "Hello there, how are you?", tools=[tool], **off),
        record("asks", "What is the weather in Madrid, city of my birth?", tools=[tool], **off),
        record("never", "What is the weather in Oslo?", tools=[tool], tool_choice="none"),
    ]
    project = SimpleNamespace(
        prompts=prompts, config=SimpleNamespace(workload=SimpleNamespace(output_tokens=64))
    )
    probes = probe_entries(project)  # type: ignore[arg-type]
    assert [p.id for p in probes] == ["asks", "vague"]
    assert all(p.chat_template_kwargs == {"enable_thinking": False} for p in probes)
    assert all(p.tool_choice == "auto" and p.max_tokens == PROBE_MAX_TOKENS for p in probes)
