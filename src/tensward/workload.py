"""The workload a run offers: prompts or chats, how many requests, and how they arrive.

The models are strict (unknown fields are refused) and frozen. Their dumps feed the project's
configuration digest, so a field added, renamed or given another default changes every
registered project's identity.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import Field, JsonValue, StringConstraints, field_validator, model_validator

from .contracts import PositiveInt, StrictModel

MAX_PROMPTS = 10_000
MAX_PROMPT_BYTES = 1_048_576
MAX_WORKLOAD_BYTES = 64 * 1024 * 1024  # as much as one input file may hold
MAX_TOKENS = 1_048_576
MAX_STOP_SEQUENCES = 4
MAX_STOP_BYTES = 64
MAX_STRUCTURED_OUTPUT_BYTES = 64 * 1024

TokenCount = Annotated[int, Field(ge=1, le=MAX_TOKENS)]
RequestCount = Annotated[int, Field(ge=1, le=10_000)]
RequestTimeout = Annotated[float, Field(gt=0, le=3600.0)]
Temperature = Annotated[float, Field(ge=0, le=2.0)]
TopP = Annotated[float, Field(gt=0, le=1)]
Seed = Annotated[int, Field(ge=0, le=2**63 - 1)]
PositiveFloat = Annotated[float, Field(gt=0)]

WorkloadApi = Literal["completions", "chat"]
TOOL_CHOICES = ("auto", "none", "required")


class ChatMessage(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def _validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a chat message must not be empty")
        return value


class ToolFunction(StrictModel):
    """An OpenAI function definition: a name and the JSON schema of its arguments."""

    name: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
    description: str | None = None
    parameters: dict[str, JsonValue]

    @field_validator("parameters")
    @classmethod
    def _validate_parameters(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        required = value.get("required", [])
        if (
            value.get("type") != "object"
            or not isinstance(value.get("properties", {}), dict)
            or not isinstance(required, list)
            or not all(isinstance(key, str) for key in required)
        ):
            raise ValueError(
                'tool parameters must be a JSON schema with "type": "object", '
                "a properties object and a list of required names"
            )
        return value


class ChatTool(StrictModel):
    type: Literal["function"]
    function: ToolFunction


class ChatRequest(StrictModel):
    """One chat prompt: its messages, the tools it offers, and an optional output cap."""

    messages: tuple[ChatMessage, ...]
    tools: tuple[ChatTool, ...] = ()
    tool_choice: str | dict[str, JsonValue] | None = None
    max_tokens: TokenCount | None = None

    @model_validator(mode="after")
    def _validate_chat(self) -> ChatRequest:
        if not self.messages or self.messages[-1].role != "user":
            raise ValueError("a chat prompt must end with a user message")
        names = [tool.function.name for tool in self.tools]
        if len(set(names)) != len(names):
            raise ValueError("tool names must be unique")
        if self.tool_choice is not None:
            named = [{"type": "function", "function": {"name": name}} for name in names]
            if not names or self.tool_choice not in (*TOOL_CHOICES, *named):
                raise ValueError(
                    'tool_choice must be "auto", "none", "required" or name an offered tool'
                )
        if len(self.model_dump_json().encode("utf-8")) > MAX_PROMPT_BYTES:
            raise ValueError("a chat prompt exceeds the maximum prompt size")
        return self


class ArrivalSpec(StrictModel):
    """How requests arrive.

    ``closed_loop`` keeps ``concurrency`` requests in flight, ``open_loop`` offers
    ``rate_rps`` requests per second whatever the server does, and ``capped`` offers that
    rate but never has more than ``max_inflight`` in flight.
    """

    kind: Literal["closed_loop", "open_loop", "capped"]
    concurrency: PositiveInt | None = None
    rate_rps: PositiveFloat | None = None
    max_inflight: PositiveInt | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> ArrivalSpec:
        takes = {
            "closed_loop": {"concurrency"},
            "open_loop": {"rate_rps"},
            "capped": {"rate_rps", "max_inflight"},
        }[self.kind]
        fields = ("concurrency", "rate_rps", "max_inflight")
        if {name for name in fields if getattr(self, name)} != takes:
            raise ValueError(f"a {self.kind} arrival declares exactly {sorted(takes)}")
        return self


class WorkloadSpec(StrictModel):
    """The requests a run offers, unchanged (``same_text``). ``api`` says which endpoint the
    prompts are for: ``completions`` carries ``prompts``, ``chat`` carries ``chats``."""

    mode: Literal["same_text"] = "same_text"
    api: WorkloadApi = "completions"
    prompts: tuple[str, ...] = ()
    chats: tuple[ChatRequest, ...] = ()
    output_tokens: TokenCount
    request_count: RequestCount
    request_timeout_s: RequestTimeout
    arrival: ArrivalSpec
    temperature: Temperature
    top_p: TopP
    seed: Seed | None = None
    stop: tuple[str, ...] = ()
    structured_output: dict[str, JsonValue] | None = None
    logprobs: Annotated[int, Field(ge=1, le=20)] | None = None

    @model_validator(mode="after")
    def _validate_prompts(self) -> WorkloadSpec:
        chat = self.api == "chat"
        offered, other = (self.chats, self.prompts) if chat else (self.prompts, self.chats)
        if not offered or other:
            raise ValueError(f"a {self.api} workload needs {self.api} prompts and no others")
        if len(offered) > MAX_PROMPTS:
            raise ValueError(f"a workload carries at most {MAX_PROMPTS} prompts")
        if any(not prompt or len(prompt.encode()) > MAX_PROMPT_BYTES for prompt in self.prompts):
            raise ValueError("a workload prompt must be non-empty and within the prompt size limit")
        size = sum(len(prompt.encode()) for prompt in self.prompts)
        size += sum(len(chat.model_dump_json().encode()) for chat in self.chats)
        if size > MAX_WORKLOAD_BYTES:
            raise ValueError("the workload exceeds the maximum total prompt size")
        return self

    @field_validator("stop")
    @classmethod
    def _validate_stop(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > MAX_STOP_SEQUENCES or len(set(value)) != len(value):
            raise ValueError(f"stop takes at most {MAX_STOP_SEQUENCES} distinct sequences")
        if any(not item or len(item.encode()) > MAX_STOP_BYTES for item in value):
            raise ValueError("a stop sequence must be non-empty and short")
        return value

    @field_validator("structured_output")
    @classmethod
    def _validate_structured_output(
        cls, value: dict[str, JsonValue] | None
    ) -> dict[str, JsonValue] | None:
        if value is not None:
            encoded = json.dumps(value, allow_nan=False, separators=(",", ":"))
            if not value or len(encoded.encode()) > MAX_STRUCTURED_OUTPUT_BYTES:
                raise ValueError("structured_output must be a non-empty, bounded schema")
        return value
