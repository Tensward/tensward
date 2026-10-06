"""The workload a run offers: prompts or chats, how many requests, and how they arrive.

The models are strict (unknown fields are refused) and frozen. Their dumps feed the project's
configuration digest, so a field added, renamed or given another default changes every
registered project's identity.
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, JsonValue, StringConstraints, field_validator, model_validator
from pydantic_core import PydanticCustomError

from .contracts import PositiveInt, StrictModel
from .errors import USER_MESSAGE_ERROR
from .images import MAX_IMAGES_PER_PROMPT

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


class TextPart(StrictModel):
    type: Literal["text"]
    text: str


class ImageUrl(StrictModel):
    url: str

    @field_validator("url")
    @classmethod
    def _relative(cls, value: str) -> str:
        """An image is a file next to the prompts file, named by its relative path: Tensward
        reads and hashes it, so a remote or inline image would not be part of the workload."""
        path = PurePosixPath(value)
        if (
            "://" in value
            or value.startswith("data:")
            or "\x00" in value
            or path.is_absolute()
            or ".." in path.parts
            or not value.strip()
        ):
            raise PydanticCustomError(
                USER_MESSAGE_ERROR,
                "an image url must be a relative path; save the image next to the prompts file "
                "(or below it) and give its relative path, e.g. images/a.png",
            )
        return value


class ImagePart(StrictModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class ChatMessage(StrictModel):
    """One chat message: its content is a text, or a list of text and image parts (user
    messages only, as OpenAI requires)."""

    role: Literal["system", "user", "assistant"]
    content: str | tuple[ContentPart, ...]

    @model_validator(mode="after")
    def _validate_content(self) -> ChatMessage:
        if not self.text.strip():
            raise ValueError("a chat message must not be empty")
        if self.image_urls and self.role != "user":
            raise ValueError("only a user message may carry images")
        return self

    @property
    def image_urls(self) -> tuple[str, ...]:
        parts = () if isinstance(self.content, str) else self.content
        return tuple(part.image_url.url for part in parts if isinstance(part, ImagePart))

    @property
    def text(self) -> str:
        """The message as one text, an image shown as ``[image]``."""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(
            "[image]" if isinstance(part, ImagePart) else part.text for part in self.content
        )


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
        if sum(len(message.image_urls) for message in self.messages) > MAX_IMAGES_PER_PROMPT:
            raise ValueError(f"a chat prompt carries at most {MAX_IMAGES_PER_PROMPT} images")
        # Images are paths here; their bytes have their own bounds (images.py).
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


STRUCTURED_CONSTRAINTS = ("json", "regex", "choice", "grammar", "json_object", "structural_tag")
STRUCTURED_OPTIONS = {
    "disable_any_whitespace": bool,
    "disable_additional_properties": bool,
    "whitespace_pattern": str,
}


def _constraint_is_valid(key: str, value: JsonValue) -> bool:
    if key == "json":
        return isinstance(value, (dict, str)) and bool(value)
    if key == "choice":
        return isinstance(value, list) and bool(value) and all(isinstance(c, str) for c in value)
    if key == "json_object":
        return value is True
    return isinstance(value, str) and bool(value)


def _check_structured_output(value: dict[str, JsonValue]) -> None:
    """vLLM's ``structured_outputs`` object: exactly one constraint, and its options."""
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode()) > MAX_STRUCTURED_OUTPUT_BYTES:
        raise ValueError("structured_output is too large")
    unknown = sorted(set(value) - set(STRUCTURED_CONSTRAINTS) - set(STRUCTURED_OPTIONS))
    given = [key for key in STRUCTURED_CONSTRAINTS if key in value]
    if unknown or len(given) != 1:
        raise ValueError(
            "structured_output takes vLLM's structured_outputs object with exactly one of "
            f"{', '.join(STRUCTURED_CONSTRAINTS)}, and optionally "
            f"{', '.join(STRUCTURED_OPTIONS)}; got {', '.join(sorted(value)) or 'nothing'}. "
            'For a JSON schema write {"json": <schema>}; the OpenAI response_format and '
            "json_schema shapes are not accepted"
        )
    if not _constraint_is_valid(given[0], value[given[0]]):
        raise ValueError(f"structured_output {given[0]} is empty or of the wrong type")
    for key, kind in STRUCTURED_OPTIONS.items():
        if key in value and not isinstance(value[key], kind):
            raise ValueError(f"structured_output {key} must be a {kind.__name__}")


class WorkloadShape(StrictModel):
    """How a workload runs: everything about it except its records."""

    mode: Literal["same_text"] = "same_text"
    api: WorkloadApi = "completions"
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
            _check_structured_output(value)
        return value


class WorkloadSpec(WorkloadShape):
    """The requests a run offers, unchanged (``same_text``). ``api`` says which endpoint the
    prompts are for: ``completions`` carries ``prompts``, ``chat`` carries ``chats``."""

    prompts: tuple[str, ...] = ()
    chats: tuple[ChatRequest, ...] = ()

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
