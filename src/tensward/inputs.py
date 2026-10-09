"""The two declared inputs of a project: the workload JSONL and the serving configuration.

Both are read once, validated strictly (unknown keys are refused, because a silently dropped
field would make the registered input a different one) and reduced to a digest. The digests
are part of each project's identity: change what they hash and every registered project
reads as changed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Sequence

from pydantic import (
    Field,
    JsonValue,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from .artifacts import ActivationDtype, WeightPrecision
from .contracts import Identifier, StrictModel
from .errors import (
    PROJECT_INPUTS_INVALID,
    USER_MESSAGE_ERROR,
    PreflightError,
    not_found_message,
    validation_summary,
)
from .files import parse_document
from .images import PromptImages
from .text import canonical_json
from .workload import (
    MAX_PROMPTS,
    ChatMessage,
    ChatRequest,
    ChatTool,
    PositiveFloat,
    TokenCount,
    WorkloadApi,
    WorkloadShape,
    WorkloadSpec,
    check_chat_template_kwargs,
    structured_from_response_format,
)

MAX_INPUT_BYTES = 64 * 1024 * 1024
CONFIG_DIGEST_DOMAIN = b"tensward:registration-config:1\x00"
WORKLOAD_DIGEST_DOMAIN = b"tensward:registration-workload:1\x00"
WORKLOAD_IMAGES_DOMAIN = b"tensward:registration-workload-images:1\x00"


class PromptEntry(StrictModel):
    """One workload record: an id, then either a plain ``prompt`` or chat ``messages``.

    A chat record may also offer ``tools`` (OpenAI function schemas), a ``tool_choice`` and a
    ``max_tokens`` cap. A chat record may also declare its own ``response_format`` (OpenAI's
    form) and ``chat_template_kwargs``. ``reference`` and ``labels`` are review metadata; they
    are part of the workload identity but the engine never sees them.
    """

    id: Identifier
    prompt: str | None = None
    messages: tuple[ChatMessage, ...] | None = None
    tools: tuple[ChatTool, ...] = ()
    tool_choice: str | dict[str, JsonValue] | None = None
    max_tokens: TokenCount | None = None
    reference: str | None = None
    labels: tuple[Identifier, ...] = ()
    response_format: dict[str, JsonValue] | None = None
    chat_template_kwargs: dict[str, JsonValue] | None = None

    @field_validator("response_format")
    @classmethod
    def _validate_response_format(
        cls, value: dict[str, JsonValue] | None
    ) -> dict[str, JsonValue] | None:
        if value is not None:
            structured_from_response_format(value)
        return value

    @field_validator("chat_template_kwargs")
    @classmethod
    def _validate_chat_template_kwargs(
        cls, value: dict[str, JsonValue] | None
    ) -> dict[str, JsonValue] | None:
        if value is not None:
            check_chat_template_kwargs(value)
        return value

    @model_validator(mode="after")
    def _validate_shape(self) -> PromptEntry:
        if (self.prompt is None) == (self.messages is None):
            raise ValueError('a workload record has either "prompt" or "messages"')
        chat_only = (
            self.tools or self.tool_choice or self.max_tokens
            or self.response_format is not None or self.chat_template_kwargs is not None
        )  # fmt: skip
        if self.messages is None and chat_only:
            raise ValueError(
                "tools, tool_choice, max_tokens, response_format and chat_template_kwargs belong "
                "to a chat record"
            )
        if self.constrained and (
            isinstance(self.tool_choice, dict) or self.tool_choice == "required"
        ):
            raise ValueError(
                'a record with a response_format cannot also set tool_choice to "required" or '
                "name a tool: the engine refuses a named tool_choice under structured output, "
                "and a required tool call leaves no text to judge as JSON"
            )
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be distinct")
        return self

    @property
    def api(self) -> WorkloadApi:
        return "completions" if self.messages is None else "chat"

    @property
    def constrained(self) -> bool:
        """Whether the record's own response_format constrains the answer (text does not)."""
        return self.response_format is not None and self.response_format.get("type") != "text"

    @property
    def chat(self) -> ChatRequest:
        own = self.response_format
        return ChatRequest(
            messages=self.messages or (),
            tools=self.tools,
            tool_choice=self.tool_choice,
            max_tokens=self.max_tokens,
            structured_output=structured_from_response_format(own) if own is not None else None,
            chat_template_kwargs=self.chat_template_kwargs,
        )

    @property
    def text(self) -> str:
        """The record as one text, for judging shared prefixes, in the order the chat template
        renders it: a leading system message, then the tools, then the other messages."""
        if self.messages is None:
            return self.prompt or ""
        tools = [tool.model_dump_json(exclude_none=True) for tool in self.tools]
        lines = [f"{m.role}: {m.text}" for m in self.messages]
        system = 1 if self.messages and self.messages[0].role == "system" else 0
        return "\n".join([*lines[:system], *tools, *lines[system:]])

    @property
    def image_urls(self) -> tuple[str, ...]:
        """The images of every message, in order of appearance."""
        return tuple(url for message in self.messages or () for url in message.image_urls)


def _read_input(path: Path, what: str) -> bytes:
    """The bytes of a declared input. Only a regular file is read, so a FIFO cannot hang us."""
    try:
        if path.is_file() and path.stat().st_size <= MAX_INPUT_BYTES:
            return path.read_bytes()
    except FileNotFoundError:
        raise PreflightError(PROJECT_INPUTS_INVALID, not_found_message(what, path)) from None
    except OSError:
        pass
    raise PreflightError(
        PROJECT_INPUTS_INVALID, f"{what} is not a readable regular file of at most 64 MiB"
    )


def load_prompt_entries(path: Path) -> tuple[PromptEntry, ...]:
    """Read the workload JSONL: one record per line, at most ``MAX_PROMPTS`` of them."""
    what = "the workload document"
    try:
        lines = _read_input(path, what).decode("utf-8").split("\n")
    except UnicodeDecodeError:
        raise PreflightError(PROJECT_INPUTS_INVALID, f"{what} is not valid UTF-8") from None
    if lines[-1] == "":
        lines.pop()
    if not 0 < len(lines) <= MAX_PROMPTS:
        raise PreflightError(
            PROJECT_INPUTS_INVALID, f"{what} must hold between 1 and {MAX_PROMPTS} records"
        )
    entries = tuple(
        parse_document(
            PromptEntry, line.encode(), f"workload record {number}", PROJECT_INPUTS_INVALID
        )
        for number, line in enumerate(lines, start=1)
    )
    if len({entry.id for entry in entries}) != len(entries):
        raise PreflightError(PROJECT_INPUTS_INVALID, "workload record ids must be unique")
    return entries


# --- the serving configuration ---------------------------------------------------------


# Engine-neutral names a configuration may use for the case fields. The stored names stay, so
# the configuration digest and every project identity are those of the stored names.
CASE_ALIASES = {
    "max_context_len": "max_model_len",
    "max_concurrent_requests": "max_num_seqs",
    "prefill_batch_tokens": "max_num_batched_tokens",
    "kv_memory_fraction": "gpu_memory_utilization",
    "prefix_caching": "prefix_cache",
}


class ServingCase(StrictModel):
    """The serving settings to start from. Every field but the precision pair may be left out:
    the engine then decides."""

    weight_precision: WeightPrecision
    activation_dtype: ActivationDtype
    max_model_len: TokenCount | None = None
    max_num_seqs: Annotated[int, Field(ge=1, le=4096)] | None = None
    max_num_batched_tokens: TokenCount | None = None
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2"] | None = None
    gpu_memory_utilization: Annotated[float, Field(gt=0, le=1)] | None = None
    prefix_cache: bool | None = None
    tool_calling: bool = False

    @model_validator(mode="before")
    @classmethod
    def _stored_names(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        stored = dict(data)
        for neutral, name in CASE_ALIASES.items():
            if neutral in stored:
                if name in stored:
                    raise PydanticCustomError(
                        USER_MESSAGE_ERROR, f"give {name} or {neutral}, not both"
                    )
                stored[name] = stored.pop(neutral)
        return stored


class DocumentWorkload(WorkloadShape):
    """The workload as the configuration declares it: no prompts (they come from the JSONL)."""


SloKey = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]


class ServingDocument(StrictModel):
    """The serving configuration document, exactly as the user writes it.

    ``engine_build``, ``rounds`` and ``objective`` are identity only (nothing else reads them),
    so they may be left out."""

    schema_version: Literal["1"]
    engine_build: Annotated[
        str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,63}$")
    ] = "unspecified"
    case: ServingCase
    workload: DocumentWorkload
    rounds: Annotated[int, Field(ge=1, le=1000)] = 1
    warmup_requests: Annotated[int, Field(ge=0, le=1000)]
    objective: Literal[
        "highest_throughput", "lowest_energy_per_useful_work", "lowest_latency", "lowest_memory"
    ] = "highest_throughput"
    slos: Annotated[dict[SloKey, PositiveFloat], Field(min_length=1, max_length=32)] | None = None


@dataclass(frozen=True, slots=True)
class ServingConfig:
    """What a run needs from the configuration: the case, and the workload with its prompts."""

    engine_build: str
    case: ServingCase
    workload: WorkloadSpec
    warmup_requests: int


def load_serving_config(path: Path, prompts: Sequence[PromptEntry]) -> tuple[ServingConfig, str]:
    """Validate the configuration document against the prompts; return it with its digest."""
    what = "the serving configuration"
    document = parse_document(
        ServingDocument, _read_input(path, what), what, PROJECT_INPUTS_INVALID
    )
    api = document.workload.api
    if any(entry.api != api for entry in prompts):
        raise PreflightError(
            PROJECT_INPUTS_INVALID,
            f"the serving configuration declares api {api!r} but some workload records are not "
            f"{api} records; every record must match the declared api",
        )
    declared = document.workload.structured_output is not None
    if declared and any(entry.response_format is not None for entry in prompts):
        raise PreflightError(
            PROJECT_INPUTS_INVALID,
            "the serving configuration declares structured_output for every request and some "
            "workload records declare their own response_format; declare it in one place",
        )
    try:
        workload = WorkloadSpec(
            prompts=tuple(entry.prompt for entry in prompts if entry.prompt is not None),
            chats=tuple(entry.chat for entry in prompts if entry.messages is not None),
            **document.workload.model_dump(),
        )
    except ValidationError as error:
        raise PreflightError(
            PROJECT_INPUTS_INVALID, f"the workload is not valid: {validation_summary(error)}"
        ) from None
    config = ServingConfig(
        engine_build=document.engine_build,
        case=document.case,
        workload=workload,
        warmup_requests=document.warmup_requests,
    )
    return config, _config_digest(document, workload)


def refuse_named_tool_under_config(config: ServingConfig) -> None:
    """Refuse a new project whose configuration constrains every answer while a record names a
    tool. Projects registered before 0.3.6 keep loading."""
    workload = config.workload
    if workload.structured_output is not None and any(
        isinstance(chat.tool_choice, dict) for chat in workload.chats
    ):
        raise PreflightError(
            PROJECT_INPUTS_INVALID,
            "the serving configuration declares structured_output and some workload records name "
            "a tool in tool_choice: the engine refuses a named tool_choice under structured output",
        )


# --- digests ---------------------------------------------------------------------------


def _config_digest(document: ServingDocument, workload: WorkloadSpec) -> str:
    """Digest of every semantic field, so a whitespace or key-order edit keeps it and any real
    change breaks it."""
    payload = {
        "schema_version": "1",
        "engine_build": document.engine_build,
        # These fields were declarable once. They stay in the payload, always unset, so the
        # digests of projects registered before they were removed still match.
        "case": {**document.case.model_dump(mode="json"), "thinking_profile_id": None},
        "workload": {
            **workload.model_dump(mode="json", exclude={"prompts", "chats"}),
            "input_tokens": None,
            "fixed_length_consent": False,
        },
        "rounds": document.rounds,
        "warmup_requests": document.warmup_requests,
        "objective": document.objective,
        "slos": document.slos,
    }
    return hashlib.sha256(CONFIG_DIGEST_DOMAIN + canonical_json(payload)).hexdigest()


# The record fields every workload digest has hashed since schema 1. A later field joins the
# digest only when a record sets it, so workloads that leave it out keep their digests.
WORKLOAD_V1_FIELDS = frozenset(
    {"id", "prompt", "messages", "tools", "tool_choice", "max_tokens", "reference", "labels"}
)


def workload_digest(prompts: Sequence[PromptEntry], images: PromptImages | None = None) -> str:
    """Digest of the records in order: order is semantic, and so is every field of each. The
    images follow in a part of their own that a workload without images does not have."""
    hasher = hashlib.sha256(WORKLOAD_DIGEST_DOMAIN)
    for entry in prompts:
        record = {
            key: value
            for key, value in entry.model_dump(mode="json").items()
            if key in WORKLOAD_V1_FIELDS or value is not None
        }
        hasher.update(canonical_json(record) + b"\n")
    if images is not None and images.images:
        hasher.update(WORKLOAD_IMAGES_DOMAIN)
        for url, image in sorted(images.images.items()):
            record = {"url": url, "sha256": image.sha256, "size": image.size}
            hasher.update(canonical_json(record) + b"\n")
    return hasher.hexdigest()
