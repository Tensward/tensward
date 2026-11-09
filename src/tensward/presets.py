"""The built-in workloads, the choices a user makes in ``init``, and the inputs they produce."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, cast

from .errors import PROJECT_INPUTS_INVALID, PreflightError
from .files import write_json, write_jsonl
from .formats.gguf import read_gguf
from .settings import Settings
from .workload import ChatTool

WORKLOADS = ("chat", "rag", "agent", "code", "batch")
GOALS = ("ttft", "tpot", "throughput")
GOAL_DEFAULT = {
    "chat": "ttft",
    "rag": "ttft",
    "agent": "ttft",
    "code": "ttft",
    "batch": "throughput",
    "own": "ttft",
}
OUTPUT_TOKENS = {"chat": 256, "rag": 256, "agent": 256, "code": 512, "batch": 256, "own": 256}
THINKING_TOKENS = 1024  # added to the answer's cap for a model whose template thinks first
THINKING_MARKERS = ("<think>", "enable_thinking")
CHOICES_FILE = "choices.json"
INPUTS_DIR = "inputs"
TOOL_PROMPT_EVERY = 2
REQUEST_COUNT = 160
WARMUP_REQUESTS = 4
TOOL_PROMPT_WORDINGS = ("Use {name} to {description}", "Please use {name} to {description}",
                        "I need you to use {name} to {description}")  # fmt: skip
EXAMPLE_VALUES: dict[str, Any] = {
    "string": "example",
    "integer": 1,
    "number": 1,
    "boolean": True,
    "array": [],
    "object": {},
}


@dataclass(frozen=True)
class Choices:
    """What the user chose in ``init``. ``tools`` is ``none``, ``default`` or an absolute path."""

    workload: str
    prompts: Path | None
    tools: str
    structured: bool
    concurrency: int
    goal: str
    baseline: Literal["command", "defaults"]


def tools_default(workload: str) -> str:
    return "default" if workload == "agent" else "none"


def concurrency_default(baseline_settings: Settings | None, hardware_kind: str) -> int:
    """The baseline's own request cap if it sets one, else 8 on a GPU and 1 on a CPU."""
    if baseline_settings is not None and baseline_settings.max_concurrent_requests:
        return baseline_settings.max_concurrent_requests
    return 1 if hardware_kind == "cpu" else 8


def write_choices(project: Path, choices: Choices) -> None:
    document = asdict(choices)
    document["prompts"] = None if choices.prompts is None else str(choices.prompts)
    write_json(project / CHOICES_FILE, document)


def read_choices(project: Path) -> Choices | None:
    path = project / CHOICES_FILE
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_bytes())
        prompts = document["prompts"]
        return Choices(**{**document, "prompts": None if prompts is None else Path(prompts)})
    except (ValueError, KeyError, TypeError):
        raise PreflightError(
            PROJECT_INPUTS_INVALID, f"{path} is not a Tensward choices file"
        ) from None


def like(project: Path, target: Path) -> dict[str, Any]:
    """``init --like``: the workload choices of ``project`` (its ``.tensward`` folder, or the
    folder holding one) as init's arguments, so a new project measures the same workload. The
    new project ``target`` must be another one."""
    found = project if (project / "project.json").is_file() else project / ".tensward"
    choices = read_choices(found)
    if choices is None:
        raise PreflightError(PROJECT_INPUTS_INVALID, f"--like: {project} is not a Tensward project")
    if found.resolve() == target.resolve():
        raise PreflightError(
            PROJECT_INPUTS_INVALID,
            "--like names this folder's own project; run init in a new folder",
        )
    if choices.prompts is not None and not choices.prompts.is_file():
        raise PreflightError(
            PROJECT_INPUTS_INVALID, f"--like: its prompts file {choices.prompts} is gone"
        )
    if problem := _like_problem(choices):
        raise PreflightError(PROJECT_INPUTS_INVALID, f"--like: {found / CHOICES_FILE} {problem}")
    own = choices.prompts is not None
    return {
        "workload": None if own else choices.workload,
        "prompts": choices.prompts,
        "tools": choices.tools,
        "structured": choices.structured,
        "concurrency": choices.concurrency,
        "goal": choices.goal,
    }


def _like_problem(choices: Choices) -> str | None:
    """What in another project's choices init itself would refuse, if anything."""
    if choices.workload not in (*WORKLOADS, "own"):
        return f"has an unknown workload {choices.workload!r}"
    if choices.goal not in GOALS:
        return f"has an unknown goal {choices.goal!r}; it is one of {', '.join(GOALS)}"
    concurrency = choices.concurrency
    if type(concurrency) is not int or concurrency < 1:
        return f"has concurrency {concurrency!r}; it is a whole number of 1 or more"
    if not isinstance(choices.structured, bool):
        return f"has structured {choices.structured!r}; it is true or false"
    if not isinstance(choices.tools, str) or (
        choices.tools not in ("none", "default") and not Path(choices.tools).is_absolute()
    ):
        return f"has tools {choices.tools!r}; it is none, default or a full path"
    return None


def load_tools(path: Path) -> list[dict[str, Any]]:
    """Read an OpenAI tools file: a JSON list of ``{"type": "function", "function": {...}}``."""
    try:
        tools = json.loads(path.read_bytes())
        if not isinstance(tools, list) or not tools:
            raise ValueError("not a non-empty list")
        for tool in tools:
            ChatTool.model_validate(tool)
            if not tool["function"].get("description"):
                raise ValueError("a tool has no description")
    except (OSError, ValueError) as error:
        raise PreflightError(
            PROJECT_INPUTS_INVALID,
            f"{path} is not a list of OpenAI function tools: {error}",
        ) from None
    return cast("list[dict[str, Any]]", tools)


def tool_prompts(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Three user messages per tool, each asking for the tool with an example of its arguments."""
    records = []
    for tool in tools:
        function = tool["function"]
        name = function["name"]
        arguments = json.dumps(_example_arguments(function["parameters"]))
        for index, wording in enumerate(TOOL_PROMPT_WORDINGS):
            text = wording.format(name=name, description=function.get("description") or name)
            records.append(
                {
                    "id": f"tool-{name}-{index}",
                    "messages": [{"role": "user", "content": f"{text}. Arguments: {arguments}"}],
                }
            )
    return records


def thinks(model: Path) -> bool:
    """Whether the model's chat template makes it reason before it answers."""
    try:
        if model.is_file():
            template = read_gguf(model).metadata.get("tokenizer.chat_template")
        else:
            config = json.loads((model / "tokenizer_config.json").read_text("utf-8"))
            template = config.get("chat_template")
            if not template and (model / "chat_template.jinja").is_file():
                template = (model / "chat_template.jinja").read_text("utf-8")
    except (OSError, ValueError, AttributeError):
        return False
    return isinstance(template, str) and any(marker in template for marker in THINKING_MARKERS)


def output_tokens(workload: str, *, thinking: bool) -> int:
    """The output cap of a workload: its answer, and room to think first for a thinking model."""
    return OUTPUT_TOKENS[workload] + (THINKING_TOKENS if thinking else 0)


def write_inputs(
    project: Path, choices: Choices, *, precision: tuple[str, str], thinking: bool = False
) -> tuple[Path, Path]:
    """Write ``inputs/config.json`` and ``inputs/prompts.jsonl`` for the choices."""
    inputs = project / INPUTS_DIR
    inputs.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": "1",
        "case": {
            "weight_precision": precision[0],
            "activation_dtype": precision[1],
            "tool_calling": choices.tools != "none",
        },
        "workload": {
            "api": "chat",
            "request_count": REQUEST_COUNT,
            "request_timeout_s": 300.0,
            "arrival": {"kind": "closed_loop", "concurrency": choices.concurrency},
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "output_tokens": output_tokens(choices.workload, thinking=thinking),
        },
        "warmup_requests": WARMUP_REQUESTS,
    }
    config_path = inputs / "config.json"
    prompts_path = inputs / "prompts.jsonl"
    write_json(config_path, config)
    write_jsonl(prompts_path, _records(choices))
    return config_path, prompts_path


def _records(choices: Choices) -> list[dict[str, Any]]:
    workload = choices.workload
    if workload == "own":
        if choices.prompts is None:
            raise PreflightError(PROJECT_INPUTS_INVALID, "the own workload needs a prompts file")
        text = choices.prompts.read_text("utf-8")
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = _bundled_records(workload)
    if choices.tools == "none":
        records = [_without_tools(r) for r in records if not _has_tool_calls(r)]
        if not records:
            raise PreflightError(PROJECT_INPUTS_INVALID, "no workload record is left without tools")
    else:
        tools = _tools_for(choices.tools)
        if choices.tools != "default" and workload == "agent":
            records = _multi_turn_prompts(tools)
        elif workload != "agent":
            offered = _bundled_records("tool-prompts") if choices.tools == "default" else None
            records = _mix_tool_prompts(
                records, offered if offered is not None else tool_prompts(tools)
            )
        records = [{**r, "tools": tools} for r in records]
    if choices.structured:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": workload, "schema": _structured_schemas()[workload]},
        }
        records = [
            r if r.get("tools") else {**r, "response_format": response_format} for r in records
        ]
    return _unique_ids(records)


def _bundled_records(name: str) -> list[dict[str, Any]]:
    if name not in (*WORKLOADS, "tool-prompts"):
        raise PreflightError(PROJECT_INPUTS_INVALID, f"{name!r} is not a built-in workload")
    text = files("tensward").joinpath("data", "workloads", f"{name}.jsonl").read_text("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _structured_schemas() -> dict[str, Any]:
    text = files("tensward").joinpath("data", "workloads", "structured.json").read_text("utf-8")
    return cast("dict[str, Any]", json.loads(text))


def _tools_for(tools: str) -> list[dict[str, Any]]:
    if tools == "default":
        text = files("tensward").joinpath("data", "tools", "sample-tools.json").read_text("utf-8")
        return cast("list[dict[str, Any]]", json.loads(text))
    return load_tools(Path(tools))


def _mix_tool_prompts(
    records: list[dict[str, Any]], prompts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Every ``TOOL_PROMPT_EVERY``-th record becomes the next tool prompt."""
    mixed = list(records)
    for slot, index in enumerate(range(TOOL_PROMPT_EVERY - 1, len(mixed), TOOL_PROMPT_EVERY)):
        mixed[index] = prompts[slot % len(prompts)]
    return mixed


def _multi_turn_prompts(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Agent prompts over the user's own tools: for each tool, a bare request that needs a call
    and a history (ask, call, result, follow-up) that does not."""
    records = []
    for tool in tools:
        name = tool["function"]["name"]
        record = tool_prompts([tool])[0]
        call_id = f"call_{name}"
        arguments = json.dumps(_example_arguments(tool["function"]["parameters"]))
        call = {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }
        history = [
            *record["messages"],
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call_id, "content": "ok"},
            {"role": "user", "content": "Thanks. Summarise what you found."},
        ]
        records.append(record)
        records.append({**record, "id": f"{record['id']}-history", "messages": history})
    return records


def _example_arguments(parameters: dict[str, Any]) -> dict[str, Any]:
    properties = parameters.get("properties", {})
    example: dict[str, Any] = {}
    for key in parameters.get("required", []):
        schema = properties.get(key, {})
        if "enum" in schema:
            example[key] = schema["enum"][0]
        else:
            example[key] = EXAMPLE_VALUES.get(schema.get("type"), "example")
    return example


def _has_tool_calls(record: dict[str, Any]) -> bool:
    return any(m.get("tool_calls") or m.get("role") == "tool" for m in record.get("messages", ()))


def _without_tools(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k not in ("tools", "tool_choice")}


def _unique_ids(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, int] = {}
    unique = []
    for record in records:
        count = seen.get(record["id"], 0)
        seen[record["id"]] = count + 1
        unique.append(record if count == 0 else {**record, "id": f"{record['id']}-{count + 1}"})
    return unique
