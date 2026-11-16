"""Ollama in ``tensward init``: find it, read the settings that decide whether long prompts are
cut, say what to change, and on a yes link or copy a model into ``~/models`` and carry on with
llama.cpp on that file.

Nothing here starts, stops or restarts Ollama, writes to its store or runs ``sudo``."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

import httpx

from . import ollama_import as store
from .cli_options import JSON_OUTPUT
from .cli_try import next_lines
from .detect import Hardware, detect_engines, detect_hardware, ollama_processes, running_servers
from .engines.llamacpp.detect import (
    ENDPOINT_MAX_BYTES,
    ENDPOINT_TIMEOUT_S,
    EndpointFacts,
    _get,
    _json,
    command_evidence,
)
from .errors import (
    ENGINE_UNAVAILABLE,
    PROJECT_CONFIG_UNSUPPORTED,
    PROJECT_INPUTS_INVALID,
    PreflightError,
    exit_code_for,
)
from .fit import CHARS_PER_TOKEN
from .probes import run_probe
from .text import printable

if TYPE_CHECKING:
    import argparse

    from .wizard import Asker

DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_PORT = 11434
DOCKER_TIMEOUT_S = 3.0
GIB = 2**30
NEAR_SHARE = 0.75
MIN_CONTEXT = 4096
CONTEXT_STEP = 1024
DEFAULT_OUTPUT_TOKENS = 1024
FOREVER = "-1"
RENDERED = frozenset({"renderer_parser", "renderer", "parser", "harmony", "go_template"})
DEFAULT_KEEP_ALIVE_S = 300
NEVER_S = 100 * 365 * 86400
STOPPED_NOTE = (
    "Ollama is installed here but not answering, so Tensward won't read its settings. "
    "Carrying on with a normal setup."
)
NO_MODELS_NOTE = "Ollama is running but has no models yet."
SCRIPTED_HINT = "Ollama is running; to set up one of its models, run tensward init in a terminal."
UNMEASURABLE = (
    "Tensward can't measure Ollama itself yet; it compares your changes against llama.cpp"
)
NOT_NAMED = "a different model (not from Ollama)"


@dataclass(slots=True)
class Where:
    """How Ollama runs here, as far as this user can read it."""

    kind: Literal["service", "docker", "plain", "unknown"] = "unknown"
    env: dict[str, str] = field(default_factory=dict)
    child_np: int | None = None
    container: str | None = None
    container_root: str | None = None
    mounts: tuple[tuple[str, str], ...] = ()
    systemd_user: str | None = None
    process: bool = False
    installed: bool = False
    published: str | None = None
    binary_on_host: bool = False
    env_read: bool = False


@dataclass(slots=True)
class State:
    """What Ollama answered."""

    url: str
    where: Where
    version: str | None = None
    tags: list[dict[str, Any]] | None = None
    loaded: list[dict[str, Any]] = field(default_factory=list)
    endpoint: bool = False
    hardware: Hardware | None = None

    @property
    def shown(self) -> str:
        """The URL without a user name or password."""
        parts = urlsplit(self.url)
        host = parts.hostname or ""
        host = f"[{host}]" if ":" in host else host
        return f"{parts.scheme}://{host}" + (f":{parts.port}" if parts.port else "")

    @property
    def local(self) -> bool:
        return urlsplit(self.url).hostname in ("127.0.0.1", "localhost", "::1")

    @property
    def answers(self) -> bool:
        return self.version is not None


def normalise_host(text: str) -> str | None:
    """A base URL from ``OLLAMA_HOST`` as Ollama reads it: ``0.0.0.0`` or a bare ``:port`` mean
    this machine, the scheme is http unless given."""
    text = text.strip()
    if not text:
        return None
    scheme, explicit = "http", "://" in text
    if explicit:
        scheme, _, text = text.partition("://")
        if scheme not in ("http", "https"):
            return None
    try:
        parts = urlsplit(f"{scheme}://{text}")
        host, port = parts.hostname, parts.port
    except ValueError:
        return None
    if host in (None, "", "0.0.0.0", "::"):
        host = "127.0.0.1"
    if port is None:
        port = 443 if scheme == "https" and explicit else DEFAULT_PORT
    shown = f"[{host}]" if ":" in host else host
    return f"{scheme}://{shown}:{port}"


def _split_environment(text: str) -> dict[str, str]:
    try:
        words = shlex.split(text)
    except ValueError:
        return {}
    return {k: v for k, _, v in (w.partition("=") for w in words if w.startswith("OLLAMA_"))}


def _systemd() -> tuple[dict[str, str], str | None, bool, bool]:
    """The environment, the user, whether a unit ``ollama`` exists and whether it runs."""
    if shutil.which("systemctl") is None:
        return {}, None, False, False
    out = run_probe(
        ["systemctl", "show", "ollama", "-p", "Environment", "-p", "User", "-p", "LoadState",
         "-p", "ActiveState"]
    )  # fmt: skip
    fields = dict(line.partition("=")[::2] for line in (out or "").splitlines())
    loaded = fields.get("LoadState") == "loaded"
    active = loaded and fields.get("ActiveState") in ("active", "activating", "reloading")
    environment = _split_environment(fields.get("Environment", ""))
    return environment, fields.get("User") or None, loaded, active


def _process_environment(pid: int) -> dict[str, str] | None:
    """The process's OLLAMA_ variables, or None when this user can't read them."""
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return None
    pairs = (item.partition("=") for item in raw.decode(errors="replace").split("\0"))
    return {key: value for key, _, value in pairs if key.startswith("OLLAMA_")}


def _process_cgroup(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cgroup").read_text()
    except OSError:
        return ""


def _child_parallel(pid: int) -> int | None:
    """``-np`` of the ``llama-server`` that the Ollama at ``pid`` started."""
    from .detect import _processes

    for child, parent, command in _processes():
        if parent != pid or not command.split()[0].endswith("llama-server"):
            continue
        words = command.split()
        for flag, value in zip(words, words[1:], strict=False):
            if flag in ("-np", "--parallel") and value.isdigit():
                return int(value)
    return None


def _docker_output(argv: list[str]) -> str | None:
    """Standard output of a docker command, given ``DOCKER_TIMEOUT_S`` to answer."""
    try:
        done = subprocess.run(
            argv, capture_output=True, text=True, stdin=subprocess.DEVNULL,
            timeout=DOCKER_TIMEOUT_S, check=False,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return done.stdout if done.returncode == 0 else None


def _docker() -> Where | None:
    """The running container made from an ollama/ollama image, its environment, published port
    and mounts, when this user may ask docker."""
    from .engines.llamacpp.detect import _LOOKALIKE_IMAGE

    if shutil.which("docker") is None:
        return None
    out = _docker_output(["docker", "ps", "--format", "{{.Names}} {{.Image}}"])
    for line in (out or "").splitlines():
        name, _, image = line.partition(" ")
        if _LOOKALIKE_IMAGE.search(image) and store.CONTAINER_NAME.match(name):
            where = Where(kind="docker", container=name, process=True)
            inspected = _docker_output(["docker", "inspect", name, "--format", "{{json .}}"])
            try:
                data = json.loads(inspected or "")
            except ValueError:
                return where
            _read_container(where, data)
            return where
    return None


def _read_container(where: Where, data: Any) -> None:
    if not isinstance(data, dict):
        return
    config = data.get("Config")
    where.env_read = isinstance(config, dict)
    for item in (config.get("Env") if isinstance(config, dict) else None) or []:
        if isinstance(item, str) and item.startswith("OLLAMA_"):
            key, _, value = item.partition("=")
            where.env[key] = value
    ports = (data.get("NetworkSettings") or {}).get("Ports") or {}
    bound = ports.get(f"{DEFAULT_PORT}/tcp") if isinstance(ports, dict) else None
    if isinstance(bound, list) and bound and isinstance(bound[0], dict):
        port = bound[0].get("HostPort")
        if isinstance(port, str) and port.isdigit():
            where.published = f"http://127.0.0.1:{port}"
    where.mounts = tuple(
        (str(m["Destination"]), str(m["Source"]))
        for m in data.get("Mounts") or []
        if isinstance(m, dict)
        and m.get("Type") == "bind"
        and m.get("Destination")
        and m.get("Source")
    )
    where.container_root = where.env.get("OLLAMA_MODELS") or store.CONTAINER_ROOT


def _loopback(url: str) -> bool:
    return urlsplit(url).hostname in ("127.0.0.1", "localhost", "::1")


def discover(arguments: argparse.Namespace | None, command: str | None) -> tuple[Where, str]:
    """How Ollama runs here and its base URL: ``--endpoint``, then ``OLLAMA_HOST`` from the
    shell, the process and the service, and the default. A remote ``--endpoint`` has none of
    this machine's settings."""
    endpoint = getattr(arguments, "endpoint", None)
    if endpoint and not _loopback(str(endpoint)):
        return Where(), str(endpoint).rstrip("/")
    where = Where(binary_on_host=shutil.which("ollama") is not None)
    where.installed = where.binary_on_host
    processes = ollama_processes()
    if processes:
        where.kind, where.process = "plain", True
        if "docker" in (group := _process_cgroup(processes[0][0])) or "containerd" in group:
            where.kind = "docker"
        if (found := _process_environment(processes[0][0])) is not None:
            where.env.update(found)
            where.env_read = True
        where.child_np = _child_parallel(processes[0][0])
    unit_env, where.systemd_user, unit, running = _systemd()
    where.installed = where.installed or unit
    if running and where.kind == "plain":
        where.kind = "service"
        where.env_read = True
    if where.kind == "service" or not processes:
        for key, value in unit_env.items():
            where.env.setdefault(key, value)
    if command:
        for key, value in _split_environment(command.replace("-e ", " ")).items():
            where.env.setdefault(key, value)
    host = os.environ.get("OLLAMA_HOST") or where.env.get("OLLAMA_HOST", "")
    url = str(endpoint).rstrip("/") if endpoint else normalise_host(host) or DEFAULT_URL
    return where, url


def _with_container(where: Where, found: Where) -> Where:
    for key, value in where.env.items():
        found.env.setdefault(key, value)
    found.binary_on_host = where.binary_on_host
    found.installed = True
    return found


def _post(url: str, body: Mapping[str, object]) -> Any:
    try:
        with (
            httpx.Client() as client,
            client.stream("POST", url, json=dict(body), timeout=ENDPOINT_TIMEOUT_S) as response,
        ):
            data = b""
            for chunk in response.iter_bytes():
                data += chunk
                if len(data) > ENDPOINT_MAX_BYTES:
                    return None
            return _body(httpx.Response(response.status_code, content=data))
    except httpx.HTTPError:
        return None


def _body(response: httpx.Response | None) -> Any:
    """The JSON of a reply, None when it is not JSON or is nested beyond what Python reads."""
    try:
        return _json(response)
    except (RecursionError, MemoryError):
        return None


def _list(value: Any, key: str) -> list[dict[str, Any]] | None:
    items = value.get(key) if isinstance(value, dict) else None
    if not isinstance(items, list):
        return None
    return [item for item in items if isinstance(item, dict)]


def _ask(state: State, facts: EndpointFacts | None) -> None:
    """Fill ``state`` from the API: the version (from the endpoint probe when there was one),
    the models and the loaded ones."""
    with httpx.Client() as client:
        state.version = printable(facts.version) if facts is not None and facts.version else None
        if state.version is None:
            body = _body(_get(client, f"{state.url}/api/version", {}))
            if isinstance(body, dict) and isinstance(body.get("version"), str):
                state.version = printable(body["version"])
        if state.version is None:
            return
        state.tags = _list(_body(_get(client, f"{state.url}/api/tags", {})), "models")
        state.loaded = _list(_body(_get(client, f"{state.url}/api/ps", {})), "models") or []


def read_state(
    arguments: argparse.Namespace | None, command: str | None, facts: EndpointFacts | None = None
) -> State:
    """What Ollama answers. Docker is asked only when there is a sign of Ollama: it answered,
    is installed, or the user named an endpoint or a command."""
    where, url = discover(arguments, command)
    endpoint = getattr(arguments, "endpoint", None)
    state = State(url, where, endpoint=bool(endpoint))
    _ask(state, facts)
    asked = state.answers or where.installed or bool(endpoint or command)
    unseen = not where.process or (where.kind == "docker" and where.container is None)
    if unseen and _loopback(url) and asked and (found := _docker()) is not None:
        state.where = _with_container(where, found)
        if (
            not state.answers
            and not endpoint
            and not os.environ.get("OLLAMA_HOST")
            and found.published
        ):
            state.url = found.published
            _ask(state, facts)
    return state


def show_model(state: State, name: str) -> dict[str, Any]:
    body = _post(f"{state.url}/api/show", {"model": name})
    return body if isinstance(body, dict) else {}


def _names(state: State) -> list[str]:
    return [m["name"] for m in state.tags or [] if isinstance(m.get("name"), str)]


def _tag(state: State, name: str) -> dict[str, Any]:
    return next((m for m in state.tags or [] if m.get("name") == name), {})


def _loaded(state: State, name: str) -> dict[str, Any] | None:
    return next((m for m in state.loaded if name in (m.get("name"), m.get("model"))), None)


def _count(value: Any) -> int | None:
    """A byte or token count from a reply: a whole number below 2**50, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = int(value)
    except (OverflowError, ValueError):
        return None
    return number if 0 <= number < 2**50 else None


def _gib(size: Any) -> str:
    number = _count(size)
    return "?" if number is None else f"{number / GIB:.1f}"


def stop_command(state: State, name: str) -> str:
    stop = f"ollama stop {shlex.quote(printable(name))}"
    if state.where.kind == "docker" and not state.where.binary_on_host and state.where.container:
        return f"docker exec {shlex.quote(state.where.container)} {stop}"
    return stop


def _holder() -> tuple[State, str, Any] | None:
    """The model Ollama holds on this machine, if it answers on the local URL."""
    url = normalise_host(os.environ.get("OLLAMA_HOST", "")) or DEFAULT_URL
    if not _loopback(url):
        return None
    with httpx.Client() as client:
        loaded = _list(_body(_get(client, f"{url}/api/ps", {})), "models")
    if not loaded or not isinstance(loaded[0].get("name"), str):
        return None
    where = Where(binary_on_host=shutil.which("ollama") is not None)
    if not where.binary_on_host and (found := _docker()) is not None:
        where = found
    return State(url, where, loaded=loaded), loaded[0]["name"], loaded[0].get("size")


def loaded_note() -> str | None:
    """The note when Ollama has a model loaded and the user sets up another way."""
    held = _holder()
    if held is None:
        return None
    state, name, size = held
    where = "in memory" if detect_hardware().kind == "cpu" else "on this GPU"
    return (
        f"Ollama has {printable(name)} loaded {where} ({_gib(size)} GiB). "
        f"Free it first: {stop_command(state, name)}"
    )


def busy_hint() -> str | None:
    """What to add to a busy-memory message when Ollama holds a model."""
    held = _holder()
    if held is None:
        return None
    state, name, _ = held
    return f"Ollama has {printable(name)} loaded. Free it: {stop_command(state, name)}"


@dataclass(frozen=True, slots=True)
class Value:
    value: int | None
    source: str
    estimate: bool = False


def _parameters(show: dict[str, Any]) -> dict[str, str]:
    text = show.get("parameters")
    found: dict[str, str] = {}
    for line in text.splitlines() if isinstance(text, str) else []:
        key, _, value = line.strip().partition(" ")
        if key and key not in found:
            found[key] = value.strip()
    return found


def _training_context(show: dict[str, Any]) -> int | None:
    info = show.get("model_info")
    if not isinstance(info, dict):
        return None
    arch = info.get("general.architecture")
    value = info.get(f"{arch}.context_length") if isinstance(arch, str) else None
    return value if isinstance(value, int) and value > 0 else None


def vram_default(total_bytes: int, overhead: int = 0) -> int:
    """Ollama's default context for the memory it sees (``server/routes.go``)."""
    free = total_bytes - overhead
    return 262144 if free >= 47 * GIB else 32768 if free >= 23 * GIB else 4096


def _positive(text: str | None) -> int | None:
    return int(text) if text and text.isdecimal() and int(text) > 0 else None


def context_value(state: State, name: str, show: dict[str, Any], gpu_bytes: int | None) -> Value:
    """The context per request, from the first source that answers; an estimate when none."""
    loaded = _loaded(state, name)
    cap = _training_context(show)
    found: Value | None = None
    if loaded and _count(loaded.get("context_length")):
        found = Value(_count(loaded.get("context_length")), "loaded")
    elif (number := _positive(_parameters(show).get("num_ctx"))) is not None:
        found = Value(number, "modelfile")
    elif (number := _positive(state.where.env.get("OLLAMA_CONTEXT_LENGTH"))) is not None:
        found = Value(number, "environment")
    if found is None:
        overhead = _positive(state.where.env.get("OLLAMA_GPU_OVERHEAD")) or 0
        default = vram_default(gpu_bytes, overhead) if gpu_bytes else MIN_CONTEXT
        found = Value(default, "default", estimate=True)
    if cap is not None and found.value is not None and found.value > cap:
        return Value(cap, found.source, found.estimate)
    return found


def keep_alive_seconds(text: str) -> int | None:
    """``OLLAMA_KEEP_ALIVE`` as seconds; negative means forever."""
    text = text.strip()
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    parts = re.findall(r"(\d+(?:\.\d+)?)(h|m|s)", text)
    if not parts or "".join(a + b for a, b in parts) != text:
        return -1 if text.startswith("-") else None
    return int(sum(float(n) * {"h": 3600, "m": 60, "s": 1}[u] for n, u in parts))


def keep_alive(state: State, name: str) -> tuple[int | None, bool]:
    """Seconds until an idle model unloads (negative: never) and whether it is the default."""
    if "OLLAMA_KEEP_ALIVE" in state.where.env:
        seconds = keep_alive_seconds(state.where.env["OLLAMA_KEEP_ALIVE"])
        if seconds is not None:
            return (-1 if seconds < 0 else seconds), False
    loaded = _loaded(state, name)
    when = loaded.get("expires_at") if loaded else None
    if isinstance(when, str):
        try:
            left = (datetime.fromisoformat(when) - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError):
            left = None
        if left is not None:
            return (-1 if left > NEVER_S else max(0, round(left))), False
    return DEFAULT_KEEP_ALIVE_S, True


def parallel(state: State) -> tuple[int, bool]:
    """Requests at once and whether that is only Ollama's default."""
    if state.where.child_np:
        return state.where.child_np, False
    found = _positive(state.where.env.get("OLLAMA_NUM_PARALLEL"))
    return (found, False) if found else (1, True)


def cut_style(show: dict[str, Any]) -> Literal["rendered", "native", "unknown"]:
    source = show.get("template_source")
    if not isinstance(source, str):
        return "unknown"
    return "rendered" if source in RENDERED else "native"


@dataclass(frozen=True, slots=True)
class Prompts:
    over: int
    near: int
    of: int
    longest: int
    output_tokens: int | None


def _message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    text = content if isinstance(content, str) else json.dumps(content or "")
    calls = message.get("tool_calls")
    return text + (json.dumps(calls) if calls else "")


def estimate_tokens(record: Mapping[str, Any]) -> int:
    """Characters of the messages and the tools over ``CHARS_PER_TOKEN``: low for tool-heavy
    JSON, so always an estimate."""
    messages = record.get("messages")
    text = (
        "".join(_message_text(m) for m in messages)
        if isinstance(messages, list)
        else str(record.get("prompt") or "")
    )
    tools = record.get("tools")
    return (len(text) + (len(json.dumps(tools)) if tools else 0)) // CHARS_PER_TOKEN


def check_prompts(records: Sequence[Mapping[str, Any]], context: int) -> Prompts | None:
    if not records:
        return None
    sizes = [estimate_tokens(record) for record in records]
    limits = [r["max_tokens"] for r in records if isinstance(r.get("max_tokens"), int)]
    return Prompts(
        over=sum(size > context for size in sizes),
        near=sum(NEAR_SHARE * context <= size <= context for size in sizes),
        of=len(sizes),
        longest=max(sizes),
        output_tokens=max(limits) if limits else None,
    )


def needed_context(prompts: Prompts) -> int:
    room = prompts.longest + (prompts.output_tokens or DEFAULT_OUTPUT_TOKENS)
    return max(MIN_CONTEXT, math.ceil(room / CONTEXT_STEP) * CONTEXT_STEP)


def prompt_records(arguments: argparse.Namespace) -> list[dict[str, Any]]:
    """The prompts ``--prompts`` or ``--workload`` names, as records."""
    from .inputs import load_prompt_entries
    from .presets import Choices, _records, tools_default

    if arguments.prompts:
        entries = load_prompt_entries(Path(arguments.prompts).expanduser())
        return [e.model_dump(mode="json", exclude_none=True) for e in entries]
    if arguments.workload:
        workload = arguments.workload
        return _records(
            Choices(workload, None, tools_default(workload), False, 1, "ttft", "defaults")
        )
    return []


@dataclass(slots=True)
class Facts:
    state: State
    name: str
    show: dict[str, Any]
    hardware: str
    context: Value
    parallel: int
    parallel_default: bool
    keep: int | None
    keep_default: bool
    prompts: Prompts | None
    concurrency: int | None
    modelfile: bool
    style: Literal["rendered", "native", "unknown"]
    fixes: list[tuple[str, str]]
    context_fix: int | None
    model_limit: int | None


def build_facts(
    state: State,
    name: str,
    show: dict[str, Any],
    records: list[dict[str, Any]],
    concurrency: int | None,
) -> Facts:
    hardware = state.hardware or detect_hardware()
    gpu = sum(size for _, size in hardware.devices) or None
    context = context_value(state, name, show, gpu)
    count, count_default = parallel(state)
    seconds, keep_default = keep_alive(state, name)
    prompts = check_prompts(records, context.value or MIN_CONTEXT)
    modelfile = "num_ctx" in _parameters(show)
    fixes: list[tuple[str, str]] = []
    context_fix = model_limit = None
    if prompts is not None:
        wanted = needed_context(prompts)
        cap = _training_context(show)
        if cap is not None and wanted > cap:
            model_limit = cap
            wanted = cap
        if wanted > (context.value or 0):
            context_fix = wanted
            if not modelfile:
                fixes.append(("OLLAMA_CONTEXT_LENGTH", str(wanted)))
    if concurrency is not None and concurrency > count:
        fixes.append(("OLLAMA_NUM_PARALLEL", str(concurrency)))
    if seconds is not None and seconds >= 0:
        fixes.append(("OLLAMA_KEEP_ALIVE", FOREVER))
    return Facts(
        state, name, show, hardware.line(), context, count, count_default, seconds,
        keep_default, prompts, concurrency, modelfile, cut_style(show), fixes, context_fix,
        model_limit,
    )  # fmt: skip


def _vram_share(loaded: dict[str, Any] | None) -> float | None:
    """The share of a loaded model that sits in GPU memory."""
    size, vram = _count((loaded or {}).get("size")), _count((loaded or {}).get("size_vram"))
    return vram / size if size and vram is not None else None


def _on_cpu(state: State) -> bool:
    return state.hardware is not None and state.hardware.kind == "cpu"


def _model_line(facts: Facts) -> str:
    tag = _tag(facts.state, facts.name)
    found = tag.get("details")
    details: dict[str, Any] = found if isinstance(found, dict) else {}
    parts = [
        printable(str(details["format"])).upper() if details.get("format") else None,
        printable(str(details["quantization_level"]))
        if details.get("quantization_level")
        else None,
        f"{_gib(tag.get('size'))} GiB" if tag.get("size") else None,
    ]
    shown = ", ".join(p for p in parts if p)
    loaded = _loaded(facts.state, facts.name)
    where = ", not loaded"
    if loaded:
        share = None if _on_cpu(facts.state) else _vram_share(loaded)
        where = ", loaded" + (f", {round(100 * share)}% on the GPU" if share is not None else "")
    return printable(facts.name) + (f" ({shown})" if shown else "") + where


def _keep_line(facts: Facts) -> str:
    if facts.keep is None:
        return "Unloads:   unknown"
    if facts.keep < 0:
        return "Unloads:   never (kept loaded)"
    text = f"after {max(1, round(facts.keep / 60))} min idle"
    if facts.keep_default:
        text += _default_note(facts)
    return f"Unloads:   {text}"


def _default_note(facts: Facts) -> str:
    if facts.state.where.env_read:
        return " (Ollama's default)"
    return " (Ollama's default; your setting could not be read)"


def _parallel_line(facts: Facts) -> str:
    if facts.parallel_default and not facts.state.local:
        return (
            "Could not read: how many requests run at once (needs root, or Ollama on this machine)."
        )
    text = f"{facts.parallel} request{'s' if facts.parallel != 1 else ''} at a time"
    if facts.parallel_default:
        text += _default_note(facts)
    return f"Parallel:  {text}"


def _context_line(facts: Facts) -> str:
    text = f"{facts.context.value} tokens per request"
    if facts.context.estimate:
        text += (
            " (Ollama's default, an estimate)"
            if _on_cpu(facts.state)
            else " (Ollama's default for this GPU, an estimate)"
        )
    return f"Context:   {text}"


def _prompt_lines(facts: Facts) -> list[str]:
    prompts, context = facts.prompts, facts.context.value or MIN_CONTEXT
    if prompts is None:
        return [
            f"Your prompts are not checked. {context} is short for agents and long documents; "
            "a longer prompt is cut with no error."
        ]
    if prompts.over:
        tail = f"{context // 2 // 100 * 100:,}"
        head = f"Too long: {prompts.over} of {prompts.of} prompts are over {context} tokens "
        head += f"(estimate: {CHARS_PER_TOKEN} characters per token)."
        say = {
            "rendered": (
                "Ollama cuts these without an error: a long chat loses its oldest messages, "
                f"and one long prompt\nkeeps only its first few words and its last {tail} "
                "tokens (half the context). Answers can be wrong and nothing tells you."
            ),
            "native": (
                "A long chat loses its oldest messages without an error. One prompt that is "
                "too long to fit is refused with an error (HTTP 400), not cut."
            ),
            "unknown": (
                "A long chat loses its oldest messages without an error. One long prompt is "
                "either cut without an error or refused with one (HTTP 400)."
            ),
        }[facts.style]
        return [head, say]
    if prompts.near:
        return [
            f"Close to the limit: your longest prompt is about {prompts.longest:,} of {context} "
            "tokens (estimate). The answer needs room too, and past the limit Ollama discards "
            "part of the prompt."
        ]
    return [
        f"Your prompts fit: the longest is about {prompts.longest:,} of {context} tokens "
        "(estimate)."
    ]


def _fix_lines(facts: Facts) -> list[str]:
    state = facts.state
    lines: list[str] = []
    settings = list(facts.fixes)
    if facts.context_fix is None and facts.prompts is None:
        lines.append("Context: give your prompts to get a number: tensward init --prompts FILE")
    if facts.concurrency is None:
        lines.append("Set OLLAMA_NUM_PARALLEL to your peak concurrent requests.")
    if facts.model_limit is not None:
        lines.append(
            f"Your longest prompt is longer than this model's {facts.model_limit}-token limit: "
            "no setting keeps it whole."
        )
    if facts.context_fix is not None and facts.modelfile:
        lines += _modelfile_lines(facts)
    if not settings:
        return lines
    if any(key == "OLLAMA_NUM_PARALLEL" for key, _ in settings):
        lines.append(
            "More slots serve more requests at once; each answer streams a little slower "
            "(measured: +9% to +154% per token)."
        )
    if state.where.kind == "docker":
        flags = " ".join(f"-e {k}={v}" for k, v in settings)
        unread = (
            [
                "Ollama runs in Docker and your user can't ask docker, "
                "so these settings come from the API only."
            ]
            if state.where.container is None
            else []
        )
        return [
            *lines,
            *unread,
            "To fix it, add these -e options to the command that starts your container "
            "(compose: under environment:), then recreate it:",
            f"  {flags}",
        ]
    pairs = " ".join(f"{k}={v}" for k, v in settings)
    if state.where.kind in ("plain", "unknown") or not state.local:
        return [*lines, f"Set these before you start it: {pairs} ollama serve"]
    check = (
        "afterwards check that free RAM stays above what Ollama uses; if it runs short, lower "
        "parallel or the context."
        if _on_cpu(state)
        else "afterwards `ollama ps` should show 100% GPU. If it shows a CPU share, lower "
        "parallel or the context."
    )
    return [
        *lines,
        "To fix it, add these to your service (sudo systemctl edit ollama), "
        "then restart it yourself:",
        "  [Service]",
        *(f'  Environment="{k}={v}"' for k, v in settings),
        "Then:   sudo systemctl restart ollama",
        "Starting points, not measured on your machine. Memory grows with context × parallel:",
        check,
    ]


def _modelfile_lines(facts: Facts) -> list[str]:
    name = printable(facts.name)
    new = f"{name}-{(facts.context_fix or 0) // 1024}k"
    state = facts.state
    on_host = state.where.kind != "docker" or state.where.binary_on_host
    box = shlex.quote(state.where.container or "CONTAINER")
    create = (
        f"ollama create {shlex.quote(new)} -f Modelfile"
        if on_host
        else f"docker cp Modelfile {box}:/tmp/Modelfile\n"
        f"        docker exec {box} ollama create "
        f"{shlex.quote(new)} -f /tmp/Modelfile"
    )
    return [
        "To fix it for this model, write a Modelfile, then create a new name and call it:",
        f"  FROM {name}",
        f"  PARAMETER num_ctx {facts.context_fix}",
        *(["  PARAMETER num_gpu N"] if _cpu_share(facts) else []),
        f"Then:   {create}",
    ]


def _cpu_share(facts: Facts) -> bool:
    share = None if _on_cpu(facts.state) else _vram_share(_loaded(facts.state, facts.name))
    return share is not None and share < 1


def header_lines(facts: Facts) -> list[str]:
    state = facts.state
    return [
        f"Ollama {state.version} is running at {state.shown}",
        f"Hardware: {facts.hardware}",
        f"Model:    {_model_line(facts)}",
        "",
        *([] if facts.context.value is None else [_context_line(facts)]),
        _parallel_line(facts),
        _keep_line(facts),
        "",
    ]


@dataclass(slots=True)
class Plan:
    """What Enter does for the chosen model, decided without writing anything."""

    kind: Literal["link", "copy", "sudo", "docker", "none"]
    refusal: str | None = None
    located: store.Located | None = None
    target: store.Target | None = None
    row: str = ""
    reused: bool = False
    rehash: bool = False
    notes: tuple[str, ...] = ()
    different: str | None = None
    size: int = 0


def store_root(state: State) -> tuple[Path | None, str | None, str | None]:
    """The store as this user sees it, and for a container the name and its path."""
    where, env = state.where, state.where.env
    if where.kind == "docker" and where.container:
        inside = where.container_root or store.CONTAINER_ROOT
        for destination, source in where.mounts:
            base = destination.rstrip("/")
            if inside == base or inside.startswith(base + "/"):
                return Path(source + inside[len(base) :]), where.container, inside
        return None, where.container, inside
    if env.get("OLLAMA_MODELS"):
        return Path(env["OLLAMA_MODELS"]), None, None
    if where.systemd_user == "ollama":
        return Path(store.STORE_ROOTS[0]), None, None
    return Path(store.STORE_ROOTS[1]).expanduser(), None, None


def plan_import(state: State, name: str, show: dict[str, Any]) -> Plan:
    """Whether and how this user can bring ``name`` into ``~/models``."""
    tag = _tag(state, name)
    label = printable(name)
    if state.where.kind == "docker" and state.where.container is None:
        return Plan("none", store.NOT_VISIBLE)
    if tag.get("remote_host") or tag.get("remote_model"):
        return Plan("none", f"{label} is a cloud model: its weights aren't on this machine.")
    details = tag.get("details")
    form = details.get("format") if isinstance(details, dict) else None
    if isinstance(form, str) and form != "gguf":
        return Plan(
            "none",
            f"{label} is stored as {printable(form)}, so llama.cpp can't load it. "
            "To tune it, download its .gguf from Hugging Face.",
        )
    if not state.local:
        return Plan(
            "none", "This Ollama runs on another machine. To copy a model, run tensward init there."
        )
    root, container, container_root = store_root(state)
    located = store.locate(
        name, show, root=root, container=container, container_root=container_root
    )
    if isinstance(located, str):
        return Plan("none", located)
    target = store.target_for(name, len(located.blobs))
    if target is None:
        return Plan("none", f"{label} has a name Tensward won't copy from. {store.NOT_VISIBLE}")
    plan = Plan("none", located=located, target=target, notes=located.notes)
    tagged = _count(tag.get("size")) or 0
    paths = [located.host_path(b) for b in located.blobs]
    real = [path for path in paths if path is not None]
    if not real:
        plan.kind, plan.size = "docker", tagged
    else:
        kinds = [
            store.source_kind(path, file) for path, file in zip(real, target.files, strict=True)
        ]
        if "unsafe" in kinds:
            plan.refusal = store.NOT_VISIBLE
            return plan
        if "unreadable" in kinds:
            if state.endpoint and not state.where.process:
                plan.refusal = store.NOT_VISIBLE
                return plan
            plan.kind = "sudo"
        else:
            if problem := store.gguf_problem(real[0], label):
                plan.refusal = problem
                return plan
            plan.kind = "link" if all(k == "link" for k in kinds) else "copy"
        plan.size = sum(os.stat(p).st_size for p in paths if p and os.path.exists(p)) or tagged
    states = [
        store.target_state(path, file) for path, file in zip(paths, target.files, strict=True)
    ]
    if all(s == "same" for s in states):
        plan.reused = True
        plan.row = _row("already", target, plan.size)
    elif all(s in ("same", "maybe") for s in states):
        plan.reused = plan.rehash = True
        plan.row = _row("maybe", target, plan.size)
    elif any(s in ("different", "symlink") for s in states):
        plan.different = target.first.name
        plan.row = _row("copy", target, plan.size)
    else:
        plan.row = _row(plan.kind, target, plan.size)
    return plan


def _row(kind: str, target: store.Target, size: int) -> str:
    tune = "tune it there"
    if kind == "link":
        return f"use this model for llama.cpp and {tune} (link, no extra disk)"
    if kind == "already":
        return f"use this model for llama.cpp and {tune} (already in ~/models)"
    if kind == "maybe":
        return f"use this model for llama.cpp and {tune} (in ~/models, checked when you continue)"
    return f"copy this model for llama.cpp and {tune} (needs {size / GIB:.1f} GiB free in ~/models)"


def _say(text: str = "") -> None:
    print(text, file=sys.stderr)


def import_commands(plan: Plan) -> list[str]:
    """Plain commands that do what Enter would, for a script or a user without a terminal."""
    if plan.located is None or plan.target is None or plan.kind == "none":
        return []
    kind = {"link": "ln", "copy": "cp", "sudo": "cp", "docker": "docker"}[plan.kind]
    return store.copy_chain(plan.located, plan.target, kind, sudo=plan.kind == "sudo")


def screen(facts: Facts, plan: Plan, *, terminal: bool) -> list[str]:
    """The settings check and its menu."""
    lines = [*header_lines(facts), *_prompt_lines(facts), ""]
    fix = _fix_lines(facts)
    if fix:
        lines += [*fix, ""]
    lines += [*plan.notes] if plan.notes else []
    rows: list[tuple[str, str]] = []
    if terminal:
        if plan.refusal:
            lines.append(plan.refusal)
        elif plan.different:
            lines.append(_different(plan))
        if plan.kind != "none" and not plan.refusal and not plan.different:
            rows.append(("Enter", plan.row))
        rows += [("n", "set up a different model"), ("q", "stop here; nothing was changed")]
    else:
        if plan.refusal:
            lines.append(plan.refusal)
        elif plan.different:
            lines.append(_different(plan))
        else:
            commands = [] if plan.reused else import_commands(plan)
            if plan.reused and plan.target is not None:
                rows.append(
                    (
                        f"tensward init --model {shlex.quote(str(plan.target.first))}",
                        "tune it with llama.cpp",
                    )
                )
            elif commands and plan.target is not None:
                rows += [(c, "bring the model into ~/models") for c in commands]
                init = f"tensward init --model {shlex.quote(str(plan.target.first))}"
                rows.append((init, "tune it with llama.cpp"))
        if not rows:
            rows.append(("tensward init --model FILE.gguf", "tune a GGUF file with llama.cpp"))
    lines += next_lines(rows)
    lines += ["", f"{UNMEASURABLE} on this file."]
    return lines


def _different(plan: Plan) -> str:
    assert plan.target is not None
    return different_text(plan.target.first)


def different_text(file: Path) -> str:
    return (
        f"{store.shown_path(file)} already exists and isn't this model. Move it away and run "
        f"tensward init again, or tune it with tensward init --model {shlex.quote(str(file))}."
    )


def document(facts: Facts, plan: Plan, message: str) -> dict[str, Any]:
    """The JSON line for a refusal with Ollama's settings."""
    prompts = facts.prompts
    commands = [] if plan.reused else import_commands(plan)
    return {
        "code": PROJECT_CONFIG_UNSUPPORTED,
        "message": message,
        "ollama": {
            "version": facts.state.version,
            "url": facts.state.shown,
            "model": facts.name,
            "context": {"value": facts.context.value, "source": facts.context.source},
            "parallel": None if facts.parallel_default else facts.parallel,
            "keep_alive": facts.keep,
            "prompts": None
            if prompts is None
            else {
                "over": prompts.over,
                "of": prompts.of,
                "longest_est": prompts.longest,
                "cut_style": facts.style,
            },
            "fixes": [f"{k}={v}" for k, v in facts.fixes],
            "import": {
                "commands": commands,
                "init": (
                    f"tensward init --model {shlex.quote(str(plan.target.first))}"
                    if plan.target
                    else None
                ),
            },
        },
    }


@dataclass(slots=True)
class Step:
    """What ``init`` does next: stop with ``code``, or go on (``reset`` after Ollama's
    evidence must not reach the project)."""

    code: int | None = None
    reset: bool = False


def _interactive(arguments: argparse.Namespace) -> bool:
    quiet = bool(arguments.json) or JSON_OUTPUT.get()
    return (
        sys.stdin.isatty()
        and sys.stderr.isatty()
        and not quiet
        and not arguments.yes
        and arguments.config is None
    )


def _named(facts: EndpointFacts | None, command: str | None) -> bool:
    found = list(facts.evidence) if facts else []
    found += command_evidence(command) if command else []
    return any(item.lookalike == "Ollama" for item in found)


def _refuse(text: str) -> Step:
    """The sentence that stops a named Ollama setup, as one JSON line under ``--json``."""
    if JSON_OUTPUT.get():
        document = {"code": PROJECT_CONFIG_UNSUPPORTED, "message": text, "ollama": None}
        print(json.dumps(document), file=sys.stderr)
    else:
        _say(text)
    return Step(code=exit_code_for(PROJECT_CONFIG_UNSUPPORTED))


def _unreachable(state: State) -> str:
    where = state.where
    if where.process:
        return (
            f"Ollama is running but didn't answer at {state.shown}. If it listens elsewhere, "
            "pass tensward init --endpoint http://HOST:PORT."
        )
    if where.installed:
        return (
            "Ollama is installed but not running. Tensward doesn't start or restart it. "
            "Start it with sudo systemctl start ollama (or ollama serve), then run "
            "tensward init again."
        )
    return (
        f"You named Ollama, but nothing answers at {state.shown} and it isn't installed here. "
        "Check the address, or run tensward init on the machine that runs it."
    )


def _is_path(text: str) -> bool:
    return os.path.exists(Path(text).expanduser())


def _drop_ollama(arguments: argparse.Namespace, *, flagged: bool) -> None:
    """Carry on with today's setup: what the user named for Ollama (``--endpoint``,
    ``--current``) and an Ollama model name do not reach it; any other command stays."""
    if flagged:
        arguments.endpoint = None
        arguments.current = None
        arguments.current_file = None
    if flagged or (arguments.model is not None and not _is_path(str(arguments.model))):
        arguments.model = None


def init_step(
    arguments: argparse.Namespace, facts: EndpointFacts | None, command: str | None
) -> Step:
    """The Ollama part of ``init``: runs before the wizard. It stops with the screen and exit
    code 3 (nothing was registered), or lets ``init`` go on, after an import with the model set
    to the copied file."""
    flagged = _named(facts, command)
    model_arg = None if arguments.model is None else str(arguments.model)
    if not flagged and (
        arguments.config is not None
        or command is not None
        or (model_arg is not None and _is_path(model_arg))
    ):
        return Step()
    hardware = detect_hardware()
    if hardware.kind not in ("nvidia", "cpu"):
        return Step()
    if not flagged and model_arg is None and running_servers():
        return Step()
    state = read_state(arguments, command, facts)
    state.hardware = hardware
    named = flagged
    if not named and model_arg is not None:
        if not state.answers or model_arg not in _names(state):
            return Step()
        named = True
    bare = not named
    if bare and not (state.answers or state.where.process or state.where.installed):
        return Step()
    if not state.answers:
        if named:
            return _refuse(_unreachable(state))
        _say(STOPPED_NOTE)
        return Step()
    if not state.tags:
        if named:
            return _refuse(
                "Could not list your models."
                if state.tags is None
                else "Ollama is running but has no models yet. Pull one (ollama pull ...), "
                "then run tensward init again."
            )
        _say(NO_MODELS_NOTE + " Carrying on with a normal setup.")
        return Step()
    terminal = _interactive(arguments)
    if bare and not terminal:
        arguments.ollama_hint = SCRIPTED_HINT
        return Step()
    return _flow(arguments, state, flagged=flagged, terminal=terminal)


def _pick_model(
    arguments: argparse.Namespace, state: State, ask: Asker | None
) -> str | None | Literal[""]:
    """The model to look at: ``--model NAME``, the only loaded one, the user's pick (``""`` for
    a model that is not Ollama's), None to list them without one."""
    names = _names(state)
    given = None if arguments.model is None else str(arguments.model)
    if given in names:
        return given
    if given is not None:
        _say(
            "You named Ollama, so --model is not used. To tune that file, run tensward init "
            f"--model {shlex.quote(given)} without --endpoint/--current."
        )
    loaded = [str(m.get("name")) for m in state.loaded if m.get("name") in names]
    if len(loaded) == 1:
        return loaded[0]
    if len(names) == 1:
        return names[0]
    if ask is not None:
        options = [*(printable(n) for n in names), NOT_NAMED]
        number = ask.choice("Which Ollama model?", options, 1)
        return "" if number == len(options) else names[number - 1]
    return loaded[0] if loaded else None


def _questions(arguments: argparse.Namespace, ask: Asker) -> None:
    """Prompts and peak requests, asked once so import doesn't ask again."""
    from .wizard import _prompts_problem

    if not arguments.prompts and not arguments.workload:
        key = ask.key(
            "Check your own prompts against Ollama's context? [Enter = skip, p = give a file]",
            {"p": "give a file"},
        )
        if key == "p":
            arguments.prompts = ask.path("Your prompts file (JSONL):", _prompts_problem)
    if arguments.concurrency is None:
        arguments.concurrency = ask.number("Peak concurrent requests?", 1)


def _flow(arguments: argparse.Namespace, state: State, *, flagged: bool, terminal: bool) -> Step:
    from . import wizard

    ask = wizard.Asker() if terminal else None
    name = _pick_model(arguments, state, ask)
    if name == "":
        _drop_ollama(arguments, flagged=flagged)
        return Step(reset=True)
    if ask is not None:
        _questions(arguments, ask)
    records = prompt_records(arguments)
    if name is None:
        return _list_models(state)
    show = show_model(state, name)
    facts = build_facts(state, name, show, records, arguments.concurrency)
    plan = plan_import(state, name, show)
    if ask is None:
        lines = screen(facts, plan, terminal=False)
        if JSON_OUTPUT.get():
            message = f"{UNMEASURABLE} on this file."
            print(json.dumps(document(facts, plan, message)), file=sys.stderr)
        else:
            for line in lines:
                _say(line)
        return Step(code=exit_code_for(PROJECT_CONFIG_UNSUPPORTED))
    for line in screen(facts, plan, terminal=True):
        _say(line)
    return _choose(arguments, ask, facts, plan, flagged=flagged)


def _list_models(state: State) -> Step:
    if JSON_OUTPUT.get():
        return _refuse("Pick an Ollama model with --model NAME.")
    _say(f"Ollama {state.version} is running at {state.shown}")
    _say("Model:    pick one with --model NAME")
    for name in _names(state):
        _say(f"  {printable(name)}")
    _say("")
    for line in next_lines([("tensward init --model NAME", "check that model's settings")]):
        _say(line)
    return Step(code=exit_code_for(PROJECT_CONFIG_UNSUPPORTED))


def _choose(
    arguments: argparse.Namespace, ask: Asker, facts: Facts, plan: Plan, *, flagged: bool
) -> Step:
    keys = {"n": "set up a different model", "q": "stop here"}
    while True:
        key = ask.key("[Enter / n / q]", keys)
        if key == "q":
            return Step(code=0)
        if key == "n":
            _drop_ollama(arguments, flagged=flagged)
            return Step(reset=True)
        if plan.kind == "none" or plan.refusal or plan.different:
            ask.write("Type n or q.")
            continue
        return _import(arguments, ask, facts, plan)


def _import(arguments: argparse.Namespace, ask: Asker, facts: Facts, plan: Plan) -> Step:
    assert plan.located is not None and plan.target is not None
    if not any(found.name == "llamacpp" for found in detect_engines()):
        _say(
            "llama.cpp isn't installed yet. See 'Don't have an engine yet?' in the README, "
            "then run tensward init again."
        )
        return Step(code=exit_code_for(ENGINE_UNAVAILABLE))
    located, target = plan.located, plan.target
    if plan.kind in ("link", "copy") and not plan.reused:
        if plan.kind == "copy" and (problem := store.free_problem(plan.size, target)):
            _say(problem)
            _say("")
            for line in next_lines([("q", "stop here")]):
                _say(line)
            return Step(code=exit_code_for(PROJECT_INPUTS_INVALID))
        how = _place(located, target, plan, ask)
    elif plan.reused:
        how = "reused"
    else:
        waited = _wait_for_user(ask, located, target, plan)
        if waited is None:
            return Step(code=0)
        how = waited
    broken = _verify(ask, located, target, printable(facts.name), rehash=plan.rehash)
    if broken:
        _say(broken)
        return Step(code=exit_code_for(PROJECT_INPUTS_INVALID))
    _told(how, target, facts, plan)
    arguments.model = target.first
    arguments.engine = "llamacpp"
    arguments.endpoint = None
    arguments.current = None
    arguments.current_file = None
    arguments.engine_defaults = True
    arguments.ollama_told = True
    return Step(reset=True)


def _place(located: store.Located, target: store.Target, plan: Plan, ask: Asker) -> str:
    how = "linked"
    for blob, file in zip(located.blobs, target.files, strict=True):
        path = located.host_path(blob)
        assert path is not None
        state = store.target_state(path, file)
        if state == "same":
            continue
        if state != "absent":
            raise PreflightError(PROJECT_INPUTS_INVALID, different_text(file))
        if plan.kind == "copy":
            ask.write(f"Copying {store.shown_path(file)} ({plan.size / GIB:.1f} GiB)...")
        try:
            done = store.link_or_copy(path, file, link=plan.kind == "link")
        except FileExistsError:
            raise PreflightError(PROJECT_INPUTS_INVALID, different_text(file)) from None
        if done == "copied":
            how = "copied"
    return how


def _wait_for_user(
    ask: Asker, located: store.Located, target: store.Target, plan: Plan
) -> str | None:
    """The command for the user's other terminal, then the check on Enter."""
    commands = store.copy_chain(
        located, target, "docker" if plan.kind == "docker" else "cp", sudo=plan.kind == "sudo"
    )
    ask.write(
        "Tensward can't read this file itself. Run this in another terminal, then press Enter here:"
    )
    for command in commands:
        ask.write(f"  {command}")
    ask.write("")
    ask.write("Tensward will check the chat template after the copy.")
    for line in next_lines([("Enter", "I ran it; check the file"), ("q", "stop here")]):
        ask.write(line)
    while True:
        if ask.key("[Enter / q]", {"q": "stop here"}) == "q":
            return None
        problem = None
        for blob, file in zip(located.blobs, target.files, strict=True):
            try:
                info = os.lstat(file)
            except OSError:
                info = None
            if info is None or not stat.S_ISREG(info.st_mode):
                problem = (
                    "That file isn't there yet. If the copy is still running, wait and press Enter."
                )
                break
            ask.write(store.check_minutes(info.st_size))
            if not store.matches_digest(file, blob):
                problem = (
                    "That file doesn't match your Ollama model. Delete it and run the "
                    "command again."
                )
                break
        if problem is None:
            return "copied"
        ask.write(problem)


def _verify(
    ask: Asker, located: store.Located, target: store.Target, label: str, *, rehash: bool
) -> str | None:
    for blob, file in zip(located.blobs, target.files, strict=True):
        if rehash:
            ask.write(store.check_minutes(os.stat(file).st_size))
            if not store.matches_digest(file, blob):
                return different_text(file)
    return store.gguf_problem(target.first, label)


def _told(how: str, target: store.Target, facts: Facts, plan: Plan) -> None:
    shown = store.shown_path(target.first)
    for note in plan.notes:
        _say(note)
    if how == "linked":
        _say(
            f"Linked {shown} to your Ollama file (no extra disk). "
            "Removing it from Ollama doesn't remove this."
        )
    elif how == "copied":
        _say(
            f"Copied {shown} from your Ollama file ({plan.size / GIB:.1f} GiB). "
            "Removing it from Ollama doesn't remove this."
        )
    else:
        _say(f"Using {shown}, already your Ollama file.")
    if facts.state.loaded:
        stops = "; ".join(stop_command(facts.state, str(m.get("name"))) for m in facts.state.loaded)
        held = "memory" if _on_cpu(facts.state) else "GPU"
        _say(
            f"Free the {held} first; a second server can't fit beside a loaded Ollama model: "
            f"{stops}"
        )
    _say("Tensward compares your changes against llama.cpp on this file, not against your Ollama.")
    _say("")
