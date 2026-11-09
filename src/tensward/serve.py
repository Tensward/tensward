"""``tensward serve``: a long-lived OpenAI-compatible endpoint for one chosen configuration.

The engine already speaks the OpenAI API; this module only launches it detached, remembers exactly
what it started in ``<project>/serve/<name>/state.json`` and stops exactly that. The API key
lives in ``<project>/serve/<name>/api_key`` (mode 0600) and is never printed.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import re
import shlex
import time
from pathlib import Path
from typing import Any, Literal, Sequence

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from .engines import ENGINES, Engine
from .engines.llamacpp.command import use_build
from .engines.vllm.quantization import require_servable_gptq
from .environment import require_available
from .files import write_json, write_private
from .gpu_busy import needs_for, refuse_if_busy
from .kept import ACCEPT_FLAG, Kept, read_kept
from .packages_file import PACKAGES_FILE, PackagesFile
from .platforms import darwin
from .probes import run_probe
from .progress import Note, emit
from .project import ResolvedProject, apply_engine_args, load_project, settings_for
from .report.build import overrides_suffix
from .runs import served_settings
from .runtime import (
    LocalProcessRuntime,
    Runtime,
    RuntimeFailure,
    ServeSpec,
    container_running,
    generate_api_key,
    out_of_memory,
    process_start_time,
    stop_container,
    stop_local_process,
    wait_until_ready,
)
from .settings import Settings

NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}")
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
HEALTH_TIMEOUT_S = 5.0


def serve_dir(project: Path, name: str) -> Path:
    if not NAME_PATTERN.fullmatch(name):
        raise RuntimeFailure("a server name is letters, digits, '.', '_' and '-'")
    return Path(project).expanduser() / "serve" / name


def resolve_settings(
    resolved: ResolvedProject,
    project: Path,
    engine: Engine,
    source: str | None,
    package: str | None,
    *,
    accept_unreviewed: bool,
) -> tuple[Settings, str, list[str]]:
    """The settings to serve, the source id and any warnings; refuses unreviewed changes.

    ``source`` other than ``current``, ``package`` and ``accept_unreviewed`` serve extensions:
    they pick among results written by an extension that searches settings.

    ``current`` is the kept setup when ``tensward set`` kept one, else the registered setup. An
    optimize result is a menu (``packages.json``); ``package`` picks one, by default the
    recommended one. Without a ``source``: the latest optimize result if the project has one,
    otherwise ``current``.
    """
    optimize_dir = Path(project).expanduser() / "optimize"
    kept = read_kept(Path(project).expanduser())
    if source is None:
        source = "latest" if any(optimize_dir.glob(f"*/{PACKAGES_FILE}")) else "current"
        if source == "latest" and kept is not None:
            emit(Note(text=f"a kept setup exists (run {kept.run}): serve it with --from current"))
    if source == "current":
        if kept is None:
            return settings_for(resolved, engine, ()), "current", []
        settings = settings_for(resolved, engine, kept.engine_args)
        return settings, f"current (kept from run {kept.run})", []
    if source == "latest":
        candidates = sorted(optimize_dir.glob(f"*/{PACKAGES_FILE}"))
        if not candidates:
            raise RuntimeFailure("no optimize result with a packages.json was found; run optimize")
        packages_path = candidates[-1]
    else:
        if not NAME_PATTERN.fullmatch(source):
            raise RuntimeFailure(f"--from {source!r} is not 'current', 'latest' or a result id")
        packages_path = optimize_dir / source / PACKAGES_FILE
    try:
        data = packages_path.read_bytes()
        document = json.loads(data)
    except (OSError, ValueError):
        raise RuntimeFailure(f"no readable optimize result {packages_path.parent.name!r}") from None
    try:
        if isinstance(document, dict) and "schema_version" not in document:
            menu = _released_optimizer_menu(document)
        else:
            menu = PackagesFile.model_validate_json(data)
    except (ValidationError, KeyError, TypeError):
        raise RuntimeFailure(
            f"optimize result {packages_path.parent.name!r} has a malformed packages.json"
        ) from None
    return _package_settings(menu, package, packages_path, accept_unreviewed)


def _released_optimizer_menu(document: dict[str, Any]) -> PackagesFile:
    """The menu of a packages.json written by an optimizer before this format, read in memory."""
    keys = ("name", "settings", "requires_review", "confirmed")
    menu = {
        "schema_version": "1",
        "recommended": document.get("recommended"),
        "packages": [{key: entry[key] for key in keys} for entry in document["packages"]],
    }
    return PackagesFile.model_validate_json(json.dumps(menu))


def _package_settings(
    menu: PackagesFile, package: str | None, packages_path: Path, accept_unreviewed: bool
) -> tuple[Settings, str, list[str]]:
    name = package or menu.recommended
    if name is None:
        raise RuntimeFailure("no package beat your current setup; serve it with --from current")
    entry = next((p for p in menu.packages if p.name == name), None)
    if entry is None:
        offered = ", ".join(p.name for p in menu.packages) or "none"
        raise RuntimeFailure(
            f"this result has no package {name!r} (it has: {offered}); "
            "a point whose settings did not beat your current setup has no package"
        )
    warnings = []
    if entry.requires_review and not accept_unreviewed:
        raise RuntimeFailure(
            "this package includes changes that can affect output quality; review "
            f"{packages_path.parent / 'summary.md'} and pass {ACCEPT_FLAG} to serve it"
        )
    if not entry.confirmed:
        warnings.append("this package was not confirmed against your current setup")
    return Settings.from_json(entry.settings), f"{packages_path.parent.name}:{name}", warnings


def served_argv(
    project: Path, engine: Engine, kept: Kept | None, *, host: str = "0.0.0.0", port: int = 8000
) -> list[str]:
    """The native engine command ``serve start`` would run for the kept setup (the registered one
    when nothing is kept), without launching it: the user's own program when the project knows
    it, else the engine's."""
    project = Path(project).expanduser()
    resolved = load_project(project)
    settings = settings_for(resolved, engine, kept.engine_args if kept is not None else ())
    settings = served_settings(project, resolved, engine, settings)
    return _without_measurement_pins(native_argv(resolved, engine, settings, host=host, port=port))


def _without_measurement_pins(argv: list[str]) -> list[str]:
    """``argv`` without the pins that exist for measuring: the all-interfaces host and the
    ``tensward-default`` model name."""
    pins = {
        ("--host", "0.0.0.0"),
        ("--served-model-name", "tensward-default"),
        ("-a", "tensward-default"),
    }
    kept: list[str] = []
    index = 0
    while index < len(argv):
        if index + 1 < len(argv) and (argv[index], argv[index + 1]) in pins:
            index += 2
            continue
        kept.append(argv[index])
        index += 1
    return kept


def native_argv(
    resolved: ResolvedProject, engine: Engine, settings: Settings, *, host: str, port: int
) -> list[str]:
    """The engine command for ``settings``, without the API key."""
    current = resolved.record.current_setup
    command = (
        [current.executable, *engine.local_command[1:]]
        if current.executable
        else list(engine.local_command)
    )
    runtime = LocalProcessRuntime(command)
    if engine.name == "llamacpp":
        use_build(shlex.join(runtime.command))
    spec = ServeSpec(
        engine=engine,
        model_dir=resolved.artifact.path,
        served_model_name="tensward-default",
        settings=settings,
        port=port,
        api_key="",
        host=host,
        serving=True,
    )
    return runtime.build_argv(spec)


def _kept_env(project: Path, source_id: str) -> tuple[str, ...]:
    """The engine environment (NAME=VALUE) of the kept setup when it is what gets served."""
    kept = read_kept(Path(project).expanduser()) if source_id.startswith("current (kept") else None
    return kept.env if kept is not None else ()


def start_server(
    project: Path,
    *,
    name: str,
    engine: Engine,
    source: str | None,
    package: str | None,
    runtime: Runtime,
    host: str,
    port: int,
    accept_unreviewed: bool,
    ready_timeout_s: float,
    engine_args: Sequence[str] = (),
    verify_weights: bool = False,
    allow_busy: bool = False,
) -> dict[str, Any]:
    """Start (or replace) the named server, wait until it is ready and record its state.

    The state is written as "starting" as soon as the server exists, so `serve stop` finds it
    even if this command is interrupted; an interrupted or failed start stops exactly that server.
    """
    directory = serve_dir(project, name)
    resolved = load_project(project, verify_weights=verify_weights)
    available = require_available(engine, runtime)
    if engine.family == "vllm":
        require_servable_gptq(resolved.artifact.path, available.version)
    settings, source_id, warnings = resolve_settings(
        resolved, project, engine, source, package, accept_unreviewed=accept_unreviewed
    )
    settings = apply_engine_args(engine, settings, engine_args)
    if source_id.startswith("current"):
        settings = served_settings(Path(project).expanduser(), resolved, engine, settings)
    env = _kept_env(project, source_id)
    if env:  # the engine's environment, as analyse launches it (also in a container)
        held = dict(var.partition("=")[::2] for var in env)
        settings = dataclasses.replace(settings, extra_env={**settings.extra_env, **held})
    if not allow_busy:
        weight_bytes = resolved.artifact.metadata.weight_bytes
        refuse_if_busy(
            runtime.platform, runtime.gpus,
            needs_for(engine, settings, weight_bytes, runtime.platform, runtime.gpus),
            None, own_pids(project),
        )  # fmt: skip
    for warning in warnings:
        emit(Note(text=f"warning: {warning}"))
    if host not in LOOPBACK_HOSTS:
        emit(Note(text=f"warning: binding to {host} exposes this server beyond this machine"))
    model_dir = resolved.artifact.path
    old = read_state(project, name)
    old = old if old is not None and is_alive(old) else None
    if old is not None and old["port"] == port:
        text = (
            f"replacing the running server on port {port}: stopping it before starting the new one"
        )
        emit(Note(text=text))
        stop_recorded(old)
        old = None
    api_key = generate_api_key()
    spec = ServeSpec(
        engine=engine,
        model_dir=model_dir,
        served_model_name=f"tensward-{name}",
        settings=settings,
        port=port,
        api_key=api_key,
        host=host,
        serving=True,
    )
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    overrides = overrides_suffix(engine_args)
    emit(Note(text=f"launching {engine.name} ({runtime.label}) for {source_id}{overrides}"))
    if (wait := _load_wait(project, resolved)) is not None:
        emit(Note(text=wait))
    server = runtime.start_persistent(spec, name=name, project=project, directory=directory)
    state: dict[str, Any] | None = None
    try:  # opened at once: from here every failure path stops exactly this server
        identity = server.identity()
        if identity.get("start_time") is None and darwin.on_mac():
            raise RuntimeFailure("cannot identify the server process")
        state = {
            "status": "starting",
            "runtime": identity["runtime"],
            "identity": identity,
            "endpoint": server.endpoint_url,
            "host": host,
            "port": port,
            "served_model_name": spec.served_model_name,
            "engine": engine.name,
            "settings": dataclasses.asdict(settings),
            "engine_args": list(engine_args),
            "engine_env": list(env),
            "engine_command": native_argv(resolved, engine, settings, host=host, port=port),
            "source": source_id,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_state(directory, state)
        asyncio.run(
            wait_until_ready(
                server,
                engine,
                api_key=api_key,
                served_model_name=spec.served_model_name,
                timeout_s=ready_timeout_s,
            )
        )
    except BaseException as error:  # includes Ctrl-C and SIGTERM (see cli)
        server.stop()
        if isinstance(error, RuntimeFailure) and (log := server.log_text()):
            path = directory / "server.log"
            said = str(error)
            if out_of_memory(said):  # one plain line; the traceback is in the log
                said = said.splitlines()[0]
                error = RuntimeFailure(said)
            with contextlib.suppress(OSError):
                write_private(path, log.replace(api_key, "***"))
                error = RuntimeFailure(f"{said}\nthe server's log: {path}")
        # The replaced server, if any, keeps running: its record goes back. If even the state
        # cannot be written, the original error is the one to report.
        record = old or (
            None
            if state is None
            else {**state, "status": "failed", "error": str(error) or type(error).__name__}
        )
        if record is not None:
            with contextlib.suppress(OSError):
                _write_state(directory, record)
        raise error
    write_private(directory / "api_key", api_key)
    state = {**state, "status": "running"}
    _write_state(directory, state)
    if old is not None:
        stop_recorded(old)
        emit(Note(text="stopped the previous server after the new one was ready"))
    return state


class ServeState(BaseModel):
    """The fields of ``state.json`` that serve reads back; any others pass through unchanged."""

    model_config = ConfigDict(extra="allow")

    status: str
    runtime: Literal["docker", "local"]
    identity: dict[str, Any]
    endpoint: str
    engine: str
    port: int
    source: str
    served_model_name: str
    started_at: str
    engine_args: list[str] = []

    @model_validator(mode="after")
    def _identity_fits_runtime(self) -> ServeState:
        keys = ("container_id",) if self.runtime == "docker" else ("pid", "start_time")
        if missing := [key for key in keys if key not in self.identity]:
            raise ValueError(f"identity lacks {', '.join(missing)}")
        return self


def read_state(project: Path, name: str) -> dict[str, Any] | None:
    """The recorded state of the named server; None if there is none. A state file that cannot
    be read as one is refused, naming the file."""
    path = serve_dir(project, name) / "state.json"
    try:
        text = path.read_text()
    except OSError:
        return None
    try:
        return ServeState.model_validate_json(text).model_dump()
    except ValidationError:
        raise RuntimeFailure(
            f"the server state {path} is unreadable; delete the file, and stop any server it "
            "described yourself"
        ) from None


def is_alive(state: dict[str, Any]) -> bool:
    """Whether exactly the recorded container or process is still running."""
    if state["status"] not in ("starting", "running"):
        return False
    identity = state["identity"]
    if state["runtime"] == "docker":
        return container_running(identity["container_id"])
    return bool(process_start_time(identity["pid"]) == identity["start_time"])


def any_running(project: Path) -> bool:
    """Whether a recorded server of the project is still running."""
    for path in (Path(project).expanduser() / "serve").glob("*/state.json"):
        try:
            state = read_state(project, path.parent.name)
        except RuntimeFailure:
            continue
        if state is not None and is_alive(state):
            return True
    return False


def own_pids(project: Path) -> frozenset[int]:
    """The host pids of the project's recorded servers that are still running."""
    found: set[int] = set()
    for path in (Path(project).expanduser() / "serve").glob("*/state.json"):
        try:
            state = read_state(project, path.parent.name)
        except RuntimeFailure:
            continue
        if state is None or not is_alive(state):
            continue
        if state["runtime"] == "local":
            found |= process_tree(state["identity"]["pid"])
        elif state["runtime"] == "docker":
            found.update(container_pids(state["identity"]["container_id"]))
    return frozenset(found)


def process_tree(pid: int) -> frozenset[int]:
    """``pid`` and every process below it, as ``ps`` lists them."""
    children: dict[int, list[int]] = {}
    for line in (run_probe(["ps", "-eo", "pid=,ppid="]) or "").splitlines():
        fields = line.split()
        if len(fields) == 2 and all(field.isdigit() for field in fields):
            children.setdefault(int(fields[1]), []).append(int(fields[0]))
    found = {pid}
    pending = [pid]
    while pending:
        for child in children.get(pending.pop(), ()):
            if child not in found:
                found.add(child)
                pending.append(child)
    return frozenset(found)


def container_pids(container_id: str) -> frozenset[int]:
    """The host pids of the processes in a running container; empty when docker does not say."""
    out = run_probe(["docker", "top", container_id, "-eo", "pid"])
    return frozenset(int(line) for line in (out or "").split() if line.isdigit())


def is_healthy(state: dict[str, Any]) -> bool:
    try:
        health = f"{state['endpoint']}{ENGINES[state['engine']].health_path}"
        return httpx.get(health, timeout=HEALTH_TIMEOUT_S).status_code == 200
    except httpx.HTTPError:
        return False


def stop_recorded(state: dict[str, Any]) -> None:
    identity = state["identity"]
    if state["runtime"] == "docker":
        stop_container(identity["container_id"])
    else:
        stop_local_process(identity["pid"], identity["start_time"])


def stop_server(project: Path, name: str) -> bool:
    """Stop the named server and mark it stopped; False if none was running."""
    state = read_state(project, name)
    if state is None:
        raise RuntimeFailure(f"no server named {name!r} in this project")
    was_alive = is_alive(state)
    if was_alive:
        stop_recorded(state)
    _write_state(serve_dir(project, name), {**state, "status": "stopped"})
    return was_alive


LOAD_ANNOUNCE_S = 10  # a load shorter than this needs no announcement


def _load_wait(project: Path, resolved: ResolvedProject) -> str | None:
    """How long loading the model usually takes, said before waiting for it."""
    from .history import last_load_s
    from .wizard import load_estimate_s

    seconds = last_load_s(project) or load_estimate_s(resolved.artifact.metadata.weight_bytes)
    if seconds < LOAD_ANNOUNCE_S:
        return None
    took = f"{round(seconds / 60)} min" if seconds >= 90 else f"{round(seconds)} s"
    return f"Loading the model takes about {took}; this command returns when the server answers."


def curl_example(state: dict[str, Any], key_file: Path) -> str:
    return (
        f"curl {state['endpoint']}/v1/chat/completions \\\n"
        f'  -H "Authorization: Bearer $(cat {shlex.quote(str(key_file))})" \\\n'
        '  -H "Content-Type: application/json" \\\n'
        f'  -d \'{{"model": "{state["served_model_name"]}", '
        '"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 16}\''
    )


def python_example(state: dict[str, Any], key_file: Path) -> str:
    """The same request with the OpenAI Python client, in two lines."""
    return (
        "from openai import OpenAI\n"
        f"print(OpenAI(base_url={state['endpoint'] + '/v1'!r}, "
        f"api_key=open({str(key_file)!r}).read().strip()).chat.completions.create("
        f"model={state['served_model_name']!r}, "
        "messages=[{'role': 'user', 'content': 'Hello'}]).choices[0].message.content)"
    )


def start_lines(
    state: dict[str, Any], key_file: Path, *, tuned: str | None = None, verbose: bool = False
) -> list[str]:
    """What ``serve start`` prints once the server answers."""
    source = state["source"]
    if source.startswith("current (kept"):
        what = (
            f"your tuned setup ({tuned} on top of your original command)"
            if tuned
            else ("your tuned setup")
        )
    elif source == "current":
        what = "your current setup"
    else:
        what = source
    what += overrides_suffix(state["engine_args"])
    if state.get("engine_env") and not tuned:
        what += f" with {' '.join(state['engine_env'])}"
    lines = [
        f"Serving {what}.",
        f"endpoint: {state['endpoint']}/v1",
        f"api key file: {key_file}",
        "",
        curl_example(state, key_file),
        "",
        python_example(state, key_file),
        "",
        "The server keeps running in the background after this command returns.",
        "Stop it from any terminal: tensward serve stop",
        "Run it as a service: tensward export --format compose (or systemd, helm)",
    ]
    if verbose:
        lines += ["", f"engine command: {shlex.join(state['engine_command'])}"]
    return lines


def _write_state(directory: Path, state: dict[str, Any]) -> None:
    write_json(directory / "state.json", state)
