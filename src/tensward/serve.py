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
import secrets
import shlex
import time
from pathlib import Path
from typing import Any, Literal, Sequence

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from .analyse import apply_engine_args, settings_for
from .engines import ENGINES, Engine
from .engines.protocol import Settings
from .environment import require_available
from .files import write_json, write_private
from .progress import say
from .project import ResolvedProject, load_project
from .report import overrides_suffix
from .runtime import (
    DockerRuntime,
    LocalProcessRuntime,
    PersistentServe,
    RunningServer,
    Runtime,
    RuntimeFailure,
    ServeSpec,
    container_running,
    generate_api_key,
    process_start_time,
    stop_container,
    stop_local_process,
    wait_until_ready,
)

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

    An optimize result is a menu (``packages.json``); ``package`` picks one, by default the
    recommended one. Without a ``source``: the latest optimize result if the project has one,
    otherwise the current setup.
    """
    optimize_dir = Path(project).expanduser() / "optimize"
    if source is None:
        source = "latest" if any(optimize_dir.glob("*/packages.json")) else "current"
    if source == "current":
        return settings_for(resolved, engine, ()), "current", []
    if source == "latest":
        candidates = sorted(optimize_dir.glob("*/packages.json"))
        if not candidates:
            raise RuntimeFailure("no optimize result with a packages.json was found; run optimize")
        packages_path = candidates[-1]
    else:
        if not NAME_PATTERN.fullmatch(source):
            raise RuntimeFailure(f"--from {source!r} is not 'current', 'latest' or a result id")
        packages_path = optimize_dir / source / "packages.json"
    try:
        menu = json.loads(packages_path.read_text("utf-8"))
    except (OSError, ValueError):
        raise RuntimeFailure(f"no readable optimize result {packages_path.parent.name!r}") from None
    try:
        return _package_settings(menu, package, packages_path, accept_unreviewed)
    except (KeyError, TypeError, AttributeError):
        raise RuntimeFailure(
            f"optimize result {packages_path.parent.name!r} has a malformed packages.json"
        ) from None


def _package_settings(
    menu: dict[str, Any], package: str | None, packages_path: Path, accept_unreviewed: bool
) -> tuple[Settings, str, list[str]]:
    name = package or menu.get("recommended")
    if name is None:
        raise RuntimeFailure("no package beat your current setup; serve it with --from current")
    entry = next((p for p in menu["packages"] if p["name"] == name), None)
    if entry is None:
        offered = ", ".join(p["name"] for p in menu["packages"]) or "none"
        raise RuntimeFailure(
            f"this result has no package {name!r} (it has: {offered}); "
            "a point whose settings did not beat your current setup has no package"
        )
    warnings = []
    if entry["requires_review"] and not accept_unreviewed:
        raise RuntimeFailure(
            "this package includes changes that can affect output quality; review "
            f"{packages_path.parent / 'summary.md'} and pass --accept-unreviewed to serve it"
        )
    if not entry["confirmed"]:
        warnings.append("this package was not confirmed against your current setup")
    return Settings.from_json(entry["settings"]), f"{packages_path.parent.name}:{name}", warnings


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
) -> dict[str, Any]:
    """Start (or replace) the named server, wait until it is ready and record its state.

    The state is written as "starting" as soon as the server exists, so `serve stop` finds it
    even if this command is interrupted; an interrupted or failed start stops exactly that server.
    """
    directory = serve_dir(project, name)
    resolved = load_project(project, verify_weights=verify_weights)
    require_available(engine, runtime)
    settings, source_id, warnings = resolve_settings(
        resolved, project, engine, source, package, accept_unreviewed=accept_unreviewed
    )
    settings = apply_engine_args(engine, settings, engine_args)
    for warning in warnings:
        say(f"warning: {warning}")
    if host not in LOOPBACK_HOSTS:
        say(f"warning: binding to {host} exposes this server beyond this machine")
    model_dir = resolved.artifact.path
    old = read_state(project, name)
    old = old if old is not None and is_alive(old) else None
    if old is not None and old["port"] == port:
        say(f"replacing the running server on port {port}: stopping it before starting the new one")
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
    )
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    say(f"launching {engine.name} ({runtime.label}) for {source_id}{overrides_suffix(engine_args)}")
    server = _launch(runtime, spec, project, name, directory)
    state: dict[str, Any] | None = None
    try:  # opened at once: from here every failure path stops exactly this server
        identity = server.identity()
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
        raise
    write_private(directory / "api_key", api_key)
    state = {**state, "status": "running"}
    _write_state(directory, state)
    if old is not None:
        stop_recorded(old)
        say("stopped the previous server after the new one was ready")
    return state


def _launch(
    runtime: Runtime, spec: ServeSpec, project: Path, name: str, directory: Path
) -> RunningServer:
    if isinstance(runtime, DockerRuntime):
        persistent = PersistentServe(name, Path(project).expanduser().resolve())
        return runtime.start_persistent(spec, persistent)
    if isinstance(runtime, LocalProcessRuntime):
        return runtime.start_persistent(spec, directory / f"logs-{secrets.token_hex(3)}")
    raise RuntimeFailure("this runtime cannot start a persistent server")


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


def curl_example(state: dict[str, Any], key_file: Path) -> str:
    return (
        f"curl {state['endpoint']}/v1/completions \\\n"
        f'  -H "Authorization: Bearer $(cat {shlex.quote(str(key_file))})" \\\n'
        '  -H "Content-Type: application/json" \\\n'
        f'  -d \'{{"model": "{state["served_model_name"]}", '
        '"prompt": "Hello", "max_tokens": 16}\''
    )


def _write_state(directory: Path, state: dict[str, Any]) -> None:
    write_json(directory / "state.json", state)
