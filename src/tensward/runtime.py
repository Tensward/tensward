"""How an engine server is started, awaited and stopped for one Tensward run.

A :class:`Runtime` turns a :class:`ServeSpec` into a :class:`RunningServer`. Two runtimes
exist: :class:`DockerRuntime` (the customer path: the ``docker`` CLI, an image that must
already be present) and :class:`LocalProcessRuntime` (a process in this container, for GPU
tests where no Docker daemon exists and for the fake server in tests). Neither uses a shell,
and the API key travels only in the environment, never on an argument vector.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence, runtime_checkable

import httpx

from .engines import Engine
from .engines.protocol import Settings
from .platforms import PLATFORMS, Platform
from .progress import say

LOOPBACK_HOST = "127.0.0.1"
WILDCARD_TO_LOOPBACK = {"0.0.0.0": LOOPBACK_HOST, "::": "::1"}
CONTAINER_PORT = 8000
CONTAINER_MODEL_PATH = "/model"
CONTAINER_TRACE_PATH = "/traces"
DEFAULT_READY_TIMEOUT_S = 900.0  # seconds to wait for a model to load
LOG_TAIL_LINES = 40
ERROR_LINE = re.compile(r"\w*(?:Error|Exception): ")
POINTS_ELSEWHERE = re.compile(r"see root cause|see above|initialization failed", re.IGNORECASE)
PROCESS_PREFIX = re.compile(r"^\(\w+ pid=\d+\)")  # vLLM prefixes each line with its process
# A pydantic summary ("1 validation error for ModelConfig") gives its detail on the next line.
VALIDATION_SUMMARY = re.compile(r"\d+ validation errors? for \w+$")
ERROR_LINE_CHARS = 300
MMAP_REFUSED = re.compile(r"unable to mmap \d+ bytes.*Cannot allocate memory", re.IGNORECASE)


def log_tail(text: str) -> str:
    """The last lines of a server log, for error messages."""
    return "\n".join(text.splitlines()[-LOG_TAIL_LINES:])


def exit_reason(log: str) -> str:
    """Why a server that exited before it was ready says it did: the first ``...Error: ...``
    line of its log that is not a wrapper pointing at an earlier one (else the last), then the
    end of the log."""
    lines = [PROCESS_PREFIX.sub("", line).strip() for line in log.splitlines()]
    errors = [index for index, line in enumerate(lines) if ERROR_LINE.search(line)]
    causes = [index for index in errors if not POINTS_ELSEWHERE.search(lines[index])]
    reason = ""
    if chosen := (causes or errors[-1:]):
        index = chosen[0]
        text = lines[index]
        if VALIDATION_SUMMARY.search(text) and index + 1 < len(lines):
            text = f"{text}: {lines[index + 1]}"
        reason = f": {text[:ERROR_LINE_CHARS]}"
    if MMAP_REFUSED.search(log):
        reason += (
            ". Likely a weight file larger than the host's RAM plus swap under "
            "vm.overcommit_memory=0: set `sudo sysctl vm.overcommit_memory=1`, add swap, use a "
            "host with more RAM, or re-shard the checkpoint into smaller files"
        )
    return f"the server exited before it was ready{reason}\n{log_tail(log)}"


STILL_LOADING_EVERY_S = 15.0
STOP_GRACE_S = 10.0
PROFILER_FLUSH_S = 180.0  # a launch wrapped by a profiler may need this long to write its report
DOCKER_TIMEOUT_S = 60.0


class RuntimeFailure(Exception):
    """A server could not be started, reached or stopped; the message is safe to print."""


@dataclass(frozen=True, slots=True)
class ServeSpec:
    """Everything needed to serve one model: an engine, a directory, a name, settings, a port
    and a key. Host, port, model and API key are owned by the runtime, not the settings.
    """

    engine: Engine
    model_dir: Path
    served_model_name: str
    settings: Settings
    port: int
    api_key: str = field(repr=False)
    host: str = LOOPBACK_HOST
    trace_dir: Path | None = None  # a traced launch: the engine's profiler writes here
    launch_prefix: tuple[str, ...] = ()  # wraps the server command, e.g. a profiler (local only)

    @property
    def bind_host(self) -> str:
        """The address to bind: an IP, since ``localhost`` is not one docker's ``-p`` accepts."""
        return LOOPBACK_HOST if self.host == "localhost" else self.host

    def public_dict(self) -> dict[str, Any]:
        """The spec as JSON, without the API key."""
        return {
            "engine": self.engine.name,
            "model_dir": str(self.model_dir),
            "served_model_name": self.served_model_name,
            "settings": dataclasses.asdict(self.settings),
            "host": self.host,
            "port": self.port,
        }


@dataclass(frozen=True, slots=True)
class PersistentServe:
    """The name and project of a long-lived server, recorded as docker labels."""

    name: str
    project: Path


@runtime_checkable
class RunningServer(Protocol):
    """One started server."""

    @property
    def endpoint_url(self) -> str: ...

    def is_running(self) -> bool: ...

    def stop(self) -> None:
        """Stop the server. Idempotent; the server is gone when this returns."""
        ...

    def log_text(self) -> str:
        """The server's whole log so far (startup lines such as the chosen kernels included)."""
        ...

    def identity(self) -> dict[str, Any]:
        """A JSON description of exactly what ran, for the run record."""
        ...


@runtime_checkable
class Runtime(Protocol):
    wraps_launch: bool  # whether ServeSpec.launch_prefix can wrap the server command
    inherits_environment: bool  # whether the engine starts with this process's environment
    label: str  # how the engine runs, for progress lines
    gpus: tuple[str, ...] | None  # device indices the engine may use; None leaves the choice open
    platform: Platform  # how the engine is pinned to ``gpus``
    kind: str  # "docker" or "local", as Engine.availability takes it
    target: str  # the image, or the server command

    def start(self, spec: ServeSpec) -> RunningServer: ...


@contextlib.contextmanager
def uninterruptible() -> Iterator[None]:
    """Ignore SIGINT and SIGTERM inside the block, so a second Ctrl-C or a SIGTERM cannot abort
    a cleanup half-way and leak a server. Usable as a decorator. Handlers are restored after."""
    if threading.current_thread() is not threading.main_thread():
        yield  # only the main thread has signal handlers
        return
    signals = (signal.SIGINT, signal.SIGTERM)
    previous = [signal.signal(sig, signal.SIG_IGN) for sig in signals]
    try:
        yield
    finally:
        for sig, handler in zip(signals, previous, strict=True):
            signal.signal(sig, handler)


def _bracketed(ip: str) -> str:
    return f"[{ip}]" if ":" in ip else ip


def _endpoint(spec: ServeSpec) -> str:
    """Where to reach the server: the bound address, or loopback when it binds every interface."""
    ip = WILDCARD_TO_LOOPBACK.get(spec.bind_host, spec.bind_host)
    return f"http://{_bracketed(ip)}:{spec.port}"


def generate_api_key() -> str:
    """A fresh private API key for one server."""
    return secrets.token_urlsafe(24)


def select_loopback_port() -> int:
    """A free port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK_HOST, 0))
        return int(probe.getsockname()[1])


# --------------------------------------------------------------------------------------
# Docker
# --------------------------------------------------------------------------------------


def _docker(
    arguments: Sequence[str], *, env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["docker", *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=DOCKER_TIMEOUT_S,
            env=None if env is None else dict(env),
        )
    except FileNotFoundError:
        raise RuntimeFailure("the docker CLI was not found on PATH") from None
    except subprocess.TimeoutExpired:
        raise RuntimeFailure(f"docker {arguments[0]} timed out") from None


class DockerRuntime:
    """Run an engine from a locally present image with the ``docker`` CLI."""

    wraps_launch = False  # a profiler would have to be inside the image
    inherits_environment = False  # only the -e variables reach the container
    kind = "docker"

    def __init__(
        self, image: str, gpus: tuple[str, ...] | None = None, platform: Platform = PLATFORMS[0]
    ) -> None:
        if not image or image.startswith("-"):
            raise RuntimeFailure(f"{image!r} is not a docker image reference")
        self.image = image
        self.target = image
        self.gpus = gpus
        self.platform = platform
        self.label = f"docker, image {image}"

    def build_run_argv(
        self, spec: ServeSpec, run_id: str, *, persistent: PersistentServe | None = None
    ) -> list[str]:
        """The exact ``docker run`` argument list. The API key is passed by name only.

        A ``persistent`` server also gets a restart policy and labels naming it and its project.
        """
        argv = [
            "docker",
            "run",
            "-d",
            "--pull",
            "never",
            "--name",
            f"tensward-{run_id}",
            "--label",
            f"tensward.run={run_id}",
            "--ipc=host",
        ]
        if persistent is not None:
            argv += [
                "--restart",
                "unless-stopped",
                "--label",
                f"tensward.serve={persistent.name}",
                "--label",
                f"tensward.project={persistent.project}",
            ]
        argv += self.platform.docker_device_args(self.gpus)
        if spec.trace_dir is not None:
            argv += ["-v", f"{spec.trace_dir}:{CONTAINER_TRACE_PATH}"]
            for name, value in spec.engine.trace_env.items():
                argv += ["-e", f"{name}={value}"]
        for name, value in spec.settings.extra_env.items():
            argv += ["-e", f"{name}={value}"]
        argv += [
            "-v",
            f"{spec.model_dir}:{CONTAINER_MODEL_PATH}:ro",
            "-p",
            f"{_bracketed(spec.bind_host)}:{spec.port}:{CONTAINER_PORT}",
            "-e",
            spec.engine.api_key_env,
            self.image,
            *spec.engine.launch_argv(
                spec.settings,
                model=CONTAINER_MODEL_PATH,
                served_model_name=spec.served_model_name,
                host="0.0.0.0",
                port=CONTAINER_PORT,
                trace_dir=None if spec.trace_dir is None else CONTAINER_TRACE_PATH,
            ),
        ]
        return argv

    def start(self, spec: ServeSpec) -> RunningServer:
        return self._run(spec, None)

    def start_persistent(self, spec: ServeSpec, persistent: PersistentServe) -> RunningServer:
        """Start a server that outlives this process; the caller stops it only on failure."""
        return self._run(spec, persistent)

    def _run(self, spec: ServeSpec, persistent: PersistentServe | None) -> RunningServer:
        inspected = _docker(["image", "inspect", "--format", "{{.Id}}", self.image])
        if inspected.returncode != 0:
            raise RuntimeFailure(
                f"docker image {self.image!r} is not present locally; Tensward never pulls "
                f"images implicitly, run `docker pull {self.image}` first"
            )
        image_id = inspected.stdout.strip()
        run_id = secrets.token_hex(6)
        argv = self.build_run_argv(spec, run_id, persistent=persistent)
        server = DockerServer(
            run_id=run_id,
            image=self.image,
            image_id=image_id,
            argv=argv,
            endpoint_url=_endpoint(spec),
        )
        try:
            completed = _docker(argv[1:], env={**os.environ, spec.engine.api_key_env: spec.api_key})
            if completed.returncode != 0:
                raise RuntimeFailure(f"docker run failed: {completed.stderr.strip()}")
            server.container_id = completed.stdout.strip()
        except BaseException:
            server.stop()
            raise
        return server


class DockerServer:
    def __init__(
        self, *, run_id: str, image: str, image_id: str, argv: list[str], endpoint_url: str
    ) -> None:
        self._name = f"tensward-{run_id}"
        self._image = image
        self._image_id = image_id
        self._argv = argv
        self._endpoint_url = endpoint_url
        self._final_log: str | None = None
        self.container_id: str | None = None

    @property
    def endpoint_url(self) -> str:
        return self._endpoint_url

    def is_running(self) -> bool:
        if self._final_log is not None:
            return False
        return container_running(self._name)

    def log_text(self) -> str:
        if self._final_log is not None:
            return self._final_log
        logs = _docker(["logs", self._name])
        return (logs.stdout + logs.stderr).strip()

    @uninterruptible()
    def stop(self) -> None:
        if self._final_log is not None:
            return
        try:
            self._final_log = self.log_text()
        except RuntimeFailure:
            self._final_log = ""
        stop_container(self._name)

    def identity(self) -> dict[str, Any]:
        return {
            "runtime": "docker",
            "image": self._image,
            "image_id": self._image_id,
            "container_name": self._name,
            "container_id": self.container_id,
            "argv": list(self._argv),
        }


# --------------------------------------------------------------------------------------
# Local process
# --------------------------------------------------------------------------------------


class LocalProcessRuntime:
    """Run ``<command> <engine arguments>`` as a local process."""

    wraps_launch = True
    inherits_environment = True
    kind = "local"

    def __init__(
        self,
        command: Sequence[str],
        gpus: tuple[str, ...] | None = None,
        platform: Platform = PLATFORMS[0],
    ) -> None:
        self.command = tuple(command)
        self.target = shlex.join(self.command)
        self.gpus = gpus
        self.platform = platform
        self.label = "local process"

    def build_argv(self, spec: ServeSpec) -> list[str]:
        return [
            *spec.launch_prefix,
            *self.command,
            *spec.engine.launch_argv(
                spec.settings,
                model=str(spec.model_dir),
                served_model_name=spec.served_model_name,
                host=spec.bind_host,
                port=spec.port,
                trace_dir=None if spec.trace_dir is None else str(spec.trace_dir),
            ),
            *(spec.engine.counters_args if spec.launch_prefix else ()),
        ]

    def start(self, spec: ServeSpec) -> RunningServer:
        return self._start(spec, Path(tempfile.mkdtemp(prefix="tensward-serve-")))

    def start_persistent(self, spec: ServeSpec, log_directory: Path) -> RunningServer:
        """Start a detached server that outlives this process, logging into ``log_directory``."""
        log_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        return self._start(spec, log_directory)

    def _start(self, spec: ServeSpec, log_directory: Path) -> RunningServer:
        argv = self.build_argv(spec)
        env = {**os.environ, spec.engine.api_key_env: spec.api_key}
        if os.sep in self.command[0]:  # as if the environment it lives in were activated
            home = os.path.dirname(os.path.abspath(self.command[0]))
            env["PATH"] = os.pathsep.join(filter(None, [home, env.get("PATH")]))
        if spec.trace_dir is not None:
            env.update(spec.engine.trace_env)
        env.update(spec.settings.extra_env)
        env.update(self.platform.select_env(self.gpus, env))
        server: LocalProcessServer | None = None
        try:
            with open(log_directory / "server.log", "wb") as log:
                process = subprocess.Popen(  # noqa: S603 - list argv, no shell
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
            server = LocalProcessServer(
                process, argv, _endpoint(spec), log_directory, wrapped=bool(spec.launch_prefix)
            )
            return server
        except BaseException:
            if server is not None:
                server.stop()
            else:
                shutil.rmtree(log_directory, ignore_errors=True)
            raise


class LocalProcessServer:
    def __init__(
        self,
        process: subprocess.Popen[bytes],
        argv: list[str],
        endpoint_url: str,
        log_directory: Path,
        *,
        wrapped: bool = False,
    ) -> None:
        self._wrapped = wrapped  # a profiler (ncu) is the process; the server is its child
        self._process = process
        self._argv = argv
        self._endpoint_url = endpoint_url
        self._log_directory = log_directory
        self._final_log: str | None = None

    @property
    def endpoint_url(self) -> str:
        return self._endpoint_url

    def is_running(self) -> bool:
        return self._final_log is None and self._process.poll() is None

    def log_text(self) -> str:
        if self._final_log is not None:
            return self._final_log
        try:
            return (self._log_directory / "server.log").read_text(errors="replace")
        except OSError:
            return ""

    @uninterruptible()
    def stop(self) -> None:
        if self._final_log is not None:
            return
        self._final_log = self.log_text()
        group = self._process.pid
        if self._wrapped:
            self._let_profiler_flush()
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self._process.wait(timeout=STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            pass
        # Always finish with SIGKILL of the whole group so no worker outlives the server.
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self._process.wait()
        shutil.rmtree(self._log_directory, ignore_errors=True)

    def _let_profiler_flush(self) -> None:
        """Ask the wrapped server, not the profiler, to exit and wait for the profiler to write
        its report (ncu writes only when it exits, and is already gone once its launch count was
        reached). Signalling the whole group would kill the profiler before it writes."""
        pid = self._process.pid
        try:
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        except OSError:
            children = []
        for child in children:
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(child), signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._process.wait(timeout=PROFILER_FLUSH_S)

    def identity(self) -> dict[str, Any]:
        return {
            "runtime": "local",
            "argv": list(self._argv),
            "pid": self._process.pid,
            "start_time": process_start_time(self._process.pid),
        }


def process_start_time(pid: int) -> str | None:
    """The kernel start time of ``pid`` (clock ticks after boot), or None if it is gone.

    A zombie counts as gone. The pair (pid, start time) identifies one process for good, so a
    recycled pid is never mistaken for the server that used to have it.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # The command name is parenthesised and may contain spaces; fields follow the last ")".
    fields = stat.rsplit(")", 1)[1].split()
    return None if fields[0] == "Z" else fields[19]


@uninterruptible()
def stop_local_process(pid: int, start_time: str) -> None:
    """Stop the detached server ``pid`` started at ``start_time``; do nothing if that is gone."""
    if process_start_time(pid) != start_time:
        return
    for sig, wait_s in ((signal.SIGTERM, STOP_GRACE_S), (signal.SIGKILL, STOP_GRACE_S)):
        try:
            os.killpg(pid, sig)  # a detached server leads its own session and group
        except ProcessLookupError:
            return
        deadline = time.monotonic() + wait_s
        while process_start_time(pid) == start_time:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        else:
            return
    raise RuntimeFailure(f"process {pid} did not exit")


@uninterruptible()
def stop_container(container_id: str) -> None:
    """Remove the docker container ``container_id``; do nothing if it is already gone."""
    removed = _docker(["rm", "-f", container_id])
    if removed.returncode != 0 and "No such container" not in removed.stderr:
        raise RuntimeFailure(f"could not remove container {container_id}: {removed.stderr}")


def container_running(container_id: str) -> bool:
    inspected = _docker(["inspect", "--format", "{{.State.Running}}", container_id])
    return inspected.returncode == 0 and inspected.stdout.strip() == "true"


# --------------------------------------------------------------------------------------
# Readiness
# --------------------------------------------------------------------------------------


async def wait_until_ready(
    server: RunningServer,
    engine: Engine,
    *,
    api_key: str,
    served_model_name: str,
    timeout_s: float,
    interval_s: float = 0.5,
) -> None:
    """Poll the health endpoint until it answers, then confirm the server serves this model.

    Raises :class:`RuntimeFailure` with the server's log tail when the server exits, serves a
    different model, or is not ready within ``timeout_s``.
    """
    began = time.monotonic()
    deadline = began + timeout_s
    next_report = began + STILL_LOADING_EVERY_S
    say("waiting for the model to load")
    healthy = False
    async with httpx.AsyncClient(
        timeout=5.0, headers={"Authorization": f"Bearer {api_key}"}
    ) as client:
        while True:
            if not server.is_running():
                raise RuntimeFailure(exit_reason(server.log_text()))
            try:
                if not healthy:
                    health = await client.get(f"{server.endpoint_url}{engine.health_path}")
                    healthy = health.status_code == 200
                if healthy:
                    models = await client.get(f"{server.endpoint_url}{engine.models_path}")
                    if models.status_code < 400:
                        served = [entry["id"] for entry in models.json()["data"]]
                        if served_model_name in served:
                            say(f"ready after {time.monotonic() - began:.0f} s")
                            return
                        raise RuntimeFailure(
                            f"the server did not serve {served_model_name!r}; "
                            f"it serves {served}\n{log_tail(server.log_text())}"
                        )
                    if models.status_code in (401, 403):
                        raise RuntimeFailure(
                            f"the server rejected this run's API key\n{log_tail(server.log_text())}"
                        )
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                healthy = False  # still loading, or not an answer we can read yet
            if time.monotonic() >= deadline:
                raise RuntimeFailure(
                    f"the server was not ready within {timeout_s:g}s\n{log_tail(server.log_text())}"
                )
            if time.monotonic() >= next_report:
                say(f"still loading... {time.monotonic() - began:.0f} s")
                next_report += STILL_LOADING_EVERY_S
            await asyncio.sleep(interval_s)
