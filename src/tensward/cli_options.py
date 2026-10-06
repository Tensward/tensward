"""Command-line arguments shared by ``tensward`` commands and its extensions."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Callable

from .engines import ENGINES, Engine
from .errors import PROJECT_CONFIG_UNSUPPORTED, RUNNER_FAILURE, PreflightError, exit_code_for
from .platforms import platform_for
from .project import LEGACY_ENGINE, project_engine, registered_setup
from .runtime import DockerRuntime, LocalProcessRuntime, Runtime
from .slo import DEFAULT_SLO, Slo

RUNTIMES = ("docker", "local")


def add_project_argument(parser: argparse.ArgumentParser, *, verify_weights: bool = True) -> None:
    parser.add_argument("--project", type=Path, required=True, help="private project directory")
    if verify_weights:
        parser.add_argument(
            "--verify-weights",
            action="store_true",
            help="hash the model weights again (by default they are hashed only when a file's "
            "size or modification time changed since the last hash)",
        )


def add_slo_arguments(parser: argparse.ArgumentParser) -> None:
    """The per-request latency limits that goodput is counted against."""
    parser.add_argument(
        "--slo-ttft-ms",
        type=float,
        default=DEFAULT_SLO.ttft_ms,
        help="a request meets the SLO if its TTFT is at most this (goodput)",
    )
    parser.add_argument(
        "--slo-tpot-ms",
        type=float,
        default=DEFAULT_SLO.tpot_ms,
        help="a request meets the SLO if its TPOT is at most this (goodput)",
    )


def slo_for(arguments: argparse.Namespace) -> Slo:
    return Slo(arguments.slo_ttft_ms, arguments.slo_tpot_ms)


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """The arguments that choose the engine and how it is run."""
    parser.add_argument(
        "--engine",
        choices=sorted(ENGINES),
        help="serving engine: must be the project's (recorded by init)",
    )
    parser.add_argument(
        "--runtime", choices=RUNTIMES, default="docker", help="how to run the engine"
    )
    parser.add_argument(
        "--image",
        help="docker image (must already be present locally; default: the one your --current "
        "command ran, else the engine's)",
    )
    parser.add_argument(
        "--gpus",
        type=_gpu_list,
        metavar="INDICES",
        help="GPU device indices to run on, e.g. 0 or 0,1 (default: the ones your --current "
        "command selected, else the engine's choice)",
    )
    parser.add_argument(
        "--local-command",
        help="server command for --runtime local (the engine's arguments are appended)",
    )


def engine_for(arguments: argparse.Namespace) -> Engine:
    """The project's engine (vLLM before a project is registered); ``--engine``, when given,
    must name it."""
    current = registered_setup(arguments.project)
    engine = project_engine(current) if current else ENGINES[LEGACY_ENGINE]
    if arguments.engine not in (None, engine.name):
        raise PreflightError(
            PROJECT_CONFIG_UNSUPPORTED,
            f"--engine {arguments.engine} is not this project's engine ({engine.name}); drop "
            "--engine, or register a new project with it",
        )
    return engine


def _gpu_list(text: str) -> tuple[str, ...]:
    devices = tuple(text.split(","))
    if not all(device.isdecimal() for device in devices):
        raise argparse.ArgumentTypeError(f"{text!r} is not a list of device indices like 0,1")
    return devices


def runtime_for(arguments: argparse.Namespace) -> Runtime:
    engine = engine_for(arguments)
    current = registered_setup(arguments.project)
    gpus = arguments.gpus or (current.gpus if current else ()) or None
    platform = platform_for(engine.platforms)
    if arguments.runtime == "docker":
        image = arguments.image or (current.image if current else None) or engine.default_image
        return DockerRuntime(image, gpus, platform)
    command = shlex.split(arguments.local_command) if arguments.local_command else None
    return LocalProcessRuntime(command or engine.local_command, gpus, platform)


def report_refusal(error: PreflightError) -> None:
    print(json.dumps({"code": error.code, "message": error.message}), file=sys.stderr)


def describe_os_error(error: OSError) -> str:
    """What failed and where, e.g. ``Permission denied: /srv/project``."""
    reason = error.strerror or type(error).__name__
    return f"{reason}: {error.filename}" if error.filename else reason


def run_guarded(
    command: str, run: Callable[[], int], *, failures: tuple[type[Exception], ...] = ()
) -> int:
    """Run one command: a refusal prints as one JSON line with its exit code; an exception of
    ``failures`` prints "tensward <command> failed: <why>" with exit 1; any other I/O error
    prints as a JSON runner failure."""
    try:
        return run()
    except PreflightError as error:
        report_refusal(error)
        return exit_code_for(error.code)
    except failures as error:
        print(f"tensward {command} failed: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        report_refusal(PreflightError(RUNNER_FAILURE, f"I/O error: {describe_os_error(error)}"))
        return exit_code_for(RUNNER_FAILURE)
