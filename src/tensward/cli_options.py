"""Command-line arguments shared by ``tensward`` commands and its extensions."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Callable

from .engines import ENGINES, Engine
from .engines.vllm.command import VLLM_VERSION
from .engines.vllm.flags import ENV_ALLOWED, env_allowed
from .errors import PROJECT_CONFIG_UNSUPPORTED, RUNNER_FAILURE, PreflightError, exit_code_for
from .platforms import platform_for
from .project import (
    LEGACY_ENGINE,
    project_engine,
    project_slos,
    registered_env,
    registered_setup,
)
from .runtime import DockerRuntime, LocalProcessRuntime, Runtime
from .slo import DEFAULT_SLO, Slo, declared_slo

RUNTIMES = ("docker", "local")
ADOPT_FLAG = "--adopt-within-noise"


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


def add_limit_arguments(parser: argparse.ArgumentParser) -> None:
    """The latency limits a user may declare; goodput is counted only against these."""
    parser.add_argument(
        "--slo-ttft-p95",
        "--slo-ttft-ms",
        dest="slo_ttft_ms",
        type=float,
        metavar="MS",
        help="declare a TTFT limit: goodput counts the requests within it, and with --goal "
        "throughput the report keeps the run's TTFT p95 within it (default: the "
        "configuration's slos.ttft_p95_ms, else none)",
    )
    parser.add_argument(
        "--slo-tpot-p95",
        "--slo-tpot-ms",
        dest="slo_tpot_ms",
        type=float,
        metavar="MS",
        help="declare a TPOT p95 limit: goodput counts the requests within it, and --capacity "
        "judges every step against it (default: the configuration's slos.tpot_p95_ms, else none)",
    )
    parser.add_argument(
        "--max-req-s-loss",
        type=share_below_100,
        metavar="PCT",
        help="with --goal ttft or tpot, a change that lowers requests per second waits under "
        "Could help once this run is more than PCT%% below the latest run of your starting "
        "settings; 0 sets no floor, also over the kept setup's",
    )


def add_remeasure_argument(parser: argparse.ArgumentParser) -> None:
    again = parser.add_mutually_exclusive_group()
    again.add_argument(
        "--remeasure",
        action="store_true",
        help="measure again instead of reusing the newest trusted run of the same setup",
    )
    again.add_argument(
        "--reuse-run",
        metavar="RUN",
        help="reuse run RUN, answers included, instead of measuring this step again: RUN "
        "measured these engine arguments and environment, kept its answers, is not itself "
        "reused and is at most 24 h old; otherwise the step is measured",
    )


def add_ttft_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ttft-from",
        choices=("first-content", "first-token"),
        default="first-content",
        help="which first token ranks the TTFT offers where the tool-call parser holds a "
        "streamed answer back: first-content (the default; the engine's own figure is used "
        "behind a hold of 4 or more decode steps) or first-token (the first streamed chunk "
        "with a choice)",
    )


def add_borrow_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--borrow-reference",
        action="store_true",
        help="reuse a reference run of the same setup up to 14 days old (24 hours otherwise) "
        "if it ran under this machine's power cap, on this machine's own GPUs (not an "
        "imported run); steps from it are judged with the cross-launch floor widened",
    )


def add_kb_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--kb",
        action="store_true",
        help="print the knowledge records behind each offer instead of the report: those that "
        "fired, with the facts that fired them, and those the run's facts could not decide",
    )


def add_window_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--window",
        type=window_choice,
        default="auto",
        metavar="auto|full|SECONDS",
        help="how many requests to measure: auto (a step after an earlier run measures the "
        "shortest window whose noise band is narrow enough for the goal; otherwise the "
        "configured count), full (the configured count), or a window of 30 to 3600 seconds "
        "at the rate of the newest run of the same setup",
    )
    parser.add_argument(
        ADOPT_FLAG,
        action="store_true",
        help="recommend keeping a change that is within noise but better on the goal with "
        "nothing limited worse beyond its band (also TENSWARD_ADOPT_WITHIN_NOISE=1)",
    )


def window_choice(text: str) -> str:
    if text in ("auto", "full"):
        return text
    try:
        seconds = float(text)
    except ValueError:
        seconds = math.nan
    if not 30 <= seconds <= 3600:
        raise argparse.ArgumentTypeError("must be auto, full, or seconds from 30 to 3600")
    return f"{seconds:g}"


def share_below_100(text: str) -> float:
    value = float(text)
    if not 0 <= value < 100:
        raise argparse.ArgumentTypeError("must be 0 or more and below 100")
    return value


def declared_limits(arguments: argparse.Namespace) -> Slo:
    """The limits ``add_limit_arguments`` read, else the project's declared ones, else none."""
    project = getattr(arguments, "project", None)
    return declared_slo(
        arguments.slo_ttft_ms, arguments.slo_tpot_ms, project_slos(project) if project else {}
    )


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
        help="the whole server command for --runtime local, subcommand included, e.g. "
        "'/path/to/vllm serve' (default 'vllm serve'); the model and the engine's arguments are "
        "appended",
    )
    parser.add_argument(
        "--pin-client",
        action="store_true",
        help="keep the engine off one CPU core, so the load generator has a core of its own "
        "(the engine's CPU affinity is every core but one)",
    )


def add_engine_env_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--engine-env",
        action="append",
        default=[],
        type=engine_env_item,
        metavar="NAME=VALUE",
        help="an engine environment variable for this run, with --runtime local (repeatable): "
        + ", ".join(sorted(set().union(*ENV_ALLOWED.values()))),
    )


def engine_env_item(text: str) -> tuple[str, str]:
    name, given, value = text.partition("=")
    if not given or not env_allowed(VLLM_VERSION, name):
        raise argparse.ArgumentTypeError(f"{name} is not an engine variable Tensward sets")
    return name, value


def engine_env(arguments: argparse.Namespace) -> dict[str, str]:
    """The ``--engine-env`` values; refused with ``--runtime docker``, where only the registered
    setup's environment reaches the container, and for a name the registered setup sets, whose
    value the engine starts with whatever the process environment says."""
    return refuse_engine_env(arguments, dict(arguments.engine_env))


def refuse_engine_env(arguments: argparse.Namespace, env: dict[str, str]) -> dict[str, str]:
    """``env`` when it can be applied: it is the given ``--engine-env`` or the kept setup's, and
    the same refusals hold for both."""
    if env and arguments.runtime == "docker":
        raise PreflightError(
            PROJECT_CONFIG_UNSUPPORTED,
            "--engine-env needs --runtime local: a container gets only the environment of the "
            "registered setup",
        )
    registered = registered_env(arguments.project)
    for name in env:
        if not env_allowed(VLLM_VERSION, name):
            raise PreflightError(
                PROJECT_CONFIG_UNSUPPORTED,
                f"--engine-env {name} is not an engine variable Tensward sets",
            )
        if name in registered:
            raise PreflightError(
                PROJECT_CONFIG_UNSUPPORTED,
                f"--engine-env {name}: the registered setup already sets {name}="
                f"{registered[name]}; Tensward cannot override it. Change the registered "
                f"command, or drop --engine-env {name}.",
            )
    return env


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
    pin_client = getattr(arguments, "pin_client", False)
    if arguments.runtime == "docker":
        image = arguments.image or (current.image if current else None) or engine.default_image
        return DockerRuntime(image, gpus, platform, pin_client)
    command = shlex.split(arguments.local_command) if arguments.local_command else None
    return LocalProcessRuntime(command or engine.local_command, gpus, platform, pin_client)


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
