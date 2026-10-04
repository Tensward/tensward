"""The ``tensward`` command line: ``env``, ``init``, ``inspect``, ``analyse``, ``compare`` and
``serve``.

Further commands, such as ``optimize`` from the separate ``tensward-optimize`` package, plug
in through the ``tensward.commands`` entry-point group.

``init`` and ``inspect`` print one line of JSON on standard output: identities and the current
setup (the ``--current`` command with secrets blanked), never an input's path or a prompt. A
refusal prints one line of JSON with a stable ``code`` and a ``message`` on standard error, and
exits with the status that code maps to (see ``errors``).
"""

from __future__ import annotations

import argparse
import functools
import json
import shlex
import signal
import sys
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from .engines import ENGINES, Engine
from .environment import unavailable_warning
from .errors import (
    EXIT_ANSWERS_DIFFER,
    EXIT_OK,
    PROJECT_CONFIG_UNSUPPORTED,
    PROJECT_INPUTS_INVALID,
    RUNNER_FAILURE,
    PreflightError,
    exit_code_for,
)
from .platforms import platform_for
from .progress import say
from .project import (
    LEGACY_ENGINE,
    hand_back_to_owner,
    init_project,
    load_project,
    project_engine,
    project_environment,
    project_fit,
    project_summary,
    registered_setup,
)
from .runtime import DEFAULT_READY_TIMEOUT_S
from .slo import DEFAULT_SLO, Slo

if TYPE_CHECKING:
    from .runtime import Runtime

COMMANDS_GROUP = "tensward.commands"
RUNTIMES = ("docker", "local")
ENV_DESCRIPTION = (
    "Show what Tensward detects on this machine: the platform and its devices, whether each "
    "engine is available as a docker image or a local command (and its version), the "
    "checkpoint formats it registers, and the combinations that work here. It starts nothing "
    "and writes nothing."
)
INIT_DESCRIPTION = (
    "Register a local checkpoint directory, a serving configuration and a JSONL workload as a "
    "private project. Registration only reads and hashes the inputs: it starts no engine."
)
ANALYSE_DESCRIPTION = (
    "Start the serving engine with the registered model and configuration, offer the "
    "registered prompts, measure the run and write a report under <project>/runs/<run_id>/. "
    "The server is always stopped afterwards."
)
SERVE_DESCRIPTION = (
    "Run the serving engine detached with a chosen configuration so applications can call its "
    "OpenAI-compatible API. `serve start` waits until the model is loaded and answering "
    "(up to --ready-timeout), then exits; the server keeps running until `serve stop`. If "
    "`start` is interrupted or times out, it stops the server it launched. State and the API key "
    "are kept under <project>/serve/<name>/."
)
COMPARE_DESCRIPTION = (
    "Compare the answers of two runs of the registered project, a baseline and a candidate: "
    "how far the candidate's answers moved, judged against the baseline's own repeats. Writes "
    "every prompt's answers side by side to <project>/compare/. Both runs must have kept "
    "their answers. With --baseline-answers, give only the candidate: it is compared with "
    "recorded production answers."
)
INSPECT_DESCRIPTION = (
    "Verify a registered project against its sources and print its identity. The checkpoint, "
    "configuration and workload identities are derived again and must match the record."
)


def build_parser() -> argparse.ArgumentParser:
    """Return the entry point's argument parser, including any installed extra commands."""
    parser = argparse.ArgumentParser(
        prog="tensward",
        description="Profile and diagnose LLM serving on your own GPU machine.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    init = subcommands.add_parser(
        "init",
        help="register one local checkpoint, configuration and workload",
        description=INIT_DESCRIPTION,
    )
    add_project_argument(init, verify_weights=False)
    init.add_argument("--model", type=Path, required=True, help="local checkpoint directory")
    init.add_argument("--config", type=Path, required=True, help="serving configuration JSON")
    init.add_argument("--prompts", type=Path, required=True, help="plain-text JSONL workload")
    init.add_argument(
        "--engine",
        choices=sorted(ENGINES),
        help="serving engine (default: the one --current runs, else the first that serves "
        "this checkpoint on this machine)",
    )
    current = init.add_mutually_exclusive_group()
    current.add_argument(
        "--current",
        metavar="COMMAND",
        help="the command you run today (`vllm serve ...`, `python -m vllm.entrypoints..."
        "api_server ...` or `docker run ...`); it is the baseline every result is compared to",
    )
    current.add_argument(
        "--current-file", type=Path, metavar="PATH", help="a file holding that command"
    )
    init.set_defaults(run=_run_registration)

    inspect = subcommands.add_parser(
        "inspect",
        help="verify and print one registered project",
        description=INSPECT_DESCRIPTION,
    )
    add_project_argument(inspect)
    inspect.set_defaults(run=_run_registration)

    env = subcommands.add_parser(
        "env", help="show what Tensward detects on this machine", description=ENV_DESCRIPTION
    )
    env.add_argument("--json", action="store_true", help="print one JSON object")
    env.add_argument("--engine", choices=sorted(ENGINES), help="check only this engine")
    env.add_argument("--runtime", choices=RUNTIMES, help="check only this runtime")
    env.add_argument("--image", help="docker image to check (default: each engine's)")
    env.add_argument("--local-command", help="server command to check (default: each engine's)")
    env.set_defaults(run=_run_env)

    analyse_parser = subcommands.add_parser(
        "analyse",
        help="serve the registered model, measure the registered prompts and diagnose",
        description=ANALYSE_DESCRIPTION,
    )
    add_project_argument(analyse_parser)
    add_runtime_arguments(analyse_parser)
    add_engine_arg_argument(analyse_parser)
    analyse_parser.add_argument(
        "--no-retain-responses",
        action="store_true",
        help="do not write generated text to the run directory",
    )
    add_slo_arguments(analyse_parser)
    add_equality_arguments(analyse_parser)
    analyse_parser.add_argument(
        "--require-gpu",
        metavar="NAME",
        help="refuse, before the model loads, unless the selected GPUs (all of them when none is "
        "selected) are this card, e.g. A10G; a leading vendor name and the case are ignored",
    )
    analyse_parser.add_argument(
        "--require-driver",
        metavar="VERSION",
        help="refuse, before the model loads, unless the driver is this version, or a dotted "
        "prefix of it (580 matches 580.95.05)",
    )
    analyse_parser.add_argument(
        "--trace",
        action="store_true",
        help="after the clean measurement, profile a short slice on a separate launch and report "
        "where the GPU goes idle (diagnostic timing, never a speed claim)",
    )
    analyse_parser.add_argument(
        "--counters",
        action="store_true",
        help="implies --trace; then profile the trace's top kernels with Nsight Compute (ncu on "
        "PATH, --runtime local) and report what the hardware did inside them",
    )
    analyse_parser.add_argument(
        "--ready-timeout",
        type=float,
        default=DEFAULT_READY_TIMEOUT_S,
        help="seconds to wait for the server to load the model",
    )
    analyse_parser.set_defaults(run=_run_analyse)

    compare_parser = subcommands.add_parser(
        "compare",
        help="compare the answers of two runs",
        description=COMPARE_DESCRIPTION,
    )
    add_project_argument(compare_parser)
    compare_parser.add_argument(
        "runs",
        nargs="+",
        metavar="RUN",
        help="run ids under runs/: the baseline, then the candidate (only the candidate with "
        "--baseline-answers)",
    )
    add_equality_arguments(compare_parser)
    compare_parser.set_defaults(run=functools.partial(_run_compare, compare_parser))

    serve_parser = subcommands.add_parser(
        "serve",
        help="run a long-lived OpenAI-compatible endpoint for a chosen configuration",
        description=SERVE_DESCRIPTION,
    )
    serve_parser.set_defaults(run=_run_serve)
    serve_commands = serve_parser.add_subparsers(dest="serve_command", required=True)
    start = serve_commands.add_parser(
        "start", help="start (or replace) a server, waiting until the model is ready"
    )
    _add_serve_arguments(start)
    add_runtime_arguments(start)
    add_engine_arg_argument(start)
    start.add_argument("--host", default="127.0.0.1", help="address to bind (default: loopback)")
    start.add_argument("--port", type=int, default=8000)
    start.add_argument(
        "--ready-timeout",
        type=float,
        default=DEFAULT_READY_TIMEOUT_S,
        help="seconds to wait for the model to load before giving up and stopping the server",
    )
    # These serve what the optional tensward-optimize package produced; without it only
    # `--from current` (the default) applies.
    optimized = start.add_argument_group("with an optimize result (tensward-optimize)")
    optimized.add_argument(
        "--from",
        dest="source",
        help="what to serve: 'current' (your current setup), 'latest' (the newest result of "
        "tensward-optimize) or the id of one such result (default: latest if the project has "
        "one, else current; the choice is printed)",
    )
    optimized.add_argument(
        "--package",
        help="which package of the optimize result to serve (default: the recommended one)",
    )
    optimized.add_argument(
        "--accept-unreviewed",
        action="store_true",
        help="serve a result whose changes can affect output quality",
    )
    for action, text in (("status", "check that a server is alive"), ("stop", "stop a server")):
        _add_serve_arguments(serve_commands.add_parser(action, help=text))

    for entry in entry_points(group=COMMANDS_GROUP):
        entry.load()(subcommands)
    if "optimize" not in subcommands.choices:
        missing = subcommands.add_parser(
            "optimize", help="search for better settings (needs the tensward-optimize package)"
        )
        missing.set_defaults(run=_optimizer_not_installed)
    return parser


def add_project_argument(parser: argparse.ArgumentParser, *, verify_weights: bool = True) -> None:
    parser.add_argument("--project", type=Path, required=True, help="private project directory")
    if verify_weights:
        parser.add_argument(
            "--verify-weights",
            action="store_true",
            help="hash the model weights again (by default they are hashed only when a file's "
            "size or modification time changed since the last hash)",
        )


def _add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    add_project_argument(parser)
    parser.add_argument("--name", default="default", help="server name within the project")


def add_engine_arg_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--engine-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="engine-specific flag, or an override of a setting, e.g. max-num-seqs=64 "
        "(repeatable); quantization=none lets the engine choose",
    )


def add_equality_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--require-equal",
        action="store_true",
        help=f"exit {EXIT_ANSWERS_DIFFER} unless every answer is one the baseline gave, or "
        "equality could not be shown",
    )
    parser.add_argument(
        "--baseline-answers",
        type=Path,
        metavar="FILE",
        help="compare with recorded production answers (JSONL: prompt_id, text, and optionally "
        "tool_calls and finish_reason) instead of a run of your current setup",
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
    from .runtime import DockerRuntime, LocalProcessRuntime

    engine = engine_for(arguments)
    current = registered_setup(arguments.project)
    gpus = arguments.gpus or (current.gpus if current else ()) or None
    platform = platform_for(engine.platforms)
    if arguments.runtime == "docker":
        image = arguments.image or (current.image if current else None) or engine.default_image
        return DockerRuntime(image, gpus, platform)
    command = shlex.split(arguments.local_command) if arguments.local_command else None
    return LocalProcessRuntime(command or engine.local_command, gpus, platform)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one invocation and return its process exit code."""
    parser = build_parser()
    arguments, unknown = parser.parse_known_args(argv)
    # Without the optimizer installed, `optimize` still answers, whatever flags it was given.
    if unknown and arguments.run is not _optimizer_not_installed:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    previous = signal.signal(signal.SIGTERM, _interrupt)
    try:
        status: int = arguments.run(arguments)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        status = 130
    finally:
        signal.signal(signal.SIGTERM, previous)
        if getattr(arguments, "project", None) is not None:
            hand_back_to_owner(arguments.project)
    return status


def _interrupt(signum: int, frame: object) -> None:
    """Treat SIGTERM like Ctrl-C, so a server that was started is stopped before we exit."""
    raise KeyboardInterrupt


def _optimizer_not_installed(arguments: argparse.Namespace) -> int:
    print(
        "tensward optimize needs the tensward-optimize package, which is not installed.",
        file=sys.stderr,
    )
    return 1


def _run_registration(arguments: argparse.Namespace) -> int:
    """Run ``init`` or ``inspect`` and print the sanitized project summary."""
    try:
        if arguments.command == "init":
            resolved = init_project(
                arguments.project,
                model=arguments.model,
                config=arguments.config,
                prompts=arguments.prompts,
                engine=arguments.engine,
                current=_current_text(arguments),
            )
        else:
            resolved = load_project(arguments.project, verify_weights=arguments.verify_weights)
        engine = project_engine(resolved.record.current_setup)
        environment = project_environment(resolved.record)
        fit = project_fit(resolved, engine)
    except PreflightError as error:
        report_refusal(error)
        return exit_code_for(error.code)
    except OSError as error:
        failure = PreflightError(RUNNER_FAILURE, f"I/O error: {_describe(error)}")
        report_refusal(failure)
        return exit_code_for(RUNNER_FAILURE)
    if arguments.command == "init":
        say(f"engine: {engine.name} ({environment['engine_choice']})")
        setup = resolved.record.current_setup
        if warning := unavailable_warning(engine, setup.text, setup.image):
            say(warning)
    summary = project_summary(resolved.record, resolved.anatomy, fit, environment)
    print(json.dumps(summary, sort_keys=True))
    return EXIT_OK


def _run_env(arguments: argparse.Namespace) -> int:
    """Print what Tensward detects on this machine."""
    from .environment import environment_report, render_environment

    report = environment_report(
        engines=[arguments.engine] if arguments.engine else list(ENGINES),
        runtimes=[arguments.runtime] if arguments.runtime else list(RUNTIMES),
        image=arguments.image,
        local_command=arguments.local_command,
    )
    print(
        json.dumps(report, sort_keys=True)
        if arguments.json
        else "\n".join(render_environment(report))
    )
    return EXIT_OK


def _describe(error: OSError) -> str:
    """What failed and where, e.g. ``Permission denied: /srv/project``."""
    reason = error.strerror or type(error).__name__
    return f"{reason}: {error.filename}" if error.filename else reason


def _current_text(arguments: argparse.Namespace) -> str | None:
    current: str | None = arguments.current
    if arguments.current_file is None:
        return current
    try:
        text: str = arguments.current_file.read_text("utf-8")
        return text
    except (OSError, ValueError):
        raise PreflightError(PROJECT_INPUTS_INVALID, "--current-file is not readable") from None


def _run_analyse(arguments: argparse.Namespace) -> int:
    """Run one analysis, print its summary and run directory, and return the exit status."""
    # Imported here so ``init`` and ``inspect`` never load the HTTP stack.
    from .analyse import AnalyseFailure, analyse
    from .report import FEEDBACK_LINE, render_headline
    from .runtime import RuntimeFailure

    try:
        result = analyse(
            arguments.project,
            engine=engine_for(arguments),
            runtime=runtime_for(arguments),
            engine_args=arguments.engine_arg,
            verify_weights=arguments.verify_weights,
            retain_responses=not arguments.no_retain_responses,
            ready_timeout_s=arguments.ready_timeout,
            trace=arguments.trace or arguments.counters,
            counters=arguments.counters,
            slo=slo_for(arguments),
            require_equal=arguments.require_equal,
            baseline_answers=arguments.baseline_answers,
            require_gpu=arguments.require_gpu,
            require_driver=arguments.require_driver,
        )
    except PreflightError as error:
        report_refusal(error)
        return exit_code_for(error.code)
    except (AnalyseFailure, RuntimeFailure, OSError) as error:
        print(f"tensward analyse failed: {error}", file=sys.stderr)
        return 1
    measurement = result.measurement
    print(f"Ran: {result.ran}")
    print("\n".join(render_headline(measurement, result.source, arguments.engine_arg)))
    print()
    if result.changes_text:
        print(result.changes_text)
        print()
    print(result.diagnosis_line)
    print()
    print(result.suggestions_text, end="")
    if result.comparison_line:
        print(result.comparison_line)
    print(f"run directory: {result.run_dir}")
    print(FEEDBACK_LINE)
    if measurement.succeeded == 0:
        print("tensward analyse failed: no request succeeded", file=sys.stderr)
        return 1
    return EXIT_ANSWERS_DIFFER if result.answers_differ else EXIT_OK


def _run_compare(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> int:
    """Compare two runs of the project, or one with recorded answers, and print the report
    section."""
    from .quality import (
        RunFile,
        RunRecord,
        compare_runs,
        render_section,
        require_same_inputs,
        write_comparison,
    )

    if len(arguments.runs) != (1 if arguments.baseline_answers else 2):
        parser.error(
            "give a baseline run and a candidate run, or only the candidate with --baseline-answers"
        )
    project_dir = arguments.project.expanduser()
    runs = project_dir / "runs"
    try:
        project = load_project(arguments.project, verify_weights=arguments.verify_weights)
        run_dirs = [runs / run_id for run_id in arguments.runs]
        for run_dir in run_dirs:
            if run_dir.resolve().parent != runs.resolve() or not run_dir.is_dir():
                raise PreflightError(PROJECT_INPUTS_INVALID, f"no run {run_dir.name} in {runs}")
        run_files = [RunFile.read(run_dir) for run_dir in run_dirs]
        require_same_inputs(project, *(run_file.snapshot_id for run_file in run_files))
        records = [
            RunRecord.from_run(run_dir, run_file)
            for run_dir, run_file in zip(run_dirs, run_files, strict=True)
        ]
        if arguments.baseline_answers:
            records.insert(0, RunRecord.from_recorded(arguments.baseline_answers, project))
        baseline, candidate = records
        comparison = compare_runs(baseline, candidate, project)
        answers = write_comparison(project_dir, comparison, baseline, candidate, project)
    except PreflightError as error:
        report_refusal(error)
        return exit_code_for(error.code)
    except OSError as error:
        failure = PreflightError(RUNNER_FAILURE, f"I/O error: {_describe(error)}")
        report_refusal(failure)
        return exit_code_for(RUNNER_FAILURE)
    print("\n".join(render_section(comparison, candidate, baseline, project)))
    print(f"answers side by side: {answers}")
    if arguments.require_equal and comparison.outcome.equal is not True:
        return EXIT_ANSWERS_DIFFER
    return EXIT_OK


def _run_serve(arguments: argparse.Namespace) -> int:
    """Run one serve subcommand and return the exit status."""
    from .analyse import AnalyseFailure
    from .report import overrides_suffix
    from .runtime import RuntimeFailure
    from .serve import (
        curl_example,
        is_alive,
        is_healthy,
        read_state,
        serve_dir,
        start_server,
        stop_server,
    )

    name = arguments.name
    try:
        if arguments.serve_command == "start":
            state = start_server(
                arguments.project,
                name=name,
                engine=engine_for(arguments),
                source=arguments.source,
                package=arguments.package,
                runtime=runtime_for(arguments),
                host=arguments.host,
                port=arguments.port,
                accept_unreviewed=arguments.accept_unreviewed,
                ready_timeout_s=arguments.ready_timeout,
                engine_args=arguments.engine_arg,
                verify_weights=arguments.verify_weights,
            )
            key_file = serve_dir(arguments.project, name) / "api_key"
            setup = f"{state['source']}{overrides_suffix(state['engine_args'])}"
            print(f"serving {setup} as {state['served_model_name']!r}")
            print(f"endpoint: {state['endpoint']}/v1")
            print(f"api key file: {key_file}")
            print(curl_example(state, key_file))
            return EXIT_OK
        if arguments.serve_command == "stop":
            stopped = stop_server(arguments.project, name)
            print(f"stopped {name}" if stopped else f"{name} was not running")
            return EXIT_OK
        recorded = read_state(arguments.project, name)
        if recorded is None:
            raise RuntimeFailure(f"no server named {name!r} in this project")
        alive = is_alive(recorded)
        healthy = alive and is_healthy(recorded)
        if recorded["status"] == "starting" and alive:
            verdict = "starting (the model is still loading)"
        else:
            verdict = "ok" if healthy else "alive but unhealthy" if alive else "not running"
        print(f"{name}: {verdict}")
        print(f"endpoint: {recorded['endpoint']}/v1 (started {recorded['started_at']})")
        print(f"serving: {recorded['source']}{overrides_suffix(recorded.get('engine_args', []))}")
        return EXIT_OK if healthy else 1
    except PreflightError as error:
        report_refusal(error)
        return exit_code_for(error.code)
    except (RuntimeFailure, AnalyseFailure, OSError) as error:
        print(f"tensward serve {arguments.serve_command} failed: {error}", file=sys.stderr)
        return 1


def report_refusal(error: PreflightError) -> None:
    print(json.dumps({"code": error.code, "message": error.message}), file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
