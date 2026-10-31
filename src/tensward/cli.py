"""The ``tensward`` command line: ``env``, ``init``, ``inspect``, ``analyse``, ``compare``,
``set``, ``reject``, ``report`` and ``serve``.

Further commands, such as ``optimize``, plug in through the ``tensward.commands`` entry-point
group.

``init`` and ``inspect`` print one line of JSON on standard output: identities and the current
setup (the ``--current`` command with secrets blanked), never an input's path or a prompt. A
refusal prints one line of JSON with a stable ``code`` and a ``message`` on standard error, and
exits with the status that code maps to (see ``errors``).
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import signal
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import httpx

from . import __version__
from .cli_options import (
    ADOPT_FLAG,
    RUNTIMES,
    add_borrow_argument,
    add_engine_env_argument,
    add_kb_argument,
    add_limit_arguments,
    add_project_argument,
    add_remeasure_argument,
    add_runtime_arguments,
    add_ttft_arguments,
    add_window_arguments,
    declared_limits,
    engine_env,
    engine_for,
    refuse_engine_env,
    run_guarded,
    runtime_for,
    runtime_kind,
)
from .context import GOALS
from .engines import ENGINES, Engine
from .engines.llamacpp.command import probe_build
from .engines.llamacpp.detect import EndpointFacts, build_mismatch, probe_endpoint, sends_key
from .engines.vllm.quantization import require_servable_gptq
from .environment import EngineChoice, require_available, require_gpus, unavailable_warning
from .errors import (
    EXIT_ANSWERS_DIFFER,
    EXIT_OK,
    PROJECT_CONFIG_UNSUPPORTED,
    PROJECT_INPUTS_INVALID,
    AnalyseFailure,
    PreflightError,
    exit_code_for,
)
from .extensions import load_commands
from .kept import (
    ACCEPT_FLAG,
    LAST,
    REGISTERED,
    Kept,
    NotKept,
    keep,
    latest_run,
    read_kept,
    set_lines,
)
from .progress import Note, emit
from .project import (
    NeedsCommand,
    hand_back_to_owner,
    init_project,
    json_answer_ids,
    linear_weight_bytes,
    load_project,
    project_engine,
    project_environment,
    project_fit,
    project_fit_advice,
    project_summary,
    registered_env,
    settings_for,
    tool_calling_warning,
    workload_sample,
)
from .runtime import (
    DEFAULT_READY_TIMEOUT_S,
    Runtime,
    RuntimeFailure,
    end_load_watch,
)
from .secrets import shown_command
from .slo import Slo

if TYPE_CHECKING:
    from .context import AnalysisContext
    from .engines.protocol import FlagRules
    from .gpu_state import GpuState
    from .guard import Armed
    from .history import PastRun
    from .project import ResolvedProject
    from .report.model import Report
    from .settings import Settings

WINDOW_COMPARED = ("matched", "figures", "deltas")  # what the report compared, for window.json
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
SET_DESCRIPTION = (
    "Keep a run's settings: `tensward analyse` and `tensward serve start --from current` then "
    "start from them. `tensward set last` keeps the newest run's; `tensward set registered` goes "
    "back to the setup registered with init."
)
REJECT_DESCRIPTION = (
    "Reject a run's change (its engine arguments beyond those of the run it was made from): "
    "`tensward analyse` and `tensward report` then never offer it again for this project. "
    "`tensward reject last` rejects the newest run's; --clear RUN takes a "
    "rejection back; --list prints them."
)
REPORT_DESCRIPTION = (
    "Write a run's report again from its run directory, with the runs before it as its history "
    "and the project's kept setup and rejections as they are now: no engine starts and no new "
    "run is made. It rewrites the run's report.md and report.json and prints what analyse "
    "printed; with --kb it prints the knowledge records behind the offers instead."
)
IMPORT_DESCRIPTION = (
    "Copy a run directory of another project into this one under its own id. Both projects must "
    "have the same registration (model, configuration, workload and current setup). The next "
    "plain analyse reuses the run while it is trusted: same engine version, GPU and driver, a "
    "full window, at most 2%% of requests failed and measured within 24 hours."
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
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    init = subcommands.add_parser(
        "init",
        help="register one local checkpoint, configuration and workload",
        description=INIT_DESCRIPTION,
    )
    add_project_argument(init, verify_weights=False)
    init.add_argument(
        "--model",
        type=Path,
        help="local checkpoint directory or GGUF file (with --endpoint: default is the file "
        "the server reports it loaded, when this machine can read it)",
    )
    init.add_argument(
        "--endpoint",
        metavar="URL",
        help="a running llama-server to read the engine and build from (GET only, nothing is "
        "sent that loads it); needs --current or --current-file to register",
    )
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
        help="the command you run today (`vllm serve ...`, `python -m vllm.entrypoints."
        "openai.api_server ...`, `docker run ...` or `llama-server ...`); it is the baseline "
        "every result is compared to",
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
    env.add_argument(
        "--local-command",
        help="the whole server command to check, subcommand included, e.g. '/path/to/vllm serve' "
        "(default: each engine's)",
    )
    env.set_defaults(run=_run_env)

    analyse_parser = subcommands.add_parser(
        "analyse",
        help="serve the registered model, measure the registered prompts and diagnose",
        description=ANALYSE_DESCRIPTION,
    )
    add_project_argument(analyse_parser)
    add_runtime_arguments(analyse_parser)
    add_engine_arg_argument(analyse_parser)
    add_engine_env_argument(analyse_parser)
    analyse_parser.add_argument(
        "--no-retain-responses",
        action="store_true",
        help="do not write generated text to the run directory",
    )
    analyse_parser.add_argument(
        "--goal",
        choices=GOALS,
        default="throughput",
        help="what to improve: throughput (requests per second), ttft (TTFT p95) or tpot "
        "(TPOT p95); it ranks the suggestions and leads the report and compare",
    )
    add_limit_arguments(analyse_parser)
    add_equality_arguments(analyse_parser)
    analyse_parser.add_argument(
        "--no-probes",
        action="store_true",
        help="skip the extra launches that measure host memory bandwidth (llama.cpp with layers "
        "or experts in host memory); the decode ceiling is then not computed",
    )
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
        "--capacity",
        action="store_true",
        help="instead of one measurement, measure steps of closed-loop clients on one launch and "
        "report the highest requests per second whose TTFT p95 and TPOT p95 are within the "
        "declared limits; needs --slo-ttft-p95 or --slo-tpot-p95 (or slos in the configuration)",
    )
    add_remeasure_argument(analyse_parser)
    add_borrow_argument(analyse_parser)
    add_ttft_arguments(analyse_parser)
    add_window_arguments(analyse_parser)
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

    set_parser = subcommands.add_parser(
        "set", help="keep a run's settings as the project's setup", description=SET_DESCRIPTION
    )
    add_project_argument(set_parser, verify_weights=False)
    set_parser.add_argument(
        "run_id",
        metavar="RUN",
        help="a run id under <project>/runs, `last` for the newest run, or `registered` for the "
        "setup registered with init",
    )
    add_accept_argument(
        set_parser, "keep a change that can change answers, after comparing them in the report"
    )
    set_parser.set_defaults(run=_run_set)

    reject_parser = subcommands.add_parser(
        "reject", help="stop offering a run's change", description=REJECT_DESCRIPTION
    )
    add_project_argument(reject_parser, verify_weights=False)
    reject_parser.add_argument(
        "run_id",
        nargs="?",
        metavar="RUN",
        help="a run id under <project>/runs, or `last` for the newest run",
    )
    reject_parser.add_argument("--reason", help="why, shown with the change in later reports")
    reject_parser.add_argument(
        "--clear", metavar="RUN", help="take back the rejection of this run's change"
    )
    reject_parser.add_argument("--list", action="store_true", help="print the rejected changes")
    reject_parser.set_defaults(run=_run_reject)

    import_parser = subcommands.add_parser(
        "import",
        help="copy a run of another project with the same registration into this one",
        description=IMPORT_DESCRIPTION,
    )
    add_project_argument(import_parser, verify_weights=False)
    import_parser.add_argument("run_dir", type=Path, metavar="RUN_DIR", help="the run to copy")
    import_parser.set_defaults(run=_run_import)

    report_parser = subcommands.add_parser(
        "report", help="write a run's report again, no GPU", description=REPORT_DESCRIPTION
    )
    add_project_argument(report_parser, verify_weights=False)
    report_parser.add_argument(
        "run_id", metavar="RUN", help="a run id under <project>/runs, or `last` for the newest run"
    )
    add_kb_argument(report_parser)
    report_parser.set_defaults(run=_run_report)

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
    # These serve an optimize result; without one only `--from current` (the default) applies.
    optimized = start.add_argument_group("with an optimize result")
    optimized.add_argument(
        "--from",
        dest="source",
        help="what to serve: 'current' (the kept setup if `tensward set` kept one, else the "
        "registered setup), 'latest' (the newest optimize "
        "result) or the id of one such result (default: latest if the project has "
        "one, else current; the choice is printed)",
    )
    optimized.add_argument(
        "--package",
        help="which package of the optimize result to serve (default: the recommended one)",
    )
    add_accept_argument(
        optimized, "serve a change that can change answers, after comparing them in the report"
    )
    for action, text in (("status", "check that a server is alive"), ("stop", "stop a server")):
        _add_serve_arguments(serve_commands.add_parser(action, help=text))

    load_commands(subcommands)
    if "optimize" not in subcommands.choices:
        missing = subcommands.add_parser(
            "optimize", help="search for better settings (not available in this installation)"
        )
        missing.set_defaults(run=_optimizer_not_installed)
    return parser


def _add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    add_project_argument(parser)
    parser.add_argument("--name", default="default", help="server name within the project")


def add_accept_argument(parser: argparse._ActionsContainer, text: str) -> None:
    """--accept-answer-changes, with its earlier spelling --accept-unreviewed as a hidden alias."""
    parser.add_argument(ACCEPT_FLAG, dest="accept_answer_changes", action="store_true", help=text)
    parser.add_argument(
        "--accept-unreviewed",
        dest="accept_answer_changes",
        action="store_true",
        help=argparse.SUPPRESS,
    )


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
        "the optimize command is not available in this installation",
        file=sys.stderr,
    )
    return 1


def _run_registration(arguments: argparse.Namespace) -> int:
    """Run ``init`` or ``inspect`` and print the sanitized project summary."""

    def run() -> int:
        facts = None
        if arguments.command == "init":
            facts = _probe_endpoint(arguments.endpoint) if arguments.endpoint else None
            try:
                resolved = init_project(
                    arguments.project,
                    model=_model_path(arguments.model, facts),
                    config=arguments.config,
                    prompts=arguments.prompts,
                    engine=arguments.engine,
                    current=_current_text(arguments),
                    endpoint=facts.evidence if facts else None,
                    current_is_file=arguments.current_file is not None,
                    check=_refuse_unservable_tools,
                )
            except NeedsCommand as needed:
                emit(Note(text=_engine_line(needed.choice)))
                raise
        else:
            resolved = load_project(arguments.project, verify_weights=arguments.verify_weights)
        engine = project_engine(resolved.record.current_setup)
        environment = project_environment(resolved.record)
        fit = project_fit(resolved, engine)
        if arguments.command == "init":
            emit(Note(text=(
                _engine_line(resolved.choice) if resolved.choice and (
                    resolved.choice.assumed
                    or (resolved.choice.evidence and engine.family != "vllm")
                ) else f"engine: {engine.name} ({environment['engine_choice']})"
            )))  # fmt: skip
            setup = resolved.record.current_setup
            if warning := unavailable_warning(engine, setup.text, setup.image, setup.executable):
                emit(Note(text=warning))
            if facts and (warning := _build_warning(facts, setup.executable)):
                emit(Note(text=warning))
            if advice := project_fit_advice(resolved, engine, setup.engine_settings):
                emit(Note(text=advice[0]))
        current = settings_for(resolved, engine, ())
        if arguments.command == "init":
            _tool_call_refusal(resolved, engine, current)
        if warning := tool_calling_warning(resolved, engine, current):
            emit(Note(text=warning))
        summary = project_summary(
            resolved.record,
            resolved.anatomy,
            fit,
            environment,
            kept=read_kept(arguments.project.expanduser()),
        )
        print(json.dumps(summary, sort_keys=True))
        return EXIT_OK

    return run_guarded(arguments.command, run)


def _engine_line(choice: EngineChoice) -> str:
    name = choice.engine.name
    if choice.assumed:
        return f"engine: {name} (assumed from {choice.evidence[0]}; pass --engine to confirm)"
    return f"engine: {name} [from: {'; '.join(choice.evidence)}]"


def _probe_endpoint(url: str) -> EndpointFacts:
    if os.environ.get("LLAMA_API_KEY") and not sends_key(url):
        emit(
            Note(text=f"warning: LLAMA_API_KEY is not sent to {url}; it is neither https nor local")
        )
    with httpx.Client() as client:
        found = probe_endpoint(url, client)
    if found is None:
        raise PreflightError(PROJECT_INPUTS_INVALID, f"nothing answers at {url}")
    return found


def _model_path(given: Path | None, facts: EndpointFacts | None) -> Path:
    """``--model``, else the file the endpoint reports it loaded when this machine can read it."""
    if given is not None:
        return given
    reported = Path(facts.model_path) if facts and facts.model_path else None
    if reported is None or not reported.is_file():
        raise PreflightError(
            PROJECT_INPUTS_INVALID,
            "--model is required"
            + ("" if facts is None else "; the endpoint's /props model_path is not a file here"),
        )
    return reported


def _build_warning(facts: EndpointFacts, executable: str | None) -> str | None:
    if executable is None:
        return None
    found = probe_build(executable)
    return build_mismatch(facts.build, found[0].build if found else None)


def _refuse_unservable_tools(resolved: ResolvedProject) -> None:
    engine = project_engine(resolved.record.current_setup)
    _tool_call_refusal(resolved, engine, settings_for(resolved, engine, ()))


def _tool_call_refusal(project: ResolvedProject, engine: Engine, settings: Settings) -> None:
    """Refuse a project whose configuration turns tool calling on but whose ``--current``
    command does not enable it, while the prompts offer tools: the server would refuse every
    one of those requests."""
    if project.record.current_setup.source != "command" or not project.config.case.tool_calling:
        return
    if (warning := tool_calling_warning(project, engine, settings)) is not None:
        raise PreflightError(PROJECT_CONFIG_UNSUPPORTED, warning.removeprefix("warning: "))


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


def _current_text(arguments: argparse.Namespace) -> str | None:
    current: str | None = arguments.current
    if arguments.current_file is None:
        return current
    try:
        text: str = arguments.current_file.read_text("utf-8")
        return text
    except (OSError, ValueError):
        raise PreflightError(PROJECT_INPUTS_INVALID, "--current-file is not readable") from None


def repeated_flags(arguments: argparse.Namespace, floor: float | None, kind: str) -> list[str]:
    """The goal, limit, floor (``floor``, a percentage), within-noise policy and runtime options
    a suggested command repeats; never ``--window``, as each step plans its own."""
    flags = ["--goal", arguments.goal] if arguments.goal != "throughput" else []
    if arguments.adopt_within_noise:
        flags.append(ADOPT_FLAG)
    if arguments.slo_ttft_ms is not None:
        flags += ["--slo-ttft-p95", f"{arguments.slo_ttft_ms:g}"]
    if arguments.slo_tpot_ms is not None:
        flags += ["--slo-tpot-p95", f"{arguments.slo_tpot_ms:g}"]
    if floor is not None and arguments.goal != "throughput":
        flags += ["--max-req-s-loss", f"{floor:g}"]
    if arguments.ttft_from != "first-content":
        flags += ["--ttft-from", arguments.ttft_from]
    flags += ["--runtime", kind]
    if arguments.gpus:
        flags += ["--gpus", ",".join(arguments.gpus)]
    for flag, given in (("--image", arguments.image), ("--local-command", arguments.local_command)):
        if given:
            flags += [flag, shown := shown_command(given)]
            if shown != given:
                emit(Note(text=(
                    f"note: suggested commands show the secrets in {flag} as ***; export them in "
                    "the environment so those commands run as printed"
                )))  # fmt: skip
    return flags


def kept_engine_args(kept: Kept | None, given: Sequence[str]) -> list[str]:
    """The engine arguments of an analysis: the kept setup's when none are given, else the given
    ones, which are the run's whole change from the registered setup."""
    if kept is None or not kept.engine_args:
        return list(given)
    if not given:
        held = ", ".join(kept.engine_args)
        emit(Note(text=f"starting from the kept setup (run {kept.run}): {held}"))
        return list(kept.engine_args)
    named = {arg.partition("=")[0]: arg for arg in given}
    for arg in kept.engine_args:
        flag = arg.partition("=")[0]
        if flag not in named:
            emit(Note(text=(
                f"note: --engine-arg leaves out the kept setup's {arg} (run {kept.run}); "
                "add it to keep it"
            )))  # fmt: skip
        elif named[flag] != arg:
            emit(Note(
                text=f"note: --engine-arg replaces the kept setup's {arg} (run {kept.run})"
            ))  # fmt: skip
    return list(given)


def kept_engine_env(kept: Kept | None, given: Mapping[str, str]) -> dict[str, str]:
    """The engine environment of an analysis: the kept setup's when none is given, else the
    given one, which is the run's whole change from the registered setup."""
    if kept is None or not kept.env:
        return dict(given)
    held = {name: value for name, _, value in (var.partition("=") for var in kept.env)}
    if not given:
        emit(Note(text=f"starting from the kept setup (run {kept.run}): {', '.join(kept.env)}"))
        return held
    for name, value in held.items():
        if name not in given:
            emit(Note(text=(
                f"note: --engine-env leaves out the kept setup's {name}={value} (run {kept.run}); "
                "add it to keep it"
            )))  # fmt: skip
        elif given[name] != value:
            emit(Note(
                text=f"note: --engine-env replaces the kept setup's {name}={value} (run {kept.run})"
            ))  # fmt: skip
    return dict(given)


def _run_analyse(arguments: argparse.Namespace) -> int:
    """Run one analysis, print its summary and run directory, and return the exit status."""
    # Imported here so ``init`` and ``inspect`` do not load the analysis modules.
    from .analyse import analyse
    from .answer_options import read_answer_options, record_answer_options
    from .answers_after import quality_document
    from .context import AnalysisContext, active
    from .files import write_json
    from .gpu_state import GpuSampler
    from .guard import arm
    from .history import cap_flag, cold_kv_tokens, cold_pools, past_runs
    from .rebuild import record_analysis
    from .rejected import read_rejected, standing
    from .runs import record_sample
    from .token_cache import cached_tokens

    def run() -> int:
        runtime = runtime_for(arguments)
        slo = declared_limits(arguments)
        engine = engine_for(arguments)
        project_dir = arguments.project.expanduser()
        kept = read_kept(project_dir)
        engine_args = kept_engine_args(kept, arguments.engine_arg)
        os.environ.update(
            refuse_engine_env(arguments, kept_engine_env(kept, engine_env(arguments)), engine)
        )
        if arguments.max_req_s_loss is not None and arguments.goal == "throughput":
            emit(Note(text="note: --max-req-s-loss applies only to --goal ttft or tpot"))
        given = floor = arguments.max_req_s_loss  # 0: no floor, also over the kept setup's
        if given is None and kept is not None and kept.max_req_s_loss is not None:
            floor = kept.max_req_s_loss * 100
            if arguments.goal != "throughput":
                emit(Note(text=f"floor of the kept setup (run {kept.run}): --max-req-s-loss "
                          f"{floor:g}"))  # fmt: skip
        reused = (
            _reuse_run(
                arguments, engine, slo, _changed(project_dir, engine_args, engine.flag_rules), floor
            )
            if arguments.reuse_run is not None
            else _reuse_reference(arguments, engine, runtime, slo, engine_args, floor)
        )
        if reused is not None:
            return reused
        if arguments.capacity:
            return _run_capacity(arguments, engine, runtime, slo, engine_args)
        runs = past_runs(project_dir, engine)
        runs_dir = project_dir / "runs"
        project = load_project(arguments.project, verify_weights=False)
        _tool_call_refusal(project, engine, settings_for(project, engine, engine_args))
        available = require_available(engine, runtime)
        if engine.family == "vllm":
            require_servable_gptq(project.artifact.path, available.version)
        changed = _changed(project_dir, engine_args, engine.flag_rules)
        plan, parent = _window_plan(arguments, project_dir, project, runs, changed, kept)
        rejected, retried = standing(read_rejected(project_dir), engine.flag_rules)
        context = AnalysisContext(
            engine=engine.name,
            goal=arguments.goal,
            ttft_p95_limit_ms=slo.declared.get("ttft_ms"),
            tpot_p95_limit_ms=slo.declared.get("tpot_ms"),
            flags=tuple(repeated_flags(arguments, floor, runtime.kind)),
            max_req_s_loss=floor / 100 if floor else None,
            engine_args=tuple(changed),
            cold_kv_tokens=cold_kv_tokens(runs, frozenset(changed), cap_flag(engine)),
            cold_pools=cold_pools(runs, frozenset(changed), cap_flag(engine)),
            sampler=GpuSampler(runtime.platform, runtime.gpus),
            project=project_dir,
            runs_before=frozenset(p.name for p in runs_dir.iterdir())
            if runs_dir.is_dir()
            else frozenset(),
            kept=kept,
            rejected=rejected,
            retried=retried,
            json_answers=json_answer_ids(project),
            read_answer_options=functools.partial(read_answer_options, engine, runtime),
            window=arguments.window,
            window_plan=plan,
            adopt_within_noise=_adopts_within_noise(arguments),
        )
        context.linear_weight_bytes = linear_weight_bytes(project)
        context.anatomy = project.anatomy
        if (observer := engine.log.cache_observer(context, runtime, os.environ)) is not None:
            context.observers.append(observer)
        if parent is not None:
            parent_run = next(run for run in runs if run.run_id == parent)
            armed = arm(parent_run, runs_dir / parent, frozenset(changed), engine)
            context.startup_guard = asdict(armed) if armed is not None else None
            if armed is not None:
                _arm_load_watch(context, armed, runs_dir / parent, engine)
        counted = cached_tokens(engine, runtime, project_dir, project, engine_args)
        with active(context):
            try:
                result = analyse(
                    arguments.project,
                    engine=counted,
                    runtime=runtime,
                    engine_args=engine_args,
                    verify_weights=arguments.verify_weights,
                    retain_responses=not arguments.no_retain_responses,
                    ready_timeout_s=arguments.ready_timeout,
                    trace=arguments.trace or arguments.counters,
                    counters=arguments.counters,
                    slo=slo,
                    require_equal=arguments.require_equal,
                    baseline_answers=arguments.baseline_answers,
                    require_gpu=arguments.require_gpu,
                    require_driver=arguments.require_driver,
                    probes=not arguments.no_probes,
                )
            except AnalyseFailure:
                stopped = _record_stop(
                    context, parent, lambda: settings_for(project, engine, engine_args)
                )
                if stopped is not None:
                    raise AnalyseFailure(stopped) from None
                raise
            finally:
                end_load_watch(context)
                counted.save()
        if (sample := workload_sample(arguments.project)) is not None:
            record_sample(result.run_dir, sample)
        record_answer_options(
            result.run_dir,
            context.answer_options,
            context.answer_options_unknown,
            context.gpu_capability,
        )
        record_analysis(result.run_dir, context, [entry.id for entry in project.prompts])
        window = result.measurement.window
        gpu = context.gpu_state(
            window.start_ns if window else None, window.end_ns if window else None
        )
        _write_window(result.run_dir, context, parent, gpu, engine)
        _write_facts(result.run_dir, context, parent, engine)
        if (after := context.answers_after) is not None and after.outcome is not None:
            write_json(result.run_dir / "quality.json", quality_document(after, result.run_id))
        measurement = result.measurement
        _print_report(result.report, result.run_dir)
        if measurement.succeeded == 0:
            print("tensward analyse failed: no request succeeded", file=sys.stderr)
            return 1
        return EXIT_ANSWERS_DIFFER if result.answers_differ else EXIT_OK

    return run_guarded("analyse", run, failures=(AnalyseFailure, RuntimeFailure, OSError))


def _window_plan(
    arguments: argparse.Namespace,
    project_dir: Path,
    project: ResolvedProject,
    runs: Sequence[PastRun],
    engine_args: Sequence[str],
    kept: Kept | None,
) -> tuple[dict[str, Any], str | None]:
    """The window this analysis measures (``window.Plan`` as a dict) and the run it is a step
    from: a step plans from its parent, ``--window SECONDS`` from the newest run of the same
    setup, a re-measure or ``--window SECONDS`` from the newest full run of the same setup when
    that gives a steady window; else a first run or a re-measure plans as a first run
    (``window.first``), and ``--window SECONDS`` the configured count with why."""
    from .combine import PREDICTED, parent_offer
    from .history import kept_run, planned_window, steady_source, step_parent
    from .window import Plan, first

    args = frozenset(engine_args)
    parent = step_parent(project_dir, runs, args, kept_run(project_dir, runs, kept))
    workload = project.config.workload
    sample = workload_sample(project_dir)
    prompts = sample.of if sample is not None else len(project.prompts)
    only = None if arguments.window in ("auto", "full") else float(arguments.window)
    source = parent
    if only is not None:
        same = [run for run in runs if run.served and run.engine_args == args]
        source = same[-1] if same else parent
    if parent is None or only is not None:
        source = steady_source(project_dir, runs, args, workload, arguments.goal) or source
    if arguments.window == "full":
        plan = Plan(workload.request_count, None, "--window full")
    elif source is None and only is None:
        arrival = workload.arrival
        clients = (arrival.concurrency or 0) if arrival.kind == "closed_loop" else 0
        latency = arguments.goal != "throughput"
        plan = first(
            clients,
            workload.request_count,
            matched=latency,
            prompts=prompts,
            ramped=arguments.goal == "ttft",
        )
    elif source is None:
        plan = Plan(workload.request_count, None, "no earlier run to plan from")
    else:
        offer = (
            parent_offer(project_dir / "runs" / parent.run_id, args - parent.engine_args)
            if parent is not None
            else None
        )
        predicted = ((offer or {}).get("predicted") or {}).get(PREDICTED.get(arguments.goal, ""))
        plan = planned_window(
            project_dir / "runs" / source.run_id, workload, goal=arguments.goal,
            predicted=predicted, only=only, quality_risk=bool((offer or {}).get("quality_risk")),
            prompts=prompts,
        )  # fmt: skip
    emit(Note(text=(
        f"window: {plan.requests} requests (about {plan.seconds:g} s); {plan.reason}"
        if plan.seconds
        else f"window: configured count ({plan.reason})"
        if plan.requests == workload.request_count
        else f"window: {plan.reason}"
    )))  # fmt: skip
    record = {k: v for k, v in asdict(plan).items() if v is not None or k == "seconds"}
    return record, parent.run_id if parent is not None else None


def _write_window(
    run_dir: Path,
    context: AnalysisContext,
    parent: str | None,
    gpu: GpuState | None,
    engine: Engine,
) -> None:
    """Write the run's ``window.json``: its plan, the run it is a step from, how its engine
    started, the GPU memory sampled over the measured window, and what its report compared."""
    from .files import write_json
    from .runs import RUN_RECORD, WINDOW_RECORD

    log = run_dir / "server.log"
    record = json.loads((run_dir / RUN_RECORD).read_text("utf-8"))
    text = log.read_text("utf-8", errors="replace") if log.is_file() else None
    started = (
        {}
        if text is None
        else engine.log.window_facts(
            text,
            record.get("compile_cache"),
            context.window_extra.get("caches"),
            (record.get("settings") or {}).get("quantization"),
        )
    )
    document: dict[str, Any] = {
        "schema_version": "1",
        "plan": context.window_plan,
        "parent": parent,
        "startup": {"ready_s": context.load_s, **started},
        "memory": {
            "peak_used_mib": gpu.memory_peak_mib if gpu else None,
            "total_mib": gpu.memory_total_mib if gpu else None,
        },
        **{key: context.window_record[key] for key in WINDOW_COMPARED
           if key in context.window_record},
    }  # fmt: skip
    if clash := document.keys() & context.window_extra.keys():
        raise ValueError(f"window.json keys set twice: {sorted(clash)}")
    write_json(run_dir / WINDOW_RECORD, {**document, **context.window_extra})


def _write_facts(
    run_dir: Path, context: AnalysisContext, parent: str | None, engine: Engine
) -> None:
    """Write the run's ``facts.json`` and, for a step measured from a known parent, append its
    training row to the local knowledge."""
    from .kb_run import append_step_row, write_facts

    if context.kb_facts is None:
        return
    write_facts(run_dir, context.kb_facts)
    if parent is not None and context.project is not None:
        append_step_row(context.project, run_dir, parent, engine, context)


def _arm_load_watch(
    context: AnalysisContext, armed: Armed, parent_dir: Path, engine: Engine
) -> None:
    """Have the load watcher stop this step's server if requests queue at the KV limit in the
    first minute of the load. It is armed only when the parent's own log shows that minute
    without that, and the step does not declare a smaller KV pool. The minute starts at the
    load's own first stats line (``load_from_line`` of the parent's window.json), not at the
    capacity probe's or the warm-up's; a parent without that record, or whose saved log was cut
    (``log_cut`` in its window.json), is not judged."""
    if armed.shrinks_kv:
        return
    try:
        window = json.loads((parent_dir / "window.json").read_text("utf-8"))
        if window.get("log_cut"):
            return
        parent = engine.log.load_queue(
            (parent_dir / "server.log").read_text("utf-8", errors="replace"),
            window["load_from_line"],
        )
    except (OSError, ValueError, KeyError):
        return
    if parent is not None and parent.seen and parent.queued is None:
        context.window_extra["load_watch"] = {"parent": armed.parent, "state": "armed"}


def _record_stop(
    context: AnalysisContext, parent: str | None, settings: Callable[[], Settings]
) -> str | None:
    """Write the ``window.json`` of a step that did not run to the end because its KV pool was
    too small (the start-up guard) or because requests queued in the first minute of the load
    (the load watcher): its plan, its parent and that record with the step's engine arguments
    (it has no run.json to carry them). A step the load watcher stopped also gets the run.json
    and report.json that let ``tensward reject`` name it. Returns the load watcher's reason."""
    from .files import write_json
    from .runs import WINDOW_RECORD

    guard_record = context.window_extra.get("startup_guard")
    watch = context.window_extra.get("load_watch")
    watched = watch if watch is not None and watch["state"] == "stopped" else None
    run_dir = context.this_run()
    if run_dir is None or (guard_record is None and watched is None):
        return None
    document: dict[str, Any] = {
        "schema_version": "1",
        "plan": context.window_plan,
        "parent": parent,
    }
    if guard_record is not None:
        document["startup_guard"] = {**guard_record, "engine_args": list(context.engine_args)}
    if watched is not None:
        document["load_watch"] = {**watched, "engine_args": list(context.engine_args)}
        _write_stopped_run(run_dir, context, watched, settings())
    write_json(run_dir / WINDOW_RECORD, document)
    return watched["reason"] if watched is not None else None


def _write_stopped_run(
    run_dir: Path, context: AnalysisContext, watch: Mapping[str, Any], settings: Settings
) -> None:
    """The run.json of a stopped step, its parent's with the step's engine arguments and
    settings, and a report.json that says why it stopped. run.json comes last, as for any run."""
    from .files import write_json
    from .runs import RUN_RECORD, RunFile

    assert context.project is not None
    record = RunFile.read(context.project / "runs" / watch["parent"]).model_dump(mode="json")
    write_json(run_dir / "report.json", {
        "schema_version": "1",
        "run_id": run_dir.name,
        "goal": {"goal": context.goal},
        "ran": watch["reason"],
        "sections": [],
        "kb": {"pack": None},
    })  # fmt: skip
    write_json(run_dir / RUN_RECORD, {
        **record, "engine_args": list(context.engine_args), "settings": asdict(settings),
    })  # fmt: skip


def _changed(project_dir: Path, engine_args: Sequence[str], rules: FlagRules) -> list[str]:
    """A run's change from the registered setup: its engine arguments, then the engine
    environment it inherits beyond the registered setup's."""
    from .history import env_args

    registered = registered_env(project_dir)
    inherited = {name: value for name, value in os.environ.items() if name not in registered}
    return [*engine_args, *env_args(inherited, rules)]


def _adopts_within_noise(arguments: argparse.Namespace) -> bool:
    return arguments.adopt_within_noise or os.environ.get("TENSWARD_ADOPT_WITHIN_NOISE") == "1"


def _reuse_reference(
    arguments: argparse.Namespace,
    engine: Engine,
    runtime: Runtime,
    slo: Slo,
    engine_args: Sequence[str],
    floor: float | None,
) -> int | None:
    """Reuse the newest trusted run of an unchanged setup in place of a measurement and return
    the exit status; None when the analysis must measure."""
    from .reuse import NO_RUN, trusted

    explicit = (
        arguments.remeasure, arguments.trace, arguments.counters, arguments.capacity,
        arguments.baseline_answers, arguments.require_equal, arguments.engine_arg,
        arguments.require_gpu, arguments.require_driver, arguments.engine_env,
        arguments.window != "auto",
    )  # fmt: skip
    if any(explicit):
        return None
    project_dir = arguments.project.expanduser()
    found = trusted(
        project_dir, engine, runtime, frozenset(engine_args),
        retain_responses=not arguments.no_retain_responses, limits=slo,
        borrow=arguments.borrow_reference,
    )  # fmt: skip
    if isinstance(found, str):
        if found != NO_RUN:
            emit(Note(text=f"not reused: {found}"))
        return None
    return _reused(
        arguments, project_dir, found, slo, floor, runtime.kind, borrowed=arguments.borrow_reference
    )


def _reuse_run(
    arguments: argparse.Namespace,
    engine: Engine,
    slo: Slo,
    changed: Sequence[str],
    floor: float | None,
) -> int | None:
    """Reuse run ``--reuse-run`` for this step and return the exit status; None, with the
    reason, when it may not stand in for it (``reusable``, and these engine arguments and
    environment are its own) and the step is measured."""
    from .reuse import reusable

    project_dir = arguments.project.expanduser()
    found = reusable(project_dir, arguments.reuse_run, engine)
    if not isinstance(found, str) and found.engine_args != frozenset(changed):
        found = f"run {found.run_id} measured other engine arguments or environment"
    if isinstance(found, str):
        emit(Note(text=f"not reused: {found}; measuring the step"))
        return None
    return _reused(arguments, project_dir, found, slo, floor, runtime_kind(arguments, engine))


def _reused(
    arguments: argparse.Namespace,
    project_dir: Path,
    found: PastRun,
    slo: Slo,
    floor: float | None,
    kind: str,
    *,
    borrowed: bool = False,
) -> int | None:
    """A new run that reuses ``found``, its report printed, and the exit status; None when its
    report cannot be written again."""
    from .context import AnalysisContext
    from .rebuild import NotReported
    from .reuse import reuse, reused_line

    context = AnalysisContext(
        goal=arguments.goal,
        flags=tuple(repeated_flags(arguments, floor, kind)),
        max_req_s_loss=floor / 100 if floor else None,
        adopt_within_noise=_adopts_within_noise(arguments),
    )
    try:
        run_dir, report = reuse(project_dir, found, context, limits=slo, borrowed=borrowed)
    except NotReported as error:
        emit(Note(text=f"not reused: run {found.run_id} cannot be reported again: {error}"))
        return None
    print(reused_line(project_dir, found, borrowed=borrowed))
    _print_report(report, run_dir)
    return EXIT_OK


def _run_import(arguments: argparse.Namespace) -> int:
    """Copy a run of another project into this one."""
    from .reuse import NotImported, import_run

    def run() -> int:
        load_project(arguments.project, verify_weights=False)
        try:
            target = import_run(arguments.run_dir, arguments.project)
        except NotImported as error:
            print(f"not imported: {error}", file=sys.stderr)
            return 1
        print(
            f"imported run {target.name}: a plain `tensward analyse` reuses it while it is trusted"
        )
        return EXIT_OK

    return run_guarded("import", run)


def _print_report(report: Report, run_dir: Path) -> None:
    from .report.build import FEEDBACK_LINE
    from .report.terminal import to_terminal

    print(to_terminal(report), end="")
    print(f"run directory: {run_dir}")
    print(FEEDBACK_LINE)


def _run_capacity(
    arguments: argparse.Namespace,
    engine: Engine,
    runtime: Runtime,
    slo: Slo,
    engine_args: Sequence[str],
) -> int:
    """Measure capacity at the declared limits and print every step."""
    from .slo_capacity import capacity_at, capacity_lines

    def refused(why: str) -> int:
        print(f"tensward analyse --capacity: {why}.", file=sys.stderr)
        return exit_code_for(PROJECT_CONFIG_UNSUPPORTED)

    if not slo.declared:
        return refused(
            "it needs a declared limit: --slo-ttft-p95 MS and/or --slo-tpot-p95 MS (or slos in "
            "the configuration); without one every step is within it"
        )
    flags = (
        ("--trace", arguments.trace),
        ("--counters", arguments.counters),
        ("--require-equal", arguments.require_equal),
        ("--baseline-answers", arguments.baseline_answers is not None),
        (f"--goal {arguments.goal}", arguments.goal != "throughput"),
    )
    for flag, given in flags:
        if given:
            return refused(f"it measures requests per second within your limits; drop {flag}")
    if arguments.require_gpu or arguments.require_driver:
        require_gpus(
            runtime.platform, runtime.gpus, arguments.require_gpu, arguments.require_driver
        )
    capacity = capacity_at(
        arguments.project,
        engine=engine,
        runtime=runtime,
        engine_args=engine_args,
        slo=slo,
        ready_timeout_s=arguments.ready_timeout,
        verify_weights=arguments.verify_weights,
    )
    print("\n".join(capacity_lines(slo, capacity)))
    return EXIT_OK


def _run_compare(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> int:
    """Compare two runs of the project, or one with recorded answers, and print the report
    section."""
    from .history import failed_share, removed_failures
    from .report.compare import recorded_goal, render_section, write_comparison
    from .runs import RunFile, RunRecord, compare_runs, recorded_sample, require_same_inputs

    if len(arguments.runs) != (1 if arguments.baseline_answers else 2):
        parser.error(
            "give a baseline run and a candidate run, or only the candidate with --baseline-answers"
        )
    project_dir = arguments.project.expanduser()
    runs = project_dir / "runs"

    def run() -> int:
        project = load_project(arguments.project, verify_weights=arguments.verify_weights)
        run_dirs = [runs / run_id for run_id in arguments.runs]
        for run_dir in run_dirs:
            if run_dir.resolve().parent != runs.resolve() or not run_dir.is_dir():
                raise PreflightError(PROJECT_INPUTS_INVALID, f"no run {run_dir.name} in {runs}")
        run_files = [RunFile.read(run_dir) for run_dir in run_dirs]
        require_same_inputs(project, *(run_file.snapshot_id for run_file in run_files))
        if recorded_sample(run_dirs[0]) != recorded_sample(run_dirs[-1]):
            print("note: the runs sent different samples of the workload; compare them with care")
        records = [
            RunRecord.from_run(run_dir, run_file)
            for run_dir, run_file in zip(run_dirs, run_files, strict=True)
        ]
        if arguments.baseline_answers:
            records.insert(0, RunRecord.from_recorded(arguments.baseline_answers, project))
        baseline, candidate = records
        comparison = compare_runs(baseline, candidate, project, project_dir)
        answers = write_comparison(project_dir, comparison, baseline, candidate, project)
        engine = project_engine(project.record.current_setup)
        advice = engine.reproducibility_advice(
            project.record.current_setup.source, project.record.current_setup.engine_settings
        )
        removed = None
        if not arguments.baseline_answers and removed_failures(
            project_dir, arguments.runs[0], run_dirs[1], ()
        ):
            removed = (failed_share(run_dirs[0]) or 0.0, failed_share(run_dirs[1]) or 0.0)
        section = render_section(
            comparison,
            candidate,
            baseline,
            project,
            advice=advice,
            show_expected=True,
            goal=recorded_goal(run_dirs[-1]),
            removed=removed,
        )
        print("\n".join(section))
        print(f"answers side by side: {answers}")
        if arguments.require_equal and comparison.outcome.equal is not True:
            return EXIT_ANSWERS_DIFFER
        return EXIT_OK

    return run_guarded("compare", run)


def _kept_command(project_dir: Path, kept: Kept | None) -> str | None:
    """The customer's own command with the kept setup written in, as it was measured."""
    from .runs import served_settings

    if kept is None:
        return None
    project = load_project(project_dir, verify_weights=False)
    setup = project.record.current_setup
    if setup.text is None:
        return None
    engine = project_engine(setup)
    settings = served_settings(
        project_dir, project, engine, settings_for(project, engine, kept.engine_args)
    )
    return engine.served_command(setup.text, settings)


def _run_set(arguments: argparse.Namespace) -> int:
    """Keep a run's settings, or go back to the registered setup, and print what changed."""
    from .answers_after import broken_in_chain
    from .rejected import carried, read_rejected
    from .serve import any_running

    def run() -> int:
        latest = arguments.run_id == LAST
        run = (latest_run(arguments.project) or LAST) if latest else arguments.run_id
        project = arguments.project.expanduser()
        found = broken_in_chain(project, run) if run != REGISTERED else None
        broken, note = found if found is not None else (run, None)
        made_from = f", which run {run} was made from" if broken != run else ""
        looks = (
            f"answers look broken in run {broken} ({note}){made_from}: read them in "
            f"{project / 'runs' / broken / 'report.md'} (Answers before / after)"
        )
        warning = f"warning: {looks}"
        try:
            accepted = arguments.accept_answer_changes
            now, before = keep(arguments.project, arguments.run_id, accept_unreviewed=accepted)
        except NotKept as error:
            if note and error.command is not None:
                print(
                    f"not kept: {looks}.\nTensward does not recommend keeping run {run}. If you "
                    f"have read the answers and still want it:\n  {error.command}",
                    file=sys.stderr,
                )
                return 1
            print(f"not kept: {error}", file=sys.stderr)
            if note:
                print(warning, file=sys.stderr)
            return 1
        serving = now != before and any_running(arguments.project)
        args = frozenset(now.engine_args) if now is not None else frozenset()
        rejected = carried(read_rejected(arguments.project), args) if args else None
        lines = set_lines(
            arguments.project, run, now, before, serving=serving, latest=latest,
            rejected_in=rejected.run if rejected else None, command=_kept_command(project, now),
        )  # fmt: skip
        if note and now is not None and now != before:
            lines.insert(1, warning)
        print("\n".join(lines))
        return EXIT_OK

    return run_guarded("set", run)


def _run_reject(arguments: argparse.Namespace) -> int:
    """Reject a run's change, take a rejection back, or list them."""
    from .rejected import (
        NotRejected,
        clear,
        clear_lines,
        list_lines,
        read_rejected,
        reject,
        reject_lines,
    )

    def run() -> int:
        given = [arguments.run_id is not None, arguments.clear is not None, arguments.list]
        if sum(given) != 1 or (arguments.reason is not None and arguments.run_id is None):
            raise PreflightError(
                PROJECT_CONFIG_UNSUPPORTED,
                "give a run to reject (with --reason, optionally), --clear RUN or --list",
            )
        load_project(arguments.project, verify_weights=False)
        if arguments.list:
            lines = list_lines(read_rejected(arguments.project))
        elif arguments.clear is not None:
            lines = clear_lines(arguments.clear, clear(arguments.project, arguments.clear))
        else:
            try:
                rejection = reject(arguments.project, arguments.run_id, arguments.reason)
            except NotRejected as error:
                print(f"not rejected: {error}", file=sys.stderr)
                return 1
            latest = arguments.run_id == LAST
            lines = reject_lines(arguments.project, rejection, latest=latest)
        print("\n".join(lines))
        return EXIT_OK

    return run_guarded("reject", run)


def _run_report(arguments: argparse.Namespace) -> int:
    """Write a run's report again and print it as analyse did."""
    from .rebuild import NotReported, rebuild

    def run() -> int:
        try:
            run_dir, report = rebuild(arguments.project, arguments.run_id)
        except NotReported as error:
            print(f"not reported: {error}", file=sys.stderr)
            return 1
        if arguments.kb:
            from .report.kb import kb_lines

            print("\n".join(kb_lines(json.loads((run_dir / "report.json").read_text("utf-8")))))
            return EXIT_OK
        _print_report(report, run_dir)
        return EXIT_OK

    return run_guarded("report", run)


def _run_serve(arguments: argparse.Namespace) -> int:
    """Run one serve subcommand and return the exit status."""
    from .report.build import overrides_suffix
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

    def run() -> int:
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
                accept_unreviewed=arguments.accept_answer_changes,
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

    return run_guarded(
        f"serve {arguments.serve_command}",
        run,
        failures=(RuntimeFailure, AnalyseFailure, OSError),
    )


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
