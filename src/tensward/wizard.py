"""The ``init`` wizard: what the machine has, four questions, and the project they make.

``plan_init`` settles the engine, the baseline, the model and the four answers, asking only
what a flag did not already say and only on a terminal; ``prepare_init`` turns the plan into the
project's inputs."""

from __future__ import annotations

import contextlib
import itertools
import math
import os
import shlex
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .cli_options import JSON_OUTPUT
from .detect import (
    WEIGHT_MEMORY_SHARE,
    EngineFound,
    Hardware,
    Machine,
    detect_machine,
    fit_consequence,
    install_hint,
    model_line,
    serving_problems,
)
from .engines import ENGINES
from .errors import (
    ENGINE_UNAVAILABLE,
    PROJECT_INPUTS_CHANGED,
    PROJECT_INPUTS_INVALID,
    PreflightError,
)
from .formats import FORMATS, detect_format
from .formats.gguf import GGUF
from .formats.hf_safetensors import HF_SAFETENSORS
from .inputs import load_prompt_entries
from .presets import (
    GOAL_DEFAULT,
    GOALS,
    INPUTS_DIR,
    OUTPUT_TOKENS,
    REQUEST_COUNT,
    WARMUP_REQUESTS,
    Choices,
    concurrency_default,
    load_tools,
    thinks,
    tools_default,
    write_choices,
    write_inputs,
)
from .progress import Note, PhaseStarted, emit
from .project import archive_existing_project
from .project_home import PROJECT_DIR
from .settings import ParsedSetup
from .text import printable

if TYPE_CHECKING:
    import argparse

    from .engines import Engine
    from .runtime import Runtime

ENGINE_LABELS = {"vllm": "vLLM", "llamacpp": "llama.cpp"}
ENGINE_PROGRAMS = {"vllm": "vllm", "llamacpp": "llama-server"}
FORMAT_ENGINE = {"a GGUF file": "llamacpp", "a safetensors folder": "vllm"}
CONTAINER_PROGRAMS = ("docker", "podman")
MODEL_QUESTION = "Path to your model (a safetensors folder or a .gguf file)"
MODEL_FOLDERS = ("~/models", ".")
SHOWN_MODELS = 8
LISTED_ENTRIES = 200  # per folder, so a huge folder never slows the question
WORKLOAD_OPTIONS = (
    "chat",
    "long documents/RAG",
    "agent (multi-turn)",
    "code",
    "batch/offline",
    "my own prompts file (JSONL)",
)
WORKLOADS = ("chat", "rag", "agent", "code", "batch", "own")
TOOL_OPTIONS = (
    "no",
    "yes, with built-in sample tools (search, calculator, weather, files, SQL)",
    "yes, with my own tools file (OpenAI format)",
)
GOAL_OPTIONS = (
    "fast first token (TTFT)",
    "smooth streaming (TPOT)",
    "most requests per GPU (throughput)",
)
DEFAULTS_LABEL = "engine defaults (not your real setup)"
BASELINE_QUESTION = (
    "Use this as your baseline? [Enter = yes, p = paste another command, d = engine defaults]"
)
NO_SERVER_LINE = (
    "No vLLM or llama.cpp server is running. That's fine: Tensward starts its own copy to measure."
)
NO_SERVER_QUESTION = "[p] paste the command you'd serve with   [Enter] engine defaults"
CONTINUE_QUESTION = "[Enter] continue · [e] edit engine/model/baseline"
SECONDS_PER_GB = 6


def _read_on_stderr(prompt: str) -> str:
    sys.stderr.write(prompt)
    sys.stderr.flush()
    return input()


def _write_on_stderr(text: str) -> None:
    print(text, file=sys.stderr)


def _short(option: str) -> str:
    """The option as the default is named in a heading: its one-word abbreviation in brackets
    when a longer phrase has one (``throughput``), else the text before its brackets."""
    text, _, tail = option.partition(" (")
    word = tail.rstrip(")")
    return word if len(text.split()) >= 3 and word.isalpha() else text


class Asker:
    """Questions on a terminal. ``read`` takes the prompt and returns the answer; ``write``
    prints a line."""

    def __init__(
        self,
        read: Callable[[str], str] = _read_on_stderr,
        write: Callable[[str], None] = _write_on_stderr,
    ) -> None:
        self.read = read
        self.write = write

    def _answer(self, prompt: str) -> str:
        try:
            return self.read(prompt).strip()
        except EOFError:
            raise KeyboardInterrupt from None

    def choice(self, question: str, options: Sequence[str], default: int) -> int:
        """The 1-based number of the option chosen; Enter takes ``default``."""
        self.write(f"{question[:-1]} [{_short(options[default - 1])}]:")
        for number, option in enumerate(options, 1):
            self.write(f"  {number}. {option}")
        while True:
            answer = self._answer(f"[{default}] ")
            if not answer:
                return default
            if answer.isdecimal() and 1 <= int(answer) <= len(options):
                return int(answer)
            self.write(f"type a number from 1 to {len(options)}")

    def number(self, question: str, default: int, minimum: int = 1) -> int:
        while True:
            answer = self._answer(f"{question} [{default}] ")
            if not answer:
                return default
            if answer.isdecimal() and int(answer) >= minimum:
                return int(answer)
            self.write(f"type a whole number of {minimum} or more")

    def path(
        self,
        question: str,
        check: Callable[[Path], str | None],
        default: Path | None = None,
        numbered: Sequence[Path] = (),
    ) -> Path:
        """A path ``check`` accepts (it returns the reason it does not, else None), or the
        number of one of ``numbered``. ``~`` and ``$VARS`` are expanded; a relative answer that
        is only missing its leading ``/`` or ``~/`` is offered as the default."""
        while True:
            answer = self._answer(f"{question} " + (f"[Enter = {default}] " if default else ""))
            if not answer and default is not None:
                return default
            if not answer:
                continue
            if answer.isdecimal() and 1 <= int(answer) <= len(numbered):
                return numbered[int(answer) - 1]
            path = Path(os.path.expandvars(answer)).expanduser()
            if reason := check(path):
                self.write(reason)
                if (meant := _meant(path, check)) is not None:
                    self.write(f"Did you mean {meant}?")
                    default = meant
                continue
            return path.resolve()

    def line(self, question: str, check: Callable[[str], str | None]) -> str:
        while True:
            answer = self._answer(f"{question} ")
            if answer and not (reason := check(answer)):
                return answer
            if answer:
                self.write(str(reason))

    def confirm(self, question: str) -> bool:
        """Enter and ``y`` are yes, ``n`` is no."""
        while True:
            answer = self._answer(f"{question} ").lower()
            if answer in ("", "y", "yes"):
                return True
            if answer in ("n", "no"):
                return False
            self.write("type Enter for yes or n for no")

    def key(self, question: str, keys: Mapping[str, str]) -> str:
        """Enter (``""``) or one of ``keys``."""
        while True:
            answer = self._answer(f"{question} ").lower()
            if not answer or answer in keys:
                return answer
            offered = ", ".join(f"{key} ({meaning})" for key, meaning in keys.items())
            self.write(f"type Enter or one of: {offered}")


@dataclass(frozen=True)
class InitPlan:
    model: Path
    engine: str
    current: str | None
    choices: Choices
    run_now: bool | None  # None: ask once the project is saved


@dataclass
class Prepared:
    """A plan whose inputs are written; ``discard`` removes the files Tensward wrote (never
    one that was already there), ``commit`` records the answers once the project is registered."""

    plan: InitPlan
    project: Path
    config: Path
    prompts: Path
    written: tuple[Path, ...]
    hardware: Hardware
    ask: Asker | None
    quiet: bool

    def discard(self) -> None:
        for path in self.written:
            path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            (self.project / INPUTS_DIR).rmdir()

    def commit(self) -> None:
        write_choices(self.project, self.plan.choices)

    def finish(
        self,
        run: Callable[[Sequence[str]], int],
        first: Callable[[], tuple[str | None, int]],
    ) -> int:
        """Say where the project is and how to repeat the setup, then run the analysis if asked."""
        say = _speaker(self.ask, self.quiet)
        home = Path.cwd() / PROJECT_DIR
        shown = "./.tensward" if self.project == home else str(self.project)
        say(f"Project saved in {shown} (runs, reports).")
        say("To repeat this setup without questions, run:")
        for line in equivalent_lines(self.plan):
            say(line)
        run_now = self.plan.run_now
        if run_now is None:
            assert self.ask is not None
            held, minutes = first()
            if held is not None:
                say(held)
                say("Stop your server, then run: tensward analyse")
                return 0
            uses = "uses the CPU" if self.hardware.kind == "cpu" else "uses the GPU"
            run_now = self.ask.confirm(
                f"Run the analysis now? Takes ~{minutes} min, {uses}. [Enter = yes]"
            )
        if not run_now:
            return 0
        command = ["analyse", "--project", str(self.project), "--yes"]
        if self.quiet:
            with contextlib.redirect_stdout(sys.stderr):
                return run(command)
        return run(command)


def load_estimate_s(weight_bytes: int) -> float:
    """Seconds one model load takes: 60 plus 6 per GB of weights."""
    return 60 + SECONDS_PER_GB * weight_bytes / 1e9


DECODE_BYTES_PER_S = 330e9  # what one stream's decode can read: about an L4's memory bandwidth
PREFILL_S = 2.0  # a request's prompt, before its first token
MEAN_OUTPUT_SHARE = 0.75  # of the output cap a request writes on average
RELAUNCH_LOAD_SHARE = 0.1  # of a cold load a second start of the same weights costs


def run_seconds(
    weight_bytes: int,
    *,
    concurrency: int = 4,
    output_tokens: int = OUTPUT_TOKENS["chat"],
    requests: int = REQUEST_COUNT,
    warmup: int = WARMUP_REQUESTS,
    relaunch: bool = False,
) -> float:
    """Seconds the first analysis takes: one load, the warm-up, the start-up wave and the
    requests, each stream decoding at the rate the weights allow. Every duration Tensward says
    before a run comes from here. ``relaunch``: the weights do not fit, so the measurement is
    made a second time with the placement the first start chose."""
    rate = min(max(DECODE_BYTES_PER_S / max(weight_bytes, 1), 3.0), 150.0)
    each = PREFILL_S + output_tokens * MEAN_OUTPUT_SHARE / rate
    rounds = math.ceil(warmup / concurrency) + 1 + math.ceil(requests / concurrency)
    load = load_estimate_s(weight_bytes)
    measured = rounds * each
    if relaunch:
        return load * (1 + RELAUNCH_LOAD_SHARE) + 2 * measured
    return load + measured


def workload_plan(workload: Any) -> dict[str, Any]:
    """The arguments of :func:`run_seconds` that a declared workload sets."""
    if workload is None:
        return {}
    return {
        "concurrency": max(getattr(workload.arrival, "concurrency", None) or 1, 1),
        "output_tokens": workload.output_tokens,
        "requests": workload.request_count,
    }


def run_minutes(weight_bytes: int, **plan: Any) -> int:
    """:func:`run_seconds` in whole minutes, rounded up."""
    return math.ceil(run_seconds(weight_bytes, **plan) / 60)


def may_relaunch(engine: Engine, runtime: Runtime, weight_bytes: int) -> bool:
    """Whether the weights may not fit the device, so that llama.cpp's own placement is
    measured first and the run is made again with it held."""
    if engine.family != "llamacpp" or runtime.platform.memory_topology == "host":
        return False
    devices, _ = runtime.platform.select(runtime.platform.detect(), runtime.gpus)
    return bool(devices) and weight_bytes > devices[0].total_bytes * WEIGHT_MEMORY_SHARE


def analysis_minutes(
    project: Path,
    weight_bytes: int,
    engine: Engine,
    runtime: Runtime,
    engine_args: Sequence[str] = (),
) -> int:
    """Minutes the first analysis of ``project`` takes: the one figure init, analyse, try and
    serve all say."""
    from .project import declared_workload

    return run_minutes(
        weight_bytes,
        **workload_plan(declared_workload(project)),
        relaunch=not engine_args and may_relaunch(engine, runtime, weight_bytes),
    )


def equivalent_lines(plan: InitPlan) -> list[str]:
    """The command that repeats ``plan`` on short lines a shell joins: the model and engine,
    the choices, then the baseline command last."""
    head, baseline, rest = _equivalent_parts(plan)
    lines = [shlex.join(head), shlex.join(rest), shlex.join(baseline)]
    return [f"  {line} \\" for line in lines[:-1]] + [f"  {lines[-1]}"]


def _equivalent_parts(plan: InitPlan) -> tuple[list[str], list[str], list[str]]:
    choices = plan.choices
    head = ["tensward", "init", "--model", str(plan.model), "--engine", plan.engine]
    baseline = ["--current", plan.current] if plan.current else ["--engine-defaults"]
    rest = (
        ["--prompts", str(choices.prompts)]
        if choices.workload == "own"
        else ["--workload", choices.workload]
    )
    rest += ["--tools", choices.tools]
    if choices.structured:
        rest.append("--structured")
    rest += ["--concurrency", str(choices.concurrency), "--goal", choices.goal, "--yes"]
    return head, baseline, rest


def missing_flags(arguments: argparse.Namespace, machine: Machine) -> list[str]:
    """The flags a run without a terminal still needs."""
    given_baseline = arguments.current or arguments.current_file or arguments.endpoint
    missing = []
    if arguments.model is None and not given_baseline:
        missing.append("--model")
    if arguments.engine is None and len(machine.engines) > 1 and not given_baseline:
        missing.append("--engine")
    if not given_baseline and not arguments.engine_defaults:
        missing.append("--current (or --engine-defaults)")
    if arguments.workload is None and arguments.prompts is None:
        missing.append("--workload")
    for flag, value in (
        ("--tools", arguments.tools),
        ("--concurrency", arguments.concurrency),
        ("--goal", arguments.goal),
    ):
        if value is None:
            missing.append(flag)
    return missing


def _speaker(ask: Asker | None, quiet: bool) -> Callable[[str], None]:
    if ask is not None:
        return ask.write
    return (lambda text: None) if quiet else print


def _joined(words: Sequence[str]) -> str:
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} and {words[-1]}"


def _engine_line(found: EngineFound) -> str:
    label = f"{ENGINE_LABELS.get(found.name, found.name)} {found.version or ''}".strip()
    command = os.path.basename(shlex.split(found.command)[0]) if found.command else found.name
    if found.backends:
        return f"{label} ({command}, {'/'.join(found.backends)} build)"
    return f"{label} ({command} command)"


def _parse(engine: str, text: str | None) -> ParsedSetup | None:
    if text is None:
        return None
    try:
        return ENGINES[engine].parse_setup(text)
    except (ValueError, PreflightError):
        return None


def _meant(path: Path, check: Callable[[Path], str | None]) -> Path | None:
    """The path the user probably meant when ``path`` lacks its leading ``/`` or ``~/``."""
    if path.is_absolute():
        return None
    for candidate in (Path("/") / path, Path.home() / path):
        if candidate.exists() and check(candidate) is None:
            return candidate
    return None


def _model_like(path: Path) -> bool:
    if path.is_file():
        return path.suffix == ".gguf"
    return (path / "config.json").is_file() and any(path.glob("*.safetensors"))


def found_models(folders: Sequence[Path]) -> list[Path]:
    """The .gguf files and safetensors folders in ``folders`` and one level below them."""
    found: list[Path] = []

    def entries(folder: Path) -> list[Path]:
        try:
            listed = sorted(itertools.islice(folder.iterdir(), LISTED_ENTRIES))
        except OSError:
            return []
        return [entry for entry in listed if not entry.name.startswith(".")]

    for folder in dict.fromkeys(f.expanduser().resolve() for f in folders):
        for entry in entries(folder):
            inner = entries(entry) if entry.is_dir() and not _model_like(entry) else []
            found += [path for path in (entry, *inner) if path not in found and _model_like(path)]
            if len(found) >= SHOWN_MODELS:
                return found[:SHOWN_MODELS]
    return found


def _ask_model(ask: Asker, missing: str | None, default: Path | None = None) -> Path:
    if missing:
        ask.write(f"The model in your command, {missing}, isn't there.")
    numbered = found_models([Path(folder) for folder in MODEL_FOLDERS])
    if numbered:
        ask.write("Models found here:")
        for number, path in enumerate(numbered, 1):
            ask.write(f"  {number}. {printable(str(path))}")
    question = f"{MODEL_QUESTION}{', or its number' if numbered else ''}:"
    return ask.path(question, _model_problem, default=default, numbered=numbered)


def _engine_missing(arguments: argparse.Namespace, machine: Machine, current: str | None) -> None:
    """Refuse a ``--current`` command whose engine is not installed here, naming it."""
    if current is None or arguments.engine:
        return
    try:
        program = os.path.basename(shlex.split(current)[0])
    except (ValueError, IndexError):
        return
    if program in CONTAINER_PROGRAMS:
        return
    installed = {found.name for found in machine.engines}
    for name, engine in ENGINES.items():
        if name not in installed and engine.recognizes(current):
            raise PreflightError(
                ENGINE_UNAVAILABLE,
                f"Your command runs {ENGINE_LABELS.get(name, name)} ({ENGINE_PROGRAMS[name]}), "
                "which is not installed here. Install it: see 'Don't have an engine yet?' in "
                "the README.",
            )


def _model_kind(path: Path) -> str | None:
    """ "a GGUF file" or "a safetensors folder", None for anything else."""
    if GGUF.detect(path):
        return "a GGUF file"
    if path.is_dir() and HF_SAFETENSORS.detect(path):
        return "a safetensors folder"
    return None


def _named_model(arguments: argparse.Namespace, current: str | None) -> Path | None:
    """The model ``--model`` or the ``--current`` command names, when it is on this machine."""
    if arguments.model:
        return Path(arguments.model).expanduser()
    try:
        words = shlex.split(current)[1:] if current else []
    except ValueError:
        return None
    return next((Path(w).expanduser() for w in words if _model_kind(Path(w).expanduser())), None)


def _format_mismatch(
    arguments: argparse.Namespace,
    machine: Machine,
    model: Path | None,
    engine: str | None = None,
    current: str | None = None,
) -> None:
    """Refuse a model whose format the chosen engine cannot load, or whose engine is not
    installed here, before any question or measurement."""
    kind = _model_kind(model) if model is not None else None
    if kind is None:
        return
    needed = FORMAT_ENGINE[kind]
    label = ENGINE_LABELS[needed]
    command = None
    if current is not None:
        try:
            program = os.path.basename(shlex.split(current)[0])
        except (ValueError, IndexError):
            program = ""
        if program in CONTAINER_PROGRAMS:
            return
        command = next((n for n, e in ENGINES.items() if e.recognizes(current)), None)
    for given, said in ((command, "Your command runs"), (arguments.engine, "--engine is")):
        if given is not None and given != needed:
            raise PreflightError(
                ENGINE_UNAVAILABLE,
                f"{said} {ENGINE_LABELS.get(given, given)}, which cannot load your model: it is "
                f"{kind}, which runs on {label}.",
            )
    if not arguments.endpoint and all(found.name != needed for found in machine.engines):
        raise PreflightError(
            ENGINE_UNAVAILABLE,
            f"Your model is {kind}, which runs on {label}; {label} is not installed here. "
            "Install it: see 'Don't have an engine yet?' in the README.",
        )
    if engine is not None and engine != needed:
        raise PreflightError(
            ENGINE_UNAVAILABLE,
            f"Your model is {kind}, which runs on {label}, not on "
            f"{ENGINE_LABELS.get(engine, engine)}: pick {label}.",
        )


def _model_problem(path: Path) -> str | None:
    if not path.exists():
        return f"{path} does not exist"
    if not any(fmt.detect(path) for fmt in FORMATS):
        return f"{path} is not a safetensors checkpoint folder or a .gguf file"
    return None


def _tools_problem(path: Path) -> str | None:
    if not path.is_file():
        return f"{path} does not exist"
    try:
        load_tools(path)
    except PreflightError as error:
        return error.message
    return None


def _prompts_problem(path: Path) -> str | None:
    if not path.is_file():
        return f"{path} does not exist"
    try:
        entries = load_prompt_entries(path)
    except PreflightError as error:
        return error.message
    for number, entry in enumerate(entries, 1):
        if entry.messages is None:
            return (
                f"record {number} is a plain prompt; init serves the chat API, so each record "
                'needs "messages" (e.g. [{"role": "user", "content": "..."}])'
            )
    return None


def _checked(path: Path, check: Callable[[Path], str | None], flag: str) -> Path:
    path = path.expanduser()
    if reason := check(path):
        raise PreflightError(PROJECT_INPUTS_INVALID, f"{flag}: {reason}")
    return path.resolve()


def _pick_engine(
    arguments: argparse.Namespace,
    machine: Machine,
    named: str | None,
    ask: Asker | None,
    current: EngineFound | None = None,
) -> EngineFound:
    """``--engine``, else the engine the baseline command names, else the only one, else the
    user's choice (the first without a terminal)."""
    by_name = {found.name: found for found in machine.engines}
    if current is None and arguments.engine:
        if arguments.engine not in by_name:
            raise PreflightError(
                ENGINE_UNAVAILABLE,
                f"--engine {arguments.engine} is not installed on this machine; "
                f"installed: {_joined([_engine_line(f) for f in machine.engines])}",
            )
        return by_name[arguments.engine]
    if current is None and named:
        for found in machine.engines:
            if ENGINES[found.name].recognizes(named):
                return found
    if len(machine.engines) == 1 or ask is None:
        return current or machine.engines[0]
    lines = [_engine_line(found) for found in machine.engines]
    default = machine.engines.index(current) + 1 if current else 1
    return machine.engines[ask.choice("Which engine do you serve with?", lines, default) - 1]


def _check_engine(engine: EngineFound, hardware: Hardware, told: set[str]) -> None:
    problems = serving_problems(engine.name, hardware, engine.backends)
    if stops := [p.text for p in problems if p.stop]:
        raise PreflightError(ENGINE_UNAVAILABLE, " ".join(stops))
    for problem in problems:
        if problem.text not in told:
            told.add(problem.text)
            emit(Note(text=problem.text))


def _pick_baseline(
    arguments: argparse.Namespace,
    machine: Machine,
    engine: EngineFound,
    current: str | None,
    ask: Asker | None,
    *,
    edit: bool = False,
) -> tuple[str | None, str]:
    """The baseline command (None for engine defaults) and the words for the Step-0 block."""
    if not edit:
        if current is not None:
            return current, current
        if arguments.engine_defaults or arguments.endpoint:
            return None, DEFAULTS_LABEL
    server = next((s for s in machine.running if s.engine == engine.name), None)
    running = f"{server.command}  (running, pid {server.pid})" if server else ""
    if ask is None:
        return (server.command, running) if server else (None, DEFAULTS_LABEL)
    if server:
        ask.write(f"A {ENGINE_LABELS.get(engine.name, engine.name)} server is running: {running}")
        key = ask.key(BASELINE_QUESTION, {"p": "paste another command", "d": "engine defaults"})
        if key == "":
            return server.command, running
    else:
        ask.write(NO_SERVER_LINE)
        key = ask.key(NO_SERVER_QUESTION, {"p": "paste the command you'd serve with"})
        if key == "":
            return None, DEFAULTS_LABEL
    if key == "d":
        return None, DEFAULTS_LABEL

    def understood(text: str) -> str | None:
        try:
            ENGINES[engine.name].parse_setup(text)
        except (ValueError, PreflightError) as error:
            return error.message if isinstance(error, PreflightError) else str(error)
        return None

    pasted = ask.line("Your command:", understood)
    return pasted, pasted


def _pick_model(
    arguments: argparse.Namespace,
    parsed: ParsedSetup | None,
    endpoint_model: Path | None,
    ask: Asker | None,
    current: Path | None = None,
) -> Path:
    """``--model``, else a local path the baseline command or endpoint names, else asked."""
    if current is not None and ask is not None:
        return _ask_model(ask, None, default=current)
    if arguments.model:
        return _checked(arguments.model, _model_problem, "--model")
    named = [c for c in (parsed.model if parsed else None, endpoint_model) if c]
    for candidate in named:
        if Path(candidate).expanduser().exists():
            return Path(candidate).expanduser().resolve()
    missing = str(named[0]) if parsed and parsed.model else None
    if ask is None:
        why = f": the model in your command, {missing}, isn't there" if missing else ""
        raise PreflightError(PROJECT_INPUTS_INVALID, f"--model is required{why}")
    return _ask_model(ask, missing)


def _block(engine: EngineFound, hardware: Hardware, model: Path, baseline: str) -> list[str]:
    try:
        described = model_line(model, hardware)
    except (OSError, ValueError, KeyError):
        described = str(model)
    try:
        consequence = fit_consequence(model, hardware)
    except (OSError, ValueError, KeyError):
        consequence = None
    return [
        f"Engine:   {_engine_line(engine)}",
        f"Hardware: {hardware.line()}",
        f"Model:    {described}",
        *([consequence] if consequence else []),
        f"Baseline: {baseline}",
    ]


def _answers(
    arguments: argparse.Namespace,
    ask: Asker | None,
    parsed: ParsedSetup | None,
    hardware: Hardware,
    baseline: Literal["command", "defaults"],
) -> Choices:
    prompts: Path | None = None
    if arguments.workload and arguments.prompts:
        raise PreflightError(PROJECT_INPUTS_INVALID, "pass --workload or --prompts, not both")
    if arguments.prompts:
        workload = "own"
        prompts = _checked(arguments.prompts, _prompts_problem, "--prompts")
    elif arguments.workload:
        workload = arguments.workload
    elif ask is not None:
        workload = WORKLOADS[ask.choice("Workload:", WORKLOAD_OPTIONS, 1) - 1]
        if workload == "own":
            prompts = ask.path("Your prompts file (JSONL):", _prompts_problem)
    else:
        workload = "chat"

    default_tools = tools_default(workload)
    if arguments.tools:
        tools = _tools_flag(arguments.tools)
    elif ask is not None:
        offered = 2 if default_tools == "default" else 1
        picked = ask.choice("Tool calls:", TOOL_OPTIONS, offered)
        tools = (
            "none"
            if picked == 1
            else "default"
            if picked == 2
            else str(ask.path("Your tools file (OpenAI format):", _tools_problem))
        )
    else:
        tools = default_tools

    settings = parsed.settings if parsed else None
    if arguments.concurrency is not None:
        concurrency = arguments.concurrency
    elif ask is not None:
        concurrency = ask.number(
            "Peak concurrent requests?", concurrency_default(settings, hardware.kind)
        )
    else:
        concurrency = concurrency_default(settings, hardware.kind)

    if arguments.goal:
        goal = arguments.goal
    elif ask is not None:
        goal = GOALS[ask.choice("Goal:", GOAL_OPTIONS, GOALS.index(GOAL_DEFAULT[workload]) + 1) - 1]
    else:
        goal = GOAL_DEFAULT[workload]
    return Choices(
        workload=workload,
        prompts=prompts,
        tools=tools,
        structured=bool(arguments.structured),
        concurrency=concurrency,
        goal=goal,
        baseline=baseline,
    )


def _tools_flag(value: str) -> str:
    if value in ("none", "default"):
        return value
    return str(_checked(Path(value), _tools_problem, "--tools"))


def plan_init(
    arguments: argparse.Namespace,
    machine: Machine,
    ask: Asker | None,
    *,
    current: str | None = None,
    endpoint_model: Path | None = None,
) -> InitPlan:
    """Everything ``init`` needs from the user. ``ask`` is None without a terminal: what a flag
    does not say then comes from what was detected, or from the defaults with ``--yes``."""
    say = _speaker(ask, bool(arguments.json) or JSON_OUTPUT.get())
    _engine_missing(arguments, machine, current)
    _format_mismatch(arguments, machine, _named_model(arguments, current), current=current)
    if not machine.engines:
        raise PreflightError(
            ENGINE_UNAVAILABLE,
            "No serving engine found. Install one for this machine: "
            f"{install_hint(machine.hardware)}",
        )
    if ask is None and not arguments.yes and (missing := missing_flags(arguments, machine)):
        raise PreflightError(
            PROJECT_INPUTS_INVALID,
            f"no terminal to ask on; pass {_joined(missing)} or --yes to take the defaults",
        )
    running = machine.running[0].command if machine.running else None
    engine = _pick_engine(arguments, machine, current or running, ask)
    told: set[str] = set()
    _check_engine(engine, machine.hardware, told)
    text, baseline = _pick_baseline(arguments, machine, engine, current, ask)
    model = _pick_model(arguments, _parse(engine.name, text), endpoint_model, ask)
    _format_mismatch(arguments, machine, model, engine.name)
    while True:
        for line in _block(engine, machine.hardware, model, baseline):
            say(line)
        if ask is None or ask.key(CONTINUE_QUESTION, {"e": "edit"}) == "":
            break
        engine = _pick_engine(arguments, machine, None, ask, engine)
        _check_engine(engine, machine.hardware, told)
        text, baseline = _pick_baseline(arguments, machine, engine, current, ask, edit=True)
        model = _pick_model(arguments, _parse(engine.name, text), endpoint_model, ask, model)
        _format_mismatch(arguments, machine, model, engine.name)
    parsed = _parse(engine.name, text)
    choices = _answers(arguments, ask, parsed, machine.hardware, "command" if text else "defaults")
    quiet = bool(arguments.json) or JSON_OUTPUT.get()
    run_now = arguments.run_now
    if run_now is None and arguments.yes and not quiet:
        run_now = True
    if ask is None and run_now is None:
        run_now = False
    return InitPlan(model, engine.name, text, choices, run_now)


def _precision(plan: InitPlan) -> tuple[str, str]:
    entry, _ = detect_format(plan.model).register(
        plan.model, engine_build="unspecified", engine_name=plan.engine
    )
    variant = entry.metadata.variants[0]
    return variant.weight_precision, variant.activation_dtype


def _place_inputs(project: Path, staged: tuple[Path, Path]) -> tuple[tuple[Path, Path], list[Path]]:
    """Move the staged inputs into ``project/inputs``. A file already there with the same bytes
    stays; one with other bytes is never overwritten. Returns the final paths and the files
    this call created."""
    inputs = project / INPUTS_DIR
    placed: list[Path] = []
    final = []
    try:
        inputs.mkdir(parents=True, exist_ok=True)
        for source in staged:
            target = inputs / source.name
            final.append(target)
            if target.exists():
                if target.read_bytes() != source.read_bytes():
                    raise PreflightError(
                        PROJECT_INPUTS_CHANGED,
                        f"{target} already exists with different contents; pass --force to "
                        "archive this project and start again, or choose another --project",
                    )
                continue
            shutil.copyfile(source, target)
            target.chmod(0o600)
            placed.append(target)
    except BaseException:
        for path in placed:
            path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            inputs.rmdir()
        raise
    return (final[0], final[1]), placed


def prepare_init(
    arguments: argparse.Namespace,
    *,
    current: str | None,
    endpoint_model: Path | None,
) -> Prepared:
    """Plan the registration, check what the plan describes, and only then touch the project:
    archive an old one (``--force``) and write the inputs the plan describes."""
    emit(PhaseStarted(phase="preflight"))
    machine = detect_machine()
    quiet = bool(arguments.json) or JSON_OUTPUT.get()
    interactive = sys.stdin.isatty() and sys.stderr.isatty() and not quiet and not arguments.yes
    ask = Asker() if interactive else None
    plan = plan_init(arguments, machine, ask, current=current, endpoint_model=endpoint_model)
    project: Path = arguments.project
    precision = _precision(plan)
    with tempfile.TemporaryDirectory() as staging:
        staged = write_inputs(
            Path(staging), plan.choices, precision=precision, thinking=thinks(plan.model)
        )
        if arguments.force and (archived := archive_existing_project(project, force=True)):
            emit(Note(text=f"archived the previous project as {archived}"))
        project.mkdir(mode=0o700, parents=True, exist_ok=True)
        (config, prompts), written = _place_inputs(project, staged)
    return Prepared(plan, project, config, prompts, tuple(written), machine.hardware, ask, quiet)
