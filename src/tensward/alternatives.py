"""Beyond the flags: another file of the same model, or another engine, that did better on a goal
in Tensward's own runs on the same GPU class, model family and workload shape. Tensward never
switches to one; the user measures it on their machine as a new project. An alternative is shown
only when every check holds, so it is never offered where it was not measured to help."""

from __future__ import annotations

import json
import math
import re
import shlex
import shutil
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from .engines.vllm.command import TOOL_CALLING_ON, VLLM_VERSION
from .errors import PreflightError
from .fit import MARGIN
from .formats.gguf import read_gguf

ALTERNATIVES_FILE = "data/alternatives.toml"
PLACEHOLDER = "WAVE-ALT"  # a value still to be measured: its record is never shown
KINDS = ("file", "engine")
WORKLOADS = ("agent", "chat")
# The measured changes a record may carry, in percent, and whether lower is better.
LOWER_IS_BETTER = {
    "tpot_p50": True, "tpot_p95": True, "ttft_p50": True, "ttft_p95": True, "req_s": False,
}  # fmt: skip
# The metric the pre-registered rule decides each goal on.
DECIDES_ON = {"tpot": "tpot_p50", "ttft": "ttft_p50", "throughput": "req_s"}
METRIC_WORDS = {
    "tpot_p50": "time per token",
    "tpot_p95": "slowest time per token",
    "ttft_p50": "first word",
    "ttft_p95": "slowest first word",
    "req_s": "requests per second",
}
REPORT_WORDS = {
    "tpot_p50": "TPOT p50",
    "tpot_p95": "TPOT p95",
    "ttft_p50": "TTFT p50",
    "ttft_p95": "TTFT p95",
    "req_s": "requests per second",
}
ENGINE_WORDS = {"vllm": ("vLLM", "vllm"), "llamacpp": ("llama.cpp", "llama-server")}
ENGINE_PLATFORMS = {"vllm": ("nvidia",), "llamacpp": ("nvidia", "apple", "cpu")}
INSTALL_COMMANDS = {"vllm": f"uv tool install vllm=={VLLM_VERSION}"}
INSTALL_README = "see 'Don't have an engine yet?' in the README"
# What the engine needs beyond its install, as README "Don't have an engine yet?" has it: vLLM
# compiles kernels when it starts.
INSTALL_NOTES = {
    "vllm": "sudo apt install -y cuda-toolkit-13-0 build-essential ninja-build python3-dev"
}
UV_INSTALL = "curl -LsSf https://astral.sh/uv/install.sh | sh"
HF_INSTALL = "uv tool install huggingface_hub"
MODEL_FLAGS = ("-m", "--model")
YOUR_FLAGS = "<your usual flags>"
SECRET_NOTE = "fill in the values shown as *** (Tensward does not keep secrets)"
MIN_RUNS = 2  # valid runs per side a record needs
MIN_CHANGE = 3.0  # percent: a smaller change is never shown, whatever its A/A range
NOISE_TIMES = 2.0  # a change counts when it is beyond this many times the A/A range
ANSWER_SE = 2.0  # answers are within the usual variation down to this many standard errors
MODEL_PREFIX = 6  # characters a per-model folder's name shares with its model file's name
CONTEXT_STEP = 1024  # a request's context is rounded up to this many tokens
MIN_CONTEXT = 4096
GIB = 2**30
DECIDES = (
    "Answers differ when the file or engine differs; your own measurement decides: the new "
    "project's analyse and its answer check."
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Alternative:
    name: str
    kind: Literal["file", "engine"]
    label: str  # "the IQ4_XS file", "vLLM with the int4 checkpoint"
    engine: str  # the engine it runs on
    quant: str  # the file or checkpoint type ("IQ4_XS", "int4")
    repo: str  # the Hugging Face repository it downloads from
    file: str  # kind "file": the file, which takes the place of the user's model file
    folder: str  # kind "engine": the checkpoint's folder, beside the user's model
    revision: str  # the repository commit the measured file or checkpoint came from
    command: str  # kind "engine": the measured command; {checkpoint} and {concurrency} filled in
    context: int  # kind "engine": the context it was measured with; 0 when the command sets none
    tool_parser: str  # kind "engine": the tool-call parser, for a workload that offers tools
    weight_gib: float
    overhead_gib: float  # what the engine used beyond the weights and the KV cache
    models: tuple[str, ...]  # the model identities it was measured on (``model_identity``)
    family: str
    params_b: tuple[float, float]
    gpus: tuple[str, ...]
    workload: Literal["agent", "chat"]
    concurrency: tuple[int, int]
    from_engine: str
    from_files: tuple[str, ...]  # the user's file types it was measured against
    model: str  # the model it was measured on, as people name it
    gpu: str  # the GPU it was measured on
    runs: int  # valid runs on each side
    same_sign: bool  # the decision metric moved the same way in every round
    changes: Mapping[str, float]  # percent, against the setup it replaces
    noise: Mapping[str, float]  # percent: the range between two runs of that setup
    identical: int  # answers identical to the setup it replaces
    of: int
    floor: int  # answers identical between two runs of that setup (the A/A floor)
    broken: int  # its broken answers: empty, repeating or with an invalid tool call
    base_broken: int  # those of the setup it replaces

    def refusal(self, facts: Mapping[str, object], goal: str) -> str | None:
        """Why it is not shown for this setup and goal; None when every check holds."""
        params, clients = facts.get("model.params_b"), facts.get("workload.clients")
        shape = "agent" if facts.get("workload.tools") else "chat"
        identity = facts.get("model.identity")
        checks = (
            (bool(identity) and identity in self.models, "another or an unknown model"),
            (_family(facts.get("model.family")) == _family(self.family), "another model family"),
            (isinstance(params, (int, float)) and _within(params, self.params_b), "another size"),
            (facts.get("gpu.model") in self.gpus and facts.get("gpu.count") == 1, "another GPU"),
            (shape == self.workload, "another workload shape"),
            (isinstance(clients, int) and _within(clients, self.concurrency), "concurrency"),
            (facts.get("engine.name") == self.from_engine, "another engine runs now"),
            (not self.from_files or facts.get("model.ftype") in self.from_files, "another file"),
            (not self.already_run(facts), "already what runs"),
            (facts.get("platform.name") in ENGINE_PLATFORMS[self.engine], "unsupported here"),
            (self.serves_tools(facts), "no tool-call parser for a workload with tools"),
            (self.holds_context(facts), "a longer context than measured"),
            (self.passes(goal), "not measured to help this goal"),
            (self.fits(facts), "does not fit"),
        )
        return next((why for holds, why in checks if not holds), None)

    def serves_tools(self, facts: Mapping[str, object]) -> bool:
        """Its command can serve tool calls when the workload offers tools."""
        tools = bool(facts.get("workload.tools"))
        return not tools or self.kind == "file" or self.engine != "vllm" or bool(self.tool_parser)

    def holds_context(self, facts: Mapping[str, object]) -> bool:
        """The workload's requests fit the context the alternative was measured with."""
        needed = needed_context(facts)
        return not self.context or (needed is not None and needed <= self.context)

    def already_run(self, facts: Mapping[str, object]) -> bool:
        if self.kind == "engine":
            return facts.get("engine.name") == self.engine
        return _family(facts.get("model.ftype")) == _family(self.quant)

    def passes(self, goal: str) -> bool:
        """The pre-registered rule on the goal's deciding metric: at least ``MIN_RUNS`` valid
        runs per side, the same sign every round, a change in the goal's direction beyond
        ``MIN_CHANGE`` and ``NOISE_TIMES`` the A/A range; and no more broken answers than the
        setup it replaces."""
        metric = DECIDES_ON[goal]
        change, noise = self.changes.get(metric), self.noise.get(metric)
        if change is None or noise is None or self.runs < MIN_RUNS or not self.same_sign:
            return False
        better = change < 0 if LOWER_IS_BETTER[metric] else change > 0
        beyond = abs(change) > max(MIN_CHANGE, NOISE_TIMES * noise)
        return better and beyond and self.broken <= self.base_broken

    def fits(self, facts: Mapping[str, object]) -> bool:
        """Its weights, the engine's overhead and the KV cache of this workload's requests fit
        the GPU's memory, within the fit estimate's margin."""
        memory, per_token = facts.get("gpu.memory_gib"), facts.get("anatomy.kv_bytes_per_token")
        clients, context = facts.get("workload.clients"), needed_context(facts)
        if not isinstance(memory, (int, float)) or not isinstance(per_token, int):
            return False
        if not isinstance(clients, int) or context is None:
            return False
        need = (self.weight_gib + self.overhead_gib) * GIB + per_token * clients * context
        return need <= memory * GIB * (1 - MARGIN)

    def answers_within(self) -> bool:
        """Answers as close to the replaced setup's as two of its own runs, within
        ``ANSWER_SE`` standard errors of the A/A share."""
        share = self.floor / self.of
        return self.identical >= self.floor - ANSWER_SE * math.sqrt(self.of * share * (1 - share))

    def gain(self, goal: str) -> float:
        return abs(self.changes[DECIDES_ON[goal]])


def _family(name: object) -> str:
    """A model family or file type as one spelling: ``lfm2_moe``, ``LFM2-MoE`` and ``lfm2moe``
    are one family."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower()) if name is not None else ""


def model_identity(model: Path) -> str | None:
    """What model this is, as one spelling, from the checkpoint's own metadata; None when it
    does not say clearly. A GGUF file: its basename, size label and fine-tune, and its name
    must say no more than those. A checkpoint folder: config.json's ``_name_or_path``, which the
    folder's name must contain."""
    if model.is_dir():
        try:
            config = json.loads((model / "config.json").read_text("utf-8"))
        except (OSError, ValueError):
            return None
        source = config.get("_name_or_path") if isinstance(config, dict) else None
        tail = _family(str(source).rstrip("/").rsplit("/", 1)[-1]) if source else ""
        return tail if tail and tail in _family(model.name) else None
    try:
        metadata = read_gguf(model).metadata
    except (OSError, PreflightError):
        return None
    name, basename = metadata.get("general.name"), metadata.get("general.basename")
    size, tune = metadata.get("general.size_label"), metadata.get("general.finetune")
    key = _family(basename or name)
    if size and not key.endswith(_family(size)):
        key += _family(size)
    key += _family(tune)
    return key if key and (not name or _family(name) == key) else None


def _within(value: float, band: tuple[float, float]) -> bool:
    return band[0] <= value <= band[1]


def needed_context(facts: Mapping[str, object]) -> int | None:
    """The context one request of this workload needs, rounded up to ``CONTEXT_STEP``."""
    prompt = facts.get("workload.prompt_tokens_max")
    output = facts.get("workload.output_tokens_limit")
    if not isinstance(prompt, int) or not isinstance(output, int):
        return None
    return max(MIN_CONTEXT, math.ceil((prompt + output) / CONTEXT_STEP) * CONTEXT_STEP)


def parse(text: str) -> tuple[Alternative, ...]:
    """The ``[[alternative]]`` tables of ``text`` that hold no placeholder; ValueError names the
    first bad one."""
    rows = tomllib.loads(text).get("alternative", [])
    return tuple(_alternative(row) for row in rows if PLACEHOLDER not in repr(row))


@cache
def table() -> tuple[Alternative, ...]:
    """The shipped alternatives; a table that cannot be read shows none."""
    try:
        return parse(files("tensward").joinpath(ALTERNATIVES_FILE).read_text("utf-8"))
    except (OSError, ValueError):
        return ()


def _alternative(row: Mapping[str, Any]) -> Alternative:
    name = row.get("name", "?")
    try:
        match, measured, answers = row["match"], row["measured"], row["answers"]
        a = Alternative(
            name=row["name"],
            kind=row["kind"],
            label=row["label"],
            engine=row["engine"],
            quant=row["quant"],
            repo=row["repo"],
            revision=row["revision"],
            file=row.get("file", ""),
            folder=row.get("folder", ""),
            command=row.get("command", ""),
            context=int(row.get("context", 0)),
            tool_parser=row.get("tool_parser", ""),
            weight_gib=float(row["weight_gib"]),
            overhead_gib=float(row["overhead_gib"]),
            models=tuple(_family(m) for m in match["models"]),
            family=match["family"],
            params_b=(float(match["params_b"][0]), float(match["params_b"][1])),
            gpus=tuple(match["gpus"]),
            workload=match["workload"],
            concurrency=(int(match["concurrency"][0]), int(match["concurrency"][1])),
            from_engine=match["engine"],
            from_files=tuple(match.get("files", ())),
            model=measured["model"],
            gpu=measured["gpu"],
            runs=int(measured["runs"]),
            same_sign=measured["same_sign"] is True,
            changes={k: float(measured[k]) for k in LOWER_IS_BETTER if k in measured},
            noise={k: float(v) for k, v in measured["noise"].items()},
            identical=int(answers["identical"]),
            of=int(answers["of"]),
            floor=int(answers["floor"]),
            broken=int(answers["broken"]),
            base_broken=int(answers["base_broken"]),
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f"alternative {name}: {error!r}") from None
    checks = (
        (a.kind in KINDS, f"kind {a.kind!r}"),
        (a.workload in WORKLOADS, f"workload {a.workload!r}"),
        (a.engine in ENGINE_WORDS and a.from_engine in ENGINE_WORDS, "an unknown engine"),
        (a.kind != "file" or (bool(a.file) and a.engine == a.from_engine), "a file names no file"),
        (a.kind != "engine" or (bool(a.folder) and "{checkpoint}" in a.command), "no command"),
        (bool(a.models) and bool(re.fullmatch(r"[0-9a-f]{40}", a.revision)), "no identity"),
        (bool(a.changes) and set(a.noise) <= set(LOWER_IS_BETTER), "no measured change"),
        (0 < a.of and 0 <= a.identical <= a.of and 0 <= a.floor <= a.of, "answers out of range"),
    )
    if problems := [said for holds, said in checks if not holds]:
        raise ValueError(f"alternative {name}: {'; '.join(problems)}")
    return a


def matching(facts: Mapping[str, object], goal: str) -> list[Alternative]:
    """The alternatives that pass every check for this setup and ``goal``, the largest gain
    first."""
    found = [a for a in table() if a.refusal(facts, goal) is None]
    return sorted(found, key=lambda a: -a.gain(goal))


def serve_command(
    alternative: Alternative, facts: Mapping[str, object], current: str | None, model: Path
) -> tuple[str, bool]:
    """The command that serves ``alternative`` here, and whether it runs as it stands. For a
    file: the user's own command with only its model path replaced, every other flag kept; when
    that path cannot be found safely, the file and ``YOUR_FLAGS`` for the user to fill in. For
    an engine: the measured command with the checkpoint beside the user's model, their
    concurrency and, when the workload offers tools, tool calling with the record's parser."""
    folder = model.parent
    if alternative.kind == "engine":
        command = alternative.command.replace(
            "{checkpoint}", shlex.quote(str(checkpoint_dir(alternative, model)))
        ).replace("{concurrency}", str(facts.get("workload.clients") or 1))
        tools = facts.get("workload.tools") and alternative.tool_parser
        if tools and TOOL_CALLING_ON not in shlex.split(command):
            command += f" {TOOL_CALLING_ON} --tool-call-parser {alternative.tool_parser}"
        return command, True
    swapped = str(folder / alternative.file)
    replaced = _with_model(current, model, swapped)
    if replaced is None:
        program = ENGINE_WORDS[alternative.engine][1]
        return f"{shlex.join([program, '-m', swapped])} {YOUR_FLAGS}", False
    return replaced, True


def _with_model(current: str | None, model: Path, swapped: str) -> str | None:
    """``current`` with the value of its one model flag (``-m X``, ``--model X``,
    ``--model=X``) replaced by ``swapped``; None when it has no such flag naming ``model``'s
    file, or more than one."""
    try:
        words = shlex.split(current) if current else []
    except ValueError:
        return None
    found = []
    for at, word in enumerate(words):
        flag, given, value = word.partition("=")
        if given and flag in MODEL_FLAGS:
            found.append((at, f"{flag}={swapped}", value))
        elif word in MODEL_FLAGS and at + 1 < len(words):
            found.append((at + 1, swapped, words[at + 1]))
    if len(found) != 1 or Path(found[0][2]).name != model.name:
        return None
    at, new, _ = found[0]
    return shlex.join([*words[:at], new, *words[at + 1 :]])


def download_command(alternative: Alternative, model: Path) -> str:
    """``hf download`` at the measured revision: the file into the user's model folder, or the
    checkpoint into a folder beside it."""
    folder = model.parent
    pinned = ["--revision", alternative.revision]
    if alternative.kind == "file":
        return shlex.join(["hf", "download", alternative.repo, alternative.file, *pinned,
                           "--local-dir", str(folder)])  # fmt: skip
    target = str(checkpoint_dir(alternative, model))
    return shlex.join(["hf", "download", alternative.repo, *pinned, "--local-dir", target])


def checkpoint_dir(alternative: Alternative, model: Path) -> Path:
    """Where an engine alternative's checkpoint goes: beside the user's model folders, the
    parent of a GGUF's folder named after its model, else ``~/models``."""
    folder = model.parent
    if model.is_dir():
        return folder / alternative.folder
    own = _family(folder.name)[:MODEL_PREFIX] == _family(model.stem)[:MODEL_PREFIX]
    root = folder.parent if own and len(_family(folder.name)) >= MODEL_PREFIX else None
    return (root or Path.home() / "models") / alternative.folder


def init_command(command: str, like: Path) -> str:
    """``tensward init --current "<command>" --like <project>``, quoted for a shell."""
    tail = f"--like {shlex.quote(str(like))}"
    if any(char in command for char in '"$`\\!'):
        return f"tensward init --current {shlex.quote(command)} {tail}"
    return f'tensward init --current "{command}" {tail}'


def new_folder(alternative: Alternative) -> str:
    """ "~/tensward-iq4xs": the folder the new project goes in."""
    name = _family(alternative.quant)
    return (
        f"~/tensward-{name}"
        if alternative.kind == "file"
        else f"~/tensward-{alternative.engine}-{name}"
    )


def on_path(program: str) -> bool:
    return shutil.which(program) is not None


def installed(engine: str) -> bool:
    """Whether the engine's server program is on this machine's PATH."""
    return on_path(ENGINE_WORDS[engine][1])


def downloaded(alternative: Alternative, model: Path) -> bool:
    if alternative.kind == "file":
        return (model.parent / alternative.file).exists()
    return checkpoint_dir(alternative, model).exists()


def steps(
    alternative: Alternative,
    facts: Mapping[str, object],
    current: str | None,
    model: Path,
    project: Path,
) -> tuple[list[tuple[str, str]], list[str]]:
    """The ``Next:`` choices that measure ``alternative`` here, and every step as the screen
    prints it, in order: installing uv, the engine and hf when they are missing, downloading
    the file or checkpoint when it is not here, a new folder, then ``tensward init --like``
    with the user's project, so the new project measures the same workload."""
    label = ENGINE_WORDS[alternative.engine][0]
    choices: list[tuple[str, str]] = []
    screen = []
    engine = alternative.kind == "engine" and not installed(alternative.engine)
    install = INSTALL_COMMANDS.get(alternative.engine) if engine else None
    fetch = not downloaded(alternative, model)
    if (install or (fetch and not on_path("hf"))) and not on_path("uv"):
        choices.append((UV_INSTALL, "install uv, then open a new shell"))
        screen.append(UV_INSTALL)
    if engine:
        if install is not None:
            choices.append((install, f"install {label}, which is not installed here"))
            screen.append(install)
        else:
            screen.append(f"install {label} ({INSTALL_README})")
        if note := INSTALL_NOTES.get(alternative.engine):
            screen.append(note)
    if fetch:
        if not on_path("hf"):
            choices.append((HF_INSTALL, "install hf, the Hugging Face download tool"))
            screen.append(HF_INSTALL)
        what = "file" if alternative.kind == "file" else "checkpoint"
        download = download_command(alternative, model)
        choices.append((download, f"get the {what} ({alternative.weight_gib:.1f} GiB)"))
        screen.append(f"get the {what} first: {download}")
    folder = new_folder(alternative)
    choices.append((f"mkdir -p {folder} && cd {folder}", "a new folder for the new project"))
    screen.append(choices[-1][0])
    command, runs = serve_command(alternative, facts, current, model)
    init = init_command(command, project)
    fill = "" if runs else f", with {YOUR_FLAGS} replaced by your own"
    if "***" in command:
        fill += ", with the values shown as *** filled in"
    choices.append((init, f"then measure {alternative.label} on your machine{fill}"))
    screen.append(init)
    return choices, screen


def option(alternative: Alternative) -> str:
    """ "vLLM, int4 checkpoint", "IQ4_XS file": the table's name for it."""
    if alternative.kind == "file":
        return f"{alternative.quant} file"
    return f"{ENGINE_WORDS[alternative.engine][0]}, {alternative.quant} checkpoint"


def short_name(alternative: Alternative) -> str:
    """ "vLLM int4", "IQ4_XS"."""
    if alternative.kind == "file":
        return alternative.quant
    return f"{ENGINE_WORDS[alternative.engine][0]} {alternative.quant}"


def wrapped(text: str, indent: str, width: int = 80) -> list[str]:
    """``text``, a shell command or a line holding one, over lines of at most ``width``: never
    broken inside single quotes, each line but the last ending in " \\" so the shell reads one
    command; a word longer than a line stays whole."""
    words = _words(text)
    lines: list[str] = []
    line = ""
    under = " " * (len(indent) + 2)
    for word in words:
        lead = indent if not lines else under
        if line and len(lead) + len(line) + 1 + len(word) + 2 > width:
            lines.append(f"{lead}{line} \\")
            line = word
        else:
            line = f"{line} {word}" if line else word
    lines.append(f"{indent if not lines else under}{line}")
    return lines


def _words(text: str) -> list[str]:
    """``text`` split at the spaces a line may break at: outside quotes, or inside double quotes
    but outside the single quotes within them, where the shell drops a " \\" line break."""
    words: list[str] = []
    word, quote, inner, escaped = "", "", False, False
    for char in text:
        literal, escaped = escaped, False
        if literal:
            pass
        elif char == "\\" and quote != "'":
            escaped = True
        elif quote == '"' and char == "'":
            inner = not inner
        elif quote:
            quote = "" if char == quote else quote
        elif char in "'\"":
            quote = char
        if char == " " and not literal and not inner and quote != "'":
            words.append(word)
            word = ""
        else:
            word += char
    words.append(word)
    return words


def percent(value: float) -> str:
    """ "−17%", "+5%"."""
    return f"{'−' if value < 0 else '+'}{abs(value):.0f}%"


def costs(alternative: Alternative, goal: str) -> list[str]:
    """The measured changes other than the deciding one that got worse beyond their A/A
    range."""
    return [
        f"{METRIC_WORDS[metric]} {percent(change)}"
        for metric, change in alternative.changes.items()
        if metric != DECIDES_ON[goal]
        and (change > 0 if LOWER_IS_BETTER[metric] else change < 0)
        and abs(change) > NOISE_TIMES * alternative.noise.get(metric, 0.0)
    ]


def answers_words(alternative: Alternative, *, short: bool) -> str:
    if alternative.answers_within():
        return "answers within the usual variation"
    if short:
        return "answers differ more than between identical runs: compare them in the new project"
    return (
        f"answers differ more than between identical runs: {alternative.identical} of "
        f"{alternative.of} identical, against {alternative.floor} of {alternative.of} between two "
        "runs of the same setup; compare them in the new project"
    )


def heading(alternative: Alternative) -> str:
    """ "Beyond flags: measured by us on an NVIDIA L4 with Qwen3.8-27B (agent workload)"."""
    return (
        f"Beyond flags: measured by us on an {alternative.gpu} with {alternative.model} "
        f"({alternative.workload} workload)"
    )


def table_row(at: int, alternative: Alternative, goal: str) -> tuple[str, str, str, str]:
    """One row of the terminal's table: number, option, the goal's change with what got worse,
    and the answers."""
    change = percent(alternative.changes[DECIDES_ON[goal]])
    worse = costs(alternative, goal)
    moved = f"{change} ({', '.join(worse)})" if worse else change
    answers = (
        "within the usual variation"
        if alternative.answers_within()
        else "differ: check them in the new project"
    )
    return str(at), option(alternative), moved, answers


def evidence_lines(alternative: Alternative) -> list[str]:
    """What report.md says of one alternative: every measured change, its A/A range and the
    answers."""
    moves = ", ".join(
        f"{REPORT_WORDS[m]} {percent(c)}"
        + (f" (A/A range {alternative.noise[m]:.1f}%)" if m in alternative.noise else "")
        for m, c in alternative.changes.items()
    )
    low, high = alternative.concurrency
    band = f"{low}" if low == high else f"{low} to {high}"
    against = ENGINE_WORDS[alternative.from_engine][0]
    if alternative.from_files:
        against += f" {' / '.join(alternative.from_files)}"
    return [
        f"- {alternative.label} ({alternative.kind}): {moves}; "
        f"{answers_words(alternative, short=False)}",
        f"  measured by us: {alternative.model} on an {alternative.gpu}, {alternative.workload} "
        f"workload at {band} clients, {alternative.runs} valid runs per side, against {against}",
        f"  broken answers (empty, repeating or an invalid tool call): {alternative.broken}, "
        f"against {alternative.base_broken}",
    ]
