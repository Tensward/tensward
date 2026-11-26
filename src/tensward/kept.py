"""The setup the user kept with ``tensward set``: the engine arguments of one run, used by plain
``analyse`` and ``serve start --from current`` in place of the registered setup's."""

from __future__ import annotations

import json
import shlex
import time
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .compare import reasoning_same
from .errors import PROJECT_RECORD_INVALID, PreflightError
from .files import write_private
from .levers import LEVERS
from .progress import Note, emit

if TYPE_CHECKING:
    from .engines import Engine
    from .settings import Settings

KEPT_FILE = "kept.json"
REGISTERED_FILE = "registered.json"
REGISTERED = "registered"
LAST = "last"
KEEP_LATEST = "latest"
OTHER_REGISTRATION = (
    "run {run} was made from another registration (checkpoint, configuration, workload or "
    "registered setup)"
)
ACCEPT_FLAG = "--accept-answer-changes"
# The levers whose change can change answers, in words.
ANSWER_LEVERS = {
    "quantization": "the weight quantization",
    "kv_cache_precision": "the KV-cache precision",
    "speculation": "speculative decoding",
    "compact_json": "compact JSON",
}


@dataclass(frozen=True, slots=True)
class Kept:
    run: str  # the run whose settings are kept
    engine_args: tuple[str, ...]  # its --engine-arg values, as run.json recorded them
    # levers that differ from the registered setup and can change answers
    answer_changes: tuple[str, ...]
    previous: str  # the run kept before, or REGISTERED
    snapshot_id: str  # the registration it was kept under
    max_req_s_loss: float | None = None  # the floor its run declared, as a share (0.2 for 20%)
    env: tuple[str, ...] = ()  # its engine environment as NAME=VALUE, as run.json recorded it
    history: tuple[dict[str, Any], ...] = ()  # the setups kept before it, oldest first


def read_kept(project: Path) -> Kept | None:
    """The kept setup; None when nothing is kept, or when it was kept under another
    registration (then a Note says it is ignored)."""
    path = project / KEPT_FILE
    try:
        document = json.loads(path.read_text("utf-8"))
        kept = Kept(
            run=document["run"],
            engine_args=tuple(document["engine_args"]),
            answer_changes=tuple(document["answer_changes"]),
            previous=document["previous"],
            snapshot_id=document["snapshot_id"],
            max_req_s_loss=document.get("max_req_s_loss"),
            env=tuple(document.get("env", ())),
            history=tuple(document.get("kept_history", ())),
        )
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError):
        raise PreflightError(
            PROJECT_RECORD_INVALID,
            f"{path} is not readable; remove it to go back to the registered setup",
        ) from None
    try:
        registered = json.loads((project / "project.json").read_text("utf-8")).get("snapshot_id")
    except (OSError, ValueError):
        registered = None
    if kept.snapshot_id != registered:
        emit(Note(text=(
            f"{path} was kept under another registration of this project; ignoring it; "
            f"{shlex.join(['tensward', 'set', REGISTERED, '--project', str(project)])} removes it"
        )))  # fmt: skip
        return None
    return kept


def went_back(project: Path) -> bool:
    """Whether the user went back to the registered setup with ``tensward set`` under the
    current registration."""
    try:
        chosen = json.loads((project / REGISTERED_FILE).read_text("utf-8")).get("snapshot_id")
        registered = json.loads((project / "project.json").read_text("utf-8")).get("snapshot_id")
    except (OSError, ValueError, AttributeError):
        return False
    return chosen is not None and chosen == registered


def _go_back(project: Path, snapshot_id: str, left: Kept | None = None) -> None:
    (project / KEPT_FILE).unlink(missing_ok=True)
    document: dict[str, Any] = {"snapshot_id": snapshot_id}
    if left is not None:
        document["left"] = {**_entry(left), "kept_history": list(left.history)}
    write_private(project / REGISTERED_FILE, json.dumps(document) + "\n")


class NotKept(Exception):
    """A run that cannot be kept; ``str(error)`` is the one line shown to the user.

    ``command`` is the ``tensward set`` command that keeps it anyway, when one exists.
    """

    def __init__(self, message: str, command: str | None = None) -> None:
        super().__init__(message)
        self.command = command


def answer_changes(
    engine: Engine, registered: Settings, settings: Settings, *, checkpoint_quantized: bool
) -> tuple[str, ...]:
    """The levers that differ between the two settings and can change answers. Quantization
    methods are compared by family; on a pre-quantized checkpoint, going from a forced method
    back to the engine's own choice is a kernel choice and does not count."""

    def value(settings: Settings, name: str) -> object:
        found = engine.lever_value(settings, name)
        return engine.quantization_family(found) if name == "quantization" and found else found

    return tuple(
        name for name, lever in LEVERS.items()
        if lever.changes_answers
        and value(registered, name) != value(settings, name)
        and not (name == "quantization" and value(settings, name) is None and checkpoint_quantized)
    )  # fmt: skip


def changed_words(
    engine: Engine,
    registered: Settings,
    engine_args: tuple[str, ...],
    changes: tuple[str, ...],
    *,
    checkpoint_quantized: bool,
) -> str:
    """``changes`` in words, each with the engine arguments that make it, as the user typed them:
    "the KV-cache precision (kv-cache-dtype=fp8_e4m3)"."""

    def typed(lever: str) -> str:
        made = [
            arg for arg in engine_args
            if lever in answer_changes(engine, registered, engine.with_engine_arg(registered, arg),
                                       checkpoint_quantized=checkpoint_quantized)
        ]  # fmt: skip
        return f"{ANSWER_LEVERS[lever]} ({', '.join(made)})" if made else ANSWER_LEVERS[lever]

    return ", ".join(typed(lever) for lever in changes)


def latest_run(project: Path) -> str | None:
    """The newest run of the project with a run.json and metrics.json (a step stopped in the
    load has no metrics.json); None when there is none."""
    runs = project.expanduser() / "runs"
    found = []
    for path in runs.iterdir() if runs.is_dir() else ():
        if (path / "run.json").is_file() and (path / "metrics.json").is_file():
            found.append(path.name)
    return max(found, default=None)


PARTIAL = "requests.partial.jsonl"
PARTIAL_COUNT = "requests.partial.json"  # {"finished": N, "planned": M}


@dataclass(frozen=True, slots=True)
class PartialRun:
    """A run stopped before its window ended: no run.json, its finished requests kept."""

    run: str
    finished: int
    planned: int | None

    def said(self) -> str:
        of = f" of {self.planned}" if self.planned else ""
        return f"stopped after {self.finished}{of} requests (partial, not reported)"


def partial_runs(project: Path) -> list[PartialRun]:
    """The project's stopped runs, oldest first."""
    runs = project.expanduser() / "runs"
    found = []
    for path in sorted(runs.iterdir()) if runs.is_dir() else ():
        if (path / "run.json").is_file() or not (path / PARTIAL).is_file():
            continue
        try:
            counted = json.loads((path / PARTIAL_COUNT).read_text("utf-8"))
            finished, planned = int(counted["finished"]), int(counted["planned"])
        except (OSError, ValueError, KeyError, TypeError):
            try:
                text = (path / PARTIAL).read_text("utf-8")
            except OSError:
                continue
            finished, planned = sum(1 for line in text.splitlines() if line.strip()), None
        found.append(PartialRun(path.name, finished, planned))
    return found


def newest_run(project: Path, refused: type[Exception]) -> str:
    """The newest run's id; raises ``refused`` when the project has no run."""
    newest = latest_run(project)
    if newest is None:
        raise refused(f"no run in {project.expanduser() / 'runs'}")
    return newest


def recorded_run(project: Path, run: str, refused: type[Exception]) -> Path:
    """The directory of ``run`` under the project's runs/; raises ``refused`` when there is no
    such complete run there."""
    from .runs import RUN_RECORD

    runs = project.expanduser() / "runs"
    run_dir = runs / run
    if run_dir.resolve().parent != runs.resolve() or not (run_dir / RUN_RECORD).is_file():
        raise refused(f"no run {run} in {runs}")
    return run_dir


def step_block(run_dir: Path) -> dict[str, Any] | None:
    """The step block of the run's report.json: the run against the one it was made from; None
    when there is none."""
    try:
        report = json.loads((run_dir / "report.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None
    for section in report["sections"]:
        for block in section["blocks"]:
            if block.get("kind") == "data" and block.get("key") == "step":
                return block  # type: ignore[no-any-return]
    return None


def keep(project: Path, run: str, *, accept_unreviewed: bool) -> tuple[Kept | None, Kept | None]:
    """Keep ``run``'s settings (``LAST``: the newest run's; ``REGISTERED``: go back to the
    registered setup); returns the kept setup now and the one before. Raises NotKept for a run
    that cannot be kept."""
    # Imported here: context imports this module, and these modules import context.
    from .history import recorded_env
    from .project import load_project, project_engine, settings_for, workload_facts
    from .runs import RUN_RECORD, RunFile, RunMetrics
    from .settings import Settings

    project = project.expanduser()
    resolved = load_project(project)
    if run == LAST:
        run = newest_run(project, NotKept)
    before = read_kept(project)
    if before is not None and run == before.run:
        return before, before
    if run == REGISTERED:
        _go_back(project, resolved.record.snapshot_id, before)
        return None, before
    run_dir = recorded_run(project, run, NotKept)
    run_file = RunFile.read(run_dir)
    if run_file.snapshot_id != resolved.record.snapshot_id:
        raise NotKept(f"{OTHER_REGISTRATION.format(run=run)}, so it cannot be this project's setup")
    metrics = run_dir / "metrics.json"
    if not metrics.is_file() or not RunMetrics.read(metrics).succeeded:
        raise NotKept(f"run {run} served no request")
    env = tuple(
        sorted(
            recorded_env(
                run_file.model_dump(), project_engine(resolved.record.current_setup).flag_rules
            )
        )
    )
    if not (run_file.engine_args or env):
        _go_back(project, resolved.record.snapshot_id, before)
        return None, before
    engine = project_engine(resolved.record.current_setup)
    registered = settings_for(resolved, engine, ())
    quantized = workload_facts(resolved).quantization is not None
    changes = answer_changes(
        engine, registered, Settings.from_json(run_file.settings), checkpoint_quantized=quantized
    )
    if not accept_unreviewed:
        from .compare import answers_pass

        step = step_block(run_dir)
        answers = step.get("answers") if step else None
        compare = f'compare them in {run_dir / "report.md"} ("Answers before / after")'
        command = set_command(project, run, unreviewed=True)
        if answers is not None and not answers_pass(answers):
            raise NotKept(
                f"run {run}'s {_refused(answers)} against the run it was made from: {compare}; "
                f"if they are acceptable, keep it with:\n  {command}",
                command,
            )
        if changes and not reasoning_same(answers):
            words = changed_words(
                engine, registered, run_file.engine_args, changes, checkpoint_quantized=quantized
            )
            raise NotKept(
                f"run {run} changes {words}, which can change answers: {compare}; if they are "
                f"acceptable, keep it with:\n  {command}",
                command,
            )
    analysed = json.loads((run_dir / RUN_RECORD).read_text("utf-8")).get("analysis") or {}
    now = Kept(
        run=run,
        engine_args=run_file.engine_args,
        answer_changes=changes,
        previous=before.run if before else REGISTERED,
        snapshot_id=run_file.snapshot_id,
        max_req_s_loss=analysed.get("max_req_s_loss"),
        env=env,
    )
    if before is not None:
        now = replace(now, history=(*before.history, _entry(before)))
    _write_kept(project, now)
    return now, before


def _refused(answers: Mapping[str, Any]) -> str:
    """Why a step's answers do not pass, as the end of "run X's ..."."""
    verdict = (answers.get("reasoning_check") or {}).get("verdict")
    if (answers.get("answer_check") or {}).get("verdict") == "changed":
        return "answers hold words your current setup would not write (the Fork check)"
    if verdict == "changed":
        return "reasoning departs from what your current setup writes (the reasoning check)"
    if verdict == "same":
        return (
            "reasoning matches your current setup's in what was produced, but its change can "
            "matter later in long answers (the reasoning check)"
        )
    return "answers did not pass the answer comparison"


def _entry(kept: Kept) -> dict[str, Any]:
    return {key: value for key, value in asdict(kept).items() if key != "history"}


def _write_kept(project: Path, kept: Kept) -> None:
    document = {
        **_entry(kept),
        "kept_history": list(kept.history),
        "set_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_private(project / KEPT_FILE, json.dumps(document, indent=2) + "\n")
    (project / REGISTERED_FILE).unlink(missing_ok=True)


def undo(project: Path) -> tuple[Kept, Kept | None] | None:
    """Take back the last kept change: the setup kept before it is kept again, or the registered
    setup when there was none. Returns the setup undone and the one now kept (None: the
    registered one); None when nothing is kept."""
    project = project.expanduser()
    undone = read_kept(project)
    if undone is None:
        return _undo_going_back(project)
    if not undone.history:
        _go_back(project, undone.snapshot_id)
        return undone, None
    *earlier, last = undone.history
    restored = _from_entry(last, tuple(earlier))
    _write_kept(project, restored)
    return undone, restored


def _from_entry(entry: Mapping[str, Any], history: tuple[dict[str, Any], ...]) -> Kept:
    fields = {key: value for key, value in entry.items() if key != "kept_history"}
    return Kept(
        **{
            **fields,
            "engine_args": tuple(entry["engine_args"]),
            "answer_changes": tuple(entry["answer_changes"]),
            "env": tuple(entry.get("env", ())),
            "history": history,
        }
    )


def _undo_going_back(project: Path) -> tuple[Kept, Kept | None] | None:
    """Undo a ``tensward set registered``: the setup it left is kept again."""
    try:
        document = json.loads((project / REGISTERED_FILE).read_text("utf-8"))
        left = document["left"]
        if document.get("snapshot_id") != left["snapshot_id"]:
            return None
        restored = _from_entry(left, tuple(left.get("kept_history", ())))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    _write_kept(project, restored)
    registered = Kept(REGISTERED, (), (), REGISTERED, restored.snapshot_id)
    return registered, restored


def set_command(project: Path, run: str, *, unreviewed: bool) -> str:
    """The ``tensward set`` command that keeps ``run`` (or ``REGISTERED``)."""
    words = ["tensward", "set", run, "--project", str(project)]
    return shlex.join([*words, ACCEPT_FLAG] if unreviewed else words)


def flag_words(engine_args: Sequence[str], env: Sequence[str] = ()) -> str:
    """The flags as a user types them to the engine: ``--threads-batch 16``, ``--no-mmap``; each
    environment variable as NAME=VALUE."""
    words = []
    for arg in engine_args:
        name, given, value = arg.partition("=")
        if given and name.replace("_", "").isupper():  # an environment variable
            words.append(arg)
        else:
            words += [f"--{name}", value] if given else [f"--{name}"]
    return shlex.join([*words, *env])


def pinned_baseline(project: Path, resolved: Any, engine: Engine) -> frozenset[str]:
    """The engine arguments the current setup already runs with that its command leaves to the
    engine (what the engine chose at the baseline's launch, pinned): an offer carries them so
    that it can be run as it was measured, but they are not part of the change tried."""
    from .project import settings_for
    from .runs import baseline_resolved

    current = settings_for(resolved, engine, ())
    pinned = engine.pinned(current, current, baseline_resolved(project, resolved))
    return frozenset(engine.engine_args_between(current, pinned))


PLACEMENT_ARGS = ("n-cpu-moe", "n-gpu-layers")
PINNED_WORDS = "placement pinned as {placed} (differs from --fit's) + {rest}"
PINNED_RESULT = "The result is for both, not for {rest} alone: {both}."


def pinned_placement(
    project: Path, resolved: Any, engine_args: Sequence[str], held: Collection[str] = ()
) -> tuple[str, str] | None:
    """The words "placement pinned as --n-cpu-moe 5 (differs from --fit's) + --threads 26" when
    a change also pins where layers or experts live and the baseline's ``--fit`` placement
    cannot be written as those flags, so the step measures that placement as well as the
    change, with the rest of the change ("--threads 26"); None otherwise."""
    from .engines.llamacpp.engine import unreadable_fit
    from .runs import baseline_resolved

    changed = [arg for arg in engine_args if arg not in held]
    placed = [arg for arg in changed if arg.partition("=")[0] in PLACEMENT_ARGS]
    if not placed or len(placed) == len(changed):
        return None
    baseline = baseline_resolved(project, resolved)
    if baseline is None or not unreadable_fit(baseline):
        return None
    rest = flag_words([arg for arg in changed if arg not in placed])
    return PINNED_WORDS.format(placed=flag_words(placed), rest=rest), rest


def tuned_words(project: Path, kept: Kept) -> str:
    """The flags the kept setup changed, without those the baseline's launch pinned."""
    from .project import load_project, project_engine

    resolved = load_project(project, verify_weights=False)
    pins = pinned_baseline(project, resolved, project_engine(resolved.record.current_setup))
    return flag_words([arg for arg in kept.engine_args if arg not in pins], kept.env)


def engine_arg_words(engine_args: Sequence[str], env: Sequence[str] = ()) -> str:
    """``--engine-arg a --engine-arg b``, then ``--engine-env`` for each variable, quoted for a
    shell."""
    words = [word for arg in engine_args for word in ("--engine-arg", arg)]
    return shlex.join([*words, *(w for var in env for w in ("--engine-env", var))])


def set_lines(
    project: Path,
    run: str,
    now: Kept | None,
    before: Kept | None,
    *,
    serving: bool,
    latest: bool = False,
    rejected_in: str | None = None,
    command: str | None = None,
) -> list[str]:
    """What ``tensward set run`` prints: the setup before and after, what can change answers,
    the change of run ``rejected_in`` it carries, which the user rejected, the user's own
    ``command`` with the change written in when the engine has one. The caller adds the "Next:"
    menu, which has the undo and serve."""
    project = project.expanduser()
    named = f"run {run} (the latest run)" if latest else f"run {run}"
    if now is not None and now == before:
        return [f"{named} is already the kept setup"]
    lines = []
    if now is None:
        if run != REGISTERED:
            lines.append(
                f"{named} measured the registered setup (no engine arguments), so keeping it "
                f"is the same as tensward set {REGISTERED}"
            )
        if before is None:
            return [
                *lines,
                "nothing is kept: tensward analyse and tensward serve already start from the "
                "registered setup",
            ]
        lines.append(
            "back to the registered setup: tensward analyse and tensward serve start from it"
        )
        lines.append(f"dropped: {flag_words(before.engine_args, before.env)} (run {before.run})")
    else:
        lines.append(f"kept the settings of {named}: tensward analyse starts from them")
        lines.append(
            "before: the registered setup (no engine arguments)"
            if before is None
            else f"before: {flag_words(before.engine_args, before.env)} (run {before.run})"
        )
        lines.append(f"after:  {flag_words(now.engine_args, now.env)}")
        if command:
            lines.append(f"your command with it, as measured: {command}")
            if "***" in command:
                lines.append("fill in the values shown as *** (Tensward does not keep secrets)")
        if now.answer_changes:
            changed = ", ".join(ANSWER_LEVERS[name] for name in now.answer_changes)
            step = step_block(project / "runs" / now.run)
            lines.append(
                f"this run changes {changed}, which can change answers; "
                + (
                    "checked: reasoning matches your current setup"
                    if reasoning_same((step or {}).get("answers"))
                    else "kept as you asked"
                )
            )
        if rejected_in is not None:
            clear = ["tensward", "reject", "--clear", rejected_in, "--project", str(project)]
            lines.append(
                f"you rejected this change (run {rejected_in}); `{shlex.join(clear)}` takes "
                "that back"
            )
    if serving:
        lines.append("a server already running keeps its settings until it is started again")
    return lines
