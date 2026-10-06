"""Reads a run's files back into records and compares two runs of one project."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from .compare import Answer, Outcome, Rates, compare_answers, share
from .contracts import DigestHex, StrictModel
from .errors import (
    PROJECT_INPUTS_CHANGED,
    PROJECT_INPUTS_INVALID,
    PreflightError,
    validation_summary,
)
from .files import parse_document
from .platforms import DeviceIdentity
from .project import ResolvedProject

RUN_RECORD = "run.json"
SPEED_LABELS = {
    "request_throughput": "requests per second",
    "output_throughput": "output tokens per second",
    "ttft_p50_ms": "TTFT p50 (ms)",
    "ttft_p95_ms": "TTFT p95 (ms)",
    "tpot_p95_ms": "TPOT p95 (ms)",
    "kv_capacity_tokens": "KV cache capacity (tokens)",
}


class RunMetrics(BaseModel):
    """The numbers of a run's ``metrics.json`` that a comparison reads. Not strict: the file
    comes from ``json.dumps``, so it may hold NaN and keys this model does not name."""

    succeeded: int
    failed: int
    request_throughput: float | None = None
    output_throughput: float | None = None
    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None
    tpot_p95_ms: float | None = None
    kv_capacity_tokens: float | None = None
    engine_output_throughput: float | None = None
    prompt_tokens_per_step: float | None = None
    ceilings: dict[str, JsonValue] | None = None
    tool_calls: dict[str, JsonValue] | None = None
    diagnosis: dict[str, JsonValue] | None = None
    window: dict[str, JsonValue] | None = None

    @property
    def failed_share(self) -> float | None:
        return share(self.failed, self.succeeded + self.failed)

    @classmethod
    def read(cls, path: Path) -> RunMetrics:
        try:
            return cls.model_validate_json(_read_run_file(path))
        except ValidationError as error:
            raise PreflightError(
                PROJECT_INPUTS_INVALID,
                f"run {path.parent.name} has an invalid {path.name}: {validation_summary(error)}",
            ) from None


@dataclass(frozen=True, slots=True)
class Machine:
    runtime: str
    gpus: tuple[str, ...] | None
    image: str | None
    identities: tuple[DeviceIdentity, ...]

    def differences(self, other: Machine) -> tuple[str, ...]:
        """What differs from ``other`` and could change speed or answers. GPUs are compared only
        when both runs recorded them."""
        both = bool(self.identities and other.identities)
        names, drivers = (
            [tuple(getattr(gpu, field) for gpu in machine.identities) for machine in (self, other)]
            for field in ("name", "driver")
        )
        return tuple(
            name
            for name, differs in (
                ("runtime", self.runtime != other.runtime),
                ("image", self.image != other.image),
                ("GPU", both and names[0] != names[1]),
                ("driver", both and drivers[0] != drivers[1]),
            )
            if differs
        )


class RunFile(StrictModel):
    """``run.json``: what produced a run."""

    schema_version: str
    snapshot_id: DigestHex
    engine_args: tuple[str, ...]
    settings: dict[str, JsonValue]
    retained_responses: bool
    temperature: float
    runtime: str
    gpus: tuple[str, ...] | None
    image: str | None
    inherited_env: dict[str, str] = Field(default_factory=dict)
    gpus_identity: tuple[DeviceIdentity, ...] = ()
    engine: str | None = None
    engine_version: str | None = None
    platform: str | None = None
    format: str | None = None
    tensward_version: str | None = None
    host: dict[str, JsonValue] | None = None

    @classmethod
    def read(cls, run_dir: Path) -> RunFile:
        path = run_dir / RUN_RECORD
        if not path.is_file():
            raise PreflightError(
                PROJECT_INPUTS_INVALID,
                f"run {run_dir.name} was made before Tensward 0.1.6 and cannot be compared; "
                "run `tensward analyse` again",
            )
        return parse_document(
            cls,
            _read_run_file(path),
            f"{RUN_RECORD} of run {run_dir.name}",
            PROJECT_INPUTS_INVALID,
        )


class _Row(StrictModel):
    """A line of a run's JSONL files: only the fields a comparison reads."""

    model_config = ConfigDict(extra="ignore")

    request_id: str
    prompt_id: str


class _RequestRow(_Row):
    outcome: str


class _CallRow(StrictModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    arguments: str


class _ResponseRow(_Row):
    text: str
    tool_calls: tuple[_CallRow, ...]
    finish_reason: str | None


def _read_run_file(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        problem = "missing"
    except OSError:
        problem = "unreadable"
    raise PreflightError(
        PROJECT_INPUTS_INVALID, f"run {path.parent.name} is incomplete: {path.name} {problem}"
    )


_StrictRow = TypeVar("_StrictRow", bound=StrictModel)


def _parse_rows(model: type[_StrictRow], name: str, data: bytes) -> list[_StrictRow]:
    return [
        parse_document(model, line, f"line {number} of {name}", PROJECT_INPUTS_INVALID)
        for number, line in enumerate(data.splitlines(), start=1)
    ]


_FileRow = TypeVar("_FileRow", bound=_Row)


def _rows(model: type[_FileRow], path: Path) -> list[_FileRow]:
    return _parse_rows(model, path.name, _read_run_file(path))


class _RecordedRow(StrictModel):
    """A line of a file of recorded production answers."""

    prompt_id: str
    text: str
    tool_calls: tuple[_CallRow, ...] | None = None
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One run held in memory: what produced it, what it measured and what it answered."""

    run_id: str
    snapshot_id: str
    temperature: float
    machine: Machine | None
    metrics: RunMetrics | None
    answers: Mapping[str, tuple[Answer, ...]]
    tensward_version: str | None = None
    engine_args: tuple[str, ...] = ()
    settings: Mapping[str, JsonValue] | None = None

    @classmethod
    def from_run(
        cls, run_dir: Path, run_file: RunFile, *, metrics: RunMetrics | None = None
    ) -> RunRecord:
        """The run in ``run_dir``, whose ``run.json`` is ``run_file``: its successful answers
        joined to their prompts. With ``metrics``, ``metrics.json`` is not read."""
        if not run_file.retained_responses:
            raise PreflightError(
                PROJECT_INPUTS_INVALID,
                f"run {run_dir.name} did not keep its answers (--no-retain-responses)",
            )
        if metrics is None:
            metrics = RunMetrics.read(run_dir / "metrics.json")
        succeeded = {
            row.request_id
            for row in _rows(_RequestRow, run_dir / "requests.jsonl")
            if row.outcome == "success"
        }
        answers: dict[str, list[Answer]] = {}
        for row in _rows(_ResponseRow, run_dir / "responses.jsonl"):
            if row.request_id in succeeded:
                calls = tuple((call.name, call.arguments) for call in row.tool_calls)
                answers.setdefault(row.prompt_id, []).append(
                    Answer(row.text, calls, row.finish_reason)
                )
        return cls(
            run_dir.name,
            run_file.snapshot_id,
            run_file.temperature,
            Machine(run_file.runtime, run_file.gpus, run_file.image, run_file.gpus_identity),
            metrics,
            {prompt_id: tuple(given) for prompt_id, given in answers.items()},
            run_file.tensward_version,
            run_file.engine_args,
            run_file.settings,
        )

    @classmethod
    def from_recorded(cls, path: Path, project: ResolvedProject) -> RunRecord:
        """Recorded production answers, one JSON object per line, as a run of ``project`` with
        no machine and no measurements. Fields a line leaves out are not compared."""
        try:
            data = path.read_bytes()
        except OSError:
            raise PreflightError(
                PROJECT_INPUTS_INVALID, f"the recorded answers in {path.name} cannot be read"
            ) from None
        known = {entry.id for entry in project.prompts}
        answers: dict[str, list[Answer]] = {}
        for row in _parse_rows(_RecordedRow, path.name, data):
            if row.prompt_id not in known:
                raise PreflightError(
                    PROJECT_INPUTS_INVALID,
                    f"{path.name} answers prompt {row.prompt_id!r}, which is not in the workload",
                )
            calls = (
                None
                if row.tool_calls is None
                else tuple((call.name, call.arguments) for call in row.tool_calls)
            )
            answers.setdefault(row.prompt_id, []).append(Answer(row.text, calls, row.finish_reason))
        return cls(
            re.sub(r"[^A-Za-z0-9._-]", "_", path.stem),
            project.record.snapshot_id,
            project.config.workload.temperature,
            None,
            None,
            {prompt_id: tuple(given) for prompt_id, given in answers.items()},
        )


@dataclass(frozen=True, slots=True)
class Comparison:
    baseline_id: str
    candidate_id: str
    outcome: Outcome
    speed: dict[str, Rates]
    temperature: float
    differences: tuple[str, ...]
    recorded: bool
    engine_rated: tuple[bool, bool] = (False, False)  # sides whose output rate is the engine's
    between_runs: bool = False  # the baseline is not the registered setup


def require_same_inputs(project: ResolvedProject, *snapshot_ids: str) -> None:
    """Refuse snapshots that differ, or that are not the project's current registration."""
    if set(snapshot_ids) != {project.record.snapshot_id}:
        raise PreflightError(
            PROJECT_INPUTS_CHANGED,
            "the runs were not made from the project's current registration (checkpoint, "
            "configuration, workload and current setup); a comparison needs the same one",
        )


def compare_runs(baseline: RunRecord, candidate: RunRecord, project: ResolvedProject) -> Comparison:
    """Compare ``candidate`` with ``baseline``, two runs of ``project``."""
    require_same_inputs(project, baseline.snapshot_id, candidate.snapshot_id)
    recorded = baseline.machine is None
    metrics = (baseline.metrics, candidate.metrics)
    outcome = compare_answers(
        baseline.answers,
        candidate.answers,
        {entry.id: entry.reference for entry in project.prompts if entry.reference is not None},
        structured=project.config.workload.structured_output is not None,
        failed=(
            baseline.metrics.failed_share if baseline.metrics else None,
            candidate.metrics.failed_share if candidate.metrics else None,
        ),
        recorded=recorded,
    )
    speed: dict[str, Rates] = {}
    engine_rated = (False, False)
    if metrics[0] and metrics[1]:
        before, after = metrics[0], metrics[1]
        engine_rated = (
            before.engine_output_throughput is not None,
            after.engine_output_throughput is not None,
        )
        speed = {name: (getattr(before, name), getattr(after, name)) for name in SPEED_LABELS}
        speed["output_throughput"] = (
            before.engine_output_throughput or before.output_throughput,
            after.engine_output_throughput or after.output_throughput,
        )
    differences = (
        baseline.machine.differences(candidate.machine)
        if baseline.machine and candidate.machine
        else ()
    )
    return Comparison(
        baseline.run_id,
        candidate.run_id,
        outcome,
        speed,
        candidate.temperature,
        differences,
        recorded,
        engine_rated,
        bool(baseline.engine_args),
    )


class NoBaseline(Exception):
    """No run of the current setup can be the baseline; the message says what to do."""


def find_baseline(project_dir: Path, project: ResolvedProject) -> RunRecord:
    """The newest run of the project's current setup that kept its answers: same snapshot, no
    engine arguments, at least one answer. Run ids sort by time to one second, so two runs
    started in the same second may be taken in either order. Runs that cannot be read are
    skipped."""
    runs = project_dir / "runs"
    unkept = False
    for run_dir in sorted(runs.iterdir(), reverse=True) if runs.is_dir() else ():
        try:
            run_file = RunFile.read(run_dir)
            if run_file.snapshot_id != project.record.snapshot_id or run_file.engine_args:
                continue
            if not run_file.retained_responses:
                unkept = True
                continue
            record = RunRecord.from_run(run_dir, run_file)
        except (PreflightError, OSError):
            continue
        if record.answers:
            return record
    reason = (
        "your latest current-setup run did not keep its answers (--no-retain-responses); run it "
        "again without that flag."
        if unkept
        else f"run `tensward analyse --project {project_dir}` first, then this command again."
    )
    raise NoBaseline(f"No run of your current setup to compare with: {reason}")
