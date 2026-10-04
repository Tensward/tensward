"""Did a change move the answers? Two runs of one project are compared prompt by prompt: how
similar the candidate's answers are to the baseline's, against how similar the baseline's own
repeats of each prompt are to each other. Correctness is not judged here; the answers are
written side by side for the user to read."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from statistics import fmean
from typing import Callable, Mapping, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from .contracts import DigestHex, StrictModel
from .errors import (
    PROJECT_INPUTS_CHANGED,
    PROJECT_INPUTS_INVALID,
    PreflightError,
    validation_summary,
)
from .files import parse_document, write_json, write_private
from .inputs import PromptEntry
from .platforms import DeviceIdentity
from .project import ResolvedProject
from .workload import ChatMessage, ImagePart

CHANGE_MARGIN = 0.15  # below the baseline's own similarity by more than this: changed
HEALTH_MARGIN = 0.05  # a format problem rate that rises by more than this is worse
MAX_ANSWERS_PER_PROMPT = 16  # per side; bounds the pairwise similarity only
RUN_RECORD = "run.json"
SPEED_METRICS = (
    "request_throughput",
    "output_throughput",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "tpot_p95_ms",
    "kv_capacity_tokens",
)
INLINE_PAIRS = 3  # prompts shown with their answers in the report
INLINE_CHARS = 400  # per answer shown in the report
FILE_CHARS = 4_000  # per answer written to answers.md
FILE_ANSWERS = 3  # distinct answers per side and prompt in answers.md
_WORD = re.compile(r"\w+")
_Key = tuple[str, tuple[tuple[str, str], ...] | None, str | None]
_Rates = tuple[float | None, float | None]  # (baseline, candidate)


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    tool_calls: tuple[tuple[str, str], ...] | None  # (name, arguments as JSON text)
    finish_reason: str | None


@dataclass(frozen=True, slots=True)
class PromptComparison:
    prompt_id: str
    agreement: float | None
    noise: float | None
    same: float | None
    changed: bool
    unanswered: bool
    baseline_count: int
    candidate_count: int
    identical: bool | None
    first_difference: int | None


@dataclass(frozen=True, slots=True)
class Outcome:
    prompts: tuple[PromptComparison, ...]
    has_noise_floor: bool
    reference: _Rates
    health: dict[str, _Rates]
    verdict: str
    equal: bool | None
    reproduction: tuple[int, int]  # prompts whose baseline answers agree, prompts with repeats
    coverage: tuple[int, int]  # prompts answered by both sides, prompts the candidate answered


def canonical_arguments(arguments: str) -> str:
    """Arguments as key-sorted compact JSON, so key order and spacing are not a change."""
    try:
        return json.dumps(json.loads(arguments), sort_keys=True, separators=(",", ":"))
    except ValueError:
        return arguments


def _calls(answer: Answer) -> tuple[tuple[str, str], ...]:
    return tuple(
        (name, canonical_arguments(arguments)) for name, arguments in answer.tool_calls or ()
    )


def _words(answer: Answer) -> Counter[str]:
    return Counter(_WORD.findall(answer.text.lower()))


def _f1(a: Counter[str], b: Counter[str]) -> float:
    total = sum(a.values()) + sum(b.values())
    return 1.0 if total == 0 else 2 * sum((a & b).values()) / total


def _pair(a: Answer, words_a: Counter[str], b: Answer, words_b: Counter[str]) -> float:
    """1 for the same tool calls or the same words, 0 for nothing in common: word-overlap F1."""
    if a.tool_calls or b.tool_calls:
        return float(_calls(a) == _calls(b))
    return _f1(words_a, words_b)


def _normal(answer: Answer) -> tuple[str, tuple[tuple[str, str], ...]]:
    return " ".join(answer.text.lower().split()), _calls(answer)


def _identity(baseline: Sequence[Answer]) -> Callable[[Answer], _Key]:
    """What makes two answers the same: the text, plus the tool calls and the finish reason when
    every baseline answer to the prompt has them (recorded answers may not)."""
    calls = all(a.tool_calls is not None for a in baseline)
    finish = all(a.finish_reason is not None for a in baseline)
    return lambda a: (a.text, _calls(a) if calls else None, a.finish_reason if finish else None)


def _render(key: _Key) -> str:
    text, calls, finish = key
    lines = [text, *(f"{name}({arguments})" for name, arguments in calls or ())]
    return "\n".join(lines if finish is None else [*lines, f"finish:{finish}"])


def _first_difference(a: str, b: str) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def _exact(
    baseline: Sequence[Answer], candidate: Sequence[Answer]
) -> tuple[bool | None, int | None]:
    """Whether every candidate answer is one of the baseline's, and where the first one that is
    not departs from the baseline's first answer. Reads every answer: no sample."""
    if not baseline or not candidate:
        return None, None
    identity = _identity(baseline)
    known = {identity(answer) for answer in baseline}
    stray = next((answer for answer in candidate if identity(answer) not in known), None)
    if stray is None:
        return True, None
    return False, _first_difference(_render(identity(stray)), _render(identity(baseline[0])))


def _compare_prompt(
    prompt_id: str, baseline: Sequence[Answer], candidate: Sequence[Answer]
) -> PromptComparison:
    identical, first_difference = _exact(baseline, candidate)
    counts = (len(baseline), len(candidate))
    if not baseline or not candidate:
        return PromptComparison(
            prompt_id, None, None, None, False, bool(baseline), *counts, identical, first_difference
        )
    base = [(a, _words(a)) for a in baseline[:MAX_ANSWERS_PER_PROMPT]]
    cand = [(a, _words(a)) for a in candidate[:MAX_ANSWERS_PER_PROMPT]]
    noise = fmean(_pair(*x, *y) for x, y in combinations(base, 2)) if len(base) > 1 else None
    agreement = fmean(_pair(*x, *y) for x, y in product(base, cand))
    same = fmean(_normal(x) == _normal(y) for (x, _), (y, _) in product(base, cand))
    changed = noise is not None and agreement < noise - CHANGE_MARGIN
    return PromptComparison(
        prompt_id, agreement, noise, same, changed, False, *counts, identical, first_difference
    )


def _order(prompt: PromptComparison) -> tuple[int, float]:
    """Unanswered prompts first, then the ones whose words differ, least agreeing first."""
    if prompt.unanswered:
        return 0, 0.0
    if prompt.same is not None and prompt.same < 1:
        return 1, prompt.agreement or 0.0
    return 2, 0.0


def _failures_rose(failed: _Rates) -> bool:
    return (failed[1] or 0) > (failed[0] or 0)


def _share(count: int, total: int) -> float | None:
    return count / total if total else None


def _health(answers: Sequence[Answer], structured: bool) -> dict[str, float | None]:
    def invalid(answer: Answer) -> bool:
        try:
            json.loads(answer.text)
        except ValueError:
            return True
        return False

    rates = {
        "cut short": sum(a.finish_reason == "length" for a in answers),
        "empty": sum(not a.text and not a.tool_calls for a in answers),
    }
    if structured:
        rates["invalid JSON"] = sum(invalid(a) for a in answers)
    return {name: _share(count, len(answers)) for name, count in rates.items()}


def _reference(
    answers: Mapping[str, Sequence[Answer]], references: Mapping[str, str]
) -> float | None:
    scores = [
        _f1(_words(answer), _words(Answer(references[prompt_id], None, None)))
        for prompt_id, given in answers.items()
        if prompt_id in references
        for answer in given
    ]
    return fmean(scores) if scores else None


def _percent(rate: float) -> str:
    return f"{rate:.0%}"


_HEALTH_NAMES = {
    "failed": "failed requests",
    "cut short": "answers cut short",
    "empty": "empty answers",
    "invalid JSON": "invalid JSON",
}


def _verdict(
    prompts: Sequence[PromptComparison],
    has_floor: bool,
    reference: _Rates,
    health: Mapping[str, _Rates],
    recorded: bool,
) -> str:
    asked = sum(p.baseline_count > 0 for p in prompts)
    unanswered = sum(p.unanswered for p in prompts)
    wrong = sum(p.changed for p in prompts) + unanswered
    worse = [
        f"{_HEALTH_NAMES[name]} {_percent(before)} → {_percent(after)}"
        for name, (before, after) in health.items()
        if before is not None and after is not None and after - before > HEALTH_MARGIN
    ]
    if reference[0] is not None and reference[1] is not None:
        if reference[1] < reference[0] - CHANGE_MARGIN:
            worse.append(f"reference match {_percent(reference[0])} → {_percent(reference[1])}")
    if wrong:
        suffix = f" ({unanswered} got no answer)" if unanswered else ""
        return f"Answers changed: {wrong} of {asked} prompts{suffix}"
    if worse:
        return f"Answers changed: {', '.join(worse)}"
    if not has_floor:
        if recorded:
            return (
                "No noise floor: recorded answers have no repeats, so only equality and "
                "agreement are reported."
            )
        return (
            "No noise floor: every prompt ran once in your current-setup run. To judge changes, "
            "raise request_count in the configuration to at least twice the number of prompts, "
            "run `tensward init` again, then analyse your current setup and the change."
        )
    unjudged = sum(
        bool(p.baseline_count and p.candidate_count and p.noise is None) for p in prompts
    )
    if not unjudged:
        return "Answers unchanged within noise"
    noun = "1 prompt ran once and was" if unjudged == 1 else f"{unjudged} prompts ran once and were"
    return f"Answers unchanged within noise ({noun} not judged)"


def compare_answers(
    baseline: Mapping[str, Sequence[Answer]],
    candidate: Mapping[str, Sequence[Answer]],
    references: Mapping[str, str],
    *,
    structured: bool,
    failed: _Rates = (None, None),
    recorded: bool = False,
) -> Outcome:
    """Compare two sets of answers by prompt id. ``failed`` is each side's share of failed
    requests; a share that is None is unknown, and counts as no failures on the candidate side.
    ``recorded`` says the baseline is recorded production answers."""
    prompts = sorted(
        (
            _compare_prompt(prompt_id, baseline.get(prompt_id, ()), candidate.get(prompt_id, ()))
            for prompt_id in dict.fromkeys([*baseline, *candidate])
        ),
        key=_order,
    )
    sides = [
        _health([a for given in side.values() for a in given], structured)
        for side in (baseline, candidate)
    ]
    health: dict[str, _Rates] = {"failed": failed}
    finish_known = any(a.finish_reason for given in baseline.values() for a in given)
    health |= {
        name: (sides[0][name], sides[1][name])
        for name in sides[0]
        if name != "cut short" or finish_known
    }
    reference = (_reference(baseline, references), _reference(candidate, references))
    has_floor = any(p.noise is not None for p in prompts)

    repeated = [given for given in baseline.values() if len(given) > 1]
    reproduced = sum(len({_identity(given)(a) for a in given}) == 1 for given in repeated)
    coverage = (
        sum(p.baseline_count > 0 and p.candidate_count > 0 for p in prompts),
        sum(p.candidate_count > 0 for p in prompts),
    )
    if not any(p.baseline_count or p.candidate_count for p in prompts):
        equal = None
    else:
        equal = (
            all(p.identical is not False and not p.unanswered for p in prompts)
            and coverage[0] == coverage[1]
            and not _failures_rose(failed)
        )
    return Outcome(
        tuple(prompts),
        has_floor,
        reference,
        health,
        _verdict(prompts, has_floor, reference, health, recorded),
        equal,
        (reproduced, len(repeated)),
        coverage,
    )


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
    tool_calls: dict[str, JsonValue] | None = None

    @property
    def failed_share(self) -> float | None:
        return _share(self.failed, self.succeeded + self.failed)

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

    @classmethod
    def from_run(cls, run_dir: Path, run_file: RunFile) -> RunRecord:
        """The run in ``run_dir``, whose ``run.json`` is ``run_file``: its successful answers
        joined to their prompts."""
        if not run_file.retained_responses:
            raise PreflightError(
                PROJECT_INPUTS_INVALID,
                f"run {run_dir.name} did not keep its answers (--no-retain-responses)",
            )
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
            project.settings.workload.temperature,
            None,
            None,
            {prompt_id: tuple(given) for prompt_id, given in answers.items()},
        )


@dataclass(frozen=True, slots=True)
class Comparison:
    baseline_id: str
    candidate_id: str
    outcome: Outcome
    speed: dict[str, _Rates]
    temperature: float
    differences: tuple[str, ...]
    recorded: bool


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
        structured=project.settings.workload.structured_output is not None,
        failed=(
            baseline.metrics.failed_share if baseline.metrics else None,
            candidate.metrics.failed_share if candidate.metrics else None,
        ),
        recorded=recorded,
    )
    speed = (
        {name: (getattr(metrics[0], name), getattr(metrics[1], name)) for name in SPEED_METRICS}
        if metrics[0] and metrics[1]
        else {}
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


def _fenced(text: str) -> str:
    """``text`` in a code fence long enough that backticks inside it cannot close it."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}\n{text}\n{fence}"


def _trimmed(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "… (trimmed)"


def _show(answer: Answer, limit: int) -> str:
    calls = [f"{name}({arguments})" for name, arguments in answer.tool_calls or ()]
    return _fenced(_trimmed("\n".join([answer.text, *calls]).strip() or "(empty)", limit))


def _cell(value: float | None, show: Callable[[float], str], note: str = "") -> str:
    return "—" if value is None else show(value) + note


def _number(value: float) -> str:
    return f"{value:,.0f}" if value >= 1000 else f"{value:.1f}"


def _decimal(value: float) -> str:
    return f"{value:.2f}"


def _change(before: float | None, after: float | None) -> str:
    return "" if before is None or after is None or before == 0 else f"{after / before - 1:+.0%}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        f"| {' | '.join(header)} |",
        f"|{'---|' * len(header)}",
        *(f"| {' | '.join(row)} |" for row in rows),
        "",
    ]


_SPEED_LABELS = {
    "request_throughput": "requests per second",
    "output_throughput": "output tokens per second",
    "ttft_p50_ms": "TTFT p50 (ms)",
    "ttft_p95_ms": "TTFT p95 (ms)",
    "tpot_p95_ms": "TPOT p95 (ms)",
    "kv_capacity_tokens": "KV cache capacity (tokens)",
}
_TOOL_CALL_LABELS = {
    "produced_call": "produced a tool call",
    "valid_json": "valid JSON arguments",
    "known_tool": "known tool",
    "schema_valid": "arguments match the schema",
}


def _tool_call_rows(baseline: RunRecord, candidate: RunRecord) -> list[list[str]]:
    sides = [(r.metrics.tool_calls if r.metrics else None) or {} for r in (baseline, candidate)]

    def share(side: Mapping[str, JsonValue], key: str) -> float | None:
        value = side.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    return [
        [f"tool calls: {label}", *(_cell(share(side, key), _percent) for side in sides)]
        for key, label in _TOOL_CALL_LABELS.items()
        if any(key in side for side in sides)
    ]


def _mean(values: Sequence[float | None]) -> float | None:
    known = [value for value in values if value is not None]
    return fmean(known) if known else None


def _baseline_name(comparison: Comparison) -> str:
    return "your recorded answers" if comparison.recorded else "your current setup"


def _quality_rows(
    comparison: Comparison, baseline: RunRecord, candidate: RunRecord
) -> list[list[str]]:
    outcome = comparison.outcome
    prompts = outcome.prompts
    rows = [
        [
            "word-overlap similarity",
            _cell(_mean([p.noise for p in prompts]), _decimal, " (its own repeats)"),
            _cell(
                _mean([p.agreement for p in prompts]),
                _decimal,
                f" (to {_baseline_name(comparison)})",
            ),
        ],
        ["answers with the same words", "—", _cell(_mean([p.same for p in prompts]), _percent)],
    ]
    if any(rate is not None for rate in outcome.reference):
        rows.append(["match with the reference", *(_cell(r, _decimal) for r in outcome.reference)])
    rows += [
        [_HEALTH_NAMES[name], *(_cell(rate, _percent) for rate in rates)]
        for name, rates in outcome.health.items()
    ]
    return [*rows, *_tool_call_rows(baseline, candidate)]


_BATCH_INVARIANT_ADVICE = (
    "Put `VLLM_BATCH_INVARIANT=1` (vLLM's batch-invariant mode: beta, compute capability 8.0 or "
    "higher, slower, and it does not support prefix caching yet, vLLM issue #27433) and "
    "`--no-enable-prefix-caching` (vLLM enables prefix caching by default) in your `--current` "
    "command, for example `VLLM_BATCH_INVARIANT=1 vllm serve … --no-enable-prefix-caching`. Run "
    "`tensward init` again, because that changes the registration, then analyse your current "
    "setup again."
)
_SEED_ADVICE = (
    "Sampling is random: set a seed in the configuration, run `tensward init` again, then "
    "analyse your current setup again."
)


def _prompts(count: int) -> str:
    return f"{count} prompt" if count == 1 else f"{count} prompts"


def _equality_lines(comparison: Comparison, project: ResolvedProject) -> list[str]:
    """Whether every answer is one the baseline gave, why not, and how well the baseline
    reproduces itself."""
    outcome = comparison.outcome
    reproduced, repeated = outcome.reproduction
    covered, answered = outcome.coverage
    complete = covered == answered
    if outcome.equal:
        same = f"All answers identical to {comparison.baseline_id}"
        lines = [f"{same} (text, tool calls and finish reason)."]
    elif outcome.equal is None:
        lines = ["Not identical: nothing to judge, no prompt was answered."]
    else:
        judged = [p for p in outcome.prompts if p.identical is not None]
        differing = [p for p in judged if p.identical is False]
        problems = []
        if differing:
            where = next((p for p in differing if p.first_difference is not None), None)
            at = (
                f" (first difference at character {where.first_difference} in {where.prompt_id})"
                if where
                else ""
            )
            problems.append(f"{len(differing)} of {len(judged)} prompts differ{at}")
        if unanswered := sum(p.unanswered for p in outcome.prompts):
            problems.append(f"{_prompts(unanswered)} got no answer")
        if _failures_rose(outcome.health["failed"]):
            problems.append("the failure rate rose")
        if not complete:
            problems.append("the coverage is incomplete")
        lines = [
            ("Not identical: " if differing else "Equality not shown: ") + "; ".join(problems) + "."
        ]
    if comparison.recorded or not complete:
        verb = (
            "The recorded answers cover" if comparison.recorded else "The current-setup run covers"
        )
        needed = "" if complete else ", and every prompt the candidate answered must be covered"
        lines.append(f"{verb} {covered} of {answered} prompts{needed}.")
    if comparison.recorded:
        lines.append("Self-reproduction not applicable: the baseline is recorded answers.")
    elif not repeated:
        lines.append("Self-reproduction not checked: each prompt ran once.")
    else:
        lines.append(
            f"Your current setup reproduced its own answers for {reproduced} of {repeated} prompts."
        )
        if reproduced < repeated:
            unseeded = comparison.temperature > 0 and project.settings.workload.seed is None
            lines.append(_SEED_ADVICE if unseeded else _BATCH_INVARIANT_ADVICE)
    return lines


def render_section(
    comparison: Comparison, candidate: RunRecord, baseline: RunRecord, project: ResolvedProject
) -> list[str]:
    """The report section that compares ``candidate`` with ``baseline``."""
    outcome = comparison.outcome
    name = _baseline_name(comparison)
    lines = [f"## Compared with {name} (run {comparison.baseline_id})", ""]
    if comparison.differences:
        lines += [
            f"Measured on a different {', '.join(comparison.differences)} than your current-setup "
            "run: speed and answers may differ for that reason alone.",
            "",
        ]
    lines += [f"**{outcome.verdict}**", ""]
    lines += [*_equality_lines(comparison, project), ""]
    if comparison.speed:
        lines += _table(
            ["", name.removeprefix("your "), "this run", "change"],
            [
                [_SPEED_LABELS[name], *(_cell(v, _number) for v in values), _change(*values)]
                for name, values in comparison.speed.items()
            ],
        )
    lines += _table(
        ["", name.removeprefix("your "), "this run"], _quality_rows(comparison, baseline, candidate)
    )
    rule = (
        "A prompt counts as changed when its answers are more than 0.15 less similar to your "
        "current setup's than your current setup's are to each other (word overlap, at most 16 "
        "answers per prompt; a single changed number barely moves it, so read the answers "
        f"below). Temperature: {comparison.temperature:g}."
    )
    if comparison.temperature > 0 and project.settings.workload.seed is not None:
        rule += (
            " With a fixed seed, repeats of a prompt sample the same answer, so the noise floor "
            "is tight and any engine change that alters sampling shows as changed answers; read "
            "the pairs."
        )
    lines += [rule, ""]
    for prompt in outcome.prompts[:INLINE_PAIRS]:
        before = baseline.answers.get(prompt.prompt_id)
        after = candidate.answers.get(prompt.prompt_id)
        lines += [f"### Prompt `{prompt.prompt_id}`", ""]
        for who, given in ((name.capitalize(), before), ("This run", after)):
            lines += [f"{who}:", "", _show(given[0], INLINE_CHARS) if given else "(no answer)", ""]
    lines.append(f"All prompts, with every distinct answer: `{_directory(comparison)}/answers.md`")
    return lines


def _directory(comparison: Comparison) -> str:
    return f"compare/{comparison.baseline_id}-vs-{comparison.candidate_id}"


def _prompt_display(entry: PromptEntry) -> str:
    """The prompt as written in the workload; an image is named by its path there."""
    if entry.messages is None:
        return entry.prompt or ""

    def text(message: ChatMessage) -> str:
        if isinstance(message.content, str):
            return message.content
        return "\n".join(
            f"[image: {part.image_url.url}]" if isinstance(part, ImagePart) else part.text
            for part in message.content
        )

    return "\n".join(f"{message.role}: {text(message)}" for message in entry.messages)


def render_answers(
    comparison: Comparison, candidate: RunRecord, baseline: RunRecord, project: ResolvedProject
) -> str:
    """Every prompt with its answers from both runs, side by side, for reading."""
    entries = {entry.id: entry for entry in project.prompts}
    name = _baseline_name(comparison)
    lines = [
        f"# Answers: {comparison.baseline_id} ({name}) and run {comparison.candidate_id}",
        "",
        comparison.outcome.verdict,
        "",
    ]
    for prompt in comparison.outcome.prompts:
        if prompt.agreement is None:
            note = "no answer" if prompt.candidate_count == 0 else f"not in {name}"
        elif prompt.noise is None:
            note = f"agreement {prompt.agreement:.2f}, not judged: it ran once"
        else:
            note = f"agreement {prompt.agreement:.2f}, the baseline's own {prompt.noise:.2f}"
        if prompt.first_difference is not None:
            note += f"; first difference at character {prompt.first_difference}"
        lines += [f"## {prompt.prompt_id}: {note}", ""]
        if prompt.prompt_id in entries:
            lines += [_fenced(_trimmed(_prompt_display(entries[prompt.prompt_id]), FILE_CHARS)), ""]
        for who, run in ((name.capitalize(), baseline), ("This run", candidate)):
            distinct = list(dict.fromkeys(run.answers.get(prompt.prompt_id, ())))[:FILE_ANSWERS]
            lines += [f"### {who}", ""]
            lines += [_show(answer, FILE_CHARS) + "\n" for answer in distinct] or [
                "(no answer)",
                "",
            ]
    return "\n".join(lines)


def write_comparison(
    project_dir: Path,
    comparison: Comparison,
    baseline: RunRecord,
    candidate: RunRecord,
    project: ResolvedProject,
) -> Path:
    """Write ``answers.md`` and ``compare.json`` privately below ``project_dir/compare`` and
    return the path of ``answers.md``."""
    directory = project_dir / _directory(comparison)
    for level in (directory.parent, directory):
        level.mkdir(mode=0o700, exist_ok=True)
    write_json(directory / "compare.json", asdict(comparison))
    answers = directory / "answers.md"
    write_private(answers, render_answers(comparison, candidate, baseline, project))
    return answers
