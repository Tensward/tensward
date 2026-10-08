"""``tensward analyse``: serve the registered model, offer the registered prompts, diagnose.

The flow is: load the verified project, start a server through a :class:`Runtime`, wait until it
is ready, run the declared warmup, scrape the engine's metrics, run the declared workload
through the streaming client, scrape the metrics again, write the report, and always stop the
server. Everything written lands in ``<project>/runs/<run_id>/``. Nothing here invents a value:
whatever the server or the configuration did not establish is reported as not measured.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence, cast

from pydantic import JsonValue

from .classify import NAMES, AnswersState, Bottleneck, Diagnosis, diagnose_run
from .compare import Outcome
from .engines import Engine
from .environment import (
    describe_run,
    require_available,
    require_gpus,
    with_logged_version,
)
from .errors import PROJECT_CONFIG_UNSUPPORTED, AnalyseFailure, PreflightError
from .files import new_run_id
from .measure import run_file_for, run_workloads, serving_gpu, summarize_run
from .measurement import Measurement
from .playbook import Situation, Suggestion
from .profiling.run import add_profiles
from .progress import Note, PhaseStarted, RunWritten, emit
from .project import (
    ResolvedProject,
    load_project,
    settings_for,
    tool_calling_warning,
    workload_facts,
)
from .report.build import RunOutputs, Subject, overrides_suffix
from .report.compare import render_changes, render_section, tpot_note, write_comparison
from .report.model import Report
from .rundir import estimate_inputs, request_rows, write_derived, write_prompt_blocks
from .runs import (
    Comparison,
    NoBaseline,
    RunFile,
    RunMetrics,
    RunRecord,
    compare_runs,
    find_baseline,
)
from .runtime import (
    DEFAULT_READY_TIMEOUT_S,
    Runtime,
    no_profiler,
)
from .slo import DEFAULT_SLO, Slo
from .suggest import suggestions


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalyseResult:
    run_id: str
    run_dir: Path
    measurement: Measurement
    diagnosis: Diagnosis
    suggestions: tuple[Suggestion, ...]
    report: Report
    answers_differ: bool = False  # --require-equal could not show equal answers


def analyse(
    project_path: Path,
    *,
    engine: Engine,
    runtime: Runtime,
    engine_args: Sequence[str] = (),
    verify_weights: bool = False,
    retain_responses: bool = True,
    ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S,
    trace: bool = False,
    counters: bool = False,
    slo: Slo = DEFAULT_SLO,
    require_equal: bool = False,
    baseline_answers: Path | None = None,
    require_gpu: str | None = None,
    require_driver: str | None = None,
) -> AnalyseResult:
    """Measure the current setup (plus any overrides) once and suggest which experiments to try.

    With ``trace``, a separate short profiled launch follows the clean measurement, never
    mixed into it: the engine's profiler must be enabled at launch and slows the engine.
    ``counters`` (which needs ``trace``) then profiles the trace's top kernels with Nsight
    Compute, one more launch each.
    """
    emit(PhaseStarted(phase="preflight"))
    if counters and not trace:
        raise AnalyseFailure("kernel counters pick their kernels from the trace: add trace")
    if trace and engine.tracing is None:
        raise AnalyseFailure(no_profiler(engine))
    if require_equal and not retain_responses:
        raise PreflightError(
            PROJECT_CONFIG_UNSUPPORTED,
            "--require-equal needs the answers: drop --no-retain-responses",
        )
    if require_equal and not (engine_args or baseline_answers):
        raise PreflightError(
            PROJECT_CONFIG_UNSUPPORTED,
            "--require-equal needs something to compare: an --engine-arg change, or "
            "--baseline-answers",
        )
    if require_gpu or require_driver:
        require_gpus(runtime.platform, runtime.gpus, require_gpu, require_driver)
    project = load_project(project_path, verify_weights=verify_weights)
    availability = require_available(engine, runtime)
    recorded = RunRecord.from_recorded(baseline_answers, project) if baseline_answers else None
    run_id = new_run_id()
    project_dir = Path(project_path).expanduser()
    runs_dir = project_dir / "runs"
    runs_dir.mkdir(mode=0o700, exist_ok=True)
    run_dir = runs_dir / run_id
    settings = settings_for(project, engine, engine_args)
    if warning := tool_calling_warning(project, engine, settings):
        emit(Note(text=warning))
    environment = describe_run(engine, runtime, project.artifact, availability)
    facts = workload_facts(project)
    try:
        (capture,) = run_workloads(
            project,
            engine=engine,
            runtime=runtime,
            settings=settings,
            run_dirs=[run_dir],
            run_projects=[project],
            retain_responses=retain_responses,
            ready_timeout_s=ready_timeout_s,
            steady_state=True,
        )
        measurement = summarize_run(
            capture, engine=engine, settings=settings, slo=slo, gpus=runtime.gpus, facts=facts
        )
    except AnalyseFailure as error:
        raise AnalyseFailure(f"your current setup did not run: {error}") from None
    environment = with_logged_version(engine, environment, capture.server_log)
    profiles = None
    if trace:
        profiles = add_profiles(
            project, engine, runtime, settings, run_dir, measurement, ready_timeout_s,
            counters=counters,
        )  # fmt: skip
        counted = profiles.counters.summary if profiles.counters else None
        measurement = replace(measurement, trace=profiles.trace.summary, counters=counted)
    run_file = run_file_for(
        project,
        engine=engine,
        runtime=runtime,
        settings=settings,
        engine_args=engine_args,
        retain_responses=retain_responses,
        image=capture.image,
        environment=environment,
    )
    emit(PhaseStarted(phase="compare"))
    advice = engine.reproducibility_advice(project.record.current_setup.source)
    compared = _compare_with_current(
        project_dir, project, run_dir, run_file, measurement, engine_args, recorded, advice
    )
    emit(PhaseStarted(phase="diagnose"))
    effective = engine.effective(settings, capture.resolved)
    gpu = serving_gpu(environment.identities, runtime.gpus)
    diagnosis = diagnose_run(
        measurement, effective, compared.answers, engine=engine, platform=environment.platform,
        gpu=gpu, version=environment.engine_version, facts=facts, resolved=capture.resolved,
    )  # fmt: skip
    changes: list[str] = []
    if (comparison := compared.comparison) is not None:
        baseline = compared.baseline
        pair = ("", "")
        if not comparison.recorded and baseline is not None:
            pair = (
                _bottleneck_name(baseline.metrics.diagnosis if baseline.metrics else None),
                NAMES[diagnosis.primary] if diagnosis.primary else "none found",
            )
        changes = render_changes(
            comparison, engine_args, pair, _caveats(compared, measurement), compared.tpot
        )
    situation = Situation(
        measurement=measurement, facts=facts, settings=effective, diagnosis=diagnosis
    )
    run = estimate_inputs(capture, runtime.gpus, measurement.kv_block_tokens)
    suggested = suggestions(
        engine,
        situation,
        current=settings_for(project, engine, ()),
        project_path=project_path,
        version=environment.engine_version,
        run=run,
    )
    emit(PhaseStarted(phase="write"))
    offered = {suggestion.entry.name for suggestion in suggested.suggestions}
    if run.prompt_blocks and measurement.kv_block_tokens and "prefix-caching" in offered:
        write_prompt_blocks(run_dir, capture, run, measurement.kv_block_tokens)
    outputs = RunOutputs(
        run_id=run_id,
        ran=environment.ran,
        measurement=measurement,
        diagnosis=diagnosis,
        suggested=suggested,
        rows=request_rows(capture),
        facts=facts,
        settings=settings,
        subject=Subject(
            "Your current setup" + overrides_suffix(engine_args), project.record.current_setup
        ),
        image=capture.image,
        slo=slo,
        engine=engine,
        gpu=gpu,
        source=project.record.current_setup.label,
        engine_args=engine_args,
        retained=retain_responses,
        changes=changes,
        compared=compared.lines,
        answers_file=compared.answers_file,
        trace=profiles.trace if profiles else None,
        counters=profiles.counters if profiles else None,
        resolved=capture.resolved,
    )
    report = write_derived(run_dir, outputs, run_file)
    emit(RunWritten(run_dir=run_dir))
    if profiles and profiles.interrupted:
        raise KeyboardInterrupt
    return AnalyseResult(
        run_id=run_id,
        run_dir=run_dir,
        measurement=measurement,
        diagnosis=diagnosis,
        suggestions=suggested.suggestions,
        report=report,
        answers_differ=require_equal and not compared.equal,
    )


def _answers_state(outcome: Outcome) -> AnswersState:
    if any(rate == 1.0 for rate in outcome.health.get("empty", ())):
        return "empty"
    if outcome.equal:
        return "equal"
    if outcome.changed is None:
        return "inconclusive"
    return "differ" if outcome.changed else "within_noise"


@dataclass(frozen=True, slots=True)
class _Compared:
    lines: Sequence[str]  # the report section, empty when nothing was compared
    answers_file: Path | None  # where the answers are side by side, when compared
    equal: bool  # the answers were shown equal
    answers: AnswersState | None  # the answers' verdict for the diagnosis; None when not compared
    comparison: Comparison | None = None
    baseline: RunRecord | None = None
    tpot: str | None = None  # why TPOT rose under a raised concurrency cap


def _bottleneck_name(recorded: Mapping[str, JsonValue] | None) -> str:
    """The primary bottleneck of a recorded diagnosis, in words."""
    if recorded is None:
        return "not diagnosed (a run from before 0.3.0)"
    primary = recorded.get("primary")
    return (
        NAMES.get(cast(Bottleneck, primary), primary) if isinstance(primary, str) else "none found"
    )


def _caveats(compared: _Compared, measurement: Measurement) -> list[str]:
    """Why the two runs' numbers may not be comparable."""
    found = []
    comparison, baseline = compared.comparison, compared.baseline
    if comparison is not None and comparison.differences:
        found.append(
            f"Measured on a different {', '.join(comparison.differences)}: the numbers may "
            "differ for that reason alone."
        )
    steady = measurement.window is not None and measurement.window.kind == "steady"
    if (
        steady
        and comparison is not None
        and not comparison.recorded
        and baseline is not None
        and baseline.metrics is not None
        and baseline.metrics.window is None
    ):
        version = baseline.tensward_version or "before 0.3.1"
        found.append(
            f"Your current setup's run was measured by Tensward {version} over a different "
            "window; run it again for a like-for-like comparison."
        )
    recorded_window = baseline.metrics.window if baseline and baseline.metrics else None
    if (
        measurement.window is not None
        and comparison is not None
        and not comparison.recorded
        and recorded_window is not None
        and (recorded_kind := str(recorded_window.get("kind"))) != measurement.window.kind
    ):
        kinds = {"steady": "steady state", "whole_run": "whole run"}
        this = kinds.get(measurement.window.kind, measurement.window.kind)
        other = kinds.get(recorded_kind, recorded_kind)
        found.append(
            f"The two runs were measured over different windows (this run: {this}; the "
            f"current setup's run: {other}); compare with care."
        )
    return found


def _compare_with_current(
    project_dir: Path,
    project: ResolvedProject,
    run_dir: Path,
    run_file: RunFile,
    measurement: Measurement,
    engine_args: Sequence[str],
    recorded: RunRecord | None,
    advice: str,
) -> _Compared:
    """The report section comparing a run with the recorded answers, or, when it changed the
    setup, with the newest run of the current setup; where its answers are side by side;
    whether the answers were shown equal; and the answers' verdict for the diagnosis (None when
    nothing was compared). A failure to compare is a line of the report, never a
    failure of the run."""
    if not engine_args and recorded is None:
        return _Compared((), None, True, None)
    try:
        metrics = RunMetrics.model_validate_json(json.dumps(asdict(measurement)))
        candidate = RunRecord.from_run(run_dir, run_file, metrics=metrics)
        baseline = recorded or find_baseline(project_dir, project)
        comparison = compare_runs(baseline, candidate, project)
        answers = write_comparison(project_dir, comparison, baseline, candidate, project)
        return _Compared(
            render_section(comparison, candidate, baseline, project, advice=advice),
            answers,
            comparison.outcome.equal is True,
            _answers_state(comparison.outcome),
            comparison,
            baseline,
            tpot_note(baseline, candidate),
        )
    except NoBaseline as error:
        return _Compared([str(error)], None, False, None)
    except (PreflightError, OSError) as error:
        return _Compared(
            [f"Comparison with your current setup skipped: {error}"], None, False, None
        )
