"""``tensward analyse``: serve the registered model, offer the registered prompts, diagnose.

The flow is: load the verified project, start a server through a :class:`Runtime`, wait until it
is ready, run the declared warmup, scrape the engine's metrics, run the declared workload
through the streaming client, scrape the metrics again, write the report, and always stop the
server. Everything written lands in ``<project>/runs/<run_id>/``. Nothing here invents a value:
whatever the server or the configuration did not establish is reported as not measured.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Collection, Mapping, Sequence

from pydantic import JsonValue

from .classify import NAMES, AnswersState, Diagnosis, diagnose_run, name_of
from .compare import Outcome
from .context import current as analysis
from .engines import Engine
from .environment import (
    describe_run,
    require_available,
    require_gpus,
    with_logged_version,
)
from .errors import PROJECT_CONFIG_UNSUPPORTED, AnalyseFailure, PreflightError
from .files import new_run_id
from .kept import pinned_baseline, read_kept
from .measure import (
    current_setup_failure,
    longest_request,
    run_file_for,
    run_workloads,
    serving_gpu,
    serving_identity,
    summarize_run,
    with_probes,
)
from .measurement import Measurement
from .playbook import Situation, Suggestion
from .profiling.run import add_profiles
from .progress import Note, PhaseStarted, RunWritten, emit
from .project import (
    ResolvedProject,
    fit_offer_name,
    load_project,
    project_fit_advice,
    settings_for,
    tool_calling_warning,
    workload_facts,
)
from .report.build import RunOutputs, Subject, overrides_suffix
from .report.compare import render_changes, render_section, tpot_note, write_comparison
from .report.model import Report
from .rundir import (
    RunCapture,
    empty_answers,
    estimate_inputs,
    rendered,
    request_rows,
    write_derived,
    write_prompt_blocks,
)
from .runs import (
    Comparison,
    NoBaseline,
    RunFile,
    RunMetrics,
    RunRecord,
    baseline_resolved,
    compare_runs,
    find_baseline,
)
from .runtime import (
    DEFAULT_READY_TIMEOUT_S,
    Runtime,
    no_profiler,
)
from .settings import Settings
from .slo import DEFAULT_SLO, Slo
from .suggest import suggestions

FIT_PINNED_NOTE = (
    "Your command leaves the placement of layers to the engine's --fit, which cannot be "
    "reproduced exactly with a fixed command. This run holds the placement the first start "
    "chose (its layers, context, slots and KV layout, with --fit off); every step is compared "
    "against it. The first start's log is in user-fit/."
)
USER_FIT_DIR = "user-fit"


def _launch(
    project: ResolvedProject,
    engine: Engine,
    runtime: Runtime,
    settings: Settings,
    run_dir: Path,
    retain_responses: bool,
    ready_timeout_s: float,
    reuse_dir: bool = False,
) -> RunCapture:
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
        reuse_dirs=reuse_dir,
    )
    return capture


def _set_aside(run_dir: Path) -> None:
    """Move what a first launch wrote into ``run_dir`` under ``user-fit/``, so the run directory
    can take the pinned launch that replaces it."""
    kept = list(run_dir.iterdir())
    aside = run_dir / USER_FIT_DIR
    aside.mkdir(mode=0o700)
    for path in kept:
        shutil.move(path, aside / path.name)


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalyseResult:
    run_id: str
    run_dir: Path
    measurement: Measurement
    diagnosis: Diagnosis
    suggestions: tuple[Suggestion, ...]
    report: Report
    answers_differ: bool = False  # --require-equal could not show equal answers


def tried_args(
    project_dir: Path, engine_args: Sequence[str], pinned: Collection[str] = ()
) -> Sequence[str]:
    """The engine arguments a run adds to the kept setup (and to what the baseline's launch
    pinned), which name it; all of them when it adds none."""
    kept = read_kept(project_dir)
    held = {*(kept.engine_args if kept is not None else ()), *pinned}
    return [arg for arg in engine_args if arg not in held] or engine_args


GIB = 2**30


def _launched(
    project: ResolvedProject,
    engine: Engine,
    project_dir: Path,
    settings: Settings,
    engine_args: Sequence[str],
) -> Settings:
    """The settings the server is started with: the baseline's pins hold under a change."""
    if not engine_args:
        return settings
    current = settings_for(project, engine, ())
    return engine.pinned(settings, current, baseline_resolved(project_dir, project))


def unstartable(
    project: ResolvedProject,
    engine: Engine,
    runtime: Runtime,
    engine_args: Sequence[str],
    project_dir: Path,
) -> str | None:
    """Stop before any duration is announced when the setup cannot start on this GPU: the
    finding and its Next menu when there is a fix to try, else a failure."""
    settings = settings_for(project, engine, engine_args)
    launched = _launched(project, engine, project_dir, settings, engine_args)
    fit_offers: list[Settings] = []
    unfit = project_fit_advice(project, engine, launched, runtime, project_dir, fit_offers)
    if not (unfit and unfit[1]):
        return None
    message = cannot_start_message(project, engine, runtime, engine_args, launched, fit_offers)
    menu = _try_menu(project_dir, engine_args)
    if not (fit_offers and menu):
        raise AnalyseFailure(message + menu)
    return message + menu


def wont_serve(project_dir: Path, engine: Engine, runtime: Runtime) -> str | None:
    """``tensward serve`` of the registered setup that cannot start on this GPU, as analyse
    finds it: the finding and a Next menu of the fix to test and serving anyway; None when it
    can start."""
    from .analyse_next import fix_choices
    from .cli_try import next_lines

    project = load_project(project_dir, verify_weights=False)
    settings = settings_for(project, engine, ())
    fit_offers: list[Settings] = []
    unfit = project_fit_advice(project, engine, settings, runtime, project_dir, fit_offers)
    if not (unfit and unfit[1]):
        return None
    message = cannot_start_message(project, engine, runtime, (), settings, fit_offers)
    choices = [*fix_choices(project_dir), ("tensward serve --anyway", "start it anyway")]
    return "\n".join([message, "", *next_lines(choices)])


FIX_WORDS = {
    "cpu-offload": "keep part of the model in CPU memory",
    "offload-experts": "keep part of the model in CPU memory",
    "offload-layers": "keep part of the model in CPU memory",
    "fit-layers": "drop --n-gpu-layers and let llama.cpp place the layers",
    "fewer-sequences": "serve fewer requests at once",
    "no-cuda-graphs": "turn off CUDA graphs (eager mode)",
    "shorter-context": "use a shorter context",
}


def cannot_start_message(
    project: ResolvedProject,
    engine: Engine,
    runtime: Runtime,
    engine_args: Sequence[str],
    launched: Settings,
    fit_offers: Sequence[Settings],
) -> str:
    """One plain line for a setup that cannot start on this GPU: the sizes, and the fix."""
    devices, _ = runtime.platform.select(runtime.platform.detect(), runtime.gpus)
    gpu = devices[0].total_bytes / GIB if devices else None
    weights = project.artifact.metadata.weight_bytes / GIB
    what = "your current setup" if not engine_args else "this setup"
    if gpu is not None and weights > gpu:
        line = (
            f"Your model is {weights:.1f} GiB and this GPU has {gpu:.1f} GiB, so {what} won't "
            "start."
        )
    elif gpu is not None:
        line = (
            f"{what.capitalize()} won't start on this GPU: it needs more memory than the "
            f"{gpu:.1f} GiB available."
        )
    else:
        line = f"{what.capitalize()} won't start on this GPU: it needs more memory than it has."
    if fit_offers:
        kind = fit_offer_name(engine.engine_args_between(launched, fit_offers[0]))
        line += f" Tensward has a fix to try: {FIX_WORDS[kind]}."
    return line


def _try_menu(project_path: Path, engine_args: Sequence[str]) -> str:
    """The Next menu after a setup that cannot start: testing the offered fix."""
    if engine_args or analysis().try_step:
        return ""
    from .analyse_next import fix_lines

    lines = fix_lines(project_path)
    return "\n\n" + "\n".join(lines) if lines else ""


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
    probes: bool = True,
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
    project_dir = Path(project_path).expanduser()
    runs_dir = project_dir / "runs"
    runs_dir.mkdir(mode=0o700, exist_ok=True)
    pins = pinned_baseline(project_dir, project, engine) if engine_args else frozenset()
    run_id = new_run_id(runs_dir, change=tried_args(project_dir, engine_args, pins))
    run_dir = runs_dir / run_id
    settings = settings_for(project, engine, engine_args)
    if warning := tool_calling_warning(project, engine, settings):
        emit(Note(text=warning))
    environment = describe_run(engine, runtime, project.artifact, availability, project.anatomy)
    facts = workload_facts(project)
    placement_note: list[str] = []
    launched = _launched(project, engine, project_dir, settings, engine_args)
    fit_offers: list[Settings] = []
    if unfit := project_fit_advice(project, engine, launched, runtime, project_path, fit_offers):
        text, fails = unfit
        if fails:
            raise AnalyseFailure(
                cannot_start_message(project, engine, runtime, engine_args, launched, fit_offers)
                + _try_menu(project_path, engine_args)
            )
        emit(Note(text=text))
    try:
        capture = _launch(
            project, engine, runtime, launched, run_dir, retain_responses, ready_timeout_s
        )
        redo = None if engine_args else engine.repinned(settings, capture.resolved)
        if redo is not None:
            emit(Note(text=FIT_PINNED_NOTE))
            placement_note.append(FIT_PINNED_NOTE)
            _set_aside(run_dir)
            settings = launched = redo
            analysis().held = tuple(
                engine.engine_args_between(settings_for(project, engine, ()), redo)
            )
            analysis().relaunched()
            capture = _launch(
                project,
                engine,
                runtime,
                settings,
                run_dir,
                retain_responses,
                ready_timeout_s,
                reuse_dir=True,
            )
        measurement = summarize_run(
            capture,
            engine=engine,
            settings=settings,
            slo=slo,
            gpus=runtime.gpus,
            facts=facts,
            platform=runtime.platform,
        )
        environment = with_logged_version(engine, environment, capture.server_log)
        capture, measurement, probed = with_probes(
            capture, measurement, engine=engine, runtime=runtime, settings=launched, slo=slo,
            facts=facts, platform=environment.platform, engine_args=engine_args,
            ready_timeout_s=ready_timeout_s, launch=probes,
        )  # fmt: skip
        facts = replace(facts, rendered=rendered(capture))
    except AnalyseFailure as error:
        raise current_setup_failure(
            error,
            engine=engine,
            project=project,
            settings=settings,
            platform=runtime.platform.name,
            engine_args=engine_args,
            run_dir=run_dir,
        ) from None
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
    analysis().topology = runtime.platform.memory_topology
    emit(PhaseStarted(phase="compare"))
    advice = engine.reproducibility_advice(
        project.record.current_setup.source, project.record.current_setup.engine_settings
    )
    unfit = project_fit_advice(project, engine, settings_for(project, engine, ()), runtime)
    compared = _compare_with_current(
        project_dir, project, run_dir, run_file, measurement, engine_args, recorded, advice,
        current_cannot_start=bool(engine_args and unfit and unfit[1]),
    )  # fmt: skip
    emit(PhaseStarted(phase="diagnose"))
    effective = engine.effective(settings, capture.resolved)
    gpu = serving_gpu(environment.identities, runtime.gpus)
    diagnosis = diagnose_run(
        measurement, effective, compared.answers, engine=engine, platform=environment.platform,
        gpu=gpu, version=environment.engine_version, facts=facts, resolved=capture.resolved,
    )  # fmt: skip
    placed = engine.placement_report(
        project.anatomy,
        measurement,
        capture.resolved,
        capture.server_log,
        environment.platform,
        longest_request(project, capture, facts.output_tokens),
        capture.host,
        settings,
        facts,
        capture.host_window,
        request_rows(capture),
    )
    changes: list[str] = list(placement_note)
    if (comparison := compared.comparison) is not None:
        baseline = compared.baseline
        pair = ("", "")
        if not comparison.recorded and baseline is not None:
            pair = (
                _bottleneck_name(baseline.metrics.diagnosis if baseline.metrics else None),
                name_of(diagnosis.primary) if diagnosis.primary else "none found",
            )
        changes = render_changes(
            comparison, engine_args, pair, _caveats(compared, measurement), compared.tpot
        )
    situation = Situation(
        measurement=measurement, facts=facts, settings=effective, diagnosis=diagnosis
    )
    run = estimate_inputs(
        capture, runtime.gpus, measurement.kv_block_tokens, environment.engine_version
    )
    context = analysis()
    context.placement, context.probes = placed, probed.block if probed else None
    context.gpu_identity = serving_identity(environment.identities, runtime.gpus)
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
        empty_answers=empty_answers(capture),
        capacity=capture.capacity,
        placement=placed,
        probes=probed.block if probed else None,
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
    if outcome.thought and outcome.equal is None:
        return "thinking"
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
    if not isinstance(primary, str):
        return "none found"
    return name_of(primary) if primary in NAMES else primary


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
    current_cannot_start: bool = False,
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
        if current_cannot_start:
            return _Compared((), None, False, None)
        return _Compared([str(error)], None, False, None)
    except (PreflightError, OSError) as error:
        return _Compared(
            [f"Comparison with your current setup skipped: {error}"], None, False, None
        )
