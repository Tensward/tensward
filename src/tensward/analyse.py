"""``tensward analyse``: serve the registered model, offer the registered prompts, diagnose.

The flow is: load the verified project, start a server through a :class:`Runtime`, wait until it
is ready, run the declared warmup, scrape the engine's metrics, run the declared workload
through the streaming client, scrape the metrics again, write the report, and always stop the
server. Everything written lands in ``<project>/runs/<run_id>/``. Nothing here invents a value:
whatever the server or the configuration did not establish is reported as not measured.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Mapping, Sequence

import httpx
from pydantic import JsonValue

from . import __version__
from .capture import CapturedResponse, CapturingTransport
from .ceilings import Measured, Unavailable, compute_ceilings, gpu_spec
from .classify import NAMES, Diagnosis, classify
from .client import (
    UNLIMITED_CONNECTIONS,
    HttpxTransport,
    RequestRecord,
    chat_body,
    run_workload,
    steady_lead,
)
from .counters import (
    COUNTER_REQUESTS,
    COUNTER_TIME_LIMIT_S,
    REQUIRED_GROUP,
    CountersSummary,
    KernelCounters,
    Launch,
    group_command,
    kernel_counters,
    match_ncu_name,
    memory_returned,
    merge_groups,
    parse_raw_csv,
    qualify,
    render_counters,
    select_kernels,
)
from .engines import Engine
from .engines.protocol import EngineSignals, Settings
from .environment import (
    RunEnvironment,
    describe_run,
    require_available,
    with_logged_version,
)
from .errors import GPU_MISMATCH, PROJECT_CONFIG_UNSUPPORTED, PreflightError
from .extensions import load_extender
from .files import new_run_id, write_json, write_jsonl
from .images import ImageSource
from .inputs import PromptEntry
from .measurement import (
    Measurement,
    Peaks,
    Window,
    group_stats,
    measurement_window,
    summarize,
    window_share,
)
from .platforms import DeviceIdentity, Platform, detect_devices
from .playbook import Entry, LowBar, WorkloadFacts, applicable, blocked_changes, ranked
from .progress import quarters, say
from .project import ResolvedProject, load_project, project_fit, runs_remote_code
from .quality import (
    RUN_RECORD,
    Comparison,
    NoBaseline,
    Outcome,
    RunFile,
    RunRecord,
    compare_runs,
    find_baseline,
    render_changes,
    render_section,
    write_comparison,
)
from .report import (
    FEEDBACK_LINE,
    Subject,
    cast_to_float16,
    checks,
    diagnosis_headline,
    engine_output_rate,
    overrides_suffix,
    put_first,
    ran_line,
    render_diagnosis,
    render_markdown,
    render_next_steps,
    window_checks,
)
from .runtime import (
    DEFAULT_READY_TIMEOUT_S,
    RunningServer,
    Runtime,
    RuntimeFailure,
    ServeSpec,
    generate_api_key,
    log_tail,
    select_loopback_port,
    wait_until_ready,
)
from .slo import DEFAULT_SLO, Slo
from .thresholds import THRESHOLDS
from .toolcalls import ToolCallStats, tool_call_stats
from .trace import (
    QUANT_KERNEL_SYMBOLS,
    Trace,
    TraceSummary,
    TraceUnavailable,
    read_trace,
    render_trace,
    summarize_trace,
)

SIGNAL_POLL_S = 1.0
WAVE_OUTLASTED = (
    "the start-up wave had not finished when the last request was sent; measuring the whole run"
)
SAVED_LOG_CHARS = 10_000_000  # the run directory keeps at most this much of the server log
TRACE_REQUESTS = 16  # the traced slice is short: profiler overhead and trace size grow with it
TRACE_TIME_LIMIT_S = 120.0  # hard bound on the profiled requests


class AnalyseFailure(Exception):
    """The run could not be completed; the message is safe to print."""


@dataclass(frozen=True, slots=True)
class AnalyseResult:
    run_id: str
    run_dir: Path
    measurement: Measurement
    suggestions: tuple[tuple[Entry, str], ...]
    not_applicable: list[tuple[str, str]]  # (entry name, the gate's or the engine's reason)
    source: str  # where the measured current setup came from
    ran: str  # what ran, as the report's first line says
    suggestions_text: str  # the next steps as rendered into the report
    diagnosis: Diagnosis
    diagnosis_line: str  # the one-line bottleneck verdict for the terminal
    comparison_line: str | None = None  # where the answers are, when compared
    answers_differ: bool = False  # --require-equal could not show equal answers
    changes_text: str = ""  # the block on what changed against the current setup, when compared


# --------------------------------------------------------------------------------------
# Building the run
# --------------------------------------------------------------------------------------


def settings_for(project: ResolvedProject, engine: Engine, engine_args: Sequence[str]) -> Settings:
    """The customer's current setup as settings, with ``KEY=VALUE`` engine flags applied."""
    settings = project.record.current_setup.engine_settings
    facts = WorkloadFacts.of(project)
    if settings.tool_parser is None and (facts.offers_tools or settings.tool_calling):
        settings = replace(settings, tool_parser=engine.default_tool_parser(project.artifact.path))
    return apply_engine_args(engine, settings, engine_args)


def apply_engine_args(engine: Engine, settings: Settings, engine_args: Sequence[str]) -> Settings:
    """``settings`` with each ``KEY=VALUE`` engine flag applied in turn."""
    trusted = runs_remote_code(settings.extra_args)
    for text in engine_args:
        try:
            settings = engine.with_engine_arg(settings, text)
        except ValueError as error:
            raise AnalyseFailure(str(error)) from None
    if runs_remote_code(settings.extra_args) and not trusted:
        raise AnalyseFailure(
            "trust-remote-code changes which code the model runs, so it is part of the model's "
            "identity: register the project again with it in --current"
        )
    return settings


async def _count_prompt_tokens(
    client: httpx.AsyncClient,
    engine: Engine,
    server_url: str,
    model: str,
    prompts: Sequence[PromptEntry],
    images: ImageSource,
) -> dict[str, int]:
    """Each prompt's token count by the server's own tokenizer, images included; empty if it
    has none."""
    counts: dict[str, int] = {}
    try:
        for entry in prompts:
            body = (
                {"prompt": entry.prompt}
                if entry.messages is None
                else await chat_body(entry.chat, images)
            )
            response = await client.post(
                f"{server_url}{engine.tokenize_path}", json={"model": model, **body}
            )
            response.raise_for_status()
            counts[entry.id] = int(response.json()["count"])
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return {}
    return counts


async def _scrape(engine: Engine, client: httpx.AsyncClient, server_url: str) -> str | None:
    """The engine's metrics text, on the run's own client: a client made per scrape loads the CA
    bundle each time, which blocks the event loop for milliseconds and skews the timings."""
    try:
        response = await client.get(f"{server_url}{engine.metrics_path}")
        response.raise_for_status()
        return response.text
    except httpx.HTTPError:
        return None


# --------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------


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
    if counters and not trace:
        raise AnalyseFailure("kernel counters pick their kernels from the trace: add trace")
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
        _require_gpus(runtime.platform, runtime.gpus, require_gpu, require_driver)
    project = load_project(project_path, verify_weights=verify_weights)
    availability = require_available(engine, runtime)
    recorded = RunRecord.from_recorded(baseline_answers, project) if baseline_answers else None
    run_id = new_run_id()
    project_dir = Path(project_path).expanduser()
    runs_dir = project_dir / "runs"
    runs_dir.mkdir(mode=0o700, exist_ok=True)
    run_dir = runs_dir / run_id
    settings = settings_for(project, engine, engine_args)
    title = "Your current setup" + overrides_suffix(engine_args)
    environment = describe_run(engine, runtime, project.artifact, availability)
    (measurement,) = measure(
        project,
        engine=engine,
        runtime=runtime,
        settings=settings,
        run_dirs=[run_dir],
        retain_responses=retain_responses,
        ready_timeout_s=ready_timeout_s,
        subject=Subject(title, project.record.current_setup),
        slo=slo,
        environment=environment,
    )
    log = run_dir / "server.log"
    if log.is_file():
        environment = with_logged_version(engine, environment, log.read_text("utf-8", "replace"))
    inherited = {
        name: value for name, value in os.environ.items() if name not in settings.extra_env
    }
    run_file = RunFile(
        schema_version="1",
        snapshot_id=project.record.snapshot_id,
        engine_args=tuple(engine_args),
        settings=asdict(settings),
        retained_responses=retain_responses,
        temperature=project.settings.workload.temperature,
        runtime=runtime.label,
        gpus=runtime.gpus,
        image=measurement.image,
        inherited_env=engine.inherited_env(inherited) if runtime.inherits_environment else {},
        gpus_identity=tuple(DeviceIdentity(**gpu) for gpu in environment.identities),
        engine=environment.engine,
        engine_version=environment.engine_version,
        platform=environment.platform,
        format=environment.format,
        tensward_version=__version__,
    )
    write_json(run_dir / RUN_RECORD, run_file.model_dump(mode="json"))
    if trace:
        measurement = _add_trace(
            project, engine, runtime, settings, run_dir, measurement, ready_timeout_s, counters
        )
    compared = _compare_with_current(project_dir, project, run_dir, engine_args, recorded)
    diagnosis = classify(measurement, settings, compared.answers)
    changes: list[str] = []
    if (comparison := compared.comparison) is not None:
        baseline = compared.baseline
        pair = ("", "")
        if not comparison.recorded and baseline is not None:
            pair = (
                _bottleneck_name(baseline.metrics.diagnosis if baseline.metrics else None),
                NAMES[diagnosis.primary] if diagnosis.primary else "none found",
            )
        changes = render_changes(comparison, engine_args, pair, _caveats(compared, measurement))
    facts = WorkloadFacts.of(project)
    suggestions: list[tuple[Entry, str]] = []
    applied: dict[str, Settings] = {}
    found, gated = applicable(
        engine.playbook(), measurement, facts, settings, allow_quality_changes=True, near=True
    )
    not_applicable = [(entry.name, why) for entry, why in gated]
    for entry, reason in ranked(found):
        proposed, note = engine.consistent(settings, entry.apply(measurement, facts, settings))
        if why := engine.unstartable(settings, proposed):
            not_applicable.append((entry.name, why))
        elif proposed != settings:
            applied[entry.name] = proposed
            if note:
                reason = (
                    reason.extended(f"the command {note}")
                    if isinstance(reason, LowBar)
                    else f"{reason}; the command {note}"
                )
            suggestions.append((entry, reason))

    current = settings_for(project, engine, ())

    def command_for(entry: Entry) -> str | None:
        if not engine.engine_args_between(settings, applied[entry.name]):
            return None
        return suggestion_command(engine, project_path, current, applied[entry.name])

    write_json(
        run_dir / "metrics.json",
        {
            **asdict(measurement),
            "diagnosis": {
                **asdict(diagnosis),
                "names": NAMES,
                "thresholds": {t.name: asdict(t) for t in THRESHOLDS},
                "not_applicable": [{"entry": n, "why": w} for n, w in not_applicable],
            },
        },
    )
    text = render_next_steps(
        diagnosis,
        suggestions,
        command_for,
        not_applicable,
        blocked_changes(engine.playbook(), measurement, settings),
        retained=retain_responses,
    )
    report = run_dir / "report.md"
    report.write_text(
        put_first(
            report.read_text("utf-8"),
            ("\n".join(changes) + "\n\n" if changes else "")
            + render_diagnosis(diagnosis, _serving_gpu(environment.identities, runtime.gpus))
            + "\n"
            + text,
        )
        + compared.section
        + "\n"
        + FEEDBACK_LINE
        + "\n",
        "utf-8",
    )
    say(f"done: {run_dir}")
    return AnalyseResult(
        run_id,
        run_dir,
        measurement,
        tuple(suggestions),
        not_applicable,
        project.record.current_setup.label,
        ran_line(environment.ran, measurement.served_as_float16),
        text,
        diagnosis,
        diagnosis_headline(diagnosis),
        compared.line,
        require_equal and not compared.equal,
        "\n".join(changes),
    )


def _bare_name(platform: Platform, name: str) -> str:
    return name.strip().casefold().removeprefix(platform.label.casefold() + " ")


def _serving_gpu(
    identities: Sequence[Mapping[str, Any]], selected: tuple[str, ...] | None
) -> str | None:
    """The name of the device serving: the first selected one, else device 0."""
    index = int(selected[0]) if selected else 0
    return str(identities[index]["name"]) if index < len(identities) else None


def _require_gpus(
    platform: Platform, selection: tuple[str, ...] | None, name: str | None, driver: str | None
) -> None:
    """Refuse a machine whose selected GPUs (all of them without a selection) are not the card
    ``name`` or do not run ``driver`` (a dotted prefix of its version also matches)."""
    gpus = detect_devices()
    if not gpus:
        raise PreflightError(GPU_MISMATCH, "no supported accelerator detected")
    if why := platform.selection_refusal(selection or ()):
        raise PreflightError(GPU_MISMATCH, f"{why} with --require-gpu or --require-driver")
    matched, missing = platform.select(gpus, selection)
    if missing:
        listed = ", ".join(gpu.index for gpu in gpus)
        raise PreflightError(
            GPU_MISMATCH, f"GPU {missing[0]} selected, but the GPUs here are {listed}"
        )
    for gpu in matched:
        actual = f"GPU {gpu.index} is {gpu.name} with driver {gpu.driver}, but"
        if name is not None and _bare_name(platform, gpu.name) != _bare_name(platform, name):
            raise PreflightError(GPU_MISMATCH, f"{actual} --require-gpu wants {name}")
        if driver is not None and not (
            gpu.driver is not None and (gpu.driver + ".").startswith(driver + ".")
        ):
            raise PreflightError(GPU_MISMATCH, f"{actual} --require-driver wants {driver}")


def _answers_state(outcome: Outcome) -> str:
    if any(rate == 1.0 for rate in outcome.health.get("empty", ())):
        return "empty"
    if outcome.equal:
        return "equal"
    if outcome.changed is None:
        return "inconclusive"
    return "differ" if outcome.changed else "within_noise"


@dataclass(frozen=True, slots=True)
class _Compared:
    section: str  # the report section, empty when nothing was compared
    line: str | None  # where the answers are, for the terminal
    equal: bool  # the answers were shown equal
    answers: str | None  # the answers' verdict for the diagnosis; None when not compared
    comparison: Comparison | None = None
    baseline: RunRecord | None = None


def _bottleneck_name(recorded: Mapping[str, JsonValue] | None) -> str:
    """The primary bottleneck of a recorded diagnosis, in words."""
    if recorded is None:
        return "not diagnosed (a run from before 0.3.0)"
    primary = recorded.get("primary")
    return NAMES.get(primary, primary) if isinstance(primary, str) else "none found"


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
    engine_args: Sequence[str],
    recorded: RunRecord | None,
) -> _Compared:
    """The report section comparing a run with the recorded answers, or, when it changed the
    setup, with the newest run of the current setup; the one-line verdict for the terminal;
    whether the answers were shown equal; and the answers' verdict for the diagnosis (None when
    nothing was compared). A failure to compare is a line of the report, never a
    failure of the run."""
    if not engine_args and recorded is None:
        return _Compared("", None, True, None)
    try:
        candidate = RunRecord.from_run(run_dir, RunFile.read(run_dir))
        baseline = recorded or find_baseline(project_dir, project)
        comparison = compare_runs(baseline, candidate, project)
        answers = write_comparison(project_dir, comparison, baseline, candidate, project)
        section = "\n".join(render_section(comparison, candidate, baseline, project))
        return _Compared(
            f"\n{section}\n",
            f"answers side by side: {answers}",
            comparison.outcome.equal is True,
            _answers_state(comparison.outcome),
            comparison,
            baseline,
        )
    except NoBaseline as error:
        return _Compared(f"\n{error}\n", None, False, None)
    except (PreflightError, OSError) as error:
        return _Compared(
            f"\nComparison with your current setup skipped: {error}\n", None, False, None
        )


def suggestion_command(
    engine: Engine, project: Path, current: Settings, suggested: Settings
) -> str:
    """The ``tensward analyse`` command that measures ``suggested``: every change from the current
    setup once, so an override already given is replaced, not repeated."""
    words = ["tensward", "analyse", "--project", str(project)]
    for text in engine.engine_args_between(current, suggested):
        words += ["--engine-arg", text]
    return shlex.join(words)


async def _poll_signals(
    engine: Engine, client: httpx.AsyncClient, server_url: str, peaks: Peaks
) -> None:
    while True:
        generation = peaks.generation
        text = await _scrape(engine, client, server_url)
        if text is not None and peaks.generation == generation:
            peaks.observe(engine.parse_signals(text))
        await asyncio.sleep(SIGNAL_POLL_S)


async def _window(
    engine: Engine,
    client: httpx.AsyncClient,
    url: str,
    peaks: Peaks,
    reached: asyncio.Event,
    log: Callable[[], str],
) -> tuple[str | None, int, str]:
    """Wait for the start-up wave to finish; then restart the peaks and scrape the engine, so
    the engine's numbers cover the steady-state window. Returns the scrape, the window's start
    and the server log as it was at that moment."""
    await reached.wait()
    log_text = await asyncio.to_thread(log)
    start_ns = time.monotonic_ns()
    peaks.reset()
    return await _scrape(engine, client, url), start_ns, log_text


def measure(
    project: ResolvedProject,
    *,
    engine: Engine,
    runtime: Runtime,
    settings: Settings,
    run_dirs: Sequence[Path],
    retain_responses: bool,
    ready_timeout_s: float,
    environment: RunEnvironment,
    subject: Subject = Subject(),
    slo: Slo = DEFAULT_SLO,
    run_projects: Sequence[ResolvedProject] | None = None,
    steady_state: bool = True,
) -> list[Measurement]:
    """Start the server once, warm it up, run the workload into each of ``run_dirs`` in turn,
    stop it, and write every run directory. Repeats share one launch.

    ``run_projects`` serves extensions (plain ``analyse`` has one run): it gives each run its
    own variant of ``project`` (a smaller or busier workload); every variant must keep the
    project's model and settings. A closed-loop run offers a start-up wave first and measures
    after it, unless ``steady_state`` is False.
    """
    for run_dir in run_dirs:
        run_dir.mkdir(mode=0o700, parents=True)
    try:
        return asyncio.run(
            _measure(
                project,
                engine=engine,
                runtime=runtime,
                settings=settings,
                run_dirs=run_dirs,
                retain_responses=retain_responses,
                ready_timeout_s=ready_timeout_s,
                subject=subject,
                slo=slo,
                run_projects=run_projects or [project] * len(run_dirs),
                environment=environment,
                steady_state=steady_state,
            )
        )
    except AnalyseFailure as error:
        if subject.setup is None:
            raise
        raise AnalyseFailure(f"your current setup did not run: {error}") from None


@dataclass(slots=True)
class _Run:
    """The raw outcome of one workload run against a live server."""

    run_dir: Path
    project: ResolvedProject  # what this run offered
    transport: CapturingTransport
    records: Sequence[RequestRecord]
    lead_records: Sequence[RequestRecord]  # the start-up wave, outside the window
    window: Window  # what the client's throughput covers
    window_ns: int  # when the engine's measurement window opened
    after_ns: int  # when the scrape that closed the engine's window began
    before: str | None
    after: str | None  # the engine as the window closed
    final: str | None  # the engine after the run, for what only it shows
    fell_back: bool  # the engine could not be read at the last request, so ``after`` is final
    outlasted: bool  # the start-up wave was still running at the last request
    peaks: Peaks
    server_log: str
    log_start: str  # the server log when the window opened
    steady_state: bool
    image: str | None  # the container image that ran, if it was a container


@dataclass(frozen=True, slots=True)
class _Live:
    """A started, ready server and the clients that talk to it."""

    server: RunningServer
    client: httpx.AsyncClient
    http: HttpxTransport
    images: ImageSource


@asynccontextmanager
async def _serving(
    engine: Engine, runtime: Runtime, spec: ServeSpec, *, ready_timeout_s: float,
    record_dirs: Sequence[Path], phase: str, images: ImageSource, may_exit: bool = False,
) -> AsyncIterator[_Live]:  # fmt: skip
    """Start the server, wait until it is ready, and always stop it; log it into ``record_dirs``.
    A server that exits during the run is a failure unless ``may_exit`` (a profiler stops it).
    ``phase`` names the launch in the progress lines; ``images`` makes the requests' image data."""
    say(f"{phase}: launching {engine.name} ({runtime.label})")
    try:
        server = runtime.start(spec)
    except RuntimeFailure as error:
        raise AnalyseFailure(str(error)) from None
    try:
        for record_dir in record_dirs:
            write_json(
                record_dir / "serve.json",
                {"spec": spec.public_dict(), "identity": server.identity()},
            )
        await wait_until_ready(
            server,
            engine,
            api_key=spec.api_key,
            served_model_name=spec.served_model_name,
            timeout_s=ready_timeout_s,
        )
        headers = {"Authorization": f"Bearer {spec.api_key}"}
        options: dict[str, Any] = {"headers": headers, "follow_redirects": False}
        # The metrics poller and scrapes get their own small pool, so a saturated workload pool
        # can never stall them (or the other way round).
        async with (
            httpx.AsyncClient(limits=httpx.Limits(max_connections=4), **options) as client,
            httpx.AsyncClient(limits=UNLIMITED_CONNECTIONS, **options) as workload_client,
        ):
            http = HttpxTransport(client=workload_client, base_url=server.endpoint_url)
            yield _Live(server, client, http, images)
        if not may_exit and not server.is_running():
            raise AnalyseFailure(f"the server died during the run\n{log_tail(server.log_text())}")
    except RuntimeFailure as error:
        raise AnalyseFailure(str(error)) from None
    finally:
        say(f"{phase}: stopping {engine.name}")
        server.stop()
        # The whole log (its end, if huge) keeps the startup lines, such as the kernels chosen.
        text = server.log_text()[-SAVED_LOG_CHARS:].replace(spec.api_key, "***")
        for record_dir in record_dirs:
            (record_dir / "server.log").write_text(text, encoding="utf-8")


def _spec_for(
    project: ResolvedProject,
    engine: Engine,
    settings: Settings,
    run_id: str,
    trace_dir: Path | None,
) -> ServeSpec:
    return ServeSpec(
        engine=engine,
        model_dir=project.artifact.path,
        served_model_name=f"tensward-{run_id.lower().replace('z-', '-')}",
        settings=settings,
        port=select_loopback_port(),
        api_key=generate_api_key(),
        trace_dir=trace_dir,
    )


async def _warm_up(project: ResolvedProject, spec: ServeSpec, live: _Live, run_id: str) -> None:
    settings = project.settings
    if settings.warmup_requests > 0:
        say(f"warmup: {settings.warmup_requests} requests")
        warmup = settings.workload.model_copy(update={"request_count": settings.warmup_requests})
        await run_workload(
            warmup,
            run_id=run_id + "-warmup",
            model=spec.served_model_name,
            transport=live.http,
            images=live.images,
        )


def _logged_gib(pattern: str, log: str) -> float | None:
    found = re.search(pattern, log)
    return float(found[1]) if found else None


def _gpu_memory_gib(selected: tuple[str, ...] | None) -> float | None:
    """The memory of the one GPU that serves, in GiB; None when it cannot be told."""
    devices = detect_devices()
    try:
        index, _, _ = gpu_spec(tuple((d.name, d.total_bytes // 2**20) for d in devices), selected)
    except Unavailable:
        return None
    return devices[index].total_bytes / 2**30


async def _measure(
    project: ResolvedProject,
    *,
    engine: Engine,
    runtime: Runtime,
    settings: Settings,
    run_dirs: Sequence[Path],
    retain_responses: bool,
    ready_timeout_s: float,
    subject: Subject,
    slo: Slo,
    run_projects: Sequence[ResolvedProject],
    environment: RunEnvironment,
    steady_state: bool,
) -> list[Measurement]:
    first_id = run_dirs[0].name
    facts = WorkloadFacts.of(project)
    spec = _spec_for(project, engine, settings, first_id, None)
    runs: list[_Run] = []

    async with _serving(
        engine, runtime, spec, ready_timeout_s=ready_timeout_s, record_dirs=run_dirs,
        phase="measurement", images=ImageSource(project.images),
    ) as live:  # fmt: skip
        prompt_token_counts = await _count_prompt_tokens(
            live.client,
            engine,
            live.server.endpoint_url,
            spec.served_model_name,
            project.prompts,
            live.images,
        )
        await _warm_up(project, spec, live, first_id)
        for run_dir, run_project in zip(run_dirs, run_projects, strict=True):
            transport = CapturingTransport(live.http)
            peaks = Peaks()
            url = live.server.endpoint_url
            workload = run_project.settings.workload
            lead = steady_lead(workload) if steady_state else 0
            reached = asyncio.Event()
            if not lead:
                reached.set()  # no wave: the window opens now, before the first request
            window = None
            if lead:
                window = asyncio.create_task(
                    _window(engine, live.client, url, peaks, reached, live.server.log_text)
                )
            else:
                before, _, log_start = await _window(
                    engine, live.client, url, peaks, reached, live.server.log_text
                )
                window_ns = time.monotonic_ns()  # the baseline scrape is not part of the window
            poller = asyncio.create_task(_poll_signals(engine, live.client, url, peaks))
            tail: asyncio.Task[str | None] | None = None
            last_ns = 0

            def last_dispatch() -> None:
                nonlocal tail, last_ns
                if not lead:
                    return
                if peaks.generation == 0:  # _window resets the peaks after taking window_ns
                    say(WAVE_OUTLASTED)
                    return
                last_ns = time.monotonic_ns()
                poller.cancel()
                tail = asyncio.create_task(_scrape(engine, live.client, url))

            count = workload.request_count
            extra = f" after a start-up wave of {lead}" if lead else ""
            say(f"measuring: {count} requests, {workload.arrival.kind}{extra}")
            try:
                offered = await run_workload(
                    workload, run_id=run_dir.name, model=spec.served_model_name,
                    transport=transport, images=live.images,
                    on_done=quarters("measuring", count), lead=lead, on_lead_done=reached.set,
                    on_last_dispatch=last_dispatch,
                )  # fmt: skip
                if window:
                    before, window_ns, log_start = await window
            except BaseException:
                if tail:
                    tail.cancel()
                raise
            finally:
                poller.cancel()
                if window:
                    window.cancel()
            records = offered[lead:]
            end_ns = time.monotonic_ns()
            final = await _scrape(engine, live.client, url)
            after, after_ns = final, end_ns
            fell_back = False
            if tail is not None:
                if (closed := await tail) is not None:
                    after, after_ns = closed, last_ns
                else:
                    fell_back = True
            span = measurement_window(
                closed_ns=last_ns if tail is not None else None,
                window_ns=window_ns,
                first_dispatch_ns=min(r.dispatch_ns for r in records) if lead else window_ns,
                end_ns=end_ns,
            )
            await transport.decode_all()
            runs.append(
                _Run(
                    run_dir=run_dir,
                    project=run_project,
                    transport=transport,
                    records=records,
                    lead_records=offered[:lead],
                    window=span,
                    window_ns=window_ns,
                    after_ns=after_ns,
                    before=before,
                    after=after,
                    final=final,
                    fell_back=fell_back,
                    outlasted=bool(lead) and tail is None,
                    peaks=peaks,
                    server_log=live.server.log_text(),
                    log_start=log_start,
                    steady_state=steady_state,
                    image=live.server.identity().get("image"),
                )
            )

    return [
        _finish(
            run,
            facts,
            engine,
            settings,
            prompt_token_counts,
            retain_responses,
            subject,
            slo,
            runtime.gpus,
            environment,
        )  # fmt: skip
        for run in runs
    ]


def _finish(
    run: _Run,
    facts: WorkloadFacts,
    engine: Engine,
    settings: Settings,
    prompt_token_counts: Mapping[str, int],
    retain_responses: bool,
    subject: Subject,
    slo: Slo,
    gpus: tuple[str, ...] | None,
    environment: RunEnvironment,
) -> Measurement:
    """Write one run's requests, metrics and report, and summarize it."""
    run_dir, records, transport, project = run.run_dir, run.records, run.transport, run.project
    run_id = run_dir.name
    workload = project.settings.workload
    prompt_ids = _prompt_ids(project, records)
    # The context window the server runs with: the engine derives it from the checkpoint unless
    # the setup names one.
    context_len = settings.max_context_len or facts.context_limit
    rows = [
        {
            **asdict(record),
            "prompt_id": prompt_ids[record.request_id],
            **_failure_reason(transport.responses.get(record.request_id), record.outcome),
        }
        for record in records
    ]
    write_jsonl(run_dir / "requests.jsonl", rows)
    if retain_responses:
        write_jsonl(
            run_dir / "responses.jsonl",
            [
                {
                    "request_id": record.request_id,
                    "prompt_id": prompt_ids[record.request_id],
                    "text": captured.text,
                    "tool_calls": [asdict(call) for call in captured.tool_calls],
                    "finish_reason": captured.finish_reason,
                    "truncated": captured.finish_reason == "length",
                }
                for record in records
                for captured in [transport.responses.get(record.request_id, CapturedResponse())]
            ],
        )
    for name, text in (("metrics_before.prom", run.before), ("metrics_after.prom", run.after)):
        if text is not None:
            (run_dir / name).write_text(text, encoding="utf-8")

    counters = _counters(engine, run.before, run.after)
    usage_tokens = [
        captured.prompt_tokens
        for captured in transport.responses.values()
        if captured.prompt_tokens is not None
    ]
    entries = {entry.id: entry for entry in project.prompts}
    measurement = summarize(
        records,
        window=run.window,
        slo=slo,
        request_prompt_tokens={
            record.request_id: captured.prompt_tokens
            if (captured := transport.responses.get(record.request_id)) and captured.prompt_tokens
            else prompt_token_counts.get(prompt_ids[record.request_id])
            for record in records
        },
        request_images={
            record.request_id: len(entries[prompt_ids[record.request_id]].image_urls)
            for record in records
        },
        peaks=run.peaks,
        preemptions=counters.preemptions,
        prefix_cache_hits=counters.prefix_cache_hits,
        prefix_cache_queries=counters.prefix_cache_queries,
        prompt_tokens=[*prompt_token_counts.values(), *usage_tokens],
        max_context_len=context_len,
        too_long=tuple(
            entry.id
            for entry in project.prompts
            if entry.id in prompt_token_counts
            and prompt_token_counts[entry.id] + (entry.max_tokens or workload.output_tokens)
            > context_len
        ),
        quant_kernels=engine.parse_quant_kernels(run.server_log),
        tool_calls=_tool_call_stats(project, records, transport.responses),
    )
    capacity = engine.parse_signals(run.final) if run.final is not None else EngineSignals()
    wave_included = (
        workload.arrival.kind == "closed_loop" and run.steady_state and not run.lead_records
    )
    run_checks = [
        *checks(
            engine,
            run.final,
            facts,
            settings,
            measurement.peak_running,
            run.server_log,
            run.log_start,
        ),
        *window_checks(
            measurement,
            counters.generation_tokens,
            run.fell_back,
            run.outlasted,
            sum(r.dispatch_ns >= run.window.start_ns for r in records),
            wave_included or run.outlasted,
        ),
    ]
    measurement = replace(
        measurement,
        image=run.image,
        served_as_float16=cast_to_float16(engine, run.server_log),
        checks=tuple(run_checks),
        engine_output_throughput=engine_output_rate(
            measurement, counters.generation_tokens, run.fell_back
        ),
        startup_wave=group_stats(run.lead_records, {}) if run.lead_records else None,
        startup_wave_included=wave_included,
        queue_share=_queue_share(counters),
        spec_acceptance_length=_acceptance_length(counters),
        spec_coverage=_coverage(counters),
        kv_capacity_tokens=capacity.kv_capacity_tokens,
        kv_max_concurrency=capacity.kv_max_concurrency,
        kv_blocks=capacity.kv_blocks,
        hybrid_cache=capacity.hybrid_cache,
        kv_available_gib=_logged_gib(engine.kv_memory_log, run.server_log),
        gpu_memory_gib=_logged_gib(engine.gpu_memory_log, run.server_log) or _gpu_memory_gib(gpus),
        kv_capacity_estimate_tokens=project_fit(project, engine, settings).capacity_tokens,
        ceilings=compute_ceilings(
            anatomy=project.anatomy,
            variant=project.artifact.metadata.variants[0],
            kv_cache_dtype=settings.kv_cache_dtype,
            selected=gpus,
            measured=Measured(
                seconds=(run.after_ns - run.window_ns) / 1e9,
                requests=sum(
                    window_share(record, run.window)
                    for record in records
                    if record.outcome == "success"
                ),
                prompt_tokens=counters.prompt_tokens,
                prefill_computed_tokens=_computed_prompt_tokens(counters),
                generation_tokens=counters.generation_tokens,
                avg_running_batch=_window_batch(counters) or measurement.mean_running,
            ),
        ),
    )
    write_json(run_dir / "metrics.json", asdict(measurement))
    (run_dir / "report.md").write_text(
        render_markdown(
            run_id,
            measurement,
            rows,
            facts,
            settings,
            subject,
            run.image,
            slo,
            engine.defaults,
            engine.added_flags,
            with_logged_version(engine, environment, run.server_log).ran,
        ),
        "utf-8",
    )
    return measurement


def _tool_call_stats(
    project: ResolvedProject,
    records: Sequence[RequestRecord],
    responses: Mapping[str, CapturedResponse],
) -> ToolCallStats | None:
    """Tool-call quality over the successful requests that offered tools (prompts cycle)."""
    pairs = []
    for record in records:
        entry = project.prompts[record.prompt_index]
        if record.outcome == "success" and entry.tools:
            pairs.append((entry.chat, responses[record.request_id].tool_calls))
    return tool_call_stats(pairs)


def _failure_reason(captured: CapturedResponse | None, outcome: str) -> dict[str, Any]:
    if outcome == "success" or captured is None or captured.http_status is None:
        return {}
    return {"http_status": captured.http_status, "error": captured.error}


def _counters(engine: Engine, before: str | None, after: str | None) -> EngineSignals:
    """The increase of the engine's cumulative counters over the run; None where unknown."""
    if before is None or after is None:
        return EngineSignals()
    start, end = engine.parse_signals(before), engine.parse_signals(after)

    def increase(first: float | None, last: float | None) -> float | None:
        return None if first is None or last is None or last < first else last - first

    return EngineSignals(
        preemptions=increase(start.preemptions, end.preemptions),
        prefix_cache_hits=increase(start.prefix_cache_hits, end.prefix_cache_hits),
        prefix_cache_queries=increase(start.prefix_cache_queries, end.prefix_cache_queries),
        prompt_tokens=increase(start.prompt_tokens, end.prompt_tokens),
        prompt_tokens_computed=increase(start.prompt_tokens_computed, end.prompt_tokens_computed),
        generation_tokens=increase(start.generation_tokens, end.generation_tokens),
        iterations=increase(start.iterations, end.iterations),
        queue_seconds=increase(start.queue_seconds, end.queue_seconds),
        prefill_seconds=increase(start.prefill_seconds, end.prefill_seconds),
        spec_drafts=increase(start.spec_drafts, end.spec_drafts),
        spec_accepted_tokens=increase(start.spec_accepted_tokens, end.spec_accepted_tokens),
    )


def _window_batch(counters: EngineSignals) -> float | None:
    """Sequences per engine step over the window: the average running batch, counted by the
    engine rather than sampled. Accepted draft tokens are extra tokens of the same sequence."""
    if not counters.generation_tokens or not counters.iterations:
        return None
    return (counters.generation_tokens - (counters.spec_accepted_tokens or 0)) / counters.iterations


def _computed_prompt_tokens(counters: EngineSignals) -> float | None:
    """The prompt tokens the GPU ran through prefill. vLLM counts them by source
    (``local_compute``). Without that family, ``prompt_tokens`` less ``prefix_cache_hits`` is
    used; the hits counter also counts lookups for the warm-up request, so it can overshoot."""
    if counters.prompt_tokens_computed is not None:
        return counters.prompt_tokens_computed
    if counters.prompt_tokens is None:
        return None
    return max(counters.prompt_tokens - (counters.prefix_cache_hits or 0.0), 0.0)


def _queue_share(counters: EngineSignals) -> float | None:
    """The share of the requests' time to first token (as the engine timed it) spent queued."""
    queued, prefill = counters.queue_seconds, counters.prefill_seconds
    if queued is None or prefill is None or queued + prefill <= 0:
        return None
    return queued / (queued + prefill)


def _coverage(counters: EngineSignals) -> float | None:
    """The share of generated tokens that came from accepted drafts."""
    if not counters.spec_drafts or counters.spec_accepted_tokens is None:
        return None
    if not counters.generation_tokens:
        return None
    return counters.spec_accepted_tokens / counters.generation_tokens


def _acceptance_length(counters: EngineSignals) -> float | None:
    """Tokens each speculative draft yielded: 1 plus the accepted draft tokens per draft."""
    if not counters.spec_drafts or counters.spec_accepted_tokens is None:
        return None
    return 1 + counters.spec_accepted_tokens / counters.spec_drafts


def _prompt_ids(project: ResolvedProject, records: Sequence[RequestRecord]) -> dict[str, str]:
    """Map each request to the registered prompt it offered."""
    return {record.request_id: project.prompts[record.prompt_index].id for record in records}


# --------------------------------------------------------------------------------------
# Trace (Level 1)
# --------------------------------------------------------------------------------------


def _add_trace(
    project: ResolvedProject,
    engine: Engine,
    runtime: Runtime,
    settings: Settings,
    run_dir: Path,
    measurement: Measurement,
    ready_timeout_s: float,
    counters: bool,
) -> Measurement:
    """Profile a short slice on its own launch; a failed trace never fails the analysis."""
    trace_dir = run_dir / "trace"
    trace_dir.mkdir(mode=0o700)
    say("trace: profiling a short slice on a separate launch (diagnostic, not a speed claim)")
    extender = load_extender()
    sections: list[str] = []
    trace = None
    try:
        asyncio.run(_traced_slice(project, engine, runtime, settings, trace_dir, ready_timeout_s))
        files = engine.collect_trace(trace_dir)
        trace = read_trace(files, engine.trace_step_scope)
        summary = summarize_trace(
            trace,
            len(files),
            expected_quant_kernels=[
                QUANT_KERNEL_SYMBOLS[kernel.name]
                for kernel in measurement.quant_kernels
                if kernel.name in QUANT_KERNEL_SYMBOLS
            ],
        )
        if extender:
            extension = extender.analyse(trace, replace(measurement, trace=summary), settings)
            summary = replace(summary, analysis=extension.result)
            sections = extension.sections
    except TraceUnavailable as error:
        summary = TraceSummary("unavailable", str(error))
    except (AnalyseFailure, httpx.HTTPError, TimeoutError, OSError, ValueError) as error:
        summary = TraceSummary("unavailable", f"{type(error).__name__}: {error}")
    measurement = replace(measurement, trace=summary)
    write_json(run_dir / "metrics.json", asdict(measurement))
    lines = render_trace(summary, analysis_available=extender is not None) + sections
    with (run_dir / "report.md").open("a", encoding="utf-8") as report:
        report.write("\n".join(lines) + "\n\n")
    if counters:
        measurement = _add_counters(
            project, engine, runtime, settings, run_dir, measurement, trace, ready_timeout_s
        )
    return measurement


async def _traced_slice(
    project: ResolvedProject,
    engine: Engine,
    runtime: Runtime,
    settings: Settings,
    trace_dir: Path,
    ready_timeout_s: float,
) -> None:
    """Launch with the profiler on, run TRACE_REQUESTS requests at the workload's concurrency
    under a hard time limit, and stop the profiler so the trace files are written."""
    run_id = trace_dir.parent.name
    spec = _spec_for(project, engine, settings, run_id + "-trace", trace_dir)
    async with _serving(
        engine, runtime, spec, ready_timeout_s=ready_timeout_s, record_dirs=[trace_dir],
        phase="trace", images=ImageSource(project.images),
    ) as live:  # fmt: skip
        await _warm_up(project, spec, live, run_id)
        url = live.server.endpoint_url
        await engine.start_trace(live.client, url)
        try:
            await _timed_requests(project, spec, live, run_id, TRACE_REQUESTS, TRACE_TIME_LIMIT_S)
        finally:
            await engine.stop_trace(live.client, url)


async def _timed_requests(
    project: ResolvedProject, spec: ServeSpec, live: _Live, run_id: str, count: int, limit_s: float
) -> None:
    """``count`` requests at the workload's concurrency under a hard time limit."""
    sliced = project.settings.workload.model_copy(update={"request_count": count})
    say(f"profiling {count} requests")
    await asyncio.wait_for(
        run_workload(
            sliced,
            run_id=run_id + "-profiled",
            model=spec.served_model_name,
            transport=live.http,
            images=live.images,
        ),
        limit_s,
    )


# --------------------------------------------------------------------------------------
# Kernel counters (Level 2)
# --------------------------------------------------------------------------------------


def _add_counters(
    project: ResolvedProject,
    engine: Engine,
    runtime: Runtime,
    settings: Settings,
    run_dir: Path,
    measurement: Measurement,
    trace: Trace | None,
    ready_timeout_s: float,
) -> Measurement:
    """Profile the trace's top kernels with ncu; failing to never fails the analysis."""
    counters_dir = run_dir / "counters"
    counters_dir.mkdir(mode=0o700)
    say("counters: profiling the top kernels with Nsight Compute, one launch per counter group")
    try:
        summary = _measure_counters(
            project, engine, runtime, settings, counters_dir, measurement, trace, ready_timeout_s
        )
    except (
        AnalyseFailure,
        RuntimeFailure,
        httpx.HTTPError,
        TimeoutError,
        OSError,
        ValueError,
    ) as error:
        summary = CountersSummary("unavailable", f"{type(error).__name__}: {error}")
    extender = load_extender()
    sections: list[str] = []
    if extender and extender.counters and summary.status == "ok":
        extension = extender.counters(summary, measurement)
        summary = replace(summary, analysis=extension.result)
        sections = extension.sections
    measurement = replace(measurement, counters=summary)
    write_json(run_dir / "metrics.json", asdict(measurement))
    with (run_dir / "report.md").open("a", encoding="utf-8") as report:
        report.write(
            "\n".join(
                [*render_counters(summary, analysis_available=extender is not None), *sections]
            )
            + "\n\n"
        )
    return measurement


def _measure_counters(
    project: ResolvedProject,
    engine: Engine,
    runtime: Runtime,
    settings: Settings,
    counters_dir: Path,
    measurement: Measurement,
    trace: Trace | None,
    ready_timeout_s: float,
) -> CountersSummary:
    if trace is None or measurement.trace is None or measurement.trace.status != "ok":
        return CountersSummary("unavailable", "there is no trusted trace to pick kernels from")
    if not runtime.wraps_launch:
        return CountersSummary(
            "unavailable", "counters need ncu inside the image; unavailable in the docker runtime"
        )
    ncu = shutil.which("ncu")
    if ncu is None:
        return CountersSummary("unavailable", "ncu not found")
    qualification = qualify(ncu, counters_dir)
    write_json(counters_dir / "qualification.json", asdict(qualification))
    if not qualification.passed:
        return CountersSummary(
            "unavailable", f"profiler qualification failed: {qualification.reason}", qualification
        )
    selected = select_kernels(trace)

    def run(name: str, prefix: list[str], log_file: Path) -> tuple[list[Launch], str | None]:
        """One engine launch under ncu; whatever ncu wrote before a failure still counts."""
        spec = replace(
            _spec_for(project, engine, settings, f"{counters_dir.parent.name}-{name}", None),
            launch_prefix=tuple(prefix),
        )
        note = None
        try:
            asyncio.run(_counted_slice(project, engine, runtime, spec, log_file, ready_timeout_s))
        except (AnalyseFailure, httpx.HTTPError, TimeoutError, OSError, ValueError) as error:
            note = f"{type(error).__name__}: {error}"
        text = log_file.read_text("utf-8", errors="replace") if log_file.exists() else ""
        return parse_raw_csv(text), note

    # One gated launch per qualified group (each fits one pass), filtered by the kernels' bare
    # function names; the trace and ncu spell full names differently, so every run's rows are
    # matched to the trace's kernels by name afterwards. The occupancy group runs first.
    groups = sorted(qualification.available, key=lambda group: group != REQUIRED_GROUP)
    runs: dict[str, tuple[list[Launch], str | None]] = {}
    returned, used = True, qualification.memory_cold_mib
    for group in groups:
        if not returned:
            break  # the profiler left memory behind: stop before it disturbs anything else
        log_file = counters_dir / f"{group}.csv"
        runs[group] = run(group, group_command(ncu, selected, log_file, group), log_file)
        returned, used = memory_returned(qualification.memory_cold_mib)
    kernels: list[KernelCounters] = []
    for kernel in selected:
        seen, _ = runs.get(REQUIRED_GROUP, ([], None))
        ncu_name, why = match_ncu_name(kernel.name, [launch.name for launch in seen])
        parts = {}
        for group, (launches, note) in runs.items():
            name, missing = match_ncu_name(kernel.name, [launch.name for launch in launches])
            rows = [launch for launch in launches if launch.name == name]
            parts[group] = kernel_counters(kernel, name or ncu_name, rows, missing or note)
        kernels.append(merge_groups(kernel, ncu_name, parts, qualification.unavailable))
    note = None if returned else "GPU memory did not return after profiling; later kernels skipped"
    return CountersSummary("ok", note, qualification, used, tuple(kernels))


async def _counted_slice(
    project: ResolvedProject,
    engine: Engine,
    runtime: Runtime,
    spec: ServeSpec,
    log_file: Path,
    ready_timeout_s: float,
) -> None:
    """Run the profiled slice between the engine's start and stop profile requests, so ncu
    (``--profile-from-start off``) counts only workload kernels. ncu ends the server itself once
    its launch count is reached, possibly mid-slice: that is expected, not a failure. Stopping
    the server afterwards lets ncu write its report even when the count was not reached."""
    async with _serving(
        engine, runtime, spec, ready_timeout_s=ready_timeout_s, record_dirs=[log_file.parent],
        phase=f"counters ({log_file.stem})", images=ImageSource(project.images), may_exit=True,
    ) as live:  # fmt: skip
        run_id = log_file.parent.parent.name
        await _warm_up(project, spec, live, run_id)
        url = live.server.endpoint_url
        await engine.start_trace(live.client, url)
        try:
            await _timed_requests(
                project, spec, live, run_id, COUNTER_REQUESTS, COUNTER_TIME_LIMIT_S
            )
            await engine.stop_trace(live.client, url)
        except (httpx.HTTPError, TimeoutError):
            if live.server.is_running():
                raise
