"""Measuring one served setup: start the server, offer the workload, scrape the engine and
write the run directory."""

from __future__ import annotations

import asyncio
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Mapping, Sequence

import httpx

from . import __version__
from .capabilities import Resolved
from .capture import CapturedResponse, CapturingTransport
from .ceilings import Measured, Unavailable, compute_ceilings, gpu_spec
from .checks import cast_to_float16, checks, engine_output_rate, window_checks
from .classify import diagnose_run
from .client import (
    UNLIMITED_CONNECTIONS,
    HttpxTransport,
    RequestRecord,
    chat_body,
    run_workload,
    steady_lead,
)
from .engines import Engine
from .environment import (
    RunEnvironment,
    with_logged_version,
)
from .errors import AnalyseFailure
from .files import write_json
from .images import ImageSource
from .inputs import PromptEntry
from .jsonanswers import StructuredAnswerStats, structured_answer_stats
from .measurement import (
    ENGINE_UNMEASURED,
    Measurement,
    Peaks,
    avg_context_tokens,
    client_batch,
    group_stats,
    measurement_window,
    summarize,
)
from .platforms import DeviceIdentity, detect_devices
from .playbook import WorkloadFacts
from .progress import Note, Phase, PhaseStarted, RequestsDone, RunWritten, WindowOpened, emit
from .project import ResolvedProject, json_answer_ids, project_fit, workload_facts
from .report.build import RunOutputs, Subject
from .rundir import (
    RunCapture,
    Scrape,
    prompt_ids,
    request_prompt_tokens,
    request_rows,
    write_derived,
    write_evidence,
)
from .runs import RunFile
from .runtime import (
    RunningServer,
    Runtime,
    RuntimeFailure,
    ServeSpec,
    generate_api_key,
    log_tail,
    select_loopback_port,
    wait_until_ready,
)
from .settings import EngineSignals, Settings
from .signal_source import RunInputs
from .slo import DEFAULT_SLO, Slo
from .toolcalls import ToolCallStats, tool_call_stats

SIGNAL_POLL_S = 1.0
WAVE_OUTLASTED = (
    "the start-up wave had not finished when the last request was sent; measuring the whole run"
)
SAVED_LOG_CHARS = 10_000_000  # the run directory keeps at most this much of the server log


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
    for entry in prompts:
        try:
            body = (
                {"prompt": entry.prompt}
                if entry.messages is None
                else await chat_body(entry.chat, images)
            )
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return {}
        count = await engine.count_prompt_tokens(client, server_url, model, body)
        if count is None:
            return {}
        counts[entry.id] = count
    return counts


async def _scrape(engine: Engine, client: httpx.AsyncClient, server_url: str) -> Scrape | None:
    """The engine's metrics, on the run's own client: a client made per scrape loads the CA
    bundle each time, which blocks the event loop for milliseconds and skews the timings."""
    if not engine.capabilities(None).polls_metrics:
        return None
    return await engine.signal_source.read(client, server_url)


async def _poll_signals(
    engine: Engine, client: httpx.AsyncClient, server_url: str, peaks: Peaks
) -> None:
    if not engine.capabilities(None).polls_metrics:
        return
    while True:
        generation = peaks.generation
        scrape = await _scrape(engine, client, server_url)
        if scrape is not None and peaks.generation == generation:
            peaks.observe(scrape.signals)
        await asyncio.sleep(SIGNAL_POLL_S)


async def _engine_now(
    engine: Engine,
    client: httpx.AsyncClient,
    url: str,
    log: Callable[[], str],
    mark: Callable[[], None] = lambda: None,
) -> tuple[Scrape | None, int, str]:
    """The server log, the moment (after ``mark`` runs) and a scrape of the engine: where a
    measurement of the engine starts."""
    log_text = await asyncio.to_thread(log)
    start_ns = time.monotonic_ns()
    mark()
    return await _scrape(engine, client, url), start_ns, log_text


async def _window(
    engine: Engine,
    client: httpx.AsyncClient,
    url: str,
    peaks: Peaks,
    reached: asyncio.Event,
    log: Callable[[], str],
    lead: int,
) -> tuple[Scrape | None, int, str]:
    """Wait for the start-up wave to finish; then restart the peaks and scrape the engine, so
    the engine's numbers cover the steady-state window. Returns the scrape, the window's start
    and the server log as it was at that moment."""
    await reached.wait()
    emit(WindowOpened(lead=lead))
    return await _engine_now(engine, client, url, log, peaks.reset)


def _edges_when_unpolled(peaks: Peaks, edges: Sequence[Scrape | None]) -> None:
    """A window shorter than one poll interval gets no polled sample: the scrapes that opened
    and closed it stand in, so its peaks are measured."""
    if (peaks.kv_usage, peaks.running, peaks.waiting) != (None, None, None):
        return
    for edge in edges:
        if edge is not None:
            peaks.observe(edge.signals)


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
    engine_args: Sequence[str] = (),
) -> list[Measurement]:
    """Measure ``settings`` on one launch: start the server, warm it up, run the workload into
    each of ``run_dirs`` in turn and stop it. Each run directory is complete on return: its
    evidence, ``metrics.json`` with the diagnosis, ``report.json``, ``report.md`` and ``run.json``.

    ``run_projects`` serves extensions (plain ``analyse`` has one run): it gives each run its
    own variant of ``project`` (a smaller or busier workload). Every variant must have the same
    artifact and current setup as ``project``, else ValueError. A closed-loop run offers a
    start-up wave first and measures after it; with ``steady_state`` False it measures the
    whole run. ``engine_args`` are recorded in each ``run.json``.
    """
    for variant in run_projects or ():
        if (variant.artifact, variant.record.current_setup) != (
            project.artifact,
            project.record.current_setup,
        ):
            raise ValueError("every run project must serve the same model and current setup")
    facts = workload_facts(project)
    measurements = []
    try:
        captures = run_workloads(
            project,
            engine=engine,
            runtime=runtime,
            settings=settings,
            run_dirs=run_dirs,
            run_projects=run_projects or [project] * len(run_dirs),
            retain_responses=retain_responses,
            ready_timeout_s=ready_timeout_s,
            steady_state=steady_state,
        )
        for capture in captures:
            measurement = summarize_run(
                capture, engine=engine, settings=settings, slo=slo, gpus=runtime.gpus, facts=facts
            )
            effective = engine.effective(settings, capture.resolved)
            logged = with_logged_version(engine, environment, capture.server_log)
            gpu = serving_gpu(logged.identities, runtime.gpus)
            diagnosis = diagnose_run(
                measurement, effective, None, engine=engine, platform=logged.platform, gpu=gpu,
                version=logged.engine_version, facts=facts, resolved=capture.resolved,
            )  # fmt: skip
            run_file = run_file_for(
                project,
                engine=engine,
                runtime=runtime,
                settings=settings,
                engine_args=engine_args,
                retain_responses=retain_responses,
                image=capture.image,
                environment=logged,
            )
            outputs = RunOutputs(
                run_id=capture.run_dir.name,
                ran=logged.ran,
                measurement=measurement,
                diagnosis=diagnosis,
                suggested=None,
                rows=request_rows(capture),
                facts=facts,
                settings=settings,
                subject=subject,
                image=capture.image,
                slo=slo,
                engine=engine,
                gpu=gpu,
                source=subject.setup.label if subject.setup else "",
                engine_args=engine_args,
                retained=retain_responses,
                feedback=False,
                resolved=capture.resolved,
            )
            write_derived(capture.run_dir, outputs, run_file)
            emit(RunWritten(run_dir=capture.run_dir))
            measurements.append(measurement)
    except AnalyseFailure as error:
        if subject.setup is None:
            raise
        raise AnalyseFailure(f"your current setup did not run: {error}") from None
    return measurements


def run_file_for(
    project: ResolvedProject,
    *,
    engine: Engine,
    runtime: Runtime,
    settings: Settings,
    engine_args: Sequence[str],
    retain_responses: bool,
    image: str | None,
    environment: RunEnvironment,
) -> RunFile:
    """The ``run.json`` of a run: what ran, where, and with which settings."""
    inherited = {
        name: value for name, value in os.environ.items() if name not in settings.extra_env
    }
    return RunFile(
        schema_version="1",
        snapshot_id=project.record.snapshot_id,
        engine_args=tuple(engine_args),
        settings=asdict(settings),
        retained_responses=retain_responses,
        temperature=project.config.workload.temperature,
        runtime=runtime.label,
        gpus=runtime.gpus,
        image=image,
        inherited_env=engine.inherited_env(inherited) if runtime.inherits_environment else {},
        gpus_identity=tuple(DeviceIdentity(**gpu) for gpu in environment.identities),
        engine=environment.engine,
        engine_version=environment.engine_version,
        platform=environment.platform,
        format=environment.format,
        tensward_version=__version__,
        host=asdict(environment.host),
    )


def serving_gpu(
    identities: Sequence[Mapping[str, Any]], selected: tuple[str, ...] | None
) -> str | None:
    """The name of the device serving: the first selected one, else device 0."""
    index = int(selected[0]) if selected else 0
    return str(identities[index]["name"]) if index < len(identities) else None


def run_workloads(
    project: ResolvedProject,
    *,
    engine: Engine,
    runtime: Runtime,
    settings: Settings,
    run_dirs: Sequence[Path],
    run_projects: Sequence[ResolvedProject],
    retain_responses: bool,
    ready_timeout_s: float,
    steady_state: bool,
) -> list[RunCapture]:
    """Start the server once, warm it up, offer each of ``run_projects`` into its run directory
    in turn and stop the server; each run's evidence is written as the launch stops."""
    for run_dir in run_dirs:
        run_dir.mkdir(mode=0o700, parents=True)
    return asyncio.run(
        _run_workloads(
            project,
            engine=engine,
            runtime=runtime,
            settings=settings,
            run_dirs=run_dirs,
            run_projects=run_projects,
            retain_responses=retain_responses,
            ready_timeout_s=ready_timeout_s,
            steady_state=steady_state,
        )
    )


@dataclass(frozen=True, slots=True)
class Live:
    """A started, ready server and the clients that talk to it."""

    server: RunningServer
    client: httpx.AsyncClient
    http: HttpxTransport
    images: ImageSource
    resolved: Resolved | None = None


@asynccontextmanager
async def serving(
    engine: Engine, runtime: Runtime, spec: ServeSpec, *, ready_timeout_s: float,
    record_dirs: Sequence[Path], phase: str, images: ImageSource, may_exit: bool = False,
) -> AsyncIterator[Live]:  # fmt: skip
    """Start the server, wait until it is ready, and always stop it; log it into ``record_dirs``.
    A server that exits during the run is a failure unless ``may_exit`` (a profiler stops it).
    ``phase`` names the launch in the progress lines; ``images`` makes the requests' image data."""
    emit(PhaseStarted(phase="launch", detail=f"{phase}: launching {engine.name} ({runtime.label})"))
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
            resolved = None
            if engine.capabilities(None).resolves_at_launch:
                launch_log = await asyncio.to_thread(server.log_text)
                resolved = await engine.resolved(client, server.endpoint_url, launch_log)
            http = HttpxTransport(client=workload_client, base_url=server.endpoint_url)
            yield Live(server, client, http, images, resolved)
        if not may_exit and not server.is_running():
            raise AnalyseFailure(f"the server died during the run\n{log_tail(server.log_text())}")
    except RuntimeFailure as error:
        raise AnalyseFailure(str(error)) from None
    finally:
        emit(PhaseStarted(phase="stop", detail=f"{phase}: stopping {engine.name}"))
        server.stop()
        # The whole log (its end, if huge) keeps the startup lines, such as the kernels chosen.
        text = server.log_text()[-SAVED_LOG_CHARS:].replace(spec.api_key, "***")
        for record_dir in record_dirs:
            (record_dir / "server.log").write_text(text, encoding="utf-8")


def spec_for(
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


async def warm_up(project: ResolvedProject, spec: ServeSpec, live: Live, run_id: str) -> None:
    config = project.config
    if config.warmup_requests > 0:
        emit(PhaseStarted(phase="warmup", detail=f"warmup: {config.warmup_requests} requests"))
        warmup = config.workload.model_copy(update={"request_count": config.warmup_requests})
        await run_workload(
            warmup,
            run_id=run_id + "-warmup",
            model=spec.engine.request_model(spec.served_model_name),
            transport=live.http,
            images=live.images,
            extras=spec.engine.request_extras,
        )


async def timed_requests(
    project: ResolvedProject,
    spec: ServeSpec,
    live: Live,
    run_id: str,
    count: int,
    limit_s: float,
    phase: Phase,
) -> None:
    """``count`` requests at the workload's concurrency under a hard time limit."""
    sliced = project.config.workload.model_copy(update={"request_count": count})
    emit(PhaseStarted(phase=phase, detail=f"profiling {count} requests"))
    await asyncio.wait_for(
        run_workload(
            sliced,
            run_id=run_id + "-profiled",
            model=spec.engine.request_model(spec.served_model_name),
            transport=live.http,
            images=live.images,
            extras=spec.engine.request_extras,
        ),
        limit_s,
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


async def _run_workloads(
    project: ResolvedProject,
    *,
    engine: Engine,
    runtime: Runtime,
    settings: Settings,
    run_dirs: Sequence[Path],
    run_projects: Sequence[ResolvedProject],
    retain_responses: bool,
    ready_timeout_s: float,
    steady_state: bool,
) -> list[RunCapture]:
    first_id = run_dirs[0].name
    spec = spec_for(project, engine, settings, first_id, None)
    captures: list[RunCapture] = []

    async with serving(
        engine, runtime, spec, ready_timeout_s=ready_timeout_s, record_dirs=run_dirs,
        phase="measurement", images=ImageSource(project.images),
    ) as live:  # fmt: skip
        prompt_token_counts = await _count_prompt_tokens(
            live.client,
            engine,
            live.server.endpoint_url,
            engine.request_model(spec.served_model_name),
            project.prompts,
            live.images,
        )
        await warm_up(project, spec, live, first_id)
        for run_dir, run_project in zip(run_dirs, run_projects, strict=True):
            transport = CapturingTransport(live.http)
            peaks = Peaks()
            url = live.server.endpoint_url
            workload = run_project.config.workload
            lead = steady_lead(workload) if steady_state else 0
            reached = asyncio.Event()
            if not lead:
                reached.set()  # no wave: the window opens now, before the first request
            window = None
            if lead:  # where the engine's counters start if the wave outlasts the run
                baseline = await _engine_now(engine, live.client, url, live.server.log_text)
                window = asyncio.create_task(
                    _window(engine, live.client, url, peaks, reached, live.server.log_text, lead)
                )
            else:
                before, _, log_start = await _window(
                    engine, live.client, url, peaks, reached, live.server.log_text, lead
                )
                window_ns = time.monotonic_ns()  # the baseline scrape is not part of the window
            poller = asyncio.create_task(_poll_signals(engine, live.client, url, peaks))
            tail: asyncio.Task[Scrape | None] | None = None
            last_ns = 0
            outlasted = False

            def last_dispatch() -> None:
                nonlocal tail, last_ns, outlasted
                if not lead:
                    return
                if peaks.generation == 0:  # _window resets the peaks after taking window_ns
                    emit(Note(text=WAVE_OUTLASTED))
                    outlasted = True  # the whole run is measured: no window opens, no reset
                    if window:
                        window.cancel()
                    return
                last_ns = time.monotonic_ns()
                poller.cancel()
                tail = asyncio.create_task(_scrape(engine, live.client, url))

            count = workload.request_count
            extra = f" after a start-up wave of {lead}" if lead else ""
            detail = f"measuring: {count} requests, {workload.arrival.kind}{extra}"
            emit(PhaseStarted(phase="measure", detail=detail))

            def requests_done(done: int) -> None:
                emit(RequestsDone(phase="measure", done=done, total=count))

            try:
                offered = await run_workload(
                    workload, run_id=run_dir.name,
                    model=engine.request_model(spec.served_model_name),
                    transport=transport, images=live.images,
                    extras=engine.request_extras,
                    on_done=requests_done, lead=lead, on_lead_done=reached.set,
                    on_last_dispatch=last_dispatch,
                )  # fmt: skip
                if outlasted:
                    before, window_ns, log_start = baseline
                elif window:
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
                    fell_back = engine.capabilities(None).polls_metrics
                _edges_when_unpolled(peaks, (before, closed))
            span = measurement_window(
                closed_ns=last_ns if tail is not None else None,
                window_ns=window_ns,
                first_dispatch_ns=min(r.dispatch_ns for r in records) if lead else window_ns,
                end_ns=end_ns,
            )
            await transport.decode_all()
            captures.append(
                RunCapture(
                    run_dir=run_dir,
                    project=run_project,
                    records=records,
                    lead_records=offered[:lead],
                    responses=transport.responses,
                    prompt_token_counts=prompt_token_counts,
                    window=span,
                    window_ns=window_ns,
                    after_ns=after_ns,
                    before=before,
                    after=after,
                    final=final,
                    fell_back=fell_back,
                    outlasted=outlasted,
                    peaks=peaks,
                    server_log=live.server.log_text(),
                    log_start=log_start,
                    steady_state=steady_state,
                    image=live.server.identity().get("image"),
                    resolved=live.resolved,
                )
            )

    for capture in captures:
        write_evidence(capture.run_dir, capture, retain_responses=retain_responses)
    return captures


def resolved_context(resolved: Resolved | None) -> int | None:
    """The context one request may use, when the engine chose it at launch."""
    value = resolved.facts.get("context_per_request") if resolved is not None else None
    return int(value) if value else None


def with_run_signals(counters: EngineSignals, neutral: EngineSignals) -> EngineSignals:
    """The window's counters, with fields the scrapes left unset filled from the run itself."""
    filled = {
        f.name: getattr(neutral, f.name)
        for f in fields(EngineSignals)
        if getattr(counters, f.name) is None and getattr(neutral, f.name) is not None
    }
    return replace(counters, **filled)


def summarize_run(
    capture: RunCapture,
    *,
    engine: Engine,
    settings: Settings,
    slo: Slo,
    gpus: tuple[str, ...] | None,
    facts: WorkloadFacts,
) -> Measurement:
    """The measurement of one captured run. ``facts`` describe the launch's registered
    workload."""
    records, responses, project = capture.records, capture.responses, capture.project
    workload = project.config.workload
    ids = prompt_ids(capture)
    counts = capture.prompt_token_counts
    # The context window the server runs with: the engine derives it from the checkpoint unless
    # the setup names one or the engine chose one at launch.
    settings = engine.effective(settings, capture.resolved)
    context_len = (
        settings.max_context_len or resolved_context(capture.resolved) or facts.context_limit
    )
    run_signals = engine.signal_source.from_run(
        RunInputs(
            records=[r for r in records if r.dispatch_ns >= capture.window.start_ns],
            responses=responses,
            window_s=capture.window.seconds,
            server_log=capture.server_log[len(capture.log_start) :],
            resolved=capture.resolved,
        )
    )
    untimed = frozenset(
        record.request_id
        for record in records
        if engine.untimed(record, responses.get(record.request_id)) is not None
    )
    counters = with_run_signals(
        signal_increase(
            capture.before.signals if capture.before else None,
            capture.after.signals if capture.after else None,
        ),
        run_signals.neutral,
    )
    supported = engine.capabilities(None, settings).signals
    declared = supported.get("generation_tokens")
    per_step = declared is None or declared.cadence != "request_end"
    answered = any(r.outcome == "success" and r.committed_output_tokens for r in records)
    still = per_step and counters.generation_tokens == 0 and answered
    if still:  # read as measured, these zeros would say no preemption, no CPU, no decode
        counters = with_run_signals(EngineSignals(), run_signals.neutral)
    # A signal the engine declares absent for these settings is not read, even when served.
    unread: dict[str, Any] = {
        f.name: f.default
        for f in fields(EngineSignals)
        if f.name in supported and supported[f.name].quality == "absent"
    }
    counters = replace(counters, **unread)
    replicas = max(
        (scrape.replicas for scrape in (capture.before, capture.after, capture.final) if scrape),
        default=1,
    )
    pooled = replicas > 1  # summed counters: no per-engine rate, batch or tokens per step
    engine_batch = (
        run_signals.window_batch if run_signals.window_batch is not None else window_batch(counters)
    )
    usage_tokens = [
        captured.prompt_tokens
        for captured in responses.values()
        if captured.prompt_tokens is not None
    ]
    entries = {entry.id: entry for entry in project.prompts}
    per_request = request_prompt_tokens(capture)
    measurement = summarize(
        records,
        window=capture.window,
        slo=slo,
        request_prompt_tokens=per_request,
        request_images={
            record.request_id: len(entries[ids[record.request_id]].image_urls) for record in records
        },
        peaks=capture.peaks,
        preemptions=counters.preemptions,
        prefix_cache_hits=counters.prefix_cache_hits,
        prefix_cache_queries=counters.prefix_cache_queries,
        prompt_tokens=[*counts.values(), *usage_tokens],
        max_context_len=context_len,
        too_long=tuple(
            entry.id
            for entry in project.prompts
            if entry.id in counts
            and counts[entry.id] + (entry.max_tokens or workload.output_tokens) > context_len
        ),
        quant_kernels=engine.parse_quant_kernels(capture.server_log),
        tool_calls=_tool_call_stats(project, records, responses),
        structured_answers=_structured_answer_stats(project, records, responses),
        untimed=untimed,
    )
    batch = client_batch(
        [r for r in records if r.request_id not in untimed],
        capture.window,
        measurement.output_throughput,
    )
    final = capture.final.signals if capture.final else None
    capacity = final or EngineSignals()
    wave_included = (
        workload.arrival.kind == "closed_loop" and capture.steady_state and not capture.lead_records
    )
    # A token counter updated only when a request ends cannot be compared with the client's
    # count over a window (SignalSupport.cadence), so the mismatch check does not see it.
    generation = counters.generation_tokens if per_step else None
    run_checks = [
        *checks(
            engine,
            final,
            facts,
            settings,
            measurement.peak_running,
            capture.server_log,
            capture.log_start,
            capture.project.record.current_setup.source,
            replicas=replicas,
        ),
        *window_checks(
            measurement,
            generation,
            capture.fell_back,
            capture.outlasted,
            sum(r.dispatch_ns >= capture.window.start_ns for r in records),
            wave_included or capture.outlasted,
        ),
        *([ENGINE_UNMEASURED] if still else []),
    ]
    return replace(
        measurement,
        image=capture.image,
        served_as_float16=cast_to_float16(engine, capture.server_log),
        checks=tuple(run_checks),
        engine_output_throughput=engine_output_rate(measurement, generation, capture.fell_back),
        startup_wave=group_stats(capture.lead_records, {}) if capture.lead_records else None,
        startup_wave_included=wave_included,
        client_batch=batch,
        engine_signals=dict(run_signals.engine),
        queue_share=queue_share(counters),
        spec_acceptance_length=acceptance_length(counters),
        spec_coverage=coverage(counters),
        replicas=replicas,
        prompt_tokens_per_step=None if pooled else prompt_tokens_per_step(counters),
        frontend_cpu_cores=None
        if pooled
        else frontend_cpu_cores(
            counters.frontend_cpu_seconds,
            (capture.after_ns - capture.window_ns) / 1e9,
            engine.frontend_processes(settings),
        ),
        kv_capacity_tokens=capacity.kv_capacity_tokens,
        kv_max_concurrency=capacity.kv_max_concurrency,
        kv_blocks=capacity.kv_blocks,
        kv_block_tokens=capacity.kv_block_tokens,
        hybrid_cache=capacity.hybrid_cache,
        kv_available_gib=_logged_gib(engine.kv_memory_log, capture.server_log),
        gpu_memory_gib=_logged_gib(engine.gpu_memory_log, capture.server_log)
        or _gpu_memory_gib(gpus),
        kv_capacity_estimate_tokens=project_fit(project, engine, settings).capacity_tokens,
        ceilings=compute_ceilings(
            anatomy=project.anatomy,
            variant=project.artifact.metadata.variants[0],
            kv_cache_dtype=settings.kv_cache_dtype,
            selected=gpus,
            measured=Measured(
                seconds=(capture.after_ns - capture.window_ns) / 1e9,
                avg_context_tokens=avg_context_tokens(records, capture.window, per_request),
                prefill_computed_tokens=None if pooled else computed_prompt_tokens(counters),
                generation_tokens=None if pooled else counters.generation_tokens,
                avg_running_batch=None
                if pooled
                else (engine_batch or batch or measurement.mean_running),
            ),
        ),
    )


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


def _structured_answer_stats(
    project: ResolvedProject,
    records: Sequence[RequestRecord],
    responses: Mapping[str, CapturedResponse],
) -> StructuredAnswerStats | None:
    """JSON quality over the successful requests whose answers must be JSON (prompts cycle)."""
    judged = json_answer_ids(project)
    triples = []
    for record in records:
        prompt = project.prompts[record.prompt_index].id
        if record.outcome == "success" and prompt in judged:
            response = responses[record.request_id]
            triples.append((prompt, response.text, response.finish_reason))
    return structured_answer_stats(triples)


def signal_increase(before: EngineSignals | None, after: EngineSignals | None) -> EngineSignals:
    """The increase of the engine's cumulative counters over the run; None where unknown."""
    if before is None or after is None:
        return EngineSignals()

    def increase(first: float | None, last: float | None) -> float | None:
        return None if first is None or last is None or last < first else last - first

    return EngineSignals(
        preemptions=increase(before.preemptions, after.preemptions),
        prefix_cache_hits=increase(before.prefix_cache_hits, after.prefix_cache_hits),
        prefix_cache_queries=increase(before.prefix_cache_queries, after.prefix_cache_queries),
        prompt_tokens=increase(before.prompt_tokens, after.prompt_tokens),
        prompt_tokens_computed=increase(
            before.prompt_tokens_computed, after.prompt_tokens_computed
        ),
        generation_tokens=increase(before.generation_tokens, after.generation_tokens),
        iterations=increase(before.iterations, after.iterations),
        queue_seconds=increase(before.queue_seconds, after.queue_seconds),
        prefill_seconds=increase(before.prefill_seconds, after.prefill_seconds),
        spec_drafts=increase(before.spec_drafts, after.spec_drafts),
        spec_accepted_tokens=increase(before.spec_accepted_tokens, after.spec_accepted_tokens),
        frontend_cpu_seconds=increase(before.frontend_cpu_seconds, after.frontend_cpu_seconds),
    )


def window_batch(counters: EngineSignals) -> float | None:
    """Sequences per engine step over the window: the average running batch, counted by the
    engine rather than sampled. Accepted draft tokens are extra tokens of the same sequence."""
    if not counters.generation_tokens or not counters.iterations:
        return None
    return (counters.generation_tokens - (counters.spec_accepted_tokens or 0)) / counters.iterations


def computed_prompt_tokens(counters: EngineSignals) -> float | None:
    """The prompt tokens the GPU ran through prefill. vLLM counts them by source
    (``local_compute``). Without that family, ``prompt_tokens`` less ``prefix_cache_hits`` is
    used; the hits counter also counts lookups for the warm-up request, so it can overshoot."""
    if counters.prompt_tokens_computed is not None:
        return counters.prompt_tokens_computed
    if counters.prompt_tokens is None:
        return None
    return max(counters.prompt_tokens - (counters.prefix_cache_hits or 0.0), 0.0)


def prompt_tokens_per_step(counters: EngineSignals) -> float | None:
    """Prompt tokens the GPU ran through prefill per engine step over the window."""
    computed = computed_prompt_tokens(counters)
    if computed is None or not counters.iterations:
        return None
    return computed / counters.iterations


def frontend_cpu_cores(
    cpu_seconds: float | None, window_s: float, frontend_processes: int | None
) -> float | None:
    """CPU cores the API server used over the window. None when unknown, or with several API-server
    processes, whose counter may cover one process only."""
    if cpu_seconds is None or window_s <= 0 or (frontend_processes or 1) > 1:
        return None
    return cpu_seconds / window_s


def queue_share(counters: EngineSignals) -> float | None:
    """The share of the requests' time to first token (as the engine timed it) spent queued."""
    queued, prefill = counters.queue_seconds, counters.prefill_seconds
    if queued is None or prefill is None or queued + prefill <= 0:
        return None
    return queued / (queued + prefill)


def coverage(counters: EngineSignals) -> float | None:
    """The share of generated tokens that came from accepted drafts."""
    if not counters.spec_drafts or counters.spec_accepted_tokens is None:
        return None
    if not counters.generation_tokens:
        return None
    return counters.spec_accepted_tokens / counters.generation_tokens


def acceptance_length(counters: EngineSignals) -> float | None:
    """Tokens each speculative draft yielded: 1 plus the accepted draft tokens per draft."""
    if not counters.spec_drafts or counters.spec_accepted_tokens is None:
        return None
    return 1 + counters.spec_accepted_tokens / counters.spec_drafts
