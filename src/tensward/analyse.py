"""``tensward analyse``: serve the registered model, offer the registered prompts, diagnose.

The flow is: load the verified project, start a server through a :class:`Runtime`, wait until it
is ready, run the declared warmup, scrape the engine's metrics, run the declared workload
through the streaming client, scrape the metrics again, write the report, and always stop the
server. Everything written lands in ``<project>/runs/<run_id>/``. Nothing here invents a value:
whatever the server or the configuration did not establish is reported as not measured.
"""

from __future__ import annotations

import asyncio
import shlex
import shutil
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, AsyncIterator, Mapping, Sequence

import httpx

from .capture import CapturedResponse, CapturingTransport
from .ceilings import Measured, compute_ceilings
from .client import UNLIMITED_CONNECTIONS, HttpxTransport, RequestRecord, chat_body, run_workload
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
from .extensions import load_extender
from .files import new_run_id, write_json, write_jsonl
from .images import ImageSource
from .inputs import PromptEntry
from .measurement import Measurement, Peaks, summarize
from .progress import quarters, say
from .project import ResolvedProject, load_project, project_fit
from .recommendations import Recipe, WorkloadFacts, suggest
from .report import Subject, checks, overrides_suffix, render_markdown, render_suggestions
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
    suggestions: tuple[tuple[Recipe, str], ...]
    source: str  # where the measured current setup came from
    suggestions_text: str  # the suggestions as rendered into the report


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
    for text in engine_args:
        try:
            settings = engine.with_engine_arg(settings, text)
        except ValueError as error:
            raise AnalyseFailure(str(error)) from None
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
) -> AnalyseResult:
    """Measure the current setup (plus any overrides) once and suggest which experiments to try.

    With ``trace``, a separate short profiled launch follows the clean measurement, never
    mixed into it: the engine's profiler must be enabled at launch and slows the engine.
    ``counters`` (which needs ``trace``) then profiles the trace's top kernels with Nsight
    Compute, one more launch each.
    """
    if counters and not trace:
        raise AnalyseFailure("kernel counters pick their kernels from the trace: add trace")
    project = load_project(project_path, verify_weights=verify_weights)
    run_id = new_run_id()
    runs_dir = Path(project_path).expanduser() / "runs"
    runs_dir.mkdir(mode=0o700, exist_ok=True)
    run_dir = runs_dir / run_id
    settings = settings_for(project, engine, engine_args)
    title = "Your current setup" + overrides_suffix(engine_args)
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
    )
    if trace:
        measurement = _add_trace(
            project, engine, runtime, settings, run_dir, measurement, ready_timeout_s, counters
        )
    suggestions = tuple(
        suggest(measurement, WorkloadFacts.of(project), settings, allow_quality_changes=True)
    )
    facts = WorkloadFacts.of(project)

    def command_for(recipe: Recipe) -> str | None:
        changed = engine.engine_args_between(settings, recipe.apply(measurement, facts, settings))
        words = ["tensward", "analyse", "--project", str(project_path)]
        for text in (*engine_args, *changed):
            words += ["--engine-arg", text]
        return shlex.join(words) if changed else None

    text = render_suggestions(suggestions, command_for)
    with (run_dir / "report.md").open("a", encoding="utf-8") as summary:
        summary.write("\n" + text)
    say(f"done: {run_dir}")
    return AnalyseResult(
        run_id, run_dir, measurement, suggestions, project.record.current_setup.label, text
    )


async def _poll_signals(
    engine: Engine, client: httpx.AsyncClient, server_url: str, peaks: Peaks
) -> None:
    while True:
        text = await _scrape(engine, client, server_url)
        if text is not None:
            peaks.observe(engine.parse_signals(text))
        await asyncio.sleep(SIGNAL_POLL_S)


def measure(
    project: ResolvedProject,
    *,
    engine: Engine,
    runtime: Runtime,
    settings: Settings,
    run_dirs: Sequence[Path],
    retain_responses: bool,
    ready_timeout_s: float,
    subject: Subject = Subject(),
    slo: Slo = DEFAULT_SLO,
    run_projects: Sequence[ResolvedProject] | None = None,
) -> list[Measurement]:
    """Start the server once, warm it up, run the workload into each of ``run_dirs`` in turn,
    stop it, and write every run directory. Repeats share one launch.

    ``run_projects`` serves extensions (plain ``analyse`` has one run): it gives each run its
    own variant of ``project`` (a smaller or busier workload); every variant must keep the
    project's model and settings.
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
    start_ns: int
    end_ns: int
    before: str | None
    after: str | None
    peaks: Peaks
    server_log: str
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
            before = await _scrape(engine, live.client, url)
            poller = asyncio.create_task(_poll_signals(engine, live.client, url, peaks))
            count = run_project.settings.workload.request_count
            say(f"measuring: {count} requests, {run_project.settings.workload.arrival.kind}")
            start_ns = time.monotonic_ns()
            try:
                records = await run_workload(
                    run_project.settings.workload,
                    run_id=run_dir.name,
                    model=spec.served_model_name,
                    transport=transport,
                    images=live.images,
                    on_done=quarters("measuring", count),
                )
            finally:
                poller.cancel()
            end_ns = time.monotonic_ns()
            after = await _scrape(engine, live.client, url)
            await transport.decode_all()
            runs.append(
                _Run(
                    run_dir=run_dir,
                    project=run_project,
                    transport=transport,
                    records=records,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    before=before,
                    after=after,
                    peaks=peaks,
                    server_log=live.server.log_text(),
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
        seconds=(run.end_ns - run.start_ns) / 1e9,
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
        mean_running=run.peaks.mean_running,
    )
    capacity = engine.parse_signals(run.after) if run.after is not None else EngineSignals()
    measurement = replace(
        measurement,
        image=run.image,
        kv_capacity_tokens=capacity.kv_capacity_tokens,
        kv_max_concurrency=capacity.kv_max_concurrency,
        kv_capacity_estimate_tokens=project_fit(project, engine, settings).capacity_tokens,
        ceilings=compute_ceilings(
            anatomy=project.anatomy,
            variant=project.artifact.metadata.variants[0],
            kv_cache_dtype=settings.kv_cache_dtype,
            selected=gpus,
            measured=Measured(
                seconds=measurement.seconds or 0.0,
                requests=measurement.succeeded,
                prompt_tokens=counters.prompt_tokens,
                prefill_computed_tokens=_computed_prompt_tokens(counters),
                generation_tokens=counters.generation_tokens,
                avg_running_batch=measurement.mean_running,
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
            checks(engine, run.after, facts, settings),
            engine.defaults,
            engine.added_flags,
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
    for index, record in enumerate(records):
        entry = project.prompts[index % len(project.prompts)]
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
        generation_tokens=increase(start.generation_tokens, end.generation_tokens),
    )


def _computed_prompt_tokens(counters: EngineSignals) -> float | None:
    """The prompt tokens the GPU ran through prefill. vLLM's ``prompt_tokens`` counter includes
    the tokens served from the prefix cache (v0.30 ``PrefillStats``: prompt = computed + cached),
    and ``prefix_cache_hits`` counts those cached tokens, so the difference is the work done."""
    if counters.prompt_tokens is None:
        return None
    return max(counters.prompt_tokens - (counters.prefix_cache_hits or 0.0), 0.0)


def _prompt_ids(project: ResolvedProject, records: Sequence[RequestRecord]) -> dict[str, str]:
    """Map each request to the registered prompt it offered (prompts cycle by arrival index)."""
    ids = [entry.id for entry in project.prompts]
    return {record.request_id: ids[index % len(ids)] for index, record in enumerate(records)}


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
