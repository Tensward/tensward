"""The run report as data: every wording decision about a run, made once, into sections that
``report.md``, ``report.json`` and the terminal lay out."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..calibration import on_devices
from ..capabilities import Resolved
from ..ceilings import Ceilings
from ..classify import NAMES, NOT_MODELLED, Diagnosis, data_parallel_note
from ..engines import Engine
from ..measurement import WAVE_OUTLASTED_CHECK, GroupStats, ImageSplit, Measurement
from ..playbook import Suggestion, WorkloadFacts, fitting_max_context_len
from ..profiling.counters import CountersResult
from ..profiling.trace import TraceResult
from ..project import CurrentSetup
from ..settings import Settings
from ..slo import Slo
from ..suggest import Suggested
from ..text import figure, percent
from ..thresholds import CALIBRATED_GPUS, MIN_CONFIDENT_REQUESTS
from .model import Block, Figure, Report, Section, SuggestionBlock, Text, item
from .profiling import ceilings_section, counters_section, trace_section

MAX_FAILURE_REASONS = 3
MAX_EXAMPLE_PROMPTS = 5
MAX_STEPS_PER_CLASS = 3
MAX_COULD_HELP = 3
FEEDBACK_LINE = "Questions, or results to share: https://github.com/Tensward/tensward/issues"


@dataclass(frozen=True, slots=True)
class Subject:
    """What a run measured, for the top of its report: a title and, for the customer's own
    setup, the record of where it came from."""

    title: str = "Configuration"
    setup: CurrentSetup | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class RunOutputs:
    """Everything a run report says, decided before any wording."""

    run_id: str
    ran: str
    measurement: Measurement
    diagnosis: Diagnosis
    suggested: Suggested | None  # None for measure()'s runs: no next steps
    rows: Sequence[Mapping[str, Any]]  # requests.jsonl rows, for the failure lines
    facts: WorkloadFacts
    settings: Settings
    subject: Subject
    image: str | None
    slo: Slo
    engine: Engine
    gpu: str | None
    source: str  # where the current setup came from, for the terminal header
    engine_args: Sequence[str]
    retained: bool
    changes: Sequence[str] = ()  # "What changed" lines, as text blocks
    compared: Sequence[str] = ()  # the comparison section's lines, as text blocks
    answers_file: Path | None = None
    trace: TraceResult | None = None
    counters: CountersResult | None = None
    feedback: bool = True
    resolved: Resolved | None = None


def overrides_suffix(engine_args: Sequence[str]) -> str:
    """ " + overrides (a=1, b=2)" when engine flags changed the setup, else nothing."""
    return f" + overrides ({', '.join(engine_args)})" if engine_args else ""


def ran_line(ran: str, cast: bool) -> str:
    """What ran, with the engine's float16 cast of a bfloat16 checkpoint when it happened."""
    return f"{ran}, served as float16" if cast else ran


def build_report(outputs: RunOutputs) -> Report:
    """The whole report of a run: what ran, what changed against the current setup, the
    diagnosis and next steps, the measurements, the profiled sections, the comparison and the
    feedback line. A section with nothing to say is left out."""
    measurement, facts = outputs.measurement, outputs.facts
    trace, counters = outputs.trace, outputs.counters
    sections = [
        _header(outputs),
        Section(id="what_changed", title=None, blocks=_lines(outputs.changes)),
        diagnosis_section(outputs.diagnosis, outputs.gpu),
        *(
            [next_steps_section(outputs.diagnosis, outputs.suggested, retained=outputs.retained)]
            if outputs.suggested is not None
            else []
        ),
        _setup(outputs.subject, outputs.settings, measurement, outputs.image),
        _requests(measurement, outputs.rows, facts),
        _performance(measurement, outputs.slo),
        _engine(measurement, outputs.engine, outputs.settings),
        _defaults(outputs.settings, outputs.engine.defaults),
        ceilings_section(
            measurement.ceilings,
            batch_note=_client_batch_note(_unstepped(outputs.engine, outputs.settings)),
        ),
        _quantization(measurement, facts),
        _tool_calls(measurement),
        _structured_answers(measurement),
        _context(measurement, facts),
        Section(
            id="checks",
            title="Checks",
            blocks=tuple(Text(key="check", text=check, item=True) for check in measurement.checks),
            title_audience="markdown",
        ),
        trace_section(trace.summary, analysis_available=trace.extended) if trace else None,
        counters_section(counters.summary, analysis_available=counters.extended)
        if counters
        else None,
        Section(
            id="extension",
            title=None,
            blocks=_lines(
                [*(trace.sections if trace else ()), *(counters.sections if counters else ())]
            ),
        ),
        _answers(outputs.compared, outputs.answers_file),
        Section(
            id="feedback",
            title=None,
            blocks=(Text(key="feedback", text=FEEDBACK_LINE, audience="markdown"),)
            if outputs.feedback
            else (),
        ),
    ]
    return Report(
        run_id=outputs.run_id,
        ran=ran_line(outputs.ran, measurement.served_as_float16),
        sections=tuple(s for s in sections if s is not None and s.blocks),
    )


def _lines(lines: Sequence[str]) -> tuple[Block, ...]:
    return tuple(Text(key="line", text=line) for line in lines)


def _header(outputs: RunOutputs) -> Section:
    measurement = outputs.measurement
    blocks: list[Block] = [
        Text(key="title", text=f"# Tensward analysis {outputs.run_id}", audience="markdown"),
        Text(key="ran", text=f"Ran: {ran_line(outputs.ran, measurement.served_as_float16)}"),
    ]
    if outputs.source:
        suffix = overrides_suffix(outputs.engine_args)
        blocks.append(
            Text(key="source", text=f"current setup{suffix}: {outputs.source}", audience="terminal")
        )
    blocks += [
        Text(key=key, text=f"  {part}", audience="terminal")
        for key, part in _headline_parts(measurement)
    ]
    trace = measurement.trace
    if trace is not None and trace.status != "unavailable" and trace.idle_share is not None:
        blocks.append(
            Text(
                key="gpu_busy",
                text=f"  GPU busy {trace.gpu_busy_share or 0:.1%}, idle {trace.idle_share:.1%} "
                "(profiled)",
                audience="terminal",
            )
        )
    if measurement.replicas > 1:
        blocks.append(
            Text(
                key="replicas",
                text=f"  {data_parallel_note(measurement.replicas)}",
                audience="terminal",
            )
        )
    return Section(id="header", title=None, blocks=tuple(blocks))


def _metric(label: str, value: float | None, unit: str, digits: int = 1) -> str:
    return f"{label} {figure(value, unit, digits)}"


def _headline_parts(measurement: Measurement) -> list[tuple[str, str]]:
    """The headline numbers of a run, each with its key."""
    total = measurement.succeeded + measurement.failed
    engine_rate = measurement.engine_output_throughput
    total_rate = measurement.total_throughput
    if engine_rate and total_rate is not None and measurement.output_throughput is not None:
        total_rate += engine_rate - measurement.output_throughput
    return [
        (
            "output",
            _metric("output", engine_rate or measurement.output_throughput, " tok/s")
            + (" (the engine's count, see Checks)" if engine_rate else ""),
        ),
        ("total", _metric("total", total_rate, " tok/s")),
        ("requests", _metric("requests", measurement.request_throughput, " req/s", 2)),
        ("goodput", _metric("goodput", measurement.goodput, " req/s", 2)),
        ("ttft_p95", _metric("TTFT p95", measurement.ttft_p95_ms, " ms", 0)),
        ("tpot_p95", _metric("TPOT p95", measurement.tpot_p95_ms, " ms")),
        ("failed", f"{measurement.failed} of {total} requests failed"),
        ("ceiling", _ceiling_metric(measurement.ceilings)),
    ]


def _ceiling_metric(ceilings: Ceilings | None) -> str:
    """How close the GPU ran to its limits: the combined figure, else the decode share."""
    if ceilings is not None and ceilings.ceiling_time_pct_of_window is not None:
        return _metric("hardware ceiling reached", ceilings.ceiling_time_pct_of_window, "%", 0)
    if ceilings is not None and ceilings.decode_pct_of_ceiling is not None:
        return f"decode at {ceilings.decode_pct_of_ceiling:.1f}% of its ceiling"
    if ceilings is not None and ceilings.unavailable:
        return f"hardware ceiling: {ceilings.unavailable}"
    if ceilings is not None and ceilings.exceeds_bound:
        shares = ", ".join(ceilings.exceeds_bound)
        return f"hardware ceiling: not available (a share exceeded its bound: {shares})"
    return _metric("hardware ceiling reached", None, "%")


def _setup(
    subject: Subject, settings: Settings, measurement: Measurement, image: str | None
) -> Section:
    """The block that opens the measurements: what was measured and its headline numbers."""
    blocks = []
    if (setup := subject.setup) is not None:
        blocks.append(item("source", f"source: {setup.label}"))
        if setup.image and image and setup.image != image:
            blocks.append(
                item(
                    "version_differs",
                    f"version differs: your setup runs {setup.image}, measured on {image}",
                    "strong",
                )
            )
    blocks.append(item("settings", f"settings: {settings.summary()}"))
    if measurement.served_as_float16:
        blocks.append(item("float16", "served as float16 (the checkpoint is bfloat16)"))
    blocks += [item("note", f"note: {note}") for note in setup.notes] if setup else []
    parts = ", ".join(part for _, part in _headline_parts(measurement))
    blocks.append(item("measured", f"measured: {parts}"))
    if setup is not None and measurement.failed:
        blocks.append(
            item(
                "fails_requests",
                "your current setup fails requests on this workload; fixing that is the first "
                "improvement",
                "strong",
            )
        )
    return Section(id="setup", title=subject.title, blocks=tuple(blocks))


def _group_text(label: str, group: GroupStats | None) -> str:
    if group is None:
        return f"{label}: no requests"
    return (
        f"{label}: {group.requests} requests, "
        f"TTFT p50 {figure(group.ttft_p50_ms, ' ms', 0)}, "
        f"TTFT p95 {figure(group.ttft_p95_ms, ' ms', 0)}, "
        f"TPOT p95 {figure(group.tpot_p95_ms, ' ms', 0)}, "
        f"mean prompt tokens {figure(group.prompt_tokens_mean, '', 0)}"
    )


def _image_split(split: ImageSplit | None) -> list[Block]:
    if split is None:
        return []
    return [
        Text(key="images", text="## Requests with and without images"),
        item("with_images", _group_text("with images", split.with_images)),
        item("without_images", _group_text("without images", split.without_images)),
        item(
            "images_per_request",
            f"images per request that carries images: {split.images_per_request:.1f}",
        ),
        item(
            "image_tokens",
            "the prompt tokens of a request with images include the image tokens the engine "
            "counted",
        ),
    ]


def _reason(row: Mapping[str, Any], *, mask: bool) -> str:
    if row.get("http_status") is None:
        return row.get("error") or f"no HTTP response ({row['outcome']})"
    message = row["error"] or ""
    if mask:  # numbers differ between requests of the same cause (token counts)
        message = re.sub(r"\d+", "N", message)
    return f"HTTP {row['http_status']}: {message}"


def _failures(failed: Sequence[Mapping[str, Any]]) -> list[Block]:
    """The most common reasons requests failed, with counts and example prompt ids."""
    if not failed:
        return []
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in failed:
        groups[_reason(row, mask=True)].append(row)
    blocks: list[Block] = [Text(key="failures", text="### Why requests failed")]
    for rows in sorted(groups.values(), key=len, reverse=True)[:MAX_FAILURE_REASONS]:
        prompts = list(dict.fromkeys(row["prompt_id"] for row in rows))[:MAX_EXAMPLE_PROMPTS]
        blocks.append(
            item(
                "failure",
                f"{len(rows)} x {_reason(rows[0], mask=False)} (prompts: {', '.join(prompts)})",
            )
        )
    return blocks


def _requests(
    measurement: Measurement, rows: Sequence[Mapping[str, Any]], facts: WorkloadFacts
) -> Section:
    """A short human summary of the requests.jsonl rows."""
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["outcome"]] += 1
    blocks: list[Block] = [
        item(
            "requests",
            f"{len(rows)} requests: {measurement.succeeded} succeeded, {measurement.failed} failed",
        )
    ]
    if breakdown := ", ".join(f"{name} {count}" for name, count in sorted(counts.items())):
        blocks.append(item("outcomes", f"outcomes: {breakdown}"))
    distinct = len(set(facts.prompts))
    if len(rows) > distinct:
        blocks.append(
            item(
                "repeated_prompts",
                f"NOTE: {len(rows)} requests from {distinct} distinct prompts: prefix-cache hit "
                "rate and similar signals are inflated by repetition; use more distinct prompts "
                "for realistic numbers",
                "warning",
            )
        )
    blocks += _failures([row for row in rows if row["outcome"] != "success"])
    return Section(id="requests", title="Requests", blocks=tuple(blocks))


def _figure(key: str, label: str, value: float | None, unit: str, digits: int = 1) -> Figure:
    return Figure(key=key, label=label, value=value, unit=unit, digits=digits)


def _performance(measurement: Measurement, slo: Slo) -> Section:
    m = measurement
    blocks: list[Block] = [
        _figure("request_throughput", "request throughput", m.request_throughput, " req/s", 2),
        _figure("output_throughput", "output throughput", m.output_throughput, " tokens/s"),
        _figure(
            "total_throughput",
            "total throughput (prompt + output)",
            m.total_throughput,
            " tokens/s",
        ),
        _figure(
            "goodput",
            f"goodput (TTFT <= {slo.ttft_ms:g} ms and TPOT <= {slo.tpot_ms:g} ms per request)",
            m.goodput,
            " req/s",
            2,
        ),
        _figure("slo_attainment", "requests meeting the SLO", percent(m.slo_attainment), "%", 0),
        _figure("ttft_p50_ms", "TTFT p50", m.ttft_p50_ms, " ms"),
        _figure("ttft_p95_ms", "TTFT p95", m.ttft_p95_ms, " ms"),
        _figure("tpot_p50_ms", "TPOT p50", m.tpot_p50_ms, " ms"),
        _figure("tpot_p95_ms", "TPOT p95", m.tpot_p95_ms, " ms"),
        _figure("e2e_p95_ms", "end-to-end p95", m.e2e_p95_ms, " ms"),
    ]
    if (wave := m.startup_wave) is not None:
        noun = "request" if wave.requests == 1 else "requests"
        where = (
            "outside these figures, inside the engine's"
            if WAVE_OUTLASTED_CHECK in m.checks
            else "outside the measured window"
        )
        blocks.append(
            item(
                "startup_wave",
                f"start-up wave (first {wave.requests} {noun}, all starting together, {where}): "
                f"{_metric('TTFT p50', wave.ttft_p50_ms, ' ms', 0)}, "
                f"{_metric('p95', wave.ttft_p95_ms, ' ms', 0)}",
            )
        )
    elif m.startup_wave_included:
        blocks.append(
            item(
                "startup_wave",
                "start-up wave: included in these numbers (fewer than two waves of requests)",
            )
        )
    blocks += _image_split(m.image_split)
    return Section(id="performance", title="Performance (client side)", blocks=tuple(blocks))


def _capacity(measurement: Measurement) -> Text:
    """The KV capacity the engine reports, beside what Tensward estimated before the run."""
    measured, estimate = measurement.kv_capacity_tokens, measurement.kv_capacity_estimate_tokens
    if measured is None:
        return item("kv_capacity", "KV cache capacity (measured by the engine): not measured")
    line = f"KV cache capacity (measured by the engine): {measured:.0f} tokens"
    if measurement.kv_max_concurrency is not None:
        line += f", {measurement.kv_max_concurrency:.1f} full-length requests at once"
    if estimate is None:
        return item("kv_capacity", line + "; no estimate was made before the run")
    error = estimate / measured - 1
    return item(
        "kv_capacity",
        f"{line}; estimated before the run: {estimate:.0f} tokens (error {error:+.0%})",
    )


def _unstepped(engine: Engine, settings: Settings) -> str:
    """Why the engine's step count is not read with these settings; empty when it is."""
    steps = engine.capabilities(None, settings).signals.get("iterations")
    return steps.absent_reason if steps and steps.quality == "absent" else ""


def _client_batch_note(unstepped: str) -> str:
    return (
        f" (from the client's timings, which read a few percent low: {unstepped})"
        if unstepped
        else ""
    )


def _engine(measurement: Measurement, engine: Engine, settings: Settings) -> Section:
    m = measurement
    reason = _unstepped(engine, settings)
    unstepped = f" ({reason})" if reason else ""
    blocks: tuple[Block, ...] = (
        _figure(
            "peak_kv_usage",
            "highest sampled KV-cache usage (1 s polls)",
            percent(m.peak_kv_usage),
            "%",
        ),
        _figure(
            "peak_running", "highest sampled running requests (1 s polls)", m.peak_running, "", 0
        ),
        _figure(
            "peak_waiting", "highest sampled waiting requests (1 s polls)", m.peak_waiting, "", 0
        ),
        _figure("preemptions", "preemptions", m.preemptions, "", 0),
        _figure("prefix_cache_hits", "prefix-cache hits", m.prefix_cache_hits, " tokens", 0),
        _figure(
            "prefix_cache_hit_rate",
            "prefix-cache hit rate",
            percent(m.prefix_cache_hit_rate),
            "%",
        ),
        Figure(
            key="prompt_tokens_per_step",
            label="prompt tokens per engine step",
            value=m.prompt_tokens_per_step,
            unit="",
            digits=0,
            note=unstepped if m.prompt_tokens_per_step is None else "",
        ),
        _figure("frontend_cpu_cores", "API-server CPU", m.frontend_cpu_cores, " cores", 2),
        _capacity(m),
        Text(key="added_flags", text=f"Tensward adds to every launch: {engine.added_flags}"),
    )
    return Section(id="engine", title="Engine signals (from the engine's metrics)", blocks=blocks)


def _defaults(settings: Settings, defaults: Mapping[str, str]) -> Section:
    """What the engine does for each setting this setup leaves unset, where that is known."""
    unset = {
        n: text for n, text in defaults.items() if getattr(settings, n) == getattr(Settings(), n)
    }
    return Section(
        id="defaults",
        title="Engine defaults in effect (settings this setup leaves unset)",
        blocks=tuple(item(name, f"{name}: {text}") for name, text in unset.items()),
    )


def _quantization(measurement: Measurement, facts: WorkloadFacts) -> Section:
    """The checkpoint's quantization and the kernels the server log says were selected."""
    blocks: tuple[Block, ...] = ()
    if facts.quantization is not None:
        kernels = ", ".join(f"{k.name} ({k.layer})" for k in measurement.quant_kernels)
        slow = " - a slower path" if any(k.slow for k in measurement.quant_kernels) else ""
        blocks = (
            item("checkpoint", f"checkpoint: {facts.quantization}"),
            item(
                "kernel",
                f"kernel: {kernels + slow if kernels else 'not detected in the server log'}",
            ),
        )
    return Section(id="quantization", title="Quantization", blocks=blocks)


def _tool_calls(measurement: Measurement) -> Section:
    """Whether the model called tools correctly. Quality observations, not performance."""
    title = "Tool calling (quality, not performance)"
    if (stats := measurement.tool_calls) is None:
        return Section(id="tool_calls", title=title, blocks=())

    def share(value: float | None) -> str:
        return "no tool call to judge" if value is None else f"{value:.0%}"

    of = f"of the {stats.calling} that did,"
    blocks = (
        item("offered", f"requests that offered tools: {stats.requests}"),
        item("produced_call", f"produced a tool call: {share(stats.produced_call)}"),
        item("valid_json", f"{of} arguments parse as JSON: {share(stats.valid_json)}"),
        item("known_tool", f"{of} tool is one of the offered tools: {share(stats.known_tool)}"),
        item(
            "schema_valid",
            f"{of} arguments satisfy the tool's required keys and property types: "
            f"{share(stats.schema_valid)}",
        ),
        item(
            "caveat",
            "caveat: counts every request that offered tools; prompts that should not call a "
            "tool count as misses",
        ),
    )
    return Section(id="tool_calls", title=title, blocks=blocks)


def _structured_answers(measurement: Measurement) -> Section:
    """Whether the answers that must be JSON are. Quality observations, not performance."""
    title = "Structured output (quality, not performance)"
    if (stats := measurement.structured_answers) is None:
        return Section(id="structured_answers", title=title, blocks=())
    blocks = [
        item("answers", f"answers that must be JSON: {stats.answers}"),
        item("invalid_json", f"invalid JSON: {stats.invalid_json:.0%}"),
        item(
            "runaway_whitespace",
            f"whitespace until the token limit (a grammar loop): {stats.runaway_whitespace:.0%}",
        ),
        item("cut_short", f"stopped at the token limit: {stats.cut_short:.0%}"),
    ]
    if prompts := stats.runaway_prompts:
        shown = ", ".join(prompts[:MAX_EXAMPLE_PROMPTS]) + (
            ", ..." if len(prompts) > MAX_EXAMPLE_PROMPTS else ""
        )
        blocks.append(
            item(
                "runaway_prompts",
                "looping prompts (often a prompt that asks for what the schema cannot hold): "
                + shown,
            )
        )
    return Section(id="structured_answers", title=title, blocks=tuple(blocks))


def _context(measurement: Measurement, facts: WorkloadFacts) -> Section:
    """Name the prompts that cannot fit the context window, and whether raising it can help."""
    title = "Prompts that do not fit the context window"
    if not measurement.too_long or measurement.max_prompt_tokens is None:
        return Section(id="context", title=title, blocks=())
    longest = measurement.max_prompt_tokens + facts.output_tokens
    blocks = [
        item(
            "too_long",
            f"{len(measurement.too_long)} prompts need more than max_context_len "
            f"{measurement.max_context_len} tokens (prompt plus {facts.output_tokens} output "
            "tokens): "
            + ", ".join(measurement.too_long[:MAX_EXAMPLE_PROMPTS])
            + (", ..." if len(measurement.too_long) > MAX_EXAMPLE_PROMPTS else ""),
        ),
        item("longest", f"the longest needs {longest} tokens"),
    ]
    if fitting_max_context_len(measurement, facts) is None:
        blocks.append(
            item(
                "unfixable",
                f"the model supports at most {facts.context_limit} tokens, so raising "
                "max_context_len cannot fix this; shorten those prompts or lower the output "
                "tokens",
            )
        )
    return Section(id="context", title=title, blocks=tuple(blocks))


def _answers(compared: Sequence[str], answers_file: Path | None) -> Section:
    blocks = [*(Text(key="line", text=line, audience="markdown") for line in compared)]
    if answers_file is not None:
        blocks.append(
            Text(
                key="answers_file",
                text=f"answers side by side: {answers_file}",
                audience="terminal",
            )
        )
    return Section(id="answers", title=None, blocks=tuple(blocks))


def _unjudged(diagnosis: Diagnosis) -> bool:
    """Every class but speculation is can't tell: the run did not measure enough to judge."""
    return all(f.state == "cant_tell" for f in diagnosis.findings if f.bottleneck != "speculation")


def diagnosis_headline(diagnosis: Diagnosis) -> str:
    few = diagnosis.requests < MIN_CONFIDENT_REQUESTS
    count = f"only {diagnosis.requests} requests succeeded"
    if diagnosis.requests == 0:
        return "Bottleneck: none named, because no request succeeded"
    if _unjudged(diagnosis):
        return "Bottleneck: not enough was measured to name one"
    if diagnosis.primary is None:
        decided = [f for f in diagnosis.findings if f.threshold]
        notes = ["no measured signal crossed its threshold"]
    else:
        decided = [f for f in diagnosis.findings if f.bottleneck == diagnosis.primary]
        notes = [f"confidence: {diagnosis.confidence}"]
    if diagnosis.primary and not all(f.calibrated for f in decided):
        some = "some " if any(f.calibrated for f in decided) else ""
        notes.append(f"{some}thresholds not yet calibrated on real GPUs")
    notes += [count] if few else []
    name = NAMES[diagnosis.primary] if diagnosis.primary else "none found"
    return f"Bottleneck: {name} ({'; '.join(notes)})"


def _outside_calibration(diagnosis: Diagnosis, gpu: str | None) -> bool:
    """A calibrated class named the bottleneck with "high" confidence on another GPU."""
    primary = next((f for f in diagnosis.findings if f.bottleneck == diagnosis.primary), None)
    if gpu is None or primary is None or not primary.calibrated or diagnosis.confidence != "high":
        return False
    return not on_devices(gpu, CALIBRATED_GPUS)


def diagnosis_section(diagnosis: Diagnosis, gpu: str | None) -> Section:
    """The bottleneck and its evidence, what else crossed, which classes did not cross and what
    could not be told, then the gates and the workload shape. The terminal shows the headline
    only."""
    by_class = {finding.bottleneck: finding for finding in diagnosis.findings}

    def detail(key: str, text: str) -> Text:
        return Text(key=key, text=text, item=True, audience="markdown")

    blocks = [Text(key="headline", text=diagnosis_headline(diagnosis))]
    if diagnosis.primary:
        blocks.append(detail("evidence", f"evidence: {by_class[diagnosis.primary].evidence}"))
        if _outside_calibration(diagnosis, gpu):
            blocks.append(
                detail(
                    "outside_calibration",
                    f"the threshold was calibrated on {' and '.join(CALIBRATED_GPUS)}; this run "
                    f"was on {gpu}",
                )
            )
    elif diagnosis.requests:
        nearest = sorted(
            (f for f in diagnosis.findings if f.state == "clear" and f.threshold),
            key=lambda f: f.margin,
            reverse=True,
        )[:2]
        blocks += [
            detail(f"nearest.{f.bottleneck}", f"nearest: {NAMES[f.bottleneck]}: {f.evidence}")
            for f in nearest
        ]
    for name in diagnosis.secondary:
        finding = by_class[name]
        blocks.append(
            detail(
                f"also_seen.{name}",
                f"also seen: {NAMES[name]} ({finding.confidence}): {finding.evidence}",
            )
        )
    decided = [
        f for f in diagnosis.findings if f.state == "clear" and f.bottleneck != "speculation"
    ]
    speculation = by_class["speculation"]
    if speculation.state == "clear" and speculation.threshold:
        decided.append(speculation)
    if decided and diagnosis.requests:
        for key, label, calibrated in (
            ("not_crossed", "not crossed", True),
            ("not_crossed_uncalibrated", "not crossed (uncalibrated thresholds)", False),
        ):
            names = [NAMES[f.bottleneck] for f in decided if bool(f.calibrated) is calibrated]
            if names:
                blocks.append(detail(key, f"{label}: {', '.join(names)}"))
    if speculation.state == "clear" and speculation.threshold is None:
        blocks.append(detail("speculation", "speculation: off or not reported"))
    if unknown := [f for f in diagnosis.findings if f.state == "cant_tell"]:
        nested = [f"\n  - {NAMES[f.bottleneck]} — {f.evidence}" for f in unknown]
        blocks.append(detail("cant_tell", "can't tell here:" + "".join(nested)))
    blocks.append(
        detail(
            "not_modelled",
            "not modelled: " + "; ".join(f"{NAMES[k]} ({why})" for k, why in NOT_MODELLED.items()),
        )
    )
    blocks += [
        detail(f"gate.{gate.bottleneck}", f"{NAMES[gate.bottleneck]}: {gate.evidence}")
        for gate in diagnosis.gates
    ]
    if diagnosis.shape:
        blocks.append(detail("shape", f"workload shape: {diagnosis.shape}"))
    return Section(
        id="diagnosis", title="Diagnosis", blocks=tuple(blocks), title_audience="markdown"
    )


def _see_more_kv_memory(suggestions: Sequence[Suggestion]) -> str:
    for suggestion in suggestions:
        if suggestion.entry.name == "more-kv-memory":
            above = suggestion.tier == "try_first"
            return "; see `more-kv-memory`" + ("" if above else " under Could help")
    return ""


def _names(suggestions: Sequence[Suggestion]) -> str:
    return ", ".join(f"`{s.entry.name}`" for s in suggestions)


def next_steps_section(diagnosis: Diagnosis, suggested: Suggested, *, retained: bool) -> Section:
    """The playbook entries worth trying. "Try first": the suggestions of that tier, grouped by
    the bottleneck they address, failed gates first, then the primary bottleneck, then the
    secondary ones, at most three in full each. "Could help": the others, at most
    MAX_COULD_HELP in full. ``suggested`` is ranked, with final tiers and, per bottleneck, why
    its change is blocked (:func:`tensward.suggest.suggestions`). ``retained`` says the run kept
    its answers, so the follow-up run compares them."""
    suggestions, blocked = suggested.suggestions, suggested.blocked
    risk = (
        " (may change the outputs: the report then compares its answers with your current setup's)"
        if retained
        else " (may change the outputs: compare them before adopting it)"
    )

    def step(suggestion: Suggestion) -> SuggestionBlock:
        return SuggestionBlock(
            suggestion=suggestion, risk=risk if suggestion.entry.quality_risk else ""
        )

    blocks: list[Block] = []
    grouped = [s for s in suggestions if s.tier == "try_first"]
    addressed = {name for s in grouped for name in s.addresses}
    order = [
        gate.bottleneck
        for gate in diagnosis.gates
        if gate.state == "critical" and gate.bottleneck in addressed
    ]
    order += [*([diagnosis.primary] if diagnosis.primary else []), *diagnosis.secondary]
    if not suggestions and not order:
        text = (
            "No change suggested: this run did not measure enough to judge (see the checks above)."
            if _unjudged(diagnosis)
            else "Nothing to change: no measured signal points to a change for this workload."
        )
        blocks.append(Text(key="nothing", text=text))
    shown: set[str] = set()
    body: list[Block] = []
    for bottleneck in order:
        applying = [s for s in grouped if bottleneck in s.addresses]
        group = [s for s in applying if s.entry.name not in shown]
        if not applying and not blocked.get(bottleneck) and suggestions:
            continue  # nothing to say here; the suggestions are under Could help
        body.append(Text(key=f"for.{bottleneck}", text=f"For {NAMES[bottleneck]}:"))
        if applying and not group:
            body.append(item("see_above", f"see {_names(applying)} above"))
            continue
        if not group:
            line = blocked.get(bottleneck)
            body.append(
                item("blocked", f"{line}{_see_more_kv_memory(suggestions)}")
                if line
                else item("no_change", "no change in this engine's playbook applies here")
            )
            continue
        # A change that may alter the answers is not vetted as a step to try first: it is
        # only named.
        steps = [s for s in group if not s.entry.quality_risk][:MAX_STEPS_PER_CLASS]
        body += [step(suggestion) for suggestion in steps]
        if rest := [s for s in group if s not in steps]:
            body.append(item("also", f"also: {_names(rest)}"))
        shown.update(s.entry.name for s in group)
    if body:
        blocks.append(Text(key="try_first", text="Try first:"))
    blocks += body
    if others := [s for s in suggestions if s.entry.name not in shown]:
        blocks.append(Text(key="could_help", text="Could help:"))
        shown_in_full = [s for at, s in enumerate(others) if at < MAX_COULD_HELP or s.cost]
        blocks += [step(suggestion) for suggestion in shown_in_full]
        if rest := [s for s in others if s not in shown_in_full]:
            blocks.append(item("also", f"also: {_names(rest)}"))
    if suggested.not_applicable:
        blocks.append(Text(key="not_applicable", text="Not applicable here:"))
        blocks += [
            item(f"not_applicable.{name}", f"`{name}`: {why}")
            for name, why in suggested.not_applicable
        ]
    if any(s.command for s in suggestions):
        blocks += [
            Text(key="gap", text=""),
            Text(
                key="serve_hint",
                text="To serve a change, give `tensward serve start` the same `--project` and "
                "`--engine-arg` options.",
            ),
        ]
    return Section(id="next_steps", title="What to try next", blocks=tuple(blocks))
