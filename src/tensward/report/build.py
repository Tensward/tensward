"""The run report as data: every wording decision about a run, made once, into sections that
``report.md``, ``report.json`` and the terminal lay out."""

from __future__ import annotations

import json
import re
import shlex
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence, cast

from .. import __version__
from ..answer_options import (
    FP8_KV_FLOOR,
    FP8_KV_RULE,
    MARLIN_FP8_ARG,
    NATIVE_FP8,
    WEIGHT_ONLY,
    mode_words,
)
from ..answers_after import AnswersAfter, counts
from ..calibration import on_devices
from ..capabilities import Resolved
from ..capacity import CapacityResult
from ..ceilings import Ceilings
from ..classify import (
    NAMES,
    NOT_MODELLED,
    UNMEASURED,
    Bottleneck,
    Diagnosis,
    Finding,
    data_parallel_note,
    limit_name,
    name_of,
    power_capped,
)
from ..combine import MAX_MEMBERS, NUMERIC, goal_gain
from ..context import current as analysis
from ..engines import Engine
from ..errors import PreflightError
from ..gpu_state import GpuState
from ..history import (
    ADOPT_MIN,
    GOAL_METRIC,
    GUARDED,
    JUDGED,
    METRIC,
    OPEN_LOOP_RATE,
    WORDS,
    Compared,
    Memory,
    changes_answers_from,
    is_env,
    metric_figure,
    trade_summary,
)
from ..kb_entries import kb_block
from ..kept import (
    ACCEPT_FLAG,
    ANSWER_LEVERS,
    LAST,
    REGISTERED,
    answer_changes,
    changed_words,
    engine_arg_words,
    latest_run,
    set_command,
)
from ..measurement import WAVE_OUTLASTED_CHECK, GroupStats, ImageSplit, Measurement
from ..playbook import (
    GOAL_WORDS,
    RELEVANT,
    Entry,
    Suggestion,
    WorkloadFacts,
    fitting_max_context_len,
    goal_bottleneck,
)
from ..pressure import TOOL_SHARE, TTFT_GAP_MS
from ..profiling.counters import CountersResult
from ..profiling.trace import TraceResult
from ..project import CurrentSetup, ProjectFacts
from ..runs import RunMetrics
from ..settings import Settings
from ..slo import Slo
from ..stands import limit_word, stands_rows, where_it_stands
from ..suggest import (
    BUNDLED,
    Suggested,
    answer_warning,
    first_try,
    group_members,
    phase_two_guard,
    try_first_order,
)
from ..text import code_span, columns, figure, percent, printable
from ..thresholds import CALIBRATED_GPUS, MIN_CONFIDENT_REQUESTS
from .beyond import beyond_section
from .compare import tool_calls_changed
from .kb import METRIC_WORDS, source_summary
from .model import (
    Block,
    Data,
    Figure,
    Report,
    Section,
    SuggestionBlock,
    Text,
    change_words,
    item,
    ordered,
)
from .placement import advice_section, placement_section
from .profiling import ceilings_section, counters_section, trace_section
from .summary import (
    GLOSSARY,
    health_line,
    limit_line,
    next_line,
    result_line,
    reveal,
    side_heading,
    side_label,
    tool_line,
    tool_warning,
)

MAX_FAILURE_REASONS = 3
MAX_EXAMPLE_PROMPTS = 5
MAX_STEPS_PER_CLASS = 3
MAX_COULD_HELP = 3
# Why the prompt-capacity probe measured nothing, by CapacityResult.source.
PROBE_OUTCOMES = {
    "unknown-setup": "the GPU, engine version, step budget or concurrency cap is not known",
    "cannot-fill": "the prompts are too short to fill an engine step",
    "not-user-turn": "a conversation does not end with a user turn",
    "too-few-completions": "too few probe requests completed",
    "short-prompts": "the chat template made the probe's prompts shorter than intended",
    "long-prompts": "text outside the last user turn made the probe's prompts longer than intended",
    "cached-prefix": "the prefix cache served part of the probe's prompts",
    "failed": "a probe request or the engine's metrics failed",
    "pooled": "several engine replicas serve together, so no single engine's rate is known",
}
GOAL_TITLES = {
    "throughput": "throughput (requests per second)",
    "ttft": "time to first token (TTFT p95)",
    "tpot": "time per output token (TPOT p95)",
}
BEYOND_RESULT = "Another file or engine did better for this goal in our runs: see Beyond flags."
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
    empty_answers: int = 0  # succeeded requests that returned no text and no tool call
    capacity: CapacityResult | None = None  # the probed prompt capacity, for run.json
    placement: Mapping[str, Any] | None = None  # the engine's placement model, for metrics.json
    probes: Mapping[str, Any] | None = None  # the engine's probe results, for run.json


def overrides_suffix(engine_args: Sequence[str]) -> str:
    """ " + overrides (a=1, b=2)" when engine flags changed the setup, else nothing."""
    return f" + overrides ({', '.join(engine_args)})" if engine_args else ""


def ran_line(ran: str, cast: bool) -> str:
    """What ran, with the engine's float16 cast of a bfloat16 checkpoint when it happened."""
    return f"{ran}, served as float16" if cast else ran


def _stands_section(outputs: RunOutputs, measurement: Measurement | None) -> Section:
    """The one-line "Where it stands" from the signals this run measured."""
    gpu = outputs.diagnosis.gpu_state
    recorded = {
        "diagnosis": {
            "primary": outputs.diagnosis.primary,
            "requests": outputs.diagnosis.requests,
            "names": NAMES,
            "gpu_state": asdict(gpu) if gpu else None,
            "unexplained": outputs.diagnosis.unexplained,
        },
        "ceilings": asdict(measurement.ceilings) if measurement and measurement.ceilings else None,
        "placement": outputs.placement,
    }
    rows = [(label, now) for label, now, _, _ in stands_rows(None, recorded, speeds=False)]
    table = columns(("Where it stands", ""), [*rows, ("Limit", limit_word(recorded))])
    return Section(
        id="stands",
        title=None,
        blocks=(
            Text(key="stands", text=where_it_stands(recorded), audience="markdown"),
            Text(key="stands", text="\n".join(table), audience="terminal"),
        ),
    )


def build_report(outputs: RunOutputs) -> Report:
    """The whole report of a run: what ran, what changed against the current setup, the
    diagnosis and next steps, the measurements, the profiled sections, the comparison and the
    feedback line. A section with nothing to say is left out."""
    measurement, facts = outputs.measurement, outputs.facts
    trace, counters = outputs.trace, outputs.counters
    keep = keep_section(outputs)
    keep, verdict = _lift_verdict(keep)
    after_section = answers_after_section(
        outputs.suggested.answers_after if outputs.suggested else None
    )
    sections = [
        _header(outputs),
        _first_screen(outputs, verdict),
        Section(id="what_changed", title=None, blocks=_lines(outputs.changes)),
        _stands_section(outputs, measurement),
        diagnosis_section(
            outputs.diagnosis,
            outputs.gpu,
            measurement,
            outputs.rows,
            facts,
            capacity=outputs.capacity,
        ),  # fmt: skip
        *(
            [next_steps_section(outputs.diagnosis, outputs.suggested, retained=outputs.retained)]
            if outputs.suggested is not None
            else []
        ),
        worth_measuring_section(outputs.suggested, retained=outputs.retained)
        if outputs.suggested is not None
        else None,
        phase_two_section(outputs.suggested, retained=outputs.retained)
        if outputs.suggested is not None
        else None,
        placement_section(outputs.placement, outputs.probes, _resolved_facts(outputs)),
        advice_section(outputs.placement),
        beyond_section(analysis().kb_facts, analysis().goal, outputs.diagnosis, analysis().project)
        if outputs.suggested is not None
        else None,
        _setup(
            _kept_subject(outputs.subject, outputs.engine_args),
            outputs.settings,
            measurement,
            outputs.image,
            outputs.slo,
            outputs.diagnosis,
            _offers_gain(outputs.suggested),
        ),  # fmt: skip
        _requests(measurement, outputs.rows, facts, outputs.empty_answers),
        _performance(measurement, outputs.slo),
        _engine(
            measurement,
            outputs.engine,
            outputs.settings,
            facts,
            outputs.suggested.warm_kv if outputs.suggested else None,
        ),
        _defaults(outputs.settings, outputs.engine.defaults),
        ceilings_section(
            measurement.ceilings,
            batch_note=_client_batch_note(_unstepped(outputs.engine, outputs.settings)),
            at_clock=_at_clock(measurement.ceilings, outputs.diagnosis.gpu_state),
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
        _with_tool_changes(outputs, after_section),
        keep,
        Section(
            id="feedback",
            title=None,
            blocks=(Text(key="feedback", text=FEEDBACK_LINE, audience="markdown"),)
            if outputs.feedback
            else (),
        ),
    ]
    ended = outputs.suggested is not None and _phase_one_ended(outputs.suggested, keep)
    return Report(
        run_id=outputs.run_id,
        ran=ran_line(outputs.ran, measurement.served_as_float16),
        sections=_with_summary(
            outputs, ordered([s for s in sections if s is not None and s.blocks], first=ended)
        ),
        kb=_kb_source(outputs.suggested),
        versions=_versions(outputs),
        goal={
            "goal": analysis().goal,
            "limits": outputs.slo.declared,
            **(
                {
                    "plan": list(suggested.plan),
                    "check": suggested.check,
                    "stop": suggested.stop,
                    **({"ttft_basis_skipped": s} if (s := suggested.ttft_skipped) else {}),
                    **({"trade": suggested.trade} if suggested.trade is not None else {}),
                    **({"floor": suggested.floor} if suggested.floor is not None else {}),
                    **(
                        {"adopted_within_noise": adopted}
                        if (adopted := analysis().window_record.get("adopted_within_noise"))
                        else {}
                    ),
                }
                if (suggested := outputs.suggested) is not None
                else {}
            ),
        },
    )


def _resolved_facts(outputs: RunOutputs) -> Mapping[str, Any]:
    return outputs.resolved.facts if outputs.resolved else {}


def _kb_source(suggested: Suggested | None) -> dict[str, Any]:
    matches = analysis().kb_matches
    source = suggested.kb_source if suggested is not None else {}
    return source_summary(source, matches.unknown if matches is not None and source else ())


def _env_args() -> tuple[str, ...]:
    """The engine environment this run changed, as NAME=VALUE, in the analysis' arguments."""
    return tuple(
        sorted(arg for arg in analysis().engine_args if is_env(arg, analysis().flag_rules))
    )


def _kept_label(engine_args: Sequence[str]) -> str | None:
    """The label "kept setup (run R)" when the run measured the kept setup's arguments."""
    kept = analysis().kept
    env = _env_args()
    if kept is None or not (engine_args or env):
        return None
    if set(engine_args) != set(kept.engine_args) or set(env) != set(kept.env):
        return None
    return f"kept setup (run {kept.run})"


def _kept_subject(subject: Subject, engine_args: Sequence[str]) -> Subject:
    """The subject of a run of the kept setup titled as one."""
    kept = _kept_label(engine_args)
    if kept is None or subject.setup is None:
        return subject
    return replace(subject, title=f"Your {kept}: {', '.join(engine_args)}")


def _lines(lines: Sequence[str]) -> tuple[Block, ...]:
    return tuple(Text(key="line", text=line) for line in lines)


VERDICTS = ("not_kept", "kept_within_noise", "kept_failures")


def baseline_kind(label: str) -> str:
    """The source of a comparison's baseline, by the label of the current setup."""
    return {"imported from --current": "command", "declared in config": "config"}.get(
        label, "defaults"
    )


def _versions(outputs: RunOutputs) -> dict[str, str | None]:
    """Tensward, the engine and its version, the hardware and the baseline, for the header and
    report.json."""
    identity = analysis().gpu_identity
    total = outputs.diagnosis.gpu_state.memory_total_mib if outputs.diagnosis.gpu_state else None
    gpu = None
    if outputs.gpu:
        size = f" {round(total * 1.048576 / 1000)} GB" if total else ""
        gpu = f"{max(outputs.measurement.replicas, 1)}× {outputs.gpu}{size}"
    elif identity is None:
        gpu = "no GPU"
    return {
        "tensward": __version__,
        "engine": outputs.ran.split(" (", 1)[0].split(", ", 1)[0],
        "hardware": gpu,
        "baseline": baseline_words(baseline_kind(outputs.source)),
    }


def _lift_verdict(keep: Section | None) -> tuple[Section | None, Text | None]:
    """``keep`` without its verdict on the terminal, and the verdict for the first screen."""
    if keep is None:
        return None, None
    for at, block in enumerate(keep.blocks):
        if isinstance(block, Text) and block.key in VERDICTS:
            shown = (*keep.blocks[:at], replace(block, audience="markdown"), *keep.blocks[at + 1 :])
            return replace(keep, blocks=shown), replace(block, audience="terminal")
    return keep, None


def _first_screen(outputs: RunOutputs, verdict: Text | None) -> Section:
    """What the terminal shows first after the header: the verdict for the goal, the offer to
    try next and the answer check."""
    blocks: list[Block] = [verdict] if verdict is not None else []
    suggested = outputs.suggested
    if warning := _tool_warning(outputs.measurement):
        blocks.append(
            Text(
                key="tool_warning", text=f"Warning: {warning}", tone="warning", audience="terminal"
            )
        )
    after = suggested.answers_after if suggested is not None else None
    if after is not None and after.outcome is not None:
        found = counts(after.outcome)
        text = (
            f"Answers: {found['identical']} of {found['prompts']} identical to run "
            f"{after.before}'s, {found['differ']} differ (see Answers before / after)"
        )
        blocks.append(Text(key="answer_check", text=text, audience="terminal"))
    return Section(id="first_screen", title=None, blocks=tuple(blocks))


OFFLOAD_SHARE_SHOWN = 25  # percent of a step reading from host memory that is named as the limit


def _host_share(diagnosis: Diagnosis) -> int | None:
    """The percent of each step spent reading the model from CPU memory, when the offload
    finding fired and measured it."""
    for finding in diagnosis.findings:
        if finding.bottleneck == "offload" and finding.state in ("warning", "critical"):
            found = re.search(r"decode step \((\d+)%\) reads", finding.evidence)
            if found and int(found.group(1)) >= OFFLOAD_SHARE_SHOWN:
                return int(found.group(1))
    return None


def _tool_warning(measurement: Measurement) -> str | None:
    stats = measurement.tool_calls
    return None if stats is None else tool_warning(stats.requests, stats.calling)


def _with_summary(outputs: RunOutputs, sections: Sequence[Section]) -> tuple[Section, ...]:
    """``sections`` with the plain-words summary under the report's title (markdown) and the
    result line for the end of the terminal (json: the CLI prints it with the menu)."""
    offers = [
        block
        for section in sections
        for block in section.blocks
        if isinstance(block, SuggestionBlock) and block.suggestion.command
    ]
    measurement, diagnosis, goal = outputs.measurement, outputs.diagnosis, analysis().goal
    first = offers[0] if offers else None
    side = first.suggestion.side if first else None
    hint = reveal(outputs.suggested.underloaded if outputs.suggested else None)
    stats = measurement.tool_calls
    valid = (
        min(
            (v for v in (stats.valid_json, stats.known_tool, stats.schema_valid) if v is not None),
            default=None,
        )
        if stats is not None
        else None
    )
    lines = [
        health_line(measurement.succeeded, measurement.failed, measurement.request_throughput),
        limit_line(
            diagnosis.primary,
            diagnosis.requests > 0 and not _unjudged(diagnosis),
            limit_name(diagnosis),
            _host_share(diagnosis),
            diagnosis.unexplained,
        ),
        tool_line(stats.requests, stats.calling, valid) if stats is not None else None,
        _tool_warning(measurement),
        next_line(
            first.change or first.suggestion.entry.name if first else None,
            goal,
            first.test_seconds if first else None,
            side=side,
            hint=hint,
        ),
    ]
    summary = Text(
        key="summary",
        text="\n".join(f"- {line}" for line in lines if line) + f"\n\n_{GLOSSARY}_",
        audience="markdown",
    )
    said = result_line(
        measurement.succeeded,
        measurement.failed,
        len(offers),
        goal,
        side=side is not None,
        hint=hint,
    )
    if (first is None or side is not None) and any(s.id == "beyond" for s in sections):
        said += f" {BEYOND_RESULT}"
    result = Text(key="result", text=said, audience="json")
    has_screen = any(s.id == "first_screen" for s in sections)
    out = []
    for section in sections:
        if section.id == "first_screen":
            section = replace(section, blocks=(summary, result, *section.blocks))
        out.append(section)
        if section.id == "header" and not has_screen:
            out.append(Section(id="first_screen", title=None, blocks=(summary, result)))
    return tuple(out)


def _with_tool_changes(outputs: RunOutputs, section: Section | None) -> Section | None:
    """``section`` with a warning when a tool-call share moved against the run it was made from."""
    after = outputs.suggested.answers_after if outputs.suggested else None
    calls = outputs.measurement.tool_calls
    if section is None or after is None or calls is None:
        return section
    changed = tool_calls_changed(
        _earlier_tool_calls(analysis().project, after.before), asdict(calls)
    )
    if changed is None:
        return section
    warning = Text(key="tool_calls_changed", text=changed, tone="warning")
    return replace(section, blocks=(*section.blocks, warning))


def _header(outputs: RunOutputs) -> Section:
    measurement = outputs.measurement
    versions = _versions(outputs)
    blocks: list[Block] = [
        Text(key="title", text=f"# Tensward analysis {outputs.run_id}", audience="markdown"),
        Text(
            key="versions",
            text=" · ".join(
                part
                for part in (
                    f"Tensward {versions['tensward']}",
                    versions["engine"],
                    versions["hardware"],
                    versions["baseline"],
                )
                if part
            ),
        ),
        Text(key="ran", text=f"Ran: {ran_line(outputs.ran, measurement.served_as_float16)}"),
    ]
    if kept := _kept_label(outputs.engine_args):
        text = f"{kept}: {', '.join(outputs.engine_args)}"
        blocks.append(Text(key="source", text=text, audience="terminal"))
    elif outputs.source:
        setup = "registered setup" if analysis().kept is not None else "current setup"
        text = (
            f"{setup} ({outputs.source}) + this run's overrides: {', '.join(outputs.engine_args)}"
            if outputs.engine_args
            else f"{setup}: {outputs.source}"
        )
        blocks.append(Text(key="source", text=text, audience="terminal"))
    parts = _headline_parts(
        measurement, outputs.slo, outputs.diagnosis, _offers_gain(outputs.suggested)
    )
    blocks += [Text(key=key, text=f"  {part}", audience="json") for key, part in parts]
    table = columns(("Measured", ""), [_split(key, part) for key, part in parts])
    blocks.append(Text(key="headline", text="\n".join(table), audience="terminal"))
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
    blocks.append(Text(key="goal", text=f"goal: {GOAL_TITLES[analysis().goal]}"))
    if outputs.slo.declared:
        blocks.append(Text(key="limits", text=f"limits: {outputs.slo.describe()}"))
    after = outputs.suggested.answers_after if outputs.suggested else None
    if after is not None and after.broken_rose:
        text = BROKEN.format(after.broken_words())
        blocks.append(Text(key="broken_answers", text=text, tone="strong"))
    return Section(id="header", title=None, blocks=tuple(blocks))


HEADLINE_LABELS = (
    "TTFT p95 (slowest 1 in 20)",
    "TPOT p95 (slowest 1 in 20)",
    "hardware ceiling reached",
    "hardware ceiling:",
    "decode at",
    "goodput",
    "output",
    "total",
    "requests",
)


def _split(key: str, part: str) -> tuple[str, str]:
    """A headline number as its label and its value, for the terminal's table."""
    if key == "failed":
        count, _, _ = part.partition(" requests failed")
        return "failed", f"{count} requests"
    label = next((label for label in HEADLINE_LABELS if part.startswith(f"{label} ")), None)
    if label is None:
        return key, part
    value = part[len(label) + 1 :]
    if label == "decode at":
        return "decode vs ceiling", value.removesuffix(" of its ceiling")
    return label.removesuffix(":"), value


def _metric(label: str, value: float | None, unit: str, digits: int = 1) -> str:
    return f"{label} {figure(value, unit, digits)}"


def _offers_gain(suggested: Suggested | None) -> bool:
    """Whether a Try-first entry addresses prefill, GPU compute or decode bandwidth, or gains."""
    return suggested is not None and any(
        s.tier == "try_first"
        and (
            s.addresses & {"prefill", "gpu_compute", "decode_bandwidth"}
            or s.entry.prior_gain
            or (goal_gain(s.predicted, suggested.goal) or 0) > 0
        )
        for s in suggested.suggestions
    )


def _headline_parts(
    measurement: Measurement,
    slo: Slo,
    diagnosis: Diagnosis | None = None,
    offered_gain: bool = False,
) -> list[tuple[str, str]]:
    """The headline numbers of a run, each with its key."""
    total = measurement.succeeded + measurement.failed
    engine_rate = measurement.engine_output_throughput
    total_rate = measurement.total_throughput
    if engine_rate and total_rate is not None and measurement.output_throughput is not None:
        total_rate += engine_rate - measurement.output_throughput
    goodput = (
        [("goodput", _metric("goodput", measurement.goodput, " req/s", 2) + f" ({slo.describe()})")]
        if slo.declared
        else []
    )
    return [
        (
            "output",
            _metric("output", engine_rate or measurement.output_throughput, " tok/s")
            + (" (the engine's count, see Checks)" if engine_rate else ""),
        ),
        ("total", _metric("total", total_rate, " tok/s")),
        ("requests", _metric("requests", measurement.request_throughput, " req/s", 2)),
        *goodput,
        ("ttft_p95", _metric("TTFT p95 (slowest 1 in 20)", measurement.ttft_p95_ms, " ms", 0)),
        ("tpot_p95", _metric("TPOT p95 (slowest 1 in 20)", measurement.tpot_p95_ms, " ms")),
        ("failed", f"{measurement.failed} of {total} requests failed"),
        ("ceiling", _ceiling_metric(measurement.ceilings, diagnosis, offered_gain)),
    ]


def clock_scaled_share(c: Ceilings, g: GpuState | None) -> float | None:
    """The window's share of the hardware ceilings with the tensor peaks scaled by the measured
    SM clock against its maximum (memory bandwidth does not scale with it). Where the bandwidth
    bound decided decode, the compute side is taken at its upper bound, the prefill ceiling, so
    the share is never overstated. None without a clock or the figures it needs."""
    if g is None or not g.sm_mhz or not g.sm_max_mhz or c.ceiling_time_pct_of_window is None:
        return None
    if c.moe_experts is not None:
        return None
    decode_ceiling, prefill_ceiling = c.decode_ceiling_tok_s, c.prefill_ceiling_tok_s
    decoded, prefilled = c.measured_decode_tok_s, c.measured_prefill_tok_s
    bandwidth, weights = c.bandwidth_gbs, c.weight_bytes_per_step
    kv, batch = c.kv_bytes_per_sequence, c.avg_running_batch
    if (
        decode_ceiling is None or not prefill_ceiling or decoded is None or prefilled is None
        or bandwidth is None or weights is None or kv is None or not batch
    ):  # fmt: skip
        return None
    ratio = min(g.sm_mhz / g.sm_max_mhz, 1.0)
    by_bandwidth = batch * bandwidth * 1e9 / (weights + batch * kv)
    compute = decode_ceiling if decode_ceiling < 0.999 * by_bandwidth else prefill_ceiling
    decode = min(by_bandwidth, compute * ratio)
    return 100 * (decoded / decode + prefilled / (prefill_ceiling * ratio))


def _at_clock(c: Ceilings | None, g: GpuState | None) -> tuple[float, float, float] | None:
    """The clock-scaled share with the SM clock and its maximum it was scaled by."""
    share = clock_scaled_share(c, g) if c is not None else None
    if share is None or g is None or not g.sm_mhz or not g.sm_max_mhz:
        return None
    return share, g.sm_mhz, g.sm_max_mhz


def _ceiling_metric(
    ceilings: Ceilings | None, diagnosis: Diagnosis | None = None, offered_gain: bool = False
) -> str:
    """How close the GPU ran to its limits: the combined figure at the measured SM clock when
    the GPU was sampled, else the combined figure, else the decode share."""
    at_clock = _at_clock(ceilings, diagnosis.gpu_state) if diagnosis is not None else None
    if at_clock is not None and diagnosis is not None:
        share, sm_mhz, sm_max_mhz = at_clock
        text = (
            f"hardware ceiling reached {share:.0f}% (the datasheet peaks at the measured SM "
            f"clock, {sm_mhz:.0f} of {sm_max_mhz:.0f} MHz)"
        )
        saturated = {f.bottleneck: f.state for f in diagnosis.findings}.get("gpu_saturated")
        capped = diagnosis.gpu_state is not None and power_capped(diagnosis.gpu_state)
        if saturated == "critical" and not offered_gain and not capped:
            text += "; the GPU was saturated, so the rest is not reachable by server settings"
        return text
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
    subject: Subject,
    settings: Settings,
    measurement: Measurement,
    image: str | None,
    slo: Slo,
    diagnosis: Diagnosis | None = None,
    offered_gain: bool = False,
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
    parts = ", ".join(
        part for _, part in _headline_parts(measurement, slo, diagnosis, offered_gain)
    )
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
    measurement: Measurement,
    rows: Sequence[Mapping[str, Any]],
    facts: WorkloadFacts,
    empty_answers: int,
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
    if empty_answers:
        blocks.append(
            item(
                "empty_answers",
                f"NOTE: {empty_answers} of {measurement.succeeded} successful requests returned "
                "no text and no tool call (a model that thinks can spend its output limit on "
                "reasoning); their speed counts, their answers do not",
                "warning",
            )
        )
    blocks += _failures([row for row in rows if row["outcome"] != "success"])
    return Section(id="requests", title="Requests", blocks=tuple(blocks))


def _figure(key: str, label: str, value: float | None, unit: str, digits: int = 1) -> Figure:
    return Figure(key=key, label=label, value=value, unit=unit, digits=digits)


def _goodput_figures(m: Measurement, slo: Slo) -> list[Block]:
    """The goodput figures; counted only against limits the user declared."""
    if not slo.declared:
        return []
    return [
        _figure("goodput", f"goodput ({slo.describe()})", m.goodput, " req/s", 2),
        _figure("slo_attainment", "requests meeting the SLO", percent(m.slo_attainment), "%", 0),
    ]


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
        *_goodput_figures(m, slo),
        _figure("ttft_p50_ms", "TTFT p50", m.ttft_p50_ms, " ms"),
        _figure("ttft_p95_ms", "TTFT p95 (slowest 1 in 20)", m.ttft_p95_ms, " ms"),
        _figure("tpot_p50_ms", "TPOT p50", m.tpot_p50_ms, " ms"),
        _figure("tpot_p95_ms", "TPOT p95 (slowest 1 in 20)", m.tpot_p95_ms, " ms"),
        _figure("e2e_p95_ms", "end-to-end p95 (slowest 1 in 20)", m.e2e_p95_ms, " ms"),
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


def kv_room(m: Measurement, facts: WorkloadFacts) -> str | None:
    """The KV pool in this workload's requests: at its longest (the longest prompt plus the
    answer cap) and at its mean (the measured prompt and answer tokens per request)."""
    if not m.kv_capacity_tokens or not m.max_prompt_tokens:
        return None
    longest = m.max_prompt_tokens + facts.output_tokens
    text = (
        f"room for about {m.kv_capacity_tokens / longest:.0f} requests of this workload at its "
        f"longest ({m.max_prompt_tokens} prompt + {facts.output_tokens} answer tokens)"
    )
    if m.request_throughput and m.total_throughput:
        mean = m.total_throughput / m.request_throughput
        text += f", {m.kv_capacity_tokens / mean:.0f} at its mean ({mean:.0f} tokens)"
    return text


def _capacity(
    m: Measurement, facts: WorkloadFacts, settings: Settings, warm: tuple[str, float] | None
) -> Text:
    """The KV capacity the engine reports, in this workload's requests, and what Tensward
    estimated before the run, against the warm pool when a cold start shrank this one. The
    estimate counts the checkpoint's own weight bytes, so none is shown for weights quantized at
    start."""
    measured, estimate = m.kv_capacity_tokens, m.kv_capacity_estimate_tokens
    if measured is None:
        return item("kv_capacity", "KV cache capacity (measured by the engine): not measured")
    line = f"KV cache capacity (measured by the engine): {measured:.0f} tokens"
    room = kv_room(m, facts)
    if room:
        line += f", {room}"
    longest = (m.max_prompt_tokens or 0) + facts.output_tokens
    near_full = bool(m.max_context_len) and 2 * longest >= (m.max_context_len or 0)
    if m.kv_max_concurrency is not None and (room is None or near_full):
        line += f", {m.kv_max_concurrency:.1f} full-length requests at once"
    reference = measured
    if m.compile_cache == "cold":
        line += (
            "; the server started with a cold compile cache, which leaves less memory for the "
            "KV cache than a warm restart"
        )
        if warm is not None:
            line += f" (a warm start of these settings in run {warm[0]} held {warm[1]:.0f} tokens)"
            reference = warm[1]
    if settings.quantization is not None and facts.quantization is None:
        return item("kv_capacity", line + "; no estimate was made for weights quantized at start")
    if estimate is None:
        return item("kv_capacity", line + "; no estimate was made before the run")
    against = " against the warm pool" if reference != measured else ""
    return item(
        "kv_capacity",
        f"{line}; estimated before the run: {estimate:.0f} tokens "
        f"(error {estimate / reference - 1:+.0%}{against})",
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


def _approximate_kv(engine: Engine, settings: Settings) -> str:
    kv = engine.capabilities(None, settings).signals.get("kv_usage")
    if kv is None or kv.quality != "approximate":
        return ")"
    return f"; approximate, {kv.how}: {kv.bias})"


def _engine(
    measurement: Measurement,
    engine: Engine,
    settings: Settings,
    facts: WorkloadFacts,
    warm: tuple[str, float] | None,
) -> Section:
    m = measurement
    reason = _unstepped(engine, settings)
    unstepped = f" ({reason})" if reason else ""
    blocks: tuple[Block, ...] = (
        _figure(
            "peak_kv_usage",
            "highest sampled KV-cache usage (1 s polls" + _approximate_kv(engine, settings),
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
        _capacity(m, facts, settings, warm),
        Text(key="added_flags", text=f"Tensward adds to every launch: {engine.added_flags}"),
    )
    title = "Engine signals (from the engine's metrics)"
    if _approximate_kv(engine, settings) != ")":
        title = "Engine signals"
    return Section(id="engine", title=title, blocks=blocks)


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


def _support() -> tuple[Block, ...]:
    """What the engine and the GPU can run in place of these weights and KV cache, or that it
    could not be read; nothing outside the CLI."""
    context = analysis()
    options = context.answer_options
    if options is None:
        if context.project is None:
            return ()
        why = context.answer_options_unknown or "see answer_options.why in run.json"
        return (item("support", "Tensward could not read which quantization this engine and GPU "
                     f"can run ({why}), so it offers no weight quantization."),)  # fmt: skip
    gpu = options.capability
    parts = [
        f"{method} weights made from this checkpoint at start ({mode_words(mode)})"
        for method, mode in options.on_the_fly.items()
    ]
    names = [f.name for f in options.checkpoints]
    parts.append(
        f"pre-quantized {', '.join(names[:-1])} or {names[-1]} checkpoints"
        if len(names) > 1
        else f"pre-quantized {names[0]} checkpoints"
        if names
        else "no pre-quantized checkpoint"
    )
    parts.append(
        "an fp8 KV cache" if "fp8" in options.kv_cache else f"no fp8 KV cache ({FP8_KV_RULE})"
    )
    capability = f"compute capability {gpu:g}" if gpu is not None else "compute capability unknown"
    line = f"This engine and GPU ({capability}) can run: {'; '.join(parts)}."
    if gpu is not None and gpu >= NATIVE_FP8:
        line += (" W8A8 fp8 and the fp8 KV cache are read from vLLM 0.30.0's source, not run by "
                 "Tensward with vLLM 0.30.0.")  # fmt: skip
    elif options.on_the_fly.get("fp8") == WEIGHT_ONLY:
        line += (f" fp8 weights here need {MARLIN_FP8_ARG} (measured with vLLM 0.30.0 on an A10G, "
                 "8.6). An fp8 KV cache also starts there for 128-wide attention heads, but it "
                 "broke the answers of the model measured (Qwen2.5-7B), so Tensward does not "
                 f"offer it below {FP8_KV_FLOOR:g}.")  # fmt: skip
    return (item("support", line),)


def _quantization(measurement: Measurement, facts: WorkloadFacts) -> Section:
    """The checkpoint's quantization and the kernels the server log says were selected, then
    what the engine and the GPU can run instead."""
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
    return Section(id="quantization", title="Quantization", blocks=(*blocks, *_support()))


def _tool_calls(measurement: Measurement) -> Section:
    """Whether the model called tools correctly. Quality observations, not performance."""
    title = "Tool calling (quality, not performance)"
    if (stats := measurement.tool_calls) is None:
        return Section(id="tool_calls", title=title, blocks=())

    def share(value: float | None) -> str:
        return "no tool call to judge" if value is None else f"{value:.0%}"

    of = f"of the {stats.calling} that did,"
    warned = tool_warning(stats.requests, stats.calling)
    blocks = (
        *([Text(key="few_calls", text=f"Warning: {warned}", tone="warning")] if warned else []),
        item("offered", f"requests that offered tools: {stats.requests}"),
        item(
            "produced_call",
            f"produced a tool call: {stats.calling} of {stats.requests} requests "
            f"({share(stats.produced_call)})",
        ),
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


EXAMPLES = 3  # prompts shown before / after
BROKEN = (
    "Warning: answers look broken after this change: {}. Tensward does not judge answer quality, "
    "but these are not answers: read them under Answers before / after before keeping it."
)
EXAMPLE_CHARS = 300  # per answer shown


def _short(text: str) -> str:
    one_line = " ".join(printable(text).split())
    if not one_line:
        return "(empty)"
    return code_span(one_line if len(one_line) <= EXAMPLE_CHARS else one_line[:EXAMPLE_CHARS] + "…")


def _rates(name: str, rates: tuple[float | None, float | None]) -> str | None:
    if rates[0] is None or rates[1] is None:
        return None
    return f"{name}: {rates[0]:.1%} before, {rates[1]:.1%} after"


def answers_after_section(shown: AnswersAfter | None) -> Section | None:
    """This run's answers next to those of the run it was made from, when its change can
    affect them. Tensward counts and shows them; it never says which are better."""
    if shown is None:
        return None
    title = "Answers before / after"
    if shown.shown_in is not None:
        text = (
            "The answers of these settings were shown before / after in run "
            f"{shown.shown_in}'s report."
        )
        return Section(id="answers_after", title=title, blocks=(Text(key="shown_in", text=text),))
    if shown.outcome is None:
        runs = " and ".join(f"run {run}" for run in shown.missing)
        text = (
            f"{runs[0].upper()}{runs[1:]} kept no answers (--no-retain-responses), so they "
            "cannot be shown before and after."
        )
        return Section(id="answers_after", title=title, blocks=(Text(key="kept_none", text=text),))
    outcome = shown.outcome
    changes = ", ".join(ANSWER_LEVERS.get(name, name) for name in shown.changes)
    found = counts(outcome)
    calls, text = found["tool_calls"], found["text"]

    def sent(count: int, noun: bool = False) -> str:
        if shown.matched:
            return f"the {count} prompts both runs sent"
        return f"{count} {'prompt' if count == 1 else 'prompts'}" if noun else str(count)

    overlap = (
        f" (word overlap {text['similarity']:.2f} on average, where 1 is the same words)"
        if text["differ"] and text["similarity"] is not None
        else ""
    )
    if calls["prompts"]:
        count = (
            f"Tool calls: {calls['identical']} of {sent(calls['prompts'])} identical (name and "
            f"arguments); text: {text['differ']} of {sent(text['prompts'])} differ from run "
            f"{shown.before}'s{overlap}"
        )
    else:
        same = found["identical"]
        count = (
            f"Answers differ from run {shown.before}'s on {found['differ']} of "
            f"{sent(found['prompts'], True)}{overlap}; {same} "
            f"{'is' if same == 1 else 'are'} identical"
        )
    if found["judged"]:
        count += (
            f"; {found['changed']} differ by more than run {shown.before}'s own repeated answers do"
        )
    if unanswered := found["unanswered"]:
        count += f"; {unanswered} {'has' if unanswered == 1 else 'have'} no answer on one side"
    (broke_before, asked), (broke_after, answered) = shown.broken
    broken = (
        f"answers that repeat themselves or are empty: {broke_before} of {asked} before, "
        f"{broke_after} of {answered} after"
    )
    before_invalid, after_invalid = outcome.health.get("invalid JSON", (None, None))
    valid = (
        (None, None)
        if before_invalid is None or after_invalid is None
        else (1 - before_invalid, 1 - after_invalid)
    )
    lines = [
        count + ".",
        broken,
        _rates("valid JSON", valid),
        _rates("failed requests", outcome.health["failed"]),
    ]
    blocks: list[Block] = [
        Text(key="changes", text=f"This run changes {changes} from run {shown.before}."),
        *(Text(key="count", text=line, item=True) for line in lines if line),
    ]
    if shown.broken_rose:
        blocks.append(Text(key="broken", text=BROKEN.format(shown.broken_words()), tone="warning"))
    differing = [pair for pair in shown.pairs if pair["identical"] is False]
    others = [pair for pair in shown.pairs if pair["identical"]]
    if shown_pairs := differing[:EXAMPLES]:
        many = len(shown_pairs) > 1
        blocks.append(Text(
            key="examples",
            text=f"The {len(shown_pairs)} least similar {'prompts' if many else 'prompt'}:",
        ))  # fmt: skip
    for pair in differing[:EXAMPLES] or others[:1]:
        note = "" if pair["identical"] is False else " (identical)"
        before, after = (
            _short(pair[side][0]) if pair[side] else "(no answer)" for side in ("before", "after")
        )
        blocks += [
            Text(key="example", text=f"Prompt `{pair['prompt_id']}`{note}:"),
            Text(key="before", text=f"before: {before}", item=True),
            Text(key="after", text=f"after: {after}", item=True),
        ]
    blocks.append(Text(
        key="where",
        text="Every pair is in quality.json in this run's directory. "
        "Tensward does not judge which answers are better.",
    ))  # fmt: skip
    return Section(id="answers_after", title=title, blocks=tuple(blocks))


def _starting_setup(text: str) -> Section:
    blocks = (Text(key="kept", text=text),)
    return Section(id="keep", title="Starting setup", blocks=blocks, title_audience="markdown")


def keep_section(outputs: RunOutputs) -> Section | None:
    """How to keep this run's settings with ``tensward set``, or that they are kept already.
    None outside a project analysis."""
    context = analysis()
    project, kept = context.project, context.kept
    if project is None or outputs.subject.setup is None:
        return None
    engine_args, env = tuple(outputs.engine_args), _env_args()
    if not (engine_args or env):
        text = "This run measured your current setup; `tensward analyse` starts from it already."
        return _starting_setup(text)
    if kept is not None and (set(engine_args), set(env)) == (set(kept.engine_args), set(kept.env)):
        return _starting_setup(
            f"This run measured the kept setup (run {kept.run}); `tensward analyse` starts from "
            "it already, and compares each run of it with the registered setup's."
        )
    title = "To keep this change"
    if outputs.diagnosis.requests == 0:
        text = "This run served no request, so its settings cannot be kept."
        return Section(id="keep", title=title, blocks=(Text(key="not_kept", text=text),))
    registered = outputs.subject.setup.engine_settings
    quantized = outputs.facts.quantization is not None
    changes = answer_changes(
        outputs.engine, registered, outputs.settings, checkpoint_quantized=quantized
    )
    newest = latest_run(project)
    run = LAST if newest is None or newest <= outputs.run_id else outputs.run_id
    command = set_command(project, run, unreviewed=bool(changes))
    if outputs.suggested is not None and outputs.suggested.carried is not None:
        text = (
            f"Tensward does not recommend keeping these settings: {outputs.suggested.carried}. "
            f"If you have read the answers and still want them: `{command}`."
        )
        return Section(
            id="keep", title=title, blocks=(Text(key="not_kept", text=text, tone="warning"),)
        )
    undo = set_command(
        project,
        kept.run if kept else REGISTERED,
        unreviewed=bool(kept and kept.answer_changes),
    )
    after = outputs.suggested.answers_after if outputs.suggested else None
    verdict = _verdict(outputs, broken=after is not None and after.broken_rose)
    blocks: list[Block] = [verdict] if verdict is not None else []
    if (step := _step_data(outputs, verdict, after)) is not None:
        blocks.append(step)
    blocks += _ahead(outputs, registered, quantized, behind=True)
    if outputs.suggested is not None and (trade := outputs.suggested.trade) is not None:
        blocks.append(Text(key="trade", text=trade_summary(trade, context.goal)))
    if after is not None and after.broken_rose:
        blocks.append(Text(
            key="broken_answers",
            text=f"Answers look broken in this run ({after.broken_words()}); keeping it is your "
            "choice.",
            tone="warning",
        ))  # fmt: skip
    refused = verdict is not None and verdict.key == "not_kept"
    shown = f"If you still want it: `{command}`" if refused else None
    by_hand = refused or not context.try_step  # `tensward try` keeps a step by itself
    title = title if by_hand else "This change"
    if by_hand:
        kept_words = f"the kept setup (run {kept.run})" if kept else "the registered setup"
        blocks += [
            Text(key="command", text=shown or f"`{command}`"),
            Text(
                key="meaning",
                text=f"makes these settings (`{engine_arg_words(engine_args, env)}`) the kept "
                "setup: `tensward analyse` and `tensward serve start --from current` then start "
                f"from them instead of {kept_words}; `{undo}` undoes it.",
            ),
        ]
    if changes and by_hand:
        blocks.append(
            Text(
                key="answer_changes",
                text=f"It needs {ACCEPT_FLAG} because this run changes "
                + changed_words(
                    outputs.engine,
                    registered,
                    engine_args,
                    changes,
                    checkpoint_quantized=quantized,
                )
                + ", which can change answers: compare them under Answers before / after first.",
                tone="warning",
            )
        )
    blocks += _ahead(outputs, registered, quantized, behind=False)
    rejection = next((r for r in context.rejected if r.run == outputs.run_id), None)
    if rejection is None:
        reject = shlex.join(["tensward", "reject", run, "--project", str(project), "--reason"])
        text = f'Not worth it? `{reject} "<why>"` tells Tensward not to offer it again here.'
    else:
        why = f" ({rejection.reason})" if rejection.reason else ""
        clear = shlex.join(
            ["tensward", "reject", "--clear", rejection.run, "--project", str(project)]
        )
        text = f"You rejected this change{why}; `{clear}` takes that back."
    blocks.append(Text(key="reject", text=text))
    return Section(id="keep", title=title, blocks=tuple(blocks))


def baseline_words(source: str) -> str:
    """What a comparison is against, by the source of the current setup."""
    return {
        "command": "vs your command",
        "defaults": "vs engine defaults (not your real setup)",
        "config": "vs your config",
    }[source]


def _step_data(
    outputs: RunOutputs, verdict: Text | None, after: AnswersAfter | None
) -> Data | None:
    """The step's verdict for report.json: the goal metric before and after, the noise band, the
    answers and tool calls compared with the run it was made from; None without that run."""
    memory = outputs.suggested.memory if outputs.suggested else None
    found = memory.parent if memory else None
    setup = outputs.subject.setup
    if memory is None or found is None or setup is None or found.before is None:
        return None
    goal = GOAL_METRIC[analysis().goal]
    side = _side_metric(memory)
    metric = side or goal
    delta, kept_on = found.deltas.get(metric), found.deltas.get(goal)
    before, now = (
        found.before.values.get(metric),
        (found.own.values.get(metric) if found.own else None),
    )
    if delta is None or before is None or now is None or kept_on is None:
        return None
    before, now, spread = _metrics_figures(
        analysis().project, found.base, outputs.measurement, metric, before, now
    )
    key = verdict.key if verdict is not None else None
    answers = None
    if after is not None and after.outcome is not None:
        shown = counts(after.outcome)
        answers = {
            "identical": shown["identical"],
            "differ": shown["differ"],
            "broken": after.broken[1][0],
            "judged": shown["judged"],
            "changed": shown["changed"],
            "total": shown["prompts"],
        }
    calls = outputs.measurement.tool_calls
    return Data(key="step", data={
        "metric": metric,
        "before": before,
        "after": now,
        "typical": spread,
        "change": delta.change,
        "band": delta.band,
        "verdict": delta.verdict,
        "recommended": key == "kept_within_noise"
        or (verdict is None and delta.verdict == "better"),
        "reason": verdict.text if key == "not_kept" and verdict is not None else None,
        "answers": answers,
        "tool_calls": None if calls is None else {
            "before": _earlier_tool_calls(analysis().project, found.base),
            "after": asdict(calls),
        },
        "baseline": setup.source,
        **({"goal": {
            "metric": goal, "change": kept_on.change, "band": kept_on.band,
            "verdict": kept_on.verdict,
        }} if side else {}),
    })  # fmt: skip


METRICS_FIELD = {
    "req_s": "request_throughput",
    "output_tok_s": "output_throughput",
    "ttft_p95_ms": "ttft_p95_ms",
    "tpot_p95_ms": "tpot_p95_ms",
    "e2e_p95_ms": "e2e_p95_ms",
}
TYPICAL_FIELD = {"ttft_p95_ms": "ttft_p50_ms", "tpot_p95_ms": "tpot_p50_ms"}


def _metrics_figures(
    project: Path | None,
    parent: str,
    own: Measurement,
    metric: str,
    before: float,
    now: float,
) -> tuple[float, float, dict[str, float] | None]:
    """The goal metric of the parent and of this run as their ``metrics.json`` holds it (the
    figures the whole run measured, not the window the noise band is judged on), with the
    typical (p50) value of a latency metric on both sides; the window figures stand in for a
    file that cannot be read."""
    field = METRICS_FIELD.get(metric)
    try:
        if project is None or field is None:
            raise OSError
        earlier = json.loads((project / "runs" / parent / "metrics.json").read_text("utf-8"))
    except (OSError, ValueError):
        return before, now, None
    was, is_ = earlier.get(field), getattr(own, field, None)
    if not isinstance(was, (int, float)) or is_ is None:
        return before, now, None
    typical = TYPICAL_FIELD.get(metric)
    spread = None
    if typical is not None:
        p50_was, p50_is = earlier.get(typical), getattr(own, typical, None)
        if isinstance(p50_was, (int, float)) and p50_is is not None:
            spread = {"before": float(p50_was), "after": float(p50_is)}
    return float(was), float(is_), spread


def _earlier_tool_calls(project: Path | None, run: str) -> dict[str, Any] | None:
    if project is None:
        return None
    try:
        return RunMetrics.read(project / "runs" / run / "metrics.json").tool_calls
    except PreflightError:
        return None


def _verdict(outputs: RunOutputs, *, broken: bool) -> Text | None:
    """Why Tensward does not recommend keeping this run's change (``not_kept``), or that it does
    only under the within-noise policy (``kept_within_noise``, recorded in the context's
    ``window_record``); None when it recommends it: a gain on the goal beyond its noise band
    against the run it was made from (for a change with no effect on the goal, nothing worse
    beyond its band), the floor and the declared limits held and, for a combined step, its
    anchor's prediction reached."""
    context = analysis()
    context.window_record.pop("adopted_within_noise", None)
    memory = outputs.suggested.memory if outputs.suggested else None
    found = memory.parent if memory else None
    if memory is None or found is None:
        return None
    metric = GOAL_METRIC[context.goal]
    delta = found.deltas.get(metric)
    longer = found.why is None and found.own is not None and found.own.requests > (found.upto or 0)
    matched = f" ({found.matched_words})" if found.why is not None or longer else ""
    held = _held(outputs, found)
    bundle = _lossless_bundle(memory.offered, outputs.engine)
    reason: str | None
    if found.removed is not None:
        return Text(
            key="kept_failures",
            text=(
                f"Speed and latency are {found.not_comparable}; this change removed the failed "
                "requests, so it is not judged a loss."
            ),
        )
    side = _side_metric(memory)
    if memory.short is not None:
        reason = memory.short
    elif delta is None:
        reason = f"{WORDS[metric]} was not measured in both runs"
    elif side is not None:
        reason = _side_reason(found, side, metric) or held
    elif metric == "req_s" and found.open_loop:
        reason = OPEN_LOOP_RATE
    elif not bundle and _no_effect(outputs, memory):
        worse = [found.deltas[m] for m in JUDGED if m in found.deltas and _worse(found, m)]
        reason = f"a loss: {found.shown(worse[0].metric, verdict=False)}" if worse else held
    elif delta.verdict == "worse":
        reason = f"a loss: {found.shown(metric, verdict=False)}"
    elif delta.verdict == "within_noise":
        reason = held or f"within noise: {found.shown(metric, verdict=False)}"
        if held is None and _adopted(found, metric, broken=broken, bundle=bundle):
            quantized = outputs.facts.quantization is not None
            if not changes_answers_from(found.base, outputs.settings, outputs.engine,
                                        quantized=quantized):  # fmt: skip
                return _adopt(found, metric, matched, bundle=bundle)
            reason += ", and it can change answers: within noise it is not kept"
    else:
        reason = held
    if reason is None:
        return None
    text = f"Tensward does not recommend keeping this change: {reason}{matched}."
    return Text(key="not_kept", text=text, tone="warning")


def _side_metric(memory: Memory | None) -> str | None:
    """The window metric of the other goal the offer this run tested was made for, when it was
    offered for that goal alone."""
    side = ((memory.offered if memory else None) or {}).get("side")
    return GOAL_METRIC.get(side) if isinstance(side, str) else None


def _side_reason(found: Compared, side: str, goal: str) -> str | None:
    """Why a change offered for the ``side`` metric is not kept: the goal metric worse beyond its
    band, or the side metric not better beyond its own; None when it may be kept."""
    if _worse(found, goal):
        return f"a loss: {found.shown(goal, verdict=False)}"
    delta = found.deltas.get(side)
    if delta is None:
        return f"{WORDS[side]} was not measured in both runs"
    if _worse(found, side):
        return f"a loss: {found.shown(side, verdict=False)}"
    if delta.verdict != "better":
        return f"within noise: {found.shown(side, verdict=False)}"
    return None


def _worse(found: Compared, metric: str) -> bool:
    return found.deltas[metric].verdict == "worse" and not (metric == "req_s" and found.open_loop)


def _no_effect(outputs: RunOutputs, memory: Memory) -> bool:
    """Whether the playbook says the change offered for this run leaves the goal alone."""
    name = (memory.offered or {}).get("name")
    entry = next((e for e in outputs.engine.playbook() if e.name == name), None)
    effect = entry.effects.get(analysis().goal) if entry is not None else None
    return effect is not None and effect.direction == "none"


def _held(outputs: RunOutputs, found: Compared) -> str | None:
    """Why the floor or a declared limit does not hold on this run's point values; a limit the
    run it was made from was past already holds while the metric is not worse beyond its band."""
    floor = outputs.suggested.floor if outputs.suggested else None
    if floor is not None and floor["crossed"]:
        then, now = floor["req_s"]
        at = "at your floor within noise" if floor.get("within_noise") else "past your floor"
        return (
            f"{at}: requests per second {now:.3g}, {1 - now / then:.0%} below run "
            f"{floor['base']}'s {then:.3g} (your floor: {floor['floor']:.0%})"
        )
    m = outputs.measurement
    for name, metric, value in (
        ("ttft_ms", "ttft_p95_ms", m.ttft_p95_ms), ("tpot_ms", "tpot_p95_ms", m.tpot_p95_ms),
    ):  # fmt: skip
        limit = outputs.slo.declared.get(name)
        if limit is None or value is None or value <= limit:
            continue
        before = found.before.values.get(metric) if found.before else None
        if before is None or before <= limit or _worse(found, metric):
            return f"past your {WORDS[metric]} limit: {value:,.1f} ms against {limit:g} ms"
    return None


def _lossless_bundle(offered: Mapping[str, Any] | None, engine: Engine) -> bool:
    """Whether the parent offered this change as a bundle of lossless small levers."""
    if offered is None or BUNDLED not in offered.get("reason", ""):
        return False
    members = offered.get("members", ())
    entries = {e.name: e for e in engine.playbook()}
    return 2 <= len(members) <= MAX_MEMBERS and all(
        name in entries and name not in NUMERIC and not entries[name].changes_answers
        for name in members
    )


def _adopted(found: Compared, metric: str, *, broken: bool, bundle: bool = False) -> bool:
    """Whether the within-noise policy adopts a step within noise on the goal: its point change
    is better by ADOPT_MIN at least, none of GUARDED is worse beyond its band, and its answers
    are not flagged broken. A lossless bundle needs no positive gain or opt-in (the floor,
    limits and a combined step's check are ``_verdict``'s)."""
    if broken or any(m in found.deltas and _worse(found, m) for m in GUARDED):
        return False
    if bundle:
        return True
    if not analysis().adopt_within_noise:
        return False
    change = found.deltas[metric].change
    gain = -change if metric in ("ttft_p95_ms", "tpot_p95_ms") else change
    return gain >= ADOPT_MIN


def _adopt(found: Compared, metric: str, matched: str, *, bundle: bool = False) -> Text:
    context = analysis()
    delta = found.deltas[metric]
    policy = "flag" if "--adopt-within-noise" in context.flags else "environment"
    context.window_record["adopted_within_noise"] = {
        "metric": metric,
        "change": delta.change,
        "band": delta.band,
        "policy": "lossless_bundle" if bundle else policy,
    }
    if bundle:
        return Text(key="kept_within_noise", text=(
            f"Within noise ({found.shown(metric, verdict=False)}){matched}; recommended: this "
            "lossless bundle holds the floor and declared limits, and nothing guarded worsened "
            "beyond its band."
        ))  # fmt: skip
    text = (
        f"Within noise ({found.shown(metric, verdict=False)}){matched}; recommended under "
        "--adopt-within-noise: the point estimate is better and nothing limited worsened beyond "
        "its band."
    )
    return Text(key="kept_within_noise", text=text)


def _ahead(
    outputs: RunOutputs, registered: Settings, quantized: bool, *, behind: bool
) -> list[Block]:
    """The ``tensward set`` line for the best earlier run on the goal: ``behind``, when this run
    is worse than it beyond noise (it comes first); else when it did as well within noise."""
    memory = outputs.suggested.memory if outputs.suggested else None
    project, goal = analysis().project, analysis().goal
    if memory is None or memory.ahead is None or memory.behind != behind or project is None:
        return []
    best = memory.ahead
    field, words, _ = METRIC[goal]
    then, now = best.metrics.get(field), getattr(outputs.measurement, field)
    if then is None or now is None:
        return []
    settings = Settings.from_json(dict(best.settings))
    changes = answer_changes(outputs.engine, registered, settings, checkpoint_quantized=quantized)
    command = set_command(
        project, best.run_id if best.engine_args else REGISTERED, unreviewed=bool(changes)
    )
    figures = f"{metric_figure(then, field)} against {metric_figure(now, field)} in this run"
    text = (
        f"This run is worse on {words} than run {best.run_id}, the best run so far ({figures}): "
        f"`{command}` keeps that run's settings instead."
        if behind
        else f"Run {best.run_id} did as well on {words}, within noise ({figures}): `{command}` "
        "keeps that run's settings instead."
    )
    return [Text(key="best", text=text)]


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
        notes = [f"how sure: {diagnosis.confidence}"]
    if diagnosis.primary and not all(f.calibrated for f in decided):
        some = "some " if any(f.calibrated for f in decided) else ""
        notes.append(f"{some}cut-offs not yet checked on real hardware")
    notes += [count] if few else []
    if diagnosis.primary in (None, "load_limited") and diagnosis.unexplained:
        return f"Bottleneck: not named; {diagnosis.unexplained} ({'; '.join(notes)})"
    if diagnosis.primary == "load_limited":
        return (
            "Bottleneck: none in the server; the GPU was not saturated at your concurrency "
            f"({'; '.join(notes)})"
        )
    name = name_of(diagnosis.primary) if diagnosis.primary else "none found"
    return f"Bottleneck: {name} ({'; '.join(notes)})"


def _outside_calibration(diagnosis: Diagnosis, gpu: str | None) -> bool:
    """A calibrated class named the bottleneck with "high" confidence on another GPU."""
    primary = next((f for f in diagnosis.findings if f.bottleneck == diagnosis.primary), None)
    if gpu is None or primary is None or not primary.calibrated or diagnosis.confidence != "high":
        return False
    return not on_devices(gpu, CALIBRATED_GPUS)


def _server_ttft(
    measurement: Measurement, rows: Sequence[Mapping[str, Any]], facts: WorkloadFacts
) -> str | None:
    """Where time to first token goes for tool calls: the answers to requests that offered tools
    reach the client well after the engine's first token."""
    stats, server = measurement.tool_calls, measurement.server_ttft_mean_ms
    if stats is None or stats.produced_call is None or server is None:
        return None
    if not isinstance(facts, ProjectFacts):
        return None
    done = [row for row in rows if row["outcome"] == "success"]
    asked = [row for row in done if row["prompt_index"] in facts.tool_prompts]
    share = stats.produced_call * len(asked) / len(done) if done else 0.0
    waits = [
        (row["first_content_ns"] - row["dispatch_ns"]) / 1e6
        for row in asked
        if row["first_content_ns"] is not None
    ]
    if share < TOOL_SHARE or not waits or fmean(waits) - server < TTFT_GAP_MS:
        return None
    client = fmean(waits)
    return (
        f"tool-call answers reach the client {(client - server) / 1000:.1f} s after the engine's "
        f"first token ({client:.0f} ms mean at the client against {server:.0f} ms in the engine; "
        f"{share:.0%} of answers are tool calls): the tool-call parser holds output until a call "
        "is complete, and API-server load adds to it"
    )


def diagnosis_section(
    diagnosis: Diagnosis,
    gpu: str | None,
    measurement: Measurement | None = None,
    rows: Sequence[Mapping[str, Any]] = (),
    facts: WorkloadFacts | None = None,
    *,
    capacity: CapacityResult | None = None,
) -> Section:
    """The bottleneck and its evidence, what else crossed, which classes did not cross and what
    could not be told, then the gates and the workload shape. Under a latency goal, what holds
    that goal back comes first and its classes lead the secondary ones. The terminal shows the
    headlines only. With ``measurement``, a tool-call run whose client TTFT is well over the
    engine's says why (right after the goal's headline under the TTFT goal)."""
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
            detail(f"nearest.{f.bottleneck}", f"nearest: {name_of(f.bottleneck)}: {f.evidence}")
            for f in nearest
        ]
    blocks += [
        detail(
            f"symptom.{name}",
            f"also seen, as its symptom: {name_of(name)}: {by_class[name].evidence}",
        )
        for name in diagnosis.symptoms
    ]
    goal = analysis().goal
    for name in sorted(diagnosis.secondary, key=lambda b: b not in RELEVANT.get(goal, ())):
        finding = by_class[name]
        blocks.append(
            detail(
                f"also_seen.{name}",
                f"also seen: {name_of(name)} (how sure: {finding.confidence}): {finding.evidence}",
            )
        )
    if (
        measurement is not None
        and facts is not None
        and (held := _server_ttft(measurement, rows, facts))
    ):
        blocks.insert(0 if goal == "ttft" else len(blocks), detail("diagnosis.server_ttft", held))
    decided = [
        f for f in diagnosis.findings if f.state == "clear" and f.bottleneck != "speculation"
    ]
    speculation = by_class["speculation"]
    if speculation.state == "clear" and speculation.threshold:
        decided.append(speculation)
    if decided and diagnosis.requests:
        for key, label, calibrated in (
            ("not_crossed", "not crossed", True),
            ("not_crossed_uncalibrated", "not crossed (cut-offs not yet checked)", False),
        ):
            names = [name_of(f.bottleneck) for f in decided if bool(f.calibrated) is calibrated]
            if names:
                blocks.append(detail(key, f"{label}: {', '.join(names)}"))
    if speculation.state == "clear" and speculation.threshold is None:
        blocks.append(detail("speculation", "speculation: off or not reported"))

    def cant_tell(listed: Sequence[Finding]) -> str:
        return "Not checked on this run:" + "".join(
            f"\n  - {name_of(f.bottleneck)} — {f.evidence}" for f in listed
        )

    unknown = [f for f in diagnosis.findings if f.state == "cant_tell"]
    unmeasured = [f for f in unknown if UNMEASURED.get(f.bottleneck) == f.evidence]
    if printed := [f for f in unknown if f not in unmeasured]:
        blocks.append(detail("cant_tell", cant_tell(printed)))
    if unmeasured:
        blocks.append(
            Text(key="cant_tell_unmeasured", text=cant_tell(unmeasured), item=True, audience="json")
        )
    blocks.append(
        detail(
            "not_modelled",
            "not modelled: "
            + "; ".join(
                f"{name_of(k)} ({why})" for k, why in NOT_MODELLED.items() if k not in by_class
            ),
        )
    )
    blocks += [
        detail(f"gate.{gate.bottleneck}", f"{name_of(gate.bottleneck)}: {gate.evidence}")
        for gate in diagnosis.gates
    ]
    if capacity is not None and capacity.source not in ("probed", "cached"):
        why = PROBE_OUTCOMES.get(capacity.source, capacity.source)
        blocks.append(
            detail("capacity_probe", f"prompt capacity: not measured ({why}), so there is no "
                   "prompt-work share this run")
        )  # fmt: skip
    if diagnosis.shape:
        blocks.append(detail("shape", f"workload shape: {diagnosis.shape}"))
    if measurement is not None and (line := _prefix_block_line(measurement)):
        blocks.append(detail("prefix_block", line))
    if goal in RELEVANT:
        named = goal_bottleneck(diagnosis, goal)
        if diagnosis.primary is not None and diagnosis.primary not in RELEVANT[goal]:
            blocks[0] = Text(
                key="headline",
                text=f"{diagnosis_headline(diagnosis)}, for the run overall; it does not hold "
                f"back {GOAL_WORDS[goal]} (see the line above)",
            )
        text = (
            f"For {GOAL_WORDS[goal]}: {name_of(named)} ({by_class[named].evidence})"
            if named in RELEVANT[goal]
            else f"For {GOAL_WORDS[goal]}: no single limit was measured as binding (checked: "
            f"{', '.join(name_of(n) for n in RELEVANT[goal])})"
        )
        m = measurement
        if named not in RELEVANT[goal] and goal == "ttft" and m and m.ttft_p95_ms and m.step_ms:
            text += (
                f"; TTFT p95 {m.ttft_p95_ms:.0f} ms is about {m.ttft_p95_ms / m.step_ms:.1f} "
                f"decode steps of {m.step_ms:.0f} ms, so first tokens wait behind whole steps"
            )
        blocks.insert(0, Text(key="goal_headline", text=text))
    return Section(
        id="diagnosis", title="Details: diagnosis", blocks=tuple(blocks), title_audience="markdown"
    )


NO_PREFIX_HITS = 0.01  # a prefix-cache hit rate under which the cache found nothing


def _prefix_block_line(m: Measurement) -> str | None:
    """Why a hybrid model's prefix cache found nothing: it reuses whole blocks only, and no
    prompt filled one."""
    block, longest = m.kv_block_tokens, m.max_prompt_tokens
    if not (m.hybrid_cache and block and longest) or longest >= block:
        return None
    if m.prefix_cache_hit_rate is None or m.prefix_cache_hit_rate >= NO_PREFIX_HITS:
        return None
    return (
        f"prefix cache: {m.prefix_cache_hit_rate:.0%} hits, because this hybrid model reuses "
        f"only whole cache blocks of {block:.0f} tokens and the longest prompt has {longest} "
        f"tokens; a shared system prompt longer than {block:.0f} tokens would let requests "
        "reuse it"
    )


def _see_more_kv_memory(suggestions: Sequence[Suggestion]) -> str:
    for suggestion in suggestions:
        if suggestion.entry.name == "more-kv-memory":
            above = suggestion.tier == "try_first"
            return "; see `more-kv-memory`" + ("" if above else " under Could help")
    return ""


def _names(suggestions: Sequence[Suggestion]) -> str:
    return _names_of([s.entry.name for s in suggestions])


def _names_of(names: Sequence[str]) -> str:
    return ", ".join(f"`{n}`" for n in names)


def _suggestion_block(
    suggested: Suggested,
    suggestion: Suggestion,
    *,
    retained: bool,
    top: bool = False,
    kb: Mapping[str, Any] | None = None,
    direction: str = "",
) -> SuggestionBlock:
    changing = next(
        (m for m in suggestion.members or (suggestion.entry,) if m.changes_answers), None
    )
    return SuggestionBlock(
        suggestion=suggestion,
        risk=answer_warning(changing, retained=retained) if changing else "",
        expected=None if suggestion.members else suggested.estimate(suggestion),
        test=(
            suggested.test_times.get(suggestion.entry.name, suggested.test_time)
            if suggestion.command
            else None
        ),
        plan=suggested.plan_lines if top else (),
        tools=suggested.tool_calls,
        kb=kb if kb is not None else suggested.kb.get(suggestion.entry.name),
        direction=direction,
        test_seconds=(
            suggested.test_seconds.get(suggestion.entry.name) if suggestion.command else None
        ),
        start_seconds=(
            suggested.start_seconds.get(suggestion.entry.name) if suggestion.command else None
        ),
        change=change_words(suggestion.command, _held_arguments()),
    )


def _held_arguments() -> frozenset[str]:
    """The engine arguments the setup already has: the kept ones and those a pinned placement
    adds to every offer."""
    context = analysis()
    kept = context.kept
    tried = context.engine_args if context.try_step else ()
    return frozenset(
        (*context.held, *tried, *(kept.engine_args + kept.env if kept else ()), *_env_args())
    )


def next_steps_section(diagnosis: Diagnosis, suggested: Suggested, *, retained: bool) -> Section:
    """The playbook entries worth trying. "Try first": the suggestions of that tier, grouped by
    the bottleneck they address in the order of :func:`tensward.suggest.try_first_order`, at
    most three in full each. "Could help": the others, at most MAX_COULD_HELP in full, then the
    changes left out because they worsen the goal. A load-limited run's Try first asks for
    more load instead of a server change. "So far:" opens it with what the project's earlier
    runs measured; a revert after a loss leads Try first; with no change in Try first,
    ``suggested.stop`` says why and whether more is possible. ``suggested`` is ranked, with
    final tiers and, per bottleneck, why its change is blocked
    (:func:`tensward.suggest.suggestions`).
    ``retained`` says the run kept its answers, so the follow-up run compares them."""
    suggestions, blocked = suggested.suggestions, suggested.blocked
    top = first_try(diagnosis, suggested)

    def step(suggestion: Suggestion) -> SuggestionBlock:
        return _suggestion_block(suggested, suggestion, retained=retained, top=suggestion is top)

    blocks: list[Block] = []
    if suggested.ttft_basis is not None:
        blocks.append(Text(key="ttft_basis", text=suggested.ttft_basis))
    memory = suggested.memory
    if memory is not None and memory.lines:
        blocks.append(Text(key="so_far", text="So far:"))
        blocks += [item("so_far.line", line) for line in memory.lines]
    revert = memory.revert if memory is not None else None
    if revert is not None:
        blocks += [Text(key="try_first", text="Try first:"), step(revert)]
    order = try_first_order(diagnosis, suggested)
    nothing = not suggestions and not order and revert is None and suggested.underloaded is None
    if nothing and _unjudged(diagnosis):
        text = (
            "No change suggested: this run did not measure enough to judge (see the checks above)."
        )
        blocks.append(Text(key="nothing", text=text))
    elif nothing and suggested.stop:
        blocks.append(Text(key="nothing", text=suggested.stop))
    shown: set[str] = set()
    body: list[Block] = []
    for bottleneck in order:
        applying = group_members(diagnosis, suggested, bottleneck)
        group = [s for s in applying if s.entry.name not in shown]
        tried = memory is not None and bottleneck in memory.tried_for
        if not applying and not blocked.get(bottleneck) and suggestions and not tried:
            continue  # nothing to say here; the suggestions are under Could help
        if bottleneck == "power_cap" and not applying and not tried:
            continue  # no flag lifts it; Beyond flags says what does, when it knows
        if (
            bottleneck == "gpu_saturated" and suggested.goal != "throughput"
            and not applying and not tried and suggested.stop
        ):  # fmt: skip
            continue  # a latency goal's stop says what a saturated GPU leaves for it
        title = GOAL_WORDS.get(bottleneck) or name_of(cast(Bottleneck, bottleneck))
        body.append(Text(key=f"for.{bottleneck}", text=f"For {title}:"))
        if applying and not group:
            body.append(item("see_above", f"see {_names(applying)} above"))
            continue
        if not group:
            line = blocked.get(bottleneck)
            if tried:
                body.append(item("tried", "tried — see So far"))
            elif line:
                body.append(item("blocked", f"{line}{_see_more_kv_memory(suggestions)}"))
            else:
                text = (
                    "held back until you go back, as the step above says"
                    if revert is not None
                    else "no change in this engine's playbook applies here"
                )
                body.append(item("no_change", text))
            continue
        # a change that can affect the answers is always shown with its warning
        steps = [*group[:MAX_STEPS_PER_CLASS],
                 *(s for s in group[MAX_STEPS_PER_CLASS:] if s.entry.changes_answers)]  # fmt: skip
        body += [step(suggestion) for suggestion in steps]
        if rest := [s for s in group if s not in steps]:
            body.append(item("also", f"also: {_names(rest)}"))
        shown.update(s.entry.name for s in group)
    if suggested.underloaded is not None:
        body.insert(
            0,
            item(
                "load_limited",
                f"The test kept about {suggested.underloaded:.0f} requests in flight and the "
                "server ran them all with room to spare: the GPU is not saturated at your "
                "concurrency, so the server has headroom at this load. The changes below act on "
                "the server; change workload.arrival.concurrency only if real traffic is higher.",
            ),
        )
    if body and revert is None:
        blocks.append(Text(key="try_first", text="Try first:"))
    blocks += body
    others = [s for s in suggestions if s.entry.name not in shown and s.side is None]
    sides = [s for s in suggestions if s.side is not None]
    if sides:
        blocks.append(Text(key="side", text=side_heading(suggested.goal)))
        full = [step(suggestion) for suggestion in sides]
        blocks += [
            Text(
                key=f"side.{block.suggestion.entry.name}",
                text=f"  {at}. {block.change or block.suggestion.entry.name}: "
                + side_label(cast(str, block.suggestion.side), suggested.goal),
            )
            for at, block in enumerate(full, 1)
        ]
        blocks += [replace(block, audience="markdown") for block in full]
    if suggested.stop and (suggestions or order):
        blocks.append(Text(key="stop", text=suggested.stop))
        if others:
            blocks.append(Text(key="stop.gap", text=""))
    if others:
        blocks.append(Text(key="could_help", text="Could help:"))
        shown_in_full = [s for at, s in enumerate(others) if at < MAX_COULD_HELP or s.cost]
        blocks += [step(suggestion) for suggestion in shown_in_full]
        if rest := [s for s in others if s not in shown_in_full]:
            blocks.append(item("also", f"also: {_names(rest)}"))
    if suggested.other_goals:
        blocks.append(
            item(
                "left_out",
                f"left out because they worsen {GOAL_WORDS[suggested.goal]}: "
                f"{_names_of(suggested.other_goals)}",
            )
        )
        blocks += [
            item(f"left_out.{name}", f"`{name}` is left out: {why}")
            for name, why in suggested.left_out_why.items()
        ]
    if suggested.not_applicable:
        blocks.append(Text(key="not_applicable", text="Not applicable here:"))
        blocks += [
            item(f"not_applicable.{name}", f"`{name}`: {why}")
            for name, why in suggested.not_applicable
        ]
    if any(s.command for s in suggestions):
        better = (
            f"it helps without making {GOAL_WORDS[suggested.goal]} worse"
            if sides
            else "it is better"
        )
        blocks += [
            Text(key="gap", text=""),
            Text(
                key="serve_hint",
                text=f"`tensward try N` tests change N and keeps it only if {better} and "
                "the answers hold; `tensward serve` then serves what is kept.",
            ),
        ]
    return Section(id="next_steps", title="What to try next", blocks=tuple(blocks))


def _phase_one_ended(suggested: Suggested, keep: Section | None) -> bool:
    """Whether the project chain has no change left for the goal: Try first is empty, or the
    step this run measured is not recommended."""
    memory = suggested.memory
    if memory is not None and memory.revert is not None:
        return False
    none_left = not any(s.tier == "try_first" for s in suggested.suggestions)
    first = keep.blocks[0] if keep is not None and keep.blocks else None
    refused = (
        memory is not None and memory.parent is not None
        and isinstance(first, Text) and first.key == "not_kept"
    )  # fmt: skip
    return none_left or refused


def phase_two_section(suggested: Suggested, *, retained: bool) -> Section | None:
    """The changes for the goal's other metrics that leave the goal alone, each with the line
    that says when Tensward keeps it."""
    if not suggested.phase_two:
        return None
    blocks: list[Block] = []
    for suggestion in suggested.phase_two:
        blocks.append(_suggestion_block(suggested, suggestion, retained=retained))
        region = suggested.kb.get(suggestion.entry.name, {}).get("region")
        guard = phase_two_guard(suggested.goal, suggestion, region)
        blocks.append(item(f"phase_two.{suggestion.entry.name}", guard))
    title = f"Then, without hurting {GOAL_WORDS[suggested.goal]}"
    return Section(id="phase_two", title=title, blocks=tuple(blocks))


def _direction(entry: Entry) -> str:
    """What a documented record expects of its change: the record's prior, else each metric's
    direction with the reason."""
    if entry.prior:
        return entry.prior
    return "; ".join(
        f"{METRIC_WORDS.get(metric, metric)} {effect.direction}: {effect.reason}"
        for metric, effect in entry.effects.items()
    )


def worth_measuring_section(suggested: Suggested, *, retained: bool) -> Section | None:
    """The documented records that fired: not offers, but changes the knowledge says are worth
    a measurement, each with its command, the direction it expects, the answer risk and the
    line that says no run of Tensward measured it yet."""
    matches = analysis().kb_matches
    if matches is None:
        return None
    blocks: list[Block] = []
    for suggestion in suggested.worth_measuring:
        name = suggestion.entry.name
        found = next((m for m in matches.worth_measuring if m.merge_key == name), None)
        found = found or matches.by_entry(name)
        blocks.append(
            _suggestion_block(
                suggested,
                suggestion,
                retained=retained,
                kb=kb_block(found, matches) if found is not None else None,
                direction=_direction(suggestion.entry),
            )  # fmt: skip
        )
    if not blocks:
        return None
    return Section(id="worth_measuring", title="Worth measuring", blocks=tuple(blocks))
