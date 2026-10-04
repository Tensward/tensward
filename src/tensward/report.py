"""The markdown report of one run: its headline numbers, checks, diagnosis and next steps."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from .ceilings import Ceilings, render_ceilings
from .classify import NAMES, NOT_MODELLED, Diagnosis
from .engines import Engine
from .engines.protocol import SPECULATION_SIGNALS, Settings
from .measurement import GroupStats, ImageSplit, Measurement
from .playbook import Entry, WorkloadFacts, fitting_max_context_len
from .project import CurrentSetup
from .slo import Slo
from .thresholds import MIN_CONFIDENT_REQUESTS

MAX_FAILURE_REASONS = 3
MAX_EXAMPLE_PROMPTS = 5


@dataclass(frozen=True, slots=True)
class Subject:
    """What a run measured, for the top of its report: a title and, for the customer's own
    setup, the record of where it came from."""

    title: str = "Configuration"
    setup: CurrentSetup | None = None


FEEDBACK_LINE = (
    "Found something worth sharing, or a suggestion that was wrong? Tell us: "
    "https://github.com/Tensward/tensward/issues"
)


def overrides_suffix(engine_args: Sequence[str]) -> str:
    """ " + overrides (a=1, b=2)" when engine flags changed the setup, else nothing."""
    return f" + overrides ({', '.join(engine_args)})" if engine_args else ""


def _percent(fraction: float | None) -> float | None:
    return None if fraction is None else fraction * 100


def _line(label: str, value: float | None, unit: str, digits: int = 1) -> str:
    return f"- {label}: " + ("not measured" if value is None else f"{value:.{digits}f}{unit}")


def _group_line(label: str, group: GroupStats | None) -> str:
    if group is None:
        return f"- {label}: no requests"

    def figure(value: float | None, unit: str) -> str:
        return "not measured" if value is None else f"{value:.0f}{unit}"

    return (
        f"- {label}: {group.requests} requests, TTFT p50 {figure(group.ttft_p50_ms, ' ms')}, "
        f"TTFT p95 {figure(group.ttft_p95_ms, ' ms')}, "
        f"TPOT p95 {figure(group.tpot_p95_ms, ' ms')}, "
        f"mean prompt tokens {figure(group.prompt_tokens_mean, '')}"
    )


def _image_split_lines(split: ImageSplit | None) -> list[str]:
    if split is None:
        return []
    return [
        "",
        "## Requests with and without images",
        "",
        _group_line("with images", split.with_images),
        _group_line("without images", split.without_images),
        f"- images per request that carries images: {split.images_per_request:.1f}",
        "- the prompt tokens of a request with images include the image tokens the engine counted",
    ]


def _capacity_line(measurement: Measurement) -> str:
    """The KV capacity the engine reports, beside what Tensward estimated before the run."""
    measured, estimate = measurement.kv_capacity_tokens, measurement.kv_capacity_estimate_tokens
    if measured is None:
        return "- KV cache capacity (measured by the engine): not measured"
    line = f"- KV cache capacity (measured by the engine): {measured:.0f} tokens"
    if measurement.kv_max_concurrency is not None:
        line += f", {measurement.kv_max_concurrency:.1f} full-length requests at once"
    if estimate is None:
        return line + "; no estimate was made before the run"
    error = estimate / measured - 1
    return f"{line}; estimated before the run: {estimate:.0f} tokens (error {error:+.0%})"


def render_markdown(
    run_id: str,
    measurement: Measurement,
    rows: Sequence[Mapping[str, Any]],
    facts: WorkloadFacts,
    settings: Settings,
    subject: Subject,
    image: str | None,
    slo: Slo,
    checks: Sequence[str],
    defaults: Mapping[str, str],
    added_flags: str,
    ran: str,
) -> str:
    """A short human summary of the requests.jsonl rows. Anything unmeasured says so."""
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["outcome"]] += 1
    lines = [f"# Tensward analysis {run_id}", "", f"Ran: {ran}", ""]
    lines += render_subject(subject, settings, measurement, image)
    lines += ["## Requests", ""]
    breakdown = ", ".join(f"{name} {count}" for name, count in sorted(counts.items()))
    lines.append(
        f"- {len(rows)} requests: {measurement.succeeded} succeeded, {measurement.failed} failed"
    )
    if breakdown:
        lines.append(f"- outcomes: {breakdown}")
    distinct = len(set(facts.prompts))
    if len(rows) > distinct:
        lines.append(
            f"- NOTE: {len(rows)} requests from {distinct} distinct prompts: prefix-cache hit "
            "rate and similar signals are inflated by repetition; use more distinct prompts "
            "for realistic numbers"
        )
    lines += _failure_lines([row for row in rows if row["outcome"] != "success"])
    lines += ["", "## Performance (client side)", ""]
    lines += [
        _line("request throughput", measurement.request_throughput, " req/s", 2),
        _line("output throughput", measurement.output_throughput, " tokens/s"),
        _line("total throughput (prompt + output)", measurement.total_throughput, " tokens/s"),
        _line(
            f"goodput (TTFT <= {slo.ttft_ms:g} ms and TPOT <= {slo.tpot_ms:g} ms per request)",
            measurement.goodput,
            " req/s",
            2,
        ),
        _line("requests meeting the SLO", _percent(measurement.slo_attainment), "%", 0),
        _line("TTFT p50", measurement.ttft_p50_ms, " ms"),
        _line("TTFT p95", measurement.ttft_p95_ms, " ms"),
        _line("TPOT p50", measurement.tpot_p50_ms, " ms"),
        _line("TPOT p95", measurement.tpot_p95_ms, " ms"),
        _line("end-to-end p95", measurement.e2e_p95_ms, " ms"),
    ]
    lines += _image_split_lines(measurement.image_split)
    lines += ["", "## Engine signals (from the engine's metrics)", ""]
    lines += [
        _line(
            "highest sampled KV-cache usage (1 s polls)", _percent(measurement.peak_kv_usage), "%"
        ),
        _line("highest sampled running requests (1 s polls)", measurement.peak_running, "", 0),
        _line("highest sampled waiting requests (1 s polls)", measurement.peak_waiting, "", 0),
        _line("preemptions", measurement.preemptions, "", 0),
        _line("prefix-cache hits", measurement.prefix_cache_hits, " tokens", 0),
        _line("prefix-cache hit rate", _percent(measurement.prefix_cache_hit_rate), "%"),
        _capacity_line(measurement),
    ]
    lines += ["", f"Tensward adds to every launch: {added_flags}"]
    lines += _default_lines(settings, defaults)
    lines += render_ceilings(measurement.ceilings)
    lines += _quantization_lines(measurement, facts)
    lines += _tool_call_lines(measurement)
    lines += _context_lines(measurement, facts)
    if checks:
        lines += ["", "## Checks", "", *(f"- {check}" for check in checks)]
    return "\n".join(lines)


def render_subject(
    subject: Subject, settings: Settings, measurement: Measurement, image: str | None
) -> list[str]:
    """The block that opens a report: what was measured and its headline numbers."""
    lines = [f"## {subject.title}", ""]
    if (setup := subject.setup) is not None:
        lines.append(f"- source: {setup.label}")
        if setup.image and image and setup.image != image:
            lines.append(
                f"- **version differs: your setup runs {setup.image}, measured on {image}**"
            )
    lines.append(f"- settings: {settings.summary()}")
    lines += [f"- note: {note}" for note in subject.setup.notes] if subject.setup else []
    lines.append(f"- measured: {', '.join(_headline_parts(measurement))}")
    if setup is not None and measurement.failed:
        lines.append(
            "- **your current setup fails requests on this workload; fixing that is the first "
            "improvement**"
        )
    return [*lines, ""]


def _headline_parts(measurement: Measurement) -> list[str]:
    total = measurement.succeeded + measurement.failed
    return [
        _metric("output", measurement.output_throughput, " tok/s"),
        _metric("total", measurement.total_throughput, " tok/s"),
        _metric("requests", measurement.request_throughput, " req/s", 2),
        _metric("goodput", measurement.goodput, " req/s", 2),
        _metric("TTFT p95", measurement.ttft_p95_ms, " ms", 0),
        _metric("TPOT p95", measurement.tpot_p95_ms, " ms"),
        f"{measurement.failed} of {total} requests failed",
        _ceiling_metric(measurement.ceilings),
    ]


def render_headline(measurement: Measurement, source: str, engine_args: Sequence[str]) -> list[str]:
    """The short terminal summary of a run: the headline numbers, one per line."""
    lines = [
        f"current setup{overrides_suffix(engine_args)}: {source}",
        *(f"  {part}" for part in _headline_parts(measurement)),
    ]
    trace = measurement.trace
    if trace is not None and trace.status != "unavailable" and trace.idle_share is not None:
        lines.append(
            f"  GPU busy {trace.gpu_busy_share or 0:.1%}, idle {trace.idle_share:.1%} (profiled)"
        )
    return lines


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


def _metric(label: str, value: float | None, unit: str, digits: int = 1) -> str:
    return f"{label} " + ("not measured" if value is None else f"{value:.{digits}f}{unit}")


def _quantization_lines(measurement: Measurement, facts: WorkloadFacts) -> list[str]:
    """The checkpoint's quantization and the kernels the server log says were selected."""
    if facts.quantization is None:
        return []
    kernels = ", ".join(f"{k.name} ({k.layer})" for k in measurement.quant_kernels)
    slow = " - a slower path" if any(k.slow for k in measurement.quant_kernels) else ""
    return [
        "",
        "## Quantization",
        "",
        f"- checkpoint: {facts.quantization}",
        f"- kernel: {kernels + slow if kernels else 'not detected in the server log'}",
    ]


def checks(
    engine: Engine,
    metrics_after: str | None,
    facts: WorkloadFacts,
    settings: Settings,
    peak_running: float | None,
) -> list[str]:
    """Problems that make this report less trustworthy or the workload unservable."""
    checks = []
    if (blocker := _tool_calling_blocker(facts, settings)) is not None:
        checks.append(f"tool calling: {blocker}")
    if metrics_after is None:
        checks.append("the engine's metrics could not be read, so engine signals are not measured")
    elif missing := [
        name
        for name, value in asdict(engine.parse_signals(metrics_after)).items()
        if value is None and name not in SPECULATION_SIGNALS
    ]:
        checks.append(
            f"the engine's metrics do not expose {', '.join(missing)}: this engine version may "
            f"have renamed its metrics, so those signals are not measured"
        )
    graph_batch = engine.max_graph_batch(settings)
    if peak_running is not None and graph_batch is not None and peak_running > graph_batch:
        checks.append(
            f"running sequences reached {peak_running:g} but CUDA graphs cover only "
            f"{graph_batch}; larger batches run without graphs"
        )
    return checks


def _tool_calling_blocker(facts: WorkloadFacts, settings: Settings) -> str | None:
    """Why the workload's tools cannot be served, with the fix; None when they can."""
    if not facts.offers_tools or settings.serves_tools:
        return None
    if settings.tool_parser is None:
        return (
            "the workload offers tools but no tool-call parser is known for this checkpoint's "
            "model family, so the server cannot return tool calls; name the parser with "
            "`--engine-arg tool-call-parser=<name> --engine-arg enable-auto-tool-choice`"
        )
    return (
        "the workload offers tools but tool calling is not enabled, so the server refuses "
        "those requests; set `tool_calling: true` in the serving configuration's case "
        f"(parser {settings.tool_parser}) or add `--engine-arg enable-auto-tool-choice`"
    )


def _default_lines(settings: Settings, defaults: Mapping[str, str]) -> list[str]:
    """What the engine does for each setting this setup leaves unset, where that is known."""
    unset = {
        n: text for n, text in defaults.items() if getattr(settings, n) == getattr(Settings(), n)
    }
    if not unset:
        return []
    return [
        "",
        "## Engine defaults in effect (settings this setup leaves unset)",
        "",
        *(f"- {name}: {text}" for name, text in unset.items()),
    ]


def _tool_call_lines(measurement: Measurement) -> list[str]:
    """Whether the model called tools correctly. Quality observations, not performance."""
    stats = measurement.tool_calls
    if stats is None:
        return []

    def share(value: float | None) -> str:
        return "no tool call to judge" if value is None else f"{value:.0%}"

    return [
        "",
        "## Tool calling (quality, not performance)",
        "",
        f"- requests that offered tools: {stats.requests}",
        f"- produced a tool call: {share(stats.produced_call)}",
        f"- of the {stats.calling} that did, arguments parse as JSON: {share(stats.valid_json)}",
        f"- of the {stats.calling} that did, tool is one of the offered tools: "
        f"{share(stats.known_tool)}",
        f"- of the {stats.calling} that did, arguments satisfy the tool's required keys and "
        f"property types: {share(stats.schema_valid)}",
        "- caveat: counts every request that offered tools; prompts that should not call a tool "
        "count as misses",
    ]


def _context_lines(measurement: Measurement, facts: WorkloadFacts) -> list[str]:
    """Name the prompts that cannot fit the context window, and whether raising it can help."""
    if not measurement.too_long or measurement.max_prompt_tokens is None:
        return []
    longest = measurement.max_prompt_tokens + facts.output_tokens
    lines = [
        "",
        "## Prompts that do not fit the context window",
        "",
        f"- {len(measurement.too_long)} prompts need more than max_context_len "
        f"{measurement.max_context_len} tokens (prompt plus {facts.output_tokens} output tokens): "
        + ", ".join(measurement.too_long[:MAX_EXAMPLE_PROMPTS])
        + (", ..." if len(measurement.too_long) > MAX_EXAMPLE_PROMPTS else ""),
        f"- the longest needs {longest} tokens",
    ]
    if fitting_max_context_len(measurement, facts) is None:
        lines.append(
            f"- the model supports at most {facts.context_limit} tokens, so raising "
            "max_context_len cannot fix this; shorten those prompts or lower the output tokens"
        )
    return lines


def _reason(row: Mapping[str, Any], *, mask: bool) -> str:
    if row.get("http_status") is None:
        return row.get("error") or f"no HTTP response ({row['outcome']})"
    message = row["error"] or ""
    if mask:  # numbers differ between requests of the same cause (token counts)
        message = re.sub(r"\d+", "N", message)
    return f"HTTP {row['http_status']}: {message}"


def _failure_lines(failed: Sequence[Mapping[str, Any]]) -> list[str]:
    """The most common reasons requests failed, with counts and example prompt ids."""
    if not failed:
        return []
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in failed:
        groups[_reason(row, mask=True)].append(row)
    lines = ["", "### Why requests failed", ""]
    for rows in sorted(groups.values(), key=len, reverse=True)[:MAX_FAILURE_REASONS]:
        prompts = list(dict.fromkeys(row["prompt_id"] for row in rows))[:MAX_EXAMPLE_PROMPTS]
        lines.append(
            f"- {len(rows)} x {_reason(rows[0], mask=False)} (prompts: {', '.join(prompts)})"
        )
    return lines


MAX_STEPS_PER_CLASS = 3


def diagnosis_headline(diagnosis: Diagnosis) -> str:
    few = diagnosis.requests < MIN_CONFIDENT_REQUESTS
    count = f"only {diagnosis.requests} requests succeeded"
    if diagnosis.requests == 0:
        return "Bottleneck: none named, because no request succeeded"
    if all(f.state == "cant_tell" for f in diagnosis.findings if f.bottleneck != "speculation"):
        return "Bottleneck: not enough was measured to name one"
    if diagnosis.primary is None:
        decided = [f for f in diagnosis.findings if f.threshold]
        notes = ["no signal crossed its threshold"]
    else:
        decided = [f for f in diagnosis.findings if f.bottleneck == diagnosis.primary]
        notes = [f"confidence: {diagnosis.confidence}"]
    if not all(f.calibrated for f in decided):
        notes.append("thresholds not yet calibrated on real GPUs")
    notes += [count] if few else []
    name = NAMES[diagnosis.primary] if diagnosis.primary else "none clear"
    return f"Bottleneck: {name} ({'; '.join(notes)})"


def render_diagnosis(diagnosis: Diagnosis) -> str:
    """The bottleneck and its evidence, what else crossed, which classes did not cross and what
    could not be told, then the gates and the workload shape."""
    by_class = {finding.bottleneck: finding for finding in diagnosis.findings}
    lines = ["## Diagnosis", "", diagnosis_headline(diagnosis)]
    if diagnosis.primary:
        lines.append(f"- evidence: {by_class[diagnosis.primary].evidence}")
    elif diagnosis.requests:
        nearest = sorted(
            (f for f in diagnosis.findings if f.state == "clear" and f.threshold),
            key=lambda f: f.margin,
            reverse=True,
        )[:2]
        lines += [f"- nearest: {NAMES[f.bottleneck]}: {f.evidence}" for f in nearest]
    for name in diagnosis.secondary:
        finding = by_class[name]
        lines.append(f"- also seen: {NAMES[name]} ({finding.confidence}): {finding.evidence}")
    decided = [
        f for f in diagnosis.findings if f.state == "clear" and f.bottleneck != "speculation"
    ]
    speculation = by_class["speculation"]
    if speculation.state == "clear" and speculation.threshold:
        decided.append(speculation)
    if decided and diagnosis.requests:
        calibrated = all(f.calibrated for f in decided if f.threshold)
        label = "ruled out" if calibrated else "not crossed (uncalibrated thresholds)"
        lines.append(f"- {label}: {', '.join(NAMES[f.bottleneck] for f in decided)}")
    if speculation.threshold is None:
        lines.append("- speculation: off or not reported")
    if unknown := [f for f in diagnosis.findings if f.state == "cant_tell"]:
        lines.append("- can't tell here:")
        lines += [f"  - {NAMES[f.bottleneck]} — {f.evidence}" for f in unknown]
    lines.append(
        "- not modelled: " + "; ".join(f"{NAMES[k]} ({why})" for k, why in NOT_MODELLED.items())
    )
    lines += [f"- {NAMES[gate.bottleneck]}: {gate.evidence}" for gate in diagnosis.gates]
    if diagnosis.shape:
        lines.append(f"- workload shape: {diagnosis.shape}")
    return "\n".join(lines) + "\n"


def _step_lines(
    entry: Entry, reason: str, command_for: Callable[[Entry], str | None], risk: str
) -> list[str]:
    lines = [
        f"- `{entry.name}`{risk if entry.quality_risk else ''}: {reason}",
        f"  evidence: {entry.evidence}; may cost: {entry.costs}",
    ]
    if command := command_for(entry):
        lines.append(f"  Try: `{command}`")
    return lines


def render_next_steps(
    diagnosis: Diagnosis,
    suggestions: Sequence[tuple[Entry, str]],
    command_for: Callable[[Entry], str | None],
    not_applicable: Sequence[tuple[str, str]] = (),
    *,
    retained: bool,
) -> str:
    """The playbook entries worth trying, grouped by the bottleneck they address: failed gates
    first, then the primary bottleneck, then the secondary ones, at most three in full each;
    then the changes the measurements support for no diagnosed class. ``suggestions`` is
    ranked. ``command_for`` gives the ``tensward analyse ...`` command of an entry, or None.
    ``retained`` says the run kept its answers, so the follow-up run compares them."""
    risk = (
        " (may change the outputs: the report then compares its answers with your current setup's)"
        if retained
        else " (may change the outputs: compare them before adopting it)"
    )
    lines = ["## What to try next", ""]
    addressed = {name for entry, _ in suggestions for name in entry.addresses}
    order = [
        gate.bottleneck
        for gate in diagnosis.gates
        if gate.state == "critical" and gate.bottleneck in addressed
    ]
    order += [*([diagnosis.primary] if diagnosis.primary else []), *diagnosis.secondary]
    if not suggestions and not order:
        lines.append("No change is suggested by the measured signals.")
    shown: set[str] = set()
    for bottleneck in order:
        applying = [(e, r) for e, r in suggestions if bottleneck in e.addresses]
        group = [(e, r) for e, r in applying if e.name not in shown]
        lines.append(f"For {NAMES[bottleneck]}:")
        if applying and not group:
            lines.append(f"- see {', '.join(f'`{entry.name}`' for entry, _ in applying)} above")
            continue
        if not group:
            lines.append("- no change in this engine's playbook applies here")
            continue
        for entry, reason in group[:MAX_STEPS_PER_CLASS]:
            lines += _step_lines(entry, reason, command_for, risk)
        if rest := group[MAX_STEPS_PER_CLASS:]:
            lines.append(f"- also: {', '.join(f'`{entry.name}`' for entry, _ in rest)}")
        shown.update(entry.name for entry, _ in group)
    if others := [(e, r) for e, r in suggestions if e.name not in shown]:
        lines.append("Other changes the measurements support:")
        for entry, reason in others:
            lines += _step_lines(entry, reason, command_for, risk)
    if not_applicable:
        lines.append("Not applicable here:")
        lines += [f"- `{name}`: {why}" for name, why in not_applicable]
    if any(command_for(entry) for entry, _ in suggestions):
        lines += [
            "",
            "To serve a change, give `tensward serve start` the same `--project` and "
            "`--engine-arg` options.",
        ]
    return "\n".join(lines) + "\n"


def put_first(report: str, sections: str) -> str:
    """``report`` with ``sections`` inserted before its first ``## `` heading."""
    at = report.index("\n## ") + 1
    return report[:at] + sections + "\n" + report[at:]
