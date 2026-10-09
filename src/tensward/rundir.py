"""One run directory: the raw facts a run captured and the files they are written to. Each
file is written once: the evidence as the launch stops, the derived files when the run is
summarized."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capabilities import Resolved
from .capacity import CapacityResult
from .capture import CapturedResponse
from .classify import NAMES, Diagnosis
from .client import RequestRecord
from .estimate import FREE_TOOL_CHOICES, RunInputs, block_hashes
from .files import write_json, write_jsonl
from .inputs import PromptEntry
from .measurement import Measurement, Peaks, Window
from .project import ResolvedProject
from .report.build import RunOutputs, build_report
from .report.markdown import to_markdown
from .report.model import Report, to_json
from .runs import RUN_RECORD, RunFile
from .signal_source import Scrape
from .thresholds import THRESHOLDS


@dataclass(frozen=True, slots=True, kw_only=True)
class RunCapture:
    """The raw facts of one workload run against a live server. Nothing here is derived."""

    run_dir: Path
    project: ResolvedProject  # what this run offered
    records: Sequence[RequestRecord]
    lead_records: Sequence[RequestRecord]  # the start-up wave, outside the window
    responses: Mapping[str, CapturedResponse]
    prompt_token_counts: Mapping[str, int]  # by prompt id, from the server's tokenizer
    window: Window  # what the client's throughput covers
    window_ns: int  # when the engine's measurement window opened
    after_ns: int  # when the scrape that closed the engine's window began
    before: Scrape | None
    after: Scrape | None  # the engine as the window closed
    final: Scrape | None  # the engine after the run, for what only it shows
    fell_back: bool  # the engine could not be read at the last request, so ``after`` is final
    outlasted: bool  # the start-up wave was still running at the last request
    peaks: Peaks
    server_log: str
    log_start: str  # the server log when the window opened
    steady_state: bool
    image: str | None  # the container image that ran, if it was a container
    resolved: Resolved | None = None  # what the engine chose at launch
    # By prompt id, the ids the server's tokenizer gave the prompt as the engine renders it.
    prompt_token_ids: Mapping[str, Sequence[int]] = field(default_factory=dict)
    capacity: CapacityResult = field(
        default_factory=lambda: CapacityResult(None, "unknown-setup", None)
    )  # the engine's prompt capacity, probed before the workload


def prompt_ids(capture: RunCapture) -> dict[str, str]:
    """Map each request to the registered prompt it offered."""
    prompts = capture.project.prompts
    return {record.request_id: prompts[record.prompt_index].id for record in capture.records}


def estimate_inputs(
    capture: RunCapture,
    gpus: tuple[str, ...] | None,
    block_tokens: float | None,
    version: str | None = None,
) -> RunInputs:
    """What the estimates read from a run."""
    project = capture.project
    dispatched = sorted(capture.records, key=lambda record: record.dispatch_ns)
    return RunInputs(
        arrival=project.config.workload.arrival.kind,
        gpu_count=len(gpus) if gpus else 1,
        forced_tools=any(entry.tool_choice not in FREE_TOOL_CHOICES for entry in project.prompts),
        prompt_order=tuple(record.prompt_index for record in dispatched),
        engine_version=version,
        prompt_blocks=prompt_blocks(
            project.prompts,
            capture.prompt_token_ids,
            capture.records,
            capture.responses,
            block_tokens,
        ),
    )


def prompt_blocks(
    prompts: Sequence[PromptEntry],
    ids: Mapping[str, Sequence[int]],
    records: Sequence[RequestRecord],
    responses: Mapping[str, CapturedResponse],
    block_tokens: float | None,
) -> dict[int, tuple[int, tuple[str, ...]]]:
    """By prompt index, each prompt's token count and block hashes; empty unless every answered
    request's prompt token count equals its prompt's ids, so that the ids are what the engine
    computed (the tokenizer endpoint and the chat endpoint can render a prompt differently)."""
    if not ids or not block_tokens:
        return {}
    for record in records:
        captured = responses.get(record.request_id)
        sent = ids.get(prompts[record.prompt_index].id)
        if captured and captured.prompt_tokens and len(sent or ()) != captured.prompt_tokens:
            return {}
    return {
        index: (len(sent), block_hashes(sent, int(block_tokens)))
        for index, entry in enumerate(prompts)
        if (sent := ids.get(entry.id)) is not None
    }


def rendered(capture: RunCapture) -> tuple[tuple[str, ...], ...]:
    """Each registered prompt's chained block hashes as the engine rendered it, in the project's
    prompt order; empty for a prompt whose token ids the run does not have."""
    block_tokens = capture.final.signals.kv_block_tokens if capture.final else None
    blocks = prompt_blocks(
        capture.project.prompts,
        capture.prompt_token_ids,
        capture.records,
        capture.responses,
        block_tokens,
    )
    return tuple(
        blocks[index][1] if index in blocks else () for index in range(len(capture.project.prompts))
    )


def write_prompt_blocks(
    run_dir: Path, capture: RunCapture, run: RunInputs, block_tokens: float
) -> None:
    """``prompt_blocks.jsonl``: per prompt, its token count and the chained hashes of its full
    KV blocks, so the prefix-caching estimate can be recomputed from the run directory. The
    hashes cannot be turned back into the prompt's text."""
    prompts = capture.project.prompts
    write_jsonl(
        run_dir / "prompt_blocks.jsonl",
        [
            {
                "prompt_id": prompts[index].id,
                "tokens": tokens,
                "block_tokens": int(block_tokens),
                "blocks": list(blocks),
            }
            for index, (tokens, blocks) in sorted(run.prompt_blocks.items())
        ],
    )


def _failure_reason(captured: CapturedResponse | None, outcome: str) -> dict[str, Any]:
    if outcome == "success" or captured is None or captured.http_status is None:
        return {}
    return {"http_status": captured.http_status, "error": captured.error}


def request_prompt_tokens(capture: RunCapture) -> dict[str, int | None]:
    """Each request's prompt tokens: the server's usage count when it sent one, else the
    registered prompt's count."""
    ids = prompt_ids(capture)
    return {
        record.request_id: captured.prompt_tokens
        if (captured := capture.responses.get(record.request_id)) and captured.prompt_tokens
        else capture.prompt_token_counts.get(ids[record.request_id])
        for record in capture.records
    }


def _response_details(captured: CapturedResponse | None) -> dict[str, Any]:
    """The usage details the server sent beyond the token count, only those it sent."""
    if captured is None:
        return {}
    found: dict[str, Any] = {}
    if captured.cached_tokens is not None:
        found["cached_tokens"] = captured.cached_tokens
    if captured.timings is not None:
        found["timings"] = dict(captured.timings)
    return found


def request_rows(capture: RunCapture) -> list[dict[str, Any]]:
    """The rows of ``requests.jsonl``: each request's record, prompt, prompt tokens and failure
    reason."""
    ids = prompt_ids(capture)
    prompt_tokens = request_prompt_tokens(capture)
    return [
        {
            **asdict(record),
            "prompt_id": ids[record.request_id],
            "prompt_tokens": prompt_tokens[record.request_id],
            **_failure_reason(capture.responses.get(record.request_id), record.outcome),
            **_response_details(capture.responses.get(record.request_id)),
        }
        for record in capture.records
    ]


def write_evidence(run_dir: Path, capture: RunCapture, *, retain_responses: bool) -> None:
    """Write what the run captured: its requests, its answers when kept, and the scrapes."""
    write_jsonl(run_dir / "requests.jsonl", request_rows(capture))
    ids = prompt_ids(capture)
    if retain_responses:
        write_jsonl(
            run_dir / "responses.jsonl",
            [
                {
                    "request_id": record.request_id,
                    "prompt_id": ids[record.request_id],
                    "text": captured.text,
                    "tool_calls": [asdict(call) for call in captured.tool_calls],
                    "finish_reason": captured.finish_reason,
                    "truncated": captured.finish_reason == "length",
                }
                for record in capture.records
                for captured in [capture.responses.get(record.request_id, CapturedResponse())]
            ],
        )
    for name, scrape in (
        ("metrics_before.prom", capture.before),
        ("metrics_after.prom", capture.after),
    ):
        if scrape is not None:
            (run_dir / name).write_text(scrape.text, encoding="utf-8")
            for extra, text in scrape.extra.items():  # metrics_after.slots.json, for example
                (run_dir / f"{Path(name).stem}.{extra}").write_text(text, encoding="utf-8")


# Measurement fields written only when an engine sets them, so a run that sets none writes none.
LEFT_OUT_AT_DEFAULT: dict[str, Any] = {
    "engine_signals": {},
    "replicas": 1,
    "untimed_requests": 0,
    **dict.fromkeys(
        (
            "mean_waiting",
            "mean_kv_usage",
            "mean_in_flight",
            "prompt_capacity_tok_s",
            "prompt_work_share",
            "scheduled_tokens_per_step",
            "step_ms",
            "step_budget_tokens",
            "compile_cache",
            "server_ttft_p50_ms",
            "server_ttft_mean_ms",
            "ttft_mean_ms",
        )
    ),
}


def metrics_document(
    measurement: Measurement,
    diagnosis: Diagnosis,
    not_applicable: Sequence[tuple[str, str]],
    resolved: Resolved | None = None,
) -> dict[str, Any]:
    """The content of ``metrics.json``: the measurement with the diagnosis beside it, and what
    the engine chose at launch when it chose anything."""
    measured = asdict(measurement)
    for name, default in LEFT_OUT_AT_DEFAULT.items():
        if measured[name] == default:
            del measured[name]
    return {
        **measured,
        **({"resolved": asdict(resolved)} if resolved is not None else {}),
        "diagnosis": {
            **asdict(diagnosis),
            "names": NAMES,
            "thresholds": {t.name: asdict(t) for t in THRESHOLDS},
            "not_applicable": [{"entry": n, "why": w} for n, w in not_applicable],
        },
    }


def write_derived(run_dir: Path, outputs: RunOutputs, run_file: RunFile) -> Report:
    """Write run.json, metrics.json (with the diagnosis), report.json and report.md, each once;
    return the report. ``run.json`` comes last: a run directory without it is never compared
    with."""
    suggested = outputs.suggested
    not_applicable = suggested.not_applicable if suggested is not None else ()
    write_json(
        run_dir / "metrics.json",
        metrics_document(outputs.measurement, outputs.diagnosis, not_applicable, outputs.resolved),
    )
    report = build_report(outputs)
    write_json(run_dir / "report.json", to_json(report))
    (run_dir / "report.md").write_text(to_markdown(report), "utf-8")
    measurement = outputs.measurement
    built = {
        "step_budget_tokens": measurement.step_budget_tokens,
        "compile_cache": measurement.compile_cache,
        "prompt_capacity": asdict(outputs.capacity) if outputs.capacity is not None else None,
    }
    known = {name: value for name, value in built.items() if value is not None}
    write_json(run_dir / RUN_RECORD, {**run_file.model_dump(mode="json"), **known})
    return report
