"""Where llama-server exports the signals Tensward reads: its Prometheus families at ``/metrics``
(``--metrics``), the slots at ``/slots`` (``--slots``) and the ``timings`` of each final stream
event. Build 11669 exports no KV cache gauge, no queue time and no per-request prefill time."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from statistics import median

import httpx

from ...prometheus import Family, SignalMap, samples, values
from ...settings import EngineSignals
from ...signal_source import PrometheusSource, RunInputs, RunSignals, Scrape

LLAMACPP_SIGNALS: SignalMap = {
    "running": Family("llamacpp:requests_processing"),
    "waiting": Family("llamacpp:requests_deferred"),
    # llamacpp:prompt_tokens_total leaves out the tokens served from the prompt cache
    "prompt_tokens_computed": Family("llamacpp:prompt_tokens_total"),
    "generation_tokens": Family("llamacpp:tokens_predicted_total"),
    "iterations": Family("llamacpp:n_decode_total"),
    "spec_drafts": Family("llamacpp:spec_decode_num_drafts_total"),
    "spec_accepted_tokens": Family("llamacpp:spec_decode_num_accepted_tokens_total"),
}

KV_FULL_LINE = "failed to find free space in the KV cache"
CONTEXT_ERROR = "context size"
SLOTS_FILE = "slots.json"
# llama-server prints metrics with six significant digits, so above this a counter difference
# over a window is rounding noise.
PRECISION_LIMIT = 1e6


def slot_fill(text: str) -> float | None:
    """The fullest busy slot's share of its context, from ``GET /slots`` (None while no slot is
    busy: an idle sample says nothing about the cache): tokens held
    (``n_prompt_tokens``, which grows with the generated tokens) over the slot's ``n_ctx``. A
    lower bound on the pool's use when the slots share a unified pool."""
    try:
        slots = json.loads(text)
        busy = [s for s in slots if s["is_processing"]]
        return max((s["n_prompt_tokens"] / s["n_ctx"] for s in busy), default=None)
    except (ValueError, KeyError, TypeError, ZeroDivisionError):
        return None


def window_batch(before: str, after: str) -> float | None:
    """Busy slots per decode call over the window. The gauge is a running average since the
    server started, so the window's figure is the difference of busy-slot totals over the
    difference of decode calls."""
    totals = []
    for text in (before, after):
        found = values(samples(text))
        batch = found.get("llamacpp:n_busy_slots_per_decode")
        decodes = found.get("llamacpp:n_decode_total")
        if batch is None or decodes is None or decodes >= PRECISION_LIMIT:
            return None
        totals.append((batch * max(decodes, 1.0), decodes))
    steps = totals[1][1] - totals[0][1]
    return (totals[1][0] - totals[0][0]) / steps if steps > 0 else None


@dataclass(frozen=True, slots=True)
class LlamaCppSource(PrometheusSource):
    async def read(self, client: httpx.AsyncClient, server_url: str) -> Scrape | None:
        scrape = await PrometheusSource.read(self, client, server_url)
        if scrape is None:
            return None
        try:
            response = await client.get(f"{server_url}/slots")
            response.raise_for_status()
        except httpx.HTTPError:
            return scrape
        signals = replace(scrape.signals, kv_usage=slot_fill(response.text))
        return replace(scrape, signals=signals, extra={SLOTS_FILE: response.text})

    def from_run(self, run: RunInputs) -> RunSignals:
        prompt = cached = prefill = queued = 0.0
        timed = 0
        thinking: list[int] = []
        exceeded = sum(
            1
            for response in run.responses.values()
            if response.http_status == 400 and CONTEXT_ERROR in (response.error or "")
        )
        for record in run.records:
            response = run.responses.get(record.request_id)
            if response is None:
                continue
            if record.outcome != "success":
                continue
            if response.reasoning and not response.text and not response.tool_calls:
                thinking.append(record.committed_output_tokens or 0)
            found = response.timings
            if not found or record.first_content_ns is None or "prompt_ms" not in found:
                continue
            timed += 1
            prompt += found.get("prompt_n", 0.0)
            cached += found.get("cache_n", 0.0)
            prefill += found["prompt_ms"] / 1000
            first = (record.first_content_ns - record.dispatch_ns) / 1e9
            queued += max(first - found["prompt_ms"] / 1000, 0.0)
        neutral = EngineSignals()
        if timed:
            neutral = EngineSignals(
                prompt_tokens=prompt + cached,
                prefix_cache_hits=cached,
                prefix_cache_queries=prompt + cached,
                queue_seconds=queued,
                prefill_seconds=prefill,
            )
        engine = {
            "llamacpp.kv_full_lines": float(run.server_log.count(KV_FULL_LINE)),
            "llamacpp.context_exceeded": float(exceeded),
        }
        if thinking:
            engine["llamacpp.thinking_answers"] = float(len(thinking))
            engine["llamacpp.thinking_tokens_p50"] = float(median(thinking))
        batch = None
        if run.before is not None and run.after is not None:
            batch = window_batch(run.before.text, run.after.text)
        return RunSignals(
            neutral=neutral,
            engine=engine,
            window_batch=batch,
            checks=_thinking_checks(run, thinking),
        )


def _thinking_checks(run: RunInputs, thinking: list[int]) -> tuple[str, ...]:
    if not thinking:
        return ()
    if len(thinking) > 1 and min(thinking) == max(thinking):
        spent = f"hits the output cap ({thinking[0]} tokens) while thinking on"
    else:
        spent = f"spends a median of {median(thinking):.0f} output tokens thinking on"
    return (
        f"{run.model} {spent} "
        f"{len(thinking)} answers that end without a final answer; to bound it, the client can "
        "send chat_template_kwargs.enable_thinking=false or the server can set "
        "--reasoning-budget; this changes answers, so it is your call",
    )
