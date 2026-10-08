"""Measurement-time checks: problems that make a run's report less trustworthy or its
workload unservable, found while the run is measured."""

from __future__ import annotations

import math
import re
from dataclasses import asdict

from .engines import Engine
from .measurement import WAVE_OUTLASTED_CHECK, Measurement, Window
from .playbook import WorkloadFacts
from .settings import GAUGE_SIGNALS, SPECULATION_SIGNALS, EngineSignals, Settings

SHORT_WINDOW_S = 2.0
USABLE_WINDOW_S = 10.0
TOKEN_MISMATCH = 0.10


def checks(
    engine: Engine,
    final: EngineSignals | None,
    facts: WorkloadFacts,
    settings: Settings,
    peak_running: float | None,
    server_log: str,
    log_at_window: str,
    source: str,
    *,
    replicas: int = 1,
) -> list[str]:
    """Problems that make this report less trustworthy or the workload unservable."""
    checks = []
    if (blocker := _tool_calling_blocker(engine, facts, settings, source)) is not None:
        checks.append(f"tool calling: {blocker}")
    if final is None:
        if engine.capabilities(None, settings).polls_metrics:
            checks.append(
                "the engine's metrics could not be read, so engine signals are not measured"
            )
    elif missing := [
        name
        for name, value in asdict(final).items()
        if name in reported(engine, settings, replicas)
        and value is None
        and name not in SPECULATION_SIGNALS | {"prompt_tokens_computed"}
    ]:
        checks.append(
            f"the engine's metrics do not expose {', '.join(missing)}: this engine version may "
            f"have renamed its metrics, so those signals are not measured"
        )
    if facts.schema_with_tools:
        checks.append(
            "structured output with tools: the grammar forces every answer to the schema, so "
            "the model cannot call the tools the prompts offer"
        )
    graph_batch = engine.max_graph_batch(settings)
    if peak_running is not None and graph_batch is not None and peak_running > graph_batch:
        checks.append(
            f"running sequences reached {peak_running:g} but CUDA graphs cover only "
            f"{graph_batch}; larger batches run without graphs"
        )
    for marker, problem in engine.log_checks.items():
        if server_log.count(marker) > log_at_window.count(marker):
            checks.append(problem)
    return checks


def _tool_calling_blocker(
    engine: Engine, facts: WorkloadFacts, settings: Settings, source: str
) -> str | None:
    """Why the workload's tools cannot be served, with the fix; None when they can."""
    return engine.tool_calling_fix(settings, source) if facts.offers_tools else None


def cast_to_float16(engine: Engine, server_log: str) -> bool:
    """Whether the engine's server log says it cast a bfloat16 checkpoint to float16."""
    return any(
        found.groups() == ("bfloat16", "float16")
        for found in re.finditer(engine.dtype_cast, server_log)
    )


def _short_window(measurement: Measurement, window: Window, in_window: int, with_wave: bool) -> str:
    """The short-window line, with the request count that would give a usable window: the
    ``in_window`` requests dispatched inside the window are scaled to fill USABLE_WINDOW_S, and
    the rest (the start-up wave and earlier dispatches) are kept as they are."""
    text = f"the measurement window was only {window.seconds:.1f} s"
    if with_wave:
        text += " and includes the start-up wave"
    text += "; throughput is noisy; raise request_count"
    sent = measurement.succeeded + measurement.failed
    if measurement.request_throughput is not None and sent:
        wanted = sent - in_window + math.ceil(in_window * USABLE_WINDOW_S / window.seconds)
        text += f" to about {-(-wanted // 10) * 10}"
    return text


def _token_mismatch(measurement: Measurement, engine_generation_tokens: float | None) -> bool:
    window = measurement.window
    if not engine_generation_tokens or window is None or measurement.output_throughput is None:
        return False
    client = measurement.output_throughput * window.seconds
    return abs(engine_generation_tokens - client) > TOKEN_MISMATCH * engine_generation_tokens


def engine_output_rate(
    measurement: Measurement, engine_generation_tokens: float | None, fell_back: bool
) -> float | None:
    """The engine's output tokens per second over the window when its token count disagrees
    with the requests'; None when they agree, the engine span is not the window, or the window
    is too short for the scrapes' timing to be trusted."""
    window = measurement.window
    if fell_back or window is None or window.kind != "steady":
        return None
    if window.seconds < USABLE_WINDOW_S:
        return None
    if _token_mismatch(measurement, engine_generation_tokens) and engine_generation_tokens:
        return engine_generation_tokens / window.seconds
    return None


def window_checks(
    measurement: Measurement,
    engine_generation_tokens: float | None,
    fell_back: bool,
    outlasted: bool,
    in_window: int,
    with_wave: bool,
) -> list[str]:
    """Problems with a measurement window: a start-up wave that outlasted the last request, a
    steady window too short or empty to trust, an engine span that is not the window, or engine
    and client token counts that disagree. ``in_window``: the requests dispatched inside it."""
    window = measurement.window
    found = []
    if outlasted:
        found.append(WAVE_OUTLASTED_CHECK)
    if window is None:
        return found
    if window.seconds < SHORT_WINDOW_S:
        found.append(_short_window(measurement, window, in_window, with_wave))
    if window.kind != "steady":
        return found
    unrated = measurement.succeeded and measurement.request_throughput is None
    if window.seconds >= SHORT_WINDOW_S and unrated:
        found.append(
            "no request streamed inside the measurement window, so throughput is not "
            "measured; raise request_count"
        )
    if fell_back:
        found.append(
            "the engine could not be read at the last request, so its counters cover the run "
            "to its end rather than the measurement window"
        )
    elif _token_mismatch(measurement, engine_generation_tokens):
        client = (measurement.output_throughput or 0.0) * window.seconds
        found.append(
            f"the engine generated {engine_generation_tokens:.0f} tokens over the window "
            f"but the requests account for {client:.0f}; throughput may be unreliable"
        )
    return found


def reported(engine: Engine, settings: Settings, replicas: int = 1) -> set[str]:
    """The neutral signals the engine declares it reports for these settings. With several
    replicas no gauge or capacity is read (each is one engine's); with several API-server
    processes there is no single process's CPU counter."""
    declared = engine.capabilities(None, settings).signals
    found = {name for name, support in declared.items() if support.quality != "absent"}
    if replicas > 1:
        found -= GAUGE_SIGNALS
    if (engine.frontend_processes(settings) or 1) > 1:
        found.discard("frontend_cpu_seconds")
    return found
