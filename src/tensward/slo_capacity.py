"""Capacity at a limit: the highest requests per second a setup was measured to hold within the
declared latency limits. Steps of closed-loop clients run on one launch, each on its own slice of
the workload's prompts; every number is a step's measurement, never an estimate between steps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

from .capacity import FILE as PROMPT_CAPACITY_FILE
from .engines import Engine
from .environment import describe_run, require_available
from .errors import PROJECT_CONFIG_UNSUPPORTED, PreflightError
from .files import new_run_id, write_json, write_private
from .inputs import PromptEntry
from .measure import measure
from .measurement import Measurement
from .project import ResolvedProject, load_project, settings_for, with_workload
from .report.build import Subject, overrides_suffix
from .runtime import Runtime
from .settings import Settings
from .slo import Slo

MAX_STEPS = 8
REQUESTS_PER_CLIENT = 4
MIN_STEP_REQUESTS = 32
MAX_FAILED_SHARE = 0.01
CLOSE_MISS = 0.10  # a first step outside the limits by at most this share is close to them


@dataclass(frozen=True, slots=True)
class Step:
    clients: int
    run_dir: Path
    request_throughput: float | None
    ttft_p95_ms: float | None
    tpot_p95_ms: float | None
    failed_share: float
    within: bool


@dataclass(frozen=True, slots=True)
class Capacity:
    """One launch's steps and what it sent: ``reused`` requests repeated a prompt already sent
    on the launch, which ``prefix_caching`` may have served from cache."""

    directory: Path
    steps: list[Step]
    prompts: int
    sent: int
    reused: int
    prefix_caching: bool | None


def client_steps(concurrency: int) -> list[int]:
    """Powers of two up to twice the registered clients, the largest MAX_STEPS of them."""
    steps = [2**k for k in range(0, 31) if 2**k <= 2 * max(concurrency, 1)]
    return steps[-MAX_STEPS:]


def step_requests(clients: int) -> int:
    """The measured requests of a step of ``clients`` clients."""
    return max(REQUESTS_PER_CLIENT * clients, MIN_STEP_REQUESTS)


def step_slices(
    prompts: Sequence[PromptEntry], clients: Sequence[int], *, warmup: int = 0
) -> tuple[list[tuple[PromptEntry, ...]], int]:
    """Each step's prompts: its measured requests, then its start-up wave of one request per
    client. All are consecutive slices of the workload's prompts on one cursor that starts after
    the launch's ``warmup`` requests, so no prompt is sent twice on the launch unless the workload
    is smaller than what the launch sends; and how many requests reused one."""
    slices, cursor = [], warmup
    for count in (step_requests(c) + c for c in clients):
        slices.append(tuple(prompts[(cursor + i) % len(prompts)] for i in range(count)))
        cursor += count
    return slices, max(cursor - len(prompts), 0)


def highest(steps: Sequence[Step]) -> tuple[Step | None, bool]:
    """The step with the highest measured requests per second within the limits, and whether
    the largest step is within them."""
    within = [s for s in steps if s.within and s.request_throughput is not None]
    best = max(within, key=lambda s: s.request_throughput or 0.0, default=None)
    return best, bool(steps) and steps[-1].within


def capacity_at(
    project: Path,
    *,
    engine: Engine,
    runtime: Runtime,
    engine_args: Sequence[str],
    slo: Slo,
    ready_timeout_s: float,
    verify_weights: bool,
) -> Capacity:
    """Measure every client step on one launch of the setup ``engine_args`` make; write each
    step's run directory under ``<project>/capacity/<id>/steps/`` and ``capacity.json``. The
    project's cached prompt capacity is copied in, so a setup already probed is not probed
    again."""
    resolved = load_project(project, verify_weights=verify_weights)
    arrival = resolved.config.workload.arrival
    if arrival.kind != "closed_loop":
        raise PreflightError(
            PROJECT_CONFIG_UNSUPPORTED,
            "--capacity steps the number of closed-loop clients; this workload's arrival is "
            f"{arrival.kind}",
        )
    availability = require_available(engine, runtime)
    settings = settings_for(resolved, engine, engine_args)
    environment = describe_run(engine, runtime, resolved.artifact, availability, resolved.anatomy)
    clients = client_steps(arrival.concurrency or 1)
    warmup = resolved.config.warmup_requests
    slices, reused = step_slices(resolved.prompts, clients, warmup=warmup)
    variants = [_variant(resolved, chunk, c) for chunk, c in zip(slices, clients, strict=True)]
    root = project.expanduser() / "capacity"
    root.mkdir(mode=0o700, exist_ok=True)
    directory = root / new_run_id()
    (directory / "steps").mkdir(mode=0o700, parents=True)
    cached = project.expanduser() / PROMPT_CAPACITY_FILE
    if cached.is_file():  # the launch reads only a row whose key is this setup's
        write_private(directory / PROMPT_CAPACITY_FILE, cached.read_text())
    run_dirs = [directory / "steps" / f"{c:04d}-clients" for c in clients]
    measured = measure(
        resolved,
        engine=engine,
        runtime=runtime,
        settings=settings,
        run_dirs=run_dirs,
        retain_responses=False,
        ready_timeout_s=ready_timeout_s,
        environment=environment,
        subject=Subject(
            "Capacity step" + overrides_suffix(engine_args), resolved.record.current_setup
        ),
        slo=slo,
        run_projects=variants,
        steady_state=True,
        engine_args=engine_args,
    )
    steps = [_step(c, d, m, slo) for c, d, m in zip(clients, run_dirs, measured, strict=True)]
    caching, source = prefix_caching(engine, settings)
    sent = warmup + sum(len(chunk) for chunk in slices)
    write_json(directory / "capacity.json", {
        "limits": slo.declared, "engine_args": list(engine_args), "prompts": len(resolved.prompts),
        "sent": sent, "reused": reused, "prefix_caching": caching, "prefix_caching_from": source,
        "steps": [{**asdict(s), "run_dir": s.run_dir.name} for s in steps],
    })  # fmt: skip
    return Capacity(directory, steps, len(resolved.prompts), sent, reused, caching)


def _variant(
    project: ResolvedProject, prompts: tuple[PromptEntry, ...], clients: int
) -> ResolvedProject:
    """The step's workload: ``clients`` clients offering the slice's measured requests; the start-up
    wave continues the cycle into the slice's last ``clients`` prompts."""
    variant = with_workload(project, requests=prompts, concurrency=clients)
    workload = variant.config.workload.model_copy(update={"request_count": len(prompts) - clients})
    return replace(variant, config=replace(variant.config, workload=workload))


def _step(clients: int, run_dir: Path, measurement: Measurement, slo: Slo) -> Step:
    total = measurement.succeeded + measurement.failed
    failed_share = measurement.failed / total if total else 0.0
    within = (
        slo.met_at_p95(measurement)
        and failed_share <= MAX_FAILED_SHARE
        and measurement.request_throughput is not None
    )
    return Step(
        clients, run_dir, measurement.request_throughput, measurement.ttft_p95_ms,
        measurement.tpot_p95_ms, failed_share, within,
    )  # fmt: skip


def prefix_caching(engine: Engine, settings: Settings) -> tuple[bool | None, str]:
    """Whether the launch cached prefixes, and where that comes from: the setup, else the
    engine's default (None when the engine states none)."""
    if settings.prefix_caching is not None:
        return settings.prefix_caching, "setup"
    default = engine.defaults.get("prefix_caching")
    return (None if default is None else default.startswith("on")), "engine default"


def capacity_lines(slo: Slo, capacity: Capacity) -> list[str]:
    """The terminal output: every step's measurement, the highest within the limits, and what
    the steps could not show."""
    limits = " and ".join(
        f"{label} p95 <= {slo.declared[name]:g} ms"
        for name, label in (("ttft_ms", "TTFT"), ("tpot_ms", "TPOT"))
        if name in slo.declared
    )
    steps, directory = capacity.steps, capacity.directory
    lines = [
        f"capacity within {limits}, measured on one launch in {len(steps)} steps:",
        "clients   req/s   TTFT p95   TPOT p95   failed   within",
    ]
    for s in steps:
        rate = "-" if s.request_throughput is None else f"{s.request_throughput:.2f}"
        lines.append(
            f"{s.clients:7d}{rate:>8}{_ms(s.ttft_p95_ms):>11}{_ms(s.tpot_p95_ms):>11}"
            f"{s.failed_share:>9.1%}   {'yes' if s.within else 'no'}"
        )
    best, largest_within = highest(steps)
    largest = steps[-1]
    every = all(s.within for s in steps)
    if best is not None and not (every and best is largest):
        lines.append(f"highest within your limits: {best.request_throughput:.2f} requests per "
                     f"second at {_clients(best.clients)}")  # fmt: skip
        if close := _close(slo, steps, best):
            lines.append(close)
    if every:
        lines.append(f"every step was within your limits: the capacity is at least "
                     f"{largest.request_throughput:.2f} requests per second "
                     f"({_clients(largest.clients)}, the largest step)")  # fmt: skip
    elif largest_within:
        lines.append(f"the largest step was within your limits: the capacity is at least "
                     f"{largest.request_throughput:.2f} requests per second "
                     f"({_clients(largest.clients)})")  # fmt: skip
    elif best is None:
        first = steps[0]
        failed = ""
        if first.failed_share > 0:
            failed = f", and {first.failed_share:.1%} of its requests failed"
        lines.append(f"no step was within your limits; the smallest, {_clients(first.clients)}, "
                     f"had TTFT p95 {_ms(first.ttft_p95_ms)} and TPOT p95 "
                     f"{_ms(first.tpot_p95_ms)}{failed}")  # fmt: skip
    if capacity.reused:
        caching = capacity.prefix_caching
        state = {True: "on", False: "off", None: "unknown"}[caching]
        low = "" if caching is False else ": those requests' TTFT may read low"
        lines.append(f"the workload has {capacity.prompts} prompts and this launch sent "
                     f"{capacity.sent} requests (warm-up and start-up waves included), so "
                     f"{capacity.reused} requests reused a prompt already sent on this launch, "
                     f"with prefix caching {state}{low}")  # fmt: skip
    lines.append(f"steps: {directory / 'steps'}/ (each with its own report); "
                 f"summary: {directory / 'capacity.json'}")  # fmt: skip
    return lines


def _close(slo: Slo, steps: Sequence[Step], best: Step) -> str | None:
    """When the first step after ``best`` missed only a latency limit, by at most CLOSE_MISS,
    that capacity lies between the two steps' measured rates."""
    after = next((s for s in steps if s.clients > best.clients), None)
    if after is None or after.within or after.request_throughput is None:
        return None
    if after.failed_share > MAX_FAILED_SHARE:
        return None
    misses = [
        (label, value / limit - 1)
        for name, label, value in (
            ("ttft_ms", "TTFT", after.ttft_p95_ms),
            ("tpot_ms", "TPOT", after.tpot_p95_ms),
        )
        if (limit := slo.declared.get(name)) and value is not None and value > limit
    ]
    if not misses or max(miss for _, miss in misses) > CLOSE_MISS:
        return None
    missed = " and ".join(f"{label} p95 by {miss:.1%}" for label, miss in misses)
    low, high = sorted((best.request_throughput or 0.0, after.request_throughput))
    return (
        f"close to the limit: {_clients(after.clients)} missed {missed}, so capacity lies between "
        f"{low:.2f} and {high:.2f} requests per second"
    )


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:,.0f} ms"


def _clients(count: int) -> str:
    return f"{count} client" if count == 1 else f"{count} clients"
