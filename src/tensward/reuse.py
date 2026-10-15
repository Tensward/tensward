"""Reuse a trusted run in place of a new measurement, and import a run from another project.

A run may stand in for a measurement when everything that decides its numbers is as it was:
the project, the engine and its version, the GPU and its driver, the environment the engine
inherited, the Tensward minor version, and a full window of enough successful requests that is
recent. ``reuse`` then writes a new run directory for the current goal and limits.
"""

from __future__ import annotations

import calendar
import json
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import __version__, environment
from .client import RequestRecord
from .context import AnalysisContext
from .engines import Engine
from .files import new_run_id, write_json
from .history import PastRun
from .measure import serving_gpu
from .measurement import Window, _tpot_ms, _ttft_ms, window_share
from .rebuild import ANALYSIS, PROFILES, rebuild
from .report.model import Report
from .runs import RUN_RECORD, WINDOW_RECORD
from .runtime import Runtime
from .settings import Settings
from .slo import Slo
from .window import MAX_FAILED

MAX_AGE_S = 24 * 3600
NO_RUN = "no earlier run of this setup"
RECOMPUTED = ("goodput", "slo_attainment")
RUN_ID_TIME = "%Y%m%dT%H%M%SZ"


class NotImported(Exception):
    """A run directory that cannot be imported; ``str(error)`` is the one line shown."""


def _json(path: Path) -> dict[str, Any]:
    try:
        document: dict[str, Any] = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return document


def measured_at(run_id: str) -> float:
    """When the run started, as seconds since the epoch, from its id."""
    return float(calendar.timegm(time.strptime(run_id[:16], RUN_ID_TIME)))


def _stamp(seconds: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(seconds))


def _minor(version: str | None) -> str | None:
    return ".".join(version.split(".")[:2]) if version else None


def _drivers(devices: Sequence[Mapping[str, Any]]) -> list[tuple[Any, Any]]:
    return [(device.get("name"), device.get("driver")) for device in devices]


def _why_not(
    run_dir: Path,
    engine: Engine,
    runtime: Runtime,
    *,
    version: str | None,
    devices: Sequence[Mapping[str, Any]],
    retain_responses: bool,
    limits: Slo,
) -> str | None:
    """Why the run in ``run_dir`` cannot stand in for a measurement, or None."""
    record = _json(run_dir / RUN_RECORD)
    metrics = _json(run_dir / "metrics.json")
    report = _json(run_dir / "report.json")
    if not metrics.get("succeeded"):
        return "served no request"
    if record.get("reused_from"):
        return "is itself a reused run"
    if ANALYSIS not in record or not (run_dir / "requests.jsonl").is_file():
        return "was written before Tensward recorded what its report needs"
    if any((metrics.get(kind) or {}).get("analysis") is not None for kind in PROFILES):
        return "was profiled (--trace/--counters)"
    if record.get("engine_version") != version:
        return (
            f"ran {engine.label} {record.get('engine_version') or 'of unknown version'}, "
            f"this machine has {version or 'a version Tensward cannot read'}"
        )
    if _drivers(record.get("gpus_identity", ())) != _drivers(devices):
        return "ran on another GPU or driver"
    if _minor(record.get("tensward_version")) != _minor(__version__):
        return f"was made by Tensward {record.get('tensward_version') or 'before 0.1.6'}"
    if record.get("runtime") != runtime.label:
        return f"ran as {record.get('runtime')}, not as {runtime.label}"
    settings = Settings.from_json(record.get("settings", {}))
    now = (
        engine.inherited_env(
            {name: value for name, value in os.environ.items() if name not in settings.extra_env}
        )
        if runtime.inherits_environment
        else {}
    )
    if record.get("inherited_env", {}) != now:
        return "ran with another inherited engine environment"
    if retain_responses and not record.get("retained_responses"):
        return "kept no answers"
    if ((_json(run_dir / WINDOW_RECORD).get("plan") or {}).get("seconds")) is not None:
        return "was measured over a planned window, not the full one"
    failed, served = metrics.get("failed", 0), metrics["succeeded"]
    if failed / (failed + served) > MAX_FAILED:
        return f"had {failed / (failed + served):.0%} of its requests fail (the limit is 2%)"
    age = time.time() - measured_at(run_dir.name)
    if age > MAX_AGE_S:
        return f"was measured {age / 3600:.0f} h ago (the limit is 24 h)"
    declared = (report.get("goal") or {}).get("limits")
    if metrics.get("window") is None and declared != limits.declared:
        return (
            "did not record its measurement window, so goodput cannot be counted under other "
            "limits; declare the limits it ran with"
        )
    return None


def trusted(
    project_dir: Path,
    engine: Engine,
    runtime: Runtime,
    args: frozenset[str],
    *,
    retain_responses: bool = True,
    limits: Slo = Slo(),
) -> PastRun | str:
    """The newest run of the project with exactly ``args`` that may stand in for a new
    measurement, else why none may (``NO_RUN`` when there is no run of ``args``)."""
    snapshot = _json(project_dir / "project.json").get("snapshot_id")
    runs = project_dir / "runs"
    candidates = sorted(
        (
            path for path in (runs.iterdir() if runs.is_dir() else ())
            if (record := _json(path / RUN_RECORD)).get("snapshot_id") == snapshot
            and frozenset(record.get("engine_args", ())) == args
        ),
        reverse=True,
    )  # fmt: skip
    if not candidates:
        return NO_RUN
    found = engine.availability(runtime.kind, runtime.target)
    selected = runtime.platform.select(environment.detect_devices(), runtime.gpus)[0]
    named = [runtime.platform.identity(device) for device in selected]
    reasons = []
    for path in candidates:
        if not named:
            return "no GPU is detected on this machine"
        why = _why_not(
            path, engine, runtime, version=found.version, devices=named,
            retain_responses=retain_responses, limits=limits,
        )  # fmt: skip
        if why is None and (run := PastRun.read(path, snapshot, engine)) is not None:
            return run
        reasons.append(f"run {path.name} {why or 'cannot be read'}")
    return reasons[0]


def _recomputed(
    rows: Sequence[Mapping[str, Any]], window: Window, limits: Slo
) -> dict[str, float | None]:
    """Goodput and limit attainment counted again from a run's requests under ``limits``, as the
    measurement counted them: over the requests sent inside the window."""
    records = [
        RequestRecord(
            request_id=row["request_id"],
            scheduled_ns=row["dispatch_ns"],
            dispatch_ns=row["dispatch_ns"],
            first_content_ns=row["first_content_ns"],
            terminal_ns=row["terminal_ns"],
            outcome=row["outcome"],
            committed_output_tokens=row["committed_output_tokens"],
            prompt_index=row["prompt_index"],
        )
        for row in rows
        if row["dispatch_ns"] >= window.start_ns
    ]
    successes = [record for record in records if record.outcome == "success"]
    shares = {record.request_id: window_share(record, window) for record in successes}
    met = [r for r in successes if limits.met(_ttft_ms(r), _tpot_ms(r))]
    rated = window.seconds > 0 and (not successes or any(shares.values()))
    good = sum(shares[record.request_id] for record in met)
    return {
        "goodput": good / window.seconds if rated else None,
        "slo_attainment": len(met) / len(records) if records else None,
    }


def reuse(
    project_dir: Path, source: PastRun, context: AnalysisContext, *, limits: Slo
) -> tuple[Path, Report]:
    """A new run directory: a copy of ``source``'s, ``run.json`` with ``reused_from`` and the
    current analysis record, goodput and attainment counted again under ``limits``, and the
    report written for the current goal, limits and flags. Raises NotReported when the report
    cannot be written; the directory is then removed."""
    source_dir = project_dir / "runs" / source.run_id
    run_id = new_run_id()
    run_dir = project_dir / "runs" / run_id
    shutil.copytree(source_dir, run_dir)
    try:
        record = _json(run_dir / RUN_RECORD)
        record["reused_from"] = {
            "run": source.run_id,
            "measured_at": _stamp(measured_at(source.run_id)),
        }
        record[ANALYSIS] = {
            **record[ANALYSIS],
            "flags": list(context.flags),
            "max_req_s_loss": context.max_req_s_loss,
            "adopt_within_noise": context.adopt_within_noise,
        }
        write_json(run_dir / RUN_RECORD, record)
        metrics = _json(run_dir / "metrics.json")
        recomputed: list[str] = []
        if metrics.get("window") is not None:
            rows = [
                json.loads(line)
                for line in (run_dir / "requests.jsonl").read_text("utf-8").splitlines()
            ]
            counted = _recomputed(rows, Window(**metrics["window"]), limits)
            write_json(run_dir / "metrics.json", {**metrics, **counted})
            recomputed = list(RECOMPUTED)
        write_json(
            run_dir / WINDOW_RECORD,
            {"schema_version": "1", **_json(run_dir / WINDOW_RECORD), "recomputed": recomputed},
        )
        return rebuild(project_dir, run_id, goal=context.goal, limits=limits, every_section=True)
    except BaseException:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise


def reused_line(project_dir: Path, source: PastRun) -> str:
    """What the report of a reused run says about its source."""
    record = _json(project_dir / "runs" / source.run_id / RUN_RECORD)
    gpu = serving_gpu(record.get("gpus_identity", ()), tuple(record.get("gpus") or ()) or None)
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(measured_at(source.run_id)))
    return (
        f"Reused run {source.run_id}, measured at {stamp} on {gpu or 'an unnamed GPU'}; not "
        "measured again (`--remeasure` measures it)"
    )


def import_run(run_dir: Path, project_dir: Path) -> Path:
    """Copy a run directory of another project into this one under its own id, marked
    ``imported_from``; refused when its snapshot id is not this project's or the id exists."""
    run_dir = run_dir.expanduser()
    record = _json(run_dir / RUN_RECORD)
    if not record:
        raise NotImported(f"{run_dir} is not a run directory (no {RUN_RECORD})")
    project_dir = project_dir.expanduser()
    snapshot = _json(project_dir / "project.json").get("snapshot_id")
    if record.get("snapshot_id") != snapshot:
        raise NotImported(
            f"run {run_dir.name} was measured for another registration (its model, "
            "configuration, workload or current setup differ from this project's)"
        )
    target = project_dir / "runs" / run_dir.name
    if target.exists():
        raise NotImported(f"this project already has run {run_dir.name}")
    shutil.copytree(run_dir, target)
    write_json(
        target / RUN_RECORD,
        {**record, "imported_from": {"run": run_dir.name, "project": str(run_dir.parent.parent)}},
    )
    return target
