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
from .gpu_state import GpuSampler
from .history import PastRun, past_runs, step_parent
from .kb_run import FACTS_RECORD
from .measure import serving_gpu
from .measurement import Window, _tpot_ms, _ttft_ms, window_share
from .project import load_project, project_engine
from .rebuild import ANALYSIS, PROFILES, rebuild
from .report.model import Report
from .runs import RUN_RECORD, WINDOW_RECORD
from .runtime import Runtime
from .settings import Settings
from .slo import Slo
from .window import MAX_FAILED

MAX_AGE_S = 24 * 3600
BORROWED_AGE_S = 14 * 24 * 3600
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


def _uuids(devices: Sequence[Mapping[str, Any]]) -> list[Any]:
    return sorted(str(device.get("uuid")) for device in devices)


def _power_cap_differs(run_dir: Path, now: float | None) -> str | None:
    then = (_json(run_dir / FACTS_RECORD).get("facts") or {}).get("gpu.power_cap_w")
    if then is None or now is None:
        return "has no power cap on record to compare with this machine's"
    if abs(then - now) > 0.5:
        return f"ran under a {then:g} W power cap, this machine enforces {now:g} W"
    return None


def _why_not(
    run_dir: Path,
    engine: Engine,
    runtime: Runtime,
    *,
    version: str | None,
    devices: Sequence[Mapping[str, Any]],
    retain_responses: bool,
    limits: Slo,
    borrow: bool = False,
    power_cap_w: float | None = None,
) -> str | None:
    """Why the run in ``run_dir`` cannot stand in for a measurement, or None. A borrowed
    reference (``borrow``) may be up to 14 days old, must have run on this machine's own GPUs
    (the same uuids, not imported) under the power cap this machine enforces now
    (``power_cap_w``)."""
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
    if borrow and record.get("imported_from"):
        return "was imported from another project, and a borrowed reference must be measured here"
    if borrow and _uuids(record.get("gpus_identity", ())) != _uuids(devices):
        return "ran on another machine's GPU (its uuid differs from this machine's)"
    if borrow and (why := _power_cap_differs(run_dir, power_cap_w)):
        return why
    age = time.time() - measured_at(run_dir.name)
    if borrow and age > BORROWED_AGE_S:
        return f"was measured {age / 86400:.0f} days ago (the limit is 14 days)"
    if not borrow and age > MAX_AGE_S:
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
    borrow: bool = False,
) -> PastRun | str:
    """The newest run of the project with exactly ``args`` that may stand in for a new
    measurement, else why none may (``NO_RUN`` when there is no run of ``args``). With ``borrow``
    a run up to 14 days old qualifies if it ran under this machine's power cap; the project's
    snapshot already binds the weights and the workload."""
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
    cap = _current_power_cap(runtime) if borrow else None
    reasons = []
    for path in candidates:
        if not named:
            return "no GPU is detected on this machine"
        why = _why_not(
            path, engine, runtime, version=found.version, devices=named,
            retain_responses=retain_responses, limits=limits, borrow=borrow, power_cap_w=cap,
        )  # fmt: skip
        if why is None and (run := PastRun.read(path, snapshot, engine)) is not None:
            return run
        reasons.append(f"run {path.name} {why or 'cannot be read'}")
    return reasons[0]


def reusable(project_dir: Path, run: str, engine: Engine) -> PastRun | str:
    """Run ``run`` when it may stand in once more for the step it measured, else why not: it
    served requests of this registration, kept its answers, is not itself reused, is at most
    24 h old by both its id and metrics.json mtime, and was made by this Tensward minor version.
    Its window may be a planned one."""
    runs = project_dir / "runs"
    run_dir = runs / run
    try:
        if (
            run != run_dir.name
            or runs.resolve() != project_dir.resolve() / "runs"
            or run_dir.resolve().parent != runs.resolve()
        ):
            return f"run {run} is not a run in this project"
        measured_epoch = measured_at(run)
    except (OSError, RuntimeError, ValueError):
        return f"run {run} is not a run in this project"
    record = _json(run_dir / RUN_RECORD)
    snapshot = _json(project_dir / "project.json").get("snapshot_id")
    if not record or record.get("snapshot_id") != snapshot:
        return f"run {run} is not a run of this registration"
    if not (run_dir / "responses.jsonl").is_file():
        return f"run {run} kept no answers"
    if record.get("reused_from"):
        return f"run {run} is itself a reused run"
    if _minor(record.get("tensward_version")) != _minor(__version__):
        return f"run {run} was made by Tensward {record.get('tensward_version') or 'before 0.1.6'}"
    age = time.time() - measured_epoch
    try:
        age = max(age, time.time() - (run_dir / "metrics.json").stat().st_mtime)
    except OSError:
        return f"run {run} kept no measurement"
    if age > MAX_AGE_S:
        return f"run {run} was measured {age / 3600:.0f} h ago (the limit is 24 h)"
    found = PastRun.read(run_dir, snapshot, engine)
    if found is None:
        return f"run {run} kept no measurement"
    if not found.served:
        return f"run {run} served no request"
    return found


def reusable_run(project_dir: Path, run: str, parent: str) -> str | None:
    """None when run ``run`` may stand in once more for the step from run ``parent``, else why
    not: ``reusable``, and its recorded window parent is that run. Legacy windows without a
    parent field use the measurement path's ``step_parent`` rule."""
    engine = project_engine(load_project(project_dir).record.current_setup)
    found = reusable(project_dir, run, engine)
    if isinstance(found, str):
        return found
    window = _json(project_dir / "runs" / run / WINDOW_RECORD)
    if "parent" in window:
        source = window["parent"]
    else:
        before = step_parent(
            project_dir, past_runs(project_dir, engine, before=run), found.engine_args
        )
        source = before.run_id if before else None
    if source != parent:
        return f"run {run} was not made from run {parent}"
    return None


def _current_power_cap(runtime: Runtime) -> float | None:
    sampler = GpuSampler(runtime.platform, runtime.gpus)
    sampler.start()
    try:
        return sampler.power_limit_w()
    finally:
        sampler.stop()


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
            first_chunk_ns=row.get("first_chunk_ns"),
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
    project_dir: Path,
    source: PastRun,
    context: AnalysisContext,
    *,
    limits: Slo,
    borrowed: bool = False,
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
            **({"borrowed": True} if borrowed else {}),
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


def reused_line(project_dir: Path, source: PastRun, *, borrowed: bool = False) -> str:
    """What the report of a reused run says about its source."""
    record = _json(project_dir / "runs" / source.run_id / RUN_RECORD)
    gpu = serving_gpu(record.get("gpus_identity", ()), tuple(record.get("gpus") or ()) or None)
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(measured_at(source.run_id)))
    kept = (
        "; borrowed as the reference, so a step from it is judged with the cross-launch floor "
        "widened"
        if borrowed
        else ""
    )
    return (
        f"Reused run {source.run_id}, measured at {stamp} on {gpu or 'an unnamed GPU'}{kept}; "
        "not measured again (`--remeasure` measures it)"
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
