"""One census for a setup ``try`` kept. A try step runs no census start of its own, so a record
that reads census readings cannot be decided on the kept run; when one waits only on them, the
kept setup gets one short census start and its report is written again."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ..context import AnalysisContext, active
from ..engines import Engine
from ..measure import serving_gpu
from ..project import load_project, settings_for
from ..rebuild import ANALYSIS, NotReported, rebuild
from ..runs import RUN_RECORD, RunFile
from ..runtime import Runtime
from ..settings import Settings
from .record import CensusRecord, read_record, write_record
from .start import expected_census_s, run_census


def _analysed(run_dir: Path) -> dict[str, object]:
    try:
        return json.loads((run_dir / RUN_RECORD).read_text("utf-8")).get(ANALYSIS) or {}
    except (OSError, ValueError, AttributeError):
        return {}


def _census_run(source: Path) -> Path | None:
    """The nearest run of the kept chain, ``source`` first, that has a census record."""
    runs = source.parent
    ids = [source.name, (_run_record(source).get("reused_from") or {}).get("census_run")]
    try:
        kept = json.loads((runs.parent / "kept.json").read_text("utf-8"))
    except (OSError, ValueError):
        kept = {}
    ids += [kept.get("previous")] if kept.get("run") == source.name else []
    ids += [entry.get("run") for entry in reversed(kept.get("kept_history") or ())]
    snapshot = _run_record(source).get("snapshot_id")
    for run in ids:
        if not isinstance(run, str):
            continue
        if run != source.name and _run_record(runs / run).get("snapshot_id") != snapshot:
            return None  # another registration: never borrow its census
        if read_record(runs / run) is not None:
            return runs / run
    return None


def _run_record(run_dir: Path) -> dict[str, Any]:
    try:
        record: dict[str, Any] = json.loads((run_dir / RUN_RECORD).read_text("utf-8"))
        return record
    except (OSError, ValueError, AttributeError):
        return {}


def _first_ok(source: Path) -> bool:
    origin = _census_run(source)
    record = None if origin is None else read_record(origin)
    return record is not None and record.status == "ok"


def reads_census(source: Path) -> bool:
    """Whether some record the matcher loaded for ``source``'s run reads a census fact."""
    try:
        kb = json.loads((source / "report.json").read_text("utf-8")).get("kb") or {}
    except (OSError, ValueError, AttributeError):
        return False
    return bool(kb.get("census_readers"))


def waits_on_census(run_dir: Path) -> bool:
    """Whether some record holds on this run but for census readings it does not have."""
    try:
        kb = json.loads((run_dir / "report.json").read_text("utf-8")).get("kb") or {}
    except (OSError, ValueError, AttributeError):
        return False
    return any(
        item.get("facts") and all(str(name).startswith("census.") for name in item["facts"])
        for item in (*(kb.get("unknown") or ()), *(kb.get("census_unknown") or ()))
    )


def expected_refresh_s(
    source: Path | None, engine: Engine, runtime: Runtime, settings: Settings
) -> float | None:
    """Seconds the census start after a kept change would add, None when none can follow."""
    if source is None or not _first_ok(source) or not reads_census(source):
        return None
    load_s = _analysed(source).get("load_s")
    return expected_census_s(
        engine, runtime, settings, float(load_s) if isinstance(load_s, (int, float)) else 0.0
    )


def refresh_census(
    project: Path,
    run_dir: Path,
    *,
    source: Path | None,
    engine: Engine,
    runtime: Runtime,
    ready_timeout_s: float,
) -> CensusRecord | None:
    """The census of the kept run ``run_dir``, made from ``source``'s offers, and its report
    written again; None when ``source``'s census was not ok, the kept run has a census already
    or nothing waits on one. Ctrl-C writes the record and is raised again."""
    if source is None or not _first_ok(source) or read_record(run_dir) is not None:
        return None
    if not waits_on_census(run_dir):
        return None
    resolved = load_project(project, verify_weights=False)
    run_file = RunFile.read(run_dir)
    load_s = _analysed(run_dir).get("load_s")
    try:
        metrics = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    except (OSError, ValueError):
        metrics = {}
    req_s = metrics.get("request_throughput") if isinstance(metrics, dict) else None
    placement = metrics.get("placement") if isinstance(metrics, dict) else None
    context = AnalysisContext(
        engine=engine.name,
        project=project,
        new_run=run_dir,
        load_s=float(load_s) if isinstance(load_s, (int, float)) else None,
        placement=placement if isinstance(placement, dict) else None,
    )
    with active(context):
        interrupted = run_census(
            resolved,
            engine,
            runtime,
            settings_for(resolved, engine, run_file.engine_args),
            run_dir,
            ready_timeout_s,
            version=run_file.engine_version,
            req_s=req_s,
            gpu_model=serving_gpu([asdict(g) for g in run_file.gpus_identity], run_file.gpus),
        )
    record = read_record(run_dir)
    if record is not None:
        record = replace(record, refresh_of=source.name)
        write_record(run_dir, record)
        try:
            rebuild(project, run_dir.name, census_refreshed=True)
        except NotReported:
            pass
    if interrupted:
        raise KeyboardInterrupt
    return record
