"""The run a try was made from, measured once more right after the try (A-B-A): a machine that
drifts between two runs can make a change look better than it is, so a change that looks like a
win is judged against the mean of its parent's two runs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .context import AnalysisContext, active
from .engines import Engine
from .files import write_json
from .progress import Note, emit
from .runtime import Runtime

AGAIN_DIR = "parent-again"
AGAIN_FILE = "parent-again.json"
FIGURES = ("request_throughput", "output_throughput", "ttft_p95_ms", "tpot_p95_ms", "e2e_p95_ms")


def measure_parent(
    project_dir: Path,
    engine: Engine,
    runtime: Runtime,
    step_dir: Path,
    parent: str,
    ready_timeout_s: float,
) -> dict[str, float] | None:
    """Measure run ``parent``'s setup once more over the try's own window, started as its first
    start was (its recorded settings with the baseline's placement pinned under a change, as
    ``analyse`` starts a step), its server log and rows in ``step_dir``/parent-again; its figures
    by metrics key, also written to ``step_dir``/parent-again.json. None when it does not start
    or serves no request."""
    from .analyse import _launch, _launched
    from .errors import AnalyseFailure
    from .history import last_load_s, recorded_env
    from .measure import summarize_run
    from .project import load_project, workload_facts
    from .reasoning_check import _environment
    from .settings import Settings
    from .slo import DEFAULT_SLO

    base_dir = project_dir / "runs" / parent
    try:
        record = json.loads((base_dir / "run.json").read_text("utf-8"))
        plan = json.loads((step_dir / "window.json").read_text("utf-8")).get("plan")
    except (OSError, ValueError):
        return None
    first = load_project(project_dir, verify_weights=False)
    (step_dir / AGAIN_FILE).unlink(missing_ok=True)  # no verdict of an earlier re-measure stays
    out = step_dir / AGAIN_DIR
    shutil.rmtree(out, ignore_errors=True)
    context = AnalysisContext(
        engine=engine.name,
        project=project_dir,
        new_run=out,
        try_step=True,
        window_plan=plan,
        model_bytes=first.artifact.metadata.weight_bytes,
        expected_load_s=last_load_s(project_dir),
        anatomy=first.anatomy,
    )
    emit(Note(text="Measuring your current setup once more, to see whether the machine drifted"))
    try:
        with _environment(engine, recorded_env(record, engine.flag_rules)), active(context):
            resolved = load_project(project_dir, verify_weights=False)  # the try's window
            settings = _launched(
                resolved, engine, project_dir, Settings.from_json(record.get("settings") or {}),
                tuple(record.get("engine_args") or ()),
            )  # fmt: skip
            capture = _launch(resolved, engine, runtime, settings, out, False, ready_timeout_s)
            measured = summarize_run(
                capture, engine=engine, settings=settings, slo=DEFAULT_SLO, gpus=runtime.gpus,
                facts=workload_facts(resolved), platform=runtime.platform,
            )  # fmt: skip
    except AnalyseFailure:
        return None
    if not measured.succeeded:
        return None
    figures = {key: value for key in FIGURES if (value := getattr(measured, key, None)) is not None}
    (step_dir / AGAIN_FILE).write_text(json.dumps({"parent": parent, **figures}), "utf-8")
    return figures


def judged_again(step: dict[str, Any], key: str, again: dict[str, float]) -> dict[str, Any] | None:
    """The step's goal change against the mean of its parent's two runs: the parent's drift
    between them, the change against their mean and whether it is still beyond the band in the
    right direction (``more_is_better``: ``step['better_up']``). None without both figures."""
    first, second = step.get("before"), again.get(key)
    if not first or second is None:
        return None
    mean = (first + second) / 2
    change = (1 + step["change"]) * first / mean - 1
    up = step.get("better_up", True)
    return {
        "first": first,
        "second": second,
        "mean": mean,
        "drift": second / first - 1,
        "change": change,
        "better": change > step["band"] if up else change < -step["band"],
    }


def common_drift(step_dir: Path, parent_dir: Path, metric: str) -> float | None:
    """The parent's second run against its first on a p95 ``metric`` (window name) over the
    prompts both of its runs and the step answered: two time-boxed windows finish different
    requests, so their own p95s are not of the same set. None when the rows cannot pair."""
    from . import window

    again, first, step = (
        window.read_rows(path) for path in (step_dir / AGAIN_DIR, parent_dir, step_dir)
    )
    if not again or not first or not step:
        return None
    common = {r.index for r in step if r.ok}
    found = window.paired(
        [r for r in again if r.index in common], [r for r in first if r.index in common], metric
    )
    return None if found is None else found[0]


def record_judged(step_dir: Path, judged: dict[str, Any]) -> None:
    """Keep the A-B-A verdict beside the parent's second figures, for keep and the held list,
    and in the step block of the try's report.json."""
    path = step_dir / AGAIN_FILE
    try:
        found = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        found = {}
    path.write_text(json.dumps({**found, "judged": judged}), "utf-8")
    report = step_dir / "report.json"
    try:
        document = json.loads(report.read_text("utf-8"))
    except (OSError, ValueError):
        return
    for section in document.get("sections") or ():
        blocks = section.get("blocks") or []
        for index, block in enumerate(blocks):
            if block.get("kind") == "data" and block.get("key") == "step":
                blocks[index] = with_judged(block, judged)
    write_json(report, document)


def with_judged(step: dict[str, Any], judged: dict[str, Any] | None) -> dict[str, Any]:
    """The step block once its parent was measured again: its goal change against the mean of
    the parent's two runs, the first run's figures kept under ``first``."""
    if judged is None or step.get("change") is None:
        return step
    if earlier := step.get("first"):
        rest = {key: value for key, value in step.items() if key not in ("first", "again")}
        step = {**rest, **earlier}
    first = {key: step.get(key) for key in ("before", "change", "verdict", "recommended")}
    change, band = judged["change"], step.get("band") or 0.0
    verdict = "better" if judged["better"] else "within_noise" if abs(change) <= band else "worse"
    return {
        **step, "first": first, "before": judged["mean"], "change": change, "verdict": verdict,
        "recommended": bool(step.get("recommended")) and bool(judged["better"]), "again": True,
    }  # fmt: skip


def read_judged(step_dir: Path) -> dict[str, Any] | None:
    """The A-B-A verdict of the try in ``step_dir``; None when its parent was not measured again."""
    try:
        found = json.loads((step_dir / AGAIN_FILE).read_text("utf-8")).get("judged")
    except (OSError, ValueError, AttributeError):
        return None
    return found if isinstance(found, dict) else None
