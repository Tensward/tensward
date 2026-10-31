"""``tensward report``: a run's report written again from its run directory, with the runs before
it as its history and the project's kept setup and rejections as they are now. No engine starts
and no run directory is made: only the run's ``report.json`` and ``report.md`` are rewritten."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields, is_dataclass, replace
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from .answer_options import AnswerOptions
from .capabilities import Resolved
from .classify import Diagnosis
from .context import AnalysisContext, Goal, active
from .estimate import FREE_TOOL_CHOICES, RunInputs
from .files import write_json, write_jsonl
from .history import cap_flag, cold_kv_tokens, cold_pools, past_runs, recorded_env
from .kb_run import write_facts
from .kept import LAST, OTHER_REGISTRATION, newest_run, read_kept, recorded_run
from .measure import serving_gpu
from .measurement import Measurement
from .playbook import Situation
from .project import (
    json_answer_ids,
    linear_weight_bytes,
    load_project,
    project_engine,
    settings_for,
    workload_facts,
)
from .rejected import read_rejected, standing
from .report.build import RunOutputs, Subject, build_report, overrides_suffix
from .report.markdown import to_markdown
from .report.model import SECTION_ORDER, Figure, Report, Section, SectionId, Table, Text, to_json
from .runs import RUN_RECORD, RunFile
from .slo import Slo
from .suggest import suggestions

ANALYSIS = "analysis"  # the run.json key of what a later report needs from the analysis
PROMPT_BLOCKS = "prompt_blocks.jsonl"
PROFILES = ("trace", "counters")  # metrics.json keys of a profiled run's analyses
REBUILT: tuple[SectionId, ...] = (
    "next_steps",
    "worth_measuring",
    "keep",
)  # the sections history and choices decide
LIMITED: tuple[SectionId, ...] = ("header", "setup", "performance")  # the sections limits decide


class NotReported(Exception):
    """A run whose report cannot be written again; ``str(error)`` is the one line shown."""


def record_analysis(run_dir: Path, context: AnalysisContext, prompt_ids: Sequence[str]) -> None:
    """Add to ``run.json`` what the analysis knew that the run directory does not keep: the
    options its suggested commands repeat, the declared floor, the launch's timings, the window
    it planned and the within-noise policy; and write ``prompt_blocks.jsonl`` when the analysis
    did not, so a later report can replay the prompts' shared blocks whatever it offers."""
    blocks = run_dir / PROMPT_BLOCKS
    if context.prompt_blocks and context.kv_block_tokens and not blocks.is_file():
        write_jsonl(blocks, [
            {
                "prompt_id": prompt_ids[index],
                "tokens": tokens,
                "block_tokens": int(context.kv_block_tokens),
                "blocks": list(hashes),
            }
            for index, (tokens, hashes) in sorted(context.prompt_blocks.items())
        ])  # fmt: skip
    path = run_dir / RUN_RECORD
    recorded = {
        "flags": list(context.flags),
        "max_req_s_loss": context.max_req_s_loss,
        "launched_s": context.launched_s,
        "load_s": context.load_s,
        "measured_s": context.measured_s,
        "window": dict(context.window_plan) if context.window_plan is not None else None,
        "adopt_within_noise": context.adopt_within_noise,
        "adopted_within_noise": context.window_record.get("adopted_within_noise"),
    }
    write_json(path, {**json.loads(path.read_text("utf-8")), ANALYSIS: recorded})


def _from_json(kind: Any, value: Any) -> Any:
    """``value`` as ``dataclasses.asdict`` and JSON left it, back as ``kind``."""
    if value is None:
        return None
    if isinstance(kind, type) and is_dataclass(kind):
        hints = get_type_hints(kind)
        return kind(**{
            f.name: _from_json(hints[f.name], value[f.name])
            for f in fields(kind) if f.init and f.name in value
        })  # fmt: skip
    origin, args = get_origin(kind), get_args(kind)
    if origin in (Union, UnionType):
        inner = [arg for arg in args if arg is not type(None)]
        return _from_json(inner[0], value) if len(inner) == 1 else value
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_from_json(args[0], item) for item in value)
        return tuple(_from_json(arg, item) for arg, item in zip(args, value, strict=True))
    if origin in (dict, Mapping):
        return {key: _from_json(args[1], item) for key, item in value.items()}
    return value


def _section(document: Mapping[str, Any]) -> Section:
    kinds = {"figure": Figure, "text": Text, "table": Table}
    return Section(
        id=document["id"],
        title=document["title"],
        title_audience=document["title_audience"],
        blocks=tuple(_from_json(kinds[block["kind"]], block) for block in document["blocks"]),
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return (
        [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        if path.is_file()
        else []
    )


def rebuild(
    project_path: Path,
    run: str,
    *,
    goal: Goal | None = None,
    limits: Slo | None = None,
    every_section: bool = False,
) -> tuple[Path, Report]:
    """Write ``run``'s report again (``LAST``: the newest run's) and return its directory and
    the report; raises NotReported for a run whose report cannot be written again. ``goal`` and
    ``limits`` replace the run's own; with ``every_section`` the sections they decide are made
    again too, and the others are kept as written."""
    project = load_project(project_path, verify_weights=False)
    project_dir = project_path.expanduser()
    run = newest_run(project_dir, NotReported) if run == LAST else run
    run_dir = recorded_run(project_dir, run, NotReported)
    run_file = RunFile.read(run_dir)
    if run_file.snapshot_id != project.record.snapshot_id:
        raise NotReported(
            f"{OTHER_REGISTRATION.format(run=run)}, so this project's history does not apply to it"
        )
    record = json.loads((run_dir / RUN_RECORD).read_text("utf-8"))
    analysed = record.get(ANALYSIS)
    if analysed is None:
        raise NotReported(
            f"run {run} was written before Tensward recorded what its report needs, or its "
            "analysis did not finish; run `tensward analyse` again"
        )
    recorded = json.loads((run_dir / "report.json").read_text("utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    if any((metrics.get(kind) or {}).get("analysis") is not None for kind in PROFILES):
        raise NotReported(
            f"run {run} was profiled (--trace/--counters): its report depends on the profile's "
            "analysis, which Tensward does not read back; run `tensward analyse` again"
        )
    engine = project_engine(project.record.current_setup)
    engine_args = run_file.engine_args
    changed = (*engine_args, *sorted(recorded_env(run_file.model_dump(), engine.flag_rules)))
    settings = settings_for(project, engine, engine_args)
    measurement: Measurement = _from_json(Measurement, metrics)
    diagnosis: Diagnosis = _from_json(Diagnosis, metrics["diagnosis"])
    index = {entry.id: at for at, entry in enumerate(project.prompts)}
    blocks = {row["prompt_id"]: row for row in _jsonl(run_dir / PROMPT_BLOCKS)}
    facts = replace(
        workload_facts(project),
        rendered=tuple(
            tuple(blocks[p.id]["blocks"]) if p.id in blocks else () for p in project.prompts
        ),
    )
    rows = _jsonl(run_dir / "requests.jsonl")
    inputs = RunInputs(
        arrival=project.config.workload.arrival.kind,
        gpu_count=len(run_file.gpus) if run_file.gpus else 1,
        forced_tools=any(p.tool_choice not in FREE_TOOL_CHOICES for p in project.prompts),
        prompt_order=tuple(
            row["prompt_index"] for row in sorted(rows, key=lambda row: row["dispatch_ns"])
        ),
        engine_version=run_file.engine_version,
        prompt_blocks={
            index[pid]: (row["tokens"], tuple(row["blocks"])) for pid, row in blocks.items()
        },
    )
    goal = goal or recorded["goal"]["goal"]
    slo = limits or Slo(
        recorded["goal"]["limits"].get("ttft_ms", math.inf),
        recorded["goal"]["limits"].get("tpot_ms", math.inf),
    )
    options = record.get("answer_options") or {}
    runs = past_runs(project_dir, engine, before=run)
    args = frozenset(changed)
    rejected, retried = standing(read_rejected(project_dir), engine.flag_rules)
    context = AnalysisContext(
        engine=engine.name,
        goal=goal,
        ttft_p95_limit_ms=slo.declared.get("ttft_ms"),
        tpot_p95_limit_ms=slo.declared.get("tpot_ms"),
        flags=tuple(analysed["flags"]),
        max_req_s_loss=analysed["max_req_s_loss"],
        engine_args=changed,
        cold_kv_tokens=cold_kv_tokens(runs, args, cap_flag(engine)),
        cold_pools=cold_pools(runs, args, cap_flag(engine)),
        launched_s=analysed["launched_s"],
        load_s=analysed["load_s"],
        measured_s=analysed["measured_s"],
        project=project_dir,
        new_run=run_dir,
        kept=read_kept(project_dir),
        rejected=rejected,
        retried=retried,
        answer_options=_from_json(AnswerOptions, options) if options.get("read") else None,
        answer_options_unknown=options.get("why"),
        gpu_capability=options.get("capability"),
        json_answers=json_answer_ids(project),
        reported_run=run,
        adopt_within_noise=bool(analysed.get("adopt_within_noise")),
        linear_weight_bytes=linear_weight_bytes(project),
        anatomy=project.anatomy,
        placement=metrics.get("placement"),
        probes=record.get("probes"),
    )
    effective = engine.effective(settings, _from_json(Resolved, metrics.get("resolved")))
    situation = Situation(
        measurement=measurement, facts=facts, settings=effective, diagnosis=diagnosis
    )
    with active(context):
        suggested = suggestions(
            engine,
            situation,
            current=settings_for(project, engine, ()),
            project_path=project_path,
            version=run_file.engine_version,
            run=inputs,
        )
        built = build_report(RunOutputs(
            run_id=run,
            ran=recorded["ran"],
            measurement=measurement,
            diagnosis=diagnosis,
            suggested=suggested,
            rows=rows,
            facts=facts,
            settings=settings,
            subject=Subject(
                "Your current setup" + overrides_suffix(engine_args), project.record.current_setup
            ),
            image=run_file.image,
            slo=slo,
            engine=engine,
            gpu=serving_gpu([asdict(g) for g in run_file.gpus_identity], run_file.gpus),
            source=project.record.current_setup.label,
            engine_args=engine_args,
            retained=run_file.retained_responses,
        ))  # fmt: skip
    if context.kb_facts is not None:
        write_facts(run_dir, context.kb_facts, replace=False)
    remade = (*REBUILT, *LIMITED) if every_section else REBUILT
    kept = [_section(s) for s in recorded["sections"] if s["id"] not in remade]
    again = [s for s in built.sections if s.id in remade]
    report = Report(
        run_id=run,
        ran=recorded["ran"],
        sections=tuple(sorted([*kept, *again], key=lambda s: SECTION_ORDER.index(s.id))),
        goal=built.goal,
        kb=built.kb,
    )
    with active(context):  # an offer's answer risk reads the checkpoint's size
        document = to_json(report)
    write_json(run_dir / "report.json", document)
    (run_dir / "report.md").write_text(to_markdown(report), "utf-8")
    return run_dir, report
