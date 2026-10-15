"""Recorded runs replayed through Tensward's diagnosis, suggestions and report.

``replays/t4-saturated``: Qwen3-4B served as float16 on a Tesla T4 by vLLM 0.30.0, 128
closed-loop clients, 96-token answers, measured by Tensward 0.3.9. ``report.json`` is that run's
report as written. ``server.log`` holds the start-up lines of a cold start of the same command on
the same machine. ``tokstats.json`` holds the workload's prompt lengths as its author counted
them; the longest, 257, stands in for the server's count, which the run did not keep.
``gpu-samples.csv`` holds nvidia-smi samples of the same command and load on that machine,
seconds first, only those inside the measured window (after the 30 s warm-up, as Tensward's
steady window excludes the start-up wave). The T4's 70 W enforced power limit, 1590 MHz maximum
SM clock and 15 GiB of memory are filled in; the throttle reasons were not recorded.

``replays/ladder-kv`` and ``replays/ladder-cap``: two 0.3.9 three-step ladders on an NVIDIA A10G
(``run.json`` without the machine's fields, ``metrics.json``, ``report.json``); ``ladder-cap``
keeps the first run of its last step.

``replays/a10g-{lt,lf8,lp}``: the 0.3.10 release ladders on an A10G, vLLM 0.30.0: ``lt``
(Qwen2.5-7B-Instruct, long prompts, 48-token answers, 128 clients, cap 8 then 16, budget 1024),
``lf8`` (the same at budget 2048, 64 clients, cap 8, 16, 32), ``lp`` (Mistral-7B-Instruct-v0.3,
512-token answers, 64 clients, cap 32 then 16, 8, 4). ``replays/a10g2-{lt,lf,lp}``: ``lt``,
``lf`` (``lf8`` from cap 16) and ``lp`` measured again (``lt`` cap 8, then 29; ``lf`` cap 16, 29,
then 32; ``lp`` ending at cap 8 with n-gram speculation). ``a10g2-lp``'s prompts (those its runs
sent) and its first run's answers keep each word as a stable id, so equal words stay equal and no
text is left.

``replays/t42-ladder``: the 0.3.10 release TTFT ladder on the T4 of ``t4-saturated``, short
answers, 128 clients: the default 2048-token step budget cold then warm, then 512 and 256. Each
run was saturated at the card's power limit; ``t4-saturated``'s samples stand in for theirs.

``replays/l4-steps``: four 300 s runs of Qwen3-0.6B on an L4, vLLM 0.30.0, 16 closed-loop
clients, the same 1,032 prompts in the same order: three launches of the default and one with
async scheduling off (``20261014T165306Z-eaf7``). ``controls.json`` holds the figures of the
same host's default, fp8-weights and fp8-KV controls. ``20261015T065602Z-33bf``: the requests of a
planned 60 s run of the default on another L4 the next day, the first 236 prompts, its drain
included.

``replays/l4-kv``: OLMo-2-1124-7B-Instruct on an L4, vLLM 0.30.0, 36 closed-loop clients short
of KV memory (81 preemptions): the full 200-request reference (``2dda``) and a planned 60 s run of
the same setting (``84d3``, 144 requests).

``replays/a10g-traffic``: five 0.3.11 runs on an A10G, vLLM 0.30.0. Phi-3.5-mini-instruct, 64
closed-loop clients under the TPOT goal: cap 32 (``ae53``), the same with fp8 weights (``6d98``)
and cap 16 (``147f``). Qwen2.5-7B-Instruct with tool calls, 64 clients, no cap: bf16 (``535d``,
whose prompt-work share was not measured) and fp8 weights (``79d5``).

Each JSON fixture keeps only the fields these tests read; a run that is analysed but never
history has no ``report.json``.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import pytest
from test_playbook import OFFERED

from tensward import window
from tensward.answer_options import AnswerOptions, Format
from tensward.calibration import profile_for
from tensward.capabilities import Resolved
from tensward.ceilings import Ceilings
from tensward.classify import Diagnosis, Finding, classify, measurement_from_metrics
from tensward.combine import PREDICTED, parent_offer
from tensward.context import AnalysisContext, Goal, active
from tensward.engines import Engine
from tensward.engines.vllm import VLLM, logs
from tensward.engines.vllm.engine import VllmEngine
from tensward.engines.vllm.startup import startup
from tensward.gpu_state import GpuState, summarize
from tensward.history import (
    PastRun,
    cap_flag,
    checked,
    cold_kv_tokens,
    cold_pools,
    command_args,
    past_runs,
    planned_window,
    seconds_to_test,
    time_to_test,
)
from tensward.kept import REGISTERED, Kept
from tensward.measurement import Measurement
from tensward.platforms.nvidia import NVIDIA
from tensward.playbook import Situation, Suggestion, WorkloadFacts
from tensward.playbook_common import running_at_cap
from tensward.predict import Prediction, held_steps, predict, predict_pair
from tensward.project import CurrentSetup
from tensward.prompt_lookup import Coverage
from tensward.rejected import Rejection
from tensward.report.build import RunOutputs, Subject, build_report, next_steps_section
from tensward.report.markdown import sections_markdown, to_markdown
from tensward.report.model import Report, Text, to_json
from tensward.settings import Settings
from tensward.slo import NO_LIMITS, Slo
from tensward.suggest import (
    Suggested,
    _combined,
    _past_limit,
    answer_warning,
    carried_gate,
    first_try,
    group_members,
    ordered_for_goal,
    suggestions,
    try_first_order,
)
from tensward.toolcalls import ToolCallStats
from tensward.workload import ArrivalSpec

REPLAYS = Path(__file__).parent / "replays"
T4 = REPLAYS / "t4-saturated"
REPLAYED = "replayed"  # where a test copies recorded runs; the golden capture skips it
FIGURES = ("request_throughput", "output_throughput", "total_throughput", "ttft_p50_ms",
           "ttft_p95_ms", "tpot_p50_ms", "tpot_p95_ms", "e2e_p95_ms", "peak_running",
           "peak_waiting", "preemptions", "prefix_cache_hits", "prompt_tokens_per_step",
           "frontend_cpu_cores")  # fmt: skip
PERCENTS = ("peak_kv_usage", "prefix_cache_hit_rate")
CEILING_FIGURES = {
    "avg_context": "avg_context_tokens", "avg_batch": "avg_running_batch",
    "decode_ceiling_batch1": "decode_ceiling_batch1_tok_s",
    "decode_ceiling": "decode_ceiling_tok_s",
    "decode_measured": "measured_decode_tok_s", "prefill_ceiling": "prefill_ceiling_tok_s",
    "prefill_measured": "measured_prefill_tok_s", "decode_share": "decode_pct_of_ceiling",
    "prefill_share": "prefill_pct_of_ceiling", "window_share": "ceiling_time_pct_of_window",
}  # fmt: skip
T4_SETTINGS = Settings(dtype="float16", max_context_len=8192)
T4_FACTS = WorkloadFacts(prompts=(), output_tokens=96, context_limit=40960)


def _blocks() -> dict[str, dict]:
    report = json.loads((T4 / "report.json").read_text())
    wanted = ("performance", "engine", "ceilings", "requests")
    return {b["key"]: b for s in report["sections"] if s["id"] in wanted for b in s["blocks"]}


def _number(text: str, pattern: str) -> float:
    found = re.search(pattern, text)
    assert found is not None, pattern
    return float(found[1])


def t4_measurement() -> Measurement:
    """The T4 run as Tensward measured it, from its report and the start-up log."""
    blocks, log = _blocks(), (T4 / "server.log").read_text()
    kv = blocks["kv_capacity"]["text"]
    ceilings = Ceilings(
        gpu="T4",
        bandwidth_gbs=_number(blocks["bandwidth"]["text"], r"([\d.]+) GB/s"),
        bandwidth_source="datasheet",
        tensor_tflops=_number(blocks["tensor_rate"]["text"], r"([\d.]+) TFLOPS"),
        weight_bytes_per_step=round(
            _number(blocks["weights_per_step"]["text"], r"([\d.]+) GB") * 1e9
        ),
        kv_bytes_per_sequence=round(blocks["kv_per_sequence"]["value"] * 2**20),
        **{field: blocks[key]["value"] for key, field in CEILING_FIGURES.items()},
    )
    return Measurement(
        int(_number(blocks["requests"]["text"], r"(\d+) succeeded")),
        int(_number(blocks["requests"]["text"], r"(\d+) failed")),
        **{name: blocks[name]["value"] for name in FIGURES},
        **{name: blocks[name]["value"] / 100 for name in PERCENTS},
        kv_capacity_tokens=_number(kv, r"(\d+) tokens"),
        kv_max_concurrency=_number(kv, r"([\d.]+) full-length"),
        kv_capacity_estimate_tokens=_number(kv, r"before the run: (\d+) tokens"),
        max_prompt_tokens=json.loads((T4 / "tokstats.json").read_text())["max_prompt_tokens"],
        max_context_len=8192,
        gpu_memory_gib=15.0,
        step_budget_tokens=logs.step_budget(log),
        compile_cache=logs.compile_cache(log),
        ceilings=ceilings,
    )


def t4_gpu() -> GpuState:
    samples = []
    for line in (T4 / "gpu-samples.csv").read_text().splitlines():
        seconds, _, rest = line.partition(", ")
        sample = NVIDIA.parse_sample(rest, round(float(seconds) * 1e9))
        assert sample is not None
        samples.append(sample)
    return summarize(samples, samples[0].at_ns, samples[-1].at_ns)  # the window only


def t4_report(
    slo: Slo = NO_LIMITS,
    gpu: bool = False,
    goal: Goal = "throughput",
    project_path: Path | None = None,
    engine: Engine = VLLM,
    settings: Settings = T4_SETTINGS,
) -> str:
    measurement = t4_measurement()
    with active(AnalysisContext(goal=goal)):
        diagnosis = classify(
            measurement, settings, None, engine=engine, version="0.30.0",
            gpu_state=t4_gpu() if gpu else None,
        )  # fmt: skip
        situation = Situation(measurement=measurement, facts=T4_FACTS, settings=settings,
                              diagnosis=diagnosis)  # fmt: skip
        suggested = suggestions(engine, situation, current=settings, project_path=project_path,
                                version="0.30.0")  # fmt: skip
        outputs = RunOutputs(
            run_id="t4", ran="vLLM 0.30.0 on Tesla T4", measurement=measurement,
            diagnosis=diagnosis, suggested=suggested, rows=(), facts=T4_FACTS,
            settings=settings, subject=Subject(), image=None, slo=slo, engine=engine,
            gpu="Tesla T4", source="", engine_args=(), retained=False,
        )  # fmt: skip
        return to_markdown(build_report(outputs))


class ResolvingVllm(VllmEngine):
    """vLLM with the launch-resolved lever values written into its effective settings, as an
    engine that resolves them does."""

    def effective(self, settings: Settings, resolved: Resolved | None) -> Settings:
        return dataclasses.replace(settings, **(resolved.levers if resolved is not None else {}))


RESOLVING = ResolvingVllm()


def test_kv_room_is_counted_in_this_workload_s_requests_and_a_cold_pool_is_flagged(
    tmp_path: Path,
) -> None:
    report = t4_report(gpu=True)
    assert (
        "room for about 77 requests of this workload at its longest (257 prompt + 96 answer tokens)"
        in report
    )
    assert re.search(r", 19\d at its mean \(13\d tokens\)", report)
    assert "full-length requests" not in report
    assert "cold compile cache" in report
    # run.json keeps the requested settings, metrics.json what the engine resolved at launch;
    # this run's situation holds the resolved settings, so only a resolved against resolved
    # comparison finds the warm pool.
    warm = tmp_path / REPLAYED / "project" / "runs" / "20261010T060000Z-aaaa"
    warm.mkdir(parents=True)
    (tmp_path / REPLAYED / "project" / "project.json").write_text(json.dumps({"snapshot_id": "s"}))
    run_json = {"snapshot_id": "s", "engine_args": [], "compile_cache": "warm",
                "settings": dataclasses.asdict(T4_SETTINGS)}  # fmt: skip
    (warm / "run.json").write_text(json.dumps(run_json))
    (warm / "metrics.json").write_text(json.dumps({
        "succeeded": 10, "kv_capacity_tokens": 36688.0,
        "resolved": {"levers": {"kv_memory_fraction": 0.92}, "facts": {}},
    }))  # fmt: skip
    (warm / "report.json").write_text(json.dumps({"sections": []}))
    resolved = dataclasses.replace(T4_SETTINGS, kv_memory_fraction=0.92)
    later = t4_report(
        gpu=True, project_path=tmp_path / REPLAYED / "project", engine=RESOLVING, settings=resolved
    )
    assert (
        "(a warm start of these settings in run 20261010T060000Z-aaaa held 36688 tokens)" in later
    )
    assert "error -24% against the warm pool" in later


def test_a_saturated_t4_run_counts_goodput_only_against_declared_limits() -> None:
    measurement = t4_measurement()
    assert (measurement.step_budget_tokens, measurement.compile_cache) == (2048, "cold")
    plain = t4_report()
    assert "goodput" not in plain and "requests meeting the SLO" not in plain
    assert "limits:" not in plain
    declared = t4_report(Slo(4220.0, math.inf))
    assert "goodput (TTFT <= 4220 ms per request)" in declared and "TPOT <=" not in declared
    assert "limits: TTFT <= 4220 ms per request" in declared


def test_a_saturated_t4_run_names_the_gpu_at_its_power_limit() -> None:
    diagnosis = classify(
        t4_measurement(), T4_SETTINGS, None, engine=VLLM, version="0.30.0", gpu_state=t4_gpu()
    )
    by_class = {finding.bottleneck: finding for finding in diagnosis.findings}
    assert diagnosis.primary == "gpu_saturated"
    assert "the GPU was saturated at its power limit" in by_class["gpu_saturated"].evidence
    assert by_class["load_limited"].state == "clear"  # decided, not "can't tell"
    report = t4_report(gpu=True)
    at_clock = r"hardware ceiling reached \d+% \(the datasheet peaks at the measured SM clock"
    assert re.search(at_clock, report)
    assert "the GPU was saturated, so the rest is not reachable" in report


def test_under_the_ttft_goal_the_smaller_step_budget_comes_first(tmp_path: Path) -> None:
    report = t4_report(gpu=True, goal="ttft")
    steps = report[report.index("## What to try next") :]
    first = re.search(r"- `([a-z0-9-]+)`", steps)
    assert first is not None and first[1] == "small-step-budget"
    assert "Try first:\nFor TTFT:\n- `small-step-budget`: " in steps  # under no crossed class
    # the T4 ladder's 256-token run: its stop explains, with no empty saturated-GPU group above
    situation, last = replay_step(tmp_path, "t42", 3, range(3))
    shown = sections_markdown([next_steps_section(situation.diagnosis, last, retained=False)])
    assert last.stop is not None and "For a saturated GPU:" not in shown
    assert last.stop.endswith("no measured limit calls for a change that leaves TTFT alone.")
    assert "expected:" not in report and "goal: time to first token (TTFT p95)" in report
    throughput = t4_report(gpu=True)
    assert (
        "`small-step-budget`" not in throughput
    )  # the step budget moves the other way for throughput


@pytest.mark.parametrize(
    ("goal", "shown"),
    [
        ("tpot", "Try first:\nFor a saturated GPU:\n- `lower-concurrency`: the GPU was "
                 "saturated with 128 sequences running at once"),  # at a throughput cost
        ("throughput", "For a saturated GPU:\n- no change in this engine's playbook applies "
                       "here\n\nNothing left in Try first for throughput: the GPU was saturated "
                       "at its power limit"),
    ],
)  # fmt: skip
def test_under_a_latency_goal_try_first_names_its_group(goal: Goal, shown: str) -> None:
    report = t4_report(gpu=True, goal=goal)
    assert shown in report and "no measured signal points to a change" not in report


def test_the_first_change_carries_a_short_plan_and_the_next_run_checks_it() -> None:
    report = t4_report(gpu=True, goal="ttft")
    assert "then (prediction)" not in report  # a saturated GPU is not a TTFT class
    plan = [
        {"change": "small-step-budget", "args": ["max-num-batched-tokens=512"],
         "limit": "gpu_saturated",
         "signal": "the GPU busy 95% or more and at its power or thermal limit"},
    ]  # fmt: skip
    measured = t4_measurement()
    diagnosis = classify(measured, T4_SETTINGS, None, engine=VLLM, gpu_state=t4_gpu())
    run = Situation(measurement=measured, facts=T4_FACTS, settings=T4_SETTINGS, diagnosis=diagnosis)
    before = PastRun(run_id="r0", engine_args=frozenset(), settings={}, metrics={}, primary=None,
                     expected={}, plan=tuple(plan), goal="ttft")  # fmt: skip
    step = frozenset({"max-num-batched-tokens=512"})
    text, check = checked([before], step, run, "ttft", engine=VLLM)
    assert check == {"run": "r0", "change": "small-step-budget", "predicted": "gpu_saturated",
                     "actual": "gpu_saturated", "confirmed": True, "held": False}  # fmt: skip
    assert text == (
        "run r0 predicted a saturated GPU would bind after `small-step-budget`: confirmed"
    )
    assert checked([before], step, run, "tpot", engine=VLLM) == (None, None)


@pytest.mark.parametrize(
    ("run_id", "changes"),
    [
        # text-only: no media encoders to drop, and no step undoes the one before it
        ("a10g-kv-capacity-mistral-20261004T131828Z-c964",
         ["more-kv-memory", "raise-concurrency", "stable-prompt-prefix"]),
        # KV projected at 10% at the raised cap: decode bandwidth, crossed now, still binds; a
        # plan never adds a change that can affect the answers
        ("a10g-queueing-mistral-20261004T125303Z-8a30", ["raise-concurrency", "fast-quant-kernel"]),
    ],
)  # fmt: skip
def test_a_plan_names_changes_that_fit_this_setup(run_id: str, changes: list[str]) -> None:
    rows = json.loads((Path(__file__).parent / "calibration_snapshots.json").read_text())
    row = next(row for row in rows if row["id"] == run_id)
    measured = measurement_from_metrics(row["metrics"])
    settings = Settings(max_concurrent_requests=row["cap"])
    facts = WorkloadFacts(prompts=(), output_tokens=128, context_limit=8192)
    diagnosis = classify(measured, settings, None)
    run = Situation(measurement=measured, facts=facts, settings=settings, diagnosis=diagnosis)
    shown = suggestions(VLLM, run, current=settings, project_path=None)
    assert [step["change"] for step in shown.plan] == changes


def test_with_a_broken_ttft_limit_the_rendered_try_first_puts_the_ttft_change_first() -> None:
    ranked, other, lead, by_list = ordered_for_goal(
        OFFERED, "throughput", Measurement(100, 0, ttft_p95_ms=900.0), 800.0, lambda s: None
    )
    diagnosis = classify(Measurement(100, 0), Settings(), None)
    suggested = Suggested(suggestions=tuple(ranked), not_applicable=(), blocked={}, lead=lead,
                          other_goals=other, by_list=by_list)  # fmt: skip
    shown = sections_markdown([next_steps_section(diagnosis, suggested, retained=False)])
    assert by_list and shown.index("`faster`") < shown.index("`cap-up`")


def ladder_step(tmp_path: Path, ladder: str, current: int, history: Sequence[int]) -> Suggested:
    """Analyse run ``current`` of a recorded ladder again, with the runs ``history`` (indices)
    as the project's earlier runs."""
    root = REPLAYS / ladder
    runs = sorted((root / "runs").iterdir())
    project = tmp_path / REPLAYED / ladder
    (project / "runs").mkdir(parents=True)
    (project / "project.json").write_text((root / "project.json").read_text())
    for index in history:
        shutil.copytree(runs[index], project / "runs" / runs[index].name)
    run_dir = runs[current]
    record = json.loads((run_dir / "run.json").read_text())
    measurement = measurement_from_metrics(json.loads((run_dir / "metrics.json").read_text()))
    settings = Settings.from_json(record["settings"])
    base = Settings.from_json(json.loads((runs[0] / "run.json").read_text())["settings"])
    diagnosis = classify(measurement, settings, None, engine=VLLM, version="0.30.0")
    facts = WorkloadFacts(prompts=(), output_tokens=256, context_limit=32768)
    situation = Situation(
        measurement=measurement, facts=facts, settings=settings, diagnosis=diagnosis
    )
    with active(AnalysisContext(engine_args=tuple(record["engine_args"]))):
        return suggestions(VLLM, situation, current=base, project_path=project, version="0.30.0")


def test_a_ladder_reverts_a_loss_first_and_never_suggests_a_tried_change_again(
    tmp_path: Path,
) -> None:
    # ladder-kv: runs 0-2 the start, 3-5 with prefix caching, 6-8 with a cap of 128 as well
    lost = ladder_step(tmp_path / "lost", "ladder-kv", 6, range(6))
    assert lost.memory is not None and lost.memory.revert is not None
    assert command_args(lost.memory.revert.command or "") == {"enable-prefix-caching"}
    assert not any("helped less than" in line for line in lost.memory.lines)
    again = ladder_step(tmp_path / "again", "ladder-kv", 5, [0, 1, 2, 3, 4, 6, 7, 8])
    names = {s.entry.name for s in again.suggestions}
    assert "raise-concurrency" not in names
    assert again.memory is not None
    assert any(
        line.startswith("`raise-concurrency` (") and " tried in run " in line
        for line in again.memory.lines
    )
    assert again.stop is not None
    assert again.stop.startswith("Every change Tensward has for throughput here was tried")
    records = sorted((REPLAYS / "ladder-kv" / "runs").glob("*/run.json"))
    for step in range(1, 9):
        done = {frozenset(json.loads(p.read_text())["engine_args"]) for p in records[:step]}
        offered = ladder_step(tmp_path / f"s{step}", "ladder-kv", step, range(step))
        assert not any(command_args(s.command) in done for s in offered.suggestions if s.command)


def test_a_ladder_keeps_a_gain_without_a_revert(tmp_path: Path) -> None:
    # ladder-cap: runs 0-5 the start, 6 with a cap of 16
    gained = ladder_step(tmp_path, "ladder-cap", 6, range(6))
    assert gained.memory is not None and gained.memory.revert is None
    assert not any("helped less than" in line for line in gained.memory.lines)


LADDERS = {
    "lt": ("a10g-lt", "throughput", 48, None),
    "lf8": ("a10g-lf8", "ttft", 48, None),
    "lp": ("a10g-lp", "tpot", 512, None),
    "lp2": ("a10g2-lp", "tpot", 512, None),
    "lt2": ("a10g2-lt", "throughput", 48, None),
    "lf2": ("a10g2-lf", "ttft", 48, None),
    "t42": ("t42-ladder", "ttft", 96, t4_gpu),
    "kv": ("ladder-kv", "throughput", 256, None),
}


def replay_step(
    tmp_path: Path, fixture: str, current: int, history: Sequence[int], *,
    context: AnalysisContext | None = None,
    plans: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
    extra: tuple[str, ...] = (),
    registered: int = 0,
    run_files: bool = False,
    goal: Goal | None = None,
    prompts: bool = False,
) -> tuple[Situation, Suggested]:  # fmt: skip
    """Analyse run ``current`` of a release ladder again under its goal, with the runs
    ``history`` as the project's earlier runs and the cold KV pool the CLI would find; with
    ``extra``, as a run of the same measurement whose engine arguments add those; with
    ``registered``, as a project whose setup is that run's settings; with ``run_files``, the
    fixture's own run directory is this run's, so its metrics scrapes are read; ``goal``
    replaces the ladder's; with ``prompts``, the fixture's prompts are the workload's."""
    directory, ladder_goal, output_tokens, gpu = LADDERS[fixture]
    goal = goal or ladder_goal
    root = REPLAYS / directory
    runs = sorted((root / "runs").iterdir())
    project = tmp_path / REPLAYED / fixture
    (project / "runs").mkdir(parents=True)
    (project / "project.json").write_text((root / "project.json").read_text())
    for index in history:
        copied = project / "runs" / runs[index].name
        shutil.copytree(runs[index], copied)
        if plans is not None and index in plans:  # the plan this code made for that run
            report = json.loads((copied / "report.json").read_text())
            report["goal"] = {**report.get("goal", {}), "goal": goal, "plan": list(plans[index])}
            (copied / "report.json").write_text(json.dumps(report))
    run_dir = runs[current]
    record = json.loads((run_dir / "run.json").read_text())
    measurement = measurement_from_metrics(json.loads((run_dir / "metrics.json").read_text()))
    settings = Settings.from_json(record["settings"])
    base = Settings.from_json(json.loads((runs[registered] / "run.json").read_text())["settings"])
    diagnosis = classify(measurement, settings, None, engine=VLLM, version="0.30.0",
                         gpu_state=gpu() if gpu else None)  # fmt: skip
    texts = tuple(
        json.loads(line)["text"] for line in (root / "prompts.jsonl").read_text().splitlines()
    ) if prompts else ()  # fmt: skip
    facts = WorkloadFacts(prompts=texts, output_tokens=output_tokens, context_limit=32768)
    situation = Situation(measurement=measurement, facts=facts, settings=settings,
                          diagnosis=diagnosis)  # fmt: skip
    args = frozenset(record["engine_args"]) | set(extra)
    past = past_runs(project, VLLM)
    cold = cold_kv_tokens(past, args, cap_flag(VLLM))
    pools = cold_pools(past, args, cap_flag(VLLM))
    context = context or AnalysisContext()
    context = dataclasses.replace(
        context, goal=goal, engine_args=tuple(sorted(args)), cold_kv_tokens=cold,
        cold_pools=pools,
        flags=("--goal", goal) if goal != "throughput" else (),
        new_run=run_dir if run_files else context.new_run,
    )  # fmt: skip
    with active(context):
        return situation, suggestions(
            VLLM, situation, current=base, project_path=project, version="0.30.0"
        )


TESTED = re.compile(r"about \d+ min on this setup \(this run: (\d+) s to load, ([\d.]+) min")


def recorded_time(run_dir: Path) -> tuple[float, float]:
    """(load s, warm-up and measure s) as the run's own report records them."""
    report = json.loads((run_dir / "report.json").read_text())
    text = next(b["test"] for s in report["sections"] for b in s["blocks"] if b.get("test"))
    found = TESTED.search(text)
    assert found is not None, text
    return float(found[1]), float(found[2]) * 60


def test_after_a_cap_halving_the_test_time_is_within_a_quarter(tmp_path: Path) -> None:
    runs = sorted((REPLAYS / "a10g-lp" / "runs").iterdir())
    for current in range(3):
        load, measure = recorded_time(runs[current])
        context = AnalysisContext(launched_s=0.0, load_s=load, measured_s=load + measure)
        situation, shown = replay_step(
            tmp_path / f"lp{current}", "lp", current, range(current), context=context
        )
        lower = next(s for s in shown.suggestions if s.entry.name == "lower-concurrency")
        assert lower.settings is not None
        said = seconds_to_test(
            context,
            cap_now=situation.settings.max_concurrent_requests,
            cap_new=lower.settings.max_concurrent_requests,
            at_cap=running_at_cap(situation.measurement, situation.settings),
        )
        actual = sum(recorded_time(runs[current + 1]))
        assert said is not None and abs(said - actual) / actual <= 0.25
        assert "lower-concurrency" in shown.test_times


def test_the_a10g_ladders_take_a_kv_sized_step_where_they_stopped(tmp_path: Path) -> None:
    # lt run 1 (cap 16, cold, KV 46% of 70,912 tokens) and lf2 run 0 (cap 16, cold, 46% of
    # 70,656): the release runs stopped, with the doubling raise blocked. lt run 0 (cap 8, KV 23%
    # of 71,056): the doubling offers 16, the sized cap is 29, so only the sized raise leads
    for fixture, current, history in (("lt", 1, [0]), ("lf2", 0, []), ("lt", 0, [])):
        situation, shown = replay_step(tmp_path / f"{fixture}{current}", fixture, current, history)
        top = first_try(situation.diagnosis, shown)
        assert top is not None and top.entry.name == "kv-sized-concurrency"
        assert command_args(top.command or "") >= {"max-num-seqs=29"}
        assert shown.stop is None
        assert not any(s.entry.name == "raise-concurrency" and s.tier == "try_first"
                       for s in shown.suggestions)  # fmt: skip
    # the second A10G ladder's lt run 1 (cap 29): prompt work at 85% of the prompt rate, and no
    # larger budget measured
    _, lt1 = replay_step(tmp_path / "lt2-1", "lt2", 1, [0])
    assert not any(s.entry.name == "raise-prefill-batch" and s.tier == "try_first"
                   for s in lt1.suggestions)  # fmt: skip
    assert lt1.stop is not None and "headroom for scheduling changes" in lt1.stop
    assert "token budget, `raise-prefill-batch` is under Could help" in lt1.stop
    assert "no change in this engine's playbook applies" not in lt1.stop
    situation, lf8 = replay_step(tmp_path / "lf8-2", "lf8", 2, [0, 1])  # same pool as lf2
    top = first_try(situation.diagnosis, lf8)
    assert top is not None and top.entry.name == "kv-sized-concurrency"
    assert command_args(top.command or "") >= {"max-num-seqs=37"}
    situation, lf2 = replay_step(tmp_path / "lf2-2", "lf2", 2, [0, 1])  # cold, cap 32, budget 2048
    top = first_try(situation.diagnosis, lf2)
    assert top is not None and top.entry.name == "kv-sized-concurrency"
    assert command_args(top.command or "") >= {"max-num-seqs=37"}  # 85% of its own 88,944 tokens
    _, tpot = replay_step(tmp_path / "lp0", "lp", 0, [])  # a raised cap worsens TPOT
    memory = next(s for s in tpot.suggestions if s.entry.name == "more-kv-memory")
    assert not any(arg.startswith("max-num-seqs=") for arg in command_args(memory.command or ""))


def replanned(tmp_path: Path, fixture: str, count: int) -> list[Suggested]:
    """Runs 0..count-1 of a ladder analysed in order, each with the runs before it as history,
    and each run's report.json given the plan this code makes, so the next run checks it."""
    shown = []
    plans: dict[int, Sequence[Mapping[str, Any]]] = {}
    for current in range(count):
        _, step = replay_step(tmp_path / f"{fixture}{current}", fixture, current, range(current),
                              plans=plans)  # fmt: skip
        shown.append(step)
        plans[current] = step.plan
    return shown


def test_the_a10g_prediction_checks_confirm_or_say_still_binds(tmp_path: Path) -> None:
    ladders = {fixture: replanned(tmp_path / fixture, fixture, count)
               for fixture, count in (("lt", 2), ("lf8", 3), ("lp", 4))}  # fmt: skip
    checks = [s.check for ladder in ladders.values() for s in ladder if s.check]
    assert len(checks) >= 3
    assert all(check["confirmed"] or check["held"] for check in checks)
    assert ladders["lf8"][2].stop is None


def test_the_tpot_ladder_stops_at_the_knee_after_its_second_halving(tmp_path: Path) -> None:
    # lp: 32 -> 16 gave TPOT p95 -12.7% for req/s -43% (0.30); 16 -> 8 gave -10.5% for -44% (0.24)
    _, kept = replay_step(tmp_path / "lp1", "lp", 1, [0])
    assert any(s.entry.name == "lower-concurrency" and s.tier == "try_first"
               for s in kept.suggestions)  # fmt: skip
    _, knee = replay_step(tmp_path / "lp2", "lp", 2, [0, 1])
    lower = next(s for s in knee.suggestions if s.entry.name == "lower-concurrency")
    assert lower.tier == "could_help" and "knee" in (lower.cost or "")
    assert knee.stop is not None and "the knee is reached" in knee.stop
    # a later run that changes only another flag: the knee at max-num-seqs=8 still holds
    _, held = replay_step(tmp_path / "lp2b", "lp", 2, [0, 1, 2], extra=("speculative-config=x",))
    lower = next(s for s in held.suggestions if s.entry.name == "lower-concurrency")
    assert lower.tier == "could_help" and "the step (max-num-seqs=8" in (lower.cost or "")
    # cap 8 plus speculation: dropping it goes back to the cap-8 run, measured, so Try first ends
    _, back = replay_step(tmp_path / "lp2-3", "lp2", 3, [0, 1, 2])
    assert back.stop is not None and back.memory is not None
    assert "drop-speculation" in back.memory.tried and back.stop.startswith("Every change")
    assert not any("drop-speculation" in line for line in back.memory.lines)
    assert "max-num-seqs=8 tried in run 20261011T061548Z-36ac" in "\n".join(back.memory.lines)
    # a project registered with speculation on: no command can drop it, so the stop still says why
    _, net = replay_step(tmp_path / "lp2-net", "lp2", 3, [0, 1, 2], registered=3)
    drop = next(s for s in net.suggestions if s.entry.name == "drop-speculation")
    assert drop.tier == "try_first" and drop.command is None and net.stop is not None
    assert knee.memory is not None and knee.memory.lines[0].startswith("since run ")
    assert knee.trade is not None and knee.trade["since"] == "20261010T224118Z-02ab"


def test_the_t4_ttft_ladder_goes_512_then_async_off_then_stops(tmp_path: Path) -> None:
    situation, at_2048 = replay_step(tmp_path / "2048", "t42", 1, [0], run_files=True)
    top = first_try(situation.diagnosis, at_2048)
    assert top is not None and top.entry.name == "small-step-budget"
    assert command_args(top.command or "") >= {"max-num-batched-tokens=512"}
    situation, at_512 = replay_step(tmp_path / "512", "t42", 2, [0, 1], run_files=True)
    top = first_try(situation.diagnosis, at_512)
    assert top is not None and top.entry.name == "no-async-scheduling"
    assert "ms per request, about " in top.reason
    assert "TPOT worsens" in (top.cost or top.entry.costs)
    assert command_args(top.command or "") >= {"no-async-scheduling"}
    # 182 prompt tokens with 128 running decodes do not fit a 256-token step
    assert not any(s.entry.name == "small-step-budget" for s in at_512.suggestions)
    _, async_off = replay_step(tmp_path / "off", "t42", 4, [1, 2], run_files=True)
    assert not any(s.tier == "try_first" for s in async_off.suggestions)
    assert async_off.stop is not None and "is about 3.5 decode steps of 122 ms" in async_off.stop
    _, throughput = replay_step(
        tmp_path / "tp", "t42", 2, [0, 1], run_files=True, goal="throughput"
    )
    assert not any(s.entry.name == "no-async-scheduling" for s in throughput.suggestions)


# --- changes that can affect answers ---

A10G_OPTIONS = AnswerOptions(
    engine_version="0.30.0", capability=8.6, on_the_fly={"fp8": "weight-only (W8A16)"}, kv_cache=(),
    checkpoints=(Format("int8 W8A8", "compressed-tensors", 8, "W8A8"),
                 Format("fp8", "fp8", 8, "weight-only (W8A16)"),
                 Format("AWQ int4", "awq", 4, "W4A16"), Format("GPTQ int4", "gptq", 4, "W4A16"),
                 Format("W4A16", "compressed-tensors", 4, "W4A16")),
    unjudged=(), plugins={"vllm-bnb-plugin": None, "vllm-gguf-plugin": None},
)  # fmt: skip


def test_a_decode_bound_run_offers_8_bit_weights_after_the_safe_change_with_its_warning(
    tmp_path: Path,
) -> None:
    context = AnalysisContext(answer_options=A10G_OPTIONS)
    situation, shown = replay_step(tmp_path, "lp", 0, [], context=context)
    names = [s.entry.name for s in shown.suggestions if s.tier == "try_first"]
    assert names.index("lower-concurrency") < names.index("fp8-weights")
    fp8 = next(s for s in shown.suggestions if s.entry.name == "fp8-weights")
    assert command_args(fp8.command or "") >= {"quantization=fp8", "linear-backend=marlin"}
    assert "the maths stays 16-bit" in fp8.reason
    top = first_try(situation.diagnosis, shown)
    assert top is not None and top.entry.name == "lower-concurrency"
    assert not any(s.entry.name == "bnb-4bit-weights" for s in shown.suggestions)  # no plugin
    checkpoint = next(s for s in shown.suggestions if s.entry.name == "quantized-checkpoint")
    assert checkpoint.command is None
    assert checkpoint.reason.index("8-bit") < checkpoint.reason.index("4-bit")
    assert answer_warning(fp8.entry, retained=True).startswith(" (answers may change")


@pytest.mark.parametrize(
    "context",
    [
        AnalysisContext(answer_options=dataclasses.replace(A10G_OPTIONS, capability=7.5)),
        AnalysisContext(gpu_capability=8.6),  # the support read failed
    ],
)
def test_an_fp8_kv_cache_is_not_offered_where_the_gpu_cannot_run_it(
    tmp_path: Path, context: AnalysisContext
) -> None:
    situation, shown = replay_step(tmp_path, "kv", 1, [0], context=context)
    assert not any(s.entry.name == "fp8-kv-cache" for s in shown.suggestions)
    assert any(name == "fp8-kv-cache" and "8.9" in why for name, why in shown.not_applicable)
    forced = VLLM.with_lever(situation.settings, "kv_cache_precision", "8bit")
    with active(context):  # a run that forces it anyway is carried: no keep, no change on top
        carried = carried_gate(
            VLLM.playbook(),
            dataclasses.replace(situation, settings=forced),
            situation.settings,
            VLLM,
        )
    assert carried is not None and "8.9" in carried


# --- the n-gram gate ---


def test_ngram_speculation_says_to_measure_it(tmp_path: Path) -> None:
    run0 = AnalysisContext(new_run=REPLAYS / "a10g2-lp" / "runs" / "20261011T060049Z-92cd")
    _, shown = replay_step(tmp_path, "lp2", 0, [], context=run0, prompts=True, goal="throughput")
    ngram = next(s for s in shown.suggestions if s.entry.name == "ngram-speculation")
    assert ngram.tier == "could_help"
    assert "drafts would supply 12.7% of all output tokens" in ngram.reason
    assert "is needed" not in ngram.reason
    assert ngram.reason.endswith("; measure it: the step overhead on this model is not known")
    assert ngram.tail_coverage is not None


@pytest.mark.parametrize(
    ("goal", "pooled", "prompt_bound", "left_out"),
    [
        ("tpot", 0.05, False, "drafts would supply 5.0% of the output; below 6% speculation"),
        ("throughput", 0.05, False, "drafts would supply 5.0% of the output; below 6% speculation"),
        ("ttft", 0.05, False, None),
        ("ttft", 0.2, True, "take step budget before waiting prompts"),
    ],
)
def test_ngram_speculation_is_left_out_below_its_coverage_and_on_a_prompt_bound_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    goal: Goal,
    pooled: float,
    prompt_bound: bool,
    left_out: str | None,
) -> None:
    run0 = AnalysisContext(new_run=REPLAYS / "a10g2-lp" / "runs" / "20261011T060049Z-92cd")
    monkeypatch.setattr("tensward.suggest.coverage", lambda pairs: Coverage(pooled, pooled))
    monkeypatch.setattr("tensward.suggest.prompt_bound", lambda state: prompt_bound)
    _, shown = replay_step(tmp_path, "lp2", 0, [], context=run0, prompts=True, goal=goal)
    offered = [s.entry.name for s in shown.suggestions]
    if left_out is None:
        assert "ngram-speculation" in offered
    else:
        assert "ngram-speculation" not in offered
        assert left_out in shown.left_out_why["ngram-speculation"]


def test_a_speculating_run_cannot_tell_whether_decode_bandwidth_binds(tmp_path: Path) -> None:
    run3 = REPLAYS / "a10g2-lp" / "runs" / "20261011T062943Z-5805"  # cap 8 with n-gram speculation
    args = json.loads((run3 / "run.json").read_text())["engine_args"]
    plan = [{"change": "ngram-speculation", "args": args, "limit": "decode_bandwidth",
             "signal": None, "tested": True}]  # fmt: skip
    _, shown = replay_step(tmp_path, "lp2", 3, [0, 1, 2], plans={2: plan})
    check = shown.check
    assert check is not None and check["cant_tell"] and not check["confirmed"] and not check["held"]
    assert shown.memory is not None
    assert any("can't tell, as Tensward does not model the decode ceiling while speculating" in line
               for line in shown.memory.lines)  # fmt: skip


def test_the_trade_line_starts_at_the_latest_run_of_the_starting_settings(tmp_path: Path) -> None:
    _, at_512 = replay_step(tmp_path / "512", "t42", 2, [0, 1], run_files=True)
    assert at_512.trade is not None and at_512.trade["since"] == "20261011T072230Z-5c00"
    assert "wave" not in at_512.trade  # 2.26 s against 1.90 s: under a fifth
    _, at_256 = replay_step(tmp_path / "256", "t42", 3, [0, 1, 2], run_files=True)
    assert at_256.trade is not None and at_256.memory is not None
    wave = at_256.trade["wave"]
    assert wave["after_ms"] > 1.2 * wave["before_ms"]
    assert at_256.memory.lines[1].startswith("the start-up wave (128 requests sent at once")


def test_a_tpot_ladder_stops_lowering_the_cap_past_the_declared_floor(tmp_path: Path) -> None:
    floor = AnalysisContext(max_req_s_loss=0.2)
    _, lp1 = replay_step(tmp_path / "lp1", "lp", 1, [0], context=floor)
    lower = next(s for s in lp1.suggestions if s.entry.name == "lower-concurrency")
    assert lower.tier == "could_help"
    assert "past your floor (predicted -45%)" in (lower.cost or "")
    assert lp1.floor is not None and lp1.floor["crossed"]
    assert lp1.memory is not None and lp1.memory.revert is not None
    assert "past your floor of 20%" in lp1.memory.revert.reason
    # registered with speculation on: the runs within the floor without it can change answers,
    # so the floor's revert goes to one only when it is the kept setup
    cap16 = Kept(run="20261011T060653Z-474d", engine_args=("max-num-seqs=16",),
                 answer_changes=("speculation",), previous=REGISTERED, snapshot_id="")  # fmt: skip
    for kept in (None, cap16):
        _, net = replay_step(tmp_path / f"net-{kept is None}", "lp2", 3, [0, 1, 2], registered=3,
                             context=AnalysisContext(max_req_s_loss=0.5, kept=kept))  # fmt: skip
        assert net.floor is not None and net.floor["crossed"] and net.memory is not None
        revert = net.memory.revert
        if kept is None:
            assert revert is None
        else:
            assert revert is not None and command_args(revert.command or "") == {"max-num-seqs=16"}


L4_STEPS = REPLAYS / "l4-steps"
DEFAULTS = ("20261014T162451Z-db0f", "20261014T164058Z-8255", "20261014T170632Z-8be6")


def _cut(run: str, into: Path, upto: int = 236) -> Path:
    """A copy of the run's directory holding only its first ``upto`` requests: a planned step."""
    (into / run).mkdir(parents=True)
    rows = (L4_STEPS / run / "requests.jsonl").read_text().splitlines()[:upto]
    (into / run / "requests.jsonl").write_text("\n".join(rows) + "\n")
    return into / run


def test_three_launches_of_one_setting_are_within_noise_with_narrow_bands(tmp_path: Path) -> None:
    for a, b in itertools.combinations(DEFAULTS, 2):
        deltas = window.compare_matched(_cut(a, tmp_path / b), L4_STEPS / b, clients=16)
        assert not isinstance(deltas, str)
        for metric in ("req_s", "tpot_p95_ms", "ttft_p95_ms"):
            assert deltas[metric].verdict == "within_noise", (a, b, deltas[metric])
        for metric in ("req_s", "tpot_p95_ms"):
            assert deltas[metric].band <= window.target(metric), (a, b, deltas[metric])


def test_a_planned_run_with_its_drain_is_within_noise_of_full_runs_of_its_setting() -> None:
    planned = L4_STEPS / "20261015T065602Z-33bf"
    for full in DEFAULTS:
        deltas = window.compare_matched(planned, L4_STEPS / full, clients=16)
        assert not isinstance(deltas, str)
        assert all(d.verdict == "within_noise" for d in deltas.values()), (full, deltas)
        for metric in ("req_s", "tpot_p95_ms"):
            assert deltas[metric].band <= window.target(metric), (full, deltas[metric])


def test_async_off_is_a_loss_beyond_noise_on_a_60_s_window(tmp_path: Path) -> None:
    deltas = window.compare_matched(
        _cut("20261014T165306Z-eaf7", tmp_path), L4_STEPS / "20261014T164058Z-8255", clients=16
    )
    assert not isinstance(deltas, str)
    tpot, rate = deltas["tpot_p95_ms"], deltas["req_s"]
    assert tpot.verdict == "worse" and 0.03 < tpot.change < 0.08
    assert rate.verdict == "worse" and -0.06 < rate.change < -0.02
    assert deltas["ttft_p95_ms"].change < -0.04  # async off cuts TTFT on the same prompts


def test_a_60_s_window_is_planned_with_or_without_a_prediction() -> None:
    for predicted in (-0.2, None):
        plan = window.plan(
            3.37, clients=16, configured=1032, metric="tpot_p95_ms", predicted=predicted
        )
        assert plan.seconds == 60 and 230 <= plan.requests <= 240


def test_a_run_short_of_kv_memory_plans_no_short_window() -> None:
    kv = REPLAYS / "l4-kv"
    reference, planned = kv / "20261015T125727Z-2dda", kv / "20261015T131927Z-84d3"
    deltas = window.compare_matched(planned, reference, clients=36)
    assert not isinstance(deltas, str)
    assert deltas["req_s"].verdict == "better" and deltas["req_s"].change > 0.10
    workload = SimpleNamespace(
        request_count=200, arrival=ArrivalSpec(kind="closed_loop", concurrency=36)
    )
    for only in (None, 60.0):
        plan = planned_window(reference, workload, goal="throughput", only=only)
        assert (plan.requests, plan.seconds) == (200, None)
        assert plan.reason == (
            "run 20261015T125727Z-2dda was short of KV memory (81 preemptions, KV cache at 100% "
            "at its peak): a short window would not reach the steady queue"
        )
    workload = SimpleNamespace(
        request_count=1032, arrival=ArrivalSpec(kind="closed_loop", concurrency=16)
    )
    assert planned_window(L4_STEPS / "20261014T164058Z-8255", workload, goal="tpot").seconds == 60


L4_PARENT = "20261014T164058Z-8255"
L4_TRACK = {  # the L4 tracks' declared limits and floor, by goal
    "ttft": {"ttft_p95_limit_ms": None, "tpot_ms": 82.377},
    "tpot": {"ttft_p95_limit_ms": 723.551, "tpot_ms": None},
    "throughput": {"ttft_p95_limit_ms": None, "tpot_ms": None},
}


def _scaled(rows: Sequence[str], tpot: float, ttft: float = 1.0) -> list[str]:
    """``rows`` with the time to each first token scaled by ``ttft`` and the time after it by
    ``tpot``."""
    out = []
    for row in rows:
        record = json.loads(row)
        sent, first, end = record["dispatch_ns"], record["first_content_ns"], record["terminal_ns"]
        moved = sent + round((first - sent) * ttft)
        out.append(json.dumps({
            **record, "first_content_ns": moved, "terminal_ns": moved + round((end - first) * tpot)
        }))  # fmt: skip
    return out


def l4_keep(
    tmp_path: Path, goal: Goal, rows: Sequence[str], args: tuple[str, ...], *,
    adopt: bool = False,
) -> tuple[Suggested, dict[str, str], Mapping[str, Any]]:  # fmt: skip
    """A planned step of the L4 cell (``rows``, measured with ``args``) against the full
    default run it was made from: what it offers, its keep section and its report's goal. The
    step's point values are the default's, so the floor and the limits hold."""
    project = tmp_path / REPLAYED / goal
    shutil.copytree(L4_STEPS / L4_PARENT, project / "runs" / L4_PARENT)
    step = project / "runs" / "20261014T180000Z-step"
    step.mkdir()
    (step / "requests.jsonl").write_text("\n".join(rows) + "\n")
    track = L4_TRACK[goal]
    situation, context = l4_situation(L4_PARENT, goal, ttft_p95_limit_ms=track["ttft_p95_limit_ms"])
    context = dataclasses.replace(
        context, project=project, new_run=step, engine_args=args, adopt_within_noise=adopt,
        flags=("--goal", goal, *(("--adopt-within-noise",) if adopt else ())),
    )  # fmt: skip
    slo = Slo(track["ttft_p95_limit_ms"] or math.inf, track["tpot_ms"] or math.inf)
    record = json.loads((L4_STEPS / L4_PARENT / "run.json").read_text())
    setup = CurrentSetup(source="defaults", settings=record["settings"])
    with active(context):
        suggested = suggestions(
            VLLM, situation, current=situation.settings, project_path=project, version="0.30.0"
        )
        report = build_report(RunOutputs(
            run_id=step.name, ran="vLLM 0.30.0 on NVIDIA L4", measurement=situation.measurement,
            diagnosis=situation.diagnosis, suggested=suggested, rows=(), facts=situation.facts,
            settings=situation.settings, subject=Subject(setup=setup), image=None, slo=slo,
            engine=VLLM, gpu="NVIDIA L4", source="", engine_args=args, retained=True,
        ))  # fmt: skip
    keep = next(s for s in report.sections if s.id == "keep")
    shown = {b.key: b.text for b in keep.blocks if isinstance(b, Text)}
    return suggested, shown, report.goal or {}


def test_async_off_on_a_60_s_window_is_kept_under_ttft_and_a_loss_under_tpot(
    tmp_path: Path,
) -> None:
    rows = (L4_STEPS / "20261014T165306Z-eaf7" / "requests.jsonl").read_text().splitlines()
    ttft, keep, _ = l4_keep(tmp_path, "ttft", rows[:236], ("no-async-scheduling",))
    assert "not_kept" not in keep and "kept_within_noise" not in keep
    assert ttft.memory is not None and ttft.memory.parent is not None
    deltas = ttft.memory.parent.deltas
    assert deltas["ttft_p95_ms"].verdict == "better"
    assert deltas["ttft_p95_ms"].change == pytest.approx(-0.085, abs=0.005)
    assert deltas["ttft_p95_ms"].band == pytest.approx(0.079, abs=0.005)
    assert ttft.memory.lines[0] == (
        f"since run {L4_PARENT} (the latest run of your starting settings), on the 236 "
        "prompts both runs sent: TTFT p95 −8% ±8% (better); full answer p95 +4% ±5% (within "
        "noise); requests per second −4% ±3% (worse); TPOT p95 +5% ±3% (worse)"
    )
    assert ttft.memory.lines[1].startswith("TTFT p95 of the middle 80% −6% ±")
    assert keep["trade"] == (
        f"Measured since run {L4_PARENT}: gains: TTFT p95 −8% ±8%; costs: requests per second "
        "−4% ±3%, TPOT p95 +5% ±3%; within noise: full answer p95 +4% ±5%."
    )
    _, keep, _ = l4_keep(tmp_path, "tpot", rows[:236], ("no-async-scheduling",))
    assert keep["not_kept"] == (
        "Tensward does not recommend keeping this change: a loss: TPOT p95 +5% ±3%."
    )
    assert keep["command"].startswith("If you still want it: `tensward set ")


def test_a_step_beyond_its_band_is_kept_and_one_within_it_is_not(tmp_path: Path) -> None:
    rows = (L4_STEPS / L4_PARENT / "requests.jsonl").read_text().splitlines()[:236]
    faster, keep, _ = l4_keep(tmp_path / "fast", "tpot", _scaled(rows, 0.8), ("max-num-seqs=8",))
    assert "not_kept" not in keep
    assert faster.memory is not None and faster.memory.parent is not None
    assert faster.memory.parent.deltas["tpot_p95_ms"].change == pytest.approx(-0.2, abs=0.001)
    assert "TPOT p95 −20% ±3%" in keep["trade"]
    _, keep, _ = l4_keep(tmp_path / "same", "throughput", rows, ("max-num-seqs=8",))
    found = re.fullmatch(
        r"Tensward does not recommend keeping this change: within noise: requests per second "
        r"\+0% ±(\d)%\.",
        keep["not_kept"],
    )
    assert found is not None and int(found[1]) < 4, keep["not_kept"]


def test_the_within_noise_policy_adopts_a_better_point_unless_a_guarded_metric_lost(
    tmp_path: Path,
) -> None:
    rows = (L4_STEPS / L4_PARENT / "requests.jsonl").read_text().splitlines()[:236]
    args = ("max-num-seqs=8",)
    _, keep, goal = l4_keep(tmp_path / "a", "throughput", _scaled(rows, 0.99), args, adopt=True)
    assert keep["kept_within_noise"].startswith("Within noise (requests per second +1% ±3%); ")
    assert goal["adopted_within_noise"]["policy"] == "flag"
    _, keep, goal = l4_keep(tmp_path / "b", "throughput", _scaled(rows, 0.99), args)
    assert keep["not_kept"].endswith("within noise: requests per second +1% ±3%.")
    assert "adopted_within_noise" not in goal
    slower = _scaled(rows, 1.05, ttft=0.97)
    _, keep, goal = l4_keep(tmp_path / "c", "ttft", slower, args, adopt=True)
    assert keep["not_kept"].endswith("within noise: TTFT p95 −3% ±4%.")
    assert "adopted_within_noise" not in goal


@pytest.mark.parametrize(
    ("run", "init", "compile_s", "gap"),
    [
        ("20261014T162451Z-db0f", 173.32, 44.66, True),
        ("20261014T164058Z-8255", 173.97, 45.12, True),
        ("20261014T165306Z-eaf7", 65.18, 43.77, False),
        ("20261014T170632Z-8be6", 173.90, 45.00, True),
    ],
)
def test_the_start_times_come_from_the_server_log(
    run: str, init: float, compile_s: float, gap: bool
) -> None:
    started = startup((REPLAYS / "l4-steps" / run / "server.log").read_text())
    assert started["init_engine_s"] == pytest.approx(init, abs=0.1)
    assert started["compile_s"] == pytest.approx(compile_s, abs=0.1)
    assert started["captures"] == 2
    assert started["compile_cache_warm"] is False
    assert (started["warmup_gap_s"] > 100) is gap
    assert gap or started["warmup_gap_s"] < 15


def recorded_situation(run_dir: Path, facts: WorkloadFacts) -> Situation:
    """A recorded run diagnosed again as ``diagnose_run`` did, with its stored GPU state."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    record = json.loads((run_dir / "run.json").read_text())
    measurement = measurement_from_metrics(metrics)
    settings = Settings.from_json(record["settings"])
    version, gpu = record["engine_version"], record["gpus_identity"][0]["name"]
    diagnosis = classify(
        measurement, settings, None, engine=VLLM, version=version,
        gpu_state=GpuState(**metrics["diagnosis"]["gpu_state"]),
        profile=profile_for(VLLM, record["platform"], gpu, version),
    )  # fmt: skip
    assert diagnosis.primary == metrics["diagnosis"]["primary"]
    return Situation(measurement=measurement, facts=facts, settings=settings, diagnosis=diagnosis)


def recorded_options(run_dir: Path) -> AnswerOptions:
    options = json.loads((run_dir / "run.json").read_text())["answer_options"]
    return AnswerOptions(
        engine_version=options["engine_version"],
        capability=options["capability"],
        on_the_fly=options["on_the_fly"],
        kv_cache=tuple(options["kv_cache"]),
        checkpoints=tuple(Format(**f) for f in options["checkpoints"]),
        unjudged=(),
        plugins=options["plugins"],
    )


def l4_situation(
    run: str,
    goal: Goal,
    *,
    model_type: str = "qwen3",
    ttft_p95_limit_ms: float | None = 723.551,
    max_req_s_loss: float | None = 0.10,
) -> tuple[Situation, AnalysisContext]:
    """An L4 run's situation and the context of the L4 track for ``goal`` (under
    throughput the driver declares a TPOT limit and no floor)."""
    context = AnalysisContext(
        goal=goal,
        ttft_p95_limit_ms=ttft_p95_limit_ms,
        max_req_s_loss=None if goal == "throughput" else max_req_s_loss,
        linear_weight_bytes=2 * 440_400_000,
        answer_options=recorded_options(L4_STEPS / run),
        tpot_p95_limit_ms=82.377 if goal == "throughput" else None,
    )
    facts = WorkloadFacts(prompts=(), output_tokens=256, context_limit=8192, model_type=model_type)
    with active(context):
        situation = recorded_situation(L4_STEPS / run, facts)
    assert situation.diagnosis.primary == "long_context"
    return situation, context


A10G_TRAFFIC = REPLAYS / "a10g-traffic"
PHI35_LINEAR_BYTES = 2 * 3_620_000_000


def _a10g(run: str) -> tuple[Situation, dict[str, float]]:
    situation = recorded_situation(A10G_TRAFFIC / run, T4_FACTS)
    m = situation.measurement
    assert m.tpot_p95_ms is not None and m.request_throughput is not None
    return situation, {"tpot": m.tpot_p95_ms, "req_s": m.request_throughput}


def _change(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    return {key: after[key] / before[key] - 1 for key in before}


def test_the_traffic_model_predicts_the_a10g_fp8_and_cap_steps() -> None:
    lp0, before = _a10g("20261012T013850Z-ae53")
    for run, entry, proposed, measured_only in (
        ("20261012T015732Z-6d98", "fp8-weights", {"quantization": "fp8"}, ()),
        ("20261012T020727Z-147f", "lower-concurrency", {"max_concurrent_requests": 16}, ("req_s",)),
    ):
        step, after = _a10g(run)
        measured = _change(before, after)
        prediction = predict(
            lp0, dataclasses.replace(lp0.settings, **proposed), entry=entry,
            linear_bytes=PHI35_LINEAR_BYTES,
        )  # fmt: skip
        assert prediction is not None and "prompt share" not in prediction.text
        figures = {"tpot": prediction.tpot, "req_s": prediction.req_s}
        for key in measured_only or figures:
            assert figures[key] == pytest.approx(measured[key], abs=0.03), (entry, key)
    qb = recorded_situation(A10G_TRAFFIC / "20261012T021303Z-535d", T4_FACTS)
    assert qb.measurement.prompt_work_share is None
    fp8 = dataclasses.replace(qb.settings, quantization="fp8")
    assert predict(qb, fp8, entry="fp8-weights", linear_bytes=PHI35_LINEAR_BYTES) is None


def test_on_the_l4_cell_the_share_bound_predicts_both_fp8_controls() -> None:
    situation, context = l4_situation("20261014T164058Z-8255", "tpot")
    controls = json.loads((L4_STEPS / "controls.json").read_text())
    keys = {"tpot": "tpot_p95_ms", "req_s": "req_s"}
    default = {k: sum(r[key] for r in controls["default"]) / 3 for k, key in keys.items()}
    for control, entry, proposed in (
        ("fp8w", "fp8-weights", {"quantization": "fp8"}),
        ("fp8kv", "fp8-kv-decode", {"kv_cache_dtype": "fp8"}),
    ):
        measured = _change(default, {k: controls[control][key] for k, key in keys.items()})
        prediction = predict(
            situation, dataclasses.replace(situation.settings, **proposed), entry=entry,
            linear_bytes=context.linear_weight_bytes,
        )  # fmt: skip
        assert prediction is not None and "prompt share at most 27%" in prediction.text
        assert prediction.tpot == pytest.approx(measured["tpot"], abs=0.05), entry
        assert prediction.req_s == pytest.approx(measured["req_s"], abs=0.05), entry
    lower = dataclasses.replace(situation.settings, max_concurrent_requests=8)
    assert predict(situation, lower, entry="lower-concurrency", linear_bytes=None) is None
    kv_short = dataclasses.replace(
        situation,
        diagnosis=dataclasses.replace(
            situation.diagnosis,
            findings=tuple(
                dataclasses.replace(f, state="critical") if f.bottleneck == "kv_capacity" else f
                for f in situation.diagnosis.findings
            ),
        ),
    )
    fp8_kv = dataclasses.replace(situation.settings, kv_cache_dtype="fp8")
    for entry in ("fp8-kv-cache", "fp8-kv-decode"):
        assert predict(kv_short, fp8_kv, entry=entry, linear_bytes=None) is None


L4_8255 = "20261014T164058Z-8255"
FP8_KV = {"fp8-kv-cache", "fp8-kv-decode"}


def _l4_offers(situation: Situation, context: AnalysisContext, project: Path) -> Suggested:
    with active(context):
        return suggestions(
            VLLM, situation, current=situation.settings, project_path=project, version="0.30.0"
        )


@pytest.mark.parametrize("goal", ["tpot", "throughput"])
def test_fp8_kv_is_offered_where_decode_reads_mostly_kv(goal: Goal, tmp_path: Path) -> None:
    situation, context = l4_situation(L4_8255, goal)
    offered = _l4_offers(situation, context, tmp_path)
    section = next_steps_section(situation.diagnosis, offered, retained=False)
    blocks = to_json(Report(run_id="r", ran="now", sections=(section,)))["sections"][0]["blocks"]
    block = next(b for b in blocks if b.get("name") == "fp8-kv-decode")
    assert block["tier"] == "try_first" and block["quality_risk"]
    command = f"tensward analyse --project {tmp_path} --engine-arg kv-cache-dtype=fp8"
    assert block["command"] == command
    assert "7.9x" in block["reason"]


@pytest.mark.parametrize(
    ("goal", "offered"), [("throughput", "fp8-kv-cache"), ("tpot", "fp8-kv-decode")]
)
def test_where_kv_memory_is_short_one_fp8_kv_change_is_offered(
    goal: Goal, offered: str, tmp_path: Path
) -> None:
    situation, context = l4_situation(L4_8255, goal)
    metrics = json.loads((L4_STEPS / L4_8255 / "metrics.json").read_text())
    record = json.loads((L4_STEPS / L4_8255 / "run.json").read_text())
    measurement = dataclasses.replace(situation.measurement, peak_kv_usage=0.99, peak_waiting=8.0)
    kv_short = dataclasses.replace(
        situation,
        measurement=measurement,
        diagnosis=classify(
            measurement, situation.settings, None, engine=VLLM, version="0.30.0",
            gpu_state=GpuState(**metrics["diagnosis"]["gpu_state"]),
            profile=profile_for(VLLM, record["platform"], "NVIDIA L4", "0.30.0"),
        ),
    )  # fmt: skip
    assert kv_short.fired("kv_capacity")
    found = _l4_offers(kv_short, context, tmp_path).suggestions
    assert [(s.entry.name, s.tier) for s in found if s.entry.name in FP8_KV] == [
        (offered, "try_first")
    ]


@pytest.mark.parametrize("goal", ["tpot", "throughput"])
def test_fp8_kv_decode_leads_try_first_by_its_predicted_gain(goal: Goal, tmp_path: Path) -> None:
    situation, context = l4_situation(L4_8255, goal)
    offered = _l4_offers(situation, context, tmp_path)
    leading, *rest = [s for s in offered.suggestions if s.tier == "try_first"]
    assert leading.entry.name == "fp8-kv-decode" and "predicted: TPOT" in leading.reason
    assert leading.predicted is not None and not leading.members
    assert leading.predicted.tpot == pytest.approx(-0.325, abs=0.05)
    assert leading.predicted.req_s == pytest.approx(0.481, abs=0.05)
    assert [s.entry.name for s in rest if s.command] == ["fp8-weights"]
    diagnosis = situation.diagnosis
    group = try_first_order(diagnosis, offered)[0]
    assert group_members(diagnosis, offered, group)[0] is leading
    assert first_try(diagnosis, offered) is leading
    section = next_steps_section(diagnosis, offered, retained=False)
    blocks = to_json(Report(run_id="r", ran="now", sections=(section,)))["sections"][0]["blocks"]
    shown = [b for b in blocks if b["kind"] == "suggestion" and b["tier"] == "try_first"]
    assert shown[0]["key"] == "fp8-kv-decode" and shown[0]["quality_risk"]


def test_a_change_predicted_past_the_floor_could_help_and_fp8_weights_leads(
    tmp_path: Path,
) -> None:
    lp0, _ = _a10g("20261012T013850Z-ae53")
    context = AnalysisContext(
        goal="tpot", max_req_s_loss=0.20, linear_weight_bytes=PHI35_LINEAR_BYTES,
        answer_options=recorded_options(A10G_TRAFFIC / "20261012T013850Z-ae53"),
    )  # fmt: skip
    with active(context):
        offered = suggestions(
            VLLM, lp0, current=lp0.settings, project_path=tmp_path, version="0.30.0"
        )
    first = first_try(lp0.diagnosis, offered)
    assert first is not None and first is offered.suggestions[0]
    assert first.entry.name == "fp8-weights"
    lower = next(s for s in offered.suggestions if s.entry.name == "lower-concurrency")
    assert lower.tier == "could_help" and "past your floor (predicted -30%)" in (lower.cost or "")


def _ceilings(situation: Situation, **changes: Any) -> Situation:
    m = situation.measurement
    assert m.ceilings is not None
    ceilings = dataclasses.replace(m.ceilings, **changes)
    return dataclasses.replace(situation, measurement=dataclasses.replace(m, ceilings=ceilings))


def _options(context: AnalysisContext, **changes: Any) -> AnalysisContext:
    assert context.answer_options is not None
    options = dataclasses.replace(context.answer_options, **changes)
    return dataclasses.replace(context, answer_options=options)


@pytest.mark.parametrize(
    ("model_type", "change", "gated"),
    [
        ("qwen2", lambda s, c: (s, c), True),
        ("qwen2", lambda s, c: (s, _options(c, capability=8.6, kv_cache=())), True),
        ("qwen3", lambda s, c: (s, _options(c, kv_cache=())), True),
        ("qwen3", lambda s, c: (dataclasses.replace(
            s, settings=dataclasses.replace(s.settings, kv_cache_dtype="fp8")), c), False),
        ("qwen3", lambda s, c: (_ceilings(s, avg_running_batch=3.0), c), False),
        ("qwen3", lambda s, c: (_ceilings(s, decode_pct_of_ceiling=55.0), c), False),
        ("qwen3", lambda s, c: (dataclasses.replace(
            s, measurement=dataclasses.replace(s.measurement, hybrid_cache=True)), c), False),
    ],
    ids=["qwen2-l4", "qwen2-a10g", "no-fp8-kv-gpu", "kv-8bit", "kv-under-2x", "decode-55",
         "hybrid"],
)  # fmt: skip
def test_no_fp8_kv_offer_where_it_must_not_fire(
    model_type: str,
    change: Callable[[Situation, AnalysisContext], tuple[Situation, AnalysisContext]],
    gated: bool,
    tmp_path: Path,
) -> None:
    situation, context = change(*l4_situation(L4_8255, "tpot", model_type=model_type))
    offered = _l4_offers(situation, context, tmp_path)
    assert not {s.entry.name for s in offered.suggestions} & FP8_KV
    assert ("fp8-kv-decode" in dict(offered.not_applicable)) is gated


@pytest.mark.parametrize(
    ("measurement", "ceilings"),
    [
        ({"spec_acceptance_length": 2.1}, {}),
        ({}, {"moe_experts": (64, 8)}),
        ({"step_ms": None}, {}),
    ],
)
def test_no_traffic_prediction_with_speculation_moe_or_no_step_time(
    measurement: dict[str, Any], ceilings: dict[str, Any]
) -> None:
    lp0, _ = _a10g("20261012T013850Z-ae53")
    m = lp0.measurement
    assert m.ceilings is not None
    changed = dataclasses.replace(
        m, ceilings=dataclasses.replace(m.ceilings, **ceilings), **measurement
    )
    fp8 = dataclasses.replace(lp0.settings, quantization="fp8")
    situation = dataclasses.replace(lp0, measurement=changed)
    assert predict(situation, fp8, entry="fp8-weights", linear_bytes=PHI35_LINEAR_BYTES) is None


def _with_tool_calls(situation: Situation, run: str) -> Situation:
    """The run's tool-call shares, which ``measurement_from_metrics`` leaves out."""
    calls = json.loads((L4_STEPS / run / "metrics.json").read_text())["tool_calls"]
    m = dataclasses.replace(situation.measurement, tool_calls=ToolCallStats(**calls))
    return dataclasses.replace(situation, measurement=m)


FP8_STEPS = (
    ("fp8-kv-decode", {"kv_cache_dtype": "fp8"}, -0.26, -0.18),
    ("fp8-weights", {"quantization": "fp8"}, -0.04, -0.01),
)


def test_on_tool_calls_client_ttft_follows_tpot_by_the_parser_s_held_steps() -> None:
    # client TTFT p50 718 ms, engine 243 ms, TPOT p50 41.2 ms; measured on the same cell: fp8 KV
    # -29% (the engine's own TTFT fell too), fp8 weights -7%
    situation, context = l4_situation(L4_8255, "ttft")
    situation = _with_tool_calls(situation, L4_8255)
    assert held_steps(situation.measurement) == pytest.approx(11.55, abs=0.01)
    for entry, proposed, low, high in FP8_STEPS:
        prediction = predict(
            situation, dataclasses.replace(situation.settings, **proposed), entry=entry,
            linear_bytes=context.linear_weight_bytes,
        )  # fmt: skip
        assert prediction is not None and prediction.ttft is not None
        assert low <= prediction.ttft <= high, entry
        assert "waits 12 decode steps for the tool-call parser" in prediction.text


@pytest.mark.parametrize(
    "change",
    [
        lambda m: dataclasses.replace(m, tool_calls=None),
        lambda m: dataclasses.replace(m, server_ttft_p50_ms=m.ttft_p50_ms),
        lambda m: dataclasses.replace(m, tpot_p50_ms=120.0),
    ],
    ids=["no-tool-calls", "engine-at-client", "held-under-4"],
)
def test_no_ttft_prediction_without_a_parser_hold(
    change: Callable[[Measurement], Measurement],
) -> None:
    situation, _ = l4_situation(L4_8255, "ttft")
    situation = _with_tool_calls(situation, L4_8255)
    situation = dataclasses.replace(situation, measurement=change(situation.measurement))
    fp8_kv = dataclasses.replace(situation.settings, kv_cache_dtype="fp8")
    prediction = predict(situation, fp8_kv, entry="fp8-kv-decode", linear_bytes=None)
    assert prediction is not None and prediction.ttft is None and "TTFT" not in prediction.text


def test_on_a_run_served_as_fp8_the_kv_prediction_reads_the_halved_linear_bytes() -> None:
    # with the prompt share measured: under its datasheet bound the weight bytes cancel out
    situation, context = l4_situation(L4_8255, "tpot")
    measured = dataclasses.replace(situation.measurement, prompt_work_share=0.2)
    situation = dataclasses.replace(situation, measurement=measured)
    linear = context.linear_weight_bytes
    assert linear is not None and situation.measurement.ceilings is not None
    weights = situation.measurement.ceilings.weight_bytes_per_step
    assert weights is not None
    served = dataclasses.replace(
        situation, settings=dataclasses.replace(situation.settings, quantization="fp8")
    )
    halved = _ceilings(situation, weight_bytes_per_step=weights - linear // 2)
    found = [
        predict(s, dataclasses.replace(s.settings, kv_cache_dtype="fp8"),
                entry="fp8-kv-decode", linear_bytes=linear)
        for s in (served, halved, situation)
    ]  # fmt: skip
    assert all(p is not None for p in found)
    on_fp8, on_halved, on_bf16 = [(p.tpot, p.req_s) for p in found if p is not None]
    assert on_fp8 == pytest.approx(on_halved)
    assert on_fp8[0] < on_bf16[0]  # KV is a larger share of the smaller step


# The declared 8B class on a 24 GB L4: 16.4 GB of weights, 13.9 GB of them linear; 3.5 GiB left
# for a 24,000-token pool (156.6 KB per token of the pool, while the step model's KV per
# sequence uses the declared 147,456 B per token: 1,085.7 tokens x 147,456 B = 160 MB); 96
# clients, peaking at 21 running with the pool 95% full, under the engine's cap of 256.
KV_BOUND = Measurement(
    succeeded=600, failed=0, step_ms=98.0, prompt_work_share=0.1, peak_running=21.0,
    peak_kv_usage=0.95, kv_capacity_tokens=24_000.0, kv_available_gib=3.5, peak_in_flight=96,
    queue_share=0.8, server_ttft_p50_ms=2000.0, ttft_p50_ms=2100.0, tpot_p95_ms=120.0,
    ceilings=Ceilings(
        weight_bytes_per_step=16_400_000_000, kv_bytes_per_sequence=160_000_000,
        avg_running_batch=21.0,
    ),
)  # fmt: skip
LINEAR_8B = 13_900_000_000


def kv_bound_situation() -> Situation:
    short = Finding("kv_capacity", "critical", "KV cache 95% full with requests waiting")
    diagnosis = Diagnosis(
        findings=(short,), gates=(), shape=None, primary="kv_capacity", confidence="likely",
        secondary=(), requests=600,
    )  # fmt: skip
    facts = WorkloadFacts(prompts=(), output_tokens=384, context_limit=8192)
    return Situation(measurement=KV_BOUND, facts=facts, settings=Settings(), diagnosis=diagnosis)


def test_on_a_kv_bound_run_a_precision_change_grows_the_batch_into_the_freed_memory() -> None:
    # t = 0.95 x 24,000 / 21 = 1,085.7 tokens a request; the batch after is
    # floor(0.9 x pool / t): 48,000 -> 39 (fp8 KV), 68,385 -> 56 (fp8 weights),
    # 136,769 -> 113, held to the 96 clients (both)
    situation = kv_bound_situation()
    settings = situation.settings
    found = {
        entry: predict(situation, dataclasses.replace(settings, **proposed), entry=entry,
                       linear_bytes=LINEAR_8B, cap=256)
        for entry, proposed in (("fp8-kv-cache", {"kv_cache_dtype": "fp8"}),
                                ("fp8-weights", {"quantization": "fp8"}))
    }  # fmt: skip
    found["pair"] = predict_pair(situation, linear_bytes=LINEAR_8B, cap=256)
    expected = {  # running, TPOT, req/s, TTFT (the queued 1,600 ms goes by (running - 21) / 75)
        "fp8-kv-cache": (39, 0.0748, 0.7279, -0.1829),
        "fp8-weights": (56, 0.1052, 1.4129, -0.3556),
        "pair": (96, 0.2374, 2.6945, -0.7619),
    }
    for name, (running, tpot, req_s, ttft) in expected.items():
        prediction = found[name]
        assert prediction is not None, name
        assert f"with about {running} running" in prediction.text
        assert (prediction.tpot, prediction.req_s, prediction.ttft) == pytest.approx(
            (tpot, req_s, ttft), abs=0.001
        )
    assert "no request waits for KV space" in (found["pair"] or found["fp8-weights"]).text
    limit = AnalysisContext(goal="throughput", tpot_p95_limit_ms=140.0)
    assert _past_limit(found["fp8-weights"], KV_BOUND, limit) is None
    assert _past_limit(found["pair"], KV_BOUND, limit) == (
        "past your TPOT limit (predicted TPOT p95 148 ms, limit 140 ms)"
    )
    assert predict(situation, settings, entry="fp8-kv-cache", linear_bytes=LINEAR_8B) is None
    assert predict_pair(situation, linear_bytes=LINEAR_8B, cap=21) is None


def _kv_bound_offers(situation: Situation) -> list:
    entries = {entry.name: entry for entry in VLLM.playbook()}
    settings = {"fp8-weights": {"quantization": "fp8"}, "fp8-kv-cache": {"kv_cache_dtype": "fp8"}}
    return [
        Suggestion(entry=entries[name], reason=name, tier="try_first", command=name,
                   settings=dataclasses.replace(situation.settings, **change),
                   predicted=Prediction(tpot=-0.1, req_s=0.2, text="", ttft=-0.1))
        for name, change in settings.items()
    ]  # fmt: skip


def _pairs_up(situation: Situation, tmp_path: Path, **context: Any) -> bool:
    offers = _kv_bound_offers(situation)
    with active(AnalysisContext(goal="throughput", linear_weight_bytes=LINEAR_8B, **context)):
        shown = _combined(offers, situation, engine=VLLM, current=Settings(),
                          project_path=tmp_path, version="0.30.0")  # fmt: skip
    return [s.members for s in shown] == [(offers[0].entry, offers[1].entry)]


def test_fp8_weights_and_fp8_kv_are_one_step_only_on_a_kv_bound_run_without_tools(
    tmp_path: Path,
) -> None:
    situation = dataclasses.replace(
        kv_bound_situation(), settings=Settings(max_concurrent_requests=256)
    )
    assert _pairs_up(situation, tmp_path)
    tools = ToolCallStats(
        requests=600, calling=500, produced_call=0.8, valid_json=1.0, known_tool=1.0,
        schema_valid=1.0,
    )  # fmt: skip
    with_tools = dataclasses.replace(
        situation, measurement=dataclasses.replace(KV_BOUND, tool_calls=tools)
    )
    assert not _pairs_up(with_tools, tmp_path)
    assert not _pairs_up(situation, tmp_path, tpot_p95_limit_ms=140.0)
    assert _pairs_up(situation, tmp_path, tpot_p95_limit_ms=160.0)
    rejected = Rejection(run="r", engine_args=(), change=("quantization=fp8", "kv-cache-dtype=fp8"),
                         made_from=None, reason=None, at="")  # fmt: skip
    assert not _pairs_up(situation, tmp_path, rejected=(rejected,))
    off_route, _ = l4_situation(L4_8255, "throughput")
    assert not _pairs_up(off_route, tmp_path)
    for run in sorted(path.name for path in A10G_TRAFFIC.iterdir()):
        assert not _pairs_up(_a10g(run)[0], tmp_path)


def test_off_the_kv_bound_route_the_a10g_traffic_runs_predict_as_before() -> None:
    for run in sorted(path.name for path in A10G_TRAFFIC.iterdir()):
        situation, _ = _a10g(run)
        assert not situation.fired("kv_capacity")
        assert predict_pair(situation, linear_bytes=PHI35_LINEAR_BYTES, cap=256) is None
    lp0, _ = _a10g("20261012T013850Z-ae53")
    fp8 = dataclasses.replace(lp0.settings, quantization="fp8")
    assert predict(lp0, fp8, entry="fp8-weights", linear_bytes=PHI35_LINEAR_BYTES, cap=256) == (
        predict(lp0, fp8, entry="fp8-weights", linear_bytes=PHI35_LINEAR_BYTES)
    )


def test_under_the_ttft_goal_a_predicted_ttft_sets_the_window_s_bar(tmp_path: Path) -> None:
    situation, context = l4_situation(L4_8255, "ttft")
    offered = _l4_offers(situation, context, tmp_path)
    step = next(s for s in offered.suggestions if s.command)
    predicted = Prediction(tpot=-0.2, req_s=0.25, text="predicted", ttft=-0.30)
    offered = dataclasses.replace(offered, suggestions=tuple(
        dataclasses.replace(s, predicted=predicted) if s is step else s
        for s in offered.suggestions
    ))  # fmt: skip
    section = next_steps_section(situation.diagnosis, offered, retained=False)
    parent = tmp_path / L4_8255
    parent.mkdir()
    shutil.copy(L4_STEPS / L4_8255 / "requests.jsonl", parent)
    (parent / "run.json").write_text(json.dumps({"engine_args": []}))
    (parent / "report.json").write_text(json.dumps(to_json(Report(
        run_id=L4_8255, ran="now", sections=(section,)
    ))))  # fmt: skip
    assert step.command is not None
    offer = parent_offer(parent, command_args(step.command))
    assert offer is not None
    workload = SimpleNamespace(
        request_count=1032, arrival=ArrivalSpec(kind="closed_loop", concurrency=16)
    )
    with_bar = planned_window(
        parent, workload, goal="ttft", predicted=offer["predicted"][PREDICTED["ttft"]]
    )
    without = planned_window(parent, workload, goal="ttft")
    assert offer["predicted"]["ttft"] == -0.30
    # 60 s projects about ±8.6% on TTFT p95: inside the bar of 0.15, outside the 8% target
    assert (with_bar.seconds, without.seconds) == (60, 90)


def test_a_step_after_the_async_off_run_is_timed_with_the_planned_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 0.3.11 timed it from the whole run: 125 s to load and 6.9 min to warm up and measure
    run = "20261014T165306Z-eaf7"
    (tmp_path / run).mkdir()
    shutil.copy(L4_STEPS / run / "requests.jsonl", tmp_path / run)
    declared = json.loads((L4_STEPS / run / "run.json").read_text())["analysis"]
    arrival = ArrivalSpec(kind="closed_loop", concurrency=16)
    monkeypatch.setattr(
        "tensward.history.declared_workload",
        lambda project: SimpleNamespace(request_count=1032, arrival=arrival),
    )
    context = AnalysisContext(
        goal="ttft", launched_s=declared["launched_s"], load_s=declared["load_s"],
        measured_s=declared["measured_s"], new_run=tmp_path / run, project=tmp_path,
    )  # fmt: skip
    assert time_to_test(context) == (
        "about 5 min on this setup (this run: 125 s to load, a 90 s window planned)"
    )
    assert seconds_to_test(context) == pytest.approx(318, abs=1)
