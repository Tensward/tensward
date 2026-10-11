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
then 32; ``lp`` ending at cap 8 with n-gram speculation).

``replays/t42-ladder``: the 0.3.10 release TTFT ladder on the T4 of ``t4-saturated``, short
answers, 128 clients: the default 2048-token step budget cold then warm, then 512 and 256. Each
run was saturated at the card's power limit; ``t4-saturated``'s samples stand in for theirs.

Each JSON fixture keeps only the fields these tests read; a run that is analysed but never
history has no ``report.json``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
from test_playbook import OFFERED

from tensward.capabilities import Resolved
from tensward.ceilings import Ceilings
from tensward.classify import classify, measurement_from_metrics
from tensward.context import AnalysisContext, Goal, active
from tensward.engines import Engine
from tensward.engines.vllm import VLLM, logs
from tensward.engines.vllm.engine import VllmEngine
from tensward.gpu_state import GpuState, summarize
from tensward.history import (
    PastRun,
    cap_flag,
    checked,
    cold_kv_tokens,
    command_args,
    past_runs,
    seconds_to_test,
)
from tensward.measurement import Measurement
from tensward.platforms.nvidia import NVIDIA
from tensward.playbook import Situation, WorkloadFacts
from tensward.playbook_common import running_at_cap
from tensward.report.build import RunOutputs, Subject, build_report, next_steps_section
from tensward.report.markdown import sections_markdown, to_markdown
from tensward.settings import Settings
from tensward.slo import NO_LIMITS, Slo
from tensward.suggest import Suggested, first_try, ordered_for_goal, suggestions

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
    text, check = checked([before], frozenset({"max-num-batched-tokens=512"}), run, "ttft")
    assert check == {"run": "r0", "change": "small-step-budget", "predicted": "gpu_saturated",
                     "actual": "gpu_saturated", "confirmed": True, "held": False}  # fmt: skip
    assert text == (
        "run r0 predicted a saturated GPU would bind after `small-step-budget`: confirmed"
    )
    assert checked([before], frozenset({"max-num-batched-tokens=512"}), run, "tpot") == (None, None)


@pytest.mark.parametrize(
    ("run_id", "changes"),
    [
        # text-only: no media encoders to drop, and no step undoes the one before it
        ("a10g-kv-capacity-mistral-20261004T131828Z-c964",
         ["more-kv-memory", "raise-concurrency", "stable-prompt-prefix"]),
        # KV projected at 10% at the raised cap: decode bandwidth, crossed now, still binds
        ("a10g-queueing-mistral-20261004T125303Z-8a30",
         ["raise-concurrency", "fast-quant-kernel", "ngram-speculation"]),
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
}


def replay_step(
    tmp_path: Path, fixture: str, current: int, history: Sequence[int], *,
    context: AnalysisContext | None = None,
    plans: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
    extra: tuple[str, ...] = (),
    registered: int = 0,
) -> tuple[Situation, Suggested]:  # fmt: skip
    """Analyse run ``current`` of a release ladder again under its goal, with the runs
    ``history`` as the project's earlier runs and the cold KV pool the CLI would find; with
    ``extra``, as a run of the same measurement whose engine arguments add those; with
    ``registered``, as a project whose setup is that run's settings."""
    directory, goal, output_tokens, gpu = LADDERS[fixture]
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
    facts = WorkloadFacts(prompts=(), output_tokens=output_tokens, context_limit=32768)
    situation = Situation(measurement=measurement, facts=facts, settings=settings,
                          diagnosis=diagnosis)  # fmt: skip
    args = frozenset(record["engine_args"]) | set(extra)
    cold = cold_kv_tokens(past_runs(project, VLLM), args, cap_flag(VLLM))
    context = context or AnalysisContext()
    context = dataclasses.replace(
        context, goal=goal, engine_args=tuple(sorted(args)), cold_kv_tokens=cold,
        flags=("--goal", goal) if goal != "throughput" else (),
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
    _, lf8 = replay_step(tmp_path / "lf8-2", "lf8", 2, [0, 1])  # a larger cap is blocked there
    assert lf8.stop is not None
    assert "the change for queueing before scheduling is blocked" in lf8.stop
    _, lf2 = replay_step(tmp_path / "lf2-2", "lf2", 2, [0, 1])  # the second A10G ladder's end
    assert lf2.stop is not None
    assert lf2.stop.endswith("no measured limit calls for a change that leaves TTFT alone.")
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
    stop = ladders["lf8"][2].stop or ""
    assert "of the prompt rate this engine measured (4,553 tokens/s)" in stop


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


def test_under_the_ttft_goal_a_512_token_step_budget_goes_on_to_256(tmp_path: Path) -> None:
    situation, shown = replay_step(tmp_path, "t42", 2, [0, 1])
    top = first_try(situation.diagnosis, shown)
    assert top is not None and top.entry.name == "small-step-budget"
    assert command_args(top.command or "") >= {"max-num-batched-tokens=256"}
