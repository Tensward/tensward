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

``replays/l4-gemma-chat``: eight 0.3.13 runs of Gemma 4 E4B on an L4, vLLM 0.30.0, chat prompts,
one session per goal. Throughput: gmu 0.95 (``62c9``), a text-only start that found no KV memory
(``f43e``: log and settings only) and fp8 KV (``114a``). TTFT: the default (``bb04``), gmu 0.95
(``567c``) and its text-only child with a 9,668-token pool (``bf38``). TPOT: the default
(``d0a1``) and fp8 weights (``898b``). The logs keep the start-up and KV lines only.

``replays/a10g-traffic``: five 0.3.11 runs on an A10G, vLLM 0.30.0. Phi-3.5-mini-instruct, 64
closed-loop clients under the TPOT goal: cap 32 (``ae53``), the same with fp8 weights (``6d98``)
and cap 16 (``147f``). Qwen2.5-7B-Instruct with tool calls, 64 clients, no cap: bf16 (``535d``,
whose prompt-work share was not measured) and fp8 weights (``79d5``).

``replays/l4-qwen3-queue``: two 0.3.16 runs of Qwen3-8B on an L4, vLLM 0.30.0, 96 closed-loop
clients at a cap of 34, chat prompts, under the TPOT goal: bf16 short of KV memory (``5a8e``, 302
preemptions) and its fp8 weights and KV child, whose pool was 29% used while 62 requests waited
for the cap (``d497``). The logs keep the KV and memory lines only.

Each JSON fixture keeps only the fields these tests read; a run that is analysed but never
history has no ``report.json``.
"""

from __future__ import annotations

import dataclasses
import gzip
import itertools
import json
import math
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import pytest
from test_anatomy import checkpoint_anatomy
from test_playbook import OFFERED

from tensward import guard, kvsize, window
from tensward.anatomy import ModelAnatomy, build_anatomy, freed_bytes
from tensward.answer_options import MARLIN_FP8_ARG, WEIGHT_ONLY, AnswerOptions, Format
from tensward.artifacts import ArtifactVariant
from tensward.calibration import profile_for
from tensward.capabilities import Resolved
from tensward.ceilings import Ceilings, Measured, compute_ceilings
from tensward.classify import Diagnosis, Finding, classify, measurement_from_metrics
from tensward.client import RequestRecord
from tensward.combine import PREDICTED, answer_risk, parent_offer
from tensward.context import AnalysisContext, Goal, active
from tensward.engines import Engine
from tensward.engines.vllm import VLLM, logs
from tensward.engines.vllm.engine import VllmEngine
from tensward.engines.vllm.predicates import (
    async_scheduling_applies,
    cuda_graphs_applies,
    explicit_kv_size_applies,
    fast_prefill_applies,
    fp8_kv_decode_applies,
    more_kv_memory_applies,
    step_budget_down_applies,
    with_cuda_graphs,
    with_explicit_kv_size,
    with_fast_prefill,
)
from tensward.engines.vllm.quantization import options_from
from tensward.engines.vllm.start_phases import CLASSES
from tensward.engines.vllm.startup import (
    forced_backend,
    kv_available_gib,
    kv_pool_tokens,
    memory_usage,
    no_kv_memory,
    startup,
)
from tensward.gpu_state import GpuState, summarize
from tensward.history import (
    PastRun,
    cap_flag,
    checked,
    cold_kv_tokens,
    cold_pools,
    command_args,
    full_window,
    past_runs,
    planned_window,
    seconds_to_test,
    time_to_test,
)
from tensward.kept import REGISTERED, Kept
from tensward.measure import _logged_gib, computed_prompt_tokens, signal_increase
from tensward.measurement import Measurement, Window, avg_context_tokens
from tensward.platforms.nvidia import NVIDIA
from tensward.playbook import Situation, Suggestion, WorkloadFacts
from tensward.playbook_common import kv_held_back, running_at_cap, without_media_encoders
from tensward.predict import Prediction, held_steps, predict, predict_pair, predict_stack
from tensward.project import CurrentSetup
from tensward.prometheus import read_signals
from tensward.prompt_lookup import Coverage
from tensward.rejected import Rejection
from tensward.report.build import RunOutputs, Subject, build_report, next_steps_section
from tensward.report.markdown import sections_markdown, to_markdown
from tensward.report.model import Report, SuggestionBlock, Text, to_json
from tensward.settings import Settings
from tensward.slo import NO_LIMITS, Slo
from tensward.suggest import (
    BUNDLE_WINDOW,
    Suggested,
    _changes_answers,
    _held_figures,
    _kv_bound_pair,
    _past_limit,
    _sized_offers,
    _stacked,
    _start_cost,
    _start_order,
    answer_warning,
    carried_gate,
    first_try,
    group_members,
    ordered_for_goal,
    suggestion_command,
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
    # the T4 ladder's 256-token run: nothing else in Try first can run, so the lossless
    # Could-help offer is promoted, with no empty saturated-GPU group above
    situation, last = replay_step(tmp_path, "t42", 3, range(3))
    shown = sections_markdown([next_steps_section(situation.diagnosis, last, retained=False)])
    assert last.stop is None and "For a saturated GPU:" not in shown
    assert "- `stream-interval`: " in shown and "measured now: nothing else in Try first" in shown
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
        ("throughput", "Nothing left in Try first for throughput: the GPU was saturated at its "
                       "power limit"),
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


def test_the_t4_ttft_ladder_goes_512_then_async_off_then_stream_interval(tmp_path: Path) -> None:
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
    assert [s.entry.name for s in async_off.suggestions if s.tier == "try_first"] == [
        "stream-interval"
    ]
    assert async_off.stop is None
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


def test_a_decode_bound_run_offers_8_bit_weights_first_with_its_warning(
    tmp_path: Path,
) -> None:
    context = AnalysisContext(answer_options=A10G_OPTIONS)
    situation, shown = replay_step(tmp_path, "lp", 0, [], context=context)
    names = [s.entry.name for s in shown.suggestions if s.tier == "try_first"]
    assert names.index("fp8-weights") < names.index("lower-concurrency")
    fp8 = next(s for s in shown.suggestions if s.entry.name == "fp8-weights")
    assert command_args(fp8.command or "") >= {"quantization=fp8", "linear-backend=marlin"}
    assert "the maths stays 16-bit" in fp8.reason
    top = first_try(situation.diagnosis, shown)
    assert top is fp8
    planned = dataclasses.replace(shown, plan_lines=("Then measure the next limit.",))
    section = next_steps_section(situation.diagnosis, planned, retained=False)
    blocks = to_json(Report(run_id="r", ran="now", sections=(section,)))["sections"][0]["blocks"]
    tried = [b for b in blocks if b["kind"] == "suggestion" and b["tier"] == "try_first"]
    assert tried[0]["key"] == "fp8-weights"
    assert [(b["key"], b["plan"]) for b in tried if "plan" in b] == [
        ("fp8-weights", ["Then measure the next limit."])
    ]
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
        ("ttft", 0.05, False, "drafts would supply 5.0% of the output; below 6% speculation"),
        ("ttft", 0.2, False, None),
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
    short = (
        "run 20261015T125727Z-2dda was short of KV memory (its KV-capacity finding fired, 81 "
        "preemptions, KV cache at 100% at its peak): "
    )
    for only in (None, 60.0):
        for goal, why in (
            ("ttft", "under the ttft goal a short window reads the queue still building up as a "
             "loss on TTFT p95"),
            ("throughput", "its steady window (a 195 s ramp, 390 s steady and a 97 s tail) is "
             "longer than its own 538 s of dispatch"),
        ):  # fmt: skip
            plan = planned_window(reference, workload, goal=goal, only=only)
            assert (plan.requests, plan.seconds, plan.reason) == (200, None, short + why)
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
    adopt: bool = False, settings: Settings | None = None,
) -> tuple[Suggested, dict[str, str], Mapping[str, Any]]:  # fmt: skip
    """A planned step of the L4 cell (``rows``, measured with ``args``, its ``settings`` the
    default's unless given) against the full default run it was made from: what it offers, its
    keep section and its report's goal. The step's point values are the default's, so the floor
    and the limits hold."""
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
            settings=settings or situation.settings, subject=Subject(setup=setup), image=None,
            slo=slo, engine=VLLM, gpu="NVIDIA L4", source="", engine_args=args, retained=True,
        ))  # fmt: skip
    keep = next(s for s in report.sections if s.id == "keep")
    shown = {b.key: b.text for b in keep.blocks if isinstance(b, Text)}
    return suggested, shown, report.goal or {}


def test_async_off_on_a_60_s_window_gains_ttft_only_in_its_ramp_and_is_a_loss_under_tpot(
    tmp_path: Path,
) -> None:
    rows = (L4_STEPS / "20261014T165306Z-eaf7" / "requests.jsonl").read_text().splitlines()
    whole = window.compare_matched(
        _cut("20261014T165306Z-eaf7", tmp_path / "cut"), L4_STEPS / L4_PARENT, clients=16
    )
    assert not isinstance(whole, str) and whole["ttft_p95_ms"].verdict == "better"
    ttft, keep, _ = l4_keep(tmp_path, "ttft", rows[:236], ("no-async-scheduling",))
    assert keep["not_kept"] == (
        "Tensward does not recommend keeping this change: within noise: "
        "TTFT p95 after the ramp −5% ±8%."
    )
    assert ttft.memory is not None and ttft.memory.parent is not None
    deltas = ttft.memory.parent.deltas
    assert deltas["ttft_p95_ms"].verdict == "within_noise"
    assert deltas["ttft_p95_ms"].change == pytest.approx(-0.046, abs=0.005)
    assert deltas["ttft_p95_ms"].band == pytest.approx(0.075, abs=0.005)
    assert ttft.memory.lines[0] == (
        f"since run {L4_PARENT} (the latest run of your starting settings), on the 236 "
        "prompts both runs sent: TTFT p95 after the ramp −5% ±8% (within noise); full answer p95 "
        "+4% ±5% (within noise); requests per second −4% ±3% (worse); TPOT p95 +5% ±3% (worse)"
    )
    assert ttft.memory.lines[1].startswith("TTFT p95 of the middle 80% −6% ±")
    assert keep["trade"] == (
        f"Measured since run {L4_PARENT}: costs: requests per second −4% ±3%, TPOT p95 +5% ±3%; "
        "within noise: TTFT p95 −5% ±8%, full answer p95 +4% ±5%."
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
    assert keep["not_kept"].endswith("within noise: TTFT p95 after the ramp −3% ±4%.")
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


def test_the_memory_facts_come_from_the_server_log() -> None:
    def log(run: str) -> str:
        return (REPLAYS / "l4-gemma-chat" / run / "server.log").read_text()

    roomy, over_reserved, refused = (
        log(run)
        for run in ("20261016T174304Z-567c", "20261016T175324Z-bf38", "20261016T175339Z-f43e")
    )
    assert kv_pool_tokens(roomy) == 126_449 and kv_pool_tokens(over_reserved) == 9_668
    assert memory_usage(roomy) == {
        "consumed_gib": 15.45,
        "peak_activation_gib": 0.82,
        "graphs_gib": 0.29,
        "fit_kv_bytes": 4531650560,
        "full_kv_bytes": 5511184384,
    }
    assert memory_usage(over_reserved) == {
        "consumed_gib": 14.52,
        "peak_activation_gib": 6.06,
        "graphs_gib": 0.31,
        "fit_kv_bytes": -103744000,
        "full_kv_bytes": 875789824,
    }
    assert memory_usage(refused) is None and kv_pool_tokens(refused) is None
    assert [no_kv_memory(text) for text in (roomy, over_reserved, refused)] == [False, False, True]
    assert forced_backend(roomy) == "TRITON_ATTN" and forced_backend("no such line") is None


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
    paired = " --engine-arg quantization=fp8" if goal == "tpot" else ""
    assert block["command"] == command + paired
    assert "7.9x" in block["reason"]


def test_on_tool_calls_the_report_writes_the_answer_risk_the_order_used(tmp_path: Path) -> None:
    situation, context = l4_situation(L4_8255, "tpot")
    context.anatomy = dense_anatomy(QWEN3_06B_CONFIG)
    situation = _with_tool_calls(situation, L4_8255)
    offered = _l4_offers(situation, context, tmp_path)
    with active(context):
        section = next_steps_section(situation.diagnosis, offered, retained=False)
        report = to_json(Report(run_id="r", ran="now", sections=(section,)))
    blocks = report["sections"][0]["blocks"]
    risks = {b["name"]: b.get("answer_risk") for b in blocks if b["kind"] == "suggestion"}
    assert risks["fp8-kv-decode"] == risks["fp8-weights"] == "high"


@pytest.mark.parametrize(
    ("goal", "waiting", "offered"),
    [
        ("throughput", 2.0, "fp8-kv-cache"),
        ("tpot", 2.0, "fp8-kv-decode"),
        ("throughput", None, "fp8-kv-decode"),
    ],
    ids=["held-back-throughput", "held-back-tpot", "full-cache-no-waiting"],
)
def test_where_kv_memory_is_short_one_fp8_kv_change_is_offered(
    goal: Goal, waiting: float | None, offered: str, tmp_path: Path
) -> None:
    situation, context = l4_situation(L4_8255, goal)
    metrics = json.loads((L4_STEPS / L4_8255 / "metrics.json").read_text())
    record = json.loads((L4_STEPS / L4_8255 / "run.json").read_text())
    measurement = dataclasses.replace(
        situation.measurement, peak_kv_usage=0.99, peak_waiting=8.0,
        mean_waiting=situation.measurement.mean_waiting if waiting is None else waiting,
    )  # fmt: skip
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
    paired = goal == "tpot"
    assert leading.predicted is not None
    assert [m.name for m in leading.members] == (["fp8-kv-decode", "fp8-weights"] if paired else [])
    assert leading.predicted.tpot == pytest.approx(-0.355 if paired else -0.325, abs=0.05)
    assert leading.predicted.req_s == pytest.approx(0.55 if paired else 0.481, abs=0.05)
    assert [s.entry.name for s in rest if s.command] == ([] if paired else ["fp8-weights"])
    diagnosis = situation.diagnosis
    group = try_first_order(diagnosis, offered)[0]
    assert group_members(diagnosis, offered, group)[0] is leading
    assert first_try(diagnosis, offered) is leading
    section = next_steps_section(diagnosis, offered, retained=False)
    blocks = to_json(Report(run_id="r", ran="now", sections=(section,)))["sections"][0]["blocks"]
    shown = [b for b in blocks if b["kind"] == "suggestion" and b["tier"] == "try_first"]
    assert shown[0]["key"] == ("fp8-kv-decode+fp8-weights" if paired else "fp8-kv-decode")
    assert shown[0]["quality_risk"]


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
        ("qwen3", lambda s, c: (_ceilings(s, avg_running_batch=0.4), c), False),
        ("qwen3", lambda s, c: (dataclasses.replace(
            s, measurement=dataclasses.replace(s.measurement, hybrid_cache=True)), c), False),
    ],
    ids=["qwen2-l4", "qwen2-a10g", "no-fp8-kv-gpu", "kv-8bit", "kv-under-a-quarter", "hybrid"],
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


@pytest.mark.parametrize(("goal", "offered"), [("tpot", True), ("throughput", False)])
def test_the_bandwidth_share_gates_fp8_kv_decode_outside_tpot(
    goal: Goal, offered: bool, tmp_path: Path
) -> None:
    situation, context = l4_situation(L4_8255, goal)
    situation = _ceilings(situation, decode_pct_of_ceiling=42.0)
    names = {s.entry.name for s in _l4_offers(situation, context, tmp_path).suggestions}
    assert ("fp8-kv-decode" in names) is offered


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
    ("fp8-kv-decode", {"kv_cache_dtype": "fp8"}, -0.30, -0.18),
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
    waiting = dataclasses.replace(situation.measurement, mean_waiting=2.0)  # no full closed loop
    situation = dataclasses.replace(situation, measurement=change(waiting))
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
        shown, _ = _stacked(offers, (), situation, engine=VLLM, current=Settings(),
                            project_path=tmp_path, version="0.30.0")  # fmt: skip
    return [s.members for s in shown] == [(offers[0].entry, offers[1].entry)]


def test_fp8_weights_and_fp8_kv_are_one_step_only_on_a_kv_bound_run(tmp_path: Path) -> None:
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
    assert kv_held_back(with_tools) and _pairs_up(with_tools, tmp_path)
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


L4_GEMMA = REPLAYS / "l4-gemma-chat"
GEMMA_RUN = L4_GEMMA / "20261016T172654Z-ed76"
BF16_VARIANT = ArtifactVariant(
    weight_precision="bf16", activation_dtype="bfloat16", quantization_method="none"
)


def gemma_e4b_anatomy() -> ModelAnatomy:
    model = L4_GEMMA / "model"
    rows = json.loads(gzip.decompress((model / "tensors.json.gz").read_bytes()))
    tensors = {name: (dtype, tuple(shape), size) for name, (dtype, shape, size) in rows.items()}
    return build_anatomy(json.loads((model / "config.json").read_text()), None, tensors)


def replay_measurement(
    run_dir: Path, anatomy: ModelAnatomy, variant: ArtifactVariant = BF16_VARIANT
) -> Measurement:
    """A stored run's measurement with the ceilings ``anatomy`` gives it. The machine is the
    recorded one (its GPU name and memory), so a test host without a GPU still gets the row.
    The window's counters are the difference of the stored scrapes; the average context is the
    sealed ``avg_context_tokens`` over the stripped request rows."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    record = json.loads((run_dir / "run.json").read_text())
    counters = signal_increase(
        *(read_signals((run_dir / f"metrics_{when}.prom").read_text(), VLLM.signals)
          for when in ("before", "after"))
    )  # fmt: skip
    rows = [json.loads(line) for line in (run_dir / "requests.jsonl").read_text().splitlines()]
    prompt_tokens = {row["request_id"]: row["prompt_tokens"] for row in rows}
    records = [
        RequestRecord(
            request_id=row["request_id"], scheduled_ns=row["dispatch_ns"],
            dispatch_ns=row["dispatch_ns"], first_content_ns=row["first_content_ns"],
            terminal_ns=row["terminal_ns"], outcome=row["outcome"],
            committed_output_tokens=row["committed_output_tokens"],
            prompt_index=row["prompt_index"],
        )
        for row in rows
    ]  # fmt: skip
    ceilings = compute_ceilings(
        anatomy=anatomy,
        variant=variant,
        kv_cache_dtype=record["settings"]["kv_cache_dtype"],
        gpus=((record["gpus_identity"][0]["name"], int(metrics["gpu_memory_gib"] * 1024)),),
        measured=Measured(
            seconds=metrics["seconds"],
            avg_context_tokens=avg_context_tokens(
                records, Window(**metrics["window"]), prompt_tokens
            ),
            prefill_computed_tokens=computed_prompt_tokens(counters),
            generation_tokens=counters.generation_tokens,
            avg_running_batch=metrics["mean_running"],
        ),
    )  # fmt: skip
    return dataclasses.replace(measurement_from_metrics(metrics), ceilings=ceilings)


def test_a_stored_gemma_run_gets_its_hardware_ceiling_from_the_new_anatomy() -> None:
    stored = measurement_from_metrics(json.loads((GEMMA_RUN / "metrics.json").read_text()))
    assert stored.ceilings is not None and stored.ceilings.decode_ceiling_tok_s is None

    ceilings = replay_measurement(GEMMA_RUN, gemma_e4b_anatomy()).ceilings

    assert ceilings is not None and ceilings.unavailable is None
    assert ceilings.gpu == "L4" and ceilings.bandwidth_source == "datasheet"
    assert ceilings.decode_ceiling_tok_s is not None and ceilings.decode_pct_of_ceiling is not None
    assert 0 < ceilings.decode_pct_of_ceiling <= 100
    assert ceilings.avg_context_tokens == pytest.approx(730 + 384 / 2, rel=0.1)


def test_the_gemma_run_s_fp8_weights_are_predicted_from_the_quantizable_bytes() -> None:
    anatomy = gemma_e4b_anatomy()
    facts = WorkloadFacts(prompts=(), output_tokens=384, context_limit=8192, model_type="gemma4")
    situation = recorded_situation(GEMMA_RUN, facts)
    situation = dataclasses.replace(situation, measurement=replay_measurement(GEMMA_RUN, anatomy))
    freed = freed_bytes(anatomy)
    assert freed is not None

    prediction = predict(
        situation, dataclasses.replace(situation.settings, quantization="fp8"),
        entry="fp8-weights", linear_bytes=freed["quantizable"],
    )  # fmt: skip

    assert prediction is not None and prediction.tpot is not None and prediction.tpot < 0


@pytest.mark.parametrize(
    ("parent", "child", "change", "entry"),
    [
        ("20261016T174845Z-d0a1", "20261016T182032Z-898b", {"quantization": "fp8"}, "fp8-weights"),
        ("20261016T172654Z-ed76", "20261016T175718Z-114a", {"kv_cache_dtype": "fp8"},
         "fp8-kv-cache"),
    ],
)  # fmt: skip
def test_with_every_client_running_ttft_follows_the_predicted_step_time(
    parent: str, child: str, change: dict[str, str], entry: str
) -> None:
    anatomy = gemma_e4b_anatomy()
    freed = freed_bytes(anatomy)
    assert freed is not None
    situation = recorded_situation(L4_GEMMA / parent, GEMMA_FACTS)
    ceilings = replay_measurement(GEMMA_RUN, anatomy).ceilings  # same cell and batch
    situation = dataclasses.replace(
        situation, measurement=dataclasses.replace(situation.measurement, ceilings=ceilings)
    )
    options = options_from(ENGINE_SUPPORT, 8.9)
    assert not isinstance(options, str)

    with active(AnalysisContext(answer_options=options)):
        prediction = predict(
            situation, dataclasses.replace(situation.settings, **change), entry=entry,
            linear_bytes=freed["quantizable"], cap=256,
        )  # fmt: skip

    ttft = [json.loads((L4_GEMMA / r / "metrics.json").read_text())["ttft_p50_ms"]
            for r in (parent, child)]  # fmt: skip
    assert prediction is not None and prediction.ttft is not None
    assert prediction.ttft == pytest.approx(ttft[1] / ttft[0] - 1, abs=0.05)
    assert "a prompt waits about 4 decode steps" in prediction.text


def test_a_kv_size_adds_the_freed_bytes_within_what_the_parent_s_gpu_can_hold() -> None:
    kv = round(json.loads((GEMMA_RUN / "metrics.json").read_text())["kv_available_gib"] * 2**30)
    usage = memory_usage((GEMMA_RUN / "server.log").read_text())
    assert usage is not None
    full, freed = usage["full_kv_bytes"], freed_bytes(gemma_e4b_anatomy())
    assert freed is not None
    towers, fp8 = freed["towers"], freed["towers"] + freed["quantizable"] // 2

    def size(freeing: int, **given: Any) -> int | None:
        memory = {"peak_used_mib": None, "total_mib": None, **given}
        return kvsize.sized(kv, freed_bytes=freeing, full_kv_bytes=full, **memory)

    alone = size(towers)
    assert alone == kv + towers and alone / 2**30 == pytest.approx(4.9, abs=0.02)
    assert alone < full + towers - kvsize.MARGIN_BYTES
    assert (size(fp8) or 0) > full
    near = size(towers, peak_used_mib=21_800.0, total_mib=22_560.0)
    assert near is not None and near < alone
    assert 21_800 * 2**20 - kv - towers + near + kvsize.MARGIN_BYTES <= 22_560 * 2**20
    assert size(towers, backed_off=1) == pytest.approx(0.85 * alone, abs=1)
    assert kvsize.sized(None, freed_bytes=towers, full_kv_bytes=full, peak_used_mib=None,
                        total_mib=None) is None  # fmt: skip


def test_the_gemma_run_s_text_only_offer_carries_its_kv_size_or_limits_every_modality() -> None:
    facts = WorkloadFacts(prompts=(), output_tokens=384, context_limit=8192, model_type="gemma4",
                          media_encoders=True)  # fmt: skip
    situation = recorded_situation(GEMMA_RUN, facts)
    unsized = dataclasses.replace(
        situation, measurement=dataclasses.replace(situation.measurement, kv_available_gib=None)
    )
    anatomy = gemma_e4b_anatomy()
    towers = (freed_bytes(anatomy) or {})["towers"]

    with active(AnalysisContext(anatomy=anatomy)):
        sized_args = VLLM.engine_args_between(
            situation.settings, without_media_encoders(situation, VLLM)
        )
        limited_args = VLLM.engine_args_between(
            unsized.settings, without_media_encoders(unsized, VLLM)
        )

    assert sized_args == [
        "language-model-only",
        f"kv-cache-memory-bytes={round(4.0 * 2**30) + towers}",
    ]
    assert limited_args == ['limit-mm-per-prompt={"audio": 0, "image": 0, "video": 0}']


def test_an_over_reserved_start_is_offered_the_kv_size_of_the_run_with_no_engine_arguments(
    tmp_path: Path,
) -> None:
    project = tmp_path / REPLAYED
    for run in ("20261016T172654Z-ed76", "20261016T175324Z-bf38"):
        shutil.copytree(L4_GEMMA / run, project / "runs" / run)
    over = project / "runs" / "20261016T175324Z-bf38"
    facts = WorkloadFacts(prompts=(), output_tokens=384, context_limit=8192, model_type="gemma4")
    situation = recorded_situation(over, facts)
    unsized = dataclasses.replace(
        situation, measurement=dataclasses.replace(situation.measurement, kv_available_gib=None)
    )

    with active(AnalysisContext(project=project, new_run=over)):
        reason = explicit_kv_size_applies(situation, VLLM)
        settings = with_explicit_kv_size(situation, VLLM)
        assert explicit_kv_size_applies(unsized, VLLM) is None
    with active(AnalysisContext(project=project, new_run=project / "runs" / GEMMA_RUN.name)):
        assert explicit_kv_size_applies(recorded_situation(GEMMA_RUN, facts), VLLM) is None

    assert reason is not None and "6.06 GiB" in reason and "7.4x the 0.82 GiB" in reason
    size = int(settings.extra_args[kvsize.KV_FLAG])
    assert size == 875789824 + round((6.06 - 0.82) * 2**30) - kvsize.MARGIN_BYTES  # the GPU's cap
    assert 4.0 < size / 2**30 < 4.66 + 0.9  # above the default pool, below gmu 0.95's plus towers


GEMMA_FACTS = WorkloadFacts(prompts=(), output_tokens=384, context_limit=8192, model_type="gemma4")
ENGINE_SUPPORT = {  # vLLM 0.30.0's registry as the support read gives it, floors x 10
    "version": "0.30.0",
    "methods": {"fp8": 75, "fp8_per_tensor": 75, "fp8_per_block": 75, "fp8_per_channel": 75,
                "compressed-tensors": 70, "awq": 75, "gptq": 60},
    "kv_dtypes": ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
    "plugins": {"vllm-bnb-plugin": None, "vllm-gguf-plugin": None},
}  # fmt: skip


def test_a_text_only_retry_adds_the_freed_towers_and_backs_off_once(tmp_path: Path) -> None:
    # gmu 0.95 (567c: 4.66 GiB, full 5.13 GiB) offered the text model alone, as bf38 ran it
    project = tmp_path / REPLAYED
    for run in ("20261016T172703Z-bb04", "20261016T174304Z-567c"):
        shutil.copytree(L4_GEMMA / run, project / "runs" / run)
    this = project / "runs" / "20261016T174304Z-567c"
    situation = recorded_situation(this, dataclasses.replace(GEMMA_FACTS, media_encoders=True))
    anatomy = gemma_e4b_anatomy()
    towers = (freed_bytes(anatomy) or {})["towers"]
    full = (memory_usage((this / "server.log").read_text()) or {})["full_kv_bytes"]
    entry = next(e for e in VLLM.playbook() if e.name == "no-media-encoders")
    context = AnalysisContext(project=project, new_run=this, anatomy=anatomy,
                              engine_args=("gpu-memory-utilization=0.95",))  # fmt: skip

    def sized() -> tuple[frozenset[str], int]:
        with active(context):
            settings = without_media_encoders(situation, VLLM)
            command = suggestion_command(VLLM, project, Settings(), settings)
            offer = Suggestion(entry=entry, reason="", tier="try_first", settings=settings,
                               command=command)  # fmt: skip
            (found,) = _sized_offers([offer], situation, engine=VLLM, current=Settings(),
                                     project_path=project, not_applicable=[])  # fmt: skip
        size = next(
            a for a in command_args(found.command or "") if a.startswith(kvsize.KV_FLAG[2:])
        )
        return command_args(command), int(size.partition("=")[2])

    failed = project / "runs" / "20261016T173000Z-0000"
    failed.mkdir()
    (failed / "run.json").write_text(json.dumps({"engine_args": ["kv-cache-memory-bytes=1"]}))
    (failed / "metrics.json").write_text(json.dumps({"failed": 50, "succeeded": 950}))
    assert sized()[1] == int((round(4.0 * 2**30) + towers) * kvsize.BACK_OFF)  # from bb04, once
    shutil.rmtree(failed)

    tried, _ = sized()
    stopped = project / "runs" / "20261016T175324Z-0000"
    stopped.mkdir()
    guard = {"engine_args": sorted(tried), "reason": "stopped at start-up"}
    (stopped / "window.json").write_text(json.dumps({"startup_guard": guard}))
    assert sized()[1] == int((full + towers - kvsize.MARGIN_BYTES) * kvsize.BACK_OFF)


def _gemma_cell(
    goal: Goal, measurement: Measurement | None = None, **extra_args: str
) -> tuple[Situation, AnalysisContext]:
    anatomy = gemma_e4b_anatomy()
    situation = recorded_situation(GEMMA_RUN, GEMMA_FACTS)
    settings = dataclasses.replace(situation.settings, extra_args=extra_args)
    situation = dataclasses.replace(
        situation, settings=settings,
        measurement=measurement or replay_measurement(GEMMA_RUN, anatomy),
    )  # fmt: skip
    options = options_from(ENGINE_SUPPORT, 8.9)
    assert not isinstance(options, str)
    freed = freed_bytes(anatomy)
    assert freed is not None
    context = AnalysisContext(
        goal=goal, anatomy=anatomy, linear_weight_bytes=freed["quantizable"],
        answer_options=options,
    )  # fmt: skip
    return situation, context


def _gemma_offers(
    goal: Goal, tmp_path: Path, measurement: Measurement | None = None, **extra_args: str
) -> Suggested:
    situation, context = _gemma_cell(goal, measurement, **extra_args)
    return _l4_offers(situation, context, tmp_path)


def _fp8_weights(offered: Suggested) -> Suggestion:
    return next(s for s in offered.suggestions if s.entry.name == "fp8-weights")


def test_on_the_l4_fp8_weights_go_per_channel_only_where_the_kv_cache_keeps_headroom(
    tmp_path: Path,
) -> None:
    roomy = _fp8_weights(_gemma_offers("tpot", tmp_path))
    assert roomy.command is not None and "quantization=fp8_per_channel" in roomy.command
    assert re.search(r"per-channel: .*; the KV cache keeps \d+% headroom", roomy.reason)

    # a 16 GiB pool at full use grows by 3.75 GiB of halved weights less per-channel's 1 GiB
    full = dataclasses.replace(
        replay_measurement(GEMMA_RUN, gemma_e4b_anatomy()), kv_available_gib=16.0,
        peak_kv_usage=1.0,
    )  # fmt: skip
    tight = _fp8_weights(_gemma_offers("tpot", tmp_path, full))
    assert tight.command is not None and tight.command.endswith("--engine-arg quantization=fp8")
    assert "the KV cache would keep 15% headroom, under 20%" in tight.reason


def test_below_8_9_fp8_weights_stay_per_tensor_on_marlin_whatever_the_engine_lists(
    tmp_path: Path,
) -> None:
    lp0, _ = _a10g("20261012T013850Z-ae53")
    options = options_from(ENGINE_SUPPORT, 8.6)
    assert not isinstance(options, str) and dict(options.on_the_fly) == {"fp8": WEIGHT_ONLY}
    context = AnalysisContext(
        goal="tpot", max_req_s_loss=0.20, linear_weight_bytes=PHI35_LINEAR_BYTES,
        answer_options=options,
    )  # fmt: skip
    with active(context):
        offered = suggestions(
            VLLM, lp0, current=lp0.settings, project_path=tmp_path, version="0.30.0"
        )
    command = _fp8_weights(offered).command
    assert command is not None and "quantization=fp8 " in f"{command} "
    assert f"--engine-arg {MARLIN_FP8_ARG}" in command


def test_after_a_per_channel_parent_the_fp8_kv_step_reads_8_bit_weights() -> None:
    # the stored fp8 run with the bf16 run's ceilings (same model, cell and batch): the ceilings
    # count the linear weights at 16 bits, as the sealed path writes them for any run
    anatomy = gemma_e4b_anatomy()
    freed = freed_bytes(anatomy)
    assert freed is not None
    served = recorded_situation(L4_GEMMA / "20261016T182032Z-898b", GEMMA_FACTS)
    ceilings = replay_measurement(GEMMA_RUN, anatomy).ceilings
    served = dataclasses.replace(
        served, measurement=dataclasses.replace(served.measurement, ceilings=ceilings)
    )
    found = {}
    for method in ("fp8_per_channel", "fp8", None):
        parent = dataclasses.replace(
            served, settings=dataclasses.replace(served.settings, quantization=method)
        )
        proposed = dataclasses.replace(parent.settings, kv_cache_dtype="fp8")
        found[method] = predict(
            parent, proposed, entry="fp8-kv-decode", linear_bytes=freed["quantizable"]
        )
    assert found["fp8_per_channel"] is not None and found["fp8_per_channel"] == found["fp8"]
    assert found[None] is not None and found["fp8_per_channel"].tpot < found[None].tpot


def test_no_fp8_kv_offer_with_flashinfer_forced_on_512_wide_heads(tmp_path: Path) -> None:
    def names(offered: Suggested) -> set[str]:
        return {m.name for s in offered.suggestions for m in (s.entry, *s.members)}

    assert {"fp8-kv-cache"} <= names(_gemma_offers("ttft", tmp_path))

    forced = _gemma_offers("ttft", tmp_path, **{"--attention-backend": "FLASHINFER"})

    assert not names(forced) & FP8_KV
    assert ("fp8-kv-cache", "fp8 KV with FlashInfer fails to start at head dim 512 on this GPU "
            "(8.9)") in forced.not_applicable  # fmt: skip


GEMMA_CLOSED_LOOP = WorkloadFacts(
    prompts=(), output_tokens=384, context_limit=8192, model_type="gemma4",
    load="96 concurrent clients", max_inflight=96,
)  # fmt: skip


def _closed_loop_offers(goal: Goal, tmp_path: Path, **declared: Any) -> dict[str, Any]:
    context = AnalysisContext(goal=goal, **declared)
    with active(context):
        situation = recorded_situation(GEMMA_RUN, GEMMA_CLOSED_LOOP)
        offered = suggestions(
            VLLM, situation, current=situation.settings, project_path=tmp_path, version="0.30.0"
        )
    return {s.entry.name: s for s in offered.suggestions}


@pytest.mark.parametrize(("goal", "tokens"), [("ttft", 1024), ("throughput", 1024), ("tpot", 512)])
def test_synchronized_closed_loop_on_the_default_budget_is_offered_a_smaller_step(
    goal: Goal, tokens: int, tmp_path: Path
) -> None:
    offers = _closed_loop_offers(goal, tmp_path)

    down = offers["step-budget-down"]
    assert down.settings is not None and down.settings.prefill_batch_tokens == tokens
    assert f"max-num-batched-tokens={tokens}" in (down.command or "")
    assert "small-step-budget" not in offers
    assert [
        s.entry.name
        for s in offers.values()
        if s.tier == "try_first" and "prefill_batch_tokens" in s.entry.levers
    ] in ([], ["step-budget-down"])


def test_stream_interval_is_offered_to_a_saturated_gpu_running_many_streams(
    tmp_path: Path,
) -> None:
    offers = _closed_loop_offers("throughput", tmp_path)

    assert "streaming every 8 tokens" in offers["stream-interval"].reason


def test_lowering_concurrency_in_a_closed_loop_could_help_and_names_the_queueing(
    tmp_path: Path,
) -> None:
    offers = _closed_loop_offers("tpot", tmp_path, ttft_p95_limit_ms=2000.0)

    lower = offers["lower-concurrency"]
    assert lower.tier == "could_help"
    assert lower.cost is not None and "in a closed loop, each request waits about" in lower.cost
    assert "weak in a closed loop" in lower.cost
    assert lower.demoted is not None and lower.demoted.startswith("in a closed loop, each request")


def test_lowering_concurrency_is_not_demoted_for_a_preempting_throughput_closed_loop(
    tmp_path: Path,
) -> None:
    def lower(goal: Goal, **declared: Any) -> Suggestion:
        with active(AnalysisContext(goal=goal, **declared)):
            run = recorded_situation(GEMMA_RUN, GEMMA_CLOSED_LOOP)
            measurement = dataclasses.replace(run.measurement, preemptions=40.0)
            diagnosis = classify(
                measurement, run.settings, None, engine=VLLM, version="0.30.0",
                gpu_state=None, profile=None,
            )  # fmt: skip
            situation = dataclasses.replace(run, measurement=measurement, diagnosis=diagnosis)
            assert situation.finding("kv_capacity").threshold == "preemption_rate"
            offered = suggestions(
                VLLM, situation, current=run.settings, project_path=tmp_path, version="0.30.0"
            )
        return {s.entry.name: s for s in offered.suggestions}["lower-concurrency"]

    kept = lower("throughput")
    assert kept.tier != "could_help"
    assert "waits" not in (kept.cost or "") and "weak in a closed loop" not in (kept.cost or "")
    assert lower("throughput", ttft_p95_limit_ms=2000.0).tier == "could_help"


def test_fast_prefill_is_offered_where_layers_share_kv_and_not_where_none_do() -> None:
    anatomy = gemma_e4b_anatomy()
    assert anatomy.kv_shared_layers > 0 and anatomy.layers
    facts = WorkloadFacts(prompts=(), output_tokens=384, context_limit=8192, model_type="gemma4")
    with active(AnalysisContext(goal="ttft", anatomy=anatomy)):
        gemma = recorded_situation(GEMMA_RUN, facts)
        reason = fast_prefill_applies(gemma, VLLM)
        stepped = with_fast_prefill(gemma, VLLM)
    assert reason is not None
    assert f"{anatomy.kv_shared_layers} of {anatomy.layers} layers reuse earlier KV" in reason
    assert "--kv-sharing-fast-prefill" in stepped.extra_args

    qwen, context = l4_situation("20261014T162451Z-db0f", "ttft")
    context.anatomy = dataclasses.replace(anatomy, kv_shared_layers=0)
    with active(context):
        assert fast_prefill_applies(qwen, VLLM) is None


L4_GEMMA_2 = REPLAYS / "l4-gemma-rerun"
L4_QWEN_LONG = REPLAYS / "l4-qwen-longctx"


@pytest.mark.parametrize(
    "run", ["20261017T005701Z-42ba", "20261017T011330Z-c4ca", "20261017T012916Z-f412"]
)
def test_the_stored_tpot_runs_load_with_their_ceilings_and_keep_the_histogram_buckets(
    run: str,
) -> None:
    run_dir = L4_GEMMA_2 / run
    measurement = replay_measurement(run_dir, gemma_e4b_anatomy())
    assert measurement.ceilings is not None and measurement.ceilings.gpu == "L4"
    for when in ("before", "after"):
        prom = (run_dir / f"metrics_{when}.prom").read_text()
        assert 'vllm:time_to_first_token_seconds_bucket{engine="0",le="' in prom
        assert 'model_name="model"' in prom
    blocks = [json.loads(line) for line in (run_dir / "prompt_blocks.jsonl").open()]
    assert blocks and all(row["block_tokens"] == 16 for row in blocks)


def test_the_stored_long_prompt_run_loads_from_its_0_3_12_metrics_and_scrapes() -> None:
    run_dir = L4_QWEN_LONG / "20261015T120253Z-cb0f"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    measurement = measurement_from_metrics(metrics)
    assert measurement.ttft_p95_ms is not None
    signals = read_signals((run_dir / "metrics_after.prom").read_text(), VLLM.signals)
    assert signals is not None
    assert "_bucket{" in (run_dir / "metrics_after.prom").read_text()


def test_the_stored_start_logs_keep_the_phase_lines_and_the_kv_memory_hint() -> None:
    log = (L4_GEMMA_2 / "20261017T011501Z-45f2" / "server.log").read_text()
    assert re.search(r"Available KV cache memory: -[\d.]+ GiB", log)
    qwen = (REPLAYS / "l4-qwen3-startcheck" / "server.log").read_text()
    assert "--kv-cache-memory=" in qwen
    for phase in ("Loading weights took", "torch.compile took", "Graph capturing finished"):
        assert phase in (L4_GEMMA_2 / "20261017T005701Z-42ba" / "server.log").read_text()


STARTS = {
    "20261017T005701Z-42ba": "cold",
    "20261017T011321Z-6cd2": "cache_hit",
    "20261017T011330Z-c4ca": "new_set_bf16",
    "20261017T012344Z-262e": "new_set_bf16",
    "20261017T012916Z-f412": "new_set_fp8",
}


@pytest.mark.parametrize(("run", "expected"), STARTS.items())
def test_a_real_vllm_030_start_reads_as_its_phases_and_class(run: str, expected: str) -> None:
    from tensward.engines.vllm.start_phases import cache_state, phases, start_class

    log = (L4_GEMMA_2 / run / "server.log").read_text()
    seen = phases(log)
    assert seen is not None
    settings = json.loads((L4_GEMMA_2 / run / "run.json").read_text())["settings"]
    ordered = ("graphs_1", "kv_alloc", "gap", "graphs_2", "ready")
    assert all(name in seen for name in ordered) and {"spawn", "weights", "compile"} <= seen.keys()
    assert [seen[name] for name in ordered] == sorted(seen[name] for name in ordered)
    assert ("mm_warmup" in seen) == (run != "20261017T012344Z-262e")
    assert start_class(seen, cache_state(log), settings.get("quantization")) == expected


def test_the_load_estimate_uses_the_median_of_this_projects_starts_of_the_same_class(
    tmp_path: Path,
) -> None:
    for name, start_class, ready_s in (
        ("20261017T010000Z-aaaa", "new_set_bf16", 200.0),
        ("20261017T010100Z-bbbb", "new_set_bf16", 300.0),
        ("20261017T010200Z-cccc", "new_set_fp8", 900.0),
    ):
        run = tmp_path / "runs" / name
        run.mkdir(parents=True)
        (run / "window.json").write_text(
            json.dumps({"startup": {"start_class": start_class, "ready_s": ready_s}})
        )
    context = AnalysisContext(launched_s=0.0, load_s=100.0, measured_s=160.0, project=tmp_path)

    assert seconds_to_test(context) == pytest.approx(60 + 250)
    assert "a start like this one takes 250 s here (median of 2 starts" in str(
        time_to_test(context)
    )


def test_requests_waiting_at_the_kv_limit_in_the_first_minute_of_the_load_are_found() -> None:
    from tensward.engines.vllm.start_phases import load_queue

    log = (L4_GEMMA_2 / "20261017T012916Z-f412" / "server.log").read_text()
    clear = load_queue(log)
    assert clear.seen > 3 and clear.queued is None and clear.closed

    starved = log.replace("GPU KV cache usage: 5.7%", "GPU KV cache usage: 96.5%")
    found = load_queue(starved).queued
    assert found is not None and (found.waiting, found.kv_percent) == (85, 96.5)


def _tier(offered: Suggested, name: str) -> str | None:
    return next((s.tier for s in offered.suggestions if s.entry.name == name), None)


def _without_kv_capacity(situation: Situation) -> Situation:
    findings = tuple(
        dataclasses.replace(f, state="ok") if f.bottleneck == "kv_capacity" else f
        for f in situation.diagnosis.findings
    )
    return dataclasses.replace(
        situation, diagnosis=dataclasses.replace(situation.diagnosis, findings=findings)
    )


def test_under_tpot_the_gemma_run_reads_enough_kv_for_fp8_kv_and_the_pair_is_predicted(
    tmp_path: Path,
) -> None:
    situation, context = _gemma_cell("tpot")
    rows = [json.loads(line) for line in (GEMMA_RUN / "prompt_blocks.jsonl").open()]
    context.prompt_blocks = {i: (row["tokens"], tuple(row["blocks"])) for i, row in enumerate(rows)}
    assert not kv_held_back(situation) and situation.fired("kv_capacity")
    with active(context):
        reason = fp8_kv_decode_applies(situation, VLLM)
        assert isinstance(reason, str)
        assert "(0.36x) at 42% of the memory-bandwidth ceiling" in reason
        m = situation.measurement
        assert m.ceilings is not None
        cap = VLLM.default_concurrency("0.30.0", m.ceilings.gpu, m.gpu_memory_gib)
        pair = predict_pair(situation, linear_bytes=context.linear_weight_bytes, cap=cap)
    assert pair is not None and "with about 96 running" in pair.text
    assert pair.tpot == pytest.approx(-0.29, abs=0.01)
    assert pair.req_s == pytest.approx(0.41, abs=0.01)
    offered = _l4_offers(situation, context, tmp_path)
    assert _tier(offered, "kv-sharing-fast-prefill") == "could_help"


def test_without_the_capacity_warning_the_pair_is_predicted_by_the_step_bytes_alone() -> None:
    situation, context = _gemma_cell("tpot")
    situation = _without_kv_capacity(situation)
    assert not situation.fired("kv_capacity")
    with active(context):
        pair = predict_pair(situation, linear_bytes=context.linear_weight_bytes, cap=None)
    assert pair is not None and "at this cap" in pair.text
    assert pair.tpot == pytest.approx(-0.29, abs=0.01)
    assert pair.req_s == pytest.approx(0.41, abs=0.01)


@pytest.mark.parametrize("goal", ["ttft", "throughput"])
def test_outside_tpot_the_gemma_run_keeps_its_capacity_pair_and_gets_no_fp8_kv_decode(
    goal: Goal, tmp_path: Path
) -> None:
    offered = _gemma_offers(goal, tmp_path)
    names = {m.name for s in offered.suggestions for m in (s.entry, *s.members)}
    assert "fp8-kv-decode" not in names and {"fp8-weights", "fp8-kv-cache"} <= names


def test_fp8_kv_decode_needs_the_ceilings_and_a_quarter_of_the_weight_bytes() -> None:
    situation, context = _gemma_cell("tpot")
    ceilings = situation.measurement.ceilings
    assert ceilings is not None and ceilings.avg_running_batch is not None
    for changed in (
        dataclasses.replace(situation.measurement, ceilings=None),
        dataclasses.replace(
            situation.measurement,
            ceilings=dataclasses.replace(
                ceilings, avg_running_batch=0.2 / 0.36 * ceilings.avg_running_batch
            ),
        ),
    ):
        with active(context):
            assert (
                fp8_kv_decode_applies(dataclasses.replace(situation, measurement=changed), VLLM)
                is None
            )


def test_a_pressured_tpot_run_at_its_cap_gets_no_bytes_pair() -> None:
    situation, context = _gemma_cell("tpot")
    cap = situation.measurement.peak_running
    assert cap is not None
    pressured = dataclasses.replace(situation.measurement, preemptions=81.0)
    held = dataclasses.replace(
        _without_kv_capacity(situation), measurement=pressured,
        settings=dataclasses.replace(situation.settings, max_concurrent_requests=int(cap)),
    )  # fmt: skip
    assert kv_held_back(held)
    with active(context):
        assert predict_pair(held, linear_bytes=context.linear_weight_bytes, cap=None) is None


def test_fp8_kv_is_offered_from_the_stored_fp8_parent() -> None:
    anatomy = gemma_e4b_anatomy()
    run = L4_GEMMA_2 / "20261017T012916Z-f412"
    situation = recorded_situation(run, GEMMA_FACTS)
    situation = dataclasses.replace(situation, measurement=replay_measurement(run, anatomy))
    _, context = _gemma_cell("tpot")
    with active(context):
        reason = fp8_kv_decode_applies(situation, VLLM)
    assert isinstance(reason, str) and "of the memory-bandwidth ceiling" in reason


def test_async_scheduling_is_offered_only_where_the_run_turned_it_off() -> None:
    situation, context = _gemma_cell("tpot")
    with active(context):
        assert async_scheduling_applies(situation, VLLM) is None
        off = dataclasses.replace(
            situation, settings=dataclasses.replace(situation.settings, async_scheduling=False)
        )
        assert async_scheduling_applies(off, VLLM) is not None


def armed_for(parent: Path, step: Path) -> guard.Armed | None:
    record = json.loads((parent / "run.json").read_text())
    past = PastRun(parent.name, frozenset(record["engine_args"]), record["settings"], {}, None, {})
    step_args = frozenset(json.loads((step / "run.json").read_text())["engine_args"])
    return guard.arm(past, parent, step_args, VLLM)


@pytest.mark.parametrize(
    ("line", "gib"),
    [
        ("[gpu_worker.py:640] Available KV cache memory: 4.0 GiB", 4.0),
        ("[gpu_worker.py:640] Available KV cache memory: 7.98 GiB", 7.98),
        ("[gpu_worker.py:640] Available KV cache memory: -0.3 GiB", -0.3),
        (
            "[gpu_worker.py:559] Initial free memory 21.84 GiB, reserved 4.89 GiB memory "
            "for KV Cache as specified by kv_cache_memory_bytes config and skipped memory "
            "profiling. This does not respect the gpu_memory_utilization config.",
            4.89,
        ),
        ("Initial free memory 21.84 GiB, reserved 4.89 GiB memory for model weights", None),
        ("non-default args: {'kv_cache_memory_bytes': 5251144064}", None),
        ("[gpu_worker.py:888] Free memory on device (21.84/22.03 GiB) on startup.", None),
    ],
)
def test_the_kv_memory_line_is_read_at_every_precision_vllm_logs(
    line: str, gib: float | None
) -> None:
    assert kv_available_gib(line) == gib
    assert _logged_gib(VLLM.kv_memory_log, line) == (gib if gib is None or gib >= 0 else None)
    for run, logged in (("20261017T012344Z-262e", 0.36), ("20261017T011321Z-6cd2", 4.66)):
        assert kv_available_gib((L4_GEMMA_2 / run / "server.log").read_text()) == logged


def test_the_guard_stops_the_ttft_rerun_s_text_only_step_on_its_gib() -> None:
    parent, step = L4_GEMMA_2 / "20261017T011321Z-6cd2", L4_GEMMA_2 / "20261017T012344Z-262e"
    armed = armed_for(parent, step)

    assert armed is not None and not armed.shrinks_kv
    reason = guard.verdict(armed, (step / "server.log").read_text(), VLLM)
    assert reason is not None and reason.startswith(
        "stopped at start-up: KV pool 0.36 GiB against the 3.6"
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_a_smaller_step_budget_that_reports_more_tokens_from_less_memory_still_stops(
    tmp_path: Path,
    explicit: bool,
) -> None:
    def start(gib: float, tokens_per_gib: int) -> str:
        memory = (
            f"INFO Initial free memory 21.84 GiB, reserved {gib} GiB memory for KV Cache "
            "as specified by kv_cache_memory_bytes config and skipped memory profiling."
            if explicit
            else f"INFO Available KV cache memory: {gib} GiB"
        )
        return memory + f"\nINFO GPU KV cache size: {round(gib * tokens_per_gib):,} tokens\n"

    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "server.log").write_text(start(4.0, 27_100))  # budget 2048
    (parent / "metrics.json").write_text(json.dumps({"peak_kv_usage": 0.9}))
    (parent / "run.json").write_text(json.dumps({"engine_args": [], "settings": {}}))
    armed = armed_for(parent, parent)
    assert armed is not None

    step = start(1.6, 44_800)  # budget 512: more tokens per GiB, fewer GiB
    assert (kv_pool_tokens(step) or 0) > guard.STOP_SHARE * armed.used_tokens
    reason = guard.verdict(armed, step, VLLM)
    assert reason is not None and "KV pool 1.60 GiB against the 3.60 GiB" in reason
    assert guard.verdict(armed, start(2.4, 44_800), VLLM) is None


def test_a_start_with_no_memory_for_its_cache_is_left_to_the_engine_s_own_refusal() -> None:
    step = L4_GEMMA_2 / "20261017T011501Z-45f2"
    armed = armed_for(L4_GEMMA_2 / "20261017T011321Z-6cd2", L4_GEMMA_2 / "20261017T012344Z-262e")
    log = (step / "server.log").read_text()

    assert armed is not None and no_kv_memory(log)
    assert guard.verdict(armed, log, VLLM) is None


def eager_starved_gemma() -> tuple[Situation, AnalysisContext]:
    facts = WorkloadFacts(prompts=(), output_tokens=384, context_limit=8192, model_type="gemma4",
                          media_encoders=True, encoder_gib=0.87)  # fmt: skip
    situation = recorded_situation(L4_GEMMA / "20261016T175324Z-bf38", facts)
    eager = dataclasses.replace(situation.settings, cuda_graphs=False, media_inputs=None,
                                extra_args={})  # fmt: skip
    return (
        dataclasses.replace(situation, settings=eager),
        AnalysisContext(anatomy=gemma_e4b_anatomy()),
    )


def test_cuda_graphs_on_a_starved_pool_come_with_the_text_model_sized_in_bytes() -> None:
    situation, context = eager_starved_gemma()
    towers = (freed_bytes(context.anatomy) or {})["towers"]
    unsized = dataclasses.replace(
        situation, measurement=dataclasses.replace(situation.measurement, kv_available_gib=None)
    )

    with active(context):
        reason = cuda_graphs_applies(situation, VLLM)
        args = VLLM.engine_args_between(situation.settings, with_cuda_graphs(situation, VLLM))
        plain = VLLM.engine_args_between(unsized.settings, with_cuda_graphs(unsized, VLLM))

    assert reason is not None and "`no-media-encoders` frees" in reason
    assert "language-model-only" in args
    assert f"kv-cache-memory-bytes={round(0.36 * 2**30) + towers}" in args
    assert "language-model-only" not in plain


def hinted_run(
    tmp_path: Path, *, peak_used_mib: float | None = None
) -> tuple[Situation, AnalysisContext]:
    """The starved gemma step with the Qwen3 start check's log: pool 3.16 GiB, the engine's
    "fully utilize" figure 4.35 GiB, 50 preemptions."""
    project = tmp_path / REPLAYED
    run = project / "runs" / "20261016T175324Z-bf38"
    shutil.copytree(L4_GEMMA / "20261016T175324Z-bf38", run)
    (run / "server.log").write_text((REPLAYS / "l4-qwen3-startcheck" / "server.log").read_text())
    if peak_used_mib is not None:
        (run / "window.json").write_text(
            json.dumps({"memory": {"peak_used_mib": peak_used_mib, "total_mib": 22_560.0}})
        )
    situation = recorded_situation(run, GEMMA_FACTS)
    situation = dataclasses.replace(
        situation,
        measurement=dataclasses.replace(situation.measurement, kv_available_gib=3.16),
        settings=dataclasses.replace(situation.settings, kv_memory_fraction=0.92, extra_args={}),
    )
    return situation, AnalysisContext(project=project, new_run=run)


def test_an_explicit_kv_size_takes_the_engine_s_figure_less_the_margin(tmp_path: Path) -> None:
    situation, context = hinted_run(tmp_path)

    with active(context):
        reason = explicit_kv_size_applies(situation, VLLM)
        settings = with_explicit_kv_size(situation, VLLM)

    assert reason is not None and "4.35 GiB against the 3.16 GiB" in reason
    assert settings.extra_args[kvsize.KV_FLAG] == "3600001536"  # 4673743360 less 1 GiB
    assert "3.35 GiB" in reason


def test_an_explicit_kv_size_replaces_the_larger_share(tmp_path: Path) -> None:
    situation, context = hinted_run(tmp_path)
    with active(context):
        assert more_kv_memory_applies(situation, VLLM) is not None  # +0.19 GiB, a share +0.66
        starved = dataclasses.replace(situation.measurement, kv_available_gib=2.0)
        assert (
            more_kv_memory_applies(dataclasses.replace(situation, measurement=starved), VLLM)
            is None
        )


def test_the_sampled_peak_caps_the_engine_s_figure(tmp_path: Path) -> None:
    peak = 20_700.0
    situation, context = hinted_run(tmp_path, peak_used_mib=peak)
    kv = round(3.16 * 2**30)

    with active(context):
        size = int(with_explicit_kv_size(situation, VLLM).extra_args[kvsize.KV_FLAG])

    assert size == round(22_560 * 2**20 - (peak * 2**20 - kv) - kvsize.MARGIN_BYTES) < 4673743360


def test_a_run_whose_requests_all_ran_gets_no_explicit_size_from_the_engine_s_figure(
    tmp_path: Path,
) -> None:
    situation, context = hinted_run(tmp_path)
    measurement = dataclasses.replace(situation.measurement, preemptions=0, mean_waiting=0.0,
                                      peak_running=96, peak_in_flight=96)  # fmt: skip
    calm = dataclasses.replace(situation, measurement=measurement)

    with active(context):
        assert explicit_kv_size_applies(calm, VLLM) is None


L4_QWEN_QUEUE = REPLAYS / "l4-qwen3-queue"
QUEUE_BF16, QUEUE_FP8 = "20261022T002722Z-5a8e", "20261022T005604Z-d497"
QUEUE_FACTS = WorkloadFacts(
    prompts=(), output_tokens=384, context_limit=40960, model_type="qwen3",
    load="96 concurrent clients", max_inflight=96,
)  # fmt: skip


@pytest.mark.parametrize("run", [QUEUE_BF16, QUEUE_FP8])
def test_first_tokens_queued_at_the_cap_are_offered_no_smaller_step_budget(run: str) -> None:
    situation = recorded_situation(L4_QWEN_QUEUE / run, QUEUE_FACTS)

    with active(AnalysisContext(goal="tpot")):
        assert step_budget_down_applies(situation, VLLM) is None


@pytest.mark.parametrize(("run", "offered"), [(QUEUE_BF16, True), (QUEUE_FP8, False)])
def test_requests_waiting_for_the_cap_are_not_kv_space_holding_them_back(
    run: str, offered: bool
) -> None:
    situation = recorded_situation(L4_QWEN_QUEUE / run, QUEUE_FACTS)

    with active(AnalysisContext(goal="tpot", new_run=L4_QWEN_QUEUE / run)):
        reason = explicit_kv_size_applies(situation, VLLM)

    assert (reason is not None) == offered


def test_a_full_window_step_is_timed_with_the_full_window(monkeypatch: pytest.MonkeyPatch) -> None:
    run = L4_QWEN_QUEUE / QUEUE_FP8
    declared = json.loads((run / "run.json").read_text())["analysis"]
    arrival = ArrivalSpec(kind="closed_loop", concurrency=96)
    monkeypatch.setattr(
        "tensward.history.declared_workload",
        lambda project: SimpleNamespace(request_count=800, arrival=arrival),
    )
    context = AnalysisContext(
        goal="tpot", launched_s=declared["launched_s"], load_s=declared["load_s"],
        measured_s=declared["measured_s"], new_run=run, project=L4_QWEN_QUEUE,
    )  # fmt: skip

    assert seconds_to_test(context) == pytest.approx(549, abs=1)  # a 132 s window planned
    # the bundle step measured from this run took 799 s, 642 s of them after its start
    assert seconds_to_test(context, full=True) == pytest.approx(863, abs=1)
    assert full_window("tensward analyse --engine-arg stream-interval=8 --window full")


@pytest.mark.parametrize("goal", ["throughput", "tpot"])
def test_a_hybrid_model_is_offered_no_change_that_enlarges_the_token_pool(
    goal: Goal, tmp_path: Path
) -> None:
    context = AnalysisContext(goal=goal)
    with active(context):
        situation = recorded_situation(L4_GEMMA / "20261016T172654Z-ed76", GEMMA_CLOSED_LOOP)
        plain = suggestions(VLLM, situation, current=situation.settings, project_path=tmp_path,
                            version="0.30.0")  # fmt: skip
    context.anatomy = checkpoint_anatomy("granite4_h_tiny")
    with active(context):
        hybrid = suggestions(VLLM, situation, current=situation.settings, project_path=tmp_path,
                             version="0.30.0")  # fmt: skip

    assert "fp8-kv-cache" in {s.entry.name for s in plain.suggestions}
    assert "fp8-kv-cache" not in {s.entry.name for s in hybrid.suggestions}
    why = dict(hybrid.not_applicable)["fp8-kv-cache"]
    assert "state for each sequence" in why
    assert {s.entry.name for s in plain.suggestions} - {"fp8-kv-cache"} == {
        s.entry.name for s in hybrid.suggestions
    }


@pytest.mark.parametrize("goal", ["throughput", "tpot"])
def test_an_mxfp4_checkpoint_is_offered_no_weight_precision_change(
    goal: Goal, tmp_path: Path
) -> None:
    situation, context = l4_situation(L4_8255, goal)
    weights = {"fp8-weights", "quantized-checkpoint", "bnb-4bit-weights"}
    stored = {s.entry.name for s in _l4_offers(situation, context, tmp_path).suggestions}
    quantized = dataclasses.replace(
        situation, facts=dataclasses.replace(situation.facts, quantization="mxfp4 mxfp4 group 32")
    )

    offered = {s.entry.name for s in _l4_offers(quantized, context, tmp_path).suggestions}

    assert weights & stored and not weights & offered


def dense_anatomy(config: Mapping[str, Any]) -> ModelAnatomy:
    """A dense decoder's anatomy from its config alone: 16-bit tensors of the config's shapes."""
    hidden, heads = config["hidden_size"], config["num_attention_heads"]
    head = config.get("head_dim", hidden // heads)
    kv, mlp, vocab = (
        config["num_key_value_heads"] * head,
        config["intermediate_size"],
        config["vocab_size"],
    )
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (vocab, hidden), "model.norm.weight": (hidden,),
        **({} if config["tie_word_embeddings"] else {"lm_head.weight": (vocab, hidden)}),
    }  # fmt: skip
    for layer in range(config["num_hidden_layers"]):
        for name, shape in (
            ("self_attn.q_proj", (heads * head, hidden)), ("self_attn.k_proj", (kv, hidden)),
            ("self_attn.v_proj", (kv, hidden)), ("self_attn.o_proj", (hidden, heads * head)),
            ("mlp.gate_proj", (mlp, hidden)), ("mlp.up_proj", (mlp, hidden)),
            ("mlp.down_proj", (hidden, mlp)),
        ):  # fmt: skip
            shapes[f"model.layers.{layer}.{name}.weight"] = shape
    tensors = {name: ("BF16", shape, 2 * math.prod(shape)) for name, shape in shapes.items()}
    return build_anatomy(dict(config), None, tensors)


GRANITE_CONFIG = {
    "model_type": "granite", "hidden_size": 4096, "num_hidden_layers": 40,
    "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 12800,
    "vocab_size": 49159, "tie_word_embeddings": True, "max_position_embeddings": 131072,
}  # fmt: skip
QWEN25_7B_CONFIG = {
    "model_type": "qwen2", "hidden_size": 3584, "num_hidden_layers": 28, "num_attention_heads": 28,
    "num_key_value_heads": 4, "intermediate_size": 18944, "vocab_size": 152064,
    "tie_word_embeddings": False, "max_position_embeddings": 32768,
}  # fmt: skip
QWEN3_06B_CONFIG = {
    "model_type": "qwen3", "hidden_size": 1024, "num_hidden_layers": 28, "num_attention_heads": 16,
    "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 3072, "vocab_size": 151936,
    "tie_word_embeddings": True, "max_position_embeddings": 40960,
}  # fmt: skip
CB0F = L4_QWEN_LONG / "20261015T120253Z-cb0f"
CB0F_PROMPT_SHARE = 0.34  # not measured in 0.3.12: between the datasheet bounds 0.17 and 0.51
# granite-3.3-8b on cb0f's 16-client tool-call workload, estimated: a 21.6k-token pool runs
# about 5 of the 16 sessions and the rest wait, with TTFT almost all queueing
GRANITE_SIGNALS: dict[str, Any] = {
    "kv_capacity_tokens": 21_600.0, "kv_available_gib": 3.3, "kv_blocks": 1350.0,
    "kv_max_concurrency": 2.6, "peak_kv_usage": 0.99, "mean_kv_usage": 0.95, "peak_running": 7.0,
    "mean_running": 5.0, "peak_waiting": 13.0, "mean_waiting": 11.0, "preemptions": 4.0,
    "queue_share": 0.85, "request_throughput": 0.24, "output_throughput": 45.0,
    "tpot_p50_ms": 88.0, "tpot_p95_ms": 110.0, "ttft_p50_ms": 46_000.0, "ttft_p95_ms": 65_000.0,
    "server_ttft_p50_ms": 45_000.0, "step_ms": 95.0, "prefix_cache_hit_rate": 0.43,
    "e2e_p95_ms": 85_000.0, "client_batch": 5.0, "goodput": 0.24, "prompt_work_share": 0.3,
}  # fmt: skip
GRANITE_LIMITS: dict[str, dict[str, float]] = {
    "tpot": {"ttft_p95_limit_ms": 100_000.0, "max_req_s_loss": 0.10},
    "ttft": {"tpot_p95_limit_ms": 165.0, "max_req_s_loss": 0.10},
    "throughput": {"ttft_p95_limit_ms": 100_000.0, "tpot_p95_limit_ms": 165.0},
}


def granite_situation(
    goal: Goal = "tpot", *, tools: bool = True
) -> tuple[Situation, AnalysisContext]:
    """granite-3.3-8b on a tool-call workload, short of KV space: the stored Qwen2.5-7B run
    ``cb0f`` (the same L4, 16 closed-loop clients offering tools) with its KV signals replaced
    by the granite estimate, and the context of the track for ``goal``."""
    anatomy = dense_anatomy(GRANITE_CONFIG)
    metrics = json.loads((CB0F / "metrics.json").read_text())
    record = json.loads((CB0F / "run.json").read_text())
    measurement = dataclasses.replace(replay_measurement(CB0F, anatomy), **GRANITE_SIGNALS)
    assert measurement.ceilings is not None
    measurement = dataclasses.replace(
        measurement,
        ceilings=dataclasses.replace(
            measurement.ceilings, avg_running_batch=5.0, kv_bytes_per_sequence=163_840 * 6_100
        ),
        tool_calls=ToolCallStats(**metrics["tool_calls"]) if tools else None,
    )
    settings = Settings.from_json({**record["settings"], "tool_parser": "granite"})
    context = AnalysisContext(
        goal=goal, anatomy=anatomy, linear_weight_bytes=(freed_bytes(anatomy) or {})["quantizable"],
        answer_options=options_from(ENGINE_SUPPORT, 8.9), **GRANITE_LIMITS[goal],
    )  # fmt: skip
    gpu = record["gpus_identity"][0]["name"]
    with active(context):
        diagnosis = classify(
            measurement, settings, None, engine=VLLM, version="0.30.0",
            gpu_state=GpuState(**metrics["diagnosis"]["gpu_state"]),
            profile=profile_for(VLLM, record["platform"], gpu, "0.30.0"),
        )  # fmt: skip
    facts = WorkloadFacts(
        prompts=(), output_tokens=256, context_limit=131072, model_type="granite",
        offers_tools=tools, load="16 concurrent clients", max_inflight=16,
    )  # fmt: skip
    situation = Situation(measurement=measurement, facts=facts, settings=settings,
                          diagnosis=diagnosis)  # fmt: skip
    return situation, context


def _members(suggestion: Suggestion | None) -> list[str]:
    return [m.name for m in suggestion.members] if suggestion is not None else []


@pytest.mark.parametrize("tools", [True, False])
def test_under_tpot_a_run_short_of_kv_space_leads_with_the_held_batch(
    tools: bool, tmp_path: Path
) -> None:
    situation, context = granite_situation("tpot", tools=tools)
    assert kv_held_back(situation)
    offered = _l4_offers(situation, context, tmp_path)
    leading = first_try(situation.diagnosis, offered)
    assert _members(leading) == ["fp8-weights", "fp8-kv-decode", "lower-concurrency"]
    assert leading is not None and leading.command is not None and leading.predicted is not None
    assert {"max-num-seqs=7", "kv-cache-dtype=fp8"} <= command_args(leading.command)
    assert not any(s.entry.name == "lower-concurrency" for s in offered.suggestions)
    assert leading.predicted.tpot == pytest.approx(-0.343, abs=0.005)
    assert leading.predicted.req_s == pytest.approx(0.523, abs=0.005)
    assert leading.reason.startswith(
        "Held batch: cap 7, predicted TPOT -34%, req/s +52% (floor -5%), TTFT p99 "
    )
    for bound in (
        "with the start-up wave (bound 97.5 s)",
        "s (limit 100.0 s)",
        "s (bound 127.5 s).",
    ):
        assert bound in leading.reason


def _held_cap(situation: Situation, context: AnalysisContext, k: int) -> tuple[bool, str]:
    changes: dict[str, dict[str, Any]] = {
        "fp8-weights": {"quantization": "fp8_per_channel"},
        "fp8-kv-decode": {"kv_cache_dtype": "fp8"},
        "lower-concurrency": {},
    }
    members = [
        (name, dataclasses.replace(situation.settings, max_concurrent_requests=k, **change))
        for name, change in changes.items()
    ]
    with active(context):
        predicted = predict_stack(
            situation, members, linear_bytes=context.linear_weight_bytes, cap=256
        )
        return _held_figures(predicted, situation, k)


def test_the_held_batch_takes_the_smallest_cap_within_the_ttft_p99_bound(tmp_path: Path) -> None:
    situation, context = granite_situation("tpot")
    quick = dataclasses.replace(situation.measurement, ttft_p95_ms=20_000.0)
    situation = dataclasses.replace(situation, measurement=quick)
    offered = _l4_offers(situation, context, tmp_path)
    leading = first_try(situation.diagnosis, offered)
    assert leading is not None and leading.settings is not None
    k = leading.settings.max_concurrent_requests
    assert k is not None and k > 7
    below, why = _held_cap(situation, context, k - 1)
    assert not below and "(bound 30.0 s)" in why
    assert _held_cap(situation, context, k)[0]


def test_no_held_batch_where_no_cap_keeps_the_ttft_p99_bound(tmp_path: Path) -> None:
    situation, context = granite_situation("tpot")
    situation = dataclasses.replace(
        situation, measurement=dataclasses.replace(situation.measurement, ttft_p95_ms=5_000.0)
    )
    offered = _l4_offers(situation, context, tmp_path)
    assert not any("lower-concurrency" in _members(s) for s in offered.suggestions)
    why = dict(offered.not_applicable)["held-batch"]
    assert why.startswith("no concurrency cap holds the batch here: cap 15, ")


def test_the_cap_model_predicts_the_measured_cap_step_and_the_floor_refuses_it() -> None:
    parent, child = (L4_GEMMA_2 / run for run in ("20261017T005701Z-42ba", "20261017T011330Z-c4ca"))
    _, context = _gemma_cell("tpot")
    situation = dataclasses.replace(
        recorded_situation(parent, GEMMA_FACTS),
        measurement=replay_measurement(parent, gemma_e4b_anatomy()),
    )
    context = dataclasses.replace(context, max_req_s_loss=0.10)
    before, after = (json.loads((run / "metrics.json").read_text()) for run in (parent, child))
    capped = dataclasses.replace(situation.settings, max_concurrent_requests=48)
    with active(context):
        predicted = predict(situation, capped, entry="lower-concurrency",
                            linear_bytes=context.linear_weight_bytes)  # fmt: skip
        assert predicted is not None and predicted.tpot is not None and predicted.req_s is not None
        passes, why = _held_figures(predicted, situation, 48)
    measured_tpot = after["tpot_p50_ms"] / before["tpot_p50_ms"] - 1  # -29% (p95: -31%)
    measured_req_s = after["request_throughput"] / before["request_throughput"] - 1  # -30%
    assert predicted.tpot == pytest.approx(measured_tpot, abs=0.05)
    assert predicted.req_s == pytest.approx(measured_req_s, abs=0.05)
    assert not passes and "req/s -32% (floor -5%)" in why


GEMMA_TPOT_TRACK = {"max_req_s_loss": 0.10, "ttft_p95_limit_ms": 1379.55}


def test_without_kv_pressure_there_is_no_held_batch(tmp_path: Path) -> None:
    situation, context = _gemma_cell("tpot")
    offered = _l4_offers(situation, dataclasses.replace(context, **GEMMA_TPOT_TRACK), tmp_path)
    assert not any("lower-concurrency" in _members(s) for s in offered.suggestions)
    assert "held-batch" not in dict(offered.not_applicable)


def test_the_gemma_run_s_tpot_pair_is_one_step_and_not_fp8_kv_first(tmp_path: Path) -> None:
    situation, context = _gemma_cell("tpot")
    context = dataclasses.replace(context, **GEMMA_TPOT_TRACK)
    with active(context):
        assert _kv_bound_pair(situation, VLLM, context, "0.30.0") is not None
    offered = _l4_offers(situation, context, tmp_path)
    assert {"fp8-weights", "fp8-kv-decode"} <= set(
        _members(first_try(situation.diagnosis, offered))
    )


def test_an_unpressured_tool_run_gets_no_fp8_pair(tmp_path: Path) -> None:
    situation, context = _gemma_cell("tpot")
    calls = json.loads((CB0F / "metrics.json").read_text())["tool_calls"]
    tools = dataclasses.replace(situation.measurement, tool_calls=ToolCallStats(**calls))
    situation = dataclasses.replace(situation, measurement=tools)
    assert not kv_held_back(situation)
    with active(context):
        assert _kv_bound_pair(situation, VLLM, context, "0.30.0") is None
    offered = _l4_offers(situation, context, tmp_path)
    assert not any(
        {"fp8-weights", "fp8-kv-decode"} <= set(_members(s)) for s in offered.suggestions
    )


@pytest.mark.parametrize("goal", ["ttft", "throughput"])
def test_with_tools_a_run_short_of_kv_space_measures_the_fp8_pair_in_one_step(
    goal: Goal, tmp_path: Path
) -> None:
    situation, context = granite_situation(goal)
    offered = _l4_offers(situation, context, tmp_path)
    leading = first_try(situation.diagnosis, offered)
    assert _members(leading) == ["fp8-weights", "fp8-kv-cache"]
    assert not offered.kv_first
    section = next_steps_section(situation.diagnosis, offered, retained=False)
    with active(context):
        blocks = to_json(Report(run_id="r", ran="now", sections=(section,)))["sections"][0][
            "blocks"
        ]
    shown = next(b for b in blocks if b.get("key") == "fp8-weights+fp8-kv-cache")
    assert shown["answer_risk"] == "medium"


def test_with_tools_a_precision_change_risks_the_answers_most_under_2b_parameters() -> None:
    situation, context = l4_situation(L4_8255, "tpot")
    entry = next(e for e in VLLM.playbook() if e.name == "fp8-kv-decode")
    offer = Suggestion(entry=entry, reason="", tier="try_first", settings=situation.settings)
    risks = {}
    for name, config in (("qwen3-0.6b", QWEN3_06B_CONFIG), ("granite", GRANITE_CONFIG)):
        with active(dataclasses.replace(context, anatomy=dense_anatomy(config))):
            risks[name] = answer_risk(entry, offer, tools=True)
    assert risks == {"qwen3-0.6b": "high", "granite": "medium"}


def test_after_the_judge_rejects_the_pair_a_tool_run_tries_fp8_kv_alone(tmp_path: Path) -> None:
    situation, context = granite_situation("ttft")
    pair = first_try(situation.diagnosis, _l4_offers(situation, context, tmp_path / "before"))
    assert pair is not None and pair.command is not None
    change = tuple(sorted(command_args(pair.command)))
    rejected = Rejection(run="r1", engine_args=change, change=change, made_from=CB0F.name,
                         reason="judge: worse", at="")  # fmt: skip
    context = dataclasses.replace(context, rejected=(rejected,))
    offered = _l4_offers(situation, context, tmp_path)
    leading = first_try(situation.diagnosis, offered)
    assert leading is not None and leading.entry.name == "fp8-kv-cache" and not leading.members
    assert leading is offered.led


def test_a_saturated_prefill_heavy_run_is_offered_fp8_weights(tmp_path: Path) -> None:
    anatomy = dense_anatomy(QWEN25_7B_CONFIG)
    situation = recorded_situation(CB0F, WorkloadFacts(
        prompts=(), output_tokens=256, context_limit=8192, model_type="qwen2", offers_tools=True,
        load="16 concurrent clients", max_inflight=16,
    ))  # fmt: skip
    assert situation.fired("gpu_saturated") and not situation.fired("decode_bandwidth")
    context = AnalysisContext(
        goal="throughput", anatomy=anatomy, answer_options=options_from(ENGINE_SUPPORT, 8.9),
        linear_weight_bytes=(freed_bytes(anatomy) or {})["quantizable"],
    )  # fmt: skip
    offered = {}
    for share in (CB0F_PROMPT_SHARE, 0.2):
        measurement = dataclasses.replace(
            replay_measurement(CB0F, anatomy), prompt_work_share=share
        )
        at = dataclasses.replace(situation, measurement=measurement)
        offered[share] = {s.entry.name: s for s in _l4_offers(at, context, tmp_path).suggestions}
    weights = offered[CB0F_PROMPT_SHARE]["fp8-weights"]
    assert weights.reason.startswith(
        "the GPU was saturated, with prompts 34% of its work, decode at 48% of the "
        "memory-bandwidth ceiling; at compute capability 8.9 8-bit weights also halve"
    )
    assert "fp8-weights" not in offered[0.2]


RERUN_DEFAULT = L4_GEMMA_2 / "20261017T005701Z-42ba"
RERUN_FP8 = L4_GEMMA_2 / "20261017T012916Z-f412"


def _rerun_keep(tmp_path: Path, goal: Goal) -> tuple[Suggested, dict[str, str]]:
    """The fp8-weights run of the gemma re-run against the full default run it was made from,
    in a project of 96 closed-loop clients: what it offers and its keep section."""
    from registration_fixtures import REGISTRATION_CONFIG, write_config

    project = tmp_path / REPLAYED / goal
    shutil.copytree(RERUN_DEFAULT, project / "runs" / RERUN_DEFAULT.name)
    step = project / "runs" / RERUN_FP8.name
    step.mkdir()
    shutil.copy(RERUN_FP8 / "requests.jsonl", step / "requests.jsonl")
    loop = {**REGISTRATION_CONFIG["workload"],
            "arrival": {"kind": "closed_loop", "concurrency": 96}}  # fmt: skip
    config = write_config(tmp_path / f"{goal}.json", {**REGISTRATION_CONFIG, "workload": loop})
    (project / "project.json").write_text(json.dumps({"config_path": str(config)}))
    record = json.loads((RERUN_FP8 / "run.json").read_text())
    parent = json.loads((RERUN_DEFAULT / "run.json").read_text())
    args = tuple(record["engine_args"])
    context = AnalysisContext(goal=goal, project=project, new_run=step, engine_args=args,
                              flags=("--goal", goal))  # fmt: skip
    with active(context):
        situation = recorded_situation(RERUN_DEFAULT, GEMMA_CLOSED_LOOP)
        suggested = suggestions(
            VLLM, situation, current=situation.settings, project_path=project, version="0.30.0"
        )
        report = build_report(RunOutputs(
            run_id=step.name, ran="vLLM 0.30.0 on NVIDIA L4", measurement=situation.measurement,
            diagnosis=situation.diagnosis, suggested=suggested, rows=(), facts=situation.facts,
            settings=Settings.from_json(record["settings"]),
            subject=Subject(setup=CurrentSetup(source="defaults", settings=parent["settings"])),
            image=None, slo=NO_LIMITS, engine=VLLM, gpu="NVIDIA L4", source="", engine_args=args,
            retained=True,
        ))  # fmt: skip
    keep = next(s for s in report.sections if s.id == "keep")
    return suggested, {b.key: b.text for b in keep.blocks if isinstance(b, Text)}


@pytest.mark.parametrize("goal", ["ttft", "tpot"])
def test_fp8_weights_on_the_gemma_rerun_are_kept_on_ttft_after_both_ramps(
    goal: Goal, tmp_path: Path
) -> None:
    whole = window.compare_matched(RERUN_FP8, RERUN_DEFAULT, clients=96)
    assert not isinstance(whole, str)
    full = whole["ttft_p95_ms"]
    assert full.verdict == "within_noise"
    assert (round(full.change, 3), round(full.band, 3)) == (-0.151, 0.161)

    suggested, keep = _rerun_keep(tmp_path, goal)

    assert suggested.memory is not None and suggested.memory.parent is not None
    ramped = suggested.memory.parent.deltas["ttft_p95_ms"]
    rows = [window.read_rows(run) or [] for run in (RERUN_FP8, RERUN_DEFAULT)]
    assert ramped == window.post_ramp(*rows, clients=96, upto=1010)
    assert ramped.verdict == "better"
    assert (round(ramped.change, 3), round(ramped.band, 3)) == (-0.182, 0.115)
    assert "not_kept" not in keep and "kept_within_noise" not in keep


def test_within_noise_an_answer_changing_step_is_not_kept_and_a_lossless_one_is(
    tmp_path: Path,
) -> None:
    rows = _scaled((L4_STEPS / L4_PARENT / "requests.jsonl").read_text().splitlines()[:236], 0.98)
    _, lossless, goal = l4_keep(tmp_path / "a", "tpot", rows, ("max-num-seqs=8",), adopt=True)
    assert lossless["kept_within_noise"].startswith("Within noise (TPOT p95 −2% ±3%); ")
    assert goal["adopted_within_noise"]["metric"] == "tpot_p95_ms"
    record = json.loads((L4_STEPS / L4_PARENT / "run.json").read_text())
    fp8 = "kv-cache-dtype=fp8_e4m3"
    settings = VLLM.with_engine_arg(Settings.from_json(record["settings"]), fp8)
    _, changed, goal = l4_keep(tmp_path / "b", "tpot", rows, (fp8,), adopt=True, settings=settings)
    assert changed["not_kept"] == (
        "Tensward does not recommend keeping this change: within noise: TPOT p95 −2% ±3%, and it "
        "can change answers: within noise it is not kept."
    )
    assert "adopted_within_noise" not in goal


def test_a_held_batch_stopped_in_load_is_not_offered_again(tmp_path: Path) -> None:
    situation, context = granite_situation("tpot")
    batch = first_try(situation.diagnosis, _l4_offers(situation, context, tmp_path / "before"))
    assert batch is not None and batch.command is not None
    stopped = tmp_path / "runs" / "20261017T010000Z-0000"
    stopped.mkdir(parents=True)
    watch = {"engine_args": sorted(command_args(batch.command)), "reason": "waiting at full KV"}
    (stopped / "window.json").write_text(json.dumps({"load_watch": watch}))

    offered = _l4_offers(situation, context, tmp_path)

    assert not any("lower-concurrency" in _members(s) for s in offered.suggestions)


@pytest.mark.parametrize("goal", ["tpot", "ttft"])
def test_with_tools_on_a_checkpoint_under_2b_parameters_the_fp8_pair_waits_in_could_help(
    goal: Goal, tmp_path: Path
) -> None:
    situation, context = granite_situation(goal)
    small = dataclasses.replace(context, anatomy=dense_anatomy(QWEN3_06B_CONFIG))
    offered = _l4_offers(situation, small, tmp_path)

    tried = [s for s in offered.suggestions if s.tier == "try_first" and s.command]
    waiting = [s for s in offered.suggestions if s.tier == "could_help" and s.members]
    assert not any(len(s.members) > 1 for s in tried)
    assert not any(s.entry.name == "lower-concurrency" and s.members for s in tried)
    if goal == "ttft":
        assert [s.entry.name for s in tried if not s.members][0] == "fp8-kv-cache"
    assert waiting and all("under 2B parameters" in (s.demoted or "") for s in waiting)
    assert {"fp8-weights", "fp8-kv-decode" if goal == "tpot" else "fp8-kv-cache"} <= {
        m.name for s in waiting for m in s.members
    }


def _held_six_steps() -> tuple[Situation, AnalysisContext]:
    situation, context = l4_situation(L4_8255, "ttft")
    situation = _with_tool_calls(situation, L4_8255)
    m = situation.measurement
    assert m.ttft_p50_ms is not None and m.tpot_p50_ms is not None
    engine = m.ttft_p50_ms - 6 * m.tpot_p50_ms
    return dataclasses.replace(
        situation, measurement=dataclasses.replace(m, server_ttft_p50_ms=engine)
    ), context


def test_ttft_offers_rank_on_the_engine_s_figure_behind_a_six_step_parser_hold(
    tmp_path: Path,
) -> None:
    situation, context = _held_six_steps()
    offered = _l4_offers(situation, context, tmp_path)

    assert offered.ttft_basis is not None
    assert "ranked on the engine's time to first token" in offered.ttft_basis
    assert "holds the client's first content for 6 decode steps" in offered.ttft_basis
    section = next_steps_section(situation.diagnosis, offered, retained=False)
    assert any(getattr(b, "key", None) == "ttft_basis" for b in section.blocks)
    quiet = dataclasses.replace(situation.measurement, tool_calls=None)
    assert _l4_offers(dataclasses.replace(situation, measurement=quiet), context, tmp_path
                      ).ttft_basis is None  # fmt: skip


@pytest.mark.parametrize("goal", ["tpot", "throughput"])
def test_the_engine_s_ttft_is_the_ranking_basis_under_the_ttft_goal_only(
    goal: Goal, tmp_path: Path
) -> None:
    situation, context = _held_six_steps()
    other = dataclasses.replace(context, goal=goal)

    offered = _l4_offers(situation, other, tmp_path)

    assert offered.ttft_basis is None and offered.ttft_skipped is None


def test_first_token_ranks_on_the_first_streamed_chunk(tmp_path: Path) -> None:
    situation, context = _held_six_steps()
    run = tmp_path / "runs" / "20261018T000000Z-0000"
    run.mkdir(parents=True)
    rows = [
        {"request_id": f"r:{i}", "dispatch_ns": i, "outcome": "success",
         "first_chunk_ns": i + (100 + i) * 1_000_000}
        for i in range(20)
    ]  # fmt: skip
    (run / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    asked = dataclasses.replace(context, flags=("--ttft-from", "first-token"), new_run=run)

    offered = _l4_offers(situation, asked, tmp_path)

    assert offered.ttft_basis is not None
    assert "first streamed chunk (118 ms p95)" in offered.ttft_basis
    nothing = dataclasses.replace(asked, new_run=tmp_path)
    assert _l4_offers(situation, nothing, tmp_path).ttft_basis is None


def test_a_queueing_tool_run_keeps_the_client_s_ttft_and_says_why(tmp_path: Path) -> None:
    situation, context = _held_six_steps()
    queued = dataclasses.replace(situation.measurement, mean_waiting=11.0)
    queueing = dataclasses.replace(situation, measurement=queued)

    offered = _l4_offers(queueing, context, tmp_path)
    plain = _l4_offers(dataclasses.replace(
        queueing, measurement=dataclasses.replace(queued, server_ttft_p50_ms=None)
    ), context, tmp_path)  # fmt: skip

    assert offered.ttft_basis is None and offered.ttft_skipped == "TTFT is queue time here"
    assert [s.entry.name for s in offered.suggestions] == [s.entry.name for s in plain.suggestions]
    section = next_steps_section(queueing.diagnosis, offered, retained=False)
    assert not any(getattr(b, "key", None) == "ttft_basis" for b in section.blocks)


@pytest.mark.parametrize(
    ("start_class", "expected"),
    [
        (None, 310.0),
        ("new_set_bf16", 310.0),
        ("cache_hit", 210.0),
        ("new_set_fp8", 410.0),
        ("cold", 160.0),
    ],
)
def test_a_step_uses_its_own_start_class_and_keeps_the_sparse_class_fallback(
    tmp_path: Path,
    start_class: str | None,
    expected: float,
) -> None:
    for i, (kind, ready) in enumerate(
        [
            ("new_set_bf16", 200),
            ("new_set_bf16", 300),
            ("cache_hit", 140),
            ("cache_hit", 160),
            ("new_set_fp8", 300),
            ("new_set_fp8", 400),
            ("cold", 900),
        ]
    ):
        run = tmp_path / "runs" / f"20261017T01000{i}Z-0000"
        run.mkdir(parents=True)
        (run / "window.json").write_text(
            json.dumps({"startup": {"start_class": kind, "ready_s": ready}})
        )
    context = AnalysisContext(launched_s=0.0, load_s=100.0, measured_s=160.0, project=tmp_path)

    assert seconds_to_test(context, start_class=start_class) == pytest.approx(expected)


def test_the_rerun_text_only_step_compiles_and_its_gmu_child_stays_adjacent(tmp_path: Path) -> None:
    from tensward.engines.vllm.start_phases import phases
    from tensward.suggest import _by_start_cost, _start_cost

    situation = recorded_situation(RERUN_DEFAULT, GEMMA_FACTS)
    entries = {entry.name: entry for entry in VLLM.playbook()}
    offers = []
    for name, run_id in (
        ("no-media-encoders", "20261017T012344Z-262e"),
        ("more-kv-memory", "20261017T011321Z-6cd2"),
    ):
        record = json.loads((L4_GEMMA_2 / run_id / "run.json").read_text())
        settings = Settings.from_json(record["settings"])
        offers.append(
            Suggestion(
                entry=entries[name],
                reason="same predicted gain",
                tier="try_first",
                addresses=entries[name].addresses,
                settings=settings,
                command="tensward analyse",
                predicted=Prediction(tpot=-0.1, req_s=0.0, text=""),
            )
        )
    stream = entries["stream-interval"]
    offers.append(
        Suggestion(
            entry=stream,
            reason="same predicted gain",
            tier="try_first",
            addresses=stream.addresses,
            settings=stream.apply(situation),
            command="tensward analyse",
            predicted=Prediction(tpot=-0.1, req_s=0.0, text=""),
        )
    )
    for run_id, kind in STARTS.items():
        seen = phases((L4_GEMMA_2 / run_id / "server.log").read_text())
        assert seen is not None
        run = tmp_path / "runs" / run_id
        run.mkdir(parents=True)
        (run / "window.json").write_text(
            json.dumps({"startup": {"start_class": kind, "ready_s": seen["ready"]}})
        )
    context = AnalysisContext(
        goal="tpot",
        project=tmp_path,
        launched_s=0.0,
        load_s=144.0,
        measured_s=444.0,
    )
    shown = Suggested(suggestions=tuple(offers), not_applicable=(), blocked={}, goal="tpot")
    with active(context):
        # The stream cache hit moves ahead; gmu still follows the parent that includes it.
        ordered = _by_start_cost(offers, shown, situation, VLLM)
        assert [s.entry.name for s in ordered] == [
            "stream-interval",
            "no-media-encoders",
            "more-kv-memory",
        ]
        assert [_start_cost(situation, VLLM, offer) for offer in ordered] == [0, 1, 0]
    assert seconds_to_test(context, start_class="cache_hit") == pytest.approx(444.0)
    assert seconds_to_test(context, start_class="new_set_bf16") == pytest.approx(512.5)


def test_near_equal_new_compiles_use_the_step_s_cap_stretched_test_seconds() -> None:
    from tensward.suggest import _by_start_cost

    situation, context = _gemma_cell("tpot")
    situation = dataclasses.replace(
        situation,
        settings=dataclasses.replace(situation.settings, max_concurrent_requests=96),
        measurement=dataclasses.replace(situation.measurement, peak_running=96),
    )
    context = dataclasses.replace(context, launched_s=0.0, load_s=200.0, measured_s=500.0)
    entries = {entry.name: entry for entry in VLLM.playbook()}
    offers = [
        Suggestion(
            entry=entries[name],
            reason="same predicted gain",
            tier="try_first",
            addresses=entries[name].addresses,
            settings=entries[name].apply(situation),
            command="tensward analyse",
            predicted=Prediction(tpot=-0.1, req_s=0.0, text=""),
        )
        for name in ("lower-concurrency", "kv-sharing-fast-prefill")
    ]
    shown = Suggested(suggestions=tuple(offers), not_applicable=(), blocked={}, goal="tpot")

    with active(context):
        assert [s.entry.name for s in _by_start_cost(offers, shown, situation, VLLM)] == [
            "kv-sharing-fast-prefill",
            "lower-concurrency",
        ]


def _try_first_order(tmp_path: Path, run: str, goal: Goal) -> list[tuple[list[str], str]]:
    """The names behind each runnable Try-first step of a gemma re-run, with its start class."""
    with active(AnalysisContext(goal=goal)):
        situation = recorded_situation(L4_GEMMA_2 / run, GEMMA_CLOSED_LOOP)
        offered = suggestions(
            VLLM, situation, current=situation.settings, project_path=tmp_path, version="0.30.0"
        )
    section = next_steps_section(situation.diagnosis, offered, retained=False)
    return [
        ([m.name for m in s.members] or [s.entry.name], CLASSES[_start_cost(situation, VLLM, s)])
        for block in section.blocks
        if isinstance(block, SuggestionBlock)
        and (s := block.suggestion).tier == "try_first"
        and s.command
    ]


def test_the_step_that_may_change_answers_leads_the_gemma_ttft_run(tmp_path: Path) -> None:
    order = _try_first_order(tmp_path, "20261017T005701Z-42ba", "ttft")
    assert order == [
        (["step-budget-down", "fp8-kv-cache"], "new_set_bf16"),
        (["step-budget-down"], "new_set_bf16"),
    ]


def test_the_gemma_cache_hit_ttft_report_names_its_first_step_and_start_class(
    tmp_path: Path,
) -> None:
    assert _try_first_order(tmp_path, "20261017T011321Z-6cd2", "ttft") == [
        (["step-budget-down"], "new_set_bf16")
    ]


@pytest.mark.parametrize("run", ["20261017T005701Z-42ba", "20261017T011321Z-6cd2"])
def test_without_a_judged_step_the_gemma_tpot_order_is_unchanged(run: str, tmp_path: Path) -> None:
    order = _try_first_order(tmp_path, run, "tpot")
    assert order == [
        (["lower-concurrency"], "new_set_bf16"),
        (["step-budget-down", "stream-interval"], "new_set_bf16"),
    ]


def test_short_starts_come_after_long_ones_and_judged_steps_before_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def step(name: str, risk: bool) -> Suggestion:
        entry = dataclasses.replace(OFFERED[0].entry, name=name, quality_risk=risk)
        return Suggestion(entry=entry, tier="try_first", reason="")

    costs = {"short": 0, "long": 1, "judged": 0}
    monkeypatch.setattr("tensward.suggest._start_cost", lambda _s, _e, s: costs[s.entry.name])
    steps = [step("short", False), step("long", False), step("judged", True)]
    assert [s.entry.name for s in _start_order(None, VLLM, steps)] == ["judged", "long", "short"]  # type: ignore[arg-type]


def _rerun_cell(
    goal: Goal, run: Path = RERUN_DEFAULT, *, track: bool = True
) -> tuple[Situation, AnalysisContext]:
    """A gemma re-run under 96 closed-loop clients, with the track's limits when ``track``."""
    _, context = _gemma_cell(goal)
    situation = dataclasses.replace(
        recorded_situation(run, GEMMA_CLOSED_LOOP),
        measurement=replay_measurement(run, gemma_e4b_anatomy()),
    )
    return situation, dataclasses.replace(context, **GEMMA_TPOT_TRACK) if track else context


def _try_first(offered: Suggested) -> list[str]:
    return [
        "+".join(_members(s)) or s.entry.name
        for s in offered.suggestions
        if s.tier == "try_first" and s.command
    ]


def _bundles(offered: Suggested) -> list[Suggestion]:
    return [s for s in offered.suggestions if s.members and not _changes_answers(s)]


@pytest.mark.parametrize(
    ("cell", "goal", "order"),
    [
        ("42ba", "tpot", ["fp8-weights+fp8-kv-decode+step-budget-down+stream-interval"]),
        ("42ba", "ttft", ["fp8-weights+step-budget-down+kv-sharing-fast-prefill+fp8-kv-cache"]),
        ("42ba", "throughput", ["fp8-weights+fp8-kv-cache"]),
        ("f412", "tpot", ["fp8-kv-decode+step-budget-down+stream-interval"]),
        ("f412", "ttft", ["kv-sharing-fast-prefill+step-budget-down"]),
        ("f412", "throughput", ["stream-interval"]),
        ("granite", "tpot", ["fp8-weights+fp8-kv-decode+lower-concurrency"]),
        ("granite", "ttft", ["fp8-weights+fp8-kv-cache", "more-kv-memory"]),
        ("granite", "throughput", ["fp8-weights+fp8-kv-cache", "more-kv-memory"]),
    ],
)  # fmt: skip
def test_the_lossless_small_levers_left_out_of_a_stack_are_one_bundle(
    cell: str, goal: Goal, order: list[str], tmp_path: Path
) -> None:
    runs = {"42ba": RERUN_DEFAULT, "f412": RERUN_FP8}
    situation, context = (
        granite_situation(goal) if cell == "granite" else _rerun_cell(goal, runs[cell])
    )
    offered = _l4_offers(situation, context, tmp_path)
    assert _try_first(offered) == order
    bundles = _bundles(offered)
    assert len(bundles) == (cell == "f412" and goal == "ttft")
    for bundle in bundles:
        assert bundle.command is not None and bundle.command.endswith(f"--window {BUNDLE_WINDOW}")
        assert "Bundled: none of these changes can change the answers" in bundle.reason


def test_a_bundle_rejected_as_worse_gives_its_members_alone(tmp_path: Path) -> None:
    situation, context = _rerun_cell("tpot", track=False)
    before = _l4_offers(situation, context, tmp_path / "before")
    assert _try_first(before) == [
        "fp8-weights",
        "fp8-kv-decode",
        "lower-concurrency",
        "step-budget-down+stream-interval",
    ]
    (bundle,) = _bundles(before)
    change = tuple(sorted(command_args(bundle.command or "")))
    worse = "goal: TPOT p95 41 worse than the best adopted run's 40"
    rejected = Rejection(run="r1", engine_args=change, change=change, made_from=RERUN_DEFAULT.name,
                         reason=worse, at="")  # fmt: skip
    after = _l4_offers(situation, dataclasses.replace(context, rejected=(rejected,)), tmp_path)
    assert _try_first(after) == [
        "fp8-weights",
        "fp8-kv-decode",
        "lower-concurrency",
        "step-budget-down",
        "stream-interval",
    ]


@pytest.mark.parametrize("reason", [None, "tensward: does not recommend keeping this run"])
def test_a_bundle_rejected_without_a_worse_verdict_does_not_give_back_singles(
    reason: str | None, tmp_path: Path
) -> None:
    situation, context = _rerun_cell("tpot", track=False)
    (bundle,) = _bundles(_l4_offers(situation, context, tmp_path / "before"))
    change = tuple(sorted(command_args(bundle.command or "")))
    rejected = Rejection(run="r1", engine_args=change, change=change,
                         made_from=RERUN_DEFAULT.name, reason=reason, at="")  # fmt: skip
    after = _l4_offers(situation, dataclasses.replace(context, rejected=(rejected,)), tmp_path)
    assert _try_first(after) == ["fp8-weights", "fp8-kv-decode", "lower-concurrency"]


@pytest.mark.parametrize("scale", [1.01, 1.2])
def test_a_measured_bundle_gives_back_singles_only_when_it_is_worse(
    scale: float, tmp_path: Path
) -> None:
    situation, context = _rerun_cell("tpot", track=False)
    (bundle,) = _bundles(_l4_offers(situation, context, tmp_path / "before"))
    project = tmp_path / REPLAYED
    base = project / "runs" / RERUN_DEFAULT.name
    shutil.copytree(RERUN_DEFAULT, base)
    step = project / "runs" / "20261017T020000Z-bundle"
    shutil.copytree(base, step)
    record = json.loads((step / "run.json").read_text())
    record["engine_args"] = sorted(command_args(bundle.command or ""))
    (step / "run.json").write_text(json.dumps(record))
    rows = _scaled((base / "requests.jsonl").read_text().splitlines(), scale)
    (step / "requests.jsonl").write_text("\n".join(rows) + "\n")
    compared = window.compare_matched(step, base, clients=96)
    assert not isinstance(compared, str)
    assert compared["tpot_p95_ms"].verdict == ("worse" if scale == 1.2 else "within_noise")
    after = _l4_offers(situation, dataclasses.replace(context, project=project), project)
    singles = ["step-budget-down", "stream-interval"] if scale == 1.2 else []
    assert _try_first(after) == ["fp8-weights", "fp8-kv-decode", "lower-concurrency", *singles]


def test_the_first_text_only_gemma_stack_keeps_the_parents_used_kv_pool(tmp_path: Path) -> None:
    situation, context = _rerun_cell("tpot")
    for media in (False, True):
        at = dataclasses.replace(
            situation,
            facts=dataclasses.replace(situation.facts, media_encoders=True, sends_images=media),
        )
        offered = _l4_offers(at, context, tmp_path / str(media))
        stack = first_try(at.diagnosis, offered)
        assert stack is not None and stack.members and stack.settings is not None
        args = command_args(stack.command or "")
        assert [m.name for m in stack.members if m.name != "no-media-encoders"] == [
            "fp8-weights",
            "fp8-kv-decode",
            "step-budget-down",
            "stream-interval",
        ]
        if media:
            assert "language-model-only" not in args
            assert kvsize.KV_FLAG not in stack.settings.extra_args
        else:
            assert "language-model-only" in args
            pool = int(stack.settings.extra_args[kvsize.KV_FLAG])
            assert f"kv-cache-memory-bytes={pool}" in args
            assert pool / 2**30 >= 4.0 * 0.9200537043817894
            configured = dataclasses.replace(
                at, settings=kvsize.with_kv_bytes(at.settings, 4 * 2**30)
            )
            configured_context = dataclasses.replace(
                context, engine_args=("kv-cache-memory-bytes=4294967296",)
            )
            explicit = first_try(
                configured.diagnosis,
                _l4_offers(configured, configured_context, tmp_path / "explicit"),
            )
            assert explicit is not None and explicit.settings is not None
            assert explicit.settings.extra_args[kvsize.KV_FLAG] == str(pool)
            assert f"kv-cache-memory-bytes={pool}" in command_args(explicit.command or "")
