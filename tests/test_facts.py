"""The facts of a recorded run, as the matcher and the training rows read them."""

from __future__ import annotations

import dataclasses
import gzip
import json
from pathlib import Path

import pytest
from test_replays import (
    CB0F,
    GEMMA_FACTS,
    GEMMA_RUN,
    L4_GEMMA_2,
    T4_FACTS,
    T4_SETTINGS,
    recorded_situation,
    replay_measurement,
    t4_gpu,
    t4_measurement,
)

from tensward.anatomy import build_anatomy
from tensward.classify import classify
from tensward.context import AnalysisContext, active
from tensward.engines.vllm import VLLM, histograms
from tensward.facts import FACTS_VERSION, UNKNOWN, VOCABULARY, Facts, extract
from tensward.measurement import Measurement
from tensward.playbook import Situation
from tensward.trace_facts import prefix_reuse

TPOT_RUN = L4_GEMMA_2 / "20261017T005701Z-42ba"
MODEL = GEMMA_RUN.parent / "model"


def test_the_t4_replay_s_facts_are_filled_where_measured_and_digest_stably() -> None:
    measurement = t4_measurement()
    context = AnalysisContext(goal="ttft", gpu_capability=7.5)
    with active(context):
        diagnosis = classify(
            measurement, T4_SETTINGS, None, engine=VLLM, version="0.30.0", gpu_state=t4_gpu()
        )
    situation = Situation(
        measurement=measurement, facts=T4_FACTS, settings=T4_SETTINGS, diagnosis=diagnosis
    )
    facts = extract(situation, context, VLLM, "0.30.0", anatomy=None)
    assert (facts["gpu.cc"], facts["gpu.arch"], facts["goal"]) == (7.5, "turing", "ttft")
    assert facts["diag.primary"] == diagnosis.primary == "gpu_saturated"
    assert facts["gpu.power_cap_w"] == 70.0
    assert facts["run.kv_usage_peak"] == measurement.peak_kv_usage
    assert facts["engine.version"] == "0.30.0"
    assert facts["settings.max_context_len"] == 8192
    assert facts["anatomy.kv_shared_layers"] is UNKNOWN
    again = extract(situation, context, VLLM, "0.30.0", anatomy=None)
    assert again.digest() == facts.digest()
    assert Facts.from_json(facts.to_json()).digest() == facts.digest()


def _gemma_anatomy():
    rows = json.loads(gzip.decompress((MODEL / "tensors.json.gz").read_bytes()))
    tensors = {name: (dtype, tuple(shape), size) for name, (dtype, shape, size) in rows.items()}
    config = json.loads((MODEL / "config.json").read_text())
    return build_anatomy(config, None, tensors, dtype="bfloat16", quantization="none")


def _facts_of(run_dir: Path, *, anatomy=None) -> Facts:
    anatomy = anatomy or _gemma_anatomy()
    situation = recorded_situation(run_dir, GEMMA_FACTS)
    measurement = replay_measurement(run_dir, anatomy)
    situation = dataclasses.replace(situation, measurement=measurement)
    blocks = [json.loads(line) for line in (run_dir / "prompt_blocks.jsonl").open()]
    context = AnalysisContext(
        goal="tpot",
        new_run=run_dir,
        anatomy=anatomy,
        prompt_blocks={i: (row["tokens"], tuple(row["blocks"])) for i, row in enumerate(blocks)},
    )
    with active(context):
        return extract(situation, context, VLLM, "0.30.0", anatomy=anatomy)


def test_the_gemma_chat_run_reads_no_kv_pressure_though_its_cache_is_full() -> None:
    facts = _facts_of(GEMMA_RUN)
    assert facts["anatomy.kv_shared_layers"] == 18
    assert facts["run.running_share"] == pytest.approx(0.99, abs=0.005)
    assert facts["run.mean_waiting"] == pytest.approx(0.03, abs=0.005)
    assert 4.06 <= facts["run.step_held_ratio"] <= 4.09
    assert facts["run.kv_pressure"] is False
    assert facts["run.saturated"] is True


def test_the_gemma_tpot_run_has_the_config_the_pressure_and_the_trace_facts() -> None:
    facts = _facts_of(TPOT_RUN)
    assert facts["model.arch"] == "Gemma4ForConditionalGeneration"
    assert (facts["model.quant"], facts["model.quant_bits"], facts["model.dtype"]) == (
        "none",
        16,
        "bfloat16",
    )
    assert facts["model.has_mtp"] is False and facts["model.gptq_desc_act"] is False
    assert facts["anatomy.linear_attention"] is False and facts["anatomy.mla"] is False
    assert facts["anatomy.per_layer_embeddings_gib"] > 1
    assert 0 < facts["anatomy.quantizable_gib"] < facts["model.weight_gib"]
    assert 0 <= facts["anatomy.unclassified_share"] < 0.01
    assert facts["workload.output_tokens_p50"] == 384
    assert facts["run.running_p50"] == facts["run.running_p90"] == 96
    assert 100 < facts["run.server_tpot_p50_ms"] < 120
    assert 0.1 < facts["run.mixed_step_share"] < 0.4
    assert 0 < facts["workload.possible_prefix_hit"] < 0.01
    assert facts["workload.common_prefix_tokens"] == 0
    assert facts["workload.shared_prefix_share"] >= facts["workload.possible_prefix_hit"]
    assert facts["run.start_class"] == "cold"
    assert facts["run.kv_blocks"] == 16387


def test_facts_without_their_source_stay_unknown() -> None:
    facts = _facts_of(GEMMA_RUN)
    for name in (
        "run.running_p50",
        "run.running_p90",
        "run.server_tpot_p50_ms",
        "run.mixed_step_share",
        "run.waiting_capacity",
        "run.waiting_deferred",
        "run.wave_period_s",
        "host.memory_gib",
    ):
        assert facts[name] is UNKNOWN, name
    assert all(name in VOCABULARY for name in facts.to_json())


def test_a_saturated_run_that_fits_its_cache_and_a_preempting_one_read_apart() -> None:
    qwen = recorded_situation(CB0F, GEMMA_FACTS)
    qwen = dataclasses.replace(qwen, measurement=replay_measurement(CB0F, _gemma_anatomy()))
    context = AnalysisContext(goal="throughput", new_run=CB0F)
    with active(context):
        facts = extract(qwen, context, VLLM, "0.30.0", anatomy=None)
    assert facts["run.saturated"] is True and facts["run.kv_pressure"] is False

    preempting = dataclasses.replace(
        qwen, measurement=Measurement(succeeded=200, failed=0, preemptions=81.0, peak_kv_usage=1.0)
    )
    with active(AnalysisContext(goal="throughput")):
        facts = extract(preempting, AnalysisContext(), VLLM, "0.30.0", anatomy=None)
    assert facts["run.kv_pressure"] is True


def test_two_requests_share_two_of_their_blocks() -> None:
    blocks = {0: (60, ("a", "b", "c")), 1: (50, ("a", "b", "x"))}
    rows = [{"prompt_index": 0}, {"prompt_index": 1}]
    possible, shared, common = prefix_reuse(rows, blocks, 16)
    assert possible == pytest.approx(32 / 110)
    assert shared == pytest.approx((32 + 32) / 110)
    assert common == 32


def test_the_histograms_grow_between_the_real_scrapes() -> None:
    before, after = (TPOT_RUN / f"metrics_{when}.prom" for when in ("before", "after"))
    steps = histograms.growth(before.read_text(), after.read_text(), "vllm:iteration_tokens_total")
    assert steps is not None and steps.counts[-1] > 3000
    queue = histograms.growth(
        before.read_text(), after.read_text(), "vllm:request_queue_time_seconds"
    )
    assert queue is not None and queue.counts[-1] > 900
    assert 0 < queue.quantile(0.5) < 0.3
    assert histograms.growth(before.read_text(), after.read_text(), "vllm:no_such") is None


def test_the_facts_version_is_three() -> None:
    assert FACTS_VERSION == 3


def test_the_host_memory_is_the_one_the_run_recorded(tmp_path: Path) -> None:
    record = json.loads((TPOT_RUN / "run.json").read_text())
    (tmp_path / "run.json").write_text(json.dumps(record | {"host": {"ram_bytes": 32 * 2**30}}))
    situation = recorded_situation(TPOT_RUN, GEMMA_FACTS)
    context = AnalysisContext(goal="tpot", new_run=tmp_path)
    facts = extract(situation, context, VLLM, "0.30.0", anatomy=None)
    assert facts["host.memory_gib"] == 32.0
