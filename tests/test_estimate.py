"""Expected effects: the two estimates on hand-checked cases, the runs they refuse, and the one
promotion they make."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Callable, Sequence

import pytest

from tensward.capture import CapturedResponse
from tensward.ceilings import Ceilings
from tensward.classify import classify
from tensward.client import RequestRecord
from tensward.engines.vllm import VLLM
from tensward.estimate import (
    Estimator,
    RunInputs,
    block_hashes,
    prefix_caching_estimate,
    raise_concurrency_estimate,
    range_text,
    step_ratio,
)
from tensward.inputs import PromptEntry
from tensward.measurement import Measurement, Window
from tensward.playbook import Situation, WorkloadFacts
from tensward.report.build import next_steps_section
from tensward.report.markdown import sections_markdown
from tensward.rundir import prompt_blocks
from tensward.settings import Settings
from tensward.suggest import _estimates, suggestions
from tensward.workload import ChatMessage

# An agent workload on an A10G whose KV cache is full: 64 clients, 384 requests over 128 prompts.
A10G = Ceilings(
    gpu="NVIDIA A10G", bandwidth_gbs=600.096, weight_bytes_per_step=14_141_238_272,
    kv_bytes_per_sequence=112_623_616,
)  # fmt: skip
FULL = Measurement(
    384, 0, request_throughput=1.9405, output_throughput=114.62, total_throughput=3849.9,
    tpot_p50_ms=357.64, tpot_p95_ms=397.46, ttft_p50_ms=12111.6, ttft_p95_ms=15588.2,
    peak_kv_usage=0.992, peak_running=44.0, peak_waiting=23.0, preemptions=0.0,
    kv_capacity_tokens=87952.0, kv_block_tokens=16.0, kv_blocks=5497.0, max_prompt_tokens=1984,
    max_context_len=8192, seconds=148.6, mean_running=42.932, queue_share=0.897,
    peak_in_flight=64, ceilings=A10G, window=Window("steady", 0, 148_600_000_000, 148.6),
)  # fmt: skip
OFF = Settings(
    dtype="bfloat16", max_concurrent_requests=64, max_context_len=8192, kv_memory_fraction=0.92,
    prefill_batch_tokens=2048, prefix_caching=False, tool_calling=True, tool_parser="hermes",
)  # fmt: skip
# Token ids in the order the chat template renders them: the system prompt, then the tools,
# then the user turn.
SYSTEM_AND_TOOLS = tuple(range(1_000, 2_900))
WORDS = tuple(" ".join(["rule"] * 1100) + f" ask {i} " + "word " * 10 for i in range(128))
FACTS = WorkloadFacts(prompts=WORDS, output_tokens=512, context_limit=32768, offers_tools=True)


def agent(i: int) -> tuple[int, ...]:
    return (*SYSTEM_AND_TOOLS, *range(100_000 + 100 * i, 100_000 + 100 * i + 25))


def dated(i: int) -> tuple[int, ...]:
    """The agent prompt with a per-request date line at the start of the system prompt."""
    return (1, 2, 3, 900_000 + i, *agent(i))


def documents() -> list[tuple[int, ...]]:
    """200 documents of 2,660 tokens, five questions on each."""
    return [
        (*range(300_000 + 5_000 * d, 300_000 + 5_000 * d + 2_660),
         *range(10_000_000 + 100 * (5 * d + q), 10_000_000 + 100 * (5 * d + q) + 27))
        for d in range(200)
        for q in range(5)
    ]  # fmt: skip


def twice() -> RunInputs:
    """192 distinct prompts of 1,000 tokens, each sent twice in a row: only a finished first
    copy can serve the second."""
    prompts = [tuple(range(1_000_000 + 2_000 * i, 1_001_000 + 2_000 * i)) for i in range(192)]
    return run_of(prompts, [i // 2 for i in range(384)])


def run_of(prompts: Sequence[tuple[int, ...]], order: Sequence[int]) -> RunInputs:
    blocks = {i: (len(ids), block_hashes(ids, 16)) for i, ids in enumerate(prompts)}
    return RunInputs(
        arrival="closed_loop", gpu_count=1, prompt_order=tuple(order), prompt_blocks=blocks
    )


def full(
    measurement: Measurement = FULL, facts: WorkloadFacts = FACTS, settings: Settings = OFF
) -> Situation:
    diagnosis = classify(measurement, settings, None)
    return Situation(measurement=measurement, facts=facts, settings=settings, diagnosis=diagnosis)


AGENT_RUN = run_of([agent(i) for i in range(128)], [i % 128 for i in range(384)])
SHUFFLED = list(range(1000))
random.Random(1).shuffle(SHUFFLED)
CACHING = replace(OFF, prefix_caching=True)
CAP256 = replace(OFF, max_concurrent_requests=256)

# A capped T4 run: 2 running, 30 waiting, the cap raised to 16.
CAPPED = Measurement(
    128, 0, request_throughput=0.49203, output_throughput=33.717, mean_running=1.97884,
    peak_running=2.0, peak_waiting=30.0, peak_in_flight=32, seconds=193.1,
    ceilings=Ceilings(gpu="T4", bandwidth_gbs=320, weight_bytes_per_step=7_672_043_520),
)  # fmt: skip
CAP2 = Settings(max_concurrent_requests=2)
CAP16 = replace(CAP2, max_concurrent_requests=16)
CLOSED = RunInputs(arrival="closed_loop", gpu_count=1)


def capped(measurement: Measurement = CAPPED) -> Situation:
    facts = WorkloadFacts(prompts=("p",), output_tokens=128, context_limit=4096)
    diagnosis = classify(measurement, CAP2, None)
    return Situation(measurement=measurement, facts=facts, settings=CAP2, diagnosis=diagnosis)


@pytest.mark.parametrize(
    ("estimator", "situation", "proposed", "run", "low", "high"),
    [
        (prefix_caching_estimate, full(), CACHING, AGENT_RUN, 2.858, 10.013),
        (prefix_caching_estimate, full(), CACHING,
         run_of([dated(i) for i in range(128)], [i % 128 for i in range(384)]), 1.0, 1.0),
        (prefix_caching_estimate, full(), CACHING, run_of(documents(), SHUFFLED), 1.0, 1.137),
        (prefix_caching_estimate, full(), CACHING, run_of(documents(), range(1000)), 1.028, 3.636),
        (prefix_caching_estimate, full(), CACHING,
         run_of([agent(i) for i in range(8)], [i % 8 for i in range(384)]), 2.936, 10.495),
        (prefix_caching_estimate, full(settings=CAP256), replace(CAP256, prefix_caching=True),
         run_of(documents()[::5][:120], range(120)), 1.0, 1.0),
        (prefix_caching_estimate, full(), CACHING, twice(), 1.028, 1.869),
        (raise_concurrency_estimate, capped(), CAP16, CLOSED, 1.556, 8.0),
    ],
    ids=[
        "shared-system-and-tools",
        "dated-system-prompt",
        "documents-shuffled",
        "documents-grouped",
        "fewer-prompts-than-in-flight",
        "cap-above-the-requests-sent",
        "prefill-grows-faster-than-decode",
        "raise-the-cap",
    ],
)  # fmt: skip
def test_each_estimate_on_hand_checked_runs(
    estimator: Estimator,
    situation: Situation,
    proposed: Settings,
    run: RunInputs,
    low: float,
    high: float,
) -> None:
    estimate = estimator(situation, proposed=proposed, run=run, engine=VLLM)
    assert estimate is not None
    assert (round(estimate.low, 3), round(estimate.high, 3)) == (low, high)


Case = tuple[Measurement, Settings, RunInputs | None]
TP2 = {"--tensor-parallel-size": "2"}


def short(m: Measurement) -> Measurement:
    return replace(m, seconds=29.0, window=Window("steady", 0, 29_000_000_000, 29.0))


BREAKS: dict[str, Callable[[Case], Case]] = {
    "tensor-parallel": lambda c: (c[0], replace(c[1], extra_args=TP2), c[2]),
    "data-parallel-replicas": lambda c: (replace(c[0], replicas=2), *c[1:]),
    "speculative-decoding": lambda c: (replace(c[0], spec_coverage=0.5), *c[1:]),
    "hybrid-cache": lambda c: (replace(c[0], hybrid_cache=True), *c[1:]),
    "two-gpus": lambda c: (c[0], c[1], replace(c[2], gpu_count=2) if c[2] else None),
    "forced-tool": lambda c: (c[0], c[1], replace(c[2], forced_tools=True) if c[2] else None),
    "open-loop": lambda c: (c[0], c[1], replace(c[2], arrival="open_loop") if c[2] else None),
    "no-run": lambda c: (c[0], c[1], None),
    "over-2%-failed": lambda c: (replace(c[0], failed=round(c[0].succeeded * 0.021)), *c[1:]),
    "short-window": lambda c: (short(c[0]), *c[1:]),
    "unchecked-gpu": lambda c: (replace(c[0], ceilings=replace(A10G, gpu="NVIDIA H100")), *c[1:]),
    "empty": lambda c: (Measurement(0, 0), *c[1:]),
}


@pytest.mark.parametrize("name", list(BREAKS))
def test_no_estimate_outside_the_runs_it_was_checked_on(name: str) -> None:
    raised = BREAKS[name]((CAPPED, CAP16, CLOSED))
    situation = capped(raised[0])
    assert (
        raise_concurrency_estimate(situation, proposed=raised[1], run=raised[2], engine=VLLM)
        is None
    )
    cached = BREAKS[name]((FULL, CACHING, AGENT_RUN))
    situation = full(cached[0])
    assert (
        prefix_caching_estimate(situation, proposed=cached[1], run=cached[2], engine=VLLM) is None
    )


@pytest.mark.parametrize(
    ("measurement", "facts", "run"),
    [
        (FULL, replace(FACTS, sends_images=True), AGENT_RUN),
        (FULL, FACTS, replace(AGENT_RUN, prompt_blocks={0: AGENT_RUN.prompt_blocks[0]})),
        (FULL, replace(FACTS, structured=True), AGENT_RUN),
        (replace(FULL, ceilings=replace(A10G, moe_experts=(8, 2))), FACTS, AGENT_RUN),
    ],
    ids=["images", "a-prompt-without-blocks", "structured-output", "routed-experts"],
)
def test_no_cache_estimate_without_the_prompts_the_engine_computed(
    measurement: Measurement, facts: WorkloadFacts, run: RunInputs
) -> None:
    situation = full(measurement, facts=facts)
    assert prefix_caching_estimate(situation, proposed=CACHING, run=run, engine=VLLM) is None


def test_the_decode_growth_is_never_clipped() -> None:
    # Floor 100 ms at the old batch (40 weights, 60 KV), 130 ms at 1.5 times the batch; true
    # efficiency 0.46 gives about x1.28, so the low end must not claim more.
    low = step_ratio(tpot=241.0, floor=100.0, growth=1.3, batch=1.5, uncached=0.02, eta=0.20)
    assert low <= 1.28
    assert round(low, 3) == 1.154


@pytest.mark.parametrize(
    ("low", "high", "text"),
    [
        (1.0, 1.0, "unchanged"),
        (0.96, 1.3, "-4% to +30%"),
        (1.0, 3.35, "unchanged to x3.4"),
        (1.4, 2.1, "+40% to x2.1"),
        (2.86, 10.0, "at least x2.9"),
    ],
)
def test_a_range_is_said_in_words_and_a_possible_loss_as_a_loss(
    low: float, high: float, text: str
) -> None:
    assert range_text(low, high) == text


def test_blocks_are_kept_only_when_the_engine_counted_the_same_tokens() -> None:
    prompts = [PromptEntry(id=f"p{i}", messages=(ChatMessage(role="user", content="hi"),))
               for i in range(2)]  # fmt: skip
    ids = {"p0": tuple(range(40)), "p1": tuple(range(41))}
    records = [RequestRecord(f"r{i}", 0, i, None, 10, "success", 4, i) for i in range(2)]
    answered = {"r0": CapturedResponse(prompt_tokens=40), "r1": CapturedResponse(prompt_tokens=41)}
    kept = prompt_blocks(prompts, ids, records, answered, 16.0)
    assert {index: tokens for index, (tokens, _) in kept.items()} == {0: 40, 1: 41}
    assert kept[0][1] == kept[1][1]  # the same first 32 tokens: the same two block hashes
    answered["r1"] = CapturedResponse(prompt_tokens=45)
    assert prompt_blocks(prompts, ids, records, answered, 16.0) == {}


def test_a_large_expected_gain_leads_try_first_but_never_above_a_critical_gate() -> None:
    def try_first(situation: Situation, run: RunInputs | None) -> str:
        suggested = suggestions(VLLM, situation, current=OFF, project_path=None, run=run)
        section = next_steps_section(situation.diagnosis, suggested, retained=True)
        return sections_markdown([section]).split("Could help:")[0]

    led = try_first(full(FULL), AGENT_RUN)
    assert led.index("For KV-cache capacity:") < led.index("`prefix-caching`")
    assert led.index("`prefix-caching`") < led.index("`more-kv-memory`")
    assert "expected: throughput at least x2.9" in led
    assert "`prefix-caching`" not in try_first(full(FULL), None)
    failing = full(replace(FULL, succeeded=382, failed=2, too_long=("p1",), max_prompt_tokens=8100))
    gated = try_first(failing, AGENT_RUN)
    assert gated.index("For fit and failed requests:") < gated.index("`prefix-caching`")
    prefill = replace(failing.diagnosis, primary="prefill", secondary=("kv_capacity",))
    ahead = try_first(replace(failing, diagnosis=prefill), AGENT_RUN)
    assert (
        ahead.index("For fit and failed requests:")
        < ahead.index("For KV-cache capacity:")
        < ahead.index("For prefill compute:")
    )
    situation = full()
    offered = suggestions(VLLM, situation, current=OFF, project_path=None).suggestions
    assert _estimates(VLLM, situation, (), offered, AGENT_RUN) == {}  # not the engine's entry
    # 1.5 and 17 prompt tokens per generated token: prompt processing dominates the second run.
    for total, lead in ((112.5, "queueing"), (810.0, None)):
        raised = capped(
            replace(CAPPED, output_throughput=45.0, total_throughput=total, peak_kv_usage=0.2)
        )
        suggested = suggestions(VLLM, raised, current=CAP2, project_path=None, run=CLOSED)
        assert suggested.lead == lead
        assert ("raise-concurrency" in suggested.estimates) == (lead is not None)
    # A date line ahead of the shared text leaves the cache nothing to serve.
    dated_run = run_of([dated(i) for i in range(128)], [i % 128 for i in range(384)])
    for run, lead in ((AGENT_RUN, "kv_capacity"), (dated_run, None)):
        suggested = suggestions(VLLM, full(FULL), current=OFF, project_path=None, run=run)
        assert suggested.lead == lead
        assert ("prefix-caching" in suggested.estimates) == (lead is not None)
