"""Expected effects: the three estimates on hand-checked cases and the runs they refuse."""

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
    raise_prefill_batch_estimate,
    range_text,
    step_ratio,
)
from tensward.inputs import PromptEntry
from tensward.measurement import Measurement, Window
from tensward.playbook import Situation, WorkloadFacts
from tensward.rundir import prompt_blocks
from tensward.settings import Settings
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
    prompt_tokens_per_step=1427.0, scheduled_tokens_per_step=1469.9, step_ms=357.6,
    prompt_capacity_tok_s=4262.0, step_budget_tokens=2048, mean_kv_usage=0.97, mean_waiting=21.0,
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

HALF = replace(A10G, kv_bytes_per_sequence=56_311_808)
STEADY = Window("steady", 0, 120_000_000_000, 120.0)
DECODE = Measurement(
    512, 0, request_throughput=0.26, output_throughput=266.2, peak_running=8.0, mean_running=7.99,
    peak_waiting=56.0, peak_in_flight=64, seconds=120.0, window=STEADY, ceilings=HALF,
    prompt_tokens_per_step=1.65, scheduled_tokens_per_step=9.64, step_ms=30.2,
    prompt_capacity_tok_s=4000.0, step_budget_tokens=2048, mean_kv_usage=0.011, mean_waiting=55.0,
)  # fmt: skip
PROMPTY = replace(
    DECODE, request_throughput=1.5, output_throughput=72.0, mean_running=7.9, peak_waiting=120.0,
    peak_in_flight=128, prompt_tokens_per_step=325.0, scheduled_tokens_per_step=332.8,
    step_ms=108.0, mean_kv_usage=0.04, mean_waiting=118.0, queue_share=0.95, peak_kv_usage=0.04,
    prompt_work_share=0.92,
)  # fmt: skip
BUDGET = replace(
    DECODE, request_throughput=2.6, output_throughput=1300.0, peak_running=70.0, mean_running=69.0,
    peak_waiting=30.0, peak_in_flight=96, prompt_tokens_per_step=60.0,
    scheduled_tokens_per_step=128.0, step_ms=47.0, step_budget_tokens=128, mean_kv_usage=0.47,
    mean_waiting=26.0,
)  # fmt: skip
CAP8 = Settings(max_concurrent_requests=8, prefix_caching=True)
B128 = Settings(max_concurrent_requests=96, prefill_batch_tokens=128, prefix_caching=True)
UNSET = replace(B128, max_concurrent_requests=None)
CLOSED = RunInputs(arrival="closed_loop", gpu_count=1)


def ran(measurement: Measurement, settings: Settings) -> Situation:
    facts = WorkloadFacts(prompts=("p",), output_tokens=128, context_limit=32768)
    diagnosis = classify(measurement, settings, None)
    return Situation(measurement=measurement, facts=facts, settings=settings, diagnosis=diagnosis)


@pytest.mark.parametrize(
    ("estimator", "situation", "proposed", "run", "low", "high", "binding", "named"),
    [
        (prefix_caching_estimate, full(), CACHING, AGENT_RUN, 2.904, 10.162, "demand", {"demand"}),
        (prefix_caching_estimate, full(), CACHING,
         run_of([agent(i) for i in range(8)], [i % 8 for i in range(384)]), 2.984, 10.659,
         "demand", {"demand"}),
        (raise_concurrency_estimate, ran(DECODE, CAP8), replace(CAP8, max_concurrent_requests=64),
         CLOSED, 3.153, 7.310, "demand", {"demand"}),
        (raise_concurrency_estimate, ran(DECODE, CAP8), replace(CAP8, max_concurrent_requests=32),
         CLOSED, 2.412, 3.847, "cap", {"cap"}),
        (raise_concurrency_estimate, ran(replace(DECODE, mean_kv_usage=0.2), CAP8),
         replace(CAP8, max_concurrent_requests=64), CLOSED, 2.480, 4.069, "kv", {"kv"}),
        (raise_concurrency_estimate, ran(PROMPTY, CAP8),
         replace(CAP8, max_concurrent_requests=128), CLOSED, 1.224, 1.259, "step_budget",
         {"step_budget", "prompt"}),
        (raise_concurrency_estimate, ran(replace(PROMPTY, prompt_capacity_tok_s=2900.0), CAP8),
         replace(CAP8, max_concurrent_requests=128), CLOSED, None, None, None, None),
        (raise_prefill_batch_estimate, ran(BUDGET, B128), replace(B128, prefill_batch_tokens=256),
         CLOSED, 1.171, 1.234, "demand", {"demand"}),
        (raise_prefill_batch_estimate, ran(replace(BUDGET, gpu_memory_gib=22.0), UNSET),
         replace(UNSET, prefill_batch_tokens=256), replace(CLOSED, engine_version="0.30.0"),
         1.171, 1.234, "demand", {"demand"}),
        (raise_prefill_batch_estimate, ran(replace(BUDGET, gpu_memory_gib=22.0), UNSET),
         replace(UNSET, prefill_batch_tokens=256), replace(CLOSED, engine_version="0.29.0"),
         None, None, None, None),
    ],
    ids=[
        "shared-system-and-tools",
        "fewer-prompts-than-in-flight",
        "raise-the-cap",
        "the-new-cap-binds",
        "the-kv-room-binds",
        "prompt-heavy-stops-at-the-step-budget",
        "the-prompt-ceiling-leaves-no-gain",
        "raise-the-step-budget",
        "no-cap-set-vllm-0.30-on-24-gb-runs-256",
        "no-cap-set-on-a-version-not-read",
    ],
)  # fmt: skip
def test_each_estimate_on_hand_checked_runs(
    estimator: Estimator, situation: Situation, proposed: Settings, run: RunInputs,
    low: float | None, high: float | None, binding: str | None, named: set[str] | None,
) -> None:  # fmt: skip
    estimate = estimator(situation, proposed=proposed, run=run, engine=VLLM)
    if low is None:
        assert estimate is None
        return
    assert estimate is not None
    assert (round(estimate.low, 3), round(estimate.high, 3), estimate.binding) == (
        low,
        high,
        binding,
    )
    assert estimate.named == named


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
    "budget-unknown": lambda c: (replace(c[0], step_budget_tokens=None), *c[1:]),
    "capacity-unknown": lambda c: (replace(c[0], prompt_capacity_tok_s=None), *c[1:]),
    "kv-use-unknown": lambda c: (replace(c[0], mean_kv_usage=None), *c[1:]),
    "t4-is-not-checked": lambda c: (
        replace(c[0], ceilings=replace(c[0].ceilings, gpu="T4")),
        *c[1:],
    ),
}


@pytest.mark.parametrize("name", list(BREAKS))
def test_no_estimate_outside_the_runs_it_was_checked_on(name: str) -> None:
    raised = BREAKS[name]((DECODE, replace(CAP8, max_concurrent_requests=64), CLOSED))
    situation = ran(raised[0], CAP8)
    assert (
        raise_concurrency_estimate(situation, proposed=raised[1], run=raised[2], engine=VLLM)
        is None
    )
    budget = BREAKS[name]((BUDGET, replace(B128, prefill_batch_tokens=256), CLOSED))
    situation = ran(budget[0], B128)
    assert (
        raise_prefill_batch_estimate(situation, proposed=budget[1], run=budget[2], engine=VLLM)
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
        (FULL, FACTS, run_of([dated(i) for i in range(128)], [i % 128 for i in range(384)])),
        (FULL, FACTS, run_of(documents(), SHUFFLED)),
        (FULL, FACTS, run_of(documents(), range(1000))),
        (FULL, FACTS, twice()),
    ],
    ids=[
        "images",
        "a-prompt-without-blocks",
        "structured-output",
        "routed-experts",
        "dated-system-prompt",
        "documents-shuffled",
        "documents-grouped",
        "each-prompt-twice",
    ],
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
        (1.051, 1.054, "about +5%"),
        (1.159, 1.219, "+15% to +22%"),
        (1.4, 2.1, "+40% to x2.1"),
        (2.904, 10.162, "at least x2.9"),
        (3.153, 7.310, "x3.1 to x7.3"),
    ],
)
def test_a_range_is_said_in_words_with_its_floor_rounded_down(
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
