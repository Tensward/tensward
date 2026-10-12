"""The playbook entries: when each applies, what it changes, and how it is offered."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_estimate import AGENT_RUN, CAP8, CLOSED, FULL, OFF, PROMPTY, dated, full, ran, run_of

from tensward.answers_after import AnswersAfter
from tensward.capabilities import Capabilities, LeverSupport
from tensward.ceilings import Ceilings
from tensward.classify import classify
from tensward.compare import Answer, compare_answers
from tensward.context import AnalysisContext, Goal, active
from tensward.engines import ENGINES
from tensward.engines.vllm import VLLM, VllmEngine
from tensward.engines.vllm.predicates import raise_prefill_batch_near
from tensward.estimate import RunInputs
from tensward.extensions import Extender
from tensward.gpu_state import GpuState
from tensward.history import (
    METRIC,
    Memory,
    PastRun,
    _below_floor,
    _past_knee,
    checked,
    command_args,
    knee,
    plan_for,
    plan_lines,
    remember,
    seconds_to_test,
    stop_reason,
)
from tensward.jsonanswers import StructuredAnswerStats
from tensward.measurement import Measurement
from tensward.playbook import (
    Effect,
    Entry,
    Situation,
    Suggestion,
    WorkloadFacts,
    fitting_max_context_len,
    load_entries,
    rungs,
)
from tensward.playbook_common import prefix_and_rest, projected_kv, with_cap_for_more_memory
from tensward.prompt_lookup import accepted, coverage, words
from tensward.report.build import answers_after_section, next_steps_section
from tensward.report.markdown import sections_markdown
from tensward.report.model import Text
from tensward.settings import Settings
from tensward.suggest import _estimates, applicable, budget_refusal, by_goal, suggestions

PLAYBOOK = VLLM.playbook()
FACTS = WorkloadFacts(prompts=("p",), output_tokens=128, context_limit=4096)


def situation(signals: Measurement, facts: WorkloadFacts, settings: Settings) -> Situation:
    diagnosis = classify(signals, settings, None)
    return Situation(measurement=signals, facts=facts, settings=settings, diagnosis=diagnosis)


def suggest(
    signals: Measurement, facts: WorkloadFacts, settings: Settings, *, allow_quality_changes: bool
) -> list[tuple[Entry, str]]:
    found, _ = applicable(
        PLAYBOOK, situation(signals, facts, settings), allow_quality_changes=allow_quality_changes
    )
    return [(s.entry, s.reason) for s in found]


def test_a_batch_below_one_media_item_is_not_suggested_while_media_inputs_are_on() -> None:
    stalled = Measurement(1, 0, tpot_p50_ms=10.0, tpot_p95_ms=50.0)
    settings = Settings(prefill_batch_tokens=4096)
    names = lambda f, s: [r.name for r, _ in suggest(stalled, f, s, allow_quality_changes=False)]  # noqa: E731
    assert "lower-prefill-batch" in names(
        WorkloadFacts(prompts=("p",), output_tokens=128, context_limit=4096), settings
    )
    media = WorkloadFacts(
        prompts=("p",), output_tokens=128, context_limit=4096, media_encoders=True
    )
    assert "lower-prefill-batch" not in names(media, settings)
    _, gated = applicable(
        PLAYBOOK, situation(stalled, media, settings), allow_quality_changes=False
    )
    assert [(entry.name, "media item" in why) for entry, why in gated] == [
        ("lower-prefill-batch", True)
    ]
    text_only = dataclasses.replace(settings, media_inputs=False)
    assert "lower-prefill-batch" in names(media, text_only)


def test_ngram_speculation_is_judged_on_the_part_of_the_prompt_not_shared() -> None:
    measurement = Measurement(1, 0, tpot_p50_ms=10.0)
    settings = Settings()

    def names(facts: WorkloadFacts) -> list[str]:
        found = suggest(measurement, facts, settings, allow_quality_changes=False)
        return [entry.name for entry, _ in found]

    unique = tuple(" ".join(f"w{i}x{j}" for j in range(300)) for i in range(8))
    agent = tuple(" ".join(["system"] * 1900) + f" turn {i} " + "ask " * 12 for i in range(8))
    assert "ngram-speculation" in names(
        WorkloadFacts(prompts=unique, output_tokens=128, context_limit=4096)
    )
    assert "ngram-speculation" not in names(
        WorkloadFacts(prompts=agent, output_tokens=128, context_limit=4096)
    )


def test_any_fp8_kv_cache_counts_as_already_applied() -> None:
    facts = WorkloadFacts(prompts=("p",), output_tokens=128, context_limit=4096)
    pressured = Measurement(1, 0, peak_kv_usage=0.97)
    settings = Settings(
        dtype="float16",
        max_concurrent_requests=8,
        max_context_len=2048,
        kv_memory_fraction=0.8,
        prefill_batch_tokens=2048,
        kv_cache_dtype="auto",
        prefix_caching=True,
    )
    names = lambda s: [r.name for r, _ in suggest(pressured, facts, s, allow_quality_changes=True)]  # noqa: E731
    assert "fp8-kv-cache" in names(settings)
    for dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
        assert "fp8-kv-cache" not in names(dataclasses.replace(settings, kv_cache_dtype=dtype))


def test_fit_context_rounds_up_and_never_exceeds_the_model_limit() -> None:
    def fit(longest_prompt: int, model_limit: int) -> int | None:
        signals = Measurement(1, 1, max_prompt_tokens=longest_prompt, too_long=("p",))
        return fitting_max_context_len(
            signals, WorkloadFacts(prompts=("p",), output_tokens=128, context_limit=model_limit)
        )

    assert fit(2100, 32768) == 2304  # 2228 tokens rounded up to a multiple of 256
    assert fit(2100, 2240) == 2240  # capped by the model, which still holds 2228
    assert fit(2100, 2200) is None  # 2228 tokens cannot fit the model at all


def test_raise_concurrency_jumps_to_observed_demand_within_kv_headroom() -> None:
    from tensward.playbook_common import _raised_concurrency

    # Gauges are floats, as vLLM exports them.
    seen = Measurement(10, 0, peak_running=8.0, peak_waiting=24.0, peak_kv_usage=0.06)

    def at(kv: float) -> Situation:
        signals = dataclasses.replace(seen, peak_kv_usage=kv)
        return situation(
            signals,
            WorkloadFacts(prompts=("p",), output_tokens=128, context_limit=4096),
            Settings(),
        )

    assert _raised_concurrency(at(0.06), 8) == 32  # running + waiting, not just double
    assert _raised_concurrency(at(0.5), 8) == 8  # 8 * 0.85 / 0.5 leaves no room to grow
    ungauged = Measurement(10, 0, peak_waiting=24.0, peak_kv_usage=0.5)
    no_gauge = situation(
        ungauged, WorkloadFacts(prompts=("p",), output_tokens=128, context_limit=4096), Settings()
    )
    assert _raised_concurrency(no_gauge, 8) == 8  # no running gauge: scaled from the current cap

    # The reason names the declared load; it never presents it as measured traffic.
    entry = next(r for r in PLAYBOOK if r.name == "raise-concurrency")
    facts = WorkloadFacts(
        prompts=("p",), output_tokens=128, context_limit=4096, load="32 concurrent clients"
    )
    reason = entry.applies(situation(seen, facts, Settings(max_concurrent_requests=8)))
    assert reason is not None and "at your declared load of 32 concurrent clients" in reason
    assert (
        "highest sampled running plus waiting was 32" in reason and "not measured traffic" in reason
    )
    assert "cuts queueing (TTFT) but slows each token (TPOT)" in reason


@pytest.mark.parametrize(
    "blocks, kv, running, hybrid, raised",
    [
        (57.0, 40 / 56, 10.0, True, 14),  # 0.3.6 run 5: 16 of 56 blocks free, 4 per request
        (57.0, 40 / 56, 10.0, False, None),  # the same numbers on a plain cache: blocked
        (30.0, 28 / 29, 7.0, True, None),  # 0.3.6 run 1's Try: 1 of 29 blocks free
    ],
)
def test_a_hybrid_cache_raises_the_cap_to_the_requests_its_blocks_hold(
    blocks: float, kv: float, running: float, hybrid: bool, raised: int | None
) -> None:
    signals = Measurement(
        160, 0, hybrid_cache=hybrid, kv_blocks=blocks, peak_kv_usage=kv, peak_running=running,
        peak_waiting=22.0, peak_in_flight=32,
    )  # fmt: skip
    facts = WorkloadFacts(prompts=("p",), output_tokens=128, context_limit=4096)
    run = situation(signals, facts, Settings(max_concurrent_requests=int(running)))
    entry = next(r for r in PLAYBOOK if r.name == "raise-concurrency")
    if raised is None:
        assert entry.applies(run) is None and entry.blocked(run)
        if hybrid:
            assert "the 29 usable blocks of this hybrid cache hold only 7" in entry.blocked(run)
        return
    assert entry.apply(run).max_concurrent_requests == raised
    assert f"the 56 usable blocks hold {raised} requests of 4 blocks" in entry.applies(run)


@pytest.mark.parametrize(
    "kv, pool, cache, cold, memory, cap",
    [
        (0.6, 103_856.0, "warm", 70_800.0, None, None),  # 0.85 x 70,800 / 3,894 = 15, not above 16
        (0.6, 103_856.0, "warm", None, None, 22),  # no cold figure known: its own pool
        (0.46, 70_912.0, "cold", None, (10.0, 22.5), 31),  # 0.95 of 22.5 GiB: pool x 10.675 / 10
    ],
)
def test_a_raised_cap_is_sized_in_kv_tokens(
    kv: float, pool: float, cache: str, cold: float | None,
    memory: tuple[float, float] | None, cap: int | None,
) -> None:  # fmt: skip
    signals = Measurement(
        100, 0, peak_running=16.0, peak_waiting=48.0, peak_in_flight=64, peak_kv_usage=kv,
        kv_capacity_tokens=pool, compile_cache=cache, request_throughput=1.0,
        total_throughput=2005.0, kv_available_gib=memory[0] if memory else None,
        gpu_memory_gib=memory[1] if memory else None,
    )  # fmt: skip
    run = situation(signals, FACTS, Settings(max_concurrent_requests=16))
    with active(AnalysisContext(cold_kv_tokens=cold)):
        if memory is None:
            sized = next(r for r in PLAYBOOK if r.name == "kv-sized-concurrency")
            assert (sized.apply(run).max_concurrent_requests if sized.applies(run) else None) == cap
            return
        more = next(r for r in PLAYBOOK if r.name == "more-kv-memory")
        raised = with_cap_for_more_memory(run, more.apply(run), VLLM)
    assert (raised.kv_memory_fraction, raised.max_concurrent_requests) == (0.95, cap)


@pytest.mark.parametrize(
    "signals, settings, offered",
    [
        # A TPOT tail below the 2.0 gate is not offered, near it or not.
        (dict(tpot_p50_ms=10.0, tpot_p95_ms=16.0), {}, None),
        (dict(tpot_p50_ms=10.0, tpot_p95_ms=25.0), {}, None),  # past the gate: exact, not near
        (
            dict(ttft_p50_ms=300.0, tpot_p50_ms=20.0, peak_waiting=3.0),  # 15x against 20
            {},
            "TTFT p50 is 15x TPOT p50 with requests waiting; the gate is 20x",
        ),
        # Queued at a cap of 32 with KV at 60%: raising it cannot fit, so more memory is offered.
        (
            dict(peak_running=32.0, peak_waiting=8.0, peak_in_flight=40, peak_kv_usage=0.6),
            dict(max_concurrent_requests=32),
            "raising max_concurrent_requests above 32 is blocked",
        ),
        (  # already at 0.95: nothing to offer
            dict(peak_running=32.0, peak_waiting=8.0, peak_in_flight=40, peak_kv_usage=0.6),
            dict(max_concurrent_requests=32, kv_memory_fraction=0.95),
            None,
        ),
        (  # exactly at the gate: crossed, so offered outright, not near
            dict(tpot_p50_ms=10.0, tpot_p95_ms=20.0),
            {},
            None,
        ),
        (  # prompt work named the bottleneck but steps are not full: offered, not Try first
            dict(
                prompt_work_share=0.8,
                peak_waiting=3.0,
                mean_waiting=3.0,
                scheduled_tokens_per_step=900.0,
                step_budget_tokens=2048,
            ),
            {},
            "prompt work takes 80% of engine time with requests waiting, but steps are not full",
        ),
        (  # at 20% the cap can be raised, so nothing is blocked
            dict(peak_running=32.0, peak_waiting=8.0, peak_in_flight=40, peak_kv_usage=0.2),
            dict(max_concurrent_requests=32),
            None,
        ),
    ],
)
def test_low_bar_entries_are_offered_only_on_request(
    signals: dict[str, Any], settings: dict[str, Any], offered: str | None
) -> None:
    measurement = Measurement(10, 0, **signals)
    facts = WorkloadFacts(prompts=("p",), output_tokens=128, context_limit=4096)
    current = Settings(**settings)
    run = situation(measurement, facts, current)
    plain, _ = applicable(PLAYBOOK, run, allow_quality_changes=True)
    found, _ = applicable(PLAYBOOK, run, allow_quality_changes=True, near=True)
    low = [s.reason for s in found if s.tier == "could_help"]
    assert all(s.tier == "try_first" for s in plain)
    assert not {s.entry.name for s in plain} & {s.entry.name for s in found if s.near}
    assert [offered in reason for reason in low] == ([True] if offered else [])


@pytest.mark.parametrize(
    "kv, fraction, see",
    [
        (0.6, None, "; see `more-kv-memory` under Could help"),
        (0.89, None, "; see `more-kv-memory` under Could help"),
        (0.92, None, None),  # KV-bound: the diagnosis is KV capacity, more-kv-memory is Try first
        (0.6, 0.95, ""),  # nothing to offer, the line still shows
    ],
)
def test_next_steps_say_why_the_cap_cannot_rise(
    kv: float, fraction: float | None, see: str | None
) -> None:
    measurement = Measurement(
        10, 0, queue_share=0.9, peak_running=32.0, peak_waiting=8.0, peak_in_flight=40,
        peak_kv_usage=kv, tpot_p50_ms=10.0, tpot_p95_ms=16.0,
    )  # fmt: skip
    settings = Settings(max_concurrent_requests=32, kv_memory_fraction=fraction)
    facts = WorkloadFacts(prompts=("p",), output_tokens=128, context_limit=4096)
    run = situation(measurement, facts, settings)
    suggested = suggestions(ENGINES["vllm"], run, current=settings, project_path=None)
    blocked = dataclasses.replace(
        suggested, not_applicable=(), blocked={"queueing": "raising the cap is blocked"}
    )
    text = sections_markdown([next_steps_section(run.diagnosis, blocked, retained=True)])
    if see is None:
        assert (
            "blocked" not in text
            and "Try first:\nFor KV-cache capacity:\n- `more-kv-memory`" in text
        )
    else:
        assert (
            f"Try first:\nFor queueing before scheduling:\n- raising the cap is blocked{see}\n"
            in text
        )


@pytest.mark.parametrize(
    ("goal", "cap", "peak", "preemptions", "expected"),
    [
        ("throughput", 64, 25.0, 0, None),  # 32 never binds at 25
        ("throughput", 64, 25.0, 5, "fewer concurrent sequences fit the KV cache"),
        ("tpot", 256, 4.0, 0, None),  # a saturated GPU, but the load never reached the cap
        ("tpot", 256, 256.0, 0, "the GPU was saturated with 256 sequences running at once, "
                                "at the cap"),
    ],
)  # fmt: skip
def test_lower_concurrency_needs_a_cap_that_binds(
    goal: Goal, cap: int, peak: float, preemptions: int, expected: str | None
) -> None:
    entry = next(r for r in PLAYBOOK if r.name == "lower-concurrency")
    facts = WorkloadFacts(prompts=("p",), output_tokens=128, context_limit=4096)
    settings = Settings(max_concurrent_requests=cap)
    measured = Measurement(10, 0, peak_running=peak, peak_waiting=0.0, preemptions=preemptions)
    gpu = GpuState(index=0, samples=60, utilization=99.0, power_w=68.0, power_limit_w=70.0)
    with active(AnalysisContext(goal=goal)):
        diagnosis = classify(measured, settings, None, gpu_state=gpu)
        run = Situation(measurement=measured, facts=facts, settings=settings, diagnosis=diagnosis)
        found = entry.applies(run)
    assert (found is None) if expected is None else (found is not None and expected in found)


def test_numeric_entries_walk_their_setting_in_steps_of_at_most_four() -> None:
    from tensward.playbook import _ladder

    assert _ladder(8, 64) == [16, 32, 64]
    assert _ladder(8, 512) == [16, 64, 128, 512]  # six doublings, four spread evenly
    assert _ladder(2048, 8192) == [4096, 8192]
    assert _ladder(8192, 3072) == [4096, 3072]  # down, never past the target
    assert _ladder(0.85, 0.95) == [0.9, 0.95]

    seen = Measurement(10, 0, peak_running=8.0, peak_waiting=24.0, peak_kv_usage=0.06)
    current = Settings(
        dtype="float16",
        max_concurrent_requests=8,
        max_context_len=2048,
        kv_memory_fraction=0.8,
        prefill_batch_tokens=2048,
        kv_cache_dtype="auto",
        prefix_caching=True,
    )
    facts = WorkloadFacts(prompts=("p",), output_tokens=128, context_limit=4096)
    raise_concurrency = next(r for r in PLAYBOOK if r.name == "raise-concurrency")
    run = situation(seen, facts, current)
    steps = rungs(raise_concurrency, run)
    assert [s.max_concurrent_requests for s in steps] == [16, 32]  # the evidence says 32
    assert all(s.prefill_batch_tokens == 2048 for s in steps)
    assert rungs(raise_concurrency, run, prepare=lambda s: None) == []
    assert rungs(raise_concurrency, run, prepare=lambda s: current) == []


def test_prefix_caching_leads_when_kv_blocks_a_higher_cap_and_prompts_share_a_prefix() -> None:
    prefix = "shared " * 1900
    facts = WorkloadFacts(
        prompts=tuple(f"{prefix}turn {i} " + "word " * 12 for i in range(20)),
        output_tokens=128,
        context_limit=8192,
    )
    measurement = Measurement(
        200, 0, ttft_p50_ms=900.0, tpot_p50_ms=60.0, tpot_p95_ms=70.0, peak_kv_usage=0.63,
        peak_running=16.0, peak_waiting=16.0, peak_in_flight=32, queue_share=0.6,
        preemptions=0.0,
    )  # fmt: skip
    settings = Settings(max_concurrent_requests=16, prefix_caching=False)
    blocked = situation(measurement, facts, settings)
    suggested = suggestions(ENGINES["vllm"], blocked, current=settings, project_path=None)
    caching = next(s for s in suggested.suggestions if s.entry.name == "prefix-caching")
    assert caching.tier == "try_first" and "queueing" in caching.addresses
    text = sections_markdown([next_steps_section(blocked.diagnosis, suggested, retained=True)])
    assert text.index("prefix-caching") < text.index("more-kv-memory")
    assert "prefix caching holds once" in caching.reason
    with active(AnalysisContext(goal="tpot")):  # no change here lowers TPOT
        tpot = suggestions(ENGINES["vllm"], blocked, current=settings, project_path=None)
    kept = next(s for s in tpot.suggestions if s.entry.name == "prefix-caching")
    assert kept.tier == "try_first" and (kept.cost or "").startswith("does not change TPOT")

    cached = dataclasses.replace(settings, prefix_caching=True)
    lighter = dataclasses.replace(measurement, peak_kv_usage=0.5)  # linear: 16 is the most
    on = situation(lighter, facts, cached)
    raised = next(
        s
        for s in suggestions(ENGINES["vllm"], on, current=cached, project_path=None).suggestions
        if s.entry.name == "raise-concurrency"
    )
    assert raised.settings is not None and raised.settings.max_concurrent_requests == 32


def test_the_kv_projection_without_prompts_or_a_running_gauge() -> None:
    measurement = Measurement(100, 0, peak_kv_usage=0.5)  # no running gauge
    empty = situation(
        measurement,
        WorkloadFacts(prompts=(), output_tokens=128, context_limit=4096),
        Settings(max_concurrent_requests=8),
    )
    assert prefix_and_rest(empty.facts) == (0.0, 64.0)
    assert projected_kv(empty, 16, caching=True) == pytest.approx(1.0)  # linear from the cap


WHITESPACE = StructuredAnswerStats(
    answers=20, invalid_json=0.2, runaway_whitespace=0.1, cut_short=0.2, runaway_prompts=("a",)
)
NO_LOOP = dataclasses.replace(WHITESPACE, runaway_whitespace=0.0, runaway_prompts=())
FLAG = "--structured-outputs-config"
XGRAMMAR = {"backend": "xgrammar", "disable_any_whitespace": True}
GUIDANCE = {"backend": "guidance", "disable_any_whitespace": True}


@pytest.mark.parametrize(
    ("stats", "given", "expected"),
    [
        (WHITESPACE, None, XGRAMMAR),
        (WHITESPACE, '{"backend": "auto"}', XGRAMMAR),
        (WHITESPACE, '{"backend": "guidance"}', GUIDANCE),
        (WHITESPACE, '{"backend": "outlines"}', None),
        (WHITESPACE, json.dumps(XGRAMMAR), None),
        (NO_LOOP, None, None),
        (None, None, None),
    ],
)
def test_compact_json_is_offered_from_the_answers_with_a_backend_that_takes_it(
    stats: StructuredAnswerStats | None, given: str | None, expected: dict | None
) -> None:
    measurement = Measurement(20, 0, structured_answers=stats)
    settings = Settings(extra_args={FLAG: given} if given else {})
    facts = WorkloadFacts(prompts=("p",), output_tokens=128, context_limit=4096, structured=True)
    found, _ = applicable(
        PLAYBOOK, situation(measurement, facts, settings), allow_quality_changes=True, near=True
    )
    offered = next((s for s in found if s.entry.name == "compact-json"), None)
    if expected is None:
        assert offered is None
        return
    assert offered.tier == "could_help" and offered.basis == "your answers"
    applied = offered.entry.apply(situation(measurement, facts, settings))
    assert json.loads(applied.extra_args[FLAG]) == expected


class NarrowEngine(VllmEngine):
    """An engine that realises speculation only by a draft model, holds back the KV-cache
    precision whenever a hybrid cache ran, and has no API-server lever."""

    name = "narrow"

    def capabilities(self, version: str | None, settings: Settings | None = None) -> Capabilities:
        declared = VLLM.capabilities(version, settings)
        levers = {
            **{k: v for k, v in declared.levers.items() if k != "api_server_count"},
            "speculation": LeverSupport(values=frozenset({"draft"})),
        }
        return dataclasses.replace(declared, levers=levers)

    hold_backs = {
        **VLLM.hold_backs,
        "kv_cache_precision": lambda s: "held here" if s.measurement.hybrid_cache else None,
    }


def test_an_entry_is_loaded_only_where_the_engine_realises_its_levers_and_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = [entry.name for entry in load_entries(NarrowEngine(), None, family="vllm")]
    vllm = [entry.name for entry in VLLM.playbook()]
    assert [n for n in vllm if n not in names] == ["ngram-speculation", "more-api-servers"]
    assert "drop-speculation" in names  # turns speculation off: no value to realise

    on = Entry(name="ext-on", addresses=frozenset({"kv_capacity"}), applies=lambda s: "go",
               apply=lambda s: s.settings, evidence="moderate", costs="c",
               levers=frozenset({"kv_cache_precision"}))  # fmt: skip
    absent = dataclasses.replace(on, name="ext-absent", levers=frozenset({"api_server_count"}))
    anywhere = dataclasses.replace(on, name="ext-anywhere", levers=frozenset())
    extender = Extender(api_version=1, analyse=lambda *a: None, entries=(on, absent, anywhere))
    monkeypatch.setattr("tensward.suggest.load_extender", lambda: extender)
    hybrid = situation(Measurement(1, 0, hybrid_cache=True), FACTS, Settings())
    found, gated = applicable((), hybrid, allow_quality_changes=True, engine=NarrowEngine())
    assert [s.entry.name for s in found] == ["ext-anywhere"]
    assert [(entry.name, why) for entry, why in gated] == [("ext-on", "held here")]


def test_each_entry_sets_the_lever_values_it_declares() -> None:
    run = situation(Measurement(1, 0), FACTS, Settings())
    declared = [entry for entry in VLLM.playbook() if entry.sets]
    assert {entry.name for entry in declared} == {
        "ngram-speculation",
        "fp8-kv-cache",
        "fp8-weights",
        "bnb-4bit-weights",
    }
    for entry in declared:
        applied = entry.apply(run)
        for lever, value in entry.sets.items():
            read = VLLM.lever_value(applied, lever)
            assert (read.method if lever == "speculation" else read) == value, entry.name


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
    # A date line ahead of the shared text leaves the cache nothing to serve.
    dated_run = run_of([dated(i) for i in range(128)], [i % 128 for i in range(384)])
    for run, lead in ((AGENT_RUN, "kv_capacity"), (dated_run, None)):
        suggested = suggestions(VLLM, full(FULL), current=OFF, project_path=None, run=run)
        assert suggested.lead == lead
        assert ("prefix-caching" in suggested.estimates) == (lead is not None)
    # Cap-bound and prompt-heavy: queueing stays a finding beside prompt processing, so
    # raise-concurrency stays in Try first.
    capped = ran(PROMPTY, CAP8)
    assert capped.diagnosis.primary == "prefill" and "queueing" in capped.diagnosis.secondary
    offered = suggestions(VLLM, capped, current=CAP8, project_path=None, run=CLOSED).suggestions
    assert any(s.entry.name == "raise-concurrency" and s.tier == "try_first" for s in offered)
    # Below the cap with full steps: queueing is the budget's symptom.
    below = ran(replace(PROMPTY, mean_running=50.0, peak_running=51.0, step_ms=268.0,
                        prompt_tokens_per_step=2000.0, scheduled_tokens_per_step=2049.0,
                        prompt_work_share=0.9),
                replace(CAP8, max_concurrent_requests=128))  # fmt: skip
    assert "queueing" in below.diagnosis.symptoms and "queueing" not in below.diagnosis.secondary
    # A load-limited run: no server change in Try first, no promotion, the test is named.
    idle = replace(
        FULL, peak_kv_usage=0.12, mean_kv_usage=0.12, mean_running=62.0, peak_running=64.0,
        mean_in_flight=64.0, mean_waiting=0.0, peak_waiting=0.0, prompt_work_share=0.1,
        scheduled_tokens_per_step=400.0,
    )  # fmt: skip
    situation = full(idle)
    assert situation.diagnosis.primary == "load_limited"
    suggested = suggestions(VLLM, situation, current=OFF, project_path=None, run=AGENT_RUN)
    assert suggested.lead is None and all(s.tier == "could_help" for s in suggested.suggestions)
    text = try_first(situation, AGENT_RUN)
    assert "raise the workload's concurrency" in text and "tensward analyse" not in text


def _offer(
    name: str, thr: str, ttft: str, tpot: str, addresses: frozenset = frozenset({"queueing"})
) -> Suggestion:
    effects = {
        m: Effect(d, f"{name} on {m}")
        for m, d in zip(("throughput", "ttft", "tpot"), (thr, ttft, tpot))
    }
    entry = Entry(name=name, addresses=addresses, applies=lambda s: None,
                  apply=lambda s: s.settings, evidence="strong", costs="its costs",
                  effects=effects)  # fmt: skip
    return Suggestion(entry=entry, reason="its signal", tier="try_first", addresses=addresses)


OFFERED = [
    _offer("cap-up", "improves", "worsens", "none"),
    _offer("faster", "improves", "improves", "worsens"),
    _offer("latency", "none", "improves", "worsens"),
    _offer("tpot-only", "worsens", "worsens", "improves"),
    _offer("fix", "none", "none", "none", frozenset({"fit"})),
]
NOW = "TTFT p95 is 900 ms; "


@pytest.mark.parametrize(
    ("goal", "limit", "expected", "left_out"),
    [
        ("throughput", None,
         [("cap-up", "try_first", "its costs"), ("faster", "try_first", "its costs"),
          ("fix", "try_first", "its costs"),
          ("latency", "could_help",
           "does not change throughput (latency on throughput); helps TTFT")],
         ("tpot-only",)),
        ("ttft", None,
         [("faster", "try_first", "TPOT worsens (faster on tpot)"),
          ("latency", "try_first", "TPOT worsens (latency on tpot)"),
          ("fix", "try_first", "its costs")],
         ("cap-up", "tpot-only")),
        ("tpot", None,
         [("tpot-only", "try_first",
           "throughput worsens (tpot-only on throughput); TTFT worsens"),
          ("fix", "try_first", "its costs"),
          ("cap-up", "could_help", "does not change TPOT (cap-up on tpot); helps throughput")],
         ("faster", "latency")),
        ("throughput", 800.0,  # past the limit: a TTFT-raising change leaves Try first
         [("faster", "try_first", "its costs"),
          ("fix", "try_first", "its costs"),
          ("cap-up", "could_help", "TTFT p95 is 900 ms, past your limit of 800 ms"),
          ("latency", "could_help", "does not change throughput")],
         ("tpot-only",)),
        ("tpot", 800.0,
         [("fix", "try_first", "its costs"),
          ("cap-up", "could_help", "does not change TPOT"),
          ("tpot-only", "could_help", "TTFT p95 is 900 ms, past your limit of 800 ms")],
         ("faster", "latency")),
        ("throughput", 1200.0,  # within the limit: order kept, the risk still named
         [("cap-up", "try_first", f"{NOW}may push it past your limit of 1200 ms"),
          ("faster", "try_first", "its costs"),
          ("fix", "try_first", "its costs"),
          ("latency", "could_help", "does not change throughput")],
         ("tpot-only",)),
    ],
)  # fmt: skip
def test_a_goal_ranks_changes_by_their_effects(
    goal: str, limit: float | None, expected: list, left_out: tuple
) -> None:
    ranked, other = by_goal(OFFERED, goal, Measurement(100, 0, ttft_p95_ms=900.0), limit)
    assert [(s.entry.name, s.tier) for s in ranked] == [(n, t) for n, t, _ in expected]
    assert all((s.cost or s.entry.costs).startswith(c) for s, (_, _, c) in zip(ranked, expected))
    assert other == left_out


def test_every_change_that_worsens_ttft_is_left_out_and_named_once() -> None:
    worse = [_offer("tpot-only", "worsens", "worsens", "improves")]
    ranked, other = by_goal(worse, "ttft", Measurement(100, 0), None)
    assert ranked == [] and other == ("tpot-only",)


def test_the_new_entries_are_offered_only_where_asked() -> None:
    entries = VLLM.playbook()
    assert {e.name for e in entries if e.opt_in} == {
        "kv-sized-concurrency",
        "small-step-budget",
        "stream-interval",
        "async-scheduling",
        "no-async-scheduling",
        "fp8-weights",
        "bnb-4bit-weights",
        "quantized-checkpoint",
    }
    run = situation(Measurement(100, 0), FACTS, Settings(async_scheduling=False))
    asked, _ = applicable(entries, run, allow_quality_changes=True, opt_in=True)
    default, _ = applicable(entries, run, allow_quality_changes=True)
    assert "async-scheduling" in {s.entry.name for s in asked} - {s.entry.name for s in default}


@pytest.mark.parametrize(
    ("cap", "memory", "refused"),
    [
        (None, 22.0, False),
        (None, 80.0, True),
        (1024, 22.0, True),
        (256, 80.0, False),
        (None, None, True),
    ],
)
def test_a_step_budget_below_the_cap_is_refused(
    cap: int | None, memory: float | None, refused: bool
) -> None:
    measured = Measurement(100, 0, gpu_memory_gib=memory, ceilings=Ceilings(gpu="NVIDIA H100"))
    proposed = Settings(prefill_batch_tokens=512, max_concurrent_requests=cap)
    assert (budget_refusal(VLLM, proposed, measured, "0.30.0") is not None) == refused


def _past(
    run_id: str,
    args: tuple[str, ...],
    rps: float | None,
    ttft: float = 500.0,
    tpot: float = 50.0,
    served: bool = True,
    **expected: float,
) -> PastRun:
    return PastRun(run_id=run_id, engine_args=frozenset(args), settings={},
                   metrics={"request_throughput": rps, "ttft_p95_ms": ttft, "tpot_p95_ms": tpot},
                   primary="queueing",
                   expected={frozenset(k.split()): ("raise-concurrency", v)
                             for k, v in expected.items()},
                   compile_cache="warm", kv_capacity_tokens=1000.0, served=served)  # fmt: skip


def _offered(name: str, *args: str) -> Suggestion:
    words = " ".join(f"--engine-arg {arg}" for arg in args)
    command = f"tensward analyse --project /p {words}"
    return dataclasses.replace(_offer(name, "improves", "improves", "none"), command=command)


@pytest.mark.parametrize(
    ("goal", "runs", "now", "args", "offered", "expect"),
    [
        ("throughput", [_past("r0", (), 10.0), _past("r1", ("a=1",), 11.0)], 10.0, (),
         [_offered("change-a", "a=1")],
         {"kept": [], "line": "`change-a` (a=1) tried in run r1: requests per second +10% "
                              "against run r0, reverted"}),
        ("throughput", [_past("r0", (), 10.0)], 9.0, ("a=1",), [],
         {"revert": (), "reason": "-10%"}),
        ("throughput", [_past("r0", (), 10.0)], 11.0, ("a=1",), [], {"revert": None}),
        ("throughput", [_past("r0", (), 10.0), _past("r1", ("pc",), 10.0),
                        _past("r2", ("pc", "mns=128"), 14.0)], 12.0, ("pc", "mns=64"), [],
         {"revert": ("pc", "mns=128"), "reason": "the last change (mns=64) moved requests per "
                                                 "second -14% against run r2",
          "line": "mns=128 tried in run r2: requests per second +40% against run r1, changed "
                  "to mns=64 in this run", "behind": "r2"}),
        ("throughput", [_past("r0", (), 10.0, **{"a=1": 1.5})], 12.0, ("a=1",), [],
         {"line": "`raise-concurrency` helped less than run r0 expected (measured x1.20, "
                  "expected at least x1.50): queueing before scheduling took over, "
                  "the next target"}),
        ("ttft", [_past("r0", (), 10.0, ttft=800.0)], 900.0, ("a=1",), [],
         {"revert": (), "reason": "+12%"}),  # no repeats: a p95 loss needs more than 5%
        ("ttft", [_past("r0", (), 10.0, ttft=800.0)], 830.0, ("a=1",), [],
         {"revert": None}),  # +4%: noise
        ("tpot", [_past("r0", (), 10.0, tpot=187.5), _past("r0b", (), 10.0, tpot=194.6)], 205.0,
         ("a=1",), [], {"revert": None}),  # +5% against repeats 3.7% apart: within 2x noise
        ("throughput", [_past("r0", (), 10.0), _past("r1", ("a=1",), None, served=False)], 10.0, (),
         [_offered("change-a", "a=1")],
         {"kept": [], "line": "`change-a` (a=1) tried in run r1: it served no request"}),
        ("ttft", [_past("r0", (), 10.0, ttft=800.0)], 600.0, ("a=1",), [],
         {"rate": 11.7, "knee": None}),  # TTFT -25% and req/s +17%: no knee
        ("ttft", [_past("r0", ("b=1024",), 10.0, ttft=800.0)], 780.0, ("b=512",),
         [_offered("further", "b=256"), _offered("back", "b=4096")],
         {"rate": 6.0, "tiers": ["could_help", "try_first"]}),  # past the knee: only further
    ],
)  # fmt: skip
def test_history_decides_what_comes_next(goal, runs, now, args, offered, expect) -> None:
    measured = Measurement(
        100, 0, **{"request_throughput": expect.get("rate"), METRIC[goal][0]: now}
    )
    diagnosis = classify(measured, Settings(), None)
    diagnosis = dataclasses.replace(diagnosis, primary="queueing")
    facts = WorkloadFacts(prompts=(), output_tokens=8, context_limit=4096)
    run = Situation(measurement=measured, facts=facts, settings=Settings(), diagnosis=diagnosis)
    kept, memory = remember(
        offered, run, goal=goal, args=frozenset(args), runs=runs, project_path=Path("/p")
    )
    if "kept" in expect:
        assert kept == expect["kept"]
    if "line" in expect:
        assert expect["line"] in memory.lines
    if "revert" in expect:
        assert (None if memory.revert is None else command_args(memory.revert.command or "")) == (
            None if expect["revert"] is None else frozenset(expect["revert"])
        )
    if "reason" in expect:
        assert memory.revert is not None and expect["reason"] in memory.revert.reason
    if "behind" in expect:
        assert memory.behind and memory.ahead is not None
        assert memory.ahead.run_id == expect["behind"]
    if "knee" in expect:
        assert knee(runs, frozenset(args), run, goal) == expect["knee"]
    if "tiers" in expect:
        moved, _ = _past_knee(kept, knee(runs, frozenset(args), run, goal), frozenset(args))
        assert [s.tier for s in moved] == expect["tiers"]


@pytest.mark.parametrize(
    ("effect", "rate", "tier", "cost"),
    [
        ("none", 6.0, "try_first", None),
        ("worsens", 6.0, "could_help", "past your floor"),
        (
            "worsens",
            9.0,
            "try_first",
            "your floor is 8 (20% below run r0's 10), and this run made 9",
        ),
    ],
)
def test_the_floor_moves_only_a_change_that_lowers_requests_per_second(
    effect: str, rate: float, tier: str, cost: str | None
) -> None:
    offer = _offer("change-a", effect, "none", "improves")
    measured = Measurement(100, 0, request_throughput=rate, ttft_p95_ms=500.0)
    run = Situation(
        measurement=measured,
        facts=WorkloadFacts(prompts=(), output_tokens=8, context_limit=4096),
        settings=Settings(),
        diagnosis=classify(measured, Settings(), None),
    )
    with active(AnalysisContext(goal="ttft", max_req_s_loss=0.2)):
        [moved], floor = _below_floor([offer], _past("r0", (), 10.0), run)
    assert floor is not None and floor["crossed"] == (rate < 8.0)
    assert moved.tier == tier
    assert (cost in (moved.cost or "")) if cost else moved.cost is None


@pytest.mark.parametrize(
    ("saturated", "goal", "tried", "expected", "extra"),
    [
        (True, "throughput", "",
         "Nothing left in Try first for throughput: the GPU was saturated", {}),
        (True, "tpot", "lower-concurrency",
         "Nothing left in Try first for TPOT: the GPU was saturated at its power limit (busy 99%, "
         "at 68 of 70 W); running fewer requests at once lowers TPOT, at a throughput cost, but "
         "`lower-concurrency` was tried (see So far); beyond it, what helps is less work", {}),
        (False, "throughput", "x", "Every change Tensward has for throughput here was tried; "
                                   "the best run is r1", {}),
        (False, "throughput", "", "Nothing left in Try first for throughput: "
                                  "no measured signal points to a change for it", {}),
        (False, "tpot", "knee", "Nothing left in Try first for TPOT: the knee is reached", {}),
        (False, "ttft", "", "prompt processing ran at 82% of the prompt rate this engine measured "
                            "(4,559 tokens/s), about 18% headroom for scheduling changes; what is "
                            "left: prefix caching (hit rate 1%), a smaller or quantized model, a "
                            "faster GPU",
         {"prompt_work_share": 0.82, "prompt_capacity_tok_s": 4559.0,
          "prefix_cache_hit_rate": 0.008, "primary": "prefill"}),
        (True, "throughput", "", "the 128 clients set the batch",
         {"peak_running": 128.0, "peak_in_flight": 128}),
        (True, "ttft", "", "`small-step-budget` is applied in this run",
         {"applied": ("small-step-budget",)}),
    ],
)  # fmt: skip
def test_a_report_with_nothing_left_says_why(
    saturated: bool, goal: str, tried: str, expected: str, extra: dict[str, Any]
) -> None:
    gpu = (
        GpuState(index=0, samples=60, utilization=99.0, power_w=68.0, power_limit_w=70.0)
        if saturated
        else None
    )
    extra = dict(extra)
    settings = Settings()
    applied = extra.pop("applied", ())
    primary = extra.pop("primary", None)
    measured = Measurement(100, 0, request_throughput=10.0, **extra)
    diagnosis = classify(measured, settings, None, gpu_state=gpu)
    if primary:
        diagnosis = dataclasses.replace(diagnosis, primary=primary)
    facts = WorkloadFacts(prompts=(), output_tokens=8, context_limit=4096)
    run = Situation(measurement=measured, facts=facts, settings=settings, diagnosis=diagnosis)
    words = "the last step moved TPOT p95 -5% for requests per second -40%"
    memory = Memory(lines=(), revert=None, tried=frozenset({tried} - {"", "knee"}),
                    best=("r1", 12.0) if tried else None)  # fmt: skip
    found = stop_reason([], run, memory, goal, knee=words if tried == "knee" else None,
                        applied=applied)  # fmt: skip
    assert found is not None and expected in found


@pytest.mark.parametrize(("scheduled", "near"), [(635.0, False), (1000.0, True)])
def test_a_larger_budget_is_not_near_while_running_sits_at_the_cap(
    scheduled: float, near: bool
) -> None:
    measured = Measurement(
        100, 0, peak_running=16.0, peak_waiting=48.0, peak_in_flight=64, prompt_work_share=0.82,
        prompt_capacity_tok_s=4559.0, scheduled_tokens_per_step=scheduled, step_budget_tokens=1024,
    )  # fmt: skip
    settings = Settings(max_concurrent_requests=16)
    run = situation(measured, FACTS, settings)
    assert run.fired("prefill")
    assert (raise_prefill_batch_near(run, VLLM) is not None) == near


def _crossing(*classes: str) -> Situation:
    """A run queued at a cap of 16 with KV at 46%, whose diagnosis crosses ``classes``."""
    signals = Measurement(100, 0, peak_running=16.0, peak_waiting=48.0, peak_in_flight=64,
                          peak_kv_usage=0.46, kv_capacity_tokens=70_912.0)  # fmt: skip
    run = situation(signals, FACTS, Settings(max_concurrent_requests=16))
    findings = tuple(
        dataclasses.replace(f, state="warning" if f.bottleneck in classes else "clear")
        for f in run.diagnosis.findings
    )
    diagnosis = dataclasses.replace(run.diagnosis, findings=findings, primary=classes[0],
                                    secondary=tuple(classes[1:]))  # fmt: skip
    return dataclasses.replace(run, diagnosis=diagnosis)


@pytest.mark.parametrize(
    ("name", "cap", "goal", "crossed", "limit", "tested", "line"),
    [
        ("kv-sized-concurrency", 29, "throughput", ("queueing", "prefill"), "prefill", False,
         "then: prefill compute likely still binds; KV-cache usage projected at about 83%"),
        ("raise-concurrency", 32, "throughput", ("queueing", "prefill"), "kv_capacity", True,
         "then (prediction): KV-cache capacity likely binds next (watch for KV-cache usage "
         "projected at about 92%"),
        ("lower-concurrency", 8, "tpot", ("queueing", "decode_bandwidth"), "decode_bandwidth",
         False, "then: decode memory bandwidth likely still binds"),
    ],
)  # fmt: skip
def test_a_plan_predicts_only_a_limit_not_crossed_now(
    name: str, cap: int, goal: str, crossed: tuple[str, ...], limit: str, tested: bool, line: str
) -> None:
    run = _crossing(*crossed)
    entry = next(e for e in PLAYBOOK if e.name == name)
    top = Suggestion(entry=entry, reason="", tier="try_first", addresses=entry.addresses,
                     command=f"tensward analyse --engine-arg max-num-seqs={cap}",
                     settings=Settings(max_concurrent_requests=cap))  # fmt: skip
    plan = plan_for(top, [], goal, None, frozenset(), situation=run)
    assert (plan[0]["limit"], plan[0]["tested"]) == (limit, tested)
    assert plan_lines(plan, ())[0].startswith(line)


@pytest.mark.parametrize(
    ("limit", "tested", "now", "shown", "held"),
    [
        ("kv_capacity", True, ("kv_capacity",), "confirmed", False),
        ("prefill", False, ("prefill",), "said prefill compute still binds after", True),
        ("kv_capacity", True, ("queueing",), "not confirmed: this run shows queueing", False),
    ],
)  # fmt: skip
def test_a_check_confirms_only_a_tested_limit(
    limit: str, tested: bool, now: tuple[str, ...], shown: str, held: bool
) -> None:
    step = {"change": "raise-concurrency", "args": ["a=1"], "limit": limit, "signal": None,
            "tested": tested}  # fmt: skip
    before = PastRun(run_id="r0", engine_args=frozenset(), settings={}, metrics={},
                     primary=None, expected={}, plan=(step,), goal="throughput")  # fmt: skip
    text, check = checked([before], frozenset({"a=1"}), _crossing(*now), "throughput", engine=VLLM)
    assert text is not None and check is not None and shown in text
    assert (check["confirmed"], check["held"]) == (shown == "confirmed", held)


@pytest.mark.parametrize(
    ("cap_now", "cap_new", "at_cap"),
    [(16, 32, True), (32, 16, False)],
    ids=["raised", "halved below the cap"],
)
def test_time_stretches_only_for_a_lowered_cap_the_run_held(
    cap_now: int, cap_new: int, at_cap: bool
) -> None:
    context = AnalysisContext(launched_s=0.0, load_s=60.0, measured_s=360.0)
    assert seconds_to_test(context, cap_now=cap_now, cap_new=cap_new, at_cap=at_cap) == 360.0


@pytest.mark.parametrize(
    ("prompt", "answer", "expected"),
    [
        (" the cat sat on the mat today", " the cat sat on the mat today", 4),
        ("alpha beta gamma", " delta epsilon zeta eta", 0),
        (" one two three four five", " one two three nine ten", 0),
    ],
)
def test_prompt_lookup_accepts_what_a_drafter_would_copy(
    prompt: str, answer: str, expected: int
) -> None:
    assert accepted(words(prompt), words(answer)) == expected


def test_coverage_pools_the_answers_and_keeps_the_least_helped_tail() -> None:
    pairs = [(" a b c d e f g h", " a b c d e f g h")] * 19 + [(" x y z", " q r s t")]
    found = coverage(pairs)
    assert found is not None
    assert found.tail == 0.0 and found.pooled == pytest.approx(76 / 156)


def test_answers_before_after_counts_changed_answers_and_json_validity() -> None:
    before = {"a": (Answer('{"x": 1}', None, None),), "b": (Answer("same", None, None),)}
    escaped = Answer("{x\x1b]52;c;eA==\x07 <b>", None, None)
    after = {"a": (escaped,), "b": (Answer("same", None, None),)}
    outcome = compare_answers(before, after, {}, json_answers=frozenset({"a"}), failed=(0.0, 0.0))
    pairs = tuple(
        {"prompt_id": name, "before": [before[name][0].text], "after": [after[name][0].text],
         "identical": name == "b", "changed": False}
        for name in before
    )  # fmt: skip
    shown = AnswersAfter("r1", ("quantization",), outcome, pairs=pairs, broken=((0, 2), (0, 2)))
    section = answers_after_section(shown)
    assert section is not None
    assert [block.text for block in section.blocks if isinstance(block, Text)] == [
        "This run changes the weight quantization from run r1.",
        "Answers differ from run r1's on 1 of 2 prompts (word overlap 0.64 on average, where 1 is "
        "the same words); 1 is identical.",
        "answers that repeat themselves or are empty: 0 of 2 before, 0 of 2 after",
        "valid JSON: 100.0% before, 0.0% after",
        "failed requests: 0.0% before, 0.0% after",
        "The 1 least similar prompt:",
        "Prompt `a`:",
        'before: `{"x": 1}`',
        "after: `{x <b>`",
        "Every pair is in quality.json in this run's directory. "
        "Tensward does not judge which answers are better.",
    ]


# --- async scheduling off ---


def _prefill_scrapes(directory: Path, ms_per_request: float) -> None:
    for name, total in (("before", 0.0), ("after", 100 * ms_per_request / 1e3)):
        count = 0 if name == "before" else 100
        (directory / f"metrics_{name}.prom").write_text(
            f'vllm:request_prefill_time_seconds_sum{{engine="0"}} {total}\n'
            f'vllm:request_prefill_time_seconds_count{{engine="0"}} {count}\n'
        )


@pytest.mark.parametrize(
    ("prefill_ms", "async_scheduling", "offered"),
    [(210, None, False), (250, False, False)],
)
def test_async_scheduling_goes_off_for_ttft_when_prefill_takes_two_steps(
    tmp_path: Path, prefill_ms: float, async_scheduling: bool | None, offered: bool
) -> None:
    _prefill_scrapes(tmp_path, prefill_ms)
    run = situation(
        Measurement(100, 0, step_ms=111.0), FACTS, Settings(async_scheduling=async_scheduling)
    )
    with active(AnalysisContext(goal="ttft", new_run=tmp_path)):
        found, _ = applicable(PLAYBOOK, run, allow_quality_changes=False, opt_in=True)
    assert ("no-async-scheduling" in [s.entry.name for s in found]) is offered


@pytest.mark.parametrize(("async_scheduling", "offered"), [(False, True), (None, False)])
def test_the_256_token_step_is_offered_sooner_once_async_scheduling_is_off(
    async_scheduling: bool | None, offered: bool
) -> None:
    measured = Measurement(
        100, 0, ttft_p95_ms=350.0, tpot_p50_ms=100.0, step_budget_tokens=512,
        max_prompt_tokens=150, peak_running=64.0,
    )  # fmt: skip
    run = situation(measured, FACTS, Settings(async_scheduling=async_scheduling))
    with active(AnalysisContext(goal="ttft")):
        found, _ = applicable(PLAYBOOK, run, allow_quality_changes=False, opt_in=True)
    assert ("small-step-budget" in [s.entry.name for s in found]) is offered
