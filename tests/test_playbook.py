"""The playbook entries: when each applies, what it changes, and how it is offered."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

from tensward.classify import classify
from tensward.engines import ENGINES
from tensward.engines.vllm.playbook import PLAYBOOK
from tensward.jsonanswers import StructuredAnswerStats
from tensward.measurement import Measurement
from tensward.playbook import Entry, Situation, WorkloadFacts, fitting_max_context_len, rungs
from tensward.playbook_common import prefix_and_rest, projected_kv
from tensward.report.build import next_steps_section
from tensward.report.markdown import sections_markdown
from tensward.settings import Settings
from tensward.suggest import applicable, suggestions


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
    "signals, settings, offered",
    [
        # TPOT tail 1.6x against the 2.0 gate: near, offered with its value and the gate.
        (dict(tpot_p50_ms=10.0, tpot_p95_ms=16.0), {}, "TPOT p95 is 1.6x p50; the gate is 2.0x"),
        (dict(tpot_p50_ms=10.0, tpot_p95_ms=13.0), {}, None),  # 1.3x is under 0.7 of the gate
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
    if not run.fired("kv_capacity"):  # KV-bound, the TPOT tail is not near its gate
        assert suggested.suggestions[-1].entry.name == "lower-prefill-batch"  # near items last
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


def test_lower_concurrency_needs_a_cap_that_binds() -> None:
    entry = next(r for r in PLAYBOOK if r.name == "lower-concurrency")
    facts = WorkloadFacts(prompts=("p",), output_tokens=128, context_limit=4096)
    settings = Settings(max_concurrent_requests=64)
    idle = Measurement(10, 0, peak_running=25.0, peak_waiting=0.0, preemptions=0)
    assert entry.applies(situation(idle, facts, settings)) is None  # 32 never binds at 25
    thrashing = Measurement(10, 0, peak_running=25.0, peak_waiting=0.0, preemptions=5)
    assert entry.applies(situation(thrashing, facts, settings)) is not None


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
