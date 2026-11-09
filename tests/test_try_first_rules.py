"""Try first ordered by predicted TTFT, and two rules from measured runs: a
runnable fallback when Try first holds no command, and KV precision before weight precision on
tool-call workloads. Stacks of two to four changes on the Gemma 4 E4B chat run, with each track's
declared limits: what leads, and what follows a judge's or a start-up's verdict."""

from __future__ import annotations

import dataclasses
import itertools
import json
import shlex
import shutil
from pathlib import Path
from typing import Any

import pytest
from test_playbook import patch_entries
from test_replays import (
    ENGINE_SUPPORT,
    GEMMA_RUN,
    L4_8255,
    REPLAYED,
    VLLM,
    _gemma_cell,
    _l4_offers,
    gemma_e4b_anatomy,
    granite_situation,
    l4_situation,
    recorded_situation,
    replay_measurement,
)

from tensward.anatomy import freed_bytes
from tensward.classify import Finding
from tensward.combine import independent
from tensward.context import AnalysisContext, Goal, active
from tensward.engines.vllm.quantization import options_from
from tensward.history import Memory, PastRun
from tensward.measurement import Measurement
from tensward.playbook import Effect, Situation, Suggestion, WorkloadFacts
from tensward.predict import Prediction
from tensward.rejected import Rejection, kind_of
from tensward.report.build import next_steps_section
from tensward.report.model import Report, to_json
from tensward.suggest import (
    SPENT,
    Suggested,
    _by_start_cost,
    _last_pass,
    _spent,
    _without_worsening,
    by_goal,
    first_try,
)
from tensward.toolcalls import ToolCallStats

ENTRIES = {entry.name: entry for entry in VLLM.playbook()}
TOOLS = ToolCallStats(
    requests=100, calling=90, produced_call=0.9, valid_json=1.0, known_tool=0.9, schema_valid=0.2
)
NGRAM_SUFFIX = "this one has no strong signal here, so measure it"


def _try_first(situation: Situation, offered: Suggested) -> list[dict[str, Any]]:
    section = next_steps_section(situation.diagnosis, offered, retained=False)
    blocks = to_json(Report(run_id="r", ran="now", sections=(section,)))["sections"][0]["blocks"]
    return [b for b in blocks if b["kind"] == "suggestion" and b["tier"] == "try_first"]


def _predicting(monkeypatch: pytest.MonkeyPatch, **by_entry: Prediction) -> None:
    monkeypatch.setattr(
        "tensward.suggest.predict", lambda situation, proposed, *, entry, **kw: by_entry.get(entry)
    )


def _offering(monkeypatch: pytest.MonkeyPatch, *found: tuple[str, str, float | None]) -> None:
    offers = [
        Suggestion(
            entry=ENTRIES[name],
            reason=f"{name} fires",
            tier=tier,  # type: ignore[arg-type]
            addresses=frozenset(ENTRIES[name].addresses),
            tail_coverage=tail,
        )
        for name, tier, tail in found
    ]
    monkeypatch.setattr("tensward.suggest.applicable", lambda *a, **k: (offers, []))


def _gain(ttft: float) -> Prediction:
    return Prediction(tpot=-0.1, req_s=0.1, text=f"predicted: TTFT {ttft:+.0%}", ttft=ttft)


@pytest.mark.parametrize("ttft", [-0.22, -0.05])
def test_a_predicted_ttft_gain_moves_fp8_weights_to_the_front_of_try_first(
    ttft: float, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    situation, context = l4_situation(L4_8255, "ttft")
    same = Effect("none", "prompt processing runs at the same speed")
    patch_entries(monkeypatch, {"fp8-weights": {"effects": {"ttft": same}}})
    _predicting(monkeypatch, **{"fp8-weights": _gain(ttft)})
    offered = _l4_offers(situation, context, tmp_path)
    tier = next((s.tier for s in offered.suggestions if s.entry.name == "fp8-weights"), "phase_two")
    if ttft == -0.22:
        assert _try_first(situation, offered)[0]["key"] == "fp8-weights"
        assert tier == "try_first"
    else:
        assert tier == "phase_two"
        assert any(s.entry.name == "fp8-weights" for s in offered.phase_two)


def test_a_change_that_worsens_the_goal_is_left_out_whatever_it_is_predicted() -> None:
    waits = Effect("worsens", "requests wait longer to be admitted")
    entry = ENTRIES["lower-concurrency"]
    lower = Suggestion(
        entry=dataclasses.replace(entry, effects={**entry.effects, "ttft": waits}),
        reason="fires",
        tier="try_first",
        predicted=_gain(-0.4),
    )
    kept, left_out = by_goal([lower], "ttft", Measurement(100, 0), None)
    assert kept == [] and left_out == ("lower-concurrency",)


def test_a_predicted_ttft_fall_is_worded_as_a_p50_gain_not_a_rise() -> None:
    entry = ENTRIES["fp8-weights"]
    silent = Effect("none", "the record names no TTFT move")
    fp8 = Suggestion(
        entry=dataclasses.replace(entry, effects={**entry.effects, "ttft": silent}),
        reason="fires",
        tier="try_first",
        predicted=_gain(-0.26),
    )  # bf16 parent: logged "+26% on TTFT" for a fall; client TTFT p50 measured -54.5%
    kept, _ = by_goal([fp8], "ttft", Measurement(100, 0), None)
    assert "predict a 26% gain on TTFT p50" in kept[0].cost


def _a5(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    goal: Goal = "tpot",
    *found: tuple[str, str, float | None],
    **limits: float,
) -> tuple[Situation, Suggested]:
    situation, context = l4_situation(L4_8255, goal)
    context = dataclasses.replace(context, **limits)
    _offering(monkeypatch, *found)
    _predicting(monkeypatch)
    return situation, _l4_offers(situation, context, tmp_path)


A5_TRY_FIRST = ("stable-prompt-prefix", "try_first", None)


def test_a5_ends_where_it_did_since_its_replay_rules_out_a_gain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ngram = ("ngram-speculation", "could_help", 0.0)
    situation, offered = _a5(monkeypatch, tmp_path, "tpot", A5_TRY_FIRST, ngram)
    assert [b["key"] for b in _try_first(situation, offered) if b.get("command")] == []
    assert offered.stop is not None


@pytest.mark.parametrize("goal", ["tpot", "throughput"])
def test_a_safe_change_with_a_possible_gain_is_promoted_when_try_first_has_no_command(
    goal: Goal, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ngram = ("ngram-speculation", "could_help", 0.10)
    situation, offered = _a5(monkeypatch, tmp_path, goal, A5_TRY_FIRST, ngram)
    runnable = [b for b in _try_first(situation, offered) if b.get("command")]
    assert [b["key"] for b in runnable] == ["ngram-speculation"]
    promoted = next(s for s in offered.suggestions if s.entry.name == "ngram-speculation")
    assert promoted.reason.endswith(NGRAM_SUFFIX) and "decode_bandwidth" in promoted.addresses
    assert offered.stop is None


@pytest.mark.parametrize(
    ("name", "limits"),
    [
        ("fp8-weights", {}),
        ("ngram-speculation", {"max_req_s_loss": 0.0}),
        ("lower-concurrency", {"ttft_p95_limit_ms": 100.0}),
    ],
)
def test_a_risky_a_limit_breaking_or_ttft_worsening_change_is_never_promoted(
    name: str, limits: dict[str, float], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    situation, context = l4_situation(L4_8255, "tpot")
    context = dataclasses.replace(context, **limits)
    waits = Effect("worsens", "requests wait longer to be admitted")
    lower = ENTRIES["lower-concurrency"]
    lower = dataclasses.replace(lower, effects={**lower.effects, "ttft": waits})
    monkeypatch.setitem(ENTRIES, "lower-concurrency", lower)
    _offering(monkeypatch, A5_TRY_FIRST, (name, "could_help", 0.10))
    if limits.get("max_req_s_loss") == 0.0:
        _predicting(monkeypatch, **{name: Prediction(tpot=-0.1, req_s=-0.2, text="predicted")})
    else:
        _predicting(monkeypatch)
    offered = _l4_offers(situation, context, tmp_path)
    assert next(s for s in offered.suggestions if s.entry.name == name).tier == "could_help"
    assert offered.stop is not None


def test_a_try_first_change_a_past_run_measured_leaves_the_fallback_to_fire(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _offering(monkeypatch, ("fp8-kv-decode", "try_first", None),
              ("ngram-speculation", "could_help", 0.10))  # fmt: skip
    _predicting(monkeypatch)
    situation, context = l4_situation(L4_8255, "tpot")
    measured = PastRun(
        run_id="r0", engine_args=frozenset({"kv-cache-dtype=fp8"}), settings={}, metrics={},
        primary=None, expected={},
    )  # fmt: skip
    monkeypatch.setattr("tensward.history.past_runs", lambda *a, **k: [measured])
    offered = _l4_offers(situation, context, tmp_path)
    assert [b["key"] for b in _try_first(situation, offered)] == ["ngram-speculation"]


def test_with_a_runnable_try_first_offer_left_nothing_moves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ngram = ("ngram-speculation", "could_help", 0.10)
    situation, offered = _a5(
        monkeypatch, tmp_path, "tpot", ("fp8-kv-decode", "try_first", None), ngram
    )
    assert [b["key"] for b in _try_first(situation, offered)] == ["fp8-kv-decode"]


def test_on_a_load_limited_run_nothing_is_promoted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    situation, context = l4_situation(L4_8255, "tpot")
    loaded = dataclasses.replace(
        situation, diagnosis=dataclasses.replace(situation.diagnosis, primary="load_limited")
    )
    _offering(monkeypatch, ("ngram-speculation", "could_help", 0.10))
    _predicting(monkeypatch)
    offered = _l4_offers(loaded, context, tmp_path)
    assert not any(s.tier == "try_first" and s.command for s in offered.suggestions)
    assert not any(s.reason.endswith(NGRAM_SUFFIX) for s in offered.suggestions)


def _with_tools(situation: Situation, tools: ToolCallStats | None) -> Situation:
    return dataclasses.replace(
        situation, measurement=dataclasses.replace(situation.measurement, tool_calls=tools)
    )


@pytest.mark.parametrize(
    ("tools", "first"), [(TOOLS, "fp8-kv-decode"), (None, "fp8-weights+fp8-kv-decode")]
)
def test_with_tool_calls_fp8_kv_comes_before_fp8_weights_though_weights_lead_more(
    tools: ToolCallStats | None, first: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    situation, context = l4_situation(L4_8255, "tpot")
    situation = _with_tools(situation, tools)
    _predicting(
        monkeypatch,
        **{"fp8-weights": Prediction(tpot=-0.5, req_s=0.5, text="predicted"),
           "fp8-kv-decode": Prediction(tpot=-0.2, req_s=0.2, text="predicted")},
    )  # fmt: skip
    offered = _l4_offers(situation, context, tmp_path)
    assert _try_first(situation, offered)[0]["key"] == first


@pytest.mark.parametrize("goal", ["throughput", "ttft"])
@pytest.mark.parametrize(("tools", "first"), [(TOOLS, "fp8-kv-decode"), (None, "fp8-weights")])
def test_with_kv_capacity_primary_and_fp8_kv_unpredicted_fp8_kv_still_comes_first(
    goal: Goal,
    tools: ToolCallStats | None,
    first: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    situation, context = l4_situation(L4_8255, goal)
    diagnosis = dataclasses.replace(
        situation.diagnosis, primary="kv_capacity", secondary=("decode_bandwidth",)
    )
    situation = _with_tools(dataclasses.replace(situation, diagnosis=diagnosis), tools)
    _offering(monkeypatch, ("more-kv-memory", "try_first", None),
              ("fp8-weights", "try_first", None), ("fp8-kv-decode", "try_first", None))  # fmt: skip
    _predicting(monkeypatch, **{"fp8-weights": _gain(-0.3)})
    offered = _l4_offers(situation, context, tmp_path)
    assert _try_first(situation, offered)[0]["key"] == first


FOUR_STACK = ("fp8-weights", "no-media-encoders", "kv-sharing-fast-prefill", "step-budget-down")
GEMMA_CHAT = WorkloadFacts(
    prompts=(), output_tokens=384, context_limit=8192, model_type="gemma4",
    load="96 concurrent clients", max_inflight=96, media_encoders=True,
)  # fmt: skip
TRACK_LIMITS: dict[str, dict[str, float]] = {
    "ttft": {"tpot_p95_limit_ms": 164.431, "max_req_s_loss": 0.10},
    "tpot": {"ttft_p95_limit_ms": 1379.55, "max_req_s_loss": 0.10},
    "throughput": {"ttft_p95_limit_ms": 1331.67, "tpot_p95_limit_ms": 162.013},
}


def _gemma(
    goal: Goal,
    tmp_path: Path,
    *verdicts: tuple[str, str],
    blocks: bool = False,
    phase_two: bool = False,
) -> tuple[list[Suggestion], list[dict[str, Any]], int]:
    """The offers (the phase-two ones when ``phase_two``) and the Try-first blocks for the E4B
    run under ``goal``, with each (command, reason) rejected as the driver rejects a step, and
    with the run's prompt blocks when ``blocks``; and the run's KV bytes."""
    project = tmp_path / REPLAYED / goal
    shutil.copytree(GEMMA_RUN, project / "runs" / GEMMA_RUN.name, dirs_exist_ok=True)
    anatomy = gemma_e4b_anatomy()
    rejected = tuple(
        Rejection(run=f"r{i}", engine_args=tuple(args), change=tuple(args),
                  made_from=GEMMA_RUN.name, reason=reason, at="")
        for i, (command, reason) in enumerate(verdicts)
        for args in [sorted(_args(command))]
    )  # fmt: skip
    context = AnalysisContext(
        goal=goal, anatomy=anatomy, linear_weight_bytes=(freed_bytes(anatomy) or {})["quantizable"],
        answer_options=options_from(ENGINE_SUPPORT, 8.9), project=project,
        new_run=project / "runs" / GEMMA_RUN.name, rejected=rejected, **TRACK_LIMITS[goal],
    )  # fmt: skip
    if blocks:
        rows = [json.loads(line) for line in (GEMMA_RUN / "prompt_blocks.jsonl").open()]
        context.prompt_blocks = {i: (r["tokens"], tuple(r["blocks"])) for i, r in enumerate(rows)}
    situation = dataclasses.replace(
        recorded_situation(GEMMA_RUN, GEMMA_CHAT),
        measurement=replay_measurement(GEMMA_RUN, anatomy),
    )
    offered = _l4_offers(situation, context, project)
    kv = round(json.loads((GEMMA_RUN / "metrics.json").read_text())["kv_available_gib"] * 2**30)
    shown = offered.phase_two if phase_two else offered.suggestions
    return list(shown), _try_first(situation, offered), kv


def _args(command: str) -> set[str]:
    words = shlex.split(command)
    return {words[i + 1] for i, word in enumerate(words[:-1]) if word == "--engine-arg"}


def test_the_four_stack_changes_act_on_different_things_for_every_goal() -> None:
    for goal, (one, two) in itertools.product(("ttft", "tpot", "throughput"),
                                              itertools.combinations(FOUR_STACK, 2)):  # fmt: skip
        assert independent(one, two, goal), (goal, one, two)


def _size(block: dict[str, Any]) -> int:
    return next(int(a.split("=")[1]) for a in _args(block["command"])
                if a.startswith("kv-cache-memory-bytes="))  # fmt: skip


STACKABLE = tuple(
    (name, "try_first", None) for name in ("fp8-weights", "no-media-encoders", "step-budget-down")
)
LOSSLESS = ["no-media-encoders", "step-budget-down"]
STACK = ["fp8-weights", "step-budget-down", "no-media-encoders"]


def _stackable(
    monkeypatch: pytest.MonkeyPatch,
    goal: Goal,
    tmp_path: Path,
    *verdicts: tuple[str, str],
) -> tuple[list[Suggestion], list[dict[str, Any]], int]:
    _offering(monkeypatch, *STACKABLE)
    _predicting(monkeypatch, **{"fp8-weights": _gain(-0.2)})
    return _gemma(goal, tmp_path, *verdicts)


def test_the_predicted_stack_leads_with_its_anchor_and_a_sized_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, shown, kv = _stackable(monkeypatch, "ttft", tmp_path)

    stack = shown[0]
    assert stack["members"] == STACK and _size(stack) > kv
    assert [b["members"] for b in shown if b.get("members")] == [STACK]


def _members_first(shown: list[dict[str, Any]]) -> list[str] | None:
    return next((b["members"] for b in shown if b.get("members")), None)


@pytest.mark.parametrize(
    ("reason", "first"),
    [
        ("judge: timeout", LOSSLESS),
        ("tensward: does not recommend keeping this run", LOSSLESS),
        ("server did not start: CUDA out of memory", STACK[:1] + STACK[2:]),
    ],
)
def test_after_a_verdict_on_the_stack_the_next_report_leads_with_what_is_left(
    reason: str, first: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, before, _ = _stackable(monkeypatch, "ttft", tmp_path / "before")

    offers, shown, _ = _stackable(monkeypatch, "ttft", tmp_path, (before[0]["command"], reason))

    assert _members_first(shown) == first
    assert all(_args(b["command"] or "") != _args(before[0]["command"]) for b in shown)
    if first == LOSSLESS:
        bundle = next(s for s in offers if [m.name for m in s.members] == first)
        assert "Bundled: none of these changes can change the answers" in bundle.reason
        assert bundle.command is not None and bundle.command.endswith("--window full")


def test_a_lossless_remainder_the_judge_also_rejected_ends_the_stack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, before, _ = _stackable(monkeypatch, "ttft", tmp_path / "before")
    stack = (before[0]["command"], "judge: worse")
    _, remainder, _ = _stackable(monkeypatch, "ttft", tmp_path / "remainder", stack)
    lossless = next(b for b in remainder if b.get("members"))
    assert lossless["members"] == LOSSLESS

    _, shown, _ = _stackable(
        monkeypatch, "ttft", tmp_path, stack, (lossless["command"], "judge: timeout")
    )

    assert all(b.get("members") != lossless["members"] for b in shown)


def test_a_stack_stopped_at_start_up_loses_its_newest_member(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, before, _ = _stackable(monkeypatch, "ttft", tmp_path / "before")
    stopped = tmp_path / REPLAYED / "ttft" / "runs" / "20261016T180000Z-0000"
    stopped.mkdir(parents=True)
    guard = {"engine_args": sorted(_args(before[0]["command"])), "reason": "the KV pool is small"}
    (stopped / "window.json").write_text(json.dumps({"startup_guard": guard}))

    _, shown, _ = _stackable(monkeypatch, "ttft", tmp_path)

    assert shown[0]["members"] == STACK[:1] + STACK[2:]


def _spent_phase_two(goal: Goal, tmp_path: Path) -> list[Suggestion]:
    """Lose the first Try-first command again and again until Try first has nothing to run; the
    phase-two offers then."""
    verdicts: list[tuple[str, str]] = []
    for turn in range(14):
        offers, shown, _ = _gemma(goal, tmp_path / str(turn), *verdicts, phase_two=True)
        runnable = [b for b in shown if b["command"]]
        if not runnable:
            return offers
        verdicts.append((runnable[0]["command"], "worse"))
    raise AssertionError("Try first was never spent")


@pytest.mark.parametrize(
    ("goal", "phase_two"),
    [("tpot", {"no-media-encoders", "step-budget-down"}), ("throughput", {"step-budget-down"})],
)
def test_once_try_first_is_spent_the_lossless_neutral_offers_stay_in_phase_two(
    goal: Goal, phase_two: set[str], tmp_path: Path
) -> None:
    found = _spent_phase_two(goal, tmp_path)

    assert {s.entry.name for s in found} == phase_two
    assert not any(s.entry.quality_risk for s in found)


def _shown(*offers: Suggestion, goal: Goal = "tpot") -> Suggested:
    return Suggested(suggestions=offers, not_applicable=(), blocked={}, goal=goal)


def _offer(name: str, tier: str = "could_help", **fields: Any) -> Suggestion:
    entry = ENTRIES[name]
    fields = {"command": f"tensward analyse --engine-arg {name}", **fields}
    return Suggestion(
        entry=entry, reason=f"{name} fires", tier=tier,  # type: ignore[arg-type]
        addresses=frozenset(entry.addresses), **fields,
    )  # fmt: skip


def test_final_offers_and_reverts_keep_their_own_numeric_estimates(tmp_path: Path) -> None:
    from tensward.history import _revert

    for i, (kind, ready) in enumerate(
        [("cache_hit", 120), ("cache_hit", 160), ("new_set_bf16", 300), ("new_set_bf16", 400)]
    ):
        run = tmp_path / "runs" / f"r{i}"
        run.mkdir(parents=True)
        (run / "window.json").write_text(
            json.dumps({"startup": {"start_class": kind, "ready_s": ready}})
        )
    situation, context = _gemma_cell("tpot")
    context = dataclasses.replace(
        context, project=tmp_path, launched_s=0.0, load_s=100.0, measured_s=160.0
    )
    offers = tuple(
        _offer(name, "try_first", settings=ENTRIES[name].apply(situation))
        for name in ("stream-interval", "kv-sharing-fast-prefill")
    )
    with active(context):
        revert = _revert(frozenset(), tmp_path, "go back")
        shown = dataclasses.replace(_shown(*offers), memory=Memory((), revert, frozenset(), None))
        changed = _last_pass(shown, _shown(), situation, VLLM)
        blocks = _try_first(situation, changed)
    assert {b["key"]: (b["test_seconds"], b["start_seconds"]) for b in blocks} == {
        revert.entry.name: (200.0, 140.0),
        "stream-interval": (200.0, 140.0),
        "kv-sharing-fast-prefill": (410.0, 350.0),
    }
    with active(dataclasses.replace(context, measured_s=None)):
        partial = _last_pass(changed, _shown(), situation, VLLM)
        blocks = _try_first(situation, partial)
    assert {b["key"]: (b["test_seconds"], b["start_seconds"]) for b in blocks} == {
        revert.entry.name: (None, 140.0),
        "stream-interval": (None, 140.0),
        "kv-sharing-fast-prefill": (None, 350.0),
    }


def test_only_a_lossless_offer_with_no_cost_against_the_goal_is_promoted_when_spent() -> None:
    situation, context = _gemma_cell("tpot")
    skipped = [
        _offer("lower-concurrency", demoted="in a closed loop, 48 of 96 clients wait for a turn"),
        _offer("fp8-weights"),
        _offer("ngram-speculation", tail_coverage=0.0),
        _offer("no-media-encoders", past_limit=True),
        _offer("step-budget-down"),
        _offer("async-scheduling", command=None),
    ]
    offers = [_offer("stable-prompt-prefix", "try_first", command=None), *skipped,
              _offer("kv-sharing-fast-prefill")]  # fmt: skip
    before = _shown(_offer("step-budget-down", "try_first"))
    shown = _shown(*offers)
    with active(context):
        moved = _spent(offers, shown, before, situation)
        busy = [_offer("stream-interval", "try_first"), *offers]
        unchanged = _spent(busy, shown, before, situation)

    assert [(s.entry.name, s.tier) for s in moved[:2]] == [
        ("kv-sharing-fast-prefill", "try_first"),
        ("stable-prompt-prefix", "try_first"),
    ]
    assert moved[0].reason.endswith(SPENT)
    assert [s.tier for s in moved[2:]] == ["could_help"] * len(skipped)
    assert unchanged == busy


@pytest.mark.parametrize("promote", [False, True])
def test_long_starts_lead_after_promotion_and_only_promotion_clears_the_stop(promote: bool) -> None:
    situation, context = _gemma_cell("tpot")
    tier = "could_help" if promote else "try_first"
    offers = tuple(
        dataclasses.replace(
            _offer(name, tier, settings=ENTRIES[name].apply(situation),
                   predicted=Prediction(tpot=-gain, req_s=0.0, text="")),
            addresses=frozenset({"gpu_saturated"}),
        )
        for name, gain in (("stream-interval", 0.20), ("kv-sharing-fast-prefill", 0.10))
    )  # fmt: skip
    shown = dataclasses.replace(
        _shown(*offers),
        stop="No runnable step.",
        plan=({"change": "stream-interval"},),
        plan_lines=("An old plan.",),
    )
    with active(context):
        changed = _last_pass(shown, _shown(), situation, VLLM)

    assert [b["key"] for b in _try_first(situation, changed)] == [
        "kv-sharing-fast-prefill",
        "stream-interval",
    ]
    assert changed.by_list is promote
    assert changed.stop == (None if promote else "No runnable step.")
    assert all(s.reason.endswith(SPENT) is promote for s in changed.suggestions)
    assert changed.plan == changed.plan_lines == ()


def test_judged_first_order_discards_the_plan_for_a_safe_step_that_now_follows() -> None:
    situation, context = _gemma_cell("tpot")
    offers = tuple(
        dataclasses.replace(_offer(name, "try_first"), addresses=frozenset({"gpu_saturated"}))
        for name in ("stream-interval", "fp8-weights")
    )
    shown = dataclasses.replace(
        _shown(*offers), plan=({"change": "stream-interval"},), plan_lines=("An old plan.",)
    )
    with active(context):
        changed = _last_pass(shown, shown, situation, VLLM)

    blocks = _try_first(situation, changed)
    assert [b["key"] for b in blocks] == ["fp8-weights", "stream-interval"]
    assert first_try(situation.diagnosis, changed) is changed.suggestions[0]
    assert changed.plan == changed.plan_lines == ()
    assert not any("plan" in b for b in blocks)


@pytest.mark.parametrize("reverting", [False, True])
def test_the_displayed_first_step_keeps_reverts_and_failed_gates_ahead_of_judged_steps(
    reverting: bool,
) -> None:
    situation, _ = _gemma_cell("tpot")
    diagnosis = dataclasses.replace(
        situation.diagnosis,
        gates=(Finding("fit", "critical", "Failed requests."),),
        primary="decode_bandwidth",
        secondary=(),
    )
    judged = dataclasses.replace(
        _offer("fp8-weights", "try_first"), addresses=frozenset({"decode_bandwidth"})
    )
    gate = dataclasses.replace(
        _offer("lower-concurrency", "try_first"), addresses=frozenset({"fit"})
    )
    revert = _offer("step-budget-down", "try_first") if reverting else None
    shown = dataclasses.replace(
        _shown(judged, gate),
        memory=Memory(lines=(), revert=revert, tried=frozenset(), best=None),
        plan_lines=("After the first step.",),
    )
    top = revert if reverting else gate
    assert first_try(diagnosis, shown) is top
    blocks = _try_first(dataclasses.replace(situation, diagnosis=diagnosis), shown)
    assert top is not None and blocks[0]["key"] == top.entry.name
    assert [b["key"] for b in blocks if "plan" in b] == [top.entry.name]


@pytest.mark.parametrize(("worse", "demoted"), [(0.03, False), (0.05, True)])
def test_an_offer_predicted_to_worsen_the_goal_beyond_the_band_goes_to_could_help(
    worse: float, demoted: bool
) -> None:
    offer = _offer("fp8-weights", "try_first", predicted=Prediction(tpot=worse, req_s=0.0, text=""))
    kept = _without_worsening(_shown(offer))[0]

    assert (kept.tier == "could_help") is demoted
    assert (kept.demoted == "predicted to worsen TPOT by 5%") is demoted


def test_a_run_short_of_kv_space_gets_no_try_first_that_the_prediction_worsens(
    tmp_path: Path,
) -> None:
    situation, context = granite_situation("tpot")
    offered = _l4_offers(situation, context, tmp_path)

    worse = next(s for s in offered.suggestions if s.entry.name == "fp8-kv-decode")
    assert worse.tier == "could_help" and [m.name for m in worse.members] == [
        "fp8-kv-decode", "fp8-weights"
    ]  # fmt: skip
    assert worse.demoted is not None and worse.demoted.startswith("predicted to worsen TPOT by ")
    assert worse.cost is not None and worse.cost.startswith(worse.demoted)
    assert [b["key"] for b in _try_first(situation, offered)][0].startswith("fp8-weights+")


def test_among_near_equal_offers_the_cheaper_start_goes_first_and_a_gap_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    situation, context = _gemma_cell("tpot")
    lossless = dataclasses.replace(ENTRIES["kv-sharing-fast-prefill"], changes_answers=False)
    monkeypatch.setitem(ENTRIES, "kv-sharing-fast-prefill", lossless)

    def placed(name: str, tpot: float, *members: str) -> Suggestion:
        settings = situation.settings
        for entry in [ENTRIES[m] for m in members] or [ENTRIES[name]]:
            settings = entry.apply(dataclasses.replace(situation, settings=settings))
        gain = Prediction(tpot=-tpot, req_s=0.0, text="")
        return _offer(name, "try_first", settings=settings, predicted=gain,
                      members=tuple(ENTRIES[m] for m in members))  # fmt: skip

    def order(*offers: Suggestion) -> list[str]:
        shown = _shown(*offers)
        with active(context):
            return [
                s.entry.name + ("+" if s.members else "")
                for s in _by_start_cost(list(offers), shown, situation, VLLM)
            ]

    both = ["stream-interval", "kv-sharing-fast-prefill"]
    assert order(placed(both[1], 0.10), placed(both[0], 0.10)) == both
    assert order(placed(both[1], 0.10), placed(both[0], 0.105)) == both
    assert order(placed(both[1], 0.20), placed(both[0], 0.10)) == both[::-1]
    stack = placed("kv-sharing-fast-prefill", 0.10, "kv-sharing-fast-prefill", "stream-interval")
    assert order(stack, placed(both[0], 0.10)) == ["kv-sharing-fast-prefill+", "stream-interval"]


@pytest.mark.parametrize(
    ("reason", "kind"),
    [
        ("judge: timeout", "transport"),
        ("judge: judge_failed", "transport"),
        ("judge: no output from the quality check", "transport"),
        ("judge: no_calibration", "transport"),
        ("judge: excess worse 0.05 > 0.03", "verdict"),
        ("judge: too_few", "dropped"),
        ("judge: budget_used", "dropped"),
        ("goal: TPOT p95 41 worse than the best adopted run's 40", "other"),
        (None, "other"),
    ],
)
def test_kind_of(reason: str | None, kind: str) -> None:
    assert kind_of(reason) == kind
