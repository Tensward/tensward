"""Try first ordered by predicted TTFT, and two rules from measured runs: a
runnable fallback when Try first holds no command, and KV precision before weight precision on
tool-call workloads."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from test_replays import L4_8255, VLLM, _l4_offers, l4_situation

from tensward.context import Goal
from tensward.history import PastRun
from tensward.measurement import Measurement
from tensward.playbook import Situation, Suggestion
from tensward.predict import Prediction
from tensward.report.build import next_steps_section
from tensward.report.model import Report, to_json
from tensward.suggest import Suggested, by_goal
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
    _predicting(monkeypatch, **{"fp8-weights": _gain(ttft)})
    offered = _l4_offers(situation, context, tmp_path)
    tier = next(s.tier for s in offered.suggestions if s.entry.name == "fp8-weights")
    if ttft == -0.22:
        assert _try_first(situation, offered)[0]["key"] == "fp8-weights"
        assert tier == "try_first"
    else:
        assert tier == "could_help"


def test_a_change_that_worsens_the_goal_is_left_out_whatever_it_is_predicted() -> None:
    lower = Suggestion(
        entry=ENTRIES["lower-concurrency"],
        reason="fires",
        tier="try_first",
        predicted=_gain(-0.4),
    )
    kept, left_out = by_goal([lower], "ttft", Measurement(100, 0), None)
    assert kept == [] and left_out == ("lower-concurrency",)


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


A5_TRY_FIRST = ("quantized-checkpoint", "try_first", None)


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


@pytest.mark.parametrize(("tools", "first"), [(TOOLS, "fp8-kv-decode"), (None, "fp8-weights")])
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
