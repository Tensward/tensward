"""The before/after comparison: agreement against the baseline's own noise, and equality."""

from dataclasses import replace

import pytest

from tensward.checks import engine_output_rate
from tensward.compare import Answer, compare_answers
from tensward.measurement import Measurement, Window
from tensward.report.build import _headline_parts
from tensward.report.compare import render_changes, tpot_note
from tensward.runs import Comparison, RunMetrics, RunRecord

A = Answer("the total is 971 dollars due 12 october", (), "stop")
A2 = Answer("The total is 971 dollars, due 12 October.", (), "stop")  # same words
B = Answer("i cannot read the invoice", (), "stop")
EMPTY = Answer("", None, "stop")
CALL = Answer("", (("file_invoice", '{"total":971,"due":"2026-10-12"}'),), "tool_calls")
CALL2 = Answer("", (("file_invoice", '{"due": "2026-10-12", "total": 971}'),), "tool_calls")


@pytest.mark.parametrize(
    ("baseline", "candidate", "changed", "verdict", "equal"),
    [
        ({"p": [A, A2]}, {"p": [A2, A]}, [], "Answers unchanged within noise", True),
        ({"p": [A, A]}, {"p": [B, B]}, ["p"], "Answers changed: 1 of 1 prompts", False),
        ({"p": [A, B]}, {"p": [B, A]}, [], "Answers unchanged within noise", True),  # noisy
        ({"p": [CALL, CALL]}, {"p": [CALL2, CALL2]}, [], "Answers unchanged within noise", True),
        ({"p": [CALL, CALL]}, {"p": [A, A]}, ["p"], "Answers changed: 1 of 1 prompts", False),
        ({"p": [A]}, {"p": [B]}, [], "No noise floor: The one prompt", False),
        ({"p": [A], "q": [A]}, {"p": [A], "q": [A]}, [], "No noise floor: Each of the 2", True),
        ({"p": [A, A]}, {"p": [EMPTY, EMPTY]}, ["p"], "Answers were empty: every one of th", None),
        ({"p": [EMPTY]}, {"p": [EMPTY]}, [], "Answers were empty: every one of the base", None),
        ({"p": [A, A], "q": [A, A]}, {"p": [A, A]}, [], "Answers changed: 1 of 2 prompts", False),
        ({"p": [A]}, {}, [], "Answers changed: 1 of 1 prompts", False),  # unanswered beats no floor
        ({"p": [A, A], "q": [A]}, {"p": [A, A], "q": [B]}, [],
         "Answers unchanged within noise (1 prompt", False),  # q ran once: not judged
        ({"p": [A] * 17}, {"p": [A] * 16 + [B]}, [], "Answers unchanged within noise", False),
        # the 17th candidate answer lies beyond the similarity cap and still breaks equality
    ],
)  # fmt: skip
def test_agreement_is_judged_against_the_baselines_own_noise(
    baseline, candidate, changed, verdict, equal
) -> None:
    outcome = compare_answers(baseline, candidate, {}, structured=False)
    assert [p.prompt_id for p in outcome.prompts if p.changed] == changed
    assert outcome.verdict.startswith(verdict)
    assert outcome.equal is equal
    expected = (
        None
        if verdict.startswith(("No noise", "Answers were empty"))
        else verdict.startswith("Answers changed")
    )
    assert outcome.changed is expected


def test_tool_calls_are_compared_per_prompt_by_name_and_arguments() -> None:
    other = Answer("", (("file_invoice", '{"total":972}'),), "tool_calls")
    outcome = compare_answers(
        {"p": [CALL, CALL], "q": [CALL, CALL], "r": [A, A]},
        {"p": [CALL2, CALL2], "q": [other, other], "r": [A, A]},
        {},
        structured=False,
    )
    assert {p.prompt_id: p.calls_identical for p in outcome.prompts} == {
        "p": True,
        "q": False,
        "r": None,
    }


def test_the_engine_rate_replaces_a_mismatched_client_rate_and_says_so() -> None:
    window = Window("steady", 0, 20_000_000_000, 20.0)
    measured = Measurement(10, 0, output_throughput=100.0, window=window)
    assert engine_output_rate(measured, 1000.0, False) == 50.0
    assert engine_output_rate(measured, 2000.0, False) is None  # the counts agree
    assert engine_output_rate(measured, 1000.0, True) is None  # engine span not the window
    short = replace(measured, window=Window("steady", 0, 5_000_000_000, 5.0))
    assert engine_output_rate(short, 250.0, False) is None  # window too short to trust
    headline = _headline_parts(replace(measured, engine_output_throughput=50.0))
    assert ("output", "output 50.0 tok/s (the engine's count, see Checks)") in headline

    outcome = compare_answers({"p": [A]}, {"p": [A]}, {}, structured=False)
    comparison = Comparison(
        "a", "b", outcome, {"output_throughput": (50.0, 60.0)}, 0.0, (), False, (False, True)
    )
    lines = render_changes(comparison, [], ("", ""), [])
    assert any("this run is the engine's own count" in line for line in lines)


@pytest.mark.parametrize(
    ("per_step", "note"),
    [
        (1400.0, "most of each step is prompt processing (about 1400 prompt tokens per step)"),
        (40.0, "more sequences share every step"),
    ],
)
def test_a_raised_cap_says_why_tpot_rose(per_step: float, note: str) -> None:
    def run(cap: int, tpot: float, **metrics: object) -> RunRecord:
        measured = RunMetrics.model_validate(
            {"succeeded": 1, "failed": 0, "tpot_p95_ms": tpot, **metrics}
        )
        settings = {"max_concurrent_requests": cap}
        return RunRecord("r", "s", 0.0, None, measured, {}, settings=settings)

    baseline = run(2, 52.0)
    candidate = run(
        32, 593.0, prompt_tokens_per_step=per_step, ceilings={"avg_running_batch": 31.0}
    )
    outcome = compare_answers({"p": [A]}, {"p": [A]}, {}, structured=False)
    speed = {"tpot_p95_ms": (52.0, 593.0)}
    comparison = Comparison("a", "b", outcome, speed, 0.0, (), False)
    tpot = tpot_note(baseline, candidate)
    lines = render_changes(comparison, [], ("", ""), [], tpot=tpot)
    assert next(line for line in lines if "TPOT p95" in line).endswith(note)
    for after in (52.01, 52.1):
        flat = replace(comparison, speed={"tpot_p95_ms": (52.0, after)})
        unchanged = render_changes(flat, [], ("", ""), [], tpot=tpot)
        assert ";" not in next(line for line in unchanged if "TPOT p95" in line)
