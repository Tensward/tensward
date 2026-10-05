"""The before/after comparison: agreement against the baseline's own noise, and equality."""

from dataclasses import replace

import pytest

from tensward.measurement import Measurement, Window
from tensward.quality import Answer, Comparison, compare_answers, render_changes
from tensward.report import engine_output_rate, render_headline

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
    headline = render_headline(replace(measured, engine_output_throughput=50.0), "x", [])
    assert any("output 50.0 tok/s" in line for line in headline)

    outcome = compare_answers({"p": [A]}, {"p": [A]}, {}, structured=False)
    comparison = Comparison(
        "a", "b", outcome, {"output_throughput": (50.0, 60.0)}, 0.0, (), False, (False, True)
    )
    lines = render_changes(comparison, [], ("", ""), [])
    assert any("this run is the engine's own count" in line for line in lines)
