"""The before/after comparison: agreement against the baseline's own noise, and equality."""

import pytest

from tensward.quality import Answer, compare_answers

A = Answer("the total is 971 dollars due 12 october", (), "stop")
A2 = Answer("The total is 971 dollars, due 12 October.", (), "stop")  # same words
B = Answer("i cannot read the invoice", (), "stop")
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
        ({"p": [A]}, {"p": [B]}, [], "No noise floor", False),  # every prompt once
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
