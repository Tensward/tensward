"""The before/after comparison: agreement against the baseline's own noise, and equality."""

import asyncio
import json
from dataclasses import replace

import pytest

from tensward import reasoning_check
from tensward.checks import engine_output_rate
from tensward.compare import Answer, compare_answers, reasoning_counts, repeats_itself
from tensward.measurement import Measurement, Window
from tensward.report.build import _headline_parts
from tensward.report.compare import render_changes, tpot_note
from tensward.runs import Comparison, RunMetrics, RunRecord
from tensward.slo import NO_LIMITS

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
    outcome = compare_answers(baseline, candidate, {}, json_answers=frozenset())
    assert [p.prompt_id for p in outcome.prompts if p.changed] == changed
    assert outcome.verdict.startswith(verdict)
    assert outcome.equal is equal
    expected = (
        None
        if verdict.startswith(("No noise", "Answers were empty"))
        else verdict.startswith("Answers changed")
    )
    assert outcome.changed is expected


CITIES = ["Oslo", "Lima", "Pune", "Kyiv", "Doha", "Baku", "Riga", "Kobe", "Nice", "Graz"]
PAGES = ["install", "configure", "serve", "analyse", "compare", "keep", "set", "report", "history"]
RECORDS = json.dumps(
    [{"id": i, "name": f"user{i}", "city": CITIES[i % 10], "on": i % 2 == 0} for i in range(30)],
    indent=2,
)
STEPS = "\n".join(
    f"{i}. Step {i}: open the settings page and check option {chr(64 + i)}." for i in range(1, 15)
)
LIST = "\n".join(f'<li class="item"><a href="/docs/{page}">{page}</a></li>' for page in PAGES)
CALLS = "\n".join(
    f'<tool_call>\n{{"name": "get_weather", "arguments": {{"city": "{city}", "unit": "celsius"}}}}'
    "\n</tool_call>"
    for city in CITIES[:3]
)


@pytest.mark.parametrize(
    ("text", "repeats"),
    [("be " * 200, True), (RECORDS, False), (STEPS, False), (f"<ul>\n{LIST}\n</ul>", False),
     (CALLS, False)],
)  # fmt: skip
def test_a_loop_repeats_itself_and_structured_answers_do_not(text: str, repeats: bool) -> None:
    assert repeats_itself(text) is repeats


def test_tool_calls_are_compared_per_prompt_by_name_and_arguments() -> None:
    other = Answer("", (("file_invoice", '{"total":972}'),), "tool_calls")
    outcome = compare_answers(
        {"p": [CALL, CALL], "q": [CALL, CALL], "r": [A, A]},
        {"p": [CALL2, CALL2], "q": [other, other], "r": [A, A]},
        {},
        json_answers=frozenset(),
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
    headline = _headline_parts(replace(measured, engine_output_throughput=50.0), NO_LIMITS)
    assert ("output", "output 50.0 tok/s (the engine's count, see Checks)") in headline

    outcome = compare_answers({"p": [A]}, {"p": [A]}, {}, json_answers=frozenset())
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
    outcome = compare_answers({"p": [A]}, {"p": [A]}, {}, json_answers=frozenset())
    speed = {"tpot_p95_ms": (52.0, 593.0)}
    comparison = Comparison("a", "b", outcome, speed, 0.0, (), False)
    tpot = tpot_note(baseline, candidate)
    lines = render_changes(comparison, [], ("", ""), [], tpot=tpot)
    assert next(line for line in lines if "TPOT p95" in line).endswith(note)
    for after in (52.01, 52.1):
        flat = replace(comparison, speed={"tpot_p95_ms": (52.0, after)})
        unchanged = render_changes(flat, [], ("", ""), [], tpot=tpot)
        assert ";" not in next(line for line in unchanged if "TPOT p95" in line)


def test_only_a_reasoning_answer_cut_by_the_output_limit_is_said_to_run_out_of_tokens() -> None:
    cut = Answer("", (), "length", True)
    stopped = Answer("", (), "stop", True)
    baseline = {"a": [A], "b": [A], "c": [A], "d": [A]}
    candidate = {"a": [A], "b": [cut], "c": [stopped], "d": []}
    outcome = compare_answers(baseline, candidate, {}, json_answers=frozenset())
    assert [p.prompt_id for p in outcome.prompts if p.ran_out] == ["b"]
    assert "(1 ran out of output tokens while thinking, 2 got no answer)" in outcome.verdict


def test_greedy_answers_cut_while_thinking_are_compared_on_their_reasoning() -> None:
    def cut(reasoning: str) -> Answer:
        return Answer("", (), "length", True, reasoning, 64)

    baseline = {"a": [cut("plan the hole pattern then the bore")], "b": [cut("one two three")]}
    candidate = {"a": [cut("plan the hole pattern then the bore and")], "b": [cut("one too")]}
    outcome = compare_answers(baseline, candidate, {}, json_answers=frozenset(), greedy=True)
    assert sorted((p.prompt_id, p.identical, p.first_difference) for p in outcome.prompts) == [
        ("a", True, None),
        ("b", False, 5),
    ]
    assert reasoning_counts(outcome.prompts) == {
        "prompts": 2, "identical": 1, "tokens": [64, 64], "diverge": [5]
    }  # fmt: skip
    assert outcome.changed is None and outcome.verdict.startswith("Your call: 1 of 2 prompts")
    sampled = compare_answers(baseline, candidate, {}, json_answers=frozenset())
    assert sampled.thought == 2 and not sampled.prompts


def _side(*prompts: tuple[int, int, int]) -> reasoning_check.Side:
    """Probes per prompt: (count, near ties the text did not take, tokens outside the top)."""
    probes = {}
    for index, (count, near, far) in enumerate(prompts):
        rows = []
        for i in range(count):
            logprob = None if i < far else -1.0 if i < far + near else -0.1
            top = 0 if logprob == -0.1 else 1
            rows.append(reasoning_check.Probe(i, i, 0, "a", logprob, top, "b", -0.1))
        probes[f"p{index}"] = rows
    return reasoning_check.Side(probes)


@pytest.mark.parametrize(
    ("base", "step", "broken", "verdict"),
    [
        ((255, 0, 0), (768, 0, 0), False, "not_run"),  # too few probes
        ((768, 31, 0), (768, 0, 0), False, "not_run"),  # 96%: the self-test fails
        ((768, 0, 0), (768, 7, 0), False, "same"),  # 0.9 pp, inside the 1 pp floor
        ((768, 0, 0), (768, 8, 0), False, "your_call"),
        ((768, 15, 0), (768, 27, 0), False, "same"),  # base 98%: the margin scales to 1.7 pp
        ((768, 0, 0), (768, 0, 3), False, "same"),
        ((768, 0, 0), (768, 0, 4), False, "your_call"),  # FAR beyond its 0.5 pp floor
        ((768, 0, 0), (768, 39, 0), False, "changed"),  # 5.1 pp
        ((768, 0, 0), (768, 0, 0), True, "changed"),  # a broken answer
    ],
)
def test_the_reasoning_check_rules_hold_at_their_boundaries(base, step, broken, verdict) -> None:
    found, _ = reasoning_check.verdict(_side(base), _side(step), broken_answers=broken)
    assert found == verdict


def test_one_prompt_that_departs_is_changed_though_the_whole_is_close() -> None:
    step = _side(*[(64, 17, 0)], *[(64, 0, 0)] * 11)  # 73% on one prompt, 2.2 pp overall
    assert reasoning_check.verdict(_side(*[(64, 0, 0)] * 12), step)[0] == "changed"


def test_the_whitespace_the_model_writes_first_is_kept_on_both_sides() -> None:
    """The reasoning parser strips what follows <think>; the replay puts it back by asking the
    current setup, and probes position 0 once for the base side."""
    seen: list[list[int]] = []

    class Engine:
        async def text_pieces(self, client, url, text):
            return [(10 + i, word.encode()) for i, word in enumerate(text.split())]

        async def next_token_logprobs(self, client, url, prefix, top, timeout_s):
            seen.append(list(prefix))
            after = len(prefix) - 2
            if after < 2:
                return [(900 + after, "\n" if after else " ", -0.1)]
            return [(10 + after - 2, "w", -0.1)]

    words = " ".join(f"w{i}" for i in range(300))
    prompt = reasoning_check.Prompt("p", [1, 2], words, words)
    base, step, cached = asyncio.run(
        reasoning_check.replay(Engine(), None, "", [prompt], None, 1.0)
    )
    assert cached["p"]["lead"] == [900, 901]
    assert all(prefix[:4] == [1, 2, 900, 901] for prefix in seen[2:])
    assert seen.count([1, 2, 900, 901]) == 2  # the last whitespace probe, then the step's 0
    assert base.share() == step.share() == 1.0


@pytest.mark.parametrize(
    ("base", "step", "result"),
    [
        ([1, 1, 1], [2, 1, 1], "explained"),  # parts at a near tie, then the setup's own pick
        ([1, 1, 1], [2, 3, 1], "changed"),  # a token far below the pick after the parting
        ([1, 1, 1], [4, 1, 1], "changed"),  # outside the top 20 where they part
        ([3, 1, 1], [2, 1, 1], "unchecked"),  # the earlier run's own copy fails the guard
        ([1, 1, 1], [1, 1], "explained"),  # cut short: nothing written past the cut
        ([1, 1], [1, 1], "unchecked"),  # rebuilt the same: nothing to tell them apart by
    ],
)
def test_the_answer_check_explains_only_tokens_near_the_current_setups_pick(
    base: list[int], step: list[int], result: str
) -> None:
    class Engine:
        async def next_token_logprobs(self, client, url, prefix, top, timeout_s):
            return [(1, "a", -0.1), (2, "b", -0.1 - 0.6), (3, "c", -3.0)]

    probing = reasoning_check._Probing(Engine(), None, "", 1.0, budget=10)
    found = asyncio.run(reasoning_check.check_answer(
        probing, "p", reasoning_check.Written([9, *base], 1, {}),
        reasoning_check.Written([9, *step], 1, {}),
    ))  # fmt: skip
    assert found.result == result
    assert probing.left is not None and 10 - probing.left <= 5  # one probe where they part


def test_the_answer_check_stops_at_the_first_changed_answer(monkeypatch) -> None:
    results = ["explained", "changed", "explained", "explained"]
    rows = [reasoning_check.Checked(f"p{i}", r, None, 0, 1) for i, r in enumerate(results)]
    budgets: list[int | None] = []

    async def check_one(probing, body, head, end, differs, answer):
        budgets.append(probing.left)
        return rows[int(differs.prompt_id[1:])]

    class Engine:
        async def end_of_turn(self, client, url):
            return 0

        async def prompt_ids(self, client, url, body):
            return [1]

        async def text_pieces(self, client, url, text):
            return [(1, b"a")]

    async def body(engine, workload, index):
        return {}

    monkeypatch.setattr(reasoning_check, "_check_one", check_one)
    monkeypatch.setattr(reasoning_check, "_body", body)
    answer = Answer("a", None, "stop")
    answers = [(i, reasoning_check.Differs(f"p{i}", (answer,), (answer,))) for i in range(4)]
    workload = type("Workload", (), {"request_timeout_s": 1.0})()
    found = asyncio.run(reasoning_check.check_answers(
        Engine(), None, "", workload, answers, {f"p{i}": 1 for i in range(4)}
    ))  # fmt: skip
    assert len(budgets) == 2 and budgets[0] == reasoning_check.ANSWER_PROBES == 6144
    assert found["verdict"] == reasoning_check._answer_document(rows)["verdict"] == "changed"
    assert found["stopped"] == {"at": 2, "of": 4, "not_replayed": ["p2", "p3"]}
    said = reasoning_check.answer_lines({"answer_check": found})[0]
    assert "changed at prompt 2 of the 4 that differ; the rest not replayed" in said


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("found", ["changed", "not_run"])
def test_a_step_too_big_to_replay_only_looks_for_a_changed_answer(
    monkeypatch, found, mixed
) -> None:
    budgets: list[int | None] = []

    async def check_one(probing, body, head, end, differs, answer):
        budgets.append(probing.left)
        if found == "changed" and differs.prompt_id == "p1":
            return reasoning_check.Checked("p1", "changed", None, 0, 1)
        for _ in range(255):
            await probing.top([])
        return reasoning_check.Checked(differs.prompt_id, "explained", None, 0, 255)

    class Engine:
        async def end_of_turn(self, client, url):
            return 0

        async def prompt_ids(self, client, url, body):
            return [1]

        async def text_pieces(self, client, url, text):
            return [(ord(text[0]), b"x"), *((i, b"x") for i in range(1000))]

        async def next_token_logprobs(self, client, url, prefix, top, timeout_s):
            return [(1, "x", 0.0)]

    async def body(engine, workload, index):
        return {}

    monkeypatch.setattr(reasoning_check, "_check_one", check_one)
    monkeypatch.setattr(reasoning_check, "_body", body)
    count = 100 if mixed else 60
    base, step = Answer("a", None, "length"), Answer("b", None, "length")
    tool = Answer("b", (("f", "{}"),), "stop")
    answers = [
        (i, reasoning_check.Differs(f"p{i}", (base,), (tool if mixed and i % 2 else step,)))
        for i in range(count)
    ]
    workload = type("Workload", (), {"request_timeout_s": 1.0})()
    try:
        checked = asyncio.run(reasoning_check.check_answers(
            Engine(), None, "", workload, answers, {f"p{i}": 1 for i in range(count)}
        ))  # fmt: skip
    except reasoning_check.CheckFailed as error:
        checked = {"verdict": "not_run", "reason": str(error)}
    assert budgets[0] == reasoning_check.ANSWER_HUNT
    assert checked["verdict"] == found
    if found == "changed":
        assert checked["stopped"]["at"] == 2
    else:
        assert len(budgets) == 5  # 1024 probes, 255 an answer
        needed = 12800 if mixed else 15300  # a tool-route answer costs one probe
        assert checked["reason"] == (
            f"too many answers differ to check in time: about {needed} probes needed (budget "
            "6144); searched the first 1024 for a change, none found; judged as before"
        )


def test_the_fork_check_is_not_same_while_differences_are_left_unchecked() -> None:
    rows = [
        reasoning_check.Checked("p0", "explained", None, 0, 1),
        reasoning_check.Checked("p1", "unchecked", reasoning_check.THOUGHT, 0, 0),
        reasoning_check.Checked("p2", "unchecked", reasoning_check.THOUGHT, 0, 0),
    ]
    found = reasoning_check._answer_document(rows)
    assert found["verdict"] == "partial"
    said = reasoning_check.answer_lines({"answer_check": found})[0]
    assert "1 of 3 differing answers explained, 2 not checked" in said
    assert reasoning_check._answer_document(rows[:1])["verdict"] == "same"
    same = reasoning_check.Checked("p3", "unchecked", reasoning_check.SAME, 0, 0)
    assert reasoning_check._answer_document([*rows[:1], same])["verdict"] == "same"
