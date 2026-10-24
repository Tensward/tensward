"""The condition language: its grammar, three-valued evaluation, the build-time checks and the
index anchors."""

from __future__ import annotations

import pytest

from tensward.facts import Facts
from tensward.kb_condition import (
    And,
    Compare,
    ConditionError,
    FactRef,
    Known,
    ListLit,
    Lit,
    Not,
    Or,
    anchors,
    check,
    evaluate,
    explain,
    from_json,
    parse,
    to_json,
)

A, B, C = FactRef("workload.tools"), FactRef("workload.structured"), FactRef("workload.media")


@pytest.mark.parametrize(
    ("text", "tree"),
    [
        ("not workload.tools and workload.structured or workload.media", Or((And((Not(A), B)), C))),
        ("workload.tools and (workload.structured or workload.media)", And((A, Or((B, C))))),
        ("not not workload.tools", Not(Not(A))),
        (
            'settings.speculation_method not in ["eagle", "eagle3"]',
            Compare(FactRef("settings.speculation_method"), "not in", ListLit(("eagle", "eagle3"))),
        ),
        ("gpu.cc >= 8.9", Compare(FactRef("gpu.cc"), ">=", Lit(8.9))),
        (
            "known(gpu.cc) and not known(anatomy.layers)",
            And((Known("gpu.cc"), Not(Known("anatomy.layers")))),
        ),
    ],
)
def test_the_grammar_binds_not_then_and_then_or(text: str, tree: object) -> None:
    node = parse(text)
    assert node == tree
    assert from_json(to_json(node)) == node


@pytest.mark.parametrize(
    ("text", "column"),
    [("goal == ", 9), ('goal == "ttft" and and', 20), ("goal ~ 1", 6), ("(goal", 6)],
)
def test_a_parse_error_names_its_column(text: str, column: int) -> None:
    with pytest.raises(ConditionError, match=f"column {column}:"):
        parse(text)


@pytest.mark.parametrize(
    ("left", "right", "both", "either"),
    [
        (True, True, True, True),
        (True, False, False, True),
        (False, False, False, False),
        (True, None, None, True),
        (False, None, False, None),
        (None, None, None, None),
    ],
)
def test_and_or_follow_kleene_logic_with_unknown(
    left: bool | None, right: bool | None, both: bool | None, either: bool | None
) -> None:
    facts = Facts({"workload.tools": left, "workload.structured": right})
    for x, y in ((A, B), (B, A)):
        assert evaluate(And((x, y)), facts) is both
        assert evaluate(Or((x, y)), facts) is either
    assert evaluate(Not(A), facts) is (None if left is None else not left)


@pytest.mark.parametrize(
    ("text", "facts", "truth"),
    [
        ('engine.version matches ">=0.30,<0.31"', {"engine.version": "0.30.0"}, True),
        ('engine.version matches ">=0.30,<0.31"', {"engine.version": "0.31.1"}, False),
        ('engine.version matches ">=0.30,<0.31"', {}, None),
        ("512 in anatomy.head_dims", {"anatomy.head_dims": [256, 512]}, True),
        ("512 not in anatomy.head_dims", {"anatomy.head_dims": [128]}, True),
        ('"kv_capacity" in diag.fired', {"diag.fired": []}, False),
        ('gpu.cc < 8.9 or goal == "ttft"', {"goal": "ttft"}, True),
        ('gpu.cc < 8.9 and goal == "ttft"', {"goal": "ttft"}, None),
        ('gpu.cc < 8.9 and goal == "ttft"', {"goal": "tpot"}, False),
        ("not known(gpu.cc)", {}, True),
        (
            "run.ttft_p95_ms > limit.ttft_p95_ms",
            {"run.ttft_p95_ms": 900.0, "limit.ttft_p95_ms": 500.0},
            True,
        ),
    ],
)
def test_conditions_evaluate_on_facts(text: str, facts: dict, truth: bool | None) -> None:
    node = parse(text)
    assert check(node) == []
    assert evaluate(node, Facts(facts)) is truth


def test_the_explanation_names_the_deciding_terms_or_the_missing_facts() -> None:
    node = parse('anatomy.kv_shared_layers > 0 and (goal == "ttft" or gpu.cc >= 8.9)')
    fired = explain(node, Facts({"anatomy.kv_shared_layers": 18, "goal": "ttft"}))
    assert [(t.fact, t.op, t.value, t.observed) for t in fired.fired_on] == [
        ("anatomy.kv_shared_layers", ">", 0, 18),
        ("goal", "==", "ttft", "ttft"),
    ]
    assert explain(node, Facts({"goal": "tpot"})).unknown == ("anatomy.kv_shared_layers", "gpu.cc")


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("model.size > 7", "model.size is not in the fact vocabulary"),
        ('gpu.cc == "ada"', "gpu.cc is float, not 'ada'"),
        ('goal == "latency"', "'latency' is not one of goal's values"),
        ('goal in ["ttft", "fast"]', "'fast' is not one of goal's values"),
        ('"fit_all" in diag.fired', "'fit_all' is not one of diag.fired's values"),
        ("workload.prompt_tokens_p50 > gpu.memory_gib", "tokens against GiB"),
        ("workload.tools > 1", "workload.tools is bool, not a number"),
        ('engine.version == "0.30.0"', "use in or matches"),
        ('engine.version matches "0.30"', "is not a version specifier"),
        ("anatomy.layers", "anatomy.layers is int, not a condition"),
    ],
)
def test_check_refuses_unknown_facts_and_mismatched_types(text: str, problem: str) -> None:
    found = check(parse(text))
    assert len(found) == 1 and problem in found[0], found


@pytest.mark.parametrize(
    ("text", "gpu_cc", "expected"),
    [
        ("gpu.cc >= 8.9", None, {"arch": ("ada", "hopper", "blackwell")}),
        ("gpu.cc >= 8.0 and gpu.cc < 9.0", None, {"arch": ("ampere", "ada")}),
        (None, ">=7.5,<8.0", {"arch": ("turing",)}),
        ('goal == "ttft" or model.family == "gemma4"', None, {}),
        ('not goal == "ttft" and "kv_capacity" in diag.fired', None, {"diag": ("kv_capacity",)}),
        (
            'model.family in ["qwen3", "gemma4"] and goal in ["tpot", "throughput"]',
            None,
            {"family": ("gemma4", "qwen3"), "goal": ("throughput", "tpot")},
        ),
        ('diag.primary == "prefill" and "queueing" in diag.fired', None, {"diag": ("prefill",)}),
    ],
)
def test_anchors_pin_only_top_level_and_terms(
    text: str | None, gpu_cc: str | None, expected: dict[str, tuple[str, ...]]
) -> None:
    found = anchors(parse(text) if text else None, engine="vllm", gpu_cc=gpu_cc)
    unpinned = {key: ("*",) for key in ("family", "arch", "diag", "goal")}
    assert found == {"engine": ("vllm",), **unpinned, **expected}
