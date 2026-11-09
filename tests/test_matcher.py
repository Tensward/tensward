"""Precedence and conflicts between the knowledge layers, on fixture records."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest

from tensward.engines.vllm.flags import merge_json
from tensward.engines.vllm.value_rules import (
    cpu_kv_offload_size,
    long_prefill_threshold,
    per_token_head_dtype,
    spec_tiers,
    suffix_acceptance,
)
from tensward.facts import UNKNOWN, Facts
from tensward.kb_condition import parse, to_json
from tensward.kb_entries import kb_block
from tensward.kb_schema import Layer, Record, lever_key, record_from_json
from tensward.matcher import Layers, Matches, annotate_offered, match
from tensward.similarity import Axis, Similarity, place, resolve

FACTS = Facts({
    "goal": "throughput", "gpu.cc": 8.9, "gpu.arch": "ada", "engine.name": "vllm",
    "engine.version": "0.30.0", "model.family": "gemma4", "diag.primary": "kv_capacity",
    "diag.fired": ["kv_capacity"], "anatomy.kv_shared_layers": 18,
})  # fmt: skip
LEVERS = frozenset({"prefix_caching", "max_context_len"})
PLAYBOOK = frozenset({"fp8-weights", "prefix-caching", "more-kv-memory"})
FIRES = "anatomy.kv_shared_layers > 0"
NEVER = "anatomy.kv_shared_layers == 0"


def _record(
    id_: str, kind: str = "lever", *, layer: Layer = "pack", when: str = FIRES, **fields: Any
) -> Record:
    data: dict[str, Any] = {
        "id": f"test/{id_}", "rev": 1, "kind": kind, "title": id_, "status": "measured",
        "engine": "vllm", "condition": to_json(parse(when)), "validity": {}, "evidence": [],
        "confidence": {"tier": "low"}, "facts_version": 1,
    }  # fmt: skip
    if kind == "lever":
        data["change"] = {"levers": ["prefix_caching"], "sets": {"prefix_caching": True}}
        data["effect"] = {"throughput": {"direction": "improves", "basis": "prior"}}
        data["costs"] = {"text": "none", "answer_risk": "none", "changes_answers": False}
    return record_from_json({**data, **fields}, layer=layer)


def _guard(id_: str, blocks: Sequence[str], **fields: Any) -> Record:
    change = {"blocks": list(blocks), "reason": f"{id_} says no", **fields.pop("change", {})}
    return _record(id_, "guard", change=change, **fields)


def _measured(median: float, noise: float = 0.02) -> dict[str, Any]:
    metric = {"median": median, "low": median, "high": median, "noise": noise}
    return {
        "setups": 1, "pairs": 2, "quality_ok": True, "expdb_version": "test",
        "metrics": {"throughput": metric},
    }  # fmt: skip


class _Pack:
    name, version, content_digest = "pack", "2026.10.20.1", "sha256:test"

    def __init__(self, records: Sequence[Record], tombstones: Mapping[str, str] = {}) -> None:
        self.records, self.retired = records, tombstones

    def candidates(self, keys: Mapping[str, Sequence[str] | None]) -> Iterator[Record]:
        return iter(self.records)

    def tombstones(self) -> Mapping[str, str]:
        return self.retired


def _match(
    pack: Sequence[Record] = (),
    local: Sequence[Record] = (),
    tombstones: Mapping[str, str] = {},
    rules: Sequence[str] = ("vllm:fast_prefill_applies",),
    facts: Facts = FACTS,
) -> Matches:
    layers = Layers(
        playbook=PLAYBOOK,
        source=_Pack(pack, tombstones),
        source_status={"pack": "2026.10.20.1", "reason": None},
        local=tuple(local),
    )
    found = match(
        facts, layers, engine_family="vllm", engine_version="0.30.0",
        tensward_version="0.3.15", rules=rules, levers=LEVERS,
    )  # fmt: skip
    again = match(
        facts, layers, engine_family="vllm", engine_version="0.30.0",
        tensward_version="0.3.15", rules=rules, levers=LEVERS,
    )  # fmt: skip
    assert found == again
    return found


def test_guards_are_a_union_the_most_restrictive_wins_and_local_cannot_lift_a_pack_guard() -> None:
    pack_guard = _guard("memory/no-prefix-caching", ["prefix_caching=on"])
    lifted = _guard(
        "memory/no-prefix-caching", ["prefix_caching=on"], layer="local", rev=5, when=NEVER
    )
    refusal = _guard("quant/act-order", ["*"], change={"refuse": True})
    local_lever = _record("memory/prefix", layer="local", change={"lever_key": "prefix_caching=on"})
    found = _match(pack=[pack_guard, refusal], local=[lifted, local_lever])
    assert found.blocked == {
        "*": "quant/act-order says no",
        "prefix_caching=on": "memory/no-prefix-caching says no",
    }
    assert found.blocked_by("prefix_caching=on") == "quant/act-order says no"

    narrow = _guard("memory/narrow", ["prefix_caching=on"])
    wide = _guard("memory/wide", ["prefix_caching=on", "*"])
    assert _match(pack=[narrow, wide]).blocked["prefix_caching=on"] == "memory/wide says no"


def test_a_pack_tombstone_hides_only_a_playbook_entry_with_its_reason() -> None:
    found = _match(tombstones={"fp8-weights": "garbles answers", "gone-entry": "x"})
    assert found.tombstoned == {"fp8-weights": "garbles answers"}


@pytest.mark.parametrize(
    ("local_median", "shown"),
    [(0.12, "local"), (None, "promoted")],
)
def test_one_merge_key_from_two_layers_is_one_offer_its_number_by_precedence(
    local_median: float | None, shown: str
) -> None:
    change = {"lever_key": "prefix_caching=on"}
    pack = [
        _record("memory/promoted", status="promoted", change=change, measured=_measured(0.20)),
        _record("memory/measured", change=change, measured=_measured(0.05)),
    ]
    numbers: dict[str, Any] = {} if local_median is None else {"measured": _measured(local_median)}
    local = _record("memory/mine", layer="local", change=change, **numbers)
    (offer,) = _match(pack=pack, local=[local]).offers
    assert (offer.record, offer.layer, offer.status) == (
        "test/memory/promoted@1",
        "pack",
        "promoted",
    )
    assert offer.measured is not None
    median = offer.measured.metrics["throughput"].median
    assert median == {"local": 0.12, "promoted": 0.20}[shown]
    assert offer.direction_note is None


def test_a_local_run_measuring_the_opposite_beyond_noise_demotes_the_offer() -> None:
    change = {"lever_key": "prefix_caching=on"}
    pack = _record("memory/promoted", status="promoted", change=change)
    worse = _record(
        "memory/mine",
        layer="local",
        change=change,
        measured=_measured(-0.15),
        evidence=[{"kind": "expdb", "ref": "run-a..run-b"}],
    )
    (offer,) = _match(pack=[pack], local=[worse]).offers
    assert offer.direction_note == "your run run-b measured throughput -15%"
    within_noise = _record("memory/mine", layer="local", change=change, measured=_measured(-0.01))
    assert _match(pack=[pack], local=[within_noise]).offers[0].direction_note is None


@pytest.mark.parametrize(
    ("local_rev", "winner"),
    [(2, "local"), (1, "pack")],
)
def test_the_same_id_in_two_layers_keeps_the_higher_rev_and_the_pack_on_a_tie(
    local_rev: int, winner: str
) -> None:
    pack = _record("memory/prefix")
    local = _record("memory/prefix", layer="local", rev=local_rev)
    (offer,) = _match(pack=[pack], local=[local]).offers
    assert offer.layer == winner


def test_records_this_tensward_cannot_use_are_skipped_and_counted() -> None:
    newer = _record("a/newer", requires_tensward="9.0")
    no_rule = _record("a/no-rule", change={"rule": "vllm:not_in_this_tensward"})
    bad_flag = _record("a/bad-flag", change={"flags": {"--enforce-eager": True}})
    good_flag = _record("a/good-flag", change={"flags": {"--stream-interval": "8"}})
    found = _match(pack=[newer, no_rule, bad_flag, good_flag])
    assert found.skipped == {"flag_not_allowed": 1, "missing_rule": 1, "requires_tensward": 1}
    assert [offer.record for offer in found.offers] == ["test/a/good-flag@1"]


def test_an_unknown_gpu_cc_under_a_validity_range_leaves_the_record_unknown() -> None:
    record = _record("gpu/needs-ada", validity={"gpu_cc": ">=8.9"})
    facts = Facts({**FACTS.to_json(), "gpu.cc": None, "gpu.arch": None})
    layers = Layers(playbook=PLAYBOOK, source=None, source_status={}, local=(record,))
    found = match(
        facts, layers, engine_family="vllm", engine_version="0.30.0",
        tensward_version="0.3.15", rules=(), levers=LEVERS,
    )  # fmt: skip
    assert found.offers == ()
    assert found.unknown == (("test/gpu/needs-ada@1", ("gpu.cc",)),)


def test_a_lever_key_guard_blocks_the_playbook_offer_that_would_set_it() -> None:
    found = _match(pack=[_guard("quant/no-fp8", ["quantization=fp8"])])
    fp8 = lever_key({"quantization": "fp8"}, {}, before={"quantization": None}, canonical=dict)
    bnb = lever_key(
        {"quantization": "bitsandbytes"}, {}, before={"quantization": None}, canonical=dict
    )
    assert found.blocked_by("fp8-weights", fp8) == "quant/no-fp8 says no"
    assert found.blocked_by("bnb-4bit-weights", bnb) is None


def test_a_guard_on_one_part_blocks_a_combined_lever_key() -> None:
    stacked = "kv_cache_precision=8bit+quantization=fp8"
    sets = {"kv_cache_precision": "8bit", "quantization": "fp8"}
    change = {"levers": list(sets), "sets": sets, "lever_key": stacked}
    lever = _record("memory/stacked", status="documented", change=change)
    assert len(_match(pack=[lever]).worth_measuring) == 1
    found = _match(pack=[_guard("memory/no-fp8-kv", ["kv_cache_precision=8bit"]), lever])
    assert found.worth_measuring == ()
    assert found.blocked_by(stacked) == "memory/no-fp8-kv says no"


def test_a_coded_record_whose_summary_is_false_leaves_the_offer_and_is_counted() -> None:
    code = {"applies": "vllm:fast_prefill_applies"}
    disagrees = _record("prefill/coded", oss_entry="prefix-caching", code=code, when=NEVER)
    agrees = _record("memory/coded", oss_entry="more-kv-memory", code=code)
    layers = Layers(
        playbook=PLAYBOOK, source=_Pack([disagrees, agrees]), source_status={}, local=()
    )
    found = _match(pack=[disagrees, agrees])
    assert found.offers == ()
    offered = annotate_offered(found, FACTS, layers, ["prefix-caching", "more-kv-memory"])
    assert offered.skipped == {"summary_disagrees": 1}
    assert offered.by_entry("prefix-caching") is None
    explained = offered.by_entry("more-kv-memory")
    assert explained is not None and explained.explanation.fired_on[0].observed == 18

    measured = {**_measured(0.2), "launches": [_gain("b")]}
    fields = {"oss_entry": "more-kv-memory", "code": code, "measured": measured, **KV}
    placed = _record("memory/placed", **fields)
    layers = Layers(playbook=PLAYBOOK, source=_Pack([placed]), source_status={}, local=())
    for attention, region in (("gqa", "same"), (UNKNOWN, "outside")):
        facts = Facts({**RUN.to_json(), "model.attention": attention})
        pending = _match(pack=[placed], facts=facts)
        found = annotate_offered(pending, facts, layers, ["more-kv-memory"])
        (offer,) = found.offers
        assert (offer.region, offer.contested, found.nulls) == (region, False, {})
        assert (offer.line or "").endswith("measure it") == (region == "outside")


KV = {"similarity": {"profile": "kv"}}
RUN = Facts({**FACTS.to_json(), "run.kv_state": "filling", "model.attention": "gqa"})
THERE = {"engine.minor": "0.30", "model.attention": "gqa", "run.kv_state": "filling"}


def _launch(id_: str, at: Mapping[str, Any] = THERE, **fields: Any) -> dict[str, Any]:
    seen = {
        "family": "gemma4",
        "arch": "ada",
        "engine_minor": "0.30",
        "outcome": {"throughput": "none"},
        "model": "Gemma 4 E4B",
        "gpu_model": "L4",
        "at": dict(at),
    }
    return {"id": id_, **{**seen, **fields}}


def _null(
    evidence: Sequence[tuple[str, str]],
    launches: Sequence[dict[str, Any]] = (),
    *,
    direction: str = "none",
    basis: str = "measured",
    median: float = 0.0,
    **fields: Any,
) -> Record:
    measured = {**_measured(median), "launches": list(launches)}
    return _record(
        "memory/gmu-null", status="documented",
        effect={"throughput": {"direction": direction, "basis": basis}},
        evidence=[{"kind": kind, "ref": ref} for kind, ref in evidence],
        measured=measured, **fields,
    )  # fmt: skip


def _offer(*launches: dict[str, Any], status: str = "measured") -> Record:
    measured = {**_measured(0.2), "launches": list(launches)}
    return _record("memory/gmu-up", layer="local", status=status, measured=measured, **KV)


def _gain(id_: str, size: float = 0.2, **fields: Any) -> dict[str, Any]:
    return _launch(
        id_, outcome={"throughput": "improves"}, change={"throughput": [size, size, size]}, **fields
    )


NULL_CASES = {
    "one launch demotes": ([("expdb", "a..b")], [_launch("b")], False, 1),
    "two aliases of one launch do not remove": (
        [("expdb", "a..b"), ("expdb", "b")], [_launch("b"), _launch("b")], False, 1,
    ),
    "two independent launches remove": (
        [("expdb", "a..b"), ("expdb", "a..c")], [_launch("b"), _launch("c")], True, 2,
    ),
    "launches out of the region do not count": (
        [("expdb", "a..b"), ("expdb", "a..c"), ("expdb", "a..d"), ("expdb", "a..e")],
        [
            _launch("b"),
            _launch("c", {**THERE, "run.kv_state": "preempting"}),
            _launch("d", {**THERE, "engine.minor": "0.29"}),
            _launch("e", outcome={"throughput": "improves"}),
        ],
        False, 1,
    ),
    "agent screens never remove": ([("agent", "screen-1"), ("agent", "screen-2")], [], False, 0),
}  # fmt: skip
OFFER_CASES = {
    "a launch in the same KV state carries its number": (
        {}, "Measured on Gemma 4 E4B, L4, same KV state (filling): req/s +20.0%", "same", True,
    ),
    "a run of unknown attention is worth measuring without a number": (
        {"model.attention": UNKNOWN},
        "Nearest measurement: Gemma 4 E4B, L4, model.attention gqa; here not recorded: "
        "outside, measure it",
        "outside", False,
    ),
    "a launch one KV band away keeps the direction only": (
        {"run.kv_state": "full"}, "Measured on Gemma 4 E4B, L4, nearby (kv state differs); "
        "direction only", "near", True,
    ),
}  # fmt: skip


@pytest.mark.parametrize("case", [*NULL_CASES, *OFFER_CASES])
def test_a_record_acts_in_the_region_it_was_measured_in(case: str) -> None:
    if case in OFFER_CASES:
        moved, line, region, offered = OFFER_CASES[case]
        found = _match(local=[_offer(_gain("b"))], facts=Facts({**RUN.to_json(), **moved}))
        (match_,) = found.offers if offered else found.worth_measuring
        assert (match_.region, match_.line, match_.nearest) == (region, line, "b")
        assert bool(found.offers) == offered and bool(found.worth_measuring) != offered
        assert (kb_block(match_, found)["measured"] is not None) == (region == "same")
        return
    evidence, launches, removes, counted = NULL_CASES[case]
    found = _match(pack=[_null(evidence, launches, when='gpu.arch == "ada"', **KV)], facts=RUN)
    assert not found.offers and not found.worth_measuring and not found.pending
    (null,) = found.nulls.values()
    agent = case.startswith("agent")
    where = "an agent screen" if agent else "Gemma 4 E4B, L4, same KV state (filling)"
    said = f"{'seen' if agent else 'measured'} null on {where} (test/memory/gmu-null@1)"
    assert (null.removes, null.launches) == (removes, counted)
    assert null.reason == (f"not offered: {said}" if removes else said)


def test_a_null_on_another_cell_makes_no_null() -> None:
    found = _match(pack=[_null([("expdb", "a..b")], [_launch("b")], when='gpu.arch == "ampere"')])
    assert found.nulls == {}


def test_a_local_loss_keeps_its_numbers_through_the_merge_and_two_launches_remove() -> None:
    key = {"lever_key": "prefix_caching=on"}
    promoted = _record(
        "memory/promoted", status="promoted", change=key, measured=_measured(0.10),
        evidence=[{"kind": "expdb", "ref": "p..q"}],
    )  # fmt: skip

    def local(*launches: dict[str, Any]) -> Record:
        return _null(
            [("expdb", "a..b"), ("expdb", "b")], launches, direction="worsens",
            layer="local", change=key, median=-0.20,
        )  # fmt: skip

    one = _match(pack=[promoted], local=[local(_launch("b", outcome={"throughput": "worsens"}))])
    (offer,) = one.offers
    assert offer.record == "test/memory/promoted@1" and offer.status == "promoted"
    assert offer.measured is not None and offer.measured.metrics["throughput"].median == -0.20
    assert offer.measured.metrics["throughput"].noise == 0.02
    assert offer.direction_note == "your run b measured throughput -20%"
    assert offer.evidence == ("pack:expdb:p..q", "local:expdb:a..b", "local:expdb:b")
    (null,) = one.nulls.values()
    assert (null.removes, null.launches, null.measured, null.direction_note) == (
        False, 1, offer.measured, offer.direction_note,
    )  # fmt: skip

    both = _match(
        pack=[promoted],
        local=[local(*(_launch(x, outcome={"throughput": "worsens"}) for x in "bc"))],
    )
    (null,) = both.nulls.values()
    assert (null.removes, null.launches) == (True, 2)
    assert null.evidence == ("local:expdb:a..b", "local:expdb:b")


def test_a_dotted_field_joins_the_capture_sizes_in_one_compilation_config() -> None:
    sizes = {"--compilation-config": '{"cudagraph_capture_sizes": [1, 2, 4]}'}
    merged = merge_json(sizes, {
        "-cc.cudagraph_mode": "FULL", "-cc.pass_config.enable_qk_norm_rope_fusion": "true",
        "--watermark": "0.03",
    })  # fmt: skip
    assert json.loads(str(merged.pop("--compilation-config"))) == {
        "cudagraph_capture_sizes": [1, 2, 4], "cudagraph_mode": "FULL",
        "pass_config": {"enable_qk_norm_rope_fusion": True},
    }  # fmt: skip
    assert merged == {"--watermark": "0.03"}


def test_value_rules() -> None:
    assert spec_tiers(16, 3, cut=None, max_batch=256) == [[1, 8, 3], [9, 16, 1], [17, 256, 0]]
    assert spec_tiers(17, 3, cut=None, max_batch=256) is None
    bimodal = [300 + i for i in range(70)] + [3000 + i for i in range(30)]
    assert long_prefill_threshold(bimodal) == 362
    assert long_prefill_threshold(list(range(100, 1100, 10))) is None
    calls = [
        json.dumps({"name": "get_weather", "arguments": {"city": city, "unit": "celsius"}})
        for city in ("Paris", "Berlin", "Rome", "Oslo", "Lima", "Kyiv")
    ]
    chat = [
        "The capital of France is Paris, known for its museums.",
        "Photosynthesis turns light into chemical energy in plants.",
        "A prime number has exactly two divisors: one and itself.",
    ]
    words = [re.findall(r"\w+|[^\w\s]", text) for text in (*calls, *chat)]
    assert (suffix_acceptance(words[:6]) or 0) >= 1.8
    assert suffix_acceptance(words[6:]) == 1.0
    assert [per_token_head_dtype(cc) for cc in (8.9, 7.5, None)] == [
        "fp8_per_token_head", "int8_per_token_head", None,
    ]  # fmt: skip
    offloads = ((32.0, 0.3), (14.0, 0.3), (32.0, 0.1), (None, 0.3), (32.0, None))
    assert [cpu_kv_offload_size(gib, gap) for gib, gap in offloads] == [15.0, 2.0, None, None, None]


def test_a_json_flag_spelled_whole_or_dotted_is_one_change_and_one_lever_key() -> None:
    from tensward.engines import canonicaliser
    from tensward.engines.vllm import VLLM
    from tensward.history import command_args
    from tensward.kb_schema import lever_key
    from tensward.rejected import as_change
    from tensward.settings import Settings

    def key_of(sets, flags):
        return lever_key(sets, flags, canonical=canonicaliser("vllm"))

    whole = 'compilation-config={"cudagraph_mode": "FULL", "cudagraph_capture_sizes": [1, 2]}'
    dotted = ['compilation-config={"cudagraph_capture_sizes":[1,2]}', "-cc.cudagraph_mode=FULL"]
    printed = [
        "tensward analyse --project p " + " ".join(f"--engine-arg '{arg}'" for arg in args)
        for args in ([whole], dotted)
    ]
    assert (
        as_change(command_args(printed[0]), VLLM.flag_rules)
        == as_change(command_args(printed[1]), VLLM.flag_rules)
        == {
            "compilation-config.cudagraph_mode=FULL",
            "compilation-config.cudagraph_capture_sizes=[1, 2]",
        }
    )
    keys = set()
    for args in ([whole], dotted):
        settings = Settings()
        for arg in args:
            settings = VLLM.with_engine_arg(settings, arg)
        keys.add(key_of({}, settings.extra_args))
    record = merge_json({}, {"-cc.cudagraph_mode": "FULL",
                             "--compilation-config.cudagraph_capture_sizes": "[1, 2]"})  # fmt: skip
    keys.add(key_of({}, record))
    assert keys == {
        "compilation_config.cudagraph_capture_sizes=[1, 2]+compilation_config.cudagraph_mode=FULL"
    }


@pytest.mark.parametrize(
    ("basis", "evidence", "launches"),
    [
        ("derived", [("expdb", "a..b")], [_launch("b")]),
        ("prior", [("doc", "playbook")], []),
        ("measured", [("doc", "playbook")], []),
        ("measured", [("doc", "playbook")], [_launch("b", outcome={"ttft": "none"})]),
    ],
)
def test_a_derived_none_or_an_unbacked_effect_is_no_null(
    basis: str, evidence: Sequence[tuple[str, str]], launches: Sequence[dict[str, Any]]
) -> None:
    found = _match(pack=[_null(evidence, launches, basis=basis, when='gpu.arch == "ada"')])
    assert found.nulls == {}


def _on(family: str, records: Sequence[Record], levers: frozenset[str] = LEVERS) -> Matches:
    layers = Layers(playbook=PLAYBOOK, source=_Pack(records), source_status={}, local=())
    return match(
        FACTS, layers, engine_family=family, engine_version="0.30.0",
        tensward_version="0.3.15", rules=(), levers=levers,
    )  # fmt: skip


def _per_engine(median: float = 0.2, status: str = "measured", **fields: Any) -> Record:
    return _record(
        "any/shared", engine="any", measured={"vllm": {**_measured(median), "status": status}},
        **fields,
    )  # fmt: skip


def test_an_any_record_resolves_through_its_on_table_for_the_engine_it_names() -> None:
    on = {"vllm": {"engine_version": ">=0.30", "flags": {"--stream-interval": "8"}}}
    record = _per_engine(change={"levers": [], "lever_key": "stream=on", "on": on})
    (offer,) = _on("vllm", [record]).offers
    assert (offer.flags, offer.measured is not None) == ({"--stream-interval": "8"}, True)
    assert _on("llamacpp", [record]).skipped == {"not_for_engine": 1}


def test_a_neutral_any_record_measured_on_another_engine_is_documented_there() -> None:
    record = _per_engine()
    assert [m.record for m in _on("vllm", [record]).offers] == ["test/any/shared@1"]
    other = _on("llamacpp", [record])
    assert not other.offers and other.worth_measuring[0].measured is None
    assert other.worth_measuring[0].status == "documented"
    assert _on("llamacpp", [record], frozenset()).skipped == {"not_for_engine": 1}


def test_a_worsens_null_measured_per_engine_demotes_only_its_own_engine() -> None:
    entry = {
        **_measured(-0.2),
        "status": "measured",
        "launches": [_launch("b", outcome={"throughput": "worsens"})],
    }
    record = _record(
        "any/worse", engine="any", status="documented", measured={"vllm": entry},
        effect={"throughput": {"direction": "worsens", "basis": "measured"}},
    )  # fmt: skip
    assert len(_on("vllm", [record]).nulls) == 1
    assert _on("llamacpp", [record]).nulls == {}


def test_the_engine_view_is_idempotent() -> None:
    from tensward.matcher import for_engine

    record = _per_engine(change={"levers": [], "lever_key": "k=on", "on": {"vllm": {}}})
    view = for_engine(record, "vllm", LEVERS)
    assert view is not None and view.engine == "vllm" and view.scope == "any"
    assert not (view.change and view.change.on)
    assert for_engine(view, "vllm", LEVERS) == view


_KV = Similarity(
    profile="kv",
    why="the effect depends on how full the KV pool is",
    hard=("engine.minor", "model.attention"),
    soft=(
        Axis(fact="run.kv_state", order=("roomy", "filling", "full", "preempting"), regime=True),
        Axis(fact="run.weights_share", cuts=(0.4, 0.75)),
    ),
)
_KERNEL = resolve({"profile": "kernel", "hard": ["run.linear_kernel"]}, "test")
_ATTN = resolve({"profile": "kernel", "hard": ["run.attn_backend"]}, "test")
_BASE: dict[str, object] = {
    "engine.minor": "11669",
    "model.attention": "gqa",
    "run.kv_state": "filling",
    "run.weights_share": 0.32,
    "run.linear_kernel": "marlin",
    "gpu.arch": "ada",
    "run.decode_share_of_ceiling": 0.6,
    "run.running_p50": 12.0,
}


@pytest.mark.parametrize(
    ("sim", "at", "here", "dropped", "expected"),
    [
        (_KV, {}, {}, (), ("same", 0, None)),
        (_KV, {}, {"run.weights_share": 0.69}, (), ("near", 0, "run.weights_share")),
        (_KV, {}, {"run.kv_state": "full"}, (), ("near", 1, "run.kv_state")),
        (
            _KV,
            {"run.kv_state": "roomy"},
            {"run.kv_state": "preempting"},
            (),
            ("outside", 3, "run.kv_state"),
        ),
        (_KV, {"model.attention": "mla"}, {}, (), ("outside", None, "model.attention")),
        (_KV, {"model.attention": UNKNOWN}, {}, (), ("outside", None, "model.attention")),
        (_KV, {"run.kv_state": UNKNOWN}, {}, (), ("near", 1, "run.kv_state")),
        (_KV, {"run.kv_state": UNKNOWN}, {}, ("run.kv_state",), ("near", 0, "run.kv_state")),
        (_KERNEL, {"run.linear_kernel": UNKNOWN}, {}, (), ("same", 0, None)),
        (
            _KERNEL,
            {"run.linear_kernel": UNKNOWN, "gpu.arch": "ada"},
            {"gpu.arch": "ampere"},
            (),
            ("outside", None, "gpu.arch"),
        ),
        (_KERNEL, {"run.linear_kernel": "cutlass"}, {}, (), ("outside", None, "run.linear_kernel")),
        (_KV, {"run.weights_share": UNKNOWN}, {}, ("run.weights_share",), ("same", 0, None)),
        (
            _ATTN,
            {"run.attn_backend": ["TRITON_ATTN", "FLASH_ATTN"]},
            {"run.attn_backend": frozenset({"FLASH_ATTN", "TRITON_ATTN"})},
            (),
            ("same", 0, None),
        ),
    ],
)
def test_band_distance_reads_same_near_and_outside(
    sim: Similarity,
    at: dict[str, object],
    here: dict[str, object],
    dropped: tuple[str, ...],
    expected: tuple[str, int | None, str | None],
) -> None:
    placed = place(sim, {**_BASE, **at}, {**_BASE, **here}, frozenset(dropped))
    assert (placed.region, placed.distance, placed.deciding) == expected
