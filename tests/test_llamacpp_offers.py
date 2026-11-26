import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from registration_fixtures import without_mtp, write_gguf

from tensward.context import AnalysisContext, active
from tensward.engines import ENGINES
from tensward.engines.llamacpp import predicates
from tensward.engines.llamacpp.flags import HelpManifest
from tensward.engines.llamacpp.predicates import slots_applies, with_more_slots
from tensward.formats.gguf import GGUF
from tensward.measurement import Measurement
from tensward.settings import Settings

LFM2_START = """\
0.05.377.206 I llama_context: n_ctx                 = 128000
0.05.377.210 I llama_context: kv_unified            = true
0.05.590.661 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 128000
"""


def test_slots_offered_when_np_is_unset(tmp_path):
    (tmp_path / "server.log").write_text(LFM2_START)
    situation = SimpleNamespace(
        measurement=Measurement(
            160, 0, peak_in_flight=8, peak_running=4.0, peak_waiting=4.0, max_prompt_tokens=517
        ),
        facts=SimpleNamespace(output_tokens=256),
        settings=Settings(),
        diagnosis=None,
    )
    engine = ENGINES["llamacpp"]
    with active(AnalysisContext(engine="llamacpp", new_run=tmp_path)):
        assert slots_applies(situation, engine)
        settings = with_more_slots(situation, engine)
    assert settings.max_concurrent_requests == 8
    assert settings.extra_args["--kv-unified"] is True
    # more slots make each stream slower: not for a per-stream speed goal
    with active(AnalysisContext(engine="llamacpp", new_run=tmp_path, goal="tpot")):
        assert slots_applies(situation, engine) is None


SPLIT_START = """\
0.05.377.206 I llama_context: n_ctx                 = 131072
0.05.377.208 I llama_context: n_ubatch              = 512
0.05.377.210 I llama_context: kv_unified            = false
0.05.428.961 I sched_reserve:      CUDA0 compute buffer size =  1008.41 MiB
0.05.590.661 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 32768
"""
MIB = 2**20


def _queued_clients(tmp_path, peak_tokens):
    (tmp_path / "server.log").write_text(SPLIT_START)
    return SimpleNamespace(
        measurement=Measurement(
            97, 0, peak_in_flight=16, peak_running=4.0, peak_waiting=12.0,
            max_prompt_tokens=16896, peak_context_tokens=peak_tokens,
        ),
        facts=SimpleNamespace(output_tokens=1024),
        settings=Settings(),
        diagnosis=None,
        fired=lambda bottleneck: bottleneck == "queueing",
    )  # fmt: skip


def _room(mib, **placed):
    fit = {"device_free": 80_000 * MIB, "device_need": (80_000 - mib) * MIB, "pool_tokens": 0}
    return {"fit": {**fit, "host_free": 200 * 2**30, "host_need": 0}, "placement": placed}


def test_queued_clients_get_a_slot_each_in_one_pool_sized_from_the_tokens_in_flight(tmp_path):
    from tensward.presets import Choices, write_choices

    situation = _queued_clients(tmp_path, 163_000)
    engine = ENGINES["llamacpp"]
    with active(AnalysisContext(engine="llamacpp", new_run=tmp_path, placement=_room(8000))):
        why = predicates.shared_pool_applies(situation, engine)
        settings = predicates.with_shared_pool(situation, engine)
    assert "every client a slot in one shared pool" in why
    # 2.09 bytes a cell per -ub token per slot over 130 MiB: 3,327 MiB, 2,319 over today's
    assert "over 130 MiB, never under today's buffer) 2,319 MiB" in why
    # 20% over the peak, rounded up to 256
    assert settings.max_concurrent_requests == 16
    assert settings.max_context_len == 195_840 and settings.extra_args["--kv-unified"] is True
    # never below the longest request the user registered, which any one request may fill
    project = tmp_path / "project"
    project.mkdir()
    write_choices(
        project,
        Choices(
            "agent", None, "default", False, 16, "throughput", "command", longest_request=200_000
        ),
    )
    context = AnalysisContext(
        engine="llamacpp", new_run=tmp_path, project=project, placement=_room(8000)
    )
    with active(context):
        assert predicates.with_shared_pool(situation, engine).max_context_len == 200_192
        assert predicates.trim_for_slots_applies(situation, engine) is None
    # a dense model whose card holds fewer slots: fewer slots than clients share a smaller pool
    with active(AnalysisContext(engine="llamacpp", new_run=tmp_path, placement=_room(2200))):
        settings = predicates.with_shared_pool(situation, engine)
        assert "for 13 of the 16 clients" in predicates.shared_pool_applies(situation, engine)
    # the compute buffer grows with the slots, so fewer slots need less of it
    assert (settings.max_concurrent_requests, settings.max_context_len) == (13, 158_976)


def test_shared_pool_pays_with_expert_layers_and_reads_a_logged_shared_start(tmp_path):
    project = tmp_path / "project"
    runs = project / "runs"
    earlier, here = runs / "0001-baseline", runs / "0002-step"
    earlier.mkdir(parents=True)
    here.mkdir()
    (earlier / "metrics.json").write_text(json.dumps({"peak_context_tokens": 170_000}))
    (earlier / "server.log").write_text(
        SPLIT_START.replace("= false", "= true").replace("1008.41", "2830.00")
        .replace("131072", "196608").replace("n_slots = 4", "n_slots = 16")
    )  # fmt: skip
    situation = _queued_clients(here, 120_000)
    anatomy = SimpleNamespace(
        layers=48, state_bytes_per_seq=0, expert_layer_bytes=(1600 * MIB,) * 48,
        model_type="qwen3next", precisions={},
    )  # fmt: skip
    context = AnalysisContext(
        engine="llamacpp", new_run=here, project=project, anatomy=anatomy,
        placement=_room(300, host_expert_layers=3, host_layers=0),
    )  # fmt: skip
    engine = ENGINES["llamacpp"]
    with active(context):
        why = predicates.shared_pool_applies(situation, engine)
        settings = predicates.with_shared_pool(situation, engine)
    # the peak of the earlier run sizes the pool; its shared start sizes the compute buffer
    assert settings.max_context_len == 204_032 and settings.max_concurrent_requests == 16
    assert why.startswith(
        "shared pool of 204,032 tokens (the most measured in flight plus 20%) for every client, "
        "paid with 2 more expert layers in host memory (--n-cpu-moe 3 to 5)"
    )
    assert "from this project's shared-pool start" in why
    assert settings.extra_args["--n-cpu-moe"] == 5


@pytest.mark.parametrize("usable, layers", [(None, 4), (3000, 4), (1200, 5)])
def test_shared_pool_keeps_a_gib_allocatable_at_its_peak(tmp_path, usable, layers):
    """rc9 on the rc7 H100 smoke: the pool kept the fit's 1,024 MiB margin, but 480 MiB of the
    driver's total is never allocatable and CUDA takes more after the start, so about 417 MiB
    would have stayed free. The run's own allocatable free at its peak, less the pool's extra,
    must stay at 1 GiB; one more expert layer pays when it would not."""
    runs = tmp_path / "project" / "runs"
    earlier, here = runs / "0001-baseline", runs / "0002-step"
    earlier.mkdir(parents=True)
    here.mkdir()
    (earlier / "metrics.json").write_text(json.dumps({"peak_context_tokens": 170_000}))
    (earlier / "server.log").write_text(
        SPLIT_START.replace("= false", "= true").replace("1008.41", "2830.00")
        .replace("131072", "196608").replace("n_slots = 4", "n_slots = 16")
    )  # fmt: skip
    situation = _queued_clients(here, 120_000)
    anatomy = SimpleNamespace(
        layers=48, state_bytes_per_seq=0, expert_layer_bytes=(1600 * MIB,) * 48,
        model_type="qwen3next", precisions={},
    )  # fmt: skip
    placement = _room(1900, host_expert_layers=3, host_layers=0)
    if usable is not None:
        placement["memory"] = {"usable_mib": usable}
    context = AnalysisContext(
        engine="llamacpp", new_run=here, project=runs.parent, anatomy=anatomy, placement=placement
    )
    engine = ENGINES["llamacpp"]
    with active(context):
        why = predicates.shared_pool_applies(situation, engine)
        settings = predicates.with_shared_pool(situation, engine)
    # 1,924 MiB more compute buffer; one more layer frees 1,600, two 3,200
    assert settings.extra_args["--n-cpu-moe"] == layers
    if usable is None:
        assert "counting 700 MiB the driver and CUDA hold beyond the fit's margin" in why
    else:
        left = usable - 1924 + 1600 * (layers - 3)
        assert f"at its peak about {left:,} MiB of the card stays allocatable" in why


def test_shared_pool_holds_the_workloads_longest_sessions_and_makes_room_with_q8_first(
    tmp_path, monkeypatch
):
    situation = _queued_clients(tmp_path, 163_000)
    (tmp_path / "server.log").write_text(  # with the KV types: a q8_0 cache can be offered
        SPLIT_START + "0.05.377.300 I llama_kv_cache: size = 3840.00 MiB (131072 cells, "
        "48 layers, 4/1 seqs), K (f16): 1920.00 MiB, V (f16): 1920.00 MiB\n"
    )  # fmt: skip
    os.utime(tmp_path / "server.log", ns=(1, 1))  # the log is cached by its mtime
    # 16 sessions of two turns, the second 16,000 tokens long (4 characters a token) and capped
    # at 1,024 output tokens: up to 16 x 17,024 tokens in flight, more than the peak + 20%
    situation.facts = SimpleNamespace(
        output_tokens=1024, context_limit=262_144,
        prompts=tuple("x" * (4_000 if turn == 0 else 64_000) for _ in range(16) for turn in (0, 1)),
        sessions=tuple(f"s{s}" for s in range(16) for _ in (0, 1)), max_outputs=(1024,) * 32,
    )  # fmt: skip
    monkeypatch.setattr(predicates, "_kv_token_bytes", lambda seen, layers: 30_720.0)
    anatomy = SimpleNamespace(
        layers=48, state_bytes_per_seq=0, expert_layer_bytes=(1600 * MIB,) * 48,
        model_type="qwen3next", precisions={},
    )  # fmt: skip
    context = AnalysisContext(
        engine="llamacpp", new_run=tmp_path, anatomy=anatomy,
        placement=_room(4700, host_expert_layers=3, host_layers=0),
    )  # fmt: skip
    engine = ENGINES["llamacpp"]
    with active(context):
        why = predicates.shared_pool_applies(situation, engine)
        settings = predicates.with_shared_pool(situation, engine)
    assert settings.max_context_len == 16 * 17_024 and settings.max_concurrent_requests == 16
    assert "(the workload's 16 longest sessions, prompt plus output cap)" in why
    # f16 needs 7.2 GiB of the 3.9 left: a q8_0 cache pays for it, then the experts are next
    assert settings.kv_cache_dtype == "q8_0" and "--n-cpu-moe" not in settings.extra_args
    assert "paid with a q8_0 KV cache (LOSSY: it changes answers)" in why
    assert "the other ways to make room, priced the same way: 3 more expert layers" in why


def test_shared_pool_sized_from_the_runs_ranks_by_price_says_its_risk_and_gives_way_once_tried(
    tmp_path, monkeypatch
):
    from tensward.rejected import Rejection

    runs = tmp_path / "project" / "runs"
    tried, here = runs / "0001-pool", runs / "0002-baseline"
    tried.mkdir(parents=True)
    here.mkdir()
    situation = _queued_clients(here, 100_000)
    (here / "server.log").write_text(
        SPLIT_START + "0.05.377.300 I llama_kv_cache: size = 3840.00 MiB (131072 cells, "
        "48 layers, 4/1 seqs), K (f16): 1920.00 MiB, V (f16): 1920.00 MiB\n"
    )  # fmt: skip
    # the workload's worst case: 16 sessions of 16,000 + 1,024 tokens; the run had 8 second turns
    # and 8 first turns of 1,000 tokens in flight at once, at their 1,024-token caps: 152,384,
    # +20% 182,861
    situation.facts = SimpleNamespace(
        output_tokens=1024, context_limit=262_144,
        prompts=tuple("x" * (4_000 if turn == 0 else 64_000) for _ in range(16) for turn in (0, 1)),
        sessions=tuple(f"s{s}" for s in range(16) for _ in (0, 1)), max_outputs=(1024,) * 32,
    )  # fmt: skip
    rows = [
        {"prompt_index": 2 * i + (i < 8), "prompt_tokens": 16_000 if i < 8 else 1_000,
         "dispatch_ns": i, "terminal_ns": 100 + i}
        for i in range(16)
    ]  # fmt: skip
    (here / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(predicates, "_kv_token_bytes", lambda seen, layers: 30_720.0)
    prices = {(False, 1, "measured"): 0.30, (True, 1, "workload"): 0.25}

    def forecast(situation, seen, found):
        price = prices.get((found.q8, found.more, found.sized), 0.2)
        return predicates.Forecast(16, 24, 4, (60, 60), 1.5, (1.5, 2), 0.2, (price, price + 0.1))

    monkeypatch.setattr(predicates, "_forecast", forecast)
    anatomy = SimpleNamespace(
        layers=48, state_bytes_per_seq=0, expert_layer_bytes=(1600 * MIB,) * 48,
        model_type="qwen4exp", precisions={},
    )  # fmt: skip
    context = AnalysisContext(
        engine="llamacpp", new_run=here, project=runs.parent, anatomy=anatomy,
        placement=_room(3200, host_expert_layers=3, host_layers=0),
    )  # fmt: skip
    engine = ENGINES["llamacpp"]
    with active(context):
        why = predicates.shared_pool_applies(situation, engine)
        first = predicates.with_shared_pool(situation, engine)
    # the pool from the runs is cheaper and ranks first by its price; its risk is said plainly
    # and the pools that hold the worst case stay on screen as the safer ways
    assert (first.max_concurrent_requests, first.max_context_len) == (16, 183_040)
    assert first.extra_args["--n-cpu-moe"] == 4 and first.kv_cache_dtype != "q8_0"
    assert "(the 16 largest requests measured in flight at their output caps, plus 20%)" in why
    assert "RISK: the pool is below the workload's worst case (272,384 tokens)" in why
    assert "the safer ways, each pool sized to hold the workload's worst case: a q8_0" in why
    # q8_0 decode is priced from this model's own measurement, another model's from the research
    with active(context):
        assert predicates._q8_speed()[0] == (0.93, 0.97)
    with active(
        replace(context, anatomy=SimpleNamespace(**{**vars(anatomy), "model_type": "qwen3next"}))
    ):
        assert predicates._q8_speed() == ((0.56, 0.85), None)
    # tried and not kept: the next way in the order is the command
    args = engine.engine_args_between(situation.settings, first)
    (tried / "run.json").write_text(json.dumps({"engine_args": args}))
    for name in ("metrics.json", "report.json"):
        (tried / name).write_text("{}")
    with active(context):
        second = predicates.with_shared_pool(situation, engine)
    assert second.kv_cache_dtype == "q8_0" and second.max_context_len == 16 * 17_024
    assert second.extra_args["--n-cpu-moe"] == 4
    (tried / "run.json").unlink()
    refused = Rejection("0001-pool", tuple(args), tuple(args), None, "not kept", "")
    with active(replace(context, rejected=(refused,))):
        assert predicates.with_shared_pool(situation, engine).kv_cache_dtype == "q8_0"


def test_shared_pool_decode_step_is_the_low_end_of_the_requests_own_token_times():
    # a request's TPOT counts the other slots' prompt batches between its tokens
    rows = [{"timings": {"predicted_n": 50, "predicted_per_token_ms": 24.0 + i}} for i in range(20)]
    rows.append({"timings": {"predicted_n": 5, "predicted_per_token_ms": 1.0}})
    assert predicates._step_alone(rows) == 26.0
    assert predicates._step_alone(rows[:5]) is None


MMAP_START = """\
0.01.000.000 W model_load: tensor overrides to CPU are used with mmap enabled
0.02.000.000 I load_tensors: loading model tensors, this can take a while... (load_mode = mmap)
"""


def _load_mode_offer(
    tmp_path, kind, host_bytes, settings=None, log=MMAP_START, environment=None, facts=None
):  # fmt: skip
    from tensward.engines.llamacpp.predicates import load_mode_none_applies

    run_dir = tmp_path / str(len(list(tmp_path.iterdir())))  # the log is cached by its mtime
    run_dir.mkdir()
    (run_dir / "server.log").write_text(log)
    situation = SimpleNamespace(settings=settings or Settings())
    placement = {
        "regime": {"placement": kind},
        "placement": {"host_weight_bytes": host_bytes},
        "environment": environment or {},
    }
    context = AnalysisContext(
        engine="llamacpp", new_run=run_dir, placement=placement, kb_facts=facts
    )
    with active(context):
        return load_mode_none_applies(situation, ENGINES["llamacpp"])


def test_load_mode_none_offered_for_cpu_offloaded_experts_read_through_mmap(tmp_path):
    from tensward.engines.llamacpp.predicates import with_load_mode_none

    assert "through mmap" in _load_mode_offer(tmp_path, "expert_offload", 8 * 2**30)
    settings = with_load_mode_none(SimpleNamespace(settings=Settings()), ENGINES["llamacpp"])
    assert settings.extra_args["--load-mode"] == "none"
    # L35's census: 97% of the host-to-GPU bytes from pageable memory at 6.5 GB/s
    from tensward.facts import Facts

    census = Facts({"census.pageable_h2d_share": 0.972, "census.h2d_gb_per_s": 6.54})
    assert _load_mode_offer(tmp_path, "expert_offload", 8 * 2**30, facts=census).startswith(
        "the GPU census saw 97% of the host-to-GPU bytes copied from pageable memory at 6.5 GB/s"
    )


def test_load_mode_none_not_offered_when_nothing_is_offloaded_or_it_is_set(tmp_path):
    assert _load_mode_offer(tmp_path, "full_gpu", 0) is None
    set_already = Settings(extra_args={"--load-mode": "mlock"})
    assert _load_mode_offer(tmp_path, "expert_offload", 8 * 2**30, set_already) is None
    direct = MMAP_START.replace("tensor overrides to CPU are used with mmap enabled", "x").replace(
        "(load_mode = mmap)", "(load_mode = dio)"
    )
    assert _load_mode_offer(tmp_path, "expert_offload", 8 * 2**30, log=direct) is None


def test_load_mode_none_reason_carries_the_ram_price(tmp_path):
    measured = {"mem_gib": 62.8, "mem_available_gib": 57.5}  # R0500L, g2-standard-16
    reason = _load_mode_offer(tmp_path, "expert_offload", 15.94e9, environment=measured)
    assert not isinstance(reason, predicates.HeldBack)
    assert "locks 14.8 GiB of host RAM" in reason and "about 42.7 GiB" in reason


def test_load_mode_none_held_back_when_host_ram_is_short(tmp_path):
    short = {"mem_gib": 31.3, "mem_available_gib": 20.0}
    reason = _load_mode_offer(tmp_path, "expert_offload", 15.94e9, environment=short)
    assert isinstance(reason, predicates.HeldBack)
    contained = {
        "mem_gib": 62.8,
        "mem_available_gib": 57.5,
        "cgroup_mem_gib": 32.0,
        "cgroup_mem_used_gib": 10.0,
    }  # a container shows the host's MemAvailable
    reason = _load_mode_offer(tmp_path, "expert_offload", 15.94e9, environment=contained)
    assert isinstance(reason, predicates.HeldBack) and "of 32 GiB" in reason


LOAD_NONE = """\
0.00.703.370 I load_tensors: loading model tensors, this can take a while... (load_mode = none)
"""


def test_load_mode_none_post_check_tells_a_lock_from_a_fallback():
    from tensward.engines.llamacpp.engine import host_lock
    from tensward.engines.llamacpp.startup import read_startup
    from tensward.engines.protocol import HostWindow
    from tensward.platforms.protocol import ProcessSample

    def lock(buffers, rss_gb, anon_gb, file_gb, placement="expert_offload"):
        mib = 1e9 / 2**20
        end = ProcessSample(0, None, None, rss_gb * mib, None, None,
                            rss_anon_mib=anon_gb * mib, rss_file_mib=file_gb * mib)  # fmt: skip
        lines = "".join(
            f"0.08.075.58{i} I load_tensors: {name:>12} model buffer size = {size} MiB\n"
            for i, (name, size) in enumerate(buffers)
        )
        started = read_startup(LOAD_NONE + lines)
        return host_lock(started, HostWindow(None, None, end), placement)

    gpu = ("CUDA0", 19979.78)  # R0500L agent-B and agent-F (forced fallback)
    assert lock([gpu, ("CUDA_Host", 15203.31)], 20.17, 4.0, 0.19)["took"]
    fell_back = lock([gpu, ("CPU", 15203.31)], 18.72, 18.52, 0.19)
    assert not fell_back["took"] and "did not take" in fell_back["words"]
    partial = lock([gpu, ("CUDA_Host", 8000.0), ("CPU", 7203.31)], 18.72, 10.3, 0.19)
    assert not partial["took"] and "pinned only 7.8 of the 14.8 GiB" in partial["words"]
    assert lock([("CPU", 35183.09)], 37.2, 36.9, 0.19, placement="cpu_only") is None
    assert lock([gpu, ("CPU", 15203.31)], 18.72, 18.52, 0.19, placement="dense_offload") is None


def test_load_mode_change_keeps_answers_like_a_placement_change():
    assert ENGINES["llamacpp"].answer_keep(["load_mode=none"])


def test_the_answer_check_keeps_hugetlb_and_backend_sampling_not_the_state_saves():
    keep = ENGINES["llamacpp"].answer_keep
    assert keep(["GLIBC_TUNABLES=glibc.malloc.hugetlb=1"]) and keep(["backend-sampling"])
    assert not keep(["GLIBC_TUNABLES=glibc.malloc.hugetlb=2"])
    assert not keep(["cache-ram=0", "ctx-checkpoints=0"])
    assert ENGINES["llamacpp"].touches_cache(["cache-ram=0", "ctx-checkpoints=0"])
    assert not ENGINES["llamacpp"].touches_cache(["GLIBC_TUNABLES=glibc.malloc.hugetlb=1"])


def test_load_mode_none_held_back_when_the_build_lacks_the_flag(monkeypatch):
    from tensward.engines.llamacpp import predicates

    manifest = SimpleNamespace(missing=lambda flags: list(flags))
    monkeypatch.setattr(predicates, "known_build", lambda: (None, manifest))
    assert "--load-mode" in predicates.HOLD_BACKS["load-mode-none"](SimpleNamespace())


def test_a_change_for_another_goal_is_offered_only_when_nothing_else_is():
    from tensward.playbook import Suggestion
    from tensward.suggest import Suggested, only_sides, side_metric

    entries = {entry.name: entry for entry in ENGINES["llamacpp"].playbook()}
    ubatch, slots = entries["raise-ubatch"], entries["llamacpp-slots"]
    assert ubatch.side is not None
    assert side_metric(ubatch, "tpot") == side_metric(ubatch, "throughput") == "ttft"
    assert side_metric(ubatch, "ttft") is None and side_metric(slots, "tpot") is None
    side = Suggestion(entry=ubatch, reason="r", tier="could_help", command="c", side="ttft")
    other = Suggestion(entry=slots, reason="r", tier="try_first", command="c")
    alone = Suggested(suggestions=(side,), not_applicable=(), blocked={})
    assert only_sides(alone).suggestions == (side,)
    assert only_sides(replace(alone, suggestions=(other, side))).suggestions == (other,)


def test_op_offload_threshold_offered_for_host_experts_and_short_computed_prompts(tmp_path):
    from tensward.engines.llamacpp.predicates import (
        op_offload_threshold_applies,
        with_op_offload_threshold,
    )
    from tensward.suggest import suggestion_command

    (tmp_path / "server.log").write_text(MMAP_START)
    placed = {"host_expert_layers": 17}
    placement = {"regime": {"placement": "expert_offload"}, "placement": placed, "environment": {}}

    def offer(prompt_tokens, hit_rate, settings=Settings(), engine_args=()):
        situation = SimpleNamespace(
            settings=settings,
            measurement=Measurement(
                160, 0, max_prompt_tokens=prompt_tokens, prefix_cache_hit_rate=hit_rate
            ),
        )
        context = AnalysisContext(
            engine="llamacpp", new_run=tmp_path, placement=placement, engine_args=engine_args
        )
        with active(context):
            return op_offload_threshold_applies(situation, ENGINES["llamacpp"]), situation

    reason, situation = offer(800, 0.83)
    assert "GGML_OP_OFFLOAD_MIN_BATCH=256" in reason
    assert offer(4000, 0.5)[0] is None
    placed["host_expert_layers"] = 7
    assert offer(800, 0.83)[0] is None
    placed["host_expert_layers"] = 17
    settings = with_op_offload_threshold(situation, ENGINES["llamacpp"])
    assert offer(800, 0.83, settings)[0] is None
    # a kept step's environment is inherited, not in its settings (L35 tries 3 and 5)
    assert (
        offer(800, 0.83, engine_args=("n-cpu-moe=17", "GGML_OP_OFFLOAD_MIN_BATCH=256"))[0] is None
    )
    with active(AnalysisContext(engine="llamacpp")):
        command = suggestion_command(ENGINES["llamacpp"], tmp_path, Settings(), settings)
    assert command.endswith("--engine-env GGML_OP_OFFLOAD_MIN_BATCH=256")


QWEN38_START = """\
0.08.357.594 I llama_context: n_ctx                 = 36096
0.08.372.764 I llama_kv_cache: size = 2256.00 MiB ( 36096 cells,  16 layers,  4/1 seqs), \
K (f16): 1128.00 MiB, V (f16): 1128.00 MiB
0.08.452.489 I sched_reserve:      CUDA0 compute buffer size =   183.74 MiB
"""


def _mtp_offer(
    tmp_path, monkeypatch, model, settings=Settings(), goal="tpot", accepted=None, tools=False
):
    monkeypatch.setattr("tensward.formats.gguf.fingerprint_files", lambda *a, **k: "0" * 64)
    run_dir = tmp_path / str(len(list(tmp_path.iterdir())))
    run_dir.mkdir()
    (run_dir / "server.log").write_text(QWEN38_START)
    (file,) = write_gguf(run_dir / "model.gguf", model)
    _, anatomy = GGUF.register(file, engine_build="b11669", engine_name="llamacpp")
    measured = Measurement(160, 0, spec_acceptance_length=accepted)
    facts = SimpleNamespace(offers_tools=tools, max_inflight=3)
    situation = SimpleNamespace(settings=settings, measurement=measured, facts=facts)
    context = AnalysisContext(engine="llamacpp", new_run=run_dir, goal=goal, anatomy=anatomy)
    with active(context):
        return anatomy.mtp_layers, predicates.mtp_speculation_applies(
            situation, ENGINES["llamacpp"]
        )


def test_mtp_speculation_needs_known_mtp_layers_and_no_speculation(tmp_path, monkeypatch):
    qwen = json.loads(
        (Path(__file__).parent / "gguf_headers/qwen3.8-27b-ud-q4_k_m.json").read_text()
    )
    layers, offer = _mtp_offer(tmp_path, monkeypatch, qwen)
    assert layers == 1 and "about 999 MiB (estimated)" in offer
    assert "x2.25 with 1 client against x1.66 with 4" in offer  # 3 requests at once
    assert _mtp_offer(tmp_path, monkeypatch, without_mtp(qwen)) == (0, None)
    unread = [t for t in qwen["tensors"] if ".nextn." not in t[0]]
    assert _mtp_offer(tmp_path, monkeypatch, {**qwen, "tensors": unread}) == (None, None)
    speculating = Settings(extra_args={"--spec-type": "ngram-mod"})
    assert _mtp_offer(tmp_path, monkeypatch, qwen, speculating) == (1, None)
    drafting = Settings(extra_args={"--spec-type": "draft-mtp", "--spec-draft-n-max": "2"})
    deeper = _mtp_offer(tmp_path, monkeypatch, qwen, drafting, accepted=2.445)[1]
    assert deeper.startswith(
        "72% of the 2 tokens drafted a step were accepted; --spec-draft-n-max 3"
    )
    assert _mtp_offer(tmp_path, monkeypatch, qwen, drafting, accepted=1.648) == (1, None)
    assert "drafts 4 tokens" in _mtp_offer(tmp_path, monkeypatch, qwen, tools=True)[1]
    at_four = Settings(extra_args={"--spec-type": "draft-mtp", "--spec-draft-n-max": "4"})
    assert _mtp_offer(tmp_path, monkeypatch, qwen, at_four, accepted=4.0, tools=True) == (1, None)
    assert _mtp_offer(tmp_path, monkeypatch, qwen, goal="ttft") == (1, None)
    sized = Settings(max_context_len=8192)
    monkeypatch.setattr(predicates, "sampled_memory", lambda run_dir, measurement: (20800, 22000))
    assert _mtp_offer(tmp_path, monkeypatch, qwen, sized) == (1, None)
    monkeypatch.setattr(predicates, "sampled_memory", lambda run_dir, measurement: (19850, 22000))
    assert "drafts 2 tokens" in _mtp_offer(tmp_path, monkeypatch, qwen, sized)[1]
    # the depth follows the card memory left: 3 for chat, 4 on tool turns, each when it fits
    monkeypatch.setattr(predicates, "sampled_memory", lambda run_dir, measurement: (19750, 22000))
    assert "drafts 3 tokens" in _mtp_offer(tmp_path, monkeypatch, qwen, sized)[1]
    assert "drafts 3 tokens" in _mtp_offer(tmp_path, monkeypatch, qwen, sized, tools=True)[1]
    monkeypatch.setattr(predicates, "sampled_memory", lambda run_dir, measurement: (19350, 22000))
    assert "drafts 4 tokens" in _mtp_offer(tmp_path, monkeypatch, qwen, sized, tools=True)[1]
    help_text = "--spec-type none,ngram-mod   kind\n--spec-draft-n-max N   most drafted\n"
    monkeypatch.setattr(
        predicates, "known_build", lambda: (None, HelpManifest.from_help(help_text))
    )
    held = predicates.HOLD_BACKS["llamacpp-mtp-speculation"](SimpleNamespace())
    assert held == "this build's --help lists no draft-mtp for --spec-type"


L27_START = """\
0.05.601.918 I llama_kv_cache: size = 5824.00 MiB ( 93184 cells,  16 layers,  4/1 seqs), \
K (f16): 2912.00 MiB, V (f16): 2912.00 MiB
0.05.703.401 I sched_reserve:      CUDA0 compute buffer size =   234.44 MiB
0.05.996.105 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 93184, kv_unified = 'true'
"""


def test_mtp_head_memory_and_the_hybrid_slots_crash_hold(tmp_path, monkeypatch):
    """L27 (b12246, Qwen3.8-27B on an L4, -np 4): with --spec-type draft-mtp at depth 2 the card
    took 335 MiB of MTP weights, a 244 MiB draft KV cache, a 130 MiB compute buffer, 40 MiB of
    hidden states and 1197 MiB more recurrent state; it aborted after 27 min (issue #31656)."""
    qwen = json.loads(
        (Path(__file__).parent / "gguf_headers/qwen3.8-27b-ud-q4_k_m.json").read_text()
    )
    monkeypatch.setattr("tensward.formats.gguf.fingerprint_files", lambda *a, **k: "0" * 64)
    (tmp_path / "server.log").write_text(L27_START)
    (file,) = write_gguf(tmp_path / "model.gguf", qwen)
    _, anatomy = GGUF.register(file, engine_build="b12246", engine_name="llamacpp")
    situation = SimpleNamespace(settings=Settings(), facts=SimpleNamespace(max_inflight=4))
    hold = predicates.HOLD_BACKS["llamacpp-mtp-speculation"]
    with active(AnalysisContext(engine="llamacpp", new_run=tmp_path, anatomy=anatomy)):
        draft = predicates._mtp_draft_bytes(predicates.view(situation), 2)
        assert 2100 * 2**20 < draft < 2250 * 2**20
        monkeypatch.setattr(predicates, "_build_number", lambda: 12246)
        assert hold(situation) == (
            "MTP drafting would raise throughput here (req/s +45% measured on Qwen3.8-27B, one "
            "L4, 3 agent clients), but this llama.cpp build can crash with it at more than one "
            "slot (#31656, fixed in b12676): upgrade llama.cpp to b12676 or later, then run "
            "tensward analyse again"
        )
        monkeypatch.setattr(predicates, "_build_number", lambda: None)
        assert hold(situation)  # a build that cannot be read is held too
        drafting = SimpleNamespace(settings=Settings(extra_args={"--spec-type": "draft-mtp"}))
        slots = predicates.HOLD_BACKS["llamacpp-slots"]
        assert slots(situation) is None
        monkeypatch.setattr(predicates, "_build_number", lambda: 12246)
        assert "your setup drafts with --spec-type draft-mtp" in slots(drafting)
        monkeypatch.setattr(predicates, "_build_number", lambda: 12676)
        assert hold(situation) is None and slots(drafting) is None


def test_defrag_threshold_from_the_environment_never_reaches_the_command_line():
    parsed = ENGINES["llamacpp"].parse_setup("LLAMA_ARG_DEFRAG_THOLD=0.1 llama-server -m /m.gguf")
    assert "--defrag-thold" not in parsed.settings.extra_args
    assert any("removed in v1.0.0" in note for note in parsed.notes)


def test_nothing_left_for_ttft_names_the_best_lever_for_another_goal(tmp_path, monkeypatch):
    from tensward.suggest import Suggested, _with_elsewhere

    qwen = json.loads(
        (Path(__file__).parent / "gguf_headers/qwen3.8-27b-ud-q4_k_m.json").read_text()
    )
    monkeypatch.setattr("tensward.formats.gguf.fingerprint_files", lambda *a, **k: "0" * 64)
    (tmp_path / "server.log").write_text(QWEN38_START)
    (file,) = write_gguf(tmp_path / "model.gguf", qwen)
    _, anatomy = GGUF.register(file, engine_build="b11669", engine_name="llamacpp")
    facts = SimpleNamespace(offers_tools=False, max_inflight=3)
    situation = SimpleNamespace(settings=Settings(), measurement=Measurement(160, 0), facts=facts)
    context = AnalysisContext(engine="llamacpp", new_run=tmp_path, goal="ttft", anatomy=anatomy)
    nothing = Suggested(suggestions=(), not_applicable=(), blocked={}, goal="ttft")
    monkeypatch.setattr(predicates, "_build_number", lambda: 12676)  # past the #31656 fix
    entries = ENGINES["llamacpp"].playbook()
    with active(context):
        line = _with_elsewhere(nothing, entries, situation, "ttft").elsewhere
    assert line.startswith("for throughput and TPOT, `llamacpp-mtp-speculation`: req/s +45%")
    assert "TPOT p50 -39.6%" in line
    assert "worsens TTFT: TTFT p50 +15%" in line


def test_moe_cache_cpu_share_is_admitted_only_from_the_build_that_lists_it():
    from tensward.engines.llamacpp.flags import allowed, flag_for_env

    assert flag_for_env("LLAMA_ARG_MOE_CACHE_CPU_SHARE") is None  # b12359 names the wrong env
    assert allowed("b12246", "--moe-cache-cpu-share")
    assert not allowed("b11981", "--moe-cache-cpu-share")
    assert allowed("b11669", "--moe-cache-mib")


WB1_START = """\
0.05.000.000 I load_tensors: offloaded 41/41 layers to GPU
0.05.590.661 I srv    load_model: initializing, n_slots = {}, n_ctx_slot = 8192
"""


GIB = 2**30


def _cache_offer(tmp_path, name, *, experts_gib, layers, host_gib, room_mib, tpot, share,
                 census=None, free_gib=199.0, types=("Q2_K",), experts=288, slots=1,
                 family="glm5-next", mtp_gib=0.0):  # fmt: skip
    """expert_cache_applies and its low bar on one launch: ``layers`` MoE layers of equal size
    behind one dense layer, and an unloaded next-token block of ``mtp_gib`` experts."""
    from tensward.engines.llamacpp.predicates import expert_cache_applies, expert_cache_low_bar

    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "server.log").write_text(WB1_START.format(slots))
    placement = {
        "regime": {"placement": "expert_offload"},
        "placement": {"host_weight_bytes": int(host_gib * GIB),
                      "host_expert_bytes": int(host_gib * GIB),
                      "device_weight_bytes": int((experts_gib - host_gib + 6) * GIB),
                      "host_layers": 0, "host_expert_layers": 14},
        "fit": {"device_free": 80_000 * MIB, "device_need": (80_000 - room_mib) * MIB},
        "environment": {"mem_gib": 230.0, "mem_available_gib": free_gib},
    }  # fmt: skip
    rate, pageable = census or (None, None)
    facts = {"run.host_time_share": share, "census.h2d_gb_per_s": rate,
             "census.pageable_h2d_share": pageable, "gpu.pcie_gbs_theoretical": 31.504}  # fmt: skip
    mtp = (int(mtp_gib * GIB),) if mtp_gib else ()
    anatomy = SimpleNamespace(
        expert_layer_bytes=(0,) + (int(experts_gib * GIB / layers),) * layers + mtp,
        experts=experts, experts_used=8, precisions={"routed_experts": types},
        model_type=family, mtp_layers=len(mtp),
    )  # fmt: skip
    args = {"--n-gpu-layers": layers + 2, "--n-cpu-moe": 14, "--load-mode": "none"}
    situation = SimpleNamespace(
        settings=Settings(extra_args=args),
        measurement=Measurement(16, 0, engine_version="b12603", tpot_p50_ms=tpot),
    )
    context = AnalysisContext(
        engine="llamacpp", new_run=run_dir, placement=placement, kb_facts=facts,
        probes={"pcie": {"gen": 4, "width": 16, "theoretical_gbs": 31.504}}, anatomy=anatomy,
    )  # fmt: skip
    engine = ENGINES["llamacpp"]
    with active(context):
        return expert_cache_applies(situation, engine), expert_cache_low_bar(situation, engine)


def test_expert_cache_is_offered_where_its_predicted_upload_beats_the_host_part(tmp_path):
    from tensward.playbook import HeldBack

    # GLM-5.3-Flash Q2 on one H100 after --load-mode none (GLM2 run 0002): try first, sized
    # without the unloaded MTP block
    glm2 = dict(experts_gib=94.68, layers=45, host_gib=24.87, room_mib=1729, tpot=31.11,
                share=0.417, census=(27.46, 0.0), mtp_gib=2.44)  # fmt: skip
    sure, low = _cache_offer(tmp_path, "q2", **glm2)
    assert "an expert cache of 73,074 MiB" in sure and "as --cpu-moe, with --load-mode none" in sure
    assert "fitted on GLM-5.3-Flash, this model's family" in sure and low is None
    assert "research, not measured by Tensward: against the launch without a cache" in sure
    # GLM-5.3-Flash Q4 (GLM2 run 0002): its own family's curve says no, which measured a loss
    glm4 = dict(experts_gib=176.8, layers=45, host_gib=111.3, room_mib=4696, tpot=47.72,
                share=0.909, census=(27.33, 0.0), free_gib=111.9)  # fmt: skip
    sure, low = _cache_offer(tmp_path, "q4", **glm4)
    assert isinstance(sure, HeldBack) and "would not pay" in sure and low is None
    assert "this model's family, predicts 1," in sure and "against 43.4 ms" in sure
    # Qwen3.5-35B-A3B Q4_K_M at a 16 GB budget on an H100 (CM1, x1.199): its family's curve pays
    cm1 = dict(experts_gib=18560 / 1024, layers=40, host_gib=15 * 464 / 1024, room_mib=2287,
               tpot=7.79, share=0.299, census=(23.1, 0.0), experts=256, types=("Q4_K",),
               family="qwen35moe")  # fmt: skip
    sure, low = _cache_offer(tmp_path, "cm1", **cm1)
    assert "fitted on Qwen3.5-35B-A3B, this model's family" in sure and low is None
    # a family with no fitted curve, where the two disagree: unsure, with both predictions
    sure, low = _cache_offer(tmp_path, "other", **{**cm1, "family": "qwen3moe", "share": 0.15})
    assert sure is None and low.count("MiB uploaded a token") == 1
    assert "unsure: no miss curve is fitted on this model's family" in low
    assert "fitted on GLM-5.3-Flash predicts" in low and "Qwen3.5-35B-A3B" in low
    # no prediction pays: the numbers are on screen, never a silent drop
    sure, _ = _cache_offer(tmp_path, "tiny", **{**glm2, "share": 0.05})
    assert isinstance(sure, HeldBack) and "would not pay" in sure and "MiB uploaded" in sure
    # no pinned census rate: 73% of the link's theoretical rate
    sure, _ = _cache_offer(tmp_path, "mapped", **{**glm2, "census": (9.5, 0.98)})
    assert "73% of the gen4 x16 link's theoretical 31.5 GB/s" in sure
    # 8-bit experts under the measured minimum: held back with the numbers
    sure, _ = _cache_offer(tmp_path, "q8", **{**glm2, "types": ("Q8_0",), "experts_gib": 18.0,
                                              "host_gib": 4.0})  # fmt: skip
    assert isinstance(sure, HeldBack) and "under the 17,900 MiB" in sure
    assert _cache_offer(tmp_path, "four", **{**glm2, "slots": 4}) == (None, None)


def test_inkling_is_never_offered_experts_on_the_host(monkeypatch):
    manifest = SimpleNamespace(missing=lambda flags: [])
    monkeypatch.setattr(predicates, "known_build", lambda: (None, manifest))

    def anatomy(expert_type):
        return SimpleNamespace(
            model_type="inkling2_loop", precisions={"routed_experts": (expert_type,)}
        )

    for name in (
        "buy-gpu-room", "offload-experts", "load-mode-none", "pin-host-weights",
        "expert-cache-to-free-vram",
    ):  # fmt: skip
        with active(AnalysisContext(anatomy=anatomy("IQ4_NL"))):
            assert "llama.cpp #32710" in predicates.HOLD_BACKS[name](SimpleNamespace())
        with active(AnalysisContext(anatomy=anatomy("Q8_0"))):
            assert predicates._no_host_experts(SimpleNamespace()) is None


def test_host_expert_guard_covers_new_builds_partial_ngl_and_gpu_override_tensor():
    from tensward.families import host_expert_flags, host_experts_unsafe

    iq4 = SimpleNamespace(model_type="inkling2_loop", precisions={"routed_experts": ("IQ4_NL",)})
    assert host_experts_unsafe(iq4, 12700)
    assert host_expert_flags({"--n-gpu-layers": "10"}, 24)
    assert not host_expert_flags({"--n-gpu-layers": "all"}, 24)
    assert not host_expert_flags({"--override-tensor": "exps=CUDA0"}, 24)
    assert host_expert_flags({"--override-tensor": "exps=CPU"}, 24)
    assert not host_expert_flags({"--cpu-moe": True, "--ubatch-size": "32"}, 24)


def test_inkling_warning_wording_budget_and_ubatch_advice():
    from tensward.families import (
        HOST_EXPERT_WARNING,
        reasoning_budget_set,
        ubatch_advice,
    )

    assert "coherent" in HOST_EXPERT_WARNING and "not identical" in HOST_EXPERT_WARNING
    assert "correct output" not in HOST_EXPERT_WARNING
    assert "micro-batch of 32 or less" in HOST_EXPERT_WARNING
    inkling = SimpleNamespace(model_type="inkling2_loop")
    assert reasoning_budget_set({"--reasoning-budget": "0"})
    assert not reasoning_budget_set({"--reasoning-budget": "512"})
    assert "-ub 256" in ubatch_advice(inkling, {})
    assert ubatch_advice(inkling, {"--ubatch-size": "256"}) is None
    assert ubatch_advice(SimpleNamespace(model_type="qwen3"), {}) is None


# Track C's RD1 arms (b12246): measured requests, the servers' minor faults over the load, C's
# faults per request and AnonHugePages (kB) at the end [run/C1, run/C2 box/results/*/arm.json].
RD1_ARMS = {
    "base.C1-RD1-chat27.r1": (64, 5599856, 87497.8, 0),
    "base.C1-RD1-chat27.r2": (64, 5557386, 86834.2, 0),
    "hugetlb.C1-RD1-chat27.r1": (64, 83388, 1302.9, 9998336),
    "base.C1-RD1-glm9.r1": (128, 587644, 4591.0, 0),
    "base.C1-RD1-glm9.r2": (128, 617593, 4824.9, 0),
    "hugetlb.C1-RD1-glm9.r2": (128, 24039, 187.8, 2449408),
    "base.C2-RD1a.r1": (192, 25745711, 134092.2, 0),
    "hugetlb.C2-RD1a.r3": (192, 383235, 1996.0, 16445440),
    "base.C2-RD1b.r2": (128, 2273120, 17758.8, 0),
}


def _proc_window(faults, hugepages, start=1_000_000):
    from tensward.engines.protocol import HostWindow
    from tensward.platforms.protocol import ProcessSample

    def sample(minor, huge):
        return ProcessSample(0, 1.0, 0.1, 100.0, 0, None, minor_faults=minor,
                             anon_hugepages_kib=huge)  # fmt: skip

    return HostWindow(None, sample(start, 0), sample(start + faults, hugepages))


def test_reading_rd1_matches_track_c_on_its_recorded_inputs():
    import os

    from tensward.engines.llamacpp.engine import host_faults
    from tensward.engines.llamacpp.logreader import fault_facts
    from tensward.platforms.cpu import CPU

    for arm, (requests, faults, per_request, hugepages) in RD1_ARMS.items():
        block = {"host_faults": host_faults(_proc_window(faults, hugepages))}
        found = fault_facts(block, requests)
        assert abs(found["run.minor_faults_per_request"] / (faults / requests) - 1) < 1e-6, arm
        assert round(found["run.minor_faults_per_request"], 1) == per_request, arm
        assert found["run.anon_hugepages_kib"] == hugepages, arm
    window = _proc_window(5599856, 0)
    for unread in (
        replace(window, start=None),  # a container: no host pid to read
        replace(window, end=replace(window.end, minor_faults=None)),  # macOS has no minflt
    ):
        assert host_faults(unread) is None and fault_facts({"host_faults": None}, 64) == {}
    assert fault_facts({"host_faults": host_faults(window)}, 0) == {}
    here = CPU.sample_process(os.getpid())
    with open(f"/proc/{os.getpid()}/stat") as stat:
        minflt = int(stat.read().rsplit(")", 1)[1].split()[7])  # C's direct.py read
    assert 0 < here.minor_faults <= minflt and here.anon_hugepages_kib is not None
    assert CPU.sample_process(os.getpid(), hugepages=False).anon_hugepages_kib is None


def test_hugetlb_state_save_is_offered_in_its_situation(tmp_path, monkeypatch):
    from tensward.engines.llamacpp.command import BuildIdentity
    from tensward.engines.llamacpp.engine import host_faults
    from tensward.suggest import suggestion_command

    engine = ENGINES["llamacpp"]
    (entry,) = (e for e in engine.playbook() if e.name == "hugetlb-state-save")
    assert not entry.changes_answers and entry.answer_risk == "none"
    (tmp_path / "server.log").write_text(LFM2_START)
    build = [12246]
    libc, thp = [(2, 39)], ["madvise"]
    monkeypatch.setattr(predicates, "glibc_version", lambda: libc[0])
    monkeypatch.setattr(predicates, "transparent_hugepages", lambda: thp[0])
    monkeypatch.setattr(
        predicates,
        "known_build",
        lambda: (BuildIdentity(build[0], "", "", ("CUDA",)), HelpManifest(frozenset())),
    )

    def offer(arm, settings=Settings(), platform="nvidia"):
        requests, faults, _, hugepages = RD1_ARMS[arm]
        placement = {
            "environment": {"platform": platform, "os_family": "linux"},
            "host_faults": host_faults(_proc_window(faults, hugepages)),
        }
        situation = SimpleNamespace(settings=settings, measurement=Measurement(requests, 0))
        with active(AnalysisContext(engine="llamacpp", new_run=tmp_path, placement=placement)):
            return entry.applies(situation), situation

    for arm in ("base.C2-RD1a.r1", "base.C1-RD1-chat27.r1", "base.C2-RD1b.r2"):
        reason, situation = offer(arm)
        assert reason and "GLIBC_TUNABLES=glibc.malloc.hugetlb=1" in reason, arm
    assert "134,092 minor page faults per request" in offer("base.C2-RD1a.r1")[0]
    for arm in ("base.C1-RD1-glm9.r1", "base.C1-RD1-glm9.r2", "hugetlb.C2-RD1a.r3"):
        assert offer(arm)[0] is None, arm
    assert offer("base.C2-RD1a.r1", platform="apple")[0] is None
    with active(AnalysisContext(engine="llamacpp", new_run=tmp_path)):
        settings = entry.apply(situation)
        command = suggestion_command(engine, tmp_path, Settings(), settings)
    assert command.endswith("--engine-env GLIBC_TUNABLES=glibc.malloc.hugetlb=1")
    assert offer("base.C2-RD1a.r1", settings)[0] is None
    build[0] = 11982
    assert offer("base.C2-RD1a.r1")[0] is None
    build[0] = 13000
    assert offer("base.C2-RD1a.r1")[0] is None
    build[0] = 12246
    libc[0] = (2, 34)
    assert offer("base.C2-RD1a.r1")[0] is None
    libc[0] = (2, 39)
    thp[0] = "never"
    assert offer("base.C2-RD1a.r1")[0] is None


# The base runs' facts of the discovery items on llama.cpp b12246, as Tensward 0.4.3 recorded
# them in each baseline's facts.json.
QWEN38_27B_GGUF = {
    "engine.build": 12246, "model.kind": "hybrid", "model.family": "qwen35",
    "model.params_b": 27.320697856, "model.active_params_b": 27.320697856,
    "model.quant": "gguf:Q4_K_M",
}  # fmt: skip
D1_FACTS = {
    **QWEN38_27B_GGUF, "gpu.arch": "hopper", "gpu.model": "H100 SXM", "run.slots": 16,
    "workload.clients": 16, "workload.tools": True, "workload.prompt_tokens_max": 785,
    "workload.output_tokens_p50": 38, "workload.output_tokens_limit": 128,
}  # fmt: skip
D1_BASE = (
    "llama-server -m /models/Qwen3.8-27B-UD-Q4_K_M.gguf --ctx-size 65536 --parallel 16 --jinja "
    "--n-gpu-layers all --fit off --metrics --reasoning off"
)


def _offered(name, facts, command):
    from tensward.engines.llamacpp.command import parse_setup
    from tensward.facts import Facts
    from tensward.suggest import suggestion_command

    engine = ENGINES["llamacpp"]
    (entry,) = (e for e in engine.playbook() if e.name == name)
    settings = parse_setup(command).settings
    situation = SimpleNamespace(settings=settings)
    with active(AnalysisContext(engine="llamacpp", kb_facts=Facts(facts))):
        reason = entry.applies(situation)
        changed = entry.apply(situation) if reason else None
    command = changed and suggestion_command(engine, Path("p"), settings, changed)
    return entry, reason, command


def test_state_copy_off_hybrid_hopper_tools_b12246_is_offered_in_its_situation():
    name = "llamacpp-state-save-off-tools"
    entry, reason, command = _offered(name, D1_FACTS, D1_BASE)
    assert entry.changes_answers and entry.answer_risk == "low"
    assert "--cache-ram 0 --ctx-checkpoints 0" in reason
    assert command.endswith("--engine-arg cache-ram=0 --engine-arg ctx-checkpoints=0")
    for other in (
        {"gpu.arch": "ada"}, {"run.slots": 8}, {"workload.tools": False},
        {"workload.prompt_tokens_max": 801}, {"engine.build": 11669},
        {"model.family": "nemotron_h"},
    ):  # fmt: skip
        assert _offered(name, D1_FACTS | other, D1_BASE)[1] is None, other
    assert _offered(name, D1_FACTS, D1_BASE + " --ctx-checkpoints 1")[1] is None


D2_FACTS = {
    **QWEN38_27B_GGUF, "gpu.arch": "ada", "gpu.model": "L4", "run.slots": 8,
    "workload.clients": 8, "workload.tools": False, "workload.prompt_tokens_max": 113,
    "workload.output_tokens_p50": 3, "workload.output_tokens_limit": 64,
}  # fmt: skip
D2_BASE = D1_BASE.replace("65536 --parallel 16", "32768 --parallel 8")


def test_state_copy_off_hybrid_short_labelled_ada_b12246_is_offered_in_its_situation():
    name = "llamacpp-state-save-off-short"
    entry, reason, command = _offered(name, D2_FACTS, D2_BASE)
    assert entry.effects["tpot"].direction == "worsens" and entry.answer_risk == "low"
    assert "--cache-ram 0 --ctx-checkpoints 0" in reason
    assert command.endswith("--engine-arg cache-ram=0 --engine-arg ctx-checkpoints=0")
    for other in (
        {"gpu.arch": "hopper"}, {"run.slots": 16}, {"workload.tools": True},
        {"workload.output_tokens_p50": 38}, {"workload.prompt_tokens_max": 785},
        {"engine.build": 12441},
    ):  # fmt: skip
        assert _offered(name, D2_FACTS | other, D2_BASE)[1] is None, other
    assert _offered("llamacpp-state-save-off-tools", D2_FACTS, D2_BASE)[1] is None
    assert _offered(name, D1_FACTS, D1_BASE)[1] is None


S5_FACTS = {
    "engine.build": 12246, "model.kind": "dense", "model.family": "gemma4",
    "model.params_b": 7.51806929, "model.quant": "gguf:Q4_K_M", "gpu.arch": "hopper",
    "gpu.model": "H100 SXM", "run.slots": 32, "workload.clients": 32, "workload.tools": False,
    "workload.prompt_tokens_max": 56, "workload.output_tokens_p50": 256,
    "workload.output_tokens_limit": 256,
}  # fmt: skip
S5_BASE = D1_BASE.replace("Qwen3.8-27B-UD-Q4_K_M", "gemma-4-E4B-it-Q4_K_M").replace(
    "65536 --parallel 16", "131072 --parallel 32"
)


def test_backend_sampling_gemma4_chat_hopper_b12246_is_offered_in_its_situation():
    name = "llamacpp-backend-sampling-gemma4"
    entry, reason, command = _offered(name, S5_FACTS, S5_BASE)
    assert entry.effects["ttft"].direction == "worsens" and entry.answer_risk == "low"
    assert "--backend-sampling samples on the GPU" in reason
    assert command.endswith("--engine-arg backend-sampling")
    for other in (
        {"gpu.arch": "ada"}, {"run.slots": 16}, {"workload.tools": True},
        {"workload.output_tokens_p50": 38}, {"model.family": "qwen35"}, {"engine.build": 12441},
        {"workload.prompt_tokens_max": 129},
    ):  # fmt: skip
        assert _offered(name, S5_FACTS | other, S5_BASE)[1] is None, other
    assert _offered(name, S5_FACTS, S5_BASE + " --backend-sampling")[1] is None
    assert _offered(name, S5_FACTS, S5_BASE + " --cache-ram 0 --ctx-checkpoints 0")[1] is None


D5_FACTS = {
    **QWEN38_27B_GGUF, "gpu.arch": "hopper", "gpu.model": "H100 SXM", "run.slots": 32,
    "workload.clients": 32, "workload.tools": False, "workload.prompt_tokens_max": 59,
    "workload.output_tokens_p50": 256, "workload.output_tokens_limit": 256,
}  # fmt: skip
D5_BASE = (
    D1_BASE.replace("65536 --parallel 16", "131072 --parallel 32")
    + " --cache-ram 0 --ctx-checkpoints 0"
)


def test_backend_sampling_hybrid_state_off_chat_hopper_b12246_is_offered_in_its_situation():
    name = "llamacpp-backend-sampling-hybrid"
    entry, reason, command = _offered(name, D5_FACTS, D5_BASE)
    assert entry.effects["ttft"].direction == "worsens" and entry.answer_risk == "low"
    assert "--backend-sampling samples on the GPU" in reason
    assert command.endswith("--engine-arg backend-sampling")
    for other in (
        {"gpu.arch": "ada"}, {"run.slots": 16}, {"workload.tools": True},
        {"model.family": "gemma4"}, {"engine.build": 11669}, {"workload.prompt_tokens_max": 129},
    ):  # fmt: skip
        assert _offered(name, D5_FACTS | other, D5_BASE)[1] is None, other
    without_state_flags = D5_BASE.removesuffix(" --cache-ram 0 --ctx-checkpoints 0")
    assert _offered(name, D5_FACTS, without_state_flags)[1] is None
    assert _offered(name, D5_FACTS, D5_BASE + " --backend-sampling")[1] is None
    assert _offered("llamacpp-backend-sampling-gemma4", D5_FACTS, D5_BASE)[1] is None


def test_a_bundle_carries_the_env_of_its_members():
    from tensward.suggest import _composed

    engine = ENGINES["llamacpp"]
    parent = Settings(max_concurrent_requests=2)
    member = SimpleNamespace(
        settings=replace(parent, extra_env={"GLIBC_TUNABLES": "glibc.malloc.hugetlb=1"})
    )
    other = SimpleNamespace(settings=replace(parent, extra_args={"--ubatch-size": 2048}))
    composed = _composed(parent, [other, member], engine)
    assert composed.extra_env == {"GLIBC_TUNABLES": "glibc.malloc.hugetlb=1"}
    assert composed.extra_args["--ubatch-size"] == "2048"


def test_a_model_with_experts_on_the_host_is_not_pointed_at_a_higher_bit_file():
    from tensward.engines.llamacpp.advice import advice

    facts = SimpleNamespace(quantization="gguf:Q4_K_M", prompts=())

    def drift(placement):
        block = {"regime": {"placement": placement, "compute": "mixed"}}
        found = advice(block, None, facts)
        return next(a for a in found if a["key"] == "quant_drift")

    assert "a higher-bit file (Q6_K, Q8_0)" in drift("gpu_only")["terminal"]
    host = drift("expert_offload")
    assert "speed-for-quality trade; judge it per solved task" in host["terminal"]
    assert "no smaller file is safe" in host["text"] and "at least 90%" in host["text"]
    assert "per solved task" in drift("dense_offload")["terminal"]


CENSUS_OFFERS = Path(__file__).parent / "fixtures" / "census-offers"
CENSUS_SNAPSHOTS = Path(__file__).parent / "census_snapshots"


def _recorded(run):
    """The facts Tensward 0.4.3 recorded for a base run of the discovery legs."""
    return json.loads((CENSUS_OFFERS / f"{run}.facts.json").read_text())


def _census(name):
    """The census readings of a recorded census window, as the census start computes them."""
    from tensward.census.readings import ReadingContext, compute_readings
    from tensward.census.snapshot import parse_snapshot, read_timeline, window

    folder = CENSUS_SNAPSHOTS / name
    meta = json.loads((folder / "meta.json").read_text())
    before, after = (
        {int(p): parse_snapshot(e) for p, e in json.loads((folder / t).read_text()).items()}
        for t in ("before.json", "after.json")
    )
    start, end = meta["window_start_ns"], meta["window_end_ns"]
    timeline = read_timeline(folder / "timeline.jsonl", start, end)
    research = meta["research"]
    win = window(
        before, after, requests=meta["requests"], start_ns=start, end_ns=end,
        timelines={meta["pid"]: timeline}, out_tokens=research.get("out_tokens", 0),
    )  # fmt: skip
    ctx = ReadingContext("llamacpp", meta["gpu"], None, True, version=meta["version"],
                         n_vocab=research.get("n_vocab"))  # fmt: skip
    return compute_readings(win, ctx)


def _goal_offered(name, facts, command, goal="throughput"):
    from tensward.engines.llamacpp.command import parse_setup
    from tensward.facts import Facts

    engine = ENGINES["llamacpp"]
    (entry,) = (e for e in engine.playbook() if e.name == name)
    situation = SimpleNamespace(settings=parse_setup(command).settings)
    with active(AnalysisContext(engine="llamacpp", kb_facts=Facts(facts), goal=goal)):
        return entry.applies(situation)


SAVES = "llamacpp-state-save-off-gpu-waits"
SAVES_OFF = " --cache-ram 0 --ctx-checkpoints 0"


def test_state_save_off_gpu_waits_is_offered_in_its_situation():
    # C2-D1: H100, Qwen3.8-27B tools at 16 slots; its census window (EA1) read busy 0.444
    facts = _recorded("c2-d1") | _census("ea1-tool27-def-b12246-h100")
    entry, reason, command = _offered(SAVES, facts, D1_BASE)
    assert reason.startswith("the GPU census found the GPU running kernels only 44% of the time")
    assert command.endswith("--engine-arg cache-ram=0 --engine-arg ctx-checkpoints=0")
    assert entry.effects["ttft"].direction == "improves" and entry.helps == {
        "throughput",
        "ttft",
        "tpot",
    }
    assert _offered("llamacpp-state-save-off-tools", facts, D1_BASE)[1] is None
    pinned = _goal_offered("llamacpp-state-save-off-tools", facts, D1_BASE, goal="ttft")
    assert pinned is not None
    assert _offered(SAVES, facts, D1_BASE + " --ctx-checkpoints 1")[1] is None
    assert _offered("llamacpp-state-save-off-tools", _recorded("c2-d1"), D1_BASE)[1]
    # its cost in this run's numbers, and never beside --slot-prompt-similarity 0 (research 469)
    longest = facts["workload.prompt_tokens_max"]
    cost = f"every returning request re-reads its whole prompt (up to {longest:,} tokens)"
    assert reason.endswith(cost)
    from tensward.facts import Facts

    measured = SimpleNamespace(measurement=SimpleNamespace(prompt_capacity_tok_s=longest / 2))
    with active(AnalysisContext(engine="llamacpp", kb_facts=Facts(facts))):
        assert predicates._rereads(measured).startswith(  # type: ignore[arg-type]
            "every returning request re-reads its whole prompt: about 2.0 s"
        )
    assert _offered(SAVES, facts, D1_BASE + " --slot-prompt-similarity 0")[1] is None
    from tensward.kb_entries import _apart

    assert _apart({"--slot-prompt-similarity": "0", "--ctx-checkpoints": "0"})


def test_the_longest_prompt_counts_its_cached_tokens():
    """O1's prompt gate reads the whole prompt a returning request sends, not the part computed
    after the cache (L35: 681 tokens, 566 from the cache)."""
    import asyncio

    from tensward.capture import _decode_capture
    from tensward.measurement import Peaks, Window, summarize
    from tensward.slo import Slo

    usage = {"prompt_tokens": 681, "prompt_tokens_details": {"cached_tokens": 566}}
    frame = {"choices": [], "usage": usage, "timings": {"cache_n": 566, "prompt_n": 115}}
    data = f"data: {json.dumps(frame)}\n\ndata: [DONE]\n\n".encode()
    captured = asyncio.run(_decode_capture(data, 200))
    assert (captured.prompt_tokens, captured.cached_tokens) == (681, 566)
    measured = summarize(
        [], window=Window("whole_run", 0, 1, 1.0), slo=Slo(200.0, 50.0), request_prompt_tokens={},
        request_images={}, peaks=Peaks(), preemptions=None, prefix_cache_hits=566,
        prefix_cache_queries=681, prompt_tokens=[640, captured.prompt_tokens],
        max_context_len=8192, too_long=(), quant_kernels=(), tool_calls=None,
        structured_answers=None,
    )  # fmt: skip
    assert measured.max_prompt_tokens == 681


def test_state_save_off_gpu_waits_not_offered_where_the_gpu_is_busy_or_outside_its_scope():
    # the L4 box of C1: its D2 facts with the census of its 27B chat at 8 slots (busy 0.880)
    busy = _recorded("c1-d2") | _census("c1-chat27-base-b12246-l4")
    assert busy["census.busy_share"] > 0.70
    assert _offered(SAVES, busy, D2_BASE)[1] is None
    assert _offered("llamacpp-state-save-off-short", busy, D2_BASE)[1]
    idle = _recorded("c2-d1") | _census("ea1-tool27-def-b12246-h100")
    for outside in (
        {"workload.clients": 4, "workload.prompt_tokens_max": 4096},  # N6: returning conv4
        {"workload.clients": 4}, {"workload.prompt_tokens_max": 2049},
        {"census.graphs_off": False}, {"anatomy.linear_attention": False},
        {"run.placement": "expert_offload"}, {"engine.build": 13000}, {"gpu.arch": "ampere"},
    ):  # fmt: skip
        assert _offered(SAVES, idle | outside, D1_BASE)[1] is None, outside
    unread = {k: v for k, v in idle.items() if not k.startswith("census.")}
    assert _offered(SAVES, unread, D1_BASE)[1] is None


def _kb_record(path):
    import sys
    import tomllib

    kb = Path(__file__).resolve().parents[3] / "tools" / "kb"
    if not kb.is_dir():
        pytest.skip("reads a record of the private knowledge base")
    sys.path.insert(0, str(kb))
    import store

    found = Path(store.KB_DIR) / "records" / "llamacpp" / f"{path}.toml"
    return store.to_record(tomllib.loads(found.read_text("utf-8")))


def _census_block(record, facts):
    from tensward.engines.llamacpp.predicates import PREDICATES
    from tensward.facts import Facts
    from tensward.kb_entries import kb_block
    from tensward.matcher import Layers, annotate_offered, match

    entry = record.oss_entry
    layers = Layers(playbook=frozenset({entry}), source=None, source_status={}, local=(record,))
    facts = Facts(facts | {"engine.name": "llamacpp", "engine.version": "12246"})
    found = match(
        facts, layers, engine_family="llamacpp", engine_version="12246",
        tensward_version="0.5.0", rules=frozenset(PREDICATES), levers=frozenset(),
    )  # fmt: skip
    offered = annotate_offered(found, facts, layers, {entry})
    match_ = offered.by_entry(entry)
    return match_ and kb_block(match_, offered), found


def test_the_state_save_offer_carries_the_census_facts_it_fired_on():
    record = _kb_record("host/state-save-off-gpu-waits")
    facts = _recorded("c2-d1") | _census("ea1-tool27-def-b12246-h100")
    block, found = _census_block(record, facts)
    assert block["record"] == f"{record.id}@1"
    assert block["census_facts"] == {
        "census.graphs_off": True, "census.busy_share": facts["census.busy_share"],
    }  # fmt: skip
    unread = {k: v for k, v in facts.items() if not k.startswith("census.")}
    found = _census_block(record, unread)[1]
    assert found.census_readers == 1
    assert found.census_unknown == ((f"{record.id}@1", ("census.graphs_off", "census.busy_share")),)


SAMPLER = "llamacpp-backend-sampling-gpu-waits"


def test_backend_sampling_gpu_waits_is_offered_in_its_situation():
    # C2-RD2 (D5): H100, Qwen3.8-27B chat at 32 slots, saves off; its census read busy 0.646
    # and 1.00 rows of logits per output token
    facts = _recorded("c2-rd2") | _census("c2-rd2-base-b12246-h100")
    entry, reason, command = _offered(SAMPLER, facts, D5_BASE)
    assert reason.startswith("the GPU census found the GPU running kernels only 65% of the time")
    assert "1.00 rows of logits per output token" in reason
    assert command.endswith("--engine-arg backend-sampling")
    assert entry.effects["ttft"].direction == "worsens" and entry.helps == {"throughput", "tpot"}
    assert _offered("llamacpp-backend-sampling-hybrid", facts, D5_BASE)[1] is None
    sixteen = facts | {"run.slots": 16, "workload.clients": 16}  # F3: no pin reaches it
    assert _offered(SAMPLER, sixteen, D5_BASE.replace("--parallel 32", "--parallel 16"))[1]
    gemma = _recorded("x-h-s5") | _census("c2-rd2-base-b12246-h100")
    assert _offered(SAMPLER, gemma, S5_BASE)[1]  # not linear attention: saves on is no block
    assert _offered("llamacpp-backend-sampling-gemma4", gemma, S5_BASE)[1] is None


def test_backend_sampling_gpu_waits_not_offered_where_the_gpu_is_busy_or_saves_fill_the_copy():
    l4 = {"gpu.arch": "ada", "gpu.model": "L4", "run.slots": 8, "workload.clients": 8}
    # N2: the C1 box's 27B chat at 8 slots, busy 0.880 and 2.52 rows per token
    for census in ("c1-chat27-base-b12246-l4", "c1-glm9-base-b12246-l4"):
        busy = _recorded("c2-rd2") | l4 | _census(census)
        assert _offered(SAMPLER, busy, D5_BASE)[1] is None, census
    sampled = _recorded("c2-rd2") | _census("c2-rd2-bs-b12246-h100")
    assert _offered(SAMPLER, sampled, D5_BASE)[1] is None
    idle = _recorded("c2-rd2") | _census("c2-rd2-base-b12246-h100")
    assert _offered(SAMPLER, idle, D5_BASE.removesuffix(SAVES_OFF))[1] is None  # O1 goes first
    half = D5_BASE.removesuffix(SAVES_OFF) + " --ctx-checkpoints 0"  # the prompt cache still on
    assert _offered(SAMPLER, idle, half)[1] is None
    assert _offered(SAMPLER, idle, D5_BASE + " --backend-sampling")[1] is None
    for outside in (
        {"workload.tools": True}, {"workload.output_tokens_p50": 63},
        {"census.graphs_off": False}, {"census.busy_share": 0.80},
        {"census.logits_on_host_share": 1.51}, {"census.logits_on_host_share": 0.89},
        {"run.placement": "expert_offload"}, {"engine.build": 11669},
    ):  # fmt: skip
        assert _offered(SAMPLER, idle | outside, D5_BASE)[1] is None, outside


TRADE = "llamacpp-raise-ubatch-expert-trade"
P2_BASE = (
    "llama-server -m /models/North-Mini-Code-1.0-Q4_K_M.gguf --ctx-size 6912 --parallel 4 "
    "--n-gpu-layers 50 --n-cpu-moe 12"
)
# North Mini Code Q4_K_M: 49 blocks, 128 experts, 8 a token; the run's 12 host blocks of experts
# hold 4,100,456,448 bytes, 341,704,704 each
NORTH_MINI = SimpleNamespace(
    model_type="cohere2moe", precisions={"routed_experts": ("Q4_K",)}, experts=128,
    experts_used=8, expert_layer_bytes=(341_704_704,) * 49,
)  # fmt: skip


def _p2_census():
    """The V100 P2 census window (tests/fixtures/census-p2-v100), with its run's host bytes."""
    from tensward.census.readings import ReadingContext, compute_readings
    from tensward.census.snapshot import parse_snapshot, read_timeline, window

    folder = Path(__file__).parent / "fixtures" / "census-p2-v100"
    saved = json.loads((folder / "after.json").read_text())
    after = {int(pid): parse_snapshot(entry) for pid, entry in saved.items()}
    start, end = 8945977000000, 9054201000000
    timeline = read_timeline(folder / "timeline.jsonl", start, end)
    win = window({}, after, requests=5, start_ns=start, end_ns=end, timelines={35384: timeline})
    ctx = ReadingContext("llamacpp", "Tesla V100-SXM2-16GB", None, True, version="12246",
                         host_weight_bytes=4100456448)  # fmt: skip
    return compute_readings(win, ctx)


def _trade(
    tmp_path, facts, command=P2_BASE, anatomy=NORTH_MINI, goal="ttft", name=TRADE,
    prompt=6500, cached=0.0, ram=None,
):  # fmt: skip
    from tensward.engines.llamacpp.command import parse_setup
    from tensward.facts import Facts

    engine = ENGINES["llamacpp"]
    (entry,) = (e for e in engine.playbook() if e.name == name)
    run_dir = tmp_path / str(len(list(tmp_path.iterdir())))  # the log is cached by its mtime
    run_dir.mkdir()
    (run_dir / "server.log").write_text((CENSUS_OFFERS / "p2-v100.server.log").read_text())
    placement = json.loads((CENSUS_OFFERS / "p2-v100.placement.json").read_text())
    if ram is not None:
        placement["environment"] |= {"mem_gib": ram[0], "mem_available_gib": ram[1]}
    prefill = SimpleNamespace(evidence="TTFT p50 15745 ms is 762x TPOT p50")
    situation = SimpleNamespace(
        settings=parse_setup(command).settings,
        measurement=Measurement(
            20, 0, engine_version="12246", max_prompt_tokens=prompt,
            prefix_cache_hit_rate=cached,
        ),
        fired=lambda name: name == "prefill", finding=lambda name: prefill,
    )  # fmt: skip
    context = AnalysisContext(
        engine="llamacpp", new_run=run_dir, placement=placement, kb_facts=Facts(facts),
        anatomy=anatomy, goal=goal,
    )  # fmt: skip
    with active(context):
        reason = entry.applies(situation)
        settings = entry.apply(situation) if reason else None
        blocked = predicates.raise_ubatch_blocked(situation, engine)
    return reason, settings, blocked


def test_raise_ubatch_expert_trade_is_offered_in_its_situation(tmp_path):
    # P2: V100 16 GB, North Mini Code Q4_K_M with 12 blocks of experts on the host, coding
    # prompts of 4k to 6.5k tokens; its census: 11.8 passes per request, copy wait 0.38
    facts = _recorded("p2-v100") | _p2_census()
    assert facts["census.pageable_h2d_share"] > 0.99  # the record: 99.3% of the H2D pageable
    reason, settings, blocked = _trade(tmp_path, facts)
    assert blocked.startswith("-ub 2048 would help but needs about 741 MiB more GPU memory")
    assert reason.startswith(
        "the GPU census saw the experts in host memory copied to the GPU 11.8 times per request, "
        "and copies with no kernel running filled 38% of the time"
    )
    assert "--n-cpu-moe 12 to 15 frees about 978 MiB" in reason
    assert (
        "the cost: decode reads the experts of 3 more layers from host memory, so TPOT rises: "
        "about +14% (estimated, not measured); it also loads the host experts with --load-mode "
        "none (pinned): it locks 4.8 GiB of host RAM while the server runs"
    ) in reason
    args = settings.extra_args
    assert (args["--ubatch-size"], args["--n-cpu-moe"]) == (2048, 15)
    assert args["--load-mode"] == "none"  # the experts were read through mmap
    # a host the lock would leave under the reserve: the trade runs without pinned loading
    reason, settings, _ = _trade(tmp_path, facts, ram=(16.0, 10.0))
    assert "--load-mode" not in settings.extra_args and "pinned" not in reason
    entry = next(e for e in ENGINES["llamacpp"].playbook() if e.name == TRADE)
    assert entry.effects["tpot"].direction == "worsens" and entry.helps == {"ttft"}


def test_raise_ubatch_expert_trade_not_offered_without_the_streaming_or_for_inkling(tmp_path):
    facts = _recorded("p2-v100") | _p2_census()
    # N5v: the same model, card and command with short chat prompts stream the experts once
    for census in ({"census.host_passes_per_request": 3.9}, {"census.copy_wait_share": 0.19},
                   {"census.graphs_off": False}):  # fmt: skip
        assert _trade(tmp_path, facts | census)[0] is None, census
    unread = {k: v for k, v in facts.items() if not k.startswith("census.")}
    assert _trade(tmp_path, unread)[0] is None
    assert _trade(tmp_path, facts | {"engine.build": 11669})[0] is None
    inkling = SimpleNamespace(**{**NORTH_MINI.__dict__, "model_type": "inkling2_loop"})
    assert _trade(tmp_path, facts, anatomy=inkling)[0] is None
    small = (341_704_704,) * 12 + (200_000_000,) * 37  # three more blocks free under 796 MiB
    smaller = SimpleNamespace(**{**NORTH_MINI.__dict__, "expert_layer_bytes": small})
    assert _trade(tmp_path, facts, anatomy=smaller)[0] is None


COPIES = "llamacpp-offload-ubatch-copies"


def test_offload_ubatch_copies_trades_on_agent_turn_readings_and_leaves_o3_its_case(tmp_path):
    # L35: agent turns at 4 clients read 0.43 passes, copy wait 0.24 and busy 0.24, under O3's
    # frozen 4 passes; the same trade is offered on the looser reading
    facts = _recorded("p2-v100") | _p2_census()
    agent = {"census.host_passes_per_request": 0.43, "census.copy_wait_share": 0.10}
    reason, settings, _ = _trade(tmp_path, facts | agent, name=COPIES)
    assert "0.43 times the experts kept in host memory" in reason
    assert "--n-cpu-moe 12 to 15 frees about 978 MiB" in reason and "research, not" in reason
    assert (settings.extra_args["--ubatch-size"], settings.extra_args["--n-cpu-moe"]) == (2048, 15)
    assert _trade(tmp_path, facts, name=COPIES)[0] is None  # O3's own case: O3 offers it
    for outside in ({"census.host_passes_per_request": 0.24}, {"census.busy_share": 0.6},
                    {"census.graphs_off": False}):  # fmt: skip
        assert _trade(tmp_path, facts | agent | outside, name=COPIES)[0] is None, outside
    # L35's real prompts: 681 tokens, 566 of them from the prompt cache, so 115 are computed and
    # never fill one 512-token micro-batch; -ub 2048 has nothing to remove
    for name, readings in ((COPIES, facts | agent), (TRADE, facts)):
        assert _trade(tmp_path, readings, name=name, prompt=681, cached=0.83)[0] is None, name


EIGHT_BIT = SimpleNamespace(**{**NORTH_MINI.__dict__, "precisions": {"routed_experts": ("Q8_0",)}})


def _cache(tmp_path, command, anatomy=EIGHT_BIT, name="llamacpp-expert-cache-share"):
    from tensward.engines.llamacpp.command import parse_setup

    (entry,) = (e for e in ENGINES["llamacpp"].playbook() if e.name == name)
    run_dir = tmp_path / str(len(list(tmp_path.iterdir())))
    run_dir.mkdir()
    (run_dir / "server.log").write_text((CENSUS_OFFERS / "p2-v100.server.log").read_text())
    placement = json.loads((CENSUS_OFFERS / "p2-v100.placement.json").read_text())
    situation = SimpleNamespace(
        settings=parse_setup(command).settings,
        measurement=Measurement(20, 0, engine_version="12246"),
    )
    context = AnalysisContext(
        engine="llamacpp", new_run=run_dir, placement=placement, anatomy=anatomy
    )
    with active(context):
        reason = entry.applies(situation)
        return reason, entry.apply(situation) if isinstance(reason, str) else None


def test_expert_cache_share_makes_room_for_its_cache_on_8_bit_experts(tmp_path):
    reason, settings = _cache(tmp_path, P2_BASE)
    assert "--moe-cache-cpu-share 0.95" in reason and "--n-cpu-moe 12 to 25" in reason
    assert "It locks" in reason
    args = settings.extra_args
    assert (args["--moe-cache-mib"], args["--moe-cache-cpu-share"]) == (4096, "0.95")
    assert (args["--n-cpu-moe"], args["--load-mode"]) == (25, "none")
    assert _cache(tmp_path, P2_BASE, anatomy=NORTH_MINI)[0] is None  # 4-bit experts
    assert _cache(tmp_path, P2_BASE + " --moe-cache-mib 2048")[0] is None


def test_the_other_share_is_measured_once_a_cache_is_set(tmp_path):
    other = "llamacpp-expert-cache-other-share"
    reason, settings = _cache(tmp_path, P2_BASE + " --moe-cache-mib 2048", name=other)
    assert "share 0.95" in reason and settings.extra_args["--moe-cache-cpu-share"] == "0.95"
    high = P2_BASE + " --moe-cache-mib 2048 --moe-cache-cpu-share 0.95"
    assert _cache(tmp_path, high, name=other)[0] is None  # share 0 at four slots: measured worse
    one = high.replace("--parallel 4", "--parallel 1")
    assert _cache(tmp_path, one, name=other)[1].extra_args["--moe-cache-cpu-share"] == "0"
    assert _cache(tmp_path, P2_BASE, name=other)[0] is None


DENSE_OFFLOAD_START = """\
0.00.256.182 I cmn  common_param:   - CUDA0   : NVIDIA L4 (22563 MiB, 22369 MiB free)
0.00.559.805 I common_params_fit_impl: filling dense layers back-to-front:
0.01.568.740 I common_params_fit_impl:   - CUDA0 (NVIDIA L4): 30 layers,  20900 MiB used, \
  1469 MiB free
0.01.991.773 I load_tensors: offloaded 31/41 layers to GPU
0.06.000.000 I llama_context: flash_attn    = enabled
0.06.630.292 I llama_kv_cache: size = 4000.00 MiB ( 32768 cells,  40 layers,  1/1 seqs), \
K (f16): 2000.00 MiB, V (f16): 2000.00 MiB
"""
MIB = 2**20
DENSE_40 = SimpleNamespace(
    model_type="qwen3", precisions={}, layers=40, layer_bytes=(400 * MIB,) * 40,
    expert_layer_bytes=(), q_heads=32, experts=0, experts_used=0,
    attention=(SimpleNamespace(window=None, kv_heads=8),),
)  # fmt: skip


def _dense_offer(
    tmp_path, monkeypatch, name, room_mib, peak_mib=21300, inflight=1, anatomy=DENSE_40
):
    """An offer of a 40-layer dense model with 10 layers on the host, ``room_mib`` of fit room
    and the card's peak at ``peak_mib`` of 22,563 MiB."""
    from tensward.engines.llamacpp.placement import DEVICE_MARGIN

    monkeypatch.setattr(predicates, "sampled_memory", lambda run_dir, m: (peak_mib, 23034))
    run_dir = tmp_path / str(len(list(tmp_path.iterdir())))
    run_dir.mkdir()
    (run_dir / "server.log").write_text(DENSE_OFFLOAD_START)
    free = 22369 * MIB
    need = free - room_mib * MIB
    placement = {
        "regime": {"placement": "dense_offload"},
        "placement": {
            "device_bytes_per_token": 30 * 400 * MIB, "device_weight_bytes": 30 * 400 * MIB,
            "host_bytes_per_token": 10 * 400 * MIB, "host_expert_bytes": 0,
            "host_expert_layers": 0, "host_layers": 10, "host_weight_bytes": 10 * 400 * MIB,
            "kv_on_host": False,
        },
        "fit": {
            "device_free": free, "device_need": need, "host_free": 40 * 2**30,
            "host_need": 4 * 2**30, "fits": True, "reason": "", "kv_bytes": 4000 * MIB,
            "compute_buffer_bytes": 500 * MIB, "pool_tokens": 32768, "ubatch": 512,
            "slots": 1, "need_tokens": 4096, "peak_running": 1,
        },
        "next_steps": [],
        "decode_model": {
            "bw_dev_gbs": 300.0, "bw_host_gbs": 50.0, "overhead_ms": 5.0, "concurrency": 1,
            "calibrated": True,
        },
    }  # fmt: skip
    assert need > DEVICE_MARGIN
    situation = SimpleNamespace(
        settings=Settings(), measurement=Measurement(10, 0, peak_in_flight=inflight),
        facts=SimpleNamespace(max_inflight=inflight), fired=lambda name: False,
    )  # fmt: skip
    engine = ENGINES["llamacpp"]
    (entry,) = (e for e in engine.playbook() if e.name == name)
    context = AnalysisContext(
        engine="llamacpp", new_run=run_dir, placement=placement, anatomy=anatomy, goal="tpot"
    )
    with active(context):
        reason = entry.applies(situation)
        return reason, entry.apply(situation) if reason else None


def test_kv_q8_is_offered_with_one_slot_when_it_brings_layers_back(tmp_path, monkeypatch):
    name = "llamacpp-kv-q8-for-layers"
    reason, settings = _dense_offer(tmp_path, monkeypatch, name, room_mib=200)
    assert reason.startswith("a q8_0 KV cache frees about 1,406 MiB, room for 3 more layers")
    assert "4 query heads per KV head" in reason
    assert settings.max_concurrent_requests == 1 and settings.kv_cache_dtype == "q8_0"
    assert settings.extra_args["--n-gpu-layers"] == 34
    assert _dense_offer(tmp_path, monkeypatch, name, room_mib=200, inflight=2)[0] is None
    grouped = SimpleNamespace(**{**DENSE_40.__dict__, "q_heads": 40})  # 5 per KV head
    assert _dense_offer(tmp_path, monkeypatch, name, room_mib=200, anatomy=grouped)[0] is None


def test_fit_target_256_is_offered_with_the_memory_it_leaves(tmp_path, monkeypatch):
    name = "llamacpp-fit-target-256"
    reason, settings = _dense_offer(tmp_path, monkeypatch, name, room_mib=300)
    assert "--fit-target 256 it places 2 more layers on the GPU" in reason
    assert "card memory left at the peak: about 263 MiB" in reason
    assert settings.extra_args["--n-gpu-layers"] == 33
    assert _dense_offer(tmp_path, monkeypatch, name, room_mib=300, peak_mib=21500)[0] is None


def test_the_report_tells_real_card_memory_and_thinking_without_an_answer(tmp_path):
    from tensward.engines.llamacpp.startup import read_startup
    from tensward.measure import thinking_no_answer
    from tensward.stands import stands_rows

    started = read_startup(DENSE_OFFLOAD_START)
    assert (started.device_total_bytes, started.fit_used_bytes) == (22563 * MIB, 20900 * MIB)
    records = [SimpleNamespace(request_id=str(i), outcome="success") for i in range(3)]
    responses = {
        "0": SimpleNamespace(reasoning="...", text="", tool_calls=[]),
        "1": SimpleNamespace(reasoning="...", text="42", tool_calls=[]),
        "2": SimpleNamespace(reasoning="", text="hi", tool_calls=[]),
    }
    assert thinking_no_answer(records, responses) == 1
    plain = {k: SimpleNamespace(reasoning="", text="a", tool_calls=[]) for k in responses}
    assert thinking_no_answer(records, plain) is None
    before = {"succeeded": 3, "thinking_no_answer": 1}
    after = {"succeeded": 3, "thinking_no_answer": 0}
    assert ("Ended in thinking, no answer", "1 of 3", "0 of 3", "") in stands_rows(before, after)


def test_pinned_loading_goes_before_the_census_ub_offers_when_copies_are_pageable(monkeypatch):
    """L35/O35/H1: pinned loading alone gave +28 to +42%; the -ub trade lost or tied."""
    from tensward import suggest
    from tensward.playbook import Suggestion

    entries = {e.name: e for e in ENGINES["llamacpp"].playbook()}
    offered = [
        Suggestion(entry=entries["llamacpp-slots"], reason="slots", tier="try_first"),
        Suggestion(entry=entries[COPIES], reason="copies", tier="try_first"),
        Suggestion(entry=entries["load-mode-none"], reason="pinned", tier="could_help"),
    ]
    monkeypatch.setattr(suggest, "_pageable_copies", lambda: False)
    assert [s.reason for s in suggest._census_early(offered)[0]] == ["copies", "slots", "pinned"]
    monkeypatch.setattr(suggest, "_pageable_copies", lambda: True)
    ordered = suggest._census_early(offered)[0]
    assert [s.reason for s in ordered] == ["pinned", "copies", "slots"]
    assert ordered[0].tier == "try_first"
    # L35: no census offer left once the -ub gate held, and an answer-changing expert cache
    # that the start order puts first
    cache = Suggestion(
        entry=entries["llamacpp-expert-cache-share"], reason="cache", tier="try_first"
    )
    ordered = suggest._census_early([cache, offered[2]])[0]
    assert [s.reason for s in ordered] == ["pinned", "cache"]
    started = suggest._start_order(None, None, ordered[::-1])  # type: ignore[arg-type]
    assert [s.reason for s in started] == ["pinned", "cache"]


GLM_START = """\
0.05.377.206 I llama_context: n_ctx                 = 65536
0.05.377.208 I llama_context: n_ubatch              = {ub}
0.05.428.961 I sched_reserve:      CUDA0 compute buffer size =  {buffer} MiB
0.05.590.661 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 65536
"""


def _ubatch_view(tmp_path, ub, buffer, room_mib, buffers, kind="expert_offload"):
    run_dir = tmp_path / f"{ub}-{room_mib}-{kind}-{len(buffers)}"
    run_dir.mkdir()
    (run_dir / "server.log").write_text(GLM_START.format(ub=ub, buffer=buffer))
    placement = {
        "regime": {"placement": kind},
        "placement": {"host_expert_bytes": 26_707_230_720, "host_layers": 0,
                      "host_expert_layers": 14},
        "fit": {"device_free": 80_000 * MIB, "device_need": (80_000 - room_mib) * MIB},
    }  # fmt: skip
    anatomy = SimpleNamespace(expert_layer_bytes=(0,) + (2_250_000_000,) * 45)
    situation = SimpleNamespace(
        settings=Settings(extra_args={"--ubatch-size": ub}, prefill_batch_tokens=ub),
        measurement=Measurement(40, 0),
    )
    probes = {"ubatch_buffers": {str(k): v * MIB for k, v in buffers.items()}}
    context = AnalysisContext(
        engine="llamacpp", new_run=run_dir, placement=placement, probes=probes, anatomy=anatomy
    )
    return situation, context


def test_ubatch_rungs_read_the_probed_compute_buffer(tmp_path):
    # GLM-5.3-Flash Q2 on one H100: the buffer grows 1,437 -> 2,069 -> 2,913 -> 5,425 MiB
    glm = {2048: 2069, 4096: 2913, 8192: 5425}
    situation, context = _ubatch_view(tmp_path, 512, 1436.89, 1729, glm)
    with active(context):
        seen = predicates.view(situation)
        assert predicates._next_ubatch(situation, seen) == 2048  # +632 MiB fits as it is
        assert predicates._ubatch_trade(situation) is None
    situation, context = _ubatch_view(tmp_path, 2048, 2069, 1097, glm)
    with active(context):
        assert predicates._next_ubatch(situation, predicates.view(situation)) == 4096
    situation, context = _ubatch_view(tmp_path, 4096, 2913, 253, glm)
    with active(context):
        assert predicates._next_ubatch(situation, predicates.view(situation)) is None
        trade = predicates._ubatch_trade(situation)
        settings = predicates.with_ubatch_trade(situation, ENGINES["llamacpp"])
    assert (trade.rung, trade.more) == (8192, 2)
    assert settings.extra_args["--n-cpu-moe"] == 16 and settings.extra_args["--ubatch-size"] == 8192
    # no probe: today's buffer scaled with the rung
    situation, context = _ubatch_view(tmp_path, 512, 1436.89, 1729, {})
    with active(context):
        assert predicates._next_ubatch(situation, predicates.view(situation)) is None
    # a dense offload stops at 2048
    situation, context = _ubatch_view(tmp_path, 2048, 1032, 9000, {}, kind="dense_offload")
    with active(context):
        assert predicates._ubatch_growth(situation, predicates.view(situation)) is None


def test_the_pcie_slope_compares_runs_that_loaded_the_host_weights_alike():
    from tensward.engines.llamacpp.probes import PrefillPoint, pcie_slope

    # GLM2 Q2: the mapped baseline, the pinned run 0002 (-ub 512, 14 layers) and the pinned
    # step 0003 (-ub 2048, 16 layers); rc10 paired the first with the last and stored 8.5 GB/s
    mapped = PrefillPoint(512, 7066, 46045.5, 26_707_233_341, "mmap")
    pinned = PrefillPoint(512, 7066, 18817.4, 26_707_233_341, "none")
    step = PrefillPoint(2048, 7066, 9165.8, 31_350_327_869, "none")
    p = SimpleNamespace(host_weight_bytes=31_350_327_869)
    assert pcie_slope(mapped, step, p, (4, 16)).gbs_effective is None
    assert round(pcie_slope(pinned, step, p, (4, 16)).gbs_effective, 1) == 25.7


def test_one_slow_first_request_does_not_project_the_run(monkeypatch):
    from tensward import measure

    clock = [0.0]
    monkeypatch.setattr(measure.time, "monotonic", lambda: clock[0])
    progress = measure.Progress()
    progress.started = 0.0
    clock[0] = 191.0  # GLM2 Q4 -ub 2048 pinned: the first request took 191 s in warm-up
    progress.first, progress.done = 191.0, 1
    assert progress.projected_s(40) is None
    clock[0], progress.done = 341.0, 3  # then two at the parent's pace, 75 s each
    assert progress.projected_s(40) == 191.0 + 150.0 * 39 / 2


def test_a_placement_with_experts_on_the_host_is_sized_from_the_logged_weights():
    from tensward.capabilities import Resolved
    from tensward.engines.llamacpp.prelaunch import _logged_weights

    # GLM2 Q2's try 3, --n-cpu-moe 13 from the baseline's 14: one layer's experts come back on
    # the device, not the whole 101.2 GiB file less 13 layers (it holds an unloaded MTP layer)
    anatomy = SimpleNamespace(expert_layer_bytes=(0,) + (1_907_659_337,) * 45)
    baseline = Resolved(
        levers={"host_expert_layers": 14, "host_layers": 0},
        facts={"device_weight_bytes": 78_779_231_764},
    )
    assert _logged_weights(anatomy, baseline, 13) == 78_779_231_764 + 1_907_659_337
    assert _logged_weights(anatomy, baseline, 16) == 78_779_231_764 - 2 * 1_907_659_337
    assert _logged_weights(anatomy, None, 13) is None


def test_split_slots_that_reserve_more_than_flies_free_cells_for_experts(tmp_path, monkeypatch):
    """FN's matched baseline: 16 split slots of 16k for 16 clients, nothing queued, experts of 6
    layers on the host; one pool sized 20% over the measured peak frees KV for expert layers."""
    (tmp_path / "server.log").write_text(
        SPLIT_START.replace("131072", "262144").replace("n_slots = 4, n_ctx_slot = 32768",
                                                        "n_slots = 16, n_ctx_slot = 16384")
    )  # fmt: skip
    situation = SimpleNamespace(
        measurement=Measurement(160, 0, peak_in_flight=16, peak_running=16.0, peak_waiting=0.0,
                                max_prompt_tokens=16000, peak_context_tokens=135_453,
                                tpot_p50_ms=50.9),
        facts=SimpleNamespace(output_tokens=1280), settings=Settings(max_concurrent_requests=16),
        diagnosis=None, fired=lambda bottleneck: False,
    )  # fmt: skip
    room = _room(1500, host_expert_layers=6, host_layers=0)
    room["fit"] |= {"pool_tokens": 262_144, "kv_bytes": 30_720 * 262_144}
    room["regime"] = {"placement": "expert_offload"}
    spent = predicates.Spent(0, 6, 0.0, 0), predicates.Spent(0, 4, 9.5, 3000 * MIB)
    monkeypatch.setattr(predicates, "_spend", lambda seen, extra, share: spent)
    monkeypatch.setattr(predicates, "_kv_token_bytes", lambda seen, layers: 30_720.0)
    engine = ENGINES["llamacpp"]
    with active(AnalysisContext(engine="llamacpp", new_run=tmp_path, placement=room)):
        why = predicates.shared_pool_applies(situation, engine)
        settings = predicates.with_shared_pool(situation, engine)
    assert why.startswith(
        "shared pool of 162,560 tokens (the most measured in flight, 135,453, plus 20%) for the "
        "same 16 slots, with the experts of 2 more layers on the GPU (--n-cpu-moe 6 to 4): "
        "nothing waited for a slot"
    )
    assert "about 9.5 ms a token less decode" in why
    assert (settings.max_concurrent_requests, settings.max_context_len) == (16, 162_560)
    assert settings.extra_args["--n-cpu-moe"] == 4 and settings.extra_args["--kv-unified"]
    # when the freed cells bring nothing back, nothing is offered
    monkeypatch.setattr(predicates, "_spend", lambda seen, extra, share: (spent[0], spent[0]))
    with active(AnalysisContext(engine="llamacpp", new_run=tmp_path, placement=room)):
        assert predicates.shared_pool_applies(situation, engine) is None


def test_an_unmeasured_host_share_is_estimated_below_the_decode_models_bound(tmp_path):
    from tensward.engines.llamacpp.startup import read_startup

    seen = predicates.View(read_startup(""), {"decode_model": {"bw_host_gbs": 20.0}}, {})
    read, tpot = 300 * 10**6, 20.0  # 15 ms at 20 GB/s: the bound is 75%
    with active(AnalysisContext(engine="llamacpp", project=tmp_path)):
        share, words = predicates._estimated_host_share(seen, read, tpot)
        assert share == pytest.approx(0.375) and "half the 75%" in words and "estimated" in words
        run = tmp_path / "runs" / "0001-baseline"
        run.mkdir(parents=True)
        probe = {"host": {"method": "launch", "mem_bw_gbs": 60.0}}
        (run / "run.json").write_text(json.dumps({"probes": probe}))
        share, words = predicates._estimated_host_share(seen, read, tpot)
        assert share == pytest.approx(0.25) and "60.0 GB/s an earlier probe" in words


def test_mtp_is_held_back_with_experts_on_the_host(tmp_path):
    (tmp_path / "server.log").write_text("")
    hold = predicates._mtp_host_experts
    situation = SimpleNamespace(settings=Settings())
    for placement, held in (
        ({"regime": {"placement": "expert_offload"}, "placement": {"host_expert_layers": 20}}, 1),
        ({"regime": {"placement": "full_gpu"}, "placement": {"host_expert_layers": 0}}, 0),
    ):
        with active(AnalysisContext(engine="llamacpp", new_run=tmp_path, placement=placement)):
            assert bool(hold(situation)) == bool(held)


def test_a_cuda_start_with_nothing_on_the_card_is_refused():
    from tensward.engines.llamacpp.startup import cpu_fallback

    cpu = (
        "ggml_backend_load_best: failed to load libggml-cuda.so: libcudart.so.12: cannot open\n"
        "load_tensors:   CPU_Mapped model buffer size =  9000.00 MiB\n"
    )
    why = cpu_fallback(cpu, {})
    assert why and "0 MiB of the model on the card" in why and "libcudart.so.12" in why
    assert cpu_fallback(cpu, {"--n-gpu-layers": "0"}) is None
    on_card = cpu + "load_tensors:        CUDA0 model buffer size =  8000.00 MiB\n"
    assert cpu_fallback(on_card, {}) is None and cpu_fallback("", {}) is None
