import json
from pathlib import Path
from types import SimpleNamespace

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


MMAP_START = """\
0.01.000.000 W model_load: tensor overrides to CPU are used with mmap enabled
0.02.000.000 I load_tensors: loading model tensors, this can take a while... (load_mode = mmap)
"""


def _load_mode_offer(tmp_path, kind, host_bytes, settings=None, log=MMAP_START):
    from tensward.engines.llamacpp.predicates import load_mode_none_applies

    run_dir = tmp_path / str(len(list(tmp_path.iterdir())))  # the log is cached by its mtime
    run_dir.mkdir()
    (run_dir / "server.log").write_text(log)
    situation = SimpleNamespace(settings=settings or Settings())
    placement = {
        "regime": {"placement": kind},
        "placement": {"host_weight_bytes": host_bytes},
        "environment": {},
    }
    context = AnalysisContext(engine="llamacpp", new_run=run_dir, placement=placement)
    with active(context):
        return load_mode_none_applies(situation, ENGINES["llamacpp"])


def test_load_mode_none_offered_for_cpu_offloaded_experts_read_through_mmap(tmp_path):
    from tensward.engines.llamacpp.predicates import with_load_mode_none

    assert "through mmap" in _load_mode_offer(tmp_path, "expert_offload", 8 * 2**30)
    settings = with_load_mode_none(SimpleNamespace(settings=Settings()), ENGINES["llamacpp"])
    assert settings.extra_args["--load-mode"] == "none"


def test_load_mode_none_not_offered_when_nothing_is_offloaded_or_it_is_set(tmp_path):
    assert _load_mode_offer(tmp_path, "full_gpu", 0) is None
    set_already = Settings(extra_args={"--load-mode": "mlock"})
    assert _load_mode_offer(tmp_path, "expert_offload", 8 * 2**30, set_already) is None
    direct = MMAP_START.replace("tensor overrides to CPU are used with mmap enabled", "x").replace(
        "(load_mode = mmap)", "(load_mode = dio)"
    )
    assert _load_mode_offer(tmp_path, "expert_offload", 8 * 2**30, log=direct) is None


def test_load_mode_none_held_back_when_the_build_lacks_the_flag(monkeypatch):
    from tensward.engines.llamacpp import predicates

    manifest = SimpleNamespace(missing=lambda flags: list(flags))
    monkeypatch.setattr(predicates, "known_build", lambda: (None, manifest))
    assert "--load-mode" in predicates.HOLD_BACKS["load-mode-none"](SimpleNamespace())


def test_a_change_for_another_goal_is_offered_only_when_nothing_else_is():
    from dataclasses import replace

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

    def offer(prompt_tokens, hit_rate, settings=Settings()):
        situation = SimpleNamespace(
            settings=settings,
            measurement=Measurement(
                160, 0, max_prompt_tokens=prompt_tokens, prefix_cache_hit_rate=hit_rate
            ),
        )
        context = AnalysisContext(engine="llamacpp", new_run=tmp_path, placement=placement)
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
    with active(AnalysisContext(engine="llamacpp")):
        command = suggestion_command(ENGINES["llamacpp"], tmp_path, Settings(), settings)
    assert command.endswith("--engine-env GGML_OP_OFFLOAD_MIN_BATCH=256")


QWEN38_START = """\
0.08.357.594 I llama_context: n_ctx                 = 36096
0.08.372.764 I llama_kv_cache: size = 2256.00 MiB ( 36096 cells,  16 layers,  4/1 seqs), \
K (f16): 1128.00 MiB, V (f16): 1128.00 MiB
0.08.452.489 I sched_reserve:      CUDA0 compute buffer size =   183.74 MiB
"""


def _mtp_offer(tmp_path, monkeypatch, model, settings=Settings(), goal="tpot"):
    monkeypatch.setattr("tensward.formats.gguf.fingerprint_files", lambda *a, **k: "0" * 64)
    run_dir = tmp_path / str(len(list(tmp_path.iterdir())))
    run_dir.mkdir()
    (run_dir / "server.log").write_text(QWEN38_START)
    (file,) = write_gguf(run_dir / "model.gguf", model)
    _, anatomy = GGUF.register(file, engine_build="b11669", engine_name="llamacpp")
    situation = SimpleNamespace(settings=settings, measurement=Measurement(160, 0))
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
    assert layers == 1 and "about 325 MiB" in offer
    assert _mtp_offer(tmp_path, monkeypatch, without_mtp(qwen)) == (0, None)
    unread = [t for t in qwen["tensors"] if ".nextn." not in t[0]]
    assert _mtp_offer(tmp_path, monkeypatch, {**qwen, "tensors": unread}) == (None, None)
    speculating = Settings(extra_args={"--spec-type": "ngram-mod"})
    assert _mtp_offer(tmp_path, monkeypatch, qwen, speculating) == (1, None)
    assert _mtp_offer(tmp_path, monkeypatch, qwen, goal="ttft") == (1, None)
    help_text = "--spec-type none,ngram-mod   kind\n--spec-draft-n-max N   most drafted\n"
    monkeypatch.setattr(
        predicates, "known_build", lambda: (None, HelpManifest.from_help(help_text))
    )
    held = predicates.HOLD_BACKS["llamacpp-mtp-speculation"](SimpleNamespace())
    assert held == "this build's --help lists no draft-mtp for --spec-type"
