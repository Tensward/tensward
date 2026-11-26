import hashlib
import json
import os
import re
import shlex
import signal
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from registration_fixtures import make_registration_inputs, write_gguf
from test_llamacpp import FAKE, L4, _analyse, _cheap_hash, _describe, _l4, _record_offers, _register

from tensward import kb_local
from tensward import runtime as runtimes
from tensward.census import inefficiency, policy
from tensward.census import library as locator
from tensward.census.library import CensusLibrary
from tensward.census.readings import ReadingContext, compute_readings
from tensward.census.record import CensusRecord, read_record
from tensward.census.snapshot import (
    Counted,
    SnapshotInvalid,
    parse_snapshot,
    read_snapshots,
    read_timeline,
    window,
)
from tensward.census.start import CENSUS_LINE
from tensward.cli import main
from tensward.cli_try import _words, offers
from tensward.engines import ENGINES
from tensward.kb_schema import record_from_json
from tensward.kept import read_kept, step_block
from tensward.platforms import nvidia
from tensward.report.census import inefficiency_section

FIXTURES = Path(__file__).parent / "census_snapshots"
NAMES = sorted(p.name for p in FIXTURES.iterdir())


def load(name: str):
    folder = FIXTURES / name
    meta = json.loads((folder / "meta.json").read_text())
    maps = [
        {
            int(pid): parse_snapshot(entry)
            for pid, entry in json.loads((folder / f"{tag}.json").read_text()).items()
        }
        for tag in ("before", "after")
    ]
    return folder, meta, *maps


def cut(name: str, **overrides):
    folder, meta, before, after = load(name)
    timelines = {
        meta["pid"]: read_timeline(
            folder / "timeline.jsonl", meta["window_start_ns"], meta["window_end_ns"]
        )
    }
    args = dict(
        requests=meta["requests"],
        start_ns=meta["window_start_ns"],
        end_ns=meta["window_end_ns"],
        timelines=timelines,
    )
    return meta, before, after, window(before, after, **{**args, **overrides})


@pytest.mark.parametrize("name", NAMES)
def test_every_recorded_snapshot_parses(name):
    _, meta, before, after = load(name)
    assert meta["pid"] in after and meta["pid"] in before
    assert after[meta["pid"]].n_kernels > 0
    assert all(row.name for s in after.values() for row in s.kernels)


@pytest.mark.parametrize("name", NAMES)
def test_the_window_is_the_difference_of_the_two_snapshots(name):
    meta, before, after, win = cut(name)
    pid = meta["pid"]
    assert pid in win.pids
    assert win.n_kernels == after[pid].n_kernels - before[pid].n_kernels
    assert sum(k.n for k in win.kernels) == win.n_kernels
    assert win.copies[1].bytes == after[pid].copies[1].bytes - (
        before[pid].copies[1].bytes if 1 in before[pid].copies else 0
    )
    assert win.timeline
    assert all(meta["window_start_ns"] <= b["t_ns"] < meta["window_end_ns"] for b in win.timeline)


def test_only_processes_that_recorded_are_in_the_window():
    meta, before, after, win = cut("multi-pid-6")
    assert win.pids == (13974,)
    assert win.timeline
    other = next(p for p in after if p != 13974)
    grown = {**after, other: replace(after[other], n_kernels=after[other].n_kernels + 5)}
    twice = window(
        before,
        grown,
        requests=1,
        start_ns=0,
        end_ns=1,
        timelines={13974: [{"t_ns": 0}], other: [{"t_ns": 0}]},
    )
    assert twice.pids == tuple(sorted((13974, other)))
    assert twice.timeline == ()


def test_a_newer_minor_keeps_its_new_fields_and_another_major_is_refused():
    folder = FIXTURES / "llamacpp-b12246-census-2.1"
    entry = json.loads((folder / "after.json").read_text())["5109"]
    snap = parse_snapshot(entry)
    assert snap.lib["sm_count"] == 58 and snap.memory["peak_bytes"] > 0
    assert snap.copies_detail["2:3:2"] == Counted(7793, 682342001, 8814686720)
    newer = parse_snapshot({**entry, "census": "2.7", "shiny": {"x": 1}})
    assert newer.minor == 7 and newer.extra == {"shiny": {"x": 1}}
    old = parse_snapshot(
        json.loads((FIXTURES / "docker-vllm-0.30.0-l4" / "after.json").read_text()).popitem()[1]
    )
    assert old.lib is None and old.memory is None and old.copies_detail == {}
    for bad in (
        {**entry, "census": "3.0"},
        {**entry, "census": "1.9"},
        {k: v for k, v in entry.items() if k != "kernels"},
        {**entry, "apis": [3]},
        {**entry, "kinds": 5},
    ):
        with pytest.raises(SnapshotInvalid):
            parse_snapshot(bad)


def test_reading_a_directory_skips_what_cannot_be_parsed(tmp_path):
    entry = json.loads((FIXTURES / "llamacpp-b12246-census-2.1" / "after.json").read_text())["5109"]
    (tmp_path / "census.5109.json").write_text(json.dumps(entry))
    (tmp_path / "census.77.json").write_text("{not json")
    (tmp_path / "census.snap-after.json").write_text(json.dumps({"5109": entry}))
    (tmp_path / "census.5109.tl.jsonl").write_text("")
    assert set(read_snapshots(tmp_path)) == {5109}


def test_a_timeline_bucket_written_twice_is_added_up(tmp_path):
    path = tmp_path / "tl.jsonl"
    path.write_text(
        '{"census":"2.0","t_ns":0}\n{"t_ns":10,"busy_ns":3}\nbroken\n{"t_ns":10,"busy_ns":4}\n{"t_ns":20,"busy_ns":1}\n'
    )
    assert read_timeline(path, 5, 20) == [{"t_ns": 10, "busy_ns": 7}]


DESK = {
    "ea1-tool27-def-b12246-h100": dict(
        E1=0.020462548735979122,
        H=0.535444003722957,
        copy_part=0.0687853667462959,
        host_part=0.46665863697666105,
        gb_per_request=0.9200298429791666,
    ),
    "ea1-tool27-ht-b12246-h100": dict(
        E1=0.02363825754299909,
        H=0.4337191900012505,
        copy_part=0.08400340849391306,
        host_part=0.3497157815073374,
        gb_per_request=0.9313017126666666,
    ),
    "a3-llama-b12246-l4": dict(
        E1=0.027388713401397556,
        H=0.14026337037915385,
        copy_part=0.011170361519797235,
        host_part=0.12909300885935662,
        gb_per_request=0.10187701846511628,
        pageable_h2d=0.2526106531889563,
        pinned_h2d=0.7473893468110437,
        label="host_work",
    ),
    "a3-vllm-0.33.0-l4": dict(E1=0.0010211837555674627),
    "vllm-0.30.0-h100-qwen3-8b-c8": dict(E1=0.040430355678197374),
}


SHOWN = {
    "ea1-tool27-def-b12246-h100": dict(kernel_us=11.43, graph_share=0.1108),
    "ea1-tool27-ht-b12246-h100": dict(kernel_us=11.45, graph_share=0.1202),
    "a3-llama-b12246-l4": dict(kernel_us=32.70, graph_share=0.0),
    "a3-vllm-0.33.0-l4": dict(kernel_us=123.80, graph_share=0.9382),
    "vllm-0.30.0-h100-qwen3-8b-c8": dict(kernel_us=13.92, graph_share=0.9159),
}


def judged(name: str):
    meta, _, _, win = cut(name)
    engine = "vllm" if meta["engine"] == "vllm" else "llamacpp"
    graphs_off = engine == "llamacpp"
    return win, ReadingContext(engine, meta["gpu"], None, graphs_off, version=meta["version"])


@pytest.mark.parametrize("name", sorted(DESK))
def test_inefficiency_matches_the_desk_values(name):
    win, ctx = judged(name)
    desk = dict(DESK[name])
    found = {c.kind: c for c in inefficiency.classes(win, ctx)}
    gaps = found.pop("launch_gaps")
    assert (gaps.share, gaps.floor) == (pytest.approx(desk.pop("E1"), rel=1e-6), 0.10)
    assert gaps.detail == pytest.approx(SHOWN[name], rel=1e-3)
    if not desk:
        assert found == {}
        return
    wait = found.pop("host_wait")
    assert (wait.share, wait.floor) == (pytest.approx(desk.pop("H"), rel=1e-6), 0.30)
    assert wait.label == desk.pop("label", None)
    assert wait.detail == pytest.approx(desk, rel=1e-6)
    kept = [c.kind for c in inefficiency.inefficiencies(win, ctx)]
    assert kept == [c for c in ("launch_gaps", "host_wait") if c in inefficiency.CONFIRMED]


@pytest.mark.parametrize("name", sorted(DESK))
def test_the_classes_add_up_to_the_idle_share(name):
    win, ctx = judged(name)
    busy = compute_readings(win, ctx)["census.busy_share"]
    found = inefficiency.classes(win, ctx)
    if len(found) == 2:
        assert sum(c.share for c in found) == pytest.approx(1 - busy)


def test_inefficiency_is_unknown_outside_its_validated_situation():
    win, ctx = judged("a3-llama-b12246-l4")
    assert inefficiency.inefficiencies(win, ctx) is not None
    two = replace(win, pids=(*win.pids, win.pids[0] + 1))
    assert inefficiency.inefficiencies(two, ctx) is None
    assert inefficiency.inefficiencies(win, replace(ctx, version="b11669")) is None
    for void in (replace(win, failed=1), replace(win, timeline=({"t_ns": 0, "busy_ns": 0},))):
        assert [c.kind for c in inefficiency.classes(void, ctx)] == ["launch_gaps"]
    unscoped = {c.kind: c for c in inefficiency.classes(replace(win, scoped=False), ctx)}
    assert unscoped["host_wait"].detail.keys() == {"copy_part", "host_part"}
    vllm, on_033 = judged("a3-vllm-0.33.0-l4")
    assert inefficiency.inefficiencies(vllm, on_033) is not None
    assert inefficiency.inefficiencies(vllm, replace(on_033, version="0.31.0")) is None


# Track C's frozen RD2 share (creport.py: census D2H MB per request / output tokens per request /
# (n_vocab x 4)) and its busy share, on the research census arms the fixtures were cut from.
TRACK_C = {
    "c2-rd2-base-b12246-h100": (1.0000000000000002, 0.6464639313153813),
    "c2-rd2-bs-b12246-h100": (1.208118556701031e-05, 0.8307393908101153),
    "c1-chat27-base-b12246-l4": (2.5207499780964344, 0.8802326496670769),
    "c1-glm9-base-b12246-l4": (1.6245810569850119, 0.877738657530274),
}


def _research(name, **context):
    meta, _, _, win = cut(name)
    research = meta["research"]
    win = replace(win, out_tokens=research["out_tokens"])
    ctx = ReadingContext("llamacpp", meta["gpu"], None, True, version=meta["version"],
                         n_vocab=research["n_vocab"], **context)  # fmt: skip
    return win, ctx, compute_readings(win, ctx)


@pytest.mark.parametrize("name", sorted(TRACK_C))
def test_reading_RD2r_matches_track_c_on_its_recorded_inputs(name):
    share, busy = TRACK_C[name]
    win, ctx, found = _research(name)
    assert found["census.logits_on_host_share"] == pytest.approx(share, rel=1e-6)
    assert found["census.busy_share"] == pytest.approx(busy, rel=1e-4)
    assert "census.logits_on_host_share" not in compute_readings(replace(win, out_tokens=0), ctx)
    assert "census.logits_on_host_share" not in compute_readings(win, replace(ctx, n_vocab=None))


def test_reading_busy_share_on_E3A_and_C2_fixtures():
    win, ctx = judged("ea1-tool27-def-b12246-h100")
    assert compute_readings(win, ctx)["census.busy_share"] == pytest.approx(0.444, abs=5e-4)
    assert _research("c2-rd2-base-b12246-h100")[2]["census.busy_share"] < 0.70
    assert _research("c1-chat27-base-b12246-l4")[2]["census.busy_share"] >= 0.80


P2_CENSUS = Path(__file__).parent / "fixtures" / "census-p2-v100"
P2_WINDOW = (8945977000000, 9054201000000)  # 108.224 s, the span its census.json records
P2_HOST_WEIGHT_BYTES = 4100456448  # run.host_weight_bytes of the same run


def p2_window():
    after = {
        int(pid): parse_snapshot(entry)
        for pid, entry in json.loads((P2_CENSUS / "after.json").read_text()).items()
    }
    start, end = P2_WINDOW
    timeline = read_timeline(P2_CENSUS / "timeline.jsonl", start, end)
    win = window({}, after, requests=5, start_ns=start, end_ns=end, timelines={35384: timeline})
    ctx = ReadingContext("llamacpp", "Tesla V100-SXM2-16GB", None, True, version="12246",
                         host_weight_bytes=P2_HOST_WEIGHT_BYTES)  # fmt: skip
    return win, ctx


def test_the_host_passes_and_the_copy_wait_on_the_v100_census():
    stored = json.loads((P2_CENSUS / "census.json").read_text())["readings"]
    win, ctx = p2_window()
    found = compute_readings(win, ctx)
    for name in ("census.busy_share", "census.h2d_bytes_per_request"):
        assert found[name] == pytest.approx(stored[name], rel=1e-5)
    assert found["census.host_passes_per_request"] == pytest.approx(
        stored["census.h2d_bytes_per_request"] / P2_HOST_WEIGHT_BYTES, rel=1e-9
    )
    assert 11.7 < found["census.host_passes_per_request"] < 11.9
    assert 0.37 < found["census.copy_wait_share"] < 0.41
    assert "census.host_passes_per_request" not in compute_readings(
        win, replace(ctx, host_weight_bytes=None)
    )
    for outside in (replace(ctx, graphs_off=False), replace(ctx, version="11669")):
        assert "census.copy_wait_share" not in compute_readings(win, outside)
    assert "census.copy_wait_share" not in compute_readings(replace(win, failed=1), ctx)


def test_the_copy_wait_reading_is_the_copy_part_of_the_host_wait():
    win, ctx = judged("ea1-tool27-def-b12246-h100")
    wait = next(c for c in inefficiency.classes(win, ctx) if c.kind == "host_wait")
    assert compute_readings(win, ctx)["census.copy_wait_share"] == wait.detail["copy_part"]


def _section(name, monkeypatch, confirmed=("launch_gaps", "host_wait"), census_run=None):
    monkeypatch.setattr(inefficiency, "CONFIRMED", frozenset(confirmed))
    win, ctx = judged(name)
    found = tuple(inefficiency.inefficiencies(win, ctx))
    record = CensusRecord("ok", "", readings=compute_readings(win, ctx), inefficiency=found)
    section = inefficiency_section(record, ctx.engine, census_run)
    return section and [block.text for block in section.blocks]


def test_the_report_says_what_the_settings_cannot_fix(monkeypatch):
    (line,) = _section("ea1-tool27-def-b12246-h100", monkeypatch)
    assert line.startswith(
        "Up to 54% of the time the GPU waited on the host; a host-side change can give back at "
        "most this share. 7% on copies between host and GPU (0.9 GB per request), 47% on host "
        "work between steps. Work that removes this: copying from pinned host buffers"
    )
    assert "overlapping copies with compute" in line and "huge-page host buffers" in line


def test_none_of_the_checked_kinds_is_large_is_said_plainly(monkeypatch):
    assert _section("a3-llama-b12246-l4", monkeypatch) == [
        "None of the checked kinds is large here (launch gaps, waiting on the host): the GPU "
        "ran kernels 83% of the time."
    ]


def test_the_section_is_absent_with_nothing_confirmed(monkeypatch):
    assert _section("ea1-tool27-def-b12246-h100", monkeypatch, confirmed=()) is None


def test_llamacpp_launch_gaps_give_no_graph_advice(monkeypatch):
    monkeypatch.setattr(inefficiency, "CONFIRMED", frozenset({"launch_gaps"}))
    gaps = inefficiency.Inefficiency("launch_gaps", 0.25, 0.10, {"kernel_us": 6.0})
    record = CensusRecord("ok", "", readings={"census.busy_share": 0.5}, inefficiency=(gaps,))
    (llama,) = inefficiency_section(record, "llamacpp").blocks
    (vllm,) = inefficiency_section(record, "vllm").blocks
    assert llama.text == (
        "Launch gaps: 25% of the time in gaps under 0.1 ms (recorded with CUDA graphs off; a "
        "setup with graphs on has fewer). Work that removes this: fusing small kernels into "
        "fewer, larger ones."
    )
    assert "as a CUDA graph" in vllm.text and "(6 µs per kernel on average)" in vllm.text


def test_the_section_names_the_setup_it_was_recorded_on(llama_project, census, capsys, monkeypatch):
    monkeypatch.setattr(inefficiency, "CONFIRMED", frozenset({"host_wait"}))
    _, _, first, _ = _analysed(llama_project, capsys, monkeypatch)
    _, _, second, again = _analysed(llama_project, capsys, monkeypatch)
    (wait,) = again.inefficiency
    assert wait.kind == "host_wait" and wait.share >= wait.floor
    text = (second / "report.md").read_text()
    section = text.split("## What the settings can't fix")[1].split("\n## ")[0]
    assert f"Recorded on the setup of run {first.name}." in section
    assert f"Up to {wait.share:.0%} of the time the GPU waited on the host" in section
    assert text.index("## GPU census") < text.index("## What the settings can't fix")
    metrics = [json.loads((run / "metrics.json").read_text())["census"] for run in (first, second)]
    assert "run" not in metrics[0] and metrics[1]["run"] == first.name
    assert metrics[1]["inefficiency"] == metrics[0]["inefficiency"]


FAKE_VLLM = Path(__file__).resolve().parent / "fake_vllm_server.py"
COLD_START = Path(__file__).resolve().parent / "fixtures" / "vllm-vh-cold-start.log"


@pytest.fixture
def census(tmp_path, monkeypatch):
    """An installed census for CUDA 12, the engine's CUPTI in TENSWARD_CUPTI_DIR, and the start
    sequence shortened; returns the file each fake start logs its census variables to."""
    home = tmp_path / "census-lib"
    home.mkdir()
    body = b"census library cu12"
    (home / "libtwcensus-v2-cu12.so").write_bytes(body)
    entry = {
        "file": "libtwcensus-v2-cu12.so",
        "cupti_major": 12,
        "cupti_api_min": 22,
        "glibc_min": "2.25",
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    manifest = {"census": "2.1", "libraries": [entry], "cupti": [], "cupti_files": []}
    (home / "manifest.json").write_text(json.dumps(manifest))
    cupti = tmp_path / "cupti"
    cupti.mkdir()
    (cupti / "libcupti.so.12").write_bytes(b"cupti 12")
    monkeypatch.setenv("TENSWARD_CENSUS_DIR", str(home))
    monkeypatch.setenv("TENSWARD_CUPTI_DIR", str(cupti))
    monkeypatch.setattr(locator, "engine_cupti_major", lambda profile, command: 12)
    for name, value in SHORT.items():
        monkeypatch.setattr(policy, name, value)
    log = tmp_path / "census-env.jsonl"
    monkeypatch.setenv("FAKE_CENSUS_ENV_LOG", str(log))
    monkeypatch.setenv("FAKE_LLAMA_BUILD", "12246")
    return log


SHORT = {"IDLE_S": 0.5, "UNPAUSE_WAIT_S": 1.2, "CENSUS_SLICE_S": 3.0, "FLUSH_WAIT_S": 1.5}


@pytest.fixture
def llama_project(tmp_path):
    """A fresh project for each test, so no run reuses an earlier one."""
    root = tmp_path / "llama"
    root.mkdir()
    (file,) = write_gguf(root / "Qwen3-0.6B-Q8_0.gguf", _describe("qwen3-0.6b-q8_0"))
    command = f"llama-server -m {file} -np 2 --port 9931"
    with _l4(), _cheap_hash(), pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_CACHE_HOME", str(root / "cache"))
        workload = {"request_count": 8, "arrival": {"kind": "closed_loop", "concurrency": 4}}
        assert _register(root / "project", file, root, "--current", command, workload=workload) == 0
    return root / "project"


def _analysed(project, capsys, monkeypatch, *, env=None):
    runs = project / "runs"
    before = set(runs.iterdir()) if runs.exists() else set()
    with _cheap_hash():
        code, _, err = _analyse(project, capsys, monkeypatch, env=env)
    (run_dir,) = set(runs.iterdir()) - before
    return code, err, run_dir, read_record(run_dir)


def _census_facts(run_dir):
    facts = json.loads((run_dir / "facts.json").read_text())["facts"]
    return {name: value for name, value in facts.items() if name.startswith("census.")}


def _gone(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_analyse_runs_a_census_start_and_reads_its_snapshot(
    llama_project, census, capsys, monkeypatch
):
    code, err, run_dir, record = _analysed(llama_project, capsys, monkeypatch)
    assert code == 0
    assert record.status == "ok", record.reason
    assert record.reason == ""
    assert CENSUS_LINE in err
    readings = dict(record.readings)
    assert readings["census.graphs_off"] is True and readings["census.recording_processes"] == 1
    assert readings["census.kernels_per_request"] > 0 and 0 < readings["census.busy_share"] < 1
    assert "census.kernel_ms_per_request" in readings
    assert _census_facts(run_dir) == readings
    assert record.requests >= 16 and record.span_s > 0 and 0 < record.wall_s <= 300
    assert record.out_tokens >= record.requests
    assert record.proof["group_empty"] and record.library["verified"] is True
    measured, started = [json.loads(line) for line in census.read_text().splitlines()]
    assert measured == dict.fromkeys(measured)
    assert started["GGML_CUDA_DISABLE_GRAPHS"] == "1"
    assert started["CUDA_INJECTION64_PATH"].endswith("libtwcensus-v2-cu12.so")
    log = (run_dir / "census" / "server.log").read_text()
    assert re.search(r"twcensus: on, pid \d+, out \S+ \(paused\), flush ondemand", log)
    assert "twcensus" not in (run_dir / "server.log").read_text()
    serve = json.loads((run_dir / "serve.json").read_text())
    census_serve = json.loads((run_dir / "census" / "serve.json").read_text())
    assert serve["identity"]["pid"] != census_serve["identity"]["pid"]
    assert serve["spec"]["served_model_name"] != census_serve["spec"]["served_model_name"]
    assert _gone(census_serve["identity"]["pid"])


def _vllm_analysed(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(type(nvidia.NVIDIA), "detect", lambda self: (L4,))
    for module in ("project", "measure"):
        monkeypatch.setattr(f"tensward.{module}.detect_devices", lambda: (L4,))
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    command = shlex.join([sys.executable, str(FAKE_VLLM)])
    assert main(["analyse", "--project", str(project), "--runtime", "local",
                 "--local-command", command, "--ready-timeout", "30"]) == 0  # fmt: skip
    capsys.readouterr()
    (run_dir,) = (project / "runs").iterdir()
    return run_dir, read_record(run_dir)


def test_the_vllm_census_start_reads_the_engine_snapshot(tmp_path, census, capsys, monkeypatch):
    run_dir, record = _vllm_analysed(tmp_path, capsys, monkeypatch)
    assert record.status == "ok", record.reason
    server = json.loads((run_dir / "census" / "serve.json").read_text())["identity"]["pid"]
    (engine,) = record.proof["pids"]
    assert engine != server
    assert (
        parse_snapshot(json.loads((run_dir / "census" / f"census.{engine}.json").read_text())).final
        is False
    )
    assert record.readings["census.graphs_off"] is False
    assert "census.kernel_ms_per_request" not in record.readings
    assert _census_facts(run_dir) == dict(record.readings)


@pytest.mark.parametrize("built", [True, False])
def test_a_vllm_start_with_one_time_builds_is_not_skipped_on_its_estimate(
    built, tmp_path, census, capsys, monkeypatch
):
    """B9 L4 vLLM, Qwen3-8B: a 289 s cold start, 145 s of it a graph compile and a kernel JIT
    the census start finds cached; the rest (spawn, weights from disk) still put the estimate
    past the limit, so the census was skipped. The hard stop inside the limit now decides."""
    monkeypatch.setattr(policy, "MARGIN_S", policy.CENSUS_CEILING_S)
    if built:
        monkeypatch.setenv("FAKE_START_LOG", str(COLD_START))
    _, record = _vllm_analysed(tmp_path, capsys, monkeypatch)
    if built:
        assert record.status == "ok", record.reason
    else:
        assert record.status == "skipped"
        assert "over its 10-minute limit, so it was skipped" in record.reason


def _wrong_kind(tmp_path, monkeypatch):
    home = Path(os.environ["TENSWARD_CENSUS_DIR"])
    (home / "libtwcensus-v2-cu12.so").rename(home / "libtwcensus-v2-cu11.so")
    manifest = json.loads((home / "manifest.json").read_text())
    manifest["libraries"][0]["file"] = "libtwcensus-v2-cu11.so"
    (home / "manifest.json").write_text(json.dumps(manifest))


def _missing_library(tmp_path, monkeypatch):
    gone = CensusLibrary(tmp_path / "libtwcensus-v2-cu12.so", 12, 22, "cu12")
    monkeypatch.setattr(locator, "find_library", lambda home, major: gone)


def _graphs_on(tmp_path, monkeypatch):
    engine = ENGINES["llamacpp"]
    profile = replace(engine.census, ship_env={"GGML_CUDA_DISABLE_GRAPHS": "0"})
    monkeypatch.setattr(engine, "census", profile)


def _env(name, value):
    return lambda tmp_path, monkeypatch: monkeypatch.setenv(name, value)


NOT_ATTACHED = {
    "missing library": (_missing_library, "the census library was not loaded into the engine"),
    "read-only out": (_env("FAKE_CENSUS_READ_ONLY", "1"), "Permission denied"),
    "zero kernels": (_env("FAKE_CENSUS_ZERO", "1"), "the census recorded no kernels"),
    "graphs on": (_graphs_on, "CUDA graphs were on in the census start"),
    "wrong kind": (_wrong_kind, "the census library is the wrong kind: cu12, expected cu11"),
}


@pytest.mark.parametrize("case", NOT_ATTACHED)
def test_a_census_that_did_not_attach_is_reported_and_never_read(
    case, llama_project, census, tmp_path, capsys, monkeypatch
):
    arrange, reason = NOT_ATTACHED[case]
    arrange(tmp_path, monkeypatch)
    code, _, run_dir, record = _analysed(llama_project, capsys, monkeypatch)
    assert code == 0
    assert record.status == "not_attached" and reason in record.reason
    assert record.readings == {} and _census_facts(run_dir) == {}
    assert record.proof["group_empty"]


def test_a_census_server_that_ignores_sigterm_is_killed_and_gone(
    llama_project, census, capsys, monkeypatch
):
    monkeypatch.setattr(runtimes, "CENSUS_STOP_GRACE_S", 2.0)
    code, _, run_dir, record = _analysed(
        llama_project, capsys, monkeypatch, env={"FAKE_CENSUS_EXIT_HANG": "1"}
    )
    assert code == 0 and record.status == "ok", record.reason
    assert record.proof["group_empty"]
    assert _gone(json.loads((run_dir / "census" / "serve.json").read_text())["identity"]["pid"])


def test_a_census_start_that_crashes_leaves_the_analysis_whole(
    llama_project, census, capsys, monkeypatch
):
    with _cheap_hash():
        code, out, _ = _analyse(
            llama_project, capsys, monkeypatch, env={"FAKE_CENSUS_CRASH_AFTER": "3"}
        )
    (run_dir,) = (llama_project / "runs").iterdir()
    record = read_record(run_dir)
    assert code == 0 and (run_dir / "report.md").is_file()
    assert record.status == "failed" and record.reason
    assert "the analysis is complete without it" in out
    assert _census_facts(run_dir) == {} and record.proof["group_empty"]
    assert _gone(json.loads((run_dir / "census" / "serve.json").read_text())["identity"]["pid"])


def test_a_census_start_that_overruns_is_stopped_inside_the_ceiling(
    llama_project, census, capsys, monkeypatch
):
    monkeypatch.setattr(policy, "CENSUS_CEILING_S", 40.0)
    for name in ("MARGIN_S", "EXPECTED_STOP_S", "SETTLE_S"):
        monkeypatch.setattr(policy, name, 0.0)
    code, _, run_dir, record = _analysed(
        llama_project, capsys, monkeypatch, env={"FAKE_CENSUS_START_DELAY_S": "45"}
    )
    assert code == 0
    assert record.status == "failed"
    assert record.reason == "the census start passed the 10-minute limit and was stopped"
    assert record.wall_s < 43 and record.proof["group_empty"]


def _cheap(project, capsys, monkeypatch, *extra, env=None):
    with _cheap_hash():
        return _analyse(project, capsys, monkeypatch, *extra, env=env)


def _out_next(out):
    assert "Next:" in out
    return out


def test_the_report_shows_the_gpu_picture_and_the_terminal_line(
    llama_project, census, capsys, monkeypatch
):
    code, out, _ = _cheap(llama_project, capsys, monkeypatch)
    assert code == 0
    (run_dir,) = (llama_project / "runs").iterdir()
    record = read_record(run_dir)
    wall = f"{record.wall_s:.0f} s"
    assert re.search(
        rf"GPU census: busy \d+%, [\d.]+ ms of kernels per request(, [\d.]+ GB \w+ per request)* "
        rf"\(diagnostic start, {wall}\)\. None of Tensward's census offers applies here \(it "
        rf"checks: [^)]*llamacpp-offload-ubatch-copies[^)]*\)\.",
        _out_next(out),
    )
    assert out.index("GPU census:") < out.index("Next:")
    markdown = (run_dir / "report.md").read_text()
    section = markdown.split("## GPU census")[1].split("\n## ")[0]
    assert re.search(r"the GPU ran kernels \d+% of the time", section)
    assert re.search(
        r"[\d.]+ ms of GPU kernels per request \(recorded with CUDA graphs off\)", section
    )
    assert markdown.index("## Checks") < markdown.index("## GPU census")
    assert "not yet verified" not in out


def test_build_11669_gets_no_census_and_the_measured_reason(
    llama_project, census, capsys, monkeypatch
):
    code, out, err = _cheap(llama_project, capsys, monkeypatch, env={"FAKE_LLAMA_BUILD": "11669"})
    assert code == 0, err
    assert code == 0
    (run_dir,) = (llama_project / "runs").iterdir()
    record = read_record(run_dir)
    assert record.status == "skipped" and record.advice
    assert not (run_dir / "census" / "serve.json").exists()
    assert f"GPU census skipped: {record.reason.rstrip('.')}" in out and record.advice in out
    assert out.index(record.advice) < out.index("Next:")


def test_no_cupti_is_said_with_the_held_offers_and_the_install(
    llama_project, census, capsys, monkeypatch
):
    monkeypatch.setattr(locator, "local_cupti", lambda profile, command, major: None)
    code, out, err = _cheap(llama_project, capsys, monkeypatch)
    assert code == 0, err
    (line,) = [text for text in out.splitlines() if text.startswith("GPU census")]
    assert line.startswith(
        "GPU census skipped: no CUPTI found; offers that need it are held back ("
    )
    assert "llamacpp-offload-ubatch-copies" in line
    assert line.endswith(
        "; install with: uv pip install --target ~/.local/share/tensward/cupti "
        "'nvidia-cuda-cupti-cu12'."
    )


def test_no_census_skips_the_start(llama_project, census, capsys, monkeypatch):
    code, out, err = _cheap(llama_project, capsys, monkeypatch, "--no-census")
    assert code == 0
    (run_dir,) = (llama_project / "runs").iterdir()
    record = read_record(run_dir)
    assert record.status == "skipped" and record.reason == "skipped (--no-census)"
    assert not (run_dir / "census" / "serve.json").exists()
    assert CENSUS_LINE not in err and "GPU census start follows" not in err
    assert "GPU census" not in out and "GPU census" not in (run_dir / "report.md").read_text()


def test_the_time_line_adds_the_census_start_and_an_unverified_build_says_so(
    llama_project, census, capsys, monkeypatch
):
    code, out, err = _cheap(llama_project, capsys, monkeypatch, env={"FAKE_LLAMA_BUILD": "13001"})
    assert code == 0
    assert re.search(r"A GPU census start follows, about \d+ min more", err)
    (run_dir,) = (llama_project / "runs").iterdir()
    record = read_record(run_dir)
    assert record.status == "ok" and record.library["verified"] is False
    assert "This engine build is not yet verified with the census." in out
    assert "not yet verified" in (run_dir / "report.md").read_text()


def test_a_reused_run_says_which_run_its_census_is_from(llama_project, census, capsys, monkeypatch):
    _, _, first, record = _analysed(llama_project, capsys, monkeypatch)
    _, err, second, again = _analysed(llama_project, capsys, monkeypatch)
    assert json.loads((second / "run.json").read_text())["reused_from"]["run"] == first.name
    assert again.readings == record.readings
    text = (second / "report.md").read_text()
    assert f"This census is from run {first.name}, not from this run." in text


def test_a_rebuilt_report_keeps_the_census_section(llama_project, census, capsys, monkeypatch):
    _, _, run_dir, _ = _analysed(llama_project, capsys, monkeypatch)
    before = (run_dir / "report.md").read_text().split("## GPU census")[1].split("\n## ")[0]
    assert main(["report", run_dir.name, "--project", str(llama_project)]) == 0
    capsys.readouterr()
    after = (run_dir / "report.md").read_text()
    assert after.split("## GPU census")[1].split("\n## ")[0] == before


def test_a_census_setup_error_never_fails_the_analysis(llama_project, census, capsys, monkeypatch):
    def broken(index):
        raise OSError("nvidia-smi vanished")

    monkeypatch.setattr("tensward.census.start.gpu_memory_mib", broken)
    code, out, run_dir, record = _analysed(llama_project, capsys, monkeypatch)
    assert code == 0 and (run_dir / "report.md").is_file()
    assert record.status == "failed" and "nvidia-smi vanished" in record.reason


def test_ctrl_c_in_the_census_start_still_stops_the_analysis_when_the_cleanup_fails(
    llama_project, census, capsys, monkeypatch
):
    from interrupts import signal_once_present

    def broken(index):
        raise OSError("nvidia-smi vanished")

    calls = iter([None])
    monkeypatch.setattr("tensward.census.start.gpu_memory_mib", lambda index: next(calls, None))
    monkeypatch.setattr("tensward.census.start.memory_returned", broken)
    signal_once_present((llama_project / "runs", "*/census/serve.json"), signal.SIGINT)
    code, _, _, record = _analysed(llama_project, capsys, monkeypatch)
    assert code != 0
    assert record is not None and record.status == "failed"


LOAD_MODE_RECORD = {
    "id": "llamacpp/test/load-mode-none",
    "rev": 1,
    "kind": "lever",
    "title": "Read the weights without mapping them",
    "status": "measured",
    "engine": "llamacpp",
    "engine_version": ">=11669,<13000",
    "condition": {"cmp": [{"fact": "census.h2d_bytes_per_request"}, ">", {"lit": 1000}]},
    "condition_text": "census.h2d_bytes_per_request > 1000",
    "change": {"lever_key": "load_mode=none", "flags": {"--load-mode": "none"}},
    "effect": {
        "throughput": {"direction": "improves", "basis": "measured", "reason": "test"},
        "reason": "test",
    },
    "costs": {"text": "test", "answer_risk": "none", "changes_answers": False},
    "validity": {"recheck_on_engine_minor": True},
    "evidence": [{"kind": "agent", "ref": "probe:test", "note": "test"}],
    "confidence": {"tier": "low", "grade": "ours"},
    "distribution": "pack",
    "facts_version": 5,
}


def test_a_census_reading_offers_a_change_that_try_runs_and_the_kept_setup_is_measured_again(
    llama_project, census, capsys, monkeypatch
):
    local = (record_from_json(LOAD_MODE_RECORD, layer="local"),)
    monkeypatch.setattr(kb_local, "local_records", lambda: local)
    code, out, _ = _cheap(llama_project, capsys, monkeypatch)
    assert code == 0
    (base,) = (llama_project / "runs").iterdir()
    first = read_record(base)
    h2d = first.readings["census.h2d_bytes_per_request"]
    assert first.status == "ok" and h2d > 1000
    names = [block["name"] for block in _record_offers(base)]
    block = _record_offers(base)[names.index("load_mode=none")]
    assert block["kb"]["census_facts"] == {"census.h2d_bytes_per_request": h2d}
    assert re.search(r"GPU census: [^\n]*; it points to `--load-mode none` below", out)
    monkeypatch.delenv("TENSWARD_JSON", raising=False)
    monkeypatch.setenv("FAKE_FAST_FLAG", "--load-mode")
    with _l4(), _cheap_hash():
        code = main(["try", str(names.index("load_mode=none") + 1), "--project",
                     str(llama_project), "--yes", "--local-command",
                     shlex.join([sys.executable, str(FAKE)])])  # fmt: skip
    tried = capsys.readouterr()
    assert code == 0, tried.err + tried.out
    kept = read_kept(llama_project)
    assert "load-mode=none" in kept.engine_args
    step = step_block(llama_project / "runs" / kept.run)
    assert step["census_facts"] == block["kb"]["census_facts"]
    assert step["kb_record"] == block["kb"]["record"]
    refreshed = read_record(llama_project / "runs" / kept.run)
    assert refreshed.status == "ok" and refreshed.refresh_of == base.name
    starts = [json.loads(line) for line in census.read_text().splitlines()]
    assert sum(bool(s["CUDA_INJECTION64_PATH"]) for s in starts) == 2
    assert re.search(r"If it is kept, a GPU census start may follow, about \d+ min more", tried.err)
    menu = tried.out.split("\nNext:\n")[1]
    following = offers(llama_project)[0]
    words = _words(following.engine_args, following.engine_env, frozenset(kept.engine_args))
    (line,) = [text for text in menu.splitlines() if text.startswith("  tensward try ")]
    shown = line.split("try", 2)[2].split()
    assert shown and set(shown) <= set(words.split())


def _kept_run(project, name, *, census=None, report=None, snapshot="s1"):
    from tensward.census.record import CensusRecord, write_record

    run = project / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    if census:
        write_record(run, CensusRecord(status=census, reason=""))
    (run / "run.json").write_text(json.dumps({"snapshot_id": snapshot}))
    if report is not None:
        (run / "report.json").write_text(json.dumps({"kb": report}))
    return run


def test_the_refresh_time_line_needs_a_loaded_record_that_reads_a_census_fact(
    tmp_path, monkeypatch
):
    from tensward.census import refresh

    monkeypatch.setattr(refresh, "expected_census_s", lambda *a: 90.0)
    plain = _kept_run(tmp_path, "r1", census="ok", report={"pack": None})
    reading = _kept_run(tmp_path, "r2", census="ok", report={"pack": None, "census_readers": 2})
    assert refresh.expected_refresh_s(plain, None, None, None) is None
    assert refresh.expected_refresh_s(reading, None, None, None) == 90.0


def test_a_kept_try_without_a_refresh_does_not_block_a_later_refresh(tmp_path, monkeypatch):
    from tensward.census import refresh

    monkeypatch.setattr(refresh, "expected_census_s", lambda *a: 90.0)
    _kept_run(tmp_path, "r1", census="ok")
    middle = _kept_run(tmp_path, "r2", report={"pack": None, "census_readers": 1})
    (tmp_path / "kept.json").write_text(
        json.dumps({"run": "r2", "previous": "r1", "kept_history": [{"run": "r1"}]})
    )
    assert refresh.expected_refresh_s(middle, None, None, None) == 90.0


def test_a_census_from_another_registration_is_never_borrowed(tmp_path, monkeypatch):
    from tensward.census import refresh

    monkeypatch.setattr(refresh, "expected_census_s", lambda *a: 90.0)
    _kept_run(tmp_path, "r1", census="ok", snapshot="old")
    middle = _kept_run(tmp_path, "r2", report={"pack": None, "census_readers": 1})
    (tmp_path / "kept.json").write_text(
        json.dumps({"run": "r2", "previous": "r1", "kept_history": [{"run": "r1"}]})
    )
    assert refresh.expected_refresh_s(middle, None, None, None) is None


P1_PROOF = Path(__file__).parent / "fixtures" / "census-p1-requests-failed.json"


def test_a_census_whose_requests_all_failed_says_so_and_not_that_it_recorded_no_kernels():
    from tensward.census.proof import AttachProof
    from tensward.census.start import _judged, _Seen

    fields = AttachProof.__dataclass_fields__
    saved = json.loads(P1_PROOF.read_text())["proof"]
    proof = AttachProof(**{k: v for k, v in saved.items() if k in fields} | {"pids": ()})
    log = "request (5385 tokens) exceeds the available context size (4096 tokens)"
    failed = window({}, {}, requests=0, start_ns=0, end_ns=1, timelines={}, failed=21)
    seen = _Seen(proof, failed, 10.0, log)
    record = _judged(seen, ENGINES["llamacpp"], "Tesla V100", True, None)
    assert record.status == "failed"
    assert record.reason == (
        "its requests failed (21 of 21): request (5385 tokens) exceeds the available "
        "context size (4096 tokens)"
    )


def test_an_analysis_with_streamed_host_experts_offers_the_ubatch_trade(
    tmp_path, census, capsys, monkeypatch
):
    """The P2 shape through ``analyse``: a 30B-A3B with 12 blocks of experts on the host, long
    prompts and too little device room for -ub 2048; the census start reads the host weights
    from the placement, sees them copied many times per request while no kernel runs, and the
    trade is offered with the readings it fired on (no knowledge pack installed)."""
    root = tmp_path / "moe"
    root.mkdir()
    (file,) = write_gguf(root / "qwen3-30b-a3b-q6_k.gguf", _describe("qwen3-30b-a3b-q6_k"))
    command = f"llama-server -m {file} -ngl all --n-cpu-moe 12 --fit off -c 8192 -np 2"
    long_chats = [
        {"id": f"doc{n}", "messages": [{"role": "user", "content": " ".join(
            f"doc{n}w{i}" for i in range(2100))}]}
        for n in range(8)
    ]  # fmt: skip
    with _l4(), _cheap_hash(), pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_CACHE_HOME", str(root / "cache"))
        workload = {"request_count": 8, "output_tokens": 16,
                    "arrival": {"kind": "closed_loop", "concurrency": 2}}  # fmt: skip
        assert _register(root / "project", file, root, "--current", command, workload=workload,
                         chats=long_chats) == 0  # fmt: skip
    env = {"FAKE_DEVICE_FREE_MIB": "1000", "FAKE_CENSUS_H2D_PER_KERNEL": "1000000000",
           "FAKE_CENSUS_COPY_SHARE": "0.4"}  # fmt: skip
    code, err, run_dir, record = _analysed(root / "project", capsys, monkeypatch, env=env)
    assert code == 0, err
    passes = record.readings["census.host_passes_per_request"]
    assert passes >= 4 and record.readings["census.copy_wait_share"] >= 0.20
    offers = {block["name"]: block for block in _record_offers(run_dir)}
    trade = offers["llamacpp-raise-ubatch-expert-trade"]
    assert "--n-cpu-moe 12 to" in trade["reason"] and "TPOT rises" in trade["reason"]
    assert trade["kb"]["record"] is None
    assert trade["kb"]["census_facts"]["census.host_passes_per_request"] == passes
