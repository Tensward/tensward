"""End-to-end tests of the llama.cpp engine through the CLI, against a fake llama-server."""

import contextvars
import hashlib
import io
import itertools
import json
import os
import re
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import (
    AbstractContextManager,
    contextmanager,
    nullcontext,
    redirect_stderr,
    redirect_stdout,
)
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from alternative_records import RECORDS
from registration_fixtures import (
    REGISTRATION_CONFIG,
    without_mtp,
    write_config,
    write_gguf,
    write_prompts,
)

from tensward import alternatives, ceilings, measure
from tensward import project as projects
from tensward.alternatives import parse as parse_alternatives
from tensward.anatomy import ModelAnatomy
from tensward.capabilities import Resolved
from tensward.cli import main
from tensward.compare import Answer
from tensward.detect import Hardware
from tensward.engines import ENGINES
from tensward.engines.llamacpp.advice import _host_cache
from tensward.engines.llamacpp.command import list_devices, working_set_mib
from tensward.engines.llamacpp.engine import LlamaCppEngine
from tensward.engines.llamacpp.placement import (
    Fit2,
    Placement,
    _wired_limit_step,
    fit_domains,
    kv_pool_bytes,
    placement,
    wired_limit_advice,
)
from tensward.engines.llamacpp.startup import read_startup
from tensward.formats.gguf import Gguf
from tensward.platforms import Device, darwin, hardware_spec, nvidia
from tensward.platforms.apple import arch_of, chip_of
from tensward.platforms.cpu import host_facts
from tensward.playbook import entry_tables
from tensward.runtime import select_loopback_port

CHAT_PROMPTS = [
    {"id": "a", "messages": [{"role": "user", "content": "synthetic private prompt"}]},
    {"id": "b", "messages": [{"role": "user", "content": "another synthetic prompt"}]},
]
HEADERS = Path(__file__).resolve().parent / "gguf_headers"
FAKE = Path(__file__).resolve().parent / "fake_llamacpp_server.py"
SMI = Path(__file__).resolve().parent / "fake_nvidia_smi.py"
MAC_M4 = Path(__file__).resolve().parent / "darwin_mac_m4.json"
_CALLS = itertools.count()
MIB = 2**20
REAL_DETECTION = (
    (ceilings, "detect_gpus", ceilings.detect_gpus),
    (ceilings, "hardware_spec", ceilings.hardware_spec),
    (projects, "detect_devices", projects.detect_devices),
    (measure, "detect_devices", measure.detect_devices),
)  # the module-scoped L4 fixture leaves its patches of these in place
L4 = Device("NVIDIA L4", 23034 * 2**20, 0)


@pytest.fixture(autouse=True, scope="module")
def _sixteen_cpus() -> Iterator[None]:
    """The fake logs six threads; hosts with fewer usable CPUs would be offered a thread cut."""
    real = host_facts
    with pytest.MonkeyPatch.context() as patch:
        for site in ("engines.llamacpp.probes", "engines.llamacpp.engine", "measure"):
            patch.setattr(
                f"tensward.{site}.host_facts",
                lambda pid=None: replace(real(pid), usable_cpus=16, logical_cpus=16),
            )
        yield


def _describe(name: str, **extra: object) -> dict:
    return {**json.loads((HEADERS / f"{name}.json").read_text()), **extra}


@contextmanager
def _cheap_hash() -> Iterator[None]:
    """Registration hashes every byte of a model; the sparse fixtures are gigabytes of holes."""

    def fixed(root: Path, declared: list[tuple[str, str]], **_: object) -> str:
        return hashlib.sha256(repr((str(root), declared)).encode()).hexdigest()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("tensward.formats.gguf.fingerprint_files", fixed)
        yield


def _register(
    project: Path,
    model: Path | None,
    tmp_path: Path,
    *extra: str,
    precision: tuple[str, str] = ("gguf", "engine"),
    workload: dict | None = None,
    engine: bool = True,
    chats: list[dict] | None = None,
    serving: dict | None = None,
) -> int:
    case = {
        **REGISTRATION_CONFIG["case"],
        **(serving or {}),
        "weight_precision": precision[0],
        "activation_dtype": precision[1],
    }
    case.pop("gpu_memory_utilization", None)
    config = write_config(
        tmp_path / "config.json",
        {
            **REGISTRATION_CONFIG,
            "case": case,
            "workload": {**REGISTRATION_CONFIG["workload"], "api": "chat", **(workload or {})},
        },
    )
    prompts = write_prompts(tmp_path / "prompts.jsonl", chats or CHAT_PROMPTS)
    return main(
        [
            "init",
            "--json",
            "--project",
            str(project),
            *(["--model", str(model)] if model else []),
            "--config",
            str(config),
        ]
        + ["--prompts", str(prompts), *(["--engine", "llamacpp"] if engine else []), *extra]
    )


def _init(
    capsys: pytest.CaptureFixture[str],
    project: Path,
    model: Path | None,
    tmp_path: Path,
    *extra: str,
    precision: tuple[str, str] = ("gguf", "engine"),
    engine: bool = True,
    serving: dict | None = None,
) -> tuple[int, str, str]:
    code = _register(
        project, model, tmp_path, *extra, precision=precision, engine=engine, serving=serving
    )
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@contextmanager
def _l4() -> Iterator[None]:
    """Detection finds one NVIDIA L4, as test_analyse.py has it. With FAKE_BW_DEV_GBS set, its
    datasheet row has that memory bandwidth, the one the fake's pacing uses."""

    def spec(name: str) -> object:
        row = hardware_spec(name)
        bandwidth = os.environ.get("FAKE_BW_DEV_GBS")
        return (
            row
            if row is None or bandwidth is None
            else replace(row, bandwidth_gbs=float(bandwidth))
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("tensward.ceilings.hardware_spec", spec)
        patch.setattr("tensward.ceilings.detect_gpus", lambda: (("NVIDIA L4", 23034),))
        for module in ("project", "measure"):
            patch.setattr(f"tensward.{module}.detect_devices", lambda: (L4,))
        patch.setattr(type(nvidia.NVIDIA), "detect", lambda self: (L4,))
        yield


@contextmanager
def _apple() -> Iterator[None]:
    """Detection finds an M4 Mac: the macOS tools answer from the recorded outputs in
    darwin_mac_m4.json, by command line."""
    recorded = json.loads(MAC_M4.read_text())["outputs"]

    def run(argv: Sequence[str]) -> str | None:
        return recorded.get(re.sub(r"-p \d+", "-p PID", " ".join(argv)))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(darwin, "RUN", run)
        patch.setattr(darwin, "on_mac", lambda: True)
        patch.setattr(type(nvidia.NVIDIA), "detect", lambda self: ())
        for module, name, original in REAL_DETECTION:
            patch.setattr(module, name, original)
        yield


@contextmanager
def _no_gpu() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("tensward.ceilings.detect_gpus", lambda: ())
        for module in ("project", "measure"):
            patch.setattr(f"tensward.{module}.detect_devices", lambda: ())
        patch.setattr(type(nvidia.NVIDIA), "detect", lambda self: ())
        yield


@pytest.fixture(scope="module")
def qwen_project(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """One llama.cpp project shared by the analyse cases: Qwen3-0.6B served with two slots to
    eight clients. The command sets no -c: --fit only adjusts unset arguments."""
    root = tmp_path_factory.mktemp("qwen")
    (file,) = write_gguf(root / "Qwen3-0.6B-Q8_0.gguf", _describe("qwen3-0.6b-q8_0"))
    command = f"llama-server -m {file} -np 2 --port 9931"
    with _l4(), _cheap_hash(), pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_CACHE_HOME", str(root / "cache"))
        workload = {"request_count": 16, "arrival": {"kind": "closed_loop", "concurrency": 8}}
        assert _register(root / "project", file, root, "--current", command, workload=workload) == 0
        yield root / "project"


def _shim(directory: Path) -> Path:
    """An executable ``llama-server`` that runs the fake: the program an ``init --current``
    command names."""
    shim = directory / "llama-server"
    shim.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(FAKE))} "$@"\n'
    )
    shim.chmod(0o755)
    return shim


@pytest.fixture(scope="module")
def apple_gemma_project(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Gemma-4-E4B registered on an M4 Mac with a Metal build; what init printed is in
    init.txt beside the project."""
    root = tmp_path_factory.mktemp("apple")
    (file,) = write_gguf(root / "gemma-4-E4B-it-Q4_K_M.gguf", _describe("gemma4-e4b-q4_k_m"))
    command = f"{_shim(root)} -m {file} -ngl 99 -np 4 -c 8192"
    printed = io.StringIO()
    with _apple(), _cheap_hash(), pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_CACHE_HOME", str(root / "cache"))
        patch.setenv("FAKE_BACKEND", "apple")
        with redirect_stdout(printed), redirect_stderr(printed):
            assert _register(root / "project", file, root, "--current", command) == 0
    (root / "init.txt").write_text(printed.getvalue())
    yield root / "project"


def _analyse(
    project: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    *extra: str,
    env: dict[str, str] | None = None,
    cache: str | None = None,
    gpu: bool = True,
    machine: Callable[[], AbstractContextManager[None]] = _l4,
    smi_row: str = "20,40.0,70.00,1230,1590,Not Active,Not Active,Not Active",
) -> tuple[int, str, str]:
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    if free := (env or {}).get("FAKE_DEVICE_FREE_MIB"):
        monkeypatch.setattr("tensward.platforms.nvidia.SAMPLE_COMMAND", (sys.executable, str(SMI)))
        monkeypatch.setenv("FAKE_SMI_ROW", smi_row)
        monkeypatch.setenv(
            "FAKE_SMI_MEMORY", f"{L4.total_bytes // 2**20 - int(free)},{L4.total_bytes // 2**20}"
        )
    shared = project.parent.parent / f"cache-{cache}" if cache else None
    monkeypatch.setenv("XDG_CACHE_HOME", str(shared or project.parent / f"cache-{next(_CALLS)}"))
    command = shlex.join([sys.executable, str(FAKE)])
    with machine() if gpu else _no_gpu():
        code = main(["analyse", "--project", str(project), "--local-command", command,
                     "--ready-timeout", "30", *extra])  # fmt: skip
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _offers(run_dir: Path) -> list[dict]:
    """The suggestions of a run's report, none of them scoped to another engine."""
    report = json.loads((run_dir / "report.json").read_text())
    found = [b for s in report["sections"] for b in s["blocks"] if b.get("kind") == "suggestion"]
    engines = {table["name"]: table["engines"] for table in entry_tables()}
    assert all("llamacpp" in engines[b["name"]] for b in found)
    return found


def _step_args(offer: dict) -> list[str]:
    words = shlex.split(offer["command"])
    return [words[i + 1] for i, word in enumerate(words) if word == "--engine-arg"]


def _throughput(run_dir: Path) -> float:
    return json.loads((run_dir / "metrics.json").read_text())["request_throughput"]


def _argv(run_dir: Path) -> list[str]:
    return json.loads((run_dir / "serve.json").read_text())["identity"]["argv"]


def _after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


@pytest.mark.parametrize(
    ("case", "kv_layers", "state_mib"),
    [
        ("qwen3.8-27b-ud-q4_k_m", 16, 149.625),
        ("qwen3.5-35b-a3b-q4_k_m", 10, 62.8125),
        ("nemotron-nano-9b-v2-q4_k_m", 4, 138.796875),
        ("lfm2-8b-a1b-q4_k_m", 6, 0.28125),
    ],
)
def test_a_hybrid_gguf_prices_attention_layers_and_state(
    case: str,
    kv_layers: int,
    state_mib: float,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (file,) = write_gguf(tmp_path / "m.gguf", _describe(case))
    command = f"llama-server -m {file} -np 2 --port 9931"
    with _l4(), _cheap_hash(), pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        code, out, err = _init(capsys, tmp_path / "project", file, tmp_path, "--current", command)
    assert code == 0, err
    anatomy = json.loads(out)["anatomy"]
    assert anatomy["linear_attention"] and anatomy["state_bytes_per_seq"] / MIB == state_mib
    assert sum(g["layers"] for g in anatomy["attention"]) == kv_layers
    with _cheap_hash():
        code, _, err = _analyse(tmp_path / "project", capsys, monkeypatch)
    assert code == 0, err


@pytest.mark.parametrize(
    "case",
    ["baseline", "pinned", "old_build_help", "fit_cut", "kv_full", "hf_refused"],
)
def test_llamacpp_analyse(
    case: str,
    qwen_project: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if case == "hf_refused":
        (file,) = write_gguf(tmp_path / "m.gguf", _describe("qwen3-0.6b-q8_0"))
        command = "llama-server -hf Qwen/Qwen3-0.6B-GGUF"
        code, _, err = _init(capsys, tmp_path / "hf", file, tmp_path, "--current", command)
        assert code == 3 and "pass the local file" in err
        return
    if case == "pinned":
        baseline = max((qwen_project / "runs").iterdir())
        levers = json.loads((baseline / "metrics.json").read_text())["resolved"]["levers"]
        (first,) = (o for o in _offers(baseline) if o["tier"] == "try_first")
        assert first["name"] == "llamacpp-slots"
        code, _, err = _analyse(
            qwen_project,
            capsys,
            monkeypatch,
            *itertools.chain(*(("--engine-arg", a) for a in _step_args(first))),
        )
        assert code == 0, err
        step = max((qwen_project / "runs").iterdir())
        argv = _argv(step)
        assert _after(argv, "--fit") == "off" and _after(_argv(baseline), "--fit") == "on"
        assert int(_after(argv, "--parallel")) > levers["max_concurrent_requests"]
        slot_ctx = levers["llamacpp.kv_pool_tokens"] // levers["max_concurrent_requests"]
        assert int(_after(argv, "--ctx-size")) == slot_ctx * int(_after(argv, "--parallel"))
        assert {o["name"] for o in _offers(step)} <= {"llamacpp-slots", "trim-max-context-len"}
        assert int(_after(argv, "--n-gpu-layers")) == levers["llamacpp.n_gpu_layers"]
        assert _throughput(step) > _throughput(baseline)
        assert main(["compare", "--project", str(qwen_project), baseline.name, step.name]) == 0
        capsys.readouterr()
        assert (
            main(["set", step.name, "--project", str(qwen_project), "--accept-answer-changes"]) == 0
        )
        line = next(t for t in capsys.readouterr().out.splitlines() if "as measured:" in t)
        command = shlex.split(line.partition("as measured: ")[2])
        assert command[0] == "llama-server" and ("--kv-unified" in command) == (
            "--kv-unified" in argv
        )
        assert ("--no-kv-unified" in command) == ("--no-kv-unified" in argv)
        assert not {"-lv", "--metrics", "--slots"} & set(command)
        (qwen_project / "kept.json").unlink()
        code, _, err = _analyse(qwen_project, capsys, monkeypatch, "--engine-arg", "flash-attn=off")
        assert code == 0, err
        flash = max((qwen_project / "runs").iterdir())
        assert _after(_argv(flash), "--flash-attn") == "off"
        assert "flash attention ran off (requested)" in (flash / "report.md").read_text()
        offers = {o["name"]: o for o in _offers(flash)}
        assert "--engine-arg flash-attn=on" in offers["flash-attn"]["command"]
        assert "kv-q8" not in offers
        return
    env = {}
    if case == "old_build_help":
        env = {
            "FAKE_LLAMA_HELP_DROP": "--n-cpu-moe,--load-mode,--cache-type-k",
            "FAKE_KV_FULL": "1",
        }
    env |= {"FAKE_DEVICE_FREE_MIB": "20000"} if case == "baseline" else {}
    env |= {"FAKE_DEVICE_FREE_MIB": "5000"} if case == "old_build_help" else {}
    env |= {"FAKE_FIT_CTX": "70", "FAKE_BACKEND": "metal"} if case == "fit_cut" else {}
    env |= {"FAKE_KV_FULL": "1", "FAKE_THINKS": "1"} if case == "kv_full" else {}
    again = ("--remeasure",) if case != "baseline" else ()
    code, out, err = _analyse(qwen_project, capsys, monkeypatch, *again, env=env)
    assert code == 0, err
    assert "Ran: llama.cpp build 11669" in out.splitlines()[1]
    assert "vLLM" not in out and "vllm" not in out
    run_dir = max((qwen_project / "runs").iterdir())
    argv = _argv(run_dir)
    assert {"--port", "--metrics", "--slots", "--fit", "on"} <= set(argv)
    assert argv[argv.index("-lv") + 1] == "4" and ("--kv-unified" in argv) != (
        "--no-kv-unified" in argv
    )
    metrics = json.loads((run_dir / "metrics.json").read_text())
    resolved, diagnosis = metrics["resolved"], metrics["diagnosis"]
    assert resolved["levers"]["llamacpp.kv_pool_tokens"] > 0
    assert resolved["levers"]["max_concurrent_requests"] == 2
    if case == "fit_cut":
        assert resolved["levers"]["llamacpp.kv_pool_tokens"] == 70
        assert resolved["facts"]["fit_changed"] == ["ctx_size"]
        assert resolved["facts"]["device_weight_bytes"] > 0
        assert resolved["levers"]["llamacpp.n_gpu_layers"] > 0
    if case == "baseline":
        assert "Bottleneck: queueing" in out
        findings = {f["bottleneck"]: f for f in diagnosis["findings"]}
        assert findings["offload"]["state"] == "clear"
        assert findings["decode_bandwidth"]["state"] != "cant_tell"
    if case == "kv_full":
        kv = next(f for f in diagnosis["findings"] if f["bottleneck"] == "kv_capacity")
        assert kv["state"] in ("critical", "warning")
        report = (run_dir / "report.md").read_text()
        assert "thinking" in report and "enable_thinking" in report
        assert "empty answers" not in report
        rows = [json.loads(t) for t in (run_dir / "requests.jsonl").read_text().splitlines()]
        refused = sum(
            r.get("http_status") == 400 and "context size" in (r.get("error") or "") for r in rows
        )
        assert refused and metrics["engine_signals"]["llamacpp.context_exceeded"] == refused
        assert f"{refused} requests were refused for exceeding the context size" in report
    offered = {offer["name"] for offer in _offers(run_dir)}
    if case == "fit_cut":
        assert "fit-context" in offered
    if case == "old_build_help":
        assert "not in this build's --help: --cache-type-k" in (run_dir / "report.md").read_text()
    missing = json.loads((run_dir / "window.json").read_text())["llamacpp"][
        "flags_missing_from_help"
    ]
    assert ("--n-cpu-moe" in missing) == (case == "old_build_help")


@pytest.mark.parametrize(
    ("case", "expect"),
    [
        ("qwen3-0.6b-q8_0", {"layers": 28, "kv_f16_mib_2048": 224, "kv_q8_mib_2048": 119}),
        ("gemma4-e4b-q4_k_m", {"sliding_window": 512}),
        ("shards", {"files": 3}),
        ("two_models", {"refused": "more than one model file"}),
        ("truncated", {"refused": "checkpoint_layout_invalid"}),
        ("bf16_case", {"refused": "weight_precision gguf with activation_dtype engine"}),
    ],
)
def test_gguf_registration(
    case: str, expect: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    project = tmp_path / "project"
    small = _describe("qwen3-0.6b-q8_0")
    with nullcontext() if case == "qwen3-0.6b-q8_0" else _cheap_hash():
        if case == "qwen3-0.6b-q8_0":
            (file,) = write_gguf(models / "Qwen3-0.6B-Q8_0.gguf", small)
            code, out, err = _init(
                capsys, project, file, tmp_path, serving={"max_model_len": 1024, "max_num_seqs": 4}
            )
            assert code == 0, err
            record = json.loads((project / "project.json").read_text())
            assert record["current_setup"]["settings"]["max_context_len"] == 1024 * 4
            anatomy = json.loads(out)["anatomy"]
            per_token = anatomy["kv_bytes_per_token"]
            assert anatomy["layers"] == expect["layers"]
            assert per_token["f16"] * 2048 / 2**20 == expect["kv_f16_mib_2048"]
            assert per_token["q8_0"] * 2048 / 2**20 == expect["kv_q8_mib_2048"]
        elif case == "gemma4-e4b-q4_k_m":
            (file,) = write_gguf(models / "gemma-4-E4B-it-Q4_K_M.gguf", _describe(case))
            code, out, err = _init(capsys, project, file, tmp_path)
            assert code == 0, err
            assert json.loads(out)["anatomy"]["sliding_window"] == expect["sliding_window"]
        elif case == "shards":
            write_gguf(models / "Qwen3-0.6B-Q8_0.gguf", {**small, "shards": 3})
            code, _, _ = _init(capsys, project, models, tmp_path)
            record = json.loads((project / "project.json").read_text())
            assert code == 0 and record["artifact"]["weights"]["shard_count"] == expect["files"]
            assert record["artifact"]["files"] == {"gguf": "Qwen3-0.6B-Q8_0-00001-of-00003.gguf"}
        elif case == "two_models":
            first, *_ = write_gguf(models / "Qwen3-0.6B-Q8_0.gguf", small)
            second, *_ = write_gguf(
                models / "gemma-4-E4B-it-Q4_K_M.gguf", _describe("gemma4-e4b-q4_k_m")
            )
            (models / "mmproj-gemma.gguf").write_bytes(b"GGUF")
            code, out, err = _init(capsys, project, models, tmp_path)
            refusal = json.loads(err)
            assert code == 3 and out == "" and refusal["code"] == "checkpoint_inventory_unexpected"
            assert expect["refused"] in refusal["message"]
            assert first.name in refusal["message"] and second.name in refusal["message"]
            assert "mmproj" not in refusal["message"]
        elif case == "truncated":
            (file,) = write_gguf(models / "Qwen3-0.6B-Q8_0.gguf", small)
            os.truncate(file, 4096)
            code, _, err = _init(capsys, project, file, tmp_path)
            assert code == 3 and json.loads(err)["code"] == expect["refused"]
        else:
            (file,) = write_gguf(models / "Qwen3-0.6B-Q8_0.gguf", small)
            code, _, err = _init(capsys, project, file, tmp_path, precision=("bf16", "bfloat16"))
            refusal = json.loads(err)
            assert code == 3 and refusal["code"] == "project_config_unsupported"
            assert expect["refused"] in refusal["message"]
            assert not (project / "project.json").exists()


def _serve_fake(
    file: Path, fingerprint: str, seen: Path, **env: str
) -> tuple[subprocess.Popen, str]:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {**os.environ, "FAKE_FINGERPRINT": fingerprint, "FAKE_SEEN_LOG": str(seen), **env}
    server = subprocess.Popen(  # noqa: S603 - the test's own fake
        [sys.executable, str(FAKE), "-m", str(file), "--port", str(port)],
        env=env, stdout=subprocess.DEVNULL,
    )  # fmt: skip
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15
    while True:
        try:
            if httpx.get(f"{url}/health").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        assert time.monotonic() < deadline, "the fake server did not start"
        time.sleep(0.05)
    seen.unlink(missing_ok=True)
    return server, url


@pytest.mark.parametrize(
    ("args", "fingerprint", "expect"),
    [
        (["--current", "/opt/llama/build/bin/llama-server -m {gguf} -ngl 99"], None,
         "llamacpp [from: serve command]"),
        (["--endpoint", "{url}"], "llamacpp",
         "llamacpp [from: endpoint /props + /v1/models owned_by=llamacpp]"),
        ([], None, "llamacpp (assumed from a GGUF file; pass --engine to confirm)"),
        (["--current", "vllm serve {gguf}"], None, "refused: vLLM with a GGUF checkpoint"),
        (["--engine", "vllm", "--current", "llama-server -m {gguf}"], None,
         "refused: --engine vllm conflicts"),
        (["--endpoint", "{url}"], "ollama",
         "refused: built on llama.cpp but not llama-server"),
        (["--current", "ollama serve"], None,
         "refused: built on llama.cpp but not llama-server"),
        (["--from-endpoint", "--endpoint", "{url}", "--current", "{wrapper} -m {gguf} --port 9931"],
         "llamacpp",
         "llamacpp [from: serve command; endpoint /props + /v1/models owned_by=llamacpp]"),
    ],
)  # fmt: skip
def test_engine_detection(
    args: list[str],
    fingerprint: str | None,
    expect: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    for name in [n for n in os.environ if n.startswith(("VLLM_", "LLAMA_ARG_"))]:
        monkeypatch.delenv(name)
    (file,) = write_gguf(tmp_path / "Qwen3-0.6B-Q8_0.gguf", _describe("qwen3-0.6b-q8_0"))
    wrapper = _shim(tmp_path)
    from_endpoint = "--from-endpoint" in args
    seen = tmp_path / "seen.txt"
    server, url = (
        _serve_fake(file, fingerprint, seen, FAKE_BUILD="b11811-1a2b3c4d")
        if fingerprint
        else (None, "")
    )
    try:
        with _cheap_hash():
            words = [
                word.format(gguf=file, url=url, wrapper=wrapper)
                for word in args
                if word != "--from-endpoint"
            ]
            code, out, err = _init(
                capsys, tmp_path / "project", None if from_endpoint else file, tmp_path,
                *words, engine=False,
            )  # fmt: skip
    finally:
        if server:
            server.terminate()
            server.wait()
    if expect.startswith("refused: "):
        assert code == 3 and out == "" and expect.removeprefix("refused: ") in err
    else:
        assert f"engine: {expect}" in err
        if from_endpoint:
            assert code == 0, err
            assert "the running server is build 11811, the command's binary is build 11669" in err
            assert "Tensward measures 11669" in err
            record = json.loads((tmp_path / "project" / "project.json").read_text())
            assert record["artifact"]["files"] == {"gguf": file.name}
        elif url:
            assert code == 3 and "Needs the command that started it" in err
        else:
            assert code == 0, err
    if fingerprint:
        served = seen.read_text().split()[1::2]
        assert served == ["/health", "/v1/models", "/props"] + (
            ["/api/version"] if fingerprint == "ollama" else []
        )


def test_apple_detect(
    apple_gemma_project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = apple_gemma_project.parent
    printed = (root / "init.txt").read_text()
    assert "vLLM" not in printed and "MLX" not in printed
    monkeypatch.setenv("XDG_CACHE_HOME", str(root / "cache"))
    with _apple():
        assert main(["env", "--json"]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["platform"]["name"] == "apple"
        (device,) = report["platform"]["devices"]
        assert device["name"] == "Apple M4" and device["gpu_cores"] == 10
        facts = darwin.host_facts()
        assert (facts.p_cores, facts.e_cores) == (4, 6)
        assert working_set_mib(list_devices(str(root / "llama-server")) or "") == 18186
        assert darwin.process_start(4321) == "Thu Oct 29 15:36:27 2026"
        assert darwin.sample_process(4321).major_faults == 0  # ps prints "-" for majflt
    arm, intel = (
        struct.pack("<II", 0xFEEDFACF, 0x0100000C),
        struct.pack("<II", 0xFEEDFACF, 0x01000007),
    )
    fat = struct.pack(">II", 0xCAFEBABE, 2) + b"".join(
        struct.pack(">IIIII", cpu, 0, 0, 0, 14) for cpu in (0x0100000C, 0x01000007)
    )
    kinds = []
    for name, header in (("arm", arm), ("intel", intel), ("fat", fat)):
        (root / name).write_bytes(header)
        kinds.append(darwin.mach_o_arch(root / name))
    assert kinds == ["arm64", "x86_64", "universal"]


APPLE_PACING = {"FAKE_BW_DEV_GBS": "80", "FAKE_BW_HOST_GBS": "30"}


def test_apple_full_metal(
    apple_gemma_project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weights in MTL0_Mapped, compute in MTL0_Private: the launch is full GPU on a 120 GB/s M4."""
    env = {"FAKE_BACKEND": "apple", "FAKE_METAL_PRIVATE": "1", **APPLE_PACING}
    with _cheap_hash():
        code, _, err = _analyse(apple_gemma_project, capsys, monkeypatch, env=env, machine=_apple)
    assert code == 0, err
    run_dir = max((apple_gemma_project / "runs").iterdir())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    block = metrics["placement"]
    assert block["regime"]["placement"] == "full_gpu" and block["fit"]["fits"]
    assert block["fit"]["device_free"] == 18186 * MIB and block["next_steps"] == []
    findings = {f["bottleneck"]: f for f in metrics["diagnosis"]["findings"]}
    assert findings["offload"]["state"] == "clear"
    assert findings["decode_bandwidth"]["confidence"] == "possible"
    assert findings["gpu_saturated"]["state"] == "cant_tell"
    assert metrics["ceilings"]["bandwidth_gbs"] == 120
    assert metrics["ceilings"]["bandwidth_source"] == "datasheet"
    assert "pcie" not in (json.loads((run_dir / "run.json").read_text()).get("probes") or {})
    report = (run_dir / "report.md").read_text()
    assert "NVIDIA" not in report and "wired_limit" not in report and "--trace" not in report
    assert "load client ran on this Mac" in report and "Apple devices cannot be sampled" in report
    facts = metrics["resolved"]["facts"]
    assert facts["working_set_bytes"] == 18186 * MIB and facts["graph_splits"] == 2
    kb = json.loads((run_dir / "facts.json").read_text())["facts"]
    assert kb["memory.topology"] == "unified" and kb["gpu.arch"] == "apple_m4"
    assert kb["engine.backend"] == "metal" and kb["os.family"] == "macos"
    assert kb["metal.chip"] == "M4" and kb["metal.gpu_cores"] == 10
    assert kb["host.p_cores"] == 4 and kb["metal.working_set_gib"] == 18186 / 1024
    noted: list[object] = []
    note = LlamaCppEngine.failure_note
    monkeypatch.setattr(
        LlamaCppEngine, "failure_note",
        lambda self, *args: noted.append(args[-1]) or note(self, *args),
    )  # fmt: skip
    with _cheap_hash():
        code, _, err = _analyse(
            apple_gemma_project, capsys, monkeypatch, env={**env, "FAKE_METAL_OOM": "1"},
            machine=_apple,
        )  # fmt: skip
    assert code == 1 and "the Metal working set is the bound" in err
    assert len(noted) == 1 and noted[0] is not None
    assert chip_of("Apple M4 Max", 32).chip == "M4 Max" and arch_of("Apple M4 Max") == "apple_m4"
    assert kb["run.graph_splits"] == 2 and kb["run.metal_bytes"] > 0


M4_PRO = {
    "sysctl -n machdep.cpu.brand_string": "Apple M4 Pro\n",
    "sysctl -n hw.memsize": f"{48 * 2**30}\n",
    "sysctl -n hw.logicalcpu": "14\n",
    "sysctl -n hw.physicalcpu": "14\n",
    "sysctl -n hw.perflevel0.physicalcpu": "10\n",
    "sysctl -n hw.perflevel0.logicalcpu": "10\n",
    "sysctl -n hw.perflevel1.physicalcpu": "4\n",
    "sysctl -n hw.perflevel1.logicalcpu": "4\n",
}


@contextmanager
def _apple_pro() -> Iterator[None]:
    """The recorded M4 outputs with an M4 Pro's chip, cores and 48 GiB of memory, 40 GiB of it
    available."""
    with _apple(), pytest.MonkeyPatch.context() as patch:
        recorded = darwin.RUN
        others = sum(
            int(n)
            for n in re.findall(
                r"Pages (?:inactive|speculative|purgeable):\s+(\d+)", recorded(["vm_stat"]) or ""
            )
        )

        def run(argv: Sequence[str]) -> str | None:
            text = M4_PRO.get(" ".join(argv)) or recorded(argv)
            if text and argv[0] == "vm_stat":
                free = 40 * 2**30 // 16384 - others
                return re.sub(r"(Pages free:\s+)\d+", rf"\g<1>{free}", text)
            if text and argv[0] == "system_profiler":
                text = text.replace('"Apple M4"', '"Apple M4 Pro"').replace(
                    '"sppci_cores" : "10"', '"sppci_cores" : "20"'
                )
            return text

        patch.setattr(darwin, "RUN", run)
        yield


@pytest.fixture(scope="module")
def apple_moe_project(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Qwen3-30B-A3B Q6_K with 28 of 49 layers on the Metal device of a Mac."""
    root = tmp_path_factory.mktemp("apple-moe")
    (file,) = write_gguf(root / "qwen3-30b-a3b-q6_k.gguf", _describe("qwen3-30b-a3b-q6_k"))
    command = f"{_shim(root)} -m {file} -ngl 28 -c 8192 -np 4"
    workload = {"request_count": 8, "output_tokens": 16}
    with _apple_pro(), _cheap_hash(), pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_CACHE_HOME", str(root / "cache"))
        patch.setenv("FAKE_BACKEND", "apple")
        assert _register(root / "project", file, root, "--current", command, workload=workload) == 0
    yield root / "project"


def test_apple_working_set(
    apple_moe_project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 16 GiB working set cannot hold the 23 GiB model on a 48 GiB M4 Pro (273 GB/s)."""
    env = {
        "FAKE_BACKEND": "apple", "FAKE_METAL_WS_MIB": "16384", "FAKE_METAL_PRIVATE": "1",
        "FAKE_BW_DEV_GBS": "273", "FAKE_BW_HOST_GBS": "100",
    }  # fmt: skip
    with _cheap_hash():
        code, _, err = _analyse(apple_moe_project, capsys, monkeypatch, env=env, machine=_apple_pro)
    assert code == 0, err
    run_dir = max((apple_moe_project / "runs").iterdir())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    block = metrics["placement"]
    assert metrics["ceilings"]["gpu"] == "M4 Pro" and metrics["ceilings"]["bandwidth_gbs"] == 273
    assert block["regime"]["placement"] == "dense_offload"
    assert block["placement"]["host_layers"] == 21 and not block["placement"]["host_expert_layers"]
    assert read_startup((run_dir / "server.log").read_text()).bytes_of("compute", "device") > 0
    assert block["fit"]["fits"] and not block["fit_full"]["fits"]
    assert "Metal working set 16.0 GiB" in block["fit_full"]["reason"]
    assert "Metal working set" not in block["fit"]["reason"]
    steps = block["next_steps"]
    levers = [step["lever"] for step in steps]
    assert levers[0] == "host_expert_layers" and "host_layers" not in levers
    (file,) = apple_moe_project.parent.glob("*.gguf")
    embedding = 255_252_480  # token_embd, which always counts as host memory
    need = file.stat().st_size - embedding + 98304 * 8192 + block["fit"]["compute_buffer_bytes"]
    assert levers[-1] == "advice:wired_limit"
    assert abs(steps[-1]["value"] * MIB / (need + 2**30) - 1) < 0.005
    report = (run_dir / "report.md").read_text()
    assert "one 48 GiB pool; Metal working set 16.0 GiB; system memory keeps 6.0 GiB back" in report
    mib = steps[-1]["value"]
    assert f"sudo sysctl iogpu.wired_limit_mb={mib}" in report
    assert "restore the limit this run started with (0 is the system default)" in report
    assert "iogpu.wired_limit_mb=0" in report and "not persistent across reboot" in report
    assert "Tensward does not run it" in report and "then: a Metal working set" not in report
    assert "offload / host memory" in report and "PCIe" not in report
    probes = _probes(run_dir)
    assert probes["host"]["mem_bw_gbs"] > 0 and not probes["pcie"]
    kb = json.loads((run_dir / "facts.json").read_text())["facts"]
    assert kb["memory.topology"] == "unified" and kb["gpu.arch"] == "apple_m4"
    assert kb["engine.backend"] == "metal" and kb["os.family"] == "macos"
    assert kb["metal.chip"] == "M4 Pro" and kb["run.placement"] == "dense_offload"


def test_unified_fit_names_the_binding_bound(tmp_path: Path) -> None:
    """Moving layers between the Metal working set and the CPU changes which bound binds and
    never the system-memory sum; the margin and the reserve are each counted once."""
    (file,) = write_gguf(tmp_path / "q.gguf", _describe("qwen3-14b-q6_k"))
    with _cheap_hash():
        _, anatomy = Gguf().register(file, engine_build="b11669", engine_name="llamacpp")
    host = replace(host_facts(), mem_gib=24.0, mem_available_gib=20.0, wired_limit_mb=0)
    facts = {"metal": True, "device_weight_bytes": 1, "host_weight_bytes": 0}

    def fit(layers: int, available: float, free: int = 10 * 2**30) -> Fit2:
        where = placement(anatomy, Resolved(levers={"host_layers": layers}, facts=facts))
        assert where is not None
        return fit_domains(
            anatomy, where, pool_tokens=8192, type_k="f16", type_v="f16",
            compute_buffer_bytes=2**28, device_free=free,
            host=replace(host, mem_available_gib=available), topology="unified",
        )  # fmt: skip

    full, offloaded = fit(0, 20.0), fit(21, 20.0)
    assert full.device_free == 10 * 2**30
    assert not full.fits and "Metal working set 10.0 GiB" in full.reason
    assert offloaded.fits and offloaded.device_need < full.device_need
    assert offloaded.device_need + offloaded.host_need == full.device_need + full.host_need
    roles = anatomy.weight_bytes_by_role
    weights = sum(roles.values()) - roles["embedding"]
    kv = kv_pool_bytes(anatomy, 8192, "f16", "f16")
    assert full.device_need == weights + kv + 2**28 + 2**30
    tight = fit(21, 6.0)
    assert not tight.fits and "system memory 24 GiB" in tight.reason
    assert "4.0 GiB kept back" in tight.reason
    need = full.device_need - 2**30
    assert wired_limit_advice(need, host) == -(-(need + 2**30) // MIB)
    assert wired_limit_advice(20 * 2**30, host) is None
    assert wired_limit_advice(need, replace(host, wired_limit_mb=None)) is None
    near, roomy, over = (fit(0, 20.0, need + delta) for delta in (100 * MIB, 600 * MIB, -MIB))
    assert near.fits and near.tight and "fit with 100 MiB to spare" in near.reason
    assert roomy.fits and not roomy.tight and not over.fits
    assert _wired_limit_step(near, host) == [] and _wired_limit_step(over, host)
    cached = {
        "topology": "unified", "regime": {"host_memory_tight": True},
        "fit": {"host_free": 4 * 2**30, "device_need": 0, "host_need": 0},
    }  # fmt: skip
    log = "prompt cache update took 13890.74 ms\nprompt cache update took 50.0 ms\n"
    note = _host_cache(cached, None, log)
    assert note and "earlier measurement on a 24 GiB M4" in note
    assert "held the server for 13.9 s (longest 13.9 s)" in note
    assert "held the server" not in (_host_cache(cached, None, "") or "")


def test_apple_dev_none(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Mac run that turns the device off is a CPU run: no device in the run, no GPU ceiling."""
    (file,) = write_gguf(tmp_path / "Qwen3-0.6B-Q8_0.gguf", _describe("qwen3-0.6b-q8_0"))
    command = f"{_shim(tmp_path)} -m {file} --device=none --port 9934"
    workload = {"request_count": 4, "output_tokens": 16}
    project = tmp_path / "project"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    with _cheap_hash():
        with _apple():
            assert _register(project, file, tmp_path, "--current", command, workload=workload) == 0
        capsys.readouterr()
        refused, _, err = _analyse(
            project, capsys, monkeypatch, "--require-gpu", "Apple M4", machine=_apple
        )
        assert refused == 2 and "need a GPU run" in err
        code, _, err = _analyse(
            project,
            capsys,
            monkeypatch,
            env={"FAKE_BACKEND": "cpu", "FAKE_METAL_DEVICE": "1"},
            machine=_apple,
        )
    assert code == 0, err
    run_dir = max((project / "runs").iterdir())
    record = json.loads((run_dir / "run.json").read_text())
    assert record["runtime_platform"] == "cpu" and record["gpus_identity"] == []
    assert json.loads((run_dir / "metrics.json").read_text())["ceilings"]["bandwidth_gbs"] is None
    kb = json.loads((run_dir / "facts.json").read_text())["facts"]
    assert kb["memory.topology"] == "host" and kb["os.family"] == "macos"
    assert kb["host.p_cores"] == 4 and kb["host.e_cores"] == 6
    assert not {name for name in kb if name.startswith(("gpu.", "metal."))}
    assert kb.get("engine.backend") == "cpu"
    assert "PCIe" not in (run_dir / "report.md").read_text()


FAKE_PCIE_GBS = 1000
OFFLOAD = {
    "FAKE_BW_DEV_GBS": "5000",
    "FAKE_BW_HOST_GBS": "1000",
}


@pytest.fixture(scope="module")
def offload_projects(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Path]]:
    """Qwen3-14B left to --fit, and Qwen3-30B-A3B with 30 layers on the device, whose prompts
    are longer than 2,048 tokens."""
    root = tmp_path_factory.mktemp("offload")
    long_chats = [
        {
            "id": f"doc{n}",
            "messages": [{"role": "user", "content": " ".join(f"doc{n}w{i}" for i in range(2100))}],
        }
        for n in range(16)
    ]
    projects = {}
    with _l4(), _cheap_hash(), pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_CACHE_HOME", str(root / "cache"))
        for name, header, flags, workload, chats in (
            ("dense", "qwen3-14b-q6_k", "", {"request_count": 12}, None),
            ("moe_fit", "qwen3-30b-a3b-q6_k", "", {"request_count": 4, "output_tokens": 16}, None),
            (
                "moe",
                "qwen3-30b-a3b-q6_k",
                "-ngl 30",
                {"request_count": 8, "output_tokens": 16},
                long_chats,
            ),
        ):
            folder = root / name
            folder.mkdir()
            (file,) = write_gguf(folder / f"{header}.gguf", _describe(header))
            command = f"llama-server -m {file} {flags} --port 9932".replace("  ", " ")
            code = _register(
                folder / "project",
                file,
                folder,
                "--current",
                command,
                workload=workload,
                chats=chats,
            )
            assert code == 0
            projects[name] = folder / "project"
        yield projects


def _probes(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text())["probes"]


@pytest.mark.parametrize("case", ["dense_offload", "moe_offload", "moe_step"])
def test_llamacpp_offload(
    case: str,
    offload_projects: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if case == "dense_offload":
        env = {"FAKE_DEVICE_FREE_MIB": "20000", "FAKE_FIT_LAYERS": "49", "FAKE_FIT_MOE": "12"}
        code, _, err = _analyse(
            offload_projects["moe_fit"], capsys, monkeypatch, env=env, cache="moefit"
        )
        assert code == 0, err
        argv = _argv(max((offload_projects["moe_fit"] / "runs").iterdir()))
        assert _after(argv, "--fit") == "off" and _after(argv, "--n-gpu-layers") == "49"
        assert _after(argv, "--n-cpu-moe") == "12"
        moe_fit = offload_projects["moe_fit"]
        tries = [_step_args(offer) for offer in _offers(max((moe_fit / "runs").iterdir()))]
        assert tries and not any(arg.startswith("fit") for step in tries for arg in step)
        step = (
            "--engine-arg",
            "n-gpu-layers=49",
            "--engine-arg",
            "n-cpu-moe=12",
            "--engine-arg",
            "threads-batch=15",
        )
        code, _, err = _analyse(moe_fit, capsys, monkeypatch, *step, env=env, cache="moefit")
        assert code == 0, err
    dense = case == "dense_offload"
    env = {**OFFLOAD, "FAKE_DEVICE_FREE_MIB": "20000" if dense else "4000"}
    env |= (
        {"FAKE_FIT_LAYERS": "20", "FAKE_FIT_MOE": ""}
        if dense
        else {"FAKE_PCIE_GBS": str(FAKE_PCIE_GBS)}
    )
    project = offload_projects["dense" if dense else "moe"]
    step = ("--engine-arg", "ubatch-size=2048") if case == "moe_step" else ()
    code, out, err = _analyse(project, capsys, monkeypatch, *step, env=env, cache="offload")
    assert code == 0, err
    run_dir = max((project / "runs").iterdir())
    if step:
        baseline = sorted((project / "runs").iterdir())[-2]
        assert _probes(baseline)["pcie"]["gbs_effective"] is None
        assert _probes(baseline)["prefill"]["ubatch"] == 512
        probes = _probes(run_dir)
        assert probes["pcie"]["ub_small"] == 512 and probes["pcie"]["ub_large"] == 2048
        assert abs(probes["pcie"]["gbs_effective"] / FAKE_PCIE_GBS - 1) < 0.15
        assert probes["host"]["cached"] is True
        assert main(["set", run_dir.name, "--project", str(project)]) == 0
        line = next(t for t in capsys.readouterr().out.splitlines() if "as measured:" in t)
        command = shlex.split(line.partition("as measured: ")[2])
        measured = _argv(run_dir)
        assert command[0] == "llama-server" and _after(command, "--port") == "9932"
        assert _after(command, "--fit") == "off" and _after(command, "--ubatch-size") == "2048"
        assert ("--kv-unified" in command) == ("--kv-unified" in measured) and (
            "--no-kv-unified" in command
        ) == ("--no-kv-unified" in measured)
        assert command.count("--n-gpu-layers") == 1 and not {"-lv", "--metrics", "--slots"} & set(
            command
        )
        return
    offers = _offers(run_dir)
    names = [offer["name"] for offer in offers]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    finding = next(f for f in metrics["diagnosis"]["findings"] if f["bottleneck"] == "offload")
    assert finding["state"] in ("critical", "warning")
    assert "CPU-offloaded" in finding["evidence"]
    block = metrics["placement"]
    steps = {step["lever"]: step for step in block["next_steps"]}
    assert block["regime"]["placement"] == "dense_offload"
    assert all(step["saved_ms_per_token"] > 0 for step in steps.values())
    probes = _probes(run_dir)
    assert abs(probes["host"]["mem_bw_gbs"] / 1000 - 1) < 0.1
    assert probes["host"]["cached"] is False
    if dense:
        assert next(o for o in offers if o["name"] == "offload-less")["changes_answers"]
        assert all(step["label"] == "likely" for step in steps.values())
        assert steps["host_layers"]["value"] < block["placement"]["host_layers"] == 21
        assert "host_expert_layers" not in steps
        argv = _argv(run_dir)
        assert _after(argv, "--fit") == "off" and _after(argv, "--n-gpu-layers") == "20"
        assert (run_dir / "user-fit").is_dir()
        report = (run_dir / "report.md").read_text()
        assert "## Where the model lives" in report
        assert "limiting resource: host memory bandwidth" in report
        assert "predicted saving" in report and "(likely)" in report
        assert re.search(r"decode time the reads do not explain: \d+% of the measured", report)
        assert "`--n-gpu-layers 41`" in report.split("Try first:")[1].split("Could help:")[0]
        assert "checkpoint: GGUF Q6_K, " in out and " bits/weight" in out
        return
    assert list(steps) == ["host_expert_layers"]
    assert "raise-ubatch" in names and "offload-less" not in names
    assert "offload-experts" in names and "kv-q8" not in names
    tried = (run_dir / "report.md").read_text().split("Try first:")[1].split("Could help:")[0]
    assert "`--n-gpu-layers 49 --n-cpu-moe 17`" in tried


@pytest.fixture(scope="module")
def cpu_project(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Qwen3-0.6B served by a CPU build on a machine with no accelerator."""
    root = tmp_path_factory.mktemp("cpu")
    (file,) = write_gguf(root / "Qwen3-0.6B-Q8_0.gguf", _describe("qwen3-0.6b-q8_0"))
    command = f"llama-server -m {file} --port 9933 -np 2"
    with _no_gpu(), _cheap_hash(), pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_CACHE_HOME", str(root / "cache"))
        workload = {"request_count": 4, "output_tokens": 16}
        assert _register(root / "project", file, root, "--current", command, workload=workload) == 0
        yield root / "project"


def test_llamacpp_probes(
    cpu_project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tensward.platforms.cpu._isa", lambda: frozenset({"avx2", "avx512f"}))
    launches = cpu_project.parent / "launches.txt"
    env = {
        "FAKE_BACKEND": "cpu", "FAKE_BW_HOST_GBS": "100", "FAKE_BW_DEV_GBS": "1000",
        "FAKE_LAUNCH_LOG": str(launches),
    }  # fmt: skip
    for cached in (False, True):
        code, _, err = _analyse(cpu_project, capsys, monkeypatch, env=env, cache="cpu", gpu=False)
        assert code == 0, err
        run_dir = max((cpu_project / "runs").iterdir())
        host = _probes(run_dir)["host"]
        assert abs(host["mem_bw_gbs"] / 100 - 1) < 0.1 and host["cached"] is cached
        assert len(launches.read_text().split()) == 2 + cached
    assert not any(a.startswith("ctx-size=") for o in _offers(run_dir) for a in _step_args(o))
    assert "offloaded 29/29 layers to GPU" in (run_dir / "server.log").read_text()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["placement"]["regime"]["placement"] == "cpu_only"
    assert "rebuild your llama.cpp with AVX512" in (run_dir / "report.md").read_text()
    local = shlex.join([sys.executable, str(FAKE)])
    with _no_gpu():
        started = main(["serve", "start", "--project", str(cpu_project), "--runtime", "local",
                        "--local-command", local, "--port", str(select_loopback_port()),
                        "--ready-timeout", "30"])  # fmt: skip
    state = json.loads((cpu_project / "serve" / "default" / "state.json").read_text())
    try:
        assert started == 0 and httpx.get(f"{state['endpoint']}/health").status_code == 200
        served = state["identity"]["argv"]
        assert (
            not {"-lv", "--metrics", "--slots"} & set(served) and _after(served, "--fit") == "off"
        )
    finally:
        main(["serve", "stop", "--project", str(cpu_project)])


@pytest.mark.parametrize(
    ("host_layers", "measured_ms"), [(11, 107.158), (21, 159.423), (31, 212.373)]
)
def test_decode_model_reproduces_the_measured_throughput(host_layers, measured_ms) -> None:
    """Qwen3-14B Q6_K on an L4 at C=1, b11669, with throughput figures recorded from a real run."""
    from tensward.engines.llamacpp.placement import Placement, calibrate, model_error

    weights = 11_477_849_210
    model = calibrate(
        weight_bytes=weights,
        bw_dev_gbs=255.0,
        full_gpu_ms=49.238,
        cpu_ms=265.968,
        bw_host_gbs=None,
        concurrency=1,
    )
    host = weights * host_layers // 41
    p = Placement(
        device_weight_bytes=weights - host,
        host_weight_bytes=host,
        host_expert_bytes=0,
        host_layers=host_layers,
        host_expert_layers=0,
        kv_on_host=False,
        device_bytes_per_token=weights - host,
        host_bytes_per_token=host,
    )
    assert model.calibrated and abs(model_error(model, p, measured_ms)) < 0.015
    assert not calibrate(
        weight_bytes=weights,
        bw_dev_gbs=255.0,
        full_gpu_ms=64.047,
        cpu_ms=None,
        bw_host_gbs=43.9,
        concurrency=4,
    ).calibrated


@pytest.mark.parametrize(
    ("flags", "n", "pinned"),
    [("-c 32768 -np 4 --fit off", 12, ""), ("-ngl all", 6, " --engine-arg ctx-size=4096")],
)
def test_llamacpp_model_larger_than_the_gpu(
    flags: str,
    n: int,
    pinned: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (file,) = write_gguf(tmp_path / "moe.gguf", _describe("qwen3-30b-a3b-q6_k"))
    command = f"llama-server -m {file} {flags} --port 9934"
    with _l4(), _cheap_hash():
        monkeypatch.setattr(
            "tensward.project.host_facts", lambda: replace(host_facts(), mem_available_gib=64.0)
        )
        code, _, err = _init(capsys, tmp_path / "project", file, tmp_path, "--current", command)
        assert code == 0 and f"`--n-gpu-layers all --n-cpu-moe {n}`" in err
        code, _, err = _analyse(tmp_path / "project", capsys, monkeypatch)
    assert code == 0
    assert "This analysis takes" not in err and "CI lower bound" not in err and "failed" not in err
    assert (
        "so your current setup won't start. Tensward has a fix to try: keep part of the model in "
        "CPU memory." in err
    )
    assert "Next:\n  tensward try" in err and "test it (~" in err and "tensward serve" not in err
    assert not list((tmp_path / "project" / "runs").glob("*/server.log"))


def test_serve_of_a_setup_that_cannot_start_says_so_instead_of_launching(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensward.engines.vllm.engine import VLLM
    from tensward.runtime import exit_reason, out_of_memory

    (file,) = write_gguf(tmp_path / "moe.gguf", _describe("qwen3-30b-a3b-q6_k"))
    command = f"llama-server -m {file} -c 32768 -np 4 --fit off --port 9934"
    project = tmp_path / "project"
    monkeypatch.delenv("TENSWARD_JSON")
    with _l4(), _cheap_hash():
        monkeypatch.setattr(
            "tensward.project.host_facts", lambda: replace(host_facts(), mem_available_gib=64.0)
        )
        assert _init(capsys, project, file, tmp_path, "--current", command)[0] == 0
        fake = shlex.join([sys.executable, str(FAKE)])
        code = main(["serve", "--project", str(project), "--local-command", fake])
    err = capsys.readouterr().err
    assert code == 1 and "so your current setup won't start. Tensward has a fix to try" in err
    assert "\nNext:\n  tensward try " in err and "tensward serve --anyway" in err
    assert not (project / "serve").exists()
    # clean L4 vLLM leg, Qwen3-8B at the checkpoint's context: one plain line, no traceback
    log = (
        "(EngineCore pid=9) ERROR 11-07 16:18:53 [core.py:1366] ValueError: To serve at least one "
        "request with the model's max seq len (40960), (5.62 GiB KV cache is needed, which is "
        "larger than the available KV cache memory (3.15 GiB). Based on the available memory, the "
        "estimated maximum model length is 22944. Try increasing `gpu_memory_utilization`\n"
        "(APIServer pid=1) Traceback (most recent call last):\n"
    )
    said = exit_reason(log, VLLM)
    assert said == (
        "the server did not start: not enough GPU memory left for its KV cache: one request of "
        "the full context (40960 tokens) needs 5.62 GiB and 3.15 GiB is left; a context of up to "
        "22944 tokens fits."
    )
    assert out_of_memory(said)


def test_try_runs_the_offload_offered_for_a_model_larger_than_the_gpu(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (file,) = write_gguf(tmp_path / "moe.gguf", _describe("qwen3-30b-a3b-q6_k"))
    command = f"llama-server -m {file} -c 32768 -np 4 --fit off --port 9934"
    project = tmp_path / "project"
    with _l4(), _cheap_hash():
        monkeypatch.setattr(
            "tensward.project.host_facts", lambda: replace(host_facts(), mem_available_gib=64.0)
        )
        code, out, err = _init(capsys, project, file, tmp_path, "--current", command, "--json")
        assert code == 0 and "does not fit the GPU" in err
        assert (
            "`--n-gpu-layers all --n-cpu-moe 12`" in err
            and "speed: measured by `tensward try`" in err
        )
        fit = json.loads(out.strip().splitlines()[-1])["fit"]
        assert fit["verdict"] == "likely does not fit" and "does not fit the GPU" in fit["reason"]
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        fake = shlex.join([sys.executable, str(FAKE)])
        code = main(["try", "--project", str(project), "--yes", "--local-command", fake])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    result = json.loads(captured.out)
    assert result["offer"]["name"] == "offload-experts" and result["kept"], result
    (run_dir,) = (project / "runs").iterdir()
    assert _after(_argv(run_dir), "--n-cpu-moe") == "12"
    assert json.loads((project / "kept.json").read_text())["run"] == run_dir.name
    project_flag = ["--project", str(project)]
    assert main(["export", "--format", "compose", *project_flag]) == 0
    assert "--n-cpu-moe" in capsys.readouterr().out
    assert main(["undo", *project_flag]) == 0
    assert "Undone:" in capsys.readouterr().out
    assert main(["export", "--format", "compose", *project_flag]) == 0
    assert "--n-cpu-moe" not in capsys.readouterr().out


def test_try_launches_a_forced_dense_ngl_dropped_for_the_build_to_place(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clean L4 llama.cpp leg: `-ngl 99` on a dense model that does not fit. try read the build
    after making its offer, so it launched `--n-gpu-layers 45 --fit off`, not --fit's placement."""
    (file,) = write_gguf(tmp_path / "dense.gguf", _describe("qwen3.8-27b-ud-q4_k_m"))
    command = f"llama-server -m {file} -ngl 99 -c 131072 -np 4 --port 9935"
    project = tmp_path / "project"
    with _l4(), _cheap_hash():
        monkeypatch.setattr(
            "tensward.project.host_facts", lambda: replace(host_facts(), mem_available_gib=64.0)
        )
        assert _init(capsys, project, file, tmp_path, "--current", command)[0] == 0
    with _cheap_hash():
        code, _, err = _analyse(project, capsys, monkeypatch)
    assert code == 0 and "Tensward has a fix to try: drop --n-gpu-layers and let llama.cpp" in err
    with _l4(), _cheap_hash():
        fake = shlex.join([sys.executable, str(FAKE)])
        # a fresh context, as a new process starts: no build read yet
        code = contextvars.Context().run(
            main, ["try", "--project", str(project), "--yes", "--local-command", fake]
        )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    (run_dir,) = (project / "runs").iterdir()
    argv = _argv(run_dir)
    assert _after(argv, "--n-gpu-layers") == "auto" and _after(argv, "--fit") == "on"


def test_first_offload_try_is_not_kept_when_its_answers_are_broken(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (file,) = write_gguf(tmp_path / "moe.gguf", _describe("qwen3-30b-a3b-q6_k"))
    command = f"llama-server -m {file} -c 32768 -np 4 --fit off --port 9934"
    project = tmp_path / "project"
    with _l4(), _cheap_hash():
        monkeypatch.setattr(
            "tensward.project.host_facts", lambda: replace(host_facts(), mem_available_gib=64.0)
        )
        code, _, err = _init(capsys, project, file, tmp_path, "--current", command)
        assert code == 0, err
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        monkeypatch.setattr(
            "tensward.runs.answers_in", lambda run_dir: {"p": (Answer("", None, "stop"),)}
        )
        fake = shlex.join([sys.executable, str(FAKE)])
        code = main(["try", "--project", str(project), "--yes", "--local-command", fake])
    result = json.loads(capsys.readouterr().out)
    assert code != 0 and not result["kept"] and result["gate"] == "answers", result
    assert not (project / "kept.json").exists()


def test_a_try_where_no_request_succeeds_is_rejected_and_not_offered_again(
    apple_gemma_project: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = shutil.copytree(apple_gemma_project, tmp_path / "project")
    env = {
        "FAKE_BACKEND": "apple", "FAKE_METAL_PRIVATE": "1", "FAKE_METAL_OOM": "1",
        "FAKE_METAL_OOM_AFTER": "3", "FAKE_METAL_WS_MIB": "3500",
    }  # fmt: skip
    with _cheap_hash():
        code, _, _ = _analyse(project, capsys, monkeypatch, env=env, machine=_apple)
        assert code == 1
        command = shlex.join([sys.executable, str(FAKE)])
        with _apple():
            code = main(["try", "--project", str(project), "--yes",
                         "--local-command", command])  # fmt: skip
    captured = capsys.readouterr()
    assert code != 0 and json.loads(captured.out)["gate"] == "requests"
    rejected = json.loads((project / "rejected.json").read_text())["rejected"]
    assert [row["reason"] for row in rejected] == ["No request succeeded."]


@pytest.mark.parametrize("models", [{"data": "x"}, {"data": ["x"]}, {"data": [[]]}, ["x"]])
def test_a_malformed_models_listing_is_no_evidence(models) -> None:
    from tensward.engines.llamacpp.detect import probe_endpoint

    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=models if request.url.path == "/v1/models" else {})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        facts = probe_endpoint("http://host", client)
    assert facts is not None and not [e for e in facts.evidence if e.engine]


def _llama_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, devices: list[str]) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    server = bin_dir / "llama-server"
    listing = "\\n".join(["Available devices:", *devices])
    server.write_text(
        f'#!/bin/sh\nif [ "$1" = "--list-devices" ]; then printf \'{listing}\\n\'; exit 0; fi\n'
        f'exec {shlex.quote(sys.executable)} {shlex.quote(str(FAKE))} "$@"\n'
    )
    server.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


@pytest.mark.parametrize(
    ("devices", "warned"),
    [(["  (none)"], True), (["  CUDA0: NVIDIA L4 (22563 MiB, 22369 MiB free)"], False)],
)
def test_init_warns_when_the_llama_server_build_cannot_use_the_gpu(
    devices: list[str],
    warned: bool,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _llama_on_path(tmp_path, monkeypatch, devices)
    (file,) = write_gguf(tmp_path / "moe.gguf", _describe("qwen3-30b-a3b-q6_k"))
    monkeypatch.setattr(
        "tensward.detect.detect_hardware",
        lambda: Hardware("nvidia", (("NVIDIA L4", 23034),), "550", True, None, None, None),
    )
    with _l4(), _cheap_hash():
        flags = ["--workload", "chat", "--tools", "none", "--concurrency", "4"]
        flags += ["--goal", "throughput", "--engine-defaults", "--yes", "--no-run", "--json"]
        code = main(["init", "--model", str(file), "--project", str(tmp_path / "p"), *flags])
    err = capsys.readouterr().err
    assert code == 0, err
    assert ("a CPU-only build: it will not use the GPU" in err) == warned


def test_a_cuda_build_is_known_by_its_device_list_for_the_exported_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensward.engines.llamacpp.command import probe_build

    _llama_on_path(tmp_path, monkeypatch, ["  CUDA0: NVIDIA L4 (22563 MiB, 22369 MiB free)"])
    found = probe_build("llama-server")
    assert found is not None and "CUDA" in found[0].backends


def test_a_metal_build_is_known_by_its_device_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensward.engines.llamacpp.command import probe_build

    _llama_on_path(
        tmp_path,
        monkeypatch,
        ["  MTL0: Apple M4 (18186 MiB, 18185 MiB free)", "  BLAS: Accelerate (0 MiB, 0 MiB free)"],
    )
    found = probe_build("llama-server")
    assert found is not None and found[0].backends == ("Metal",)


def test_a_defaults_baseline_larger_than_the_gpu_lets_llama_cpp_fit_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (file,) = write_gguf(tmp_path / "moe.gguf", _describe("qwen3-30b-a3b-q6_k"))
    project = tmp_path / "project"
    with _l4(), _cheap_hash():
        monkeypatch.setattr(
            "tensward.project.host_facts", lambda: replace(host_facts(), mem_available_gib=64.0)
        )
        code, _, err = _init(capsys, project, file, tmp_path, "--engine-defaults")
        assert code == 0, err
        code, _, err = _analyse(project, capsys, monkeypatch, env={"FAKE_FIT_CTX": "4096"})
    assert code == 0, err
    assert _after(_argv(project / "runs" / "0001-baseline"), "--fit") == "on"


def test_unified_fit_counts_the_recurrent_state_of_a_hybrid_model() -> None:
    """Qwen3.8-27B Q4_K_M on a 24 GiB M4 (b11982, -np 4, 25856 tokens): the buffers left 1.0 GiB
    of the Metal working set and the launch died of Metal out-of-memory 47 s into the run. The
    KV buffer in the log leaves out the 598.5 MiB state buffer and the checkpoints the server
    keeps of that state in host memory."""
    state = 156_893_184  # the log's RS buffer is 4 x this
    anatomy = replace(
        ModelAnatomy(
            None,
            False,
            None,
            65,
            None,
            False,
            None,
            None,
            {},
            (),
            None,
            None,
            (),
            linear_attention=True,
        ),
        state_bytes_per_seq=state,
    )
    where = Placement(
        device_weight_bytes=15_738_281_984, host_weight_bytes=0, host_expert_bytes=0,
        host_layers=0, host_expert_layers=0, kv_on_host=False,
        device_bytes_per_token=0, host_bytes_per_token=0,
    )  # fmt: skip
    host = replace(host_facts(), mem_gib=24.0, mem_available_gib=18.85, wired_limit_mb=0)
    fit = fit_domains(
        anatomy, where, pool_tokens=25856, type_k="f16", type_v="f16",
        compute_buffer_bytes=251_175_895, device_free=19_069_670_000, host=host, slots=4,
        topology="unified", kv_logged=1_694_498_816,
    )  # fmt: skip
    assert fit.kv_bytes == 1_694_498_816 + 4 * state
    assert not fit.fits and "system memory" in fit.reason


def test_a_metal_run_that_dies_late_and_does_not_fit_offers_a_smaller_pool(
    apple_gemma_project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every request after the warm-up fails with Metal out-of-memory: the report still names the
    context and slots that fit, and `tensward try` has a command to run."""
    env = {
        "FAKE_BACKEND": "apple", "FAKE_METAL_PRIVATE": "1", "FAKE_METAL_OOM": "1",
        "FAKE_METAL_OOM_AFTER": "3", "FAKE_METAL_WS_MIB": "3500",
    }  # fmt: skip
    with _cheap_hash():
        code, _, err = _analyse(apple_gemma_project, capsys, monkeypatch, env=env, machine=_apple)
    assert code == 1 and "no request succeeded" in err
    run_dir = max((apple_gemma_project / "runs").iterdir())
    block = json.loads((run_dir / "metrics.json").read_text())["placement"]
    assert not block["fit"]["fits"]
    (offer,) = _offers(run_dir)
    assert offer["name"] == "buy-gpu-room" and _step_args(offer) == ["ctx-size=2048"]
    assert "largest setup that fits this Mac's Metal working set" in json.dumps(offer)
    assert "gives up context" in json.dumps(offer) and "iogpu.wired_limit_mb" in json.dumps(offer)


def test_a_metal_run_that_fits_on_paper_but_runs_out_of_memory_still_offers_a_smaller_pool(
    apple_gemma_project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Qwen3.8-27B on the 24 GiB M4 (b11982): the log's buffers added to 16.7 GiB of a 17.8 GiB
    working set, the fit held, and every request then failed with kIOGPUCommandBuffer...
    OutOfMemory; `tensward try` said there was nothing to try."""
    env = {
        "FAKE_BACKEND": "apple", "FAKE_METAL_PRIVATE": "1", "FAKE_METAL_OOM": "1",
        "FAKE_METAL_OOM_AFTER": "3", "FAKE_METAL_WS_MIB": "18185",
    }  # fmt: skip
    with _cheap_hash():
        code, _, err = _analyse(apple_gemma_project, capsys, monkeypatch, env=env, machine=_apple)
    assert code == 1 and "no request succeeded" in err
    run_dir = max((apple_gemma_project / "runs").iterdir())
    block = json.loads((run_dir / "metrics.json").read_text())["placement"]
    assert not block["fit"]["fits"] and "ran out of memory" in block["fit"]["reason"]
    (offer,) = _offers(run_dir)
    assert offer["name"] == "buy-gpu-room"


def test_a_run_too_slow_for_its_count_offers_fewer_requests_and_says_so(
    apple_gemma_project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tensward.measure.PACE_AFTER_S", 0.5)
    monkeypatch.setattr("tensward.measure.PACE_CEILING_S", 10.0)
    monkeypatch.setattr("tensward.measure.MIN_REQUESTS", 5)
    monkeypatch.setattr("tensward.measure.LOG_SYNC_S", 0.2)
    env = {"FAKE_BACKEND": "apple", "FAKE_TOKEN_DELAY_S": "0.05"}
    with _cheap_hash():
        code, _, err = _analyse(apple_gemma_project, capsys, monkeypatch, env=env, machine=_apple)
    assert code == 0 and "the noise band is wider" in err
    run_dir = max((apple_gemma_project / "runs").iterdir())
    rows = (run_dir / "requests.jsonl").read_text().splitlines()
    assert 5 <= len(rows) < 10


def test_a_run_too_slow_even_for_the_least_is_stopped_with_the_pace_in_the_message(
    apple_gemma_project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tensward.measure.PACE_AFTER_S", 1.0)
    monkeypatch.setattr("tensward.measure.PACE_CEILING_S", 0.5)
    monkeypatch.setattr("tensward.measure.LOG_SYNC_S", 0.2)
    env = {"FAKE_BACKEND": "apple", "FAKE_TOKEN_DELAY_S": "0.05"}
    with _cheap_hash():
        code, _, err = _analyse(apple_gemma_project, capsys, monkeypatch, env=env, machine=_apple)
    assert code == 1 and "too slow to measure" in err and "for the whole run" in err


def test_a_try_that_starts_where_the_current_setup_did_not_but_slowly_can_be_kept(
    apple_gemma_project: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qwen3.8-27B on the 24 GiB M2 Mac mini: no request of the current setup succeeded, the
    offered smaller setup answered at 5 requests in 191 s, and the pace control reported it as
    "did not run" and rejected it."""
    project = shutil.copytree(apple_gemma_project, tmp_path / "project")
    env = {
        "FAKE_BACKEND": "apple", "FAKE_METAL_PRIVATE": "1", "FAKE_METAL_OOM": "1",
        "FAKE_METAL_OOM_AFTER": "3", "FAKE_METAL_WS_MIB": "3500",
    }  # fmt: skip
    with _cheap_hash():
        assert _analyse(project, capsys, monkeypatch, env=env, machine=_apple)[0] == 1
        monkeypatch.delenv("FAKE_METAL_OOM")
        monkeypatch.setenv("FAKE_TOKEN_DELAY_S", "0.05")
        monkeypatch.setenv("TENSWARD_JSON", "0")
        monkeypatch.setattr("tensward.measure.PACE_AFTER_S", 1.0)
        monkeypatch.setattr("tensward.measure.PACE_CEILING_S", 0.5)
        monkeypatch.setattr("tensward.measure.LOG_SYNC_S", 0.2)
        at = ["--project", str(project)]
        command = shlex.join([sys.executable, str(FAKE)])
        with _apple():
            code = main(["try", *at, "--yes", "--local-command", command])
            out = capsys.readouterr().out
            assert code == 3 and "It starts now (your current setup does not)" in out, out
            assert "tensward try --keep" in out and "did not run" not in out
            assert not (project / "rejected.json").exists() and not (project / "kept.json").exists()
            assert main(["try", *at, "--yes", "--json", "--local-command", command]) == 3
            document = json.loads(capsys.readouterr().out)
            assert document["gate"] == "slow" and document["answers"]["ok"] is True
            assert document["answers"]["non_empty"] == document["answers"]["total"] > 0
            assert main(["try", "--keep", *at]) == 0
    assert "slow on this machine" in capsys.readouterr().out
    assert (project / "kept.json").exists()


def test_a_baseline_where_every_request_failed_does_not_time_the_next_try(
    apple_gemma_project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Qwen3.8-27B on the 24 GiB M2 Mac mini: every baseline request hit a Metal OOM within a
    second, the offer said "about 1 min" and the try took 356 s."""
    env = {
        "FAKE_BACKEND": "apple", "FAKE_METAL_PRIVATE": "1", "FAKE_METAL_OOM": "1",
        "FAKE_METAL_OOM_AFTER": "3", "FAKE_METAL_WS_MIB": "3500",
    }  # fmt: skip
    with _cheap_hash():
        code, out, _ = _analyse(apple_gemma_project, capsys, monkeypatch, env=env, machine=_apple)
    assert code == 1 and "testing it:" not in out
    menu = "--ctx-size 2048 (time not known yet: no request finished)"
    assert f"tensward try          test the first suggested change, {menu}" in out


def test_a_server_that_stops_answering_is_stopped_and_its_log_kept(
    apple_gemma_project: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tensward.measure.stall_bound", lambda workload: 2.0)
    monkeypatch.setattr("tensward.measure.LOG_SYNC_S", 0.2)
    env = {"FAKE_BACKEND": "apple", "FAKE_HANG_AFTER": "6"}
    with _cheap_hash():
        code, _, err = _analyse(apple_gemma_project, capsys, monkeypatch, env=env, machine=_apple)
    assert code == 1 and "the server stopped answering" in err
    run_dir = max((apple_gemma_project / "runs").iterdir())
    assert "llama" in (run_dir / "server.log").read_text().lower()


def _tester_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompts: str, *, mtp: bool = True
) -> Path:
    """The 0.4.2 tester's setup: Qwen3.8-27B Q4_K_M on one L4, `llama-server -m <gguf>` and
    nothing else, three clients with tools, goal TPOT. ``prompts``: a built-in workload or a
    prompts file; ``mtp``: whether the file keeps the model's next-token-prediction layer."""
    _llama_on_path(tmp_path, monkeypatch, ["  CUDA0: NVIDIA L4 (22563 MiB, 22369 MiB free)"])
    description = _describe("qwen3.8-27b-ud-q4_k_m")
    model = description if mtp else without_mtp(description)
    (file,) = write_gguf(tmp_path / "Qwen3.8-27B-Q4_K_M.gguf", model)
    source = ["--workload", prompts] if prompts in ("agent", "chat") else ["--prompts", prompts]
    flags = [*source, "--tools", "default", "--concurrency", "3", "--goal", "tpot"]
    flags += ["--current", f"llama-server -m {file}", "--yes", "--no-run", "--json"]
    project = tmp_path / "project"
    with _l4(), _cheap_hash():
        assert main(["init", "--model", str(file), "--project", str(project), *flags]) == 0
    return project


def test_a_run_with_nothing_for_tpot_says_what_would_show_a_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _tester_project(tmp_path, monkeypatch, "agent", mtp=False)
    capsys.readouterr()
    env = {"FAKE_DEVICE_FREE_MIB": "22000", "FAKE_FIT_CTX": "16384"}
    with _cheap_hash():
        code, out, err = _analyse(project, capsys, monkeypatch, "--yes", env=env)
    assert code == 0, err
    assert "no change is worth testing" not in out and "nothing yet" not in out
    assert (
        "Result: your server answered every request; nothing found that lowers time per token "
        "here. Your test kept only 3 requests in flight; if real traffic is higher, run "
        "`tensward init` again with that concurrency."
    ) in out


def _tpot_tester_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    rules = " ".join(
        f"rule {i}: answer briefly and call a tool for fresh data." for i in range(250)
    )
    rows = [
        {"id": f"p{i}", "messages": [{"role": "user", "content": f"Question {i} first. {rules}"}]}
        for i in range(64)
    ]
    prompts = write_prompts(tmp_path / "prompts.jsonl", rows)
    project = _tester_project(tmp_path, monkeypatch, str(prompts), mtp=False)
    env = {
        "FAKE_DEVICE_FREE_MIB": "22000",
        "FAKE_FIT_CTX": "16384",
        "FAKE_PREFILL_S_PER_PASS": "0.02",
    }
    return project, env


def test_a_run_with_nothing_for_tpot_offers_a_change_for_another_goal_and_try_keeps_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(alternatives, "table", lambda: ())
    project, env = _tpot_tester_project(tmp_path, monkeypatch)
    capsys.readouterr()
    with _cheap_hash():
        code, out, err = _analyse(project, capsys, monkeypatch, "--yes", env=env)
    assert code == 0, err
    assert (
        "Nothing found that lowers time per token here. These may help elsewhere without making "
        "it worse:\n  1. --ubatch-size 2048: a shorter time to first token (TPOT is checked so "
        "it doesn't get worse)"
    ) in out
    assert (
        "Result: your server answered every request; nothing found that lowers time per token "
        "here, but 1 change listed above may help elsewhere without making it worse."
    ) in out
    assert "tensward try          test --ubatch-size 2048 for a shorter time to first token" in out
    assert "a higher-bit file (Q6_K, Q8_0)" in out and "Measured on" not in out
    report = (project / "runs" / "0001-baseline" / "report.md").read_text()
    assert "how well backed" not in out and "how well backed" in report
    assert "Measured on Gemma 4 E4B" in report
    monkeypatch.delenv("TENSWARD_JSON", raising=False)
    with _l4():
        code = main(["try", "--project", str(project), "--yes", "--local-command",
                     shlex.join([sys.executable, str(FAKE)])])  # fmt: skip
    captured = capsys.readouterr()
    tried = captured.out
    assert code == 0, captured.out + captured.err
    assert "Tried: --ubatch-size 2048" in tried and "a real gain" in tried
    assert "Per token, slowest 1 in 20: no significant change." in tried and "±" not in tried
    assert re.search(r"^ +before +after +change$", tried, re.M)
    assert "Kept in this project" in tried


def test_a_capped_gpu_with_nothing_for_tpot_leads_with_measured_alternatives(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(alternatives, "table", lambda: parse_alternatives(RECORDS))
    monkeypatch.setattr(alternatives, "on_path", lambda program: program != "vllm")
    project, env = _tpot_tester_project(tmp_path, monkeypatch)
    capsys.readouterr()
    capped = "88,69.0,72.00,1081,2040,Active,Not Active,Not Active"
    with _cheap_hash():
        code, out, err = _analyse(project, capsys, monkeypatch, "--yes", env=env, smi_row=capped)
    assert code == 0, err
    block = out.split("No server flag lifts the GPU's power cap")[1].split("\nThis run measured")[0]
    block = block.split("\n", 1)[1]
    long = [line for line in block.splitlines() if len(line) > 80]
    assert all(" " not in line.strip().removesuffix(" \\") for line in long)
    heading = "Beyond flags: measured by us on an NVIDIA L4 with Qwen3.8-27B (agent workload)"
    assert f"\n\n{heading}\n\n" in block
    differ = "differ: check them in the new project"
    assert re.search(rf"^1  vLLM, int4 checkpoint +−17% +{differ}$", block, re.M)
    assert re.search(rf"^2  IQ4_XS file +−9% +{differ}$", block, re.M)
    steps = block.split("  To try option 1:\n")[1].split("\n  Steps for option 2:")[0]
    joined = re.sub(r" \\\n +", " ", steps)
    numbered = dict(re.findall(r"^    (\d)\. (.*)$", joined, re.M))
    assert numbered["1"] == "uv tool install vllm==0.30.0"
    assert numbered["2"].startswith("sudo apt install -y cuda-toolkit-13-0")
    download = "get the checkpoint first: hf download example/Qwen3.8-27B-int4"
    assert numbered["3"].startswith(download)
    assert numbered["4"] == "mkdir -p ~/tensward-vllm-int4 && cd ~/tensward-vllm-int4"
    init = shlex.split(numbered["5"])
    assert init[:3] == ["tensward", "init", "--current"] and init[4:] == ["--like", str(project)]
    served = ENGINES["vllm"].parse_setup(init[3])
    assert served.model == str(Path.home() / "models" / "Qwen3.8-27B-int4")
    assert '  Steps for option 2: report.md, section "Beyond flags".' in block
    assert "Answers differ with another file or engine; the new project's check decides." in block
    assert "see Beyond flags" in out.split("Result: ")[1]
    menu = out.split("Next:\n")[1].rstrip("\n").splitlines()
    measure = "measure vLLM int4 as a new project (steps above, under Beyond flags)"
    assert re.fullmatch(rf"  uv tool install vllm==0\.30\.0 +{re.escape(measure)}", menu[0])
    assert len(menu) == 3 and menu[1].startswith("  tensward try") and "serve" in menu[2]
    report = (project / "runs" / "0001-baseline" / "report.md").read_text()
    assert "## Beyond flags" in report and "TPOT p50 −17% (A/A range 3.0%)" in report
    file_steps = report.split("- IQ4_XS file")[1] if "- IQ4_XS file" in report else report
    assert "`get the file first: hf download unsloth/Qwen3.8-27B-GGUF" in file_steps
    monkeypatch.delenv("TENSWARD_JSON", raising=False)
    with _l4():
        main(["try", "--project", str(project), "--yes", "--local-command",
              shlex.join([sys.executable, str(FAKE)])])  # fmt: skip
    tried = capsys.readouterr().out
    options = "2 measured options (vLLM int4 −17%, IQ4_XS −9%)"
    assert f"Beyond flags: {options}; see the first report or report.md\nNext:" in tried
    assert "To try option 1" not in tried


def test_init_like_measures_the_same_workload_in_a_new_project(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    project, _ = _tpot_tester_project(tmp_path, monkeypatch)
    (other,) = write_gguf(tmp_path / "Qwen3.8-27B-UD-IQ4_XS.gguf",
                          _describe("qwen3.8-27b-ud-q4_k_m"))  # fmt: skip
    fresh = tmp_path / "iq4xs" / ".tensward"
    flags = ["--current", f"llama-server -m {other} --jinja", "--yes", "--no-run", "--json"]
    with _l4(), _cheap_hash():
        assert main(["init", "--project", str(fresh), "--like", str(project), *flags]) == 0
        capsys.readouterr()
        assert main(["init", "--project", str(project), "--like", str(project), *flags]) != 0
    assert "run init in a new folder" in capsys.readouterr().err
    choices = [json.loads((p / "choices.json").read_text()) for p in (project, fresh)]
    assert choices[0] == choices[1]
    prompts = [(p / "inputs" / "prompts.jsonl").read_bytes() for p in (project, fresh)]
    assert prompts[0] == prompts[1]
    assert json.loads((project / "project.json").read_text())["current_setup"]["text"].endswith(
        "Qwen3.8-27B-Q4_K_M.gguf"
    )


def test_a_faster_change_that_rewords_answers_names_what_keeping_it_leads_to(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(alternatives, "table", lambda: ())
    project, env = _tpot_tester_project(tmp_path, monkeypatch)
    capsys.readouterr()
    with _cheap_hash():
        assert _analyse(project, capsys, monkeypatch, "--yes", env=env)[0] == 0
    monkeypatch.delenv("TENSWARD_JSON", raising=False)
    monkeypatch.setenv("FAKE_REWORD_FLAG", "--ubatch-size")
    at = ["--project", str(project)]
    with _l4():
        code = main(["try", *at, "--yes", "--local-command",
                     shlex.join([sys.executable, str(FAKE)])])  # fmt: skip
    tried = capsys.readouterr().out
    assert code == 3, tried
    table = tried.split("\n\n")[1].splitlines()
    assert table[0].split() == ["before", "after", "change"] and set(table[1]) == {"─"}
    assert all(len(line) <= 80 for line in table)
    assert re.search(r"^Answers +\d+ of 64 identical +your call$", tried, re.M)
    assert "Not kept: " in tried and "Measured on" not in tried
    assert "  tensward try --keep   keep it; nothing else to try after it\n" in tried
    assert main(["try", "--keep", *at]) == 0
    assert "Kept --ubatch-size 2048" in capsys.readouterr().out


def test_a_gpu_at_its_power_cap_at_half_clock_is_named_with_no_clock_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    other = RECORDS.replace('models = ["Qwen3.8-27B"]', 'models = ["Qwen3.5-27B"]')
    monkeypatch.setattr(alternatives, "table", lambda: parse_alternatives(other))
    project = _tester_project(tmp_path, monkeypatch, "agent", mtp=False)
    capsys.readouterr()
    env = {"FAKE_DEVICE_FREE_MIB": "22000", "FAKE_FIT_CTX": "16384", "FAKE_ECHO": "1"}
    capped = "88,69.0,72.00,1081,2040,Active,Not Active,Not Active"
    with _cheap_hash():
        code, out, err = _analyse(project, capsys, monkeypatch, "--yes", env=env, smi_row=capped)
    assert code == 0, err
    words = "the GPU's power cap (72 W): it ran at about half clock (1081 of 2040 MHz)"
    flat = " ".join(out.split())
    assert f"Limit {words}" in flat and re.search(r"^GPU active +88% of the time$", out, re.M)
    stands = out.split("Where it stands")[1].split("\n\n")[0]
    assert all(len(line) <= 80 for line in stands.splitlines())
    assert "Bottleneck: the GPU's power cap" in out and "Bottleneck: none" not in out
    assert "nvidia-smi" not in out and "For the GPU's power cap" not in out
    assert "--spec-type" not in out and "Beyond flags" not in out


def test_mtp_speculation_runs_in_try_with_its_context_sized_beside_the_draft(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _tester_project(tmp_path, monkeypatch, "agent")
    capsys.readouterr()
    env = {"FAKE_DEVICE_FREE_MIB": "22000", "FAKE_FIT_CTX": "16384", "FAKE_MTP_FIT_CTX": "8192"}
    capped = "88,69.0,72.00,1081,2040,Active,Not Active,Not Active"
    with _cheap_hash():
        code, out, err = _analyse(project, capsys, monkeypatch, "--yes", env=env, smi_row=capped)
    assert code == 0, err
    assert "\n- --spec-type draft-mtp --spec-draft-n-max 2  (answers may change)" in out
    monkeypatch.delenv("TENSWARD_JSON", raising=False)
    monkeypatch.setenv("FAKE_FAST_FLAG", "--spec-type")
    with _l4():
        code = main(["try", "1", "--project", str(project), "--yes", "--json"])
    captured = capsys.readouterr()
    assert code == 0, captured.err[-2000:]
    result = json.loads(captured.out)
    assert result["offer"]["name"] == "llamacpp-mtp-speculation" and result["kept"] is True
    log = (project / "runs" / result["run_id"] / "server.log").read_text()
    assert "out of memory" not in log and "n_ctx_seq             = 8192" in log


def test_the_op_offload_variable_runs_in_try_to_a_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _llama_on_path(tmp_path, monkeypatch, ["  CUDA0: NVIDIA L4 (22563 MiB, 22369 MiB free)"])
    (file,) = write_gguf(tmp_path / "Qwen3-30B-A3B-Q6_K.gguf", _describe("qwen3-30b-a3b-q6_k"))
    flags = ["--workload", "agent", "--tools", "default", "--concurrency", "4"]
    flags += ["--goal", "throughput", "--yes", "--no-run", "--json"]
    flags += ["--current", f"llama-server -m {file} --jinja -ngl all --n-cpu-moe 17"]
    project = tmp_path / "project"
    with _l4(), _cheap_hash():
        assert main(["init", "--model", str(file), "--project", str(project), *flags]) == 0
    capsys.readouterr()
    env = {"FAKE_DEVICE_FREE_MIB": "22000", "FAKE_FIT_CTX": "16384"}
    with _cheap_hash():
        code, _, err = _analyse(project, capsys, monkeypatch, "--yes", env=env)
    assert code == 0, err
    names = [offer["name"] for offer in _offers(max((project / "runs").iterdir()))]
    monkeypatch.delenv("TENSWARD_JSON", raising=False)
    with _l4():
        code = main(["try", str(names.index("op-offload-min-batch") + 1), "--project",
                     str(project), "--yes", "--json"])  # fmt: skip
    captured = capsys.readouterr()
    assert code in (0, 3), captured.err[-2000:]
    result = json.loads(captured.out)
    assert result["offer"]["engine_env"] == ["GGML_OP_OFFLOAD_MIN_BATCH=256"]
    assert result["run_id"] and result["gate"] not in ("start", None)


def test_the_tester_case_offers_mtp_speculation_first(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(alternatives, "on_path", lambda program: program != "vllm")
    project = _tester_project(tmp_path, monkeypatch, "agent")
    capsys.readouterr()
    env = {"FAKE_DEVICE_FREE_MIB": "22000", "FAKE_FIT_CTX": "16384"}
    with _cheap_hash():
        code, out, err = _analyse(project, capsys, monkeypatch, "--yes", env=env)
    assert code == 0, err
    offer = out.split("\n- --spec-type draft-mtp")[1].splitlines()[:3]
    assert offer == [
        " --spec-draft-n-max 2  (answers may change)",
        "    why: the model file carries a next-token-prediction layer",
        "    Test it: tensward try   (about 1 min on this setup)",
    ]
    report = (project / "runs" / "0001-baseline" / "report.md").read_text()
    assert "about 272 MiB" in report
    assert "test the first suggested change, --spec-type draft-mtp --spec-draft-n-max 2" in out
    facts = json.loads((project / "runs" / "0001-baseline" / "facts.json").read_text())["facts"]
    assert facts["anatomy.mtp_layers"] == 1
    assert re.search(r"^1  vLLM, int4 checkpoint +−17% ", out, re.M)
    assert re.search(r"^2  IQ4_XS file +−10% ", out, re.M)
    assert out.index("Try first:") < out.index("Beyond flags: measured")
    menu = out.split("Next:\n")[1].rstrip("\n").splitlines()
    assert len(menu) == 3 and "measure vLLM int4 as a new project" in menu[1]
    assert "Answers differ with another file or engine; the new project's check decides." in out
