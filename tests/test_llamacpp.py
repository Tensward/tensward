"""End-to-end tests of the llama.cpp engine through the CLI, against a fake llama-server."""

import hashlib
import io
import itertools
import json
import os
import re
import shlex
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
from registration_fixtures import REGISTRATION_CONFIG, write_config, write_gguf, write_prompts

from tensward import ceilings, measure
from tensward import project as projects
from tensward.capabilities import Resolved
from tensward.cli import main
from tensward.engines.llamacpp.advice import _host_cache
from tensward.engines.llamacpp.command import list_devices, working_set_mib
from tensward.engines.llamacpp.engine import LlamaCppEngine
from tensward.engines.llamacpp.placement import (
    Fit2,
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
) -> tuple[int, str, str]:
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    if free := (env or {}).get("FAKE_DEVICE_FREE_MIB"):
        monkeypatch.setattr("tensward.platforms.nvidia.SAMPLE_COMMAND", (sys.executable, str(SMI)))
        monkeypatch.setenv(
            "FAKE_SMI_ROW", "20,40.0,70.00,1230,1590,Not Active,Not Active,Not Active"
        )
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
        assert command[0] == "llama-server" and ("-kvu" in command) == ("-kvu" in argv)
        assert ("-no-kvu" in command) == ("-no-kvu" in argv)
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
    assert out.startswith("Ran: llama.cpp build 11669")
    assert "vLLM" not in out and "vllm" not in out
    run_dir = max((qwen_project / "runs").iterdir())
    argv = _argv(run_dir)
    assert {"--port", "--metrics", "--slots", "--fit", "on"} <= set(argv)
    assert argv[argv.index("-lv") + 1] == "4" and ("-kvu" in argv) != ("-no-kvu" in argv)
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
        assert ("-kvu" in command) == ("-kvu" in measured) and ("-no-kvu" in command) == (
            "-no-kvu" in measured
        )
        assert command.count("--n-gpu-layers") == 1 and not {"-lv", "--metrics", "--slots"} & set(
            command
        )
        return
    offers = _offers(run_dir)
    names = [offer["name"] for offer in offers]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    finding = next(f for f in metrics["diagnosis"]["findings"] if f["bottleneck"] == "offload")
    assert finding["state"] in ("critical", "warning")
    assert "host memory bandwidth" in finding["evidence"]
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
        assert "`offload-less`" in report.split("Try first:")[1].split("Could help:")[0]
        assert "checkpoint: GGUF Q6_K, " in out and " bits/weight" in out
        return
    assert list(steps) == ["host_expert_layers"]
    assert "raise-ubatch" in names and "offload-less" not in names
    assert "offload-experts" in names and "kv-q8" not in names
    tried = (run_dir / "report.md").read_text().split("Try first:")[1].split("Could help:")[0]
    assert "`offload-experts`" in tried


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
        assert code == 0 and f"`-ngl all --n-cpu-moe {n}`" in err
        code, _, err = _analyse(tmp_path / "project", capsys, monkeypatch)
    assert code != 0 and "your current setup cannot start" in err
    assert f"`-ngl all --n-cpu-moe {n}`" in err and "offload-experts" in err
    assert "tensward analyse --project" in err and f"--engine-arg n-cpu-moe={n}" in err
    assert (" --engine-arg ctx-size=4096" in err) == bool(pinned)
    assert not list((tmp_path / "project" / "runs").glob("*/server.log"))


@pytest.mark.parametrize("models", [{"data": "x"}, {"data": ["x"]}, {"data": [[]]}, ["x"]])
def test_a_malformed_models_listing_is_no_evidence(models) -> None:
    from tensward.engines.llamacpp.detect import probe_endpoint

    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=models if request.url.path == "/v1/models" else {})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        facts = probe_endpoint("http://host", client)
    assert facts is not None and not [e for e in facts.evidence if e.engine]
