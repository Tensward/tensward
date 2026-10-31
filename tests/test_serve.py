"""``tensward serve`` end to end with the local runtime and the fake server."""

from __future__ import annotations

import dataclasses
import json
import os
import select
import shlex
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from interrupts import signal_once_present
from registration_fixtures import (
    REGISTRATION_CONFIG,
    make_checkpoint,
    registered_project,
    write_config,
    write_prompts,
)

from tensward.cli import main
from tensward.engines import ENGINES
from tensward.kept import keep
from tensward.runtime import (
    DockerRuntime,
    LocalProcessRuntime,
    PersistentServe,
    ServeSpec,
    select_loopback_port,
)
from tensward.settings import Settings

FAKE_SERVER = Path(__file__).resolve().parent / "fake_vllm_server.py"
LAN_HOST = ".".join(
    ["10", "0", "0", "5"]
)  # a private address, built so no literal IP is checked in
SETTINGS_OBJ = Settings(
    dtype="bfloat16",
    max_concurrent_requests=8,
    max_context_len=2048,
    kv_memory_fraction=0.8,
    prefill_batch_tokens=2048,
    kv_cache_dtype="auto",
    prefix_caching=False,
)
SETTINGS = dataclasses.asdict(SETTINGS_OBJ)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return "Z" not in Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]


def _start_default(project: Path, *words: str) -> dict:
    command = shlex.join([sys.executable, str(FAKE_SERVER)])
    assert main(["serve", "start", "--project", str(project), "--runtime", "local",
                 "--local-command", command, "--port", str(select_loopback_port()),
                 "--ready-timeout", "30", *words]) == 0  # fmt: skip
    return json.loads((project / "serve" / "default" / "state.json").read_text())


def test_serve_start_without_an_optimize_result_serves_the_current_setup(
    tmp_path: Path,
) -> None:
    project = registered_project(tmp_path)

    state = _start_default(project)

    assert state["source"] == "current"
    assert main(["serve", "stop", "--project", str(project)]) == 0


def test_serve_start_loads_the_project_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tensward.serve as serve

    project = registered_project(tmp_path)
    loads = []
    real = serve.load_project
    monkeypatch.setattr(serve, "load_project", lambda *a, **k: loads.append(1) or real(*a, **k))

    _start_default(project)

    assert len(loads) == 1
    assert main(["serve", "stop", "--project", str(project)]) == 0


PACKAGE = {"name": "balanced", "confirmed": True, "requires_review": False,
           "settings": {**SETTINGS, "max_concurrent_requests": 4}}  # fmt: skip
METRICS = {"failed": 0, "goodput": 1.2, "output_throughput": 80.0, "request_throughput": 1.3,
           "slo_attainment": 1.0, "total_throughput": 900.0, "tpot_p95_ms": 9.0,
           "ttft_p95_ms": 120.0}  # fmt: skip
RELEASED_OPTIMIZER_PACKAGE = {
    **PACKAGE, "objective": "goodput", "limits": {"max_tpot_p95_ms": None, "max_ttft_p95_ms": None},
    "adopted_recipes": ["raise-concurrency"], "metrics": METRICS,
    "vs_current": {name: 0.1 for name in METRICS}, "max_users_under_slo": 8,
    "sweep": [{"concurrency": 8, "meets_slo": True, "metrics": METRICS}], "sweep_not_measured": [],
}  # fmt: skip
SERVE_SOURCES = {
    "packages-file": {"schema_version": "1", "packages": [PACKAGE], "recommended": "balanced"},
    "optimizer-0.3.1": {
        "opt_id": "20260101T000000Z-abcd", "slo": {"tpot_ms": 50.0, "ttft_ms": 2000.0},
        "declared_concurrency": 8, "screening_requests": 8, "workload_requests": 16,
        "stopped": None, "note": "",
        "current": {"settings": SETTINGS, "metrics": METRICS, "sweep": [],
                    "sweep_not_measured": [], "max_users_under_slo": 4},
        "recommended": "balanced", "packages": [RELEASED_OPTIMIZER_PACKAGE],
        "no_improvement": [], "failed_confirmation": [], "dropped": [],
    },
}  # fmt: skip


@pytest.mark.parametrize("source", SERVE_SOURCES)
def test_serve_start_status_stop_with_a_detached_local_server(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], source: str
) -> None:
    project = registered_project(tmp_path)
    local = shlex.join([sys.executable, str(FAKE_SERVER)])
    assert main(["analyse", "--project", str(project), "--runtime", "local",
                 "--local-command", local, "--engine-arg", "max-num-seqs=4"]) == 0  # fmt: skip
    (run_dir,) = (project / "runs").iterdir()
    keep(project, run_dir.name, accept_unreviewed=False)
    opt_dir = project / "optimize" / "20260101T000000Z-abcd"
    opt_dir.mkdir(parents=True)
    written = json.dumps(SERVE_SOURCES[source])
    (opt_dir / "packages.json").write_text(written)
    capsys.readouterr()

    state = _start_default(project)
    out, err = capsys.readouterr()
    assert f"a kept setup exists (run {run_dir.name}): serve it with --from current" in err
    key_file = project / "serve" / "default" / "api_key"
    assert str(key_file) in out and key_file.read_text() not in out
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600

    pid = state["identity"]["pid"]
    assert (
        state["source"] == f"{opt_dir.name}:balanced"
        and state["settings"]["max_concurrent_requests"] == 4
    )
    assert (opt_dir / "packages.json").read_text() == written
    reply = httpx.post(
        f"{state['endpoint']}/v1/completions",
        headers={"Authorization": f"Bearer {key_file.read_text()}"},
        json={"model": state["served_model_name"], "prompt": "Hello", "max_tokens": 4},
    )
    assert reply.status_code == 200

    time.sleep(0.5)  # main() has returned; the server must not have been torn down with it
    assert _alive(pid)
    assert main(["serve", "status", "--project", str(project)]) == 0

    assert main(["serve", "stop", "--project", str(project)]) == 0
    assert not _alive(pid)
    assert (
        json.loads((project / "serve" / "default" / "state.json").read_text())["status"]
        == "stopped"
    )
    assert main(["serve", "status", "--project", str(project)]) == 1

    kept = _start_default(project, "--from", "current")
    assert kept["source"] == f"current (kept from run {run_dir.name})"
    assert kept["settings"]["max_concurrent_requests"] == 4
    capsys.readouterr()
    assert main(["set", "registered", "--project", str(project)]) == 0
    running = "a server already running keeps its settings until it is started again"
    assert running in capsys.readouterr().out.splitlines()
    assert main(["serve", "stop", "--project", str(project)]) == 0


@pytest.mark.parametrize("content", ["{not json", '{"status": "running"}', "[]"])
def test_serve_refuses_a_corrupted_state_file_naming_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str
) -> None:
    project = registered_project(tmp_path)
    state_file = project / "serve" / "default" / "state.json"
    state_file.parent.mkdir(parents=True)
    state_file.write_text(content)
    for command in ("status", "stop"):
        assert main(["serve", command, "--project", str(project)]) == 1
    assert str(state_file) in capsys.readouterr().err


def test_docker_persistent_argv_restarts_labels_and_hides_the_key(tmp_path: Path) -> None:
    secret = "s3cret-api-key-value"
    spec = ServeSpec(ENGINES["vllm"], tmp_path, "m", SETTINGS_OBJ, port=18000, api_key=secret)
    argv = DockerRuntime("img:test").build_run_argv(
        spec, "abc", persistent=PersistentServe("prod", tmp_path)
    )
    assert argv[argv.index("--restart") + 1] == "unless-stopped"
    assert "tensward.serve=prod" in argv
    assert secret not in " ".join(argv)


def _slow_start_args(project: Path, *more: str) -> list[str]:
    command = shlex.join([sys.executable, str(FAKE_SERVER)])
    return ["serve", "start", "--project", str(project), "--runtime", "local",
            "--local-command", command, "--port", str(select_loopback_port()), *more]  # fmt: skip


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_serve_start_interrupted_while_loading_stops_its_server_and_marks_the_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sig: signal.Signals
) -> None:
    monkeypatch.setenv("FAKE_LOAD_SECONDS", "60")
    project = registered_project(tmp_path)
    signal_once_present((project / "serve" / "default", "state.json"), sig)

    assert main(_slow_start_args(project, "--ready-timeout", "120")) == 130

    state = json.loads((project / "serve" / "default" / "state.json").read_text())
    assert state["status"] == "failed" and not _alive(state["identity"]["pid"])
    assert main(["serve", "stop", "--project", str(project)]) == 0  # nothing left to stop


def test_serve_start_that_times_out_stops_its_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FAKE_LOAD_SECONDS", "60")
    monkeypatch.setattr("tensward.runtime.STILL_LOADING_EVERY_S", 0.5)
    project = registered_project(tmp_path)

    assert main(_slow_start_args(project, "--ready-timeout", "2")) == 1

    state = json.loads((project / "serve" / "default" / "state.json").read_text())
    assert state["status"] == "failed" and "not ready within 2s" in state["error"]
    assert not _alive(state["identity"]["pid"])
    assert "still loading... " in capsys.readouterr().err


def test_serve_stop_finds_a_server_that_is_still_starting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = registered_project(tmp_path)
    state_file = project / "serve" / "default" / "state.json"
    arguments = [sys.executable, "-m", "tensward.cli", *_slow_start_args(project)]
    arguments += ["--ready-timeout", "120", "--engine-arg", "max-num-seqs=4"]
    start = subprocess.Popen(arguments, env={**os.environ, "FAKE_LOAD_SECONDS": "60"})
    try:
        for _ in range(200):
            if state_file.exists():
                break
            time.sleep(0.05)
        state = json.loads(state_file.read_text())
        pid = state["identity"]["pid"]
        assert state["status"] == "starting" and state["engine_args"] == ["max-num-seqs=4"]
        assert state["settings"]["max_concurrent_requests"] == 4  # the override is what serves
        capsys.readouterr()
        assert main(["serve", "status", "--project", str(project)]) == 1  # not ready yet
        status = capsys.readouterr().out
        assert "default: starting" in status and "current + overrides (max-num-seqs=4)" in status

        assert main(["serve", "stop", "--project", str(project)]) == 0

        assert not _alive(pid)
        assert start.wait(timeout=30) == 1  # the start command notices and gives up
    finally:
        start.kill()


def test_serve_start_whose_state_write_fails_stops_its_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import tensward.serve as serve

    started = []
    real = LocalProcessRuntime.start_persistent

    def recorded(runtime: LocalProcessRuntime, *args: Any, **kwargs: Any) -> Any:
        started.append(real(runtime, *args, **kwargs))
        return started[-1]

    monkeypatch.setattr(LocalProcessRuntime, "start_persistent", recorded)

    def failing_write(directory: Path, state: dict) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(serve, "_write_state", failing_write)
    project = registered_project(tmp_path)

    assert main(_slow_start_args(project)) == 1

    assert "No space left on device" in capsys.readouterr().err
    assert not _alive(started[0].identity()["pid"])


def test_serve_start_names_a_malformed_packages_json_instead_of_crashing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = registered_project(tmp_path)
    opt_dir = project / "optimize" / "20260101T000000Z-abcd"
    opt_dir.mkdir(parents=True)
    (opt_dir / "packages.json").write_text(
        json.dumps({"schema_version": "1", "recommended": "balanced"})
    )

    assert main(_slow_start_args(project)) == 1

    err = capsys.readouterr().err
    assert "20260101T000000Z-abcd" in err and "malformed packages.json" in err
    assert "Traceback" not in err


def test_docker_start_that_times_out_is_a_one_line_failure_and_removes_the_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    calls = []

    class Done:
        returncode, stdout, stderr = 0, "sha256:abc\n", ""

    def fake_run(argv, **kwargs):
        calls.append(list(argv[1:3]))
        if argv[1] == "run":
            raise subprocess.TimeoutExpired(argv, 60)
        return Done()

    monkeypatch.setattr("tensward.runtime.subprocess.run", fake_run)
    project = registered_project(tmp_path)

    code = main(["serve", "start", "--project", str(project), "--image", "img:1"])

    err = capsys.readouterr().err
    assert code == 1 and err.strip().splitlines()[-1].endswith("docker run timed out")
    assert "Traceback" not in err
    assert ["rm", "-f"] in calls  # the container that may have been created is removed


def test_signals_during_cleanup_do_not_abort_it_and_handlers_come_back() -> None:
    from tensward.runtime import uninterruptible

    before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    with uninterruptible():
        os.kill(os.getpid(), signal.SIGINT)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.05)  # neither raised: the block completes
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before


def test_a_second_ctrl_c_while_the_interrupted_start_stops_its_server_does_not_leak_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_LOAD_SECONDS", "60")
    real_killpg = os.killpg

    def killpg_after_a_second_ctrl_c(group: int, sig: int) -> None:
        os.kill(os.getpid(), signal.SIGINT)  # the user presses Ctrl-C again mid-cleanup
        real_killpg(group, sig)

    monkeypatch.setattr("tensward.runtime.os.killpg", killpg_after_a_second_ctrl_c)
    project = registered_project(tmp_path)
    signal_once_present((project / "serve" / "default", "state.json"), signal.SIGINT)

    assert main(_slow_start_args(project, "--ready-timeout", "120")) == 130

    state = json.loads((project / "serve" / "default" / "state.json").read_text())
    assert state["status"] == "failed" and not _alive(state["identity"]["pid"])


@pytest.mark.parametrize(
    ("host", "published", "endpoint"),
    [
        ("127.0.0.1", "127.0.0.1:18000:8000", "http://127.0.0.1:18000"),
        ("localhost", "127.0.0.1:18000:8000", "http://127.0.0.1:18000"),
        ("0.0.0.0", "0.0.0.0:18000:8000", "http://127.0.0.1:18000"),
        ("::1", "[::1]:18000:8000", "http://[::1]:18000"),
        ("::", "[::]:18000:8000", "http://[::1]:18000"),
        (LAN_HOST, f"{LAN_HOST}:18000:8000", f"http://{LAN_HOST}:18000"),
    ],
)
def test_the_bound_host_is_what_docker_publishes_and_what_readiness_polls(
    tmp_path: Path, host: str, published: str, endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    spec = ServeSpec(
        ENGINES["vllm"], tmp_path, "m", SETTINGS_OBJ, port=18000, api_key="k", host=host
    )
    argv = DockerRuntime("img:test").build_run_argv(spec, "abc")
    assert argv[argv.index("-p") + 1] == published

    class Done:
        returncode, stdout, stderr = 0, "sha256:abc\n", ""

    monkeypatch.setattr("tensward.runtime.subprocess.run", lambda *a, **k: Done())
    assert DockerRuntime("img:test").start(spec).endpoint_url == endpoint
    local = LocalProcessRuntime(["vllm", "serve"])
    assert local.build_argv(spec)[local.build_argv(spec).index("--host") + 1] == spec.bind_host


def test_docker_gpu_selection_is_quoted_for_docker_and_defaults_to_all(tmp_path: Path) -> None:
    spec = ServeSpec(ENGINES["vllm"], tmp_path, "m", SETTINGS_OBJ, port=18000, api_key="k")
    argv = lambda gpus: DockerRuntime("img", gpus).build_run_argv(spec, "abc")  # noqa: E731
    assert argv(None)[argv(None).index("--gpus") + 1] == "all"
    assert argv(("1",))[argv(("1",)).index("--gpus") + 1] == '"device=1"'
    assert argv(("0", "1"))[argv(("0", "1")).index("--gpus") + 1] == '"device=0,1"'


def test_runtime_gpus_come_from_the_flag_else_the_current_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import argparse

    from tensward.cli_options import runtime_for

    model = make_checkpoint(tmp_path / "model")
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(write_config(tmp_path / "serving.json", REGISTRATION_CONFIG)),
                 "--prompts", str(write_prompts(tmp_path / "workload.jsonl")),
                 "--current", f"docker run --gpus device=1 -v {model}:/m img:1 --model /m",
                 ]) == 0  # fmt: skip

    def gpus(chosen: tuple[str, ...] | None, runtime: str = "docker"):
        arguments = argparse.Namespace(
            engine="vllm", runtime=runtime, image=None, project=project, gpus=chosen,
            local_command=None,
        )  # fmt: skip
        return runtime_for(arguments).gpus

    assert gpus(None) == ("1",) and gpus(("0", "2")) == ("0", "2") and gpus(None, "local") == ("1",)


def test_serve_from_must_be_a_plain_result_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = registered_project(tmp_path)

    assert main(_slow_start_args(project, "--from", "../../etc")) == 1

    assert "--from '../../etc' is not" in capsys.readouterr().err


def test_the_curl_example_quotes_a_key_file_path_with_spaces() -> None:
    from tensward.serve import curl_example

    state = {"endpoint": "http://127.0.0.1:8000", "served_model_name": "m"}
    assert "$(cat '/my project/api_key')" in curl_example(state, Path("/my project/api_key"))


def test_state_is_replaced_atomically_and_privately(tmp_path: Path) -> None:
    from tensward.serve import _write_state

    _write_state(tmp_path, {"status": "starting"})
    _write_state(tmp_path, {"status": "running"})

    assert json.loads((tmp_path / "state.json").read_text()) == {"status": "running"}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]  # no temporary left behind


def test_concurrent_state_writes_never_collide(tmp_path: Path) -> None:
    """`serve stop` can write the state while `serve start` is still writing it. With a fixed
    temporary name one writer renamed the other's file away (FileNotFoundError, seen in CI)."""
    from concurrent.futures import ThreadPoolExecutor

    from tensward.files import write_json

    target = tmp_path / "state.json"

    def writes(worker: int) -> None:
        for step in range(200):
            write_json(target, {"worker": worker, "step": step})

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(writes, range(8)))  # re-raises any writer's exception
    assert json.loads(target.read_text())["step"] == 199
    assert [path.name for path in tmp_path.iterdir()] == ["state.json"]  # no temp files left


def test_docker_keeps_the_engine_compile_caches_between_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    spec = ServeSpec(ENGINES["vllm"], tmp_path, "m", SETTINGS_OBJ, port=18000, api_key="k")
    argv = DockerRuntime("img:test").build_run_argv(spec, "abc", image_id="sha256:" + "ab12" * 16)
    root = tmp_path / "cache" / "tensward" / "engine-cache" / "ab12ab12ab12ab12"
    for name in ("vllm", "triton", "flashinfer"):
        assert f"{root / name}:/root/.cache/{name}" in argv
        assert (root / name).stat().st_mode & 0o777 == 0o700
    assert "TRITON_CACHE_DIR=/root/.cache/triton" in argv
    assert "/root/.cache" not in argv


def test_window_records_which_compile_caches_were_warm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiled = tmp_path / "vllm-root" / "torch_compile_cache"
    compiled.mkdir(parents=True)
    (compiled / "graph").write_bytes(b"x" * 100)
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm-root"))
    project = registered_project(tmp_path)
    local = shlex.join([sys.executable, str(FAKE_SERVER)])
    assert main(["analyse", "--project", str(project), "--runtime", "local",
                 "--local-command", local, "--engine-arg", "max-num-seqs=4"]) == 0  # fmt: skip
    (run_dir,) = (project / "runs").iterdir()
    caches = json.loads((run_dir / "window.json").read_text())["caches"]
    assert caches.keys() == {"vllm", "triton", "inductor", "flashinfer"}
    assert caches["inductor"] == {"before": 100, "after": 100, "warm": True}
    assert caches["flashinfer"]["warm"] is False


GEMMA_CHAT = Path(__file__).parent / "replays" / "l4-gemma-chat"


@pytest.mark.parametrize(
    ("parent", "step", "step_args", "stops"),
    [
        ("20261016T174304Z-567c", "20261016T175324Z-bf38",
         {"gpu-memory-utilization=0.95", "language-model-only"}, True),
        ("20261016T172654Z-ed76", "20261016T174256Z-62c9", {"gpu-memory-utilization=0.95"}, False),
        ("20261016T172654Z-ed76", "20261016T175324Z-bf38", {"gpu-memory-utilization=0.5"}, False),
    ],
)  # fmt: skip
def test_the_start_up_guard_stops_only_a_pool_under_half_what_its_parent_used(
    parent: str, step: str, step_args: set[str], stops: bool
) -> None:
    from tensward.guard import arm, verdict
    from tensward.history import PastRun

    run = PastRun.read(GEMMA_CHAT / parent, None, ENGINES["vllm"])
    assert run is not None
    armed = arm(run, GEMMA_CHAT / parent, frozenset(step_args), ENGINES["vllm"])
    assert armed is not None

    reason = verdict(armed, (GEMMA_CHAT / step / "server.log").read_text(), ENGINES["vllm"])

    if stops:
        assert reason is not None
        assert reason.startswith("stopped at start-up: KV pool 0.36 GiB against the 3.6")
    else:
        assert reason is None


def test_pin_client_leaves_one_core_to_the_client_and_the_startup_plan_switch_reaches_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensward.runtime import client_cores

    cores = client_cores()
    if not cores:
        pytest.skip("one CPU core")
    monkeypatch.setenv("TENSWARD_STARTUP_PLAN", "1")
    server_script = tmp_path / "server"
    server_script.write_text("#!/bin/sh\nsleep 30\n")
    server_script.chmod(0o755)
    spec = ServeSpec(ENGINES["vllm"], tmp_path, "m", SETTINGS_OBJ, port=18000, api_key="k")

    server = LocalProcessRuntime([str(server_script)], pin_client=True).start(spec)
    try:
        pid = server.identity()["pid"]
        assert os.sched_getaffinity(pid) == set(cores)
        assert b"VLLM_ENABLE_STARTUP_PLAN=1" in Path(f"/proc/{pid}/environ").read_bytes()
    finally:
        server.stop()

    argv = DockerRuntime("img:test", pin_client=True).build_run_argv(spec, "abc")
    assert argv[argv.index("--cpuset-cpus") + 1] == ",".join(map(str, cores))
    assert "VLLM_ENABLE_STARTUP_PLAN=1" in argv


LAUNCHER = Path(__file__).resolve().parent / "launcher.py"


def _answers(port: int) -> bool:
    try:
        return httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200
    except httpx.HTTPError:
        return False


def test_an_engine_launcher_starts_the_server_with_its_env_as_pairs_and_stops_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "launcher.jsonl"
    monkeypatch.setenv("LAUNCHER_RECORD", str(record))
    monkeypatch.setenv("TENSWARD_STARTUP_PLAN", "1")
    monkeypatch.setenv("TENSWARD_ENGINE_LAUNCHER", f"{sys.executable} {LAUNCHER} --")
    monkeypatch.setenv("T9_CALLER_ONLY", "must-not-reach-engine")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    monkeypatch.setattr("tensward.measure.generate_api_key", lambda: "synthetic-t9-api-key")
    monkeypatch.setattr("tensward.platforms.nvidia.NvidiaPlatform.detect", lambda self: ())
    model = make_checkpoint(tmp_path / "model")
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(write_config(tmp_path / "serving.json")),
                 "--prompts", str(write_prompts(tmp_path / "workload.jsonl")),
                 "--current", f"VLLM_USE_V2_MODEL_RUNNER=0 vllm serve {model}"]) == 0  # fmt: skip
    observed = tmp_path / "child-env.json"
    expected = {
        "VLLM_API_KEY": "synthetic-t9-api-key",
        "VLLM_ENABLE_STARTUP_PLAN": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "CUDA_VISIBLE_DEVICES": "2,5",
        "CUDA_DEVICE_ORDER": "FASTEST_FIRST",
    }
    names = ("PATH", "LANG", *expected, "T9_CALLER_ONLY")
    entry = tmp_path / "engine.py"
    entry.write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        f"names = {names!r}\n"
        f"Path({str(observed)!r}).write_text(json.dumps("
        "{k: os.environ[k] for k in names if k in os.environ}))\n"
        f"os.execv(sys.executable, [sys.executable, {str(FAKE_SERVER)!r}, *sys.argv[1:]])\n"
    )
    command = shlex.join([sys.executable, str(entry)])

    assert main(["analyse", "--project", str(project), "--runtime", "local",
                 "--local-command", command, "--gpus", "2,5",
                 "--engine-arg", "max-num-seqs=4"]) == 0  # fmt: skip

    (launch,) = map(json.loads, record.read_text().splitlines())
    assert launch["command"][:2] == [sys.executable, str(entry)]
    assert "--api-key" not in launch["command"]
    assert dict(pair.split("=", 1) for pair in launch["env"]) == expected
    assert json.loads(observed.read_text()) == {"PATH": os.defpath, "LANG": "C.UTF-8", **expected}
    (run_dir,) = (project / "runs").iterdir()
    serving = json.loads((run_dir / "serve.json").read_text())
    assert serving["identity"]["argv"] == launch["command"]
    run = json.loads((run_dir / "run.json").read_text())
    assert run["runtime"] == "local process"
    assert run["gpus"] == ["2", "5"]
    assert run["settings"]["extra_env"] == {"VLLM_USE_V2_MODEL_RUNNER": "0"}
    assert str(LAUNCHER) not in json.dumps(run)
    assert not _alive(launch["pid"])


@pytest.mark.parametrize("launcher", [None, "env"])
def test_without_a_trailing_double_dash_the_engine_inherits_the_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launcher: str | None
) -> None:
    if launcher is None:
        monkeypatch.delenv("TENSWARD_ENGINE_LAUNCHER", raising=False)
    else:
        monkeypatch.setenv("TENSWARD_ENGINE_LAUNCHER", launcher)
    monkeypatch.setenv("TENSWARD_STARTUP_PLAN", "1")
    monkeypatch.setenv("T9_CALLER_ONLY", "inherited")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    expected = {
        "VLLM_API_KEY": "synthetic-inherited",
        "VLLM_ENABLE_STARTUP_PLAN": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "CUDA_VISIBLE_DEVICES": "2,5",
        "CUDA_DEVICE_ORDER": "FASTEST_FIRST",
        "T9_CALLER_ONLY": "inherited",
    }
    observed = tmp_path / "inherited.json"
    script = tmp_path / "engine.py"
    script.write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        f"names = {tuple(expected)!r}\n"
        f"Path({str(observed)!r}).write_text(json.dumps({{\n"
        "    'env': {k: os.environ.get(k) for k in names}, 'argv': sys.argv,\n"
        "    'path': os.environ['PATH']}))\n"
    )
    settings = dataclasses.replace(SETTINGS_OBJ, extra_env={"VLLM_USE_V2_MODEL_RUNNER": "0"})
    spec = ServeSpec(
        ENGINES["vllm"], tmp_path, "m", settings, port=18000, api_key="synthetic-inherited"
    )

    server = LocalProcessRuntime([sys.executable, str(script)], gpus=("2", "5")).start(spec)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and _alive(server.identity()["pid"]):
            time.sleep(0.05)
        child = json.loads(observed.read_text())
        assert child["env"] == expected
        assert child["path"].split(os.pathsep)[0] == str(Path(sys.executable).parent)
        assert server.identity()["argv"] == [sys.executable, *child["argv"]]
    finally:
        server.stop()


def test_a_server_started_through_a_launcher_can_be_stopped_and_started_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tmp_path / "launcher.jsonl"
    marker = tmp_path / "busy.json"
    monkeypatch.setenv("LAUNCHER_RECORD", str(record))
    monkeypatch.setenv("LAUNCHER_BUSY_MARKER", str(marker))
    monkeypatch.setenv("TENSWARD_ENGINE_LAUNCHER", f"{sys.executable} {LAUNCHER} --")
    port = select_loopback_port()
    spec = ServeSpec(ENGINES["vllm"], tmp_path, "m", SETTINGS_OBJ, port=port, api_key="k")
    runtime = LocalProcessRuntime([sys.executable, str(FAKE_SERVER)])

    keeper = subprocess.Popen(
        [sys.executable, str(LAUNCHER), "--busy-keeper", str(marker)],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    servers = []
    try:
        assert keeper.stdout is not None
        assert select.select([keeper.stdout], [], [], 5)[0]
        assert keeper.stdout.readline() == "ready\n"
        server = runtime.start(spec)
        servers.append(server)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not _answers(port):
            time.sleep(0.1)
        assert _answers(port)
        server.stop()
        stopped = json.loads(marker.read_text())
        assert keeper.poll() is None
        assert not _answers(port)

        restarted = runtime.start(spec)
        servers.append(restarted)
        assert time.monotonic() - stopped["stopped_at"] < 2
        deadline = time.monotonic() + 35
        while time.monotonic() < deadline and not _answers(port):
            time.sleep(0.1)
        assert _answers(port)
        launches = [json.loads(line) for line in record.read_text().splitlines()]
        first, second = launches
        assert first["wait_started_at"] is None
        assert second["busy"] == stopped
        assert second["wait_started_at"] - stopped["stopped_at"] < 2
        assert second["launched_at"] - stopped["stopped_at"] >= 20
        assert second["launched_at"] >= stopped["release_at"]
        assert not marker.exists()
        assert "GPU KV cache size" in restarted.log_text()
        restarted.stop()
    finally:
        for server in servers:
            server.stop()
        keeper.terminate()
        keeper.wait(timeout=5)
        if keeper.stdout is not None:
            keeper.stdout.close()
    assert not _alive(keeper.pid)
    for line in record.read_text().splitlines():
        launch = json.loads(line)
        assert not _alive(launch["pid"])
        assert not _alive(launch["child_pid"])


def test_a_profiler_launch_prefix_bypasses_the_engine_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TENSWARD_ENGINE_LAUNCHER", "false")
    observed = tmp_path / "profiled.json"
    script = tmp_path / "engine.py"
    script.write_text(
        "import json, os\nfrom pathlib import Path\n"
        f"Path({str(observed)!r}).write_text(json.dumps(os.environ['T9_PROFILER_PREFIX']))\n"
    )
    spec = ServeSpec(
        ENGINES["vllm"],
        tmp_path,
        "m",
        SETTINGS_OBJ,
        port=18000,
        api_key="synthetic",
        launch_prefix=("env", "T9_PROFILER_PREFIX=active"),
    )
    server = LocalProcessRuntime([sys.executable, str(script)]).start(spec)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and _alive(server.identity()["pid"]):
            time.sleep(0.05)
        assert json.loads(observed.read_text()) == "active"
        assert not _alive(server.identity()["pid"])
    finally:
        server.stop()
