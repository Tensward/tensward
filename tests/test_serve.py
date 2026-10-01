"""``tensward serve`` end to end with the local runtime and the fake server."""

from __future__ import annotations

import dataclasses
import json
import os
import shlex
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from interrupts import signal_once_present
from registration_fixtures import REGISTRATION_CONFIG, make_checkpoint, write_config, write_prompts

from tensward.cli import main
from tensward.engines import ENGINES
from tensward.engines.protocol import Settings
from tensward.runtime import (
    DockerRuntime,
    LocalProcessRuntime,
    PersistentServe,
    ServeSpec,
    select_loopback_port,
)

FAKE_SERVER = Path(__file__).resolve().parent / "fake_vllm_server.py"
LAN_HOST = ".".join(
    ["10", "0", "0", "5"]
)  # a private address, built so no literal IP is checked in
SETTINGS_OBJ = Settings("bfloat16", 8, 2048, 0.8, 2048, "auto", False)
SETTINGS = dataclasses.asdict(SETTINGS_OBJ)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return "Z" not in Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]


def _register(tmp_path: Path) -> Path:
    model = make_checkpoint(tmp_path / "model")
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(write_config(tmp_path / "serving.json", REGISTRATION_CONFIG)),
                 "--prompts", str(write_prompts(tmp_path / "workload.jsonl"))]) == 0  # fmt: skip
    return project


def _start_default(project: Path) -> dict:
    command = shlex.join([sys.executable, str(FAKE_SERVER)])
    assert main(["serve", "start", "--project", str(project), "--runtime", "local",
                 "--local-command", command, "--port", str(select_loopback_port()),
                 "--ready-timeout", "30"]) == 0  # fmt: skip
    return json.loads((project / "serve" / "default" / "state.json").read_text())


def test_serve_start_without_an_optimize_result_serves_the_current_setup(
    tmp_path: Path,
) -> None:
    project = _register(tmp_path)

    state = _start_default(project)

    assert state["source"] == "current"
    assert main(["serve", "stop", "--project", str(project)]) == 0


def test_serve_start_loads_the_project_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tensward.serve as serve

    project = _register(tmp_path)
    loads = []
    real = serve.load_project
    monkeypatch.setattr(serve, "load_project", lambda *a, **k: loads.append(1) or real(*a, **k))

    _start_default(project)

    assert len(loads) == 1
    assert main(["serve", "stop", "--project", str(project)]) == 0


def test_serve_start_status_stop_with_a_detached_local_server(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _register(tmp_path)
    opt_dir = project / "optimize" / "20260101T000000Z-abcd"
    opt_dir.mkdir(parents=True)
    package = {"name": "balanced", "confirmed": True, "requires_review": False,
               "settings": {**SETTINGS, "max_concurrent_requests": 4}}  # fmt: skip
    (opt_dir / "packages.json").write_text(
        json.dumps({"packages": [package], "recommended": "balanced"})
    )
    capsys.readouterr()

    state = _start_default(project)
    out = capsys.readouterr().out
    key_file = project / "serve" / "default" / "api_key"
    assert str(key_file) in out and key_file.read_text() not in out
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600

    pid = state["identity"]["pid"]
    assert (
        state["source"] == f"{opt_dir.name}:balanced"
        and state["settings"]["max_concurrent_requests"] == 4
    )
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
    project = _register(tmp_path)
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
    project = _register(tmp_path)

    assert main(_slow_start_args(project, "--ready-timeout", "2")) == 1

    state = json.loads((project / "serve" / "default" / "state.json").read_text())
    assert state["status"] == "failed" and "not ready within 2s" in state["error"]
    assert not _alive(state["identity"]["pid"])
    assert "still loading... " in capsys.readouterr().err


def test_serve_stop_finds_a_server_that_is_still_starting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _register(tmp_path)
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
    real = serve._launch
    monkeypatch.setattr(serve, "_launch", lambda *a: started.append(real(*a)) or started[-1])

    def failing_write(directory: Path, state: dict) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(serve, "_write_state", failing_write)
    project = _register(tmp_path)

    assert main(_slow_start_args(project)) == 1

    assert "No space left on device" in capsys.readouterr().err
    assert not _alive(started[0].identity()["pid"])


def test_serve_start_names_a_malformed_packages_json_instead_of_crashing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _register(tmp_path)
    opt_dir = project / "optimize" / "20260101T000000Z-abcd"
    opt_dir.mkdir(parents=True)
    (opt_dir / "packages.json").write_text(json.dumps({"recommended": "balanced"}))

    assert main(_slow_start_args(project)) == 1

    err = capsys.readouterr().err
    assert "20260101T000000Z-abcd" in err and "malformed packages.json" in err
    assert "Traceback" not in err


def test_docker_start_that_times_out_is_a_one_line_failure_and_removes_the_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = []

    class Done:
        returncode, stdout, stderr = 0, "sha256:abc\n", ""

    def fake_run(argv, **kwargs):
        calls.append(list(argv[1:3]))
        if argv[1] == "run":
            raise subprocess.TimeoutExpired(argv, 60)
        return Done()

    monkeypatch.setattr("tensward.runtime.subprocess.run", fake_run)
    project = _register(tmp_path)

    code = main(["serve", "start", "--project", str(project), "--image", "img:1"])

    err = capsys.readouterr().err
    assert code == 1 and err.strip().splitlines()[-1].endswith("docker run timed out")
    assert "Traceback" not in err
    assert ["rm", "-f"] in calls  # the container that may have been created is removed


def test_analyse_with_a_bad_image_is_a_one_line_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _register(tmp_path)

    code = main(["analyse", "--project", str(project), "--image=-x"])

    err = capsys.readouterr().err
    assert code == 1 and err.strip().splitlines()[-1].endswith(
        "'-x' is not a docker image reference"
    )
    assert "Traceback" not in err


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
    project = _register(tmp_path)
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

    from tensward.cli import runtime_for

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
    project = _register(tmp_path)

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
