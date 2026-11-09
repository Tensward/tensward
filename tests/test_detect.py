import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tensward.detect import Hardware, running_servers, serving_problems


@pytest.mark.parametrize(
    "engine, kind, backends, stop, words",
    [
        ("vllm", "apple", (), True, "llama.cpp is supported on this machine"),
        ("vllm", "cpu", (), True, "llama.cpp is supported on this machine"),
        ("vllm", "nvidia", (), False, None),
        ("llamacpp", "nvidia", ("CPU",), False, "will not use the GPU"),
        ("llamacpp", "nvidia", (), False, None),
        ("llamacpp", "nvidia", ("CUDA",), False, None),
        ("llamacpp", "apple", ("CPU",), True, "Metal"),
        ("llamacpp", "cpu", ("CPU",), False, None),
    ],
)
def test_engine_and_hardware_pairs_are_checked_before_any_question(
    engine, kind, backends, stop, words
):
    nvidia = kind == "nvidia"
    hardware = Hardware(
        kind=kind,
        devices=(("NVIDIA L4", 24 * 10**9),) if nvidia else (),
        driver="580.95" if nvidia else None,
        cuda=nvidia,
        ram_bytes=64 * 2**30,
        physical_cores=8,
        logical_cores=16,
    )
    problems = serving_problems(engine, hardware, backends)
    assert any(p.stop for p in problems) is stop
    assert (words is None and not problems) or any(words in p.text for p in problems)

    no_driver = Hardware("nvidia", (("NVIDIA L4", 24 * 10**9),), None, False, None, None, None)
    assert any(p.stop and "driver" in p.text for p in serving_problems("vllm", no_driver, ()))

    two = Hardware("nvidia", (("NVIDIA L4", 24 * 10**9),) * 2, "580.95", True, None, None, None)
    (problem,) = serving_problems("vllm", two, ())
    assert not problem.stop
    assert "Multi-GPU tuning is in Tensward Enterprise" in problem.text


def _orphan(arguments: list[str]) -> int:
    """Start ``python arguments`` outside this process's tree and return its pid."""
    out = subprocess.run(
        [
            "sh",
            "-c",
            f'{shlex.quote(sys.executable)} "$@" >/dev/null 2>&1 & echo $!',
            "sh",
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return int(out)


def test_running_servers_see_a_pip_vllm_and_skip_help_queries(tmp_path: Path) -> None:
    launcher = tmp_path / "vllm"
    launcher.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    llama = tmp_path / "llama-server"
    llama.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    pids = [_orphan([str(launcher), "serve", "/models/m"]), _orphan([str(llama), "--help"])]
    served, queried = pids
    try:
        deadline = time.monotonic() + 10
        found = {}
        while time.monotonic() < deadline and served not in found:
            found = {server.pid: server for server in running_servers()}
            time.sleep(0.1)
        assert found[served].engine == "vllm"
        assert queried not in found
    finally:
        for pid in pids:
            os.kill(pid, signal.SIGKILL)
