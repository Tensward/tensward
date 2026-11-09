import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensward import serve
from tensward.errors import GPU_BUSY, PreflightError
from tensward.gpu_busy import refuse_if_busy
from tensward.platforms.protocol import GIB, MIB, Device

SMI = Path(__file__).parent / "fake_nvidia_smi.py"


def _machine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, used_mib: int, apps: str
) -> SimpleNamespace:
    shim = tmp_path / "nvidia-smi"
    shim.write_text(
        f"#!{sys.executable}\nimport runpy\nrunpy.run_path({str(SMI)!r}, run_name='__main__')\n"
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SMI_APPS", apps)
    device = Device("NVIDIA L4", 24576 * MIB, used_mib * MIB, "0", "GPU-0")
    return SimpleNamespace(
        memory_topology="discrete",
        detect=lambda: (device,),
        select=lambda devices, selection: (devices, ()),
    )


def test_a_busy_gpu_is_refused_and_nothing_is_signalled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    kills: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: kills.append((pid, sig)))
    needs = int(0.9 * 24576 * MIB)

    busy = _machine(monkeypatch, tmp_path, 21811, "4121,21800,GPU-0")
    with pytest.raises(PreflightError) as refused:
        refuse_if_busy(busy, None, needs, 6)  # type: ignore[arg-type]
    assert refused.value.code == GPU_BUSY
    assert "in use by" in refused.value.message
    assert "pid 4121" in refused.value.message
    assert "It will not stop your server" in refused.value.message
    assert kills == []

    free = _machine(monkeypatch, tmp_path, 500, "")
    assert refuse_if_busy(free, None, needs, 6) is None  # type: ignore[arg-type]


def test_a_mac_with_its_own_share_in_use_is_not_busy_and_a_short_one_says_how_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensward import gpu_busy
    from tensward.engines.llamacpp.engine import LlamaCppEngine
    from tensward.settings import Settings

    mac = SimpleNamespace(memory_topology="unified")
    weights = 16_500_000_000
    needs = gpu_busy.needs_for(LlamaCppEngine(), Settings(), weights, mac, None)  # type: ignore[arg-type]
    assert needs == weights
    monkeypatch.setattr(gpu_busy.darwin, "sysctl", lambda *names: {"hw.memsize": str(24 * GIB)})
    monkeypatch.setattr(gpu_busy.darwin, "available_gib", lambda: 18.0)
    assert refuse_if_busy(mac, None, needs, 10) is None  # type: ignore[arg-type]
    monkeypatch.setattr(gpu_busy.darwin, "available_gib", lambda: 9.0)
    with pytest.raises(PreflightError) as refused:
        refuse_if_busy(mac, None, needs, 10)  # type: ignore[arg-type]
    assert "the server needs 15.4 GB and 9 GB is free" in refused.value.message


def test_own_pids_include_the_processes_of_a_docker_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = {"runtime": "docker", "identity": {"container_id": "abc123"}}
    monkeypatch.setattr(serve, "read_state", lambda project, name: state)
    monkeypatch.setattr(serve, "is_alive", lambda state: True)
    (tmp_path / "serve" / "main").mkdir(parents=True)
    (tmp_path / "serve" / "main" / "state.json").write_text("{}")
    seen: list[list[str]] = []

    def probe(argv: list[str]) -> str:
        seen.append(argv)
        return "PID\n4242\n4300\n"

    monkeypatch.setattr(serve, "run_probe", probe)
    assert serve.own_pids(tmp_path) == {4242, 4300}
    assert seen == [["docker", "top", "abc123", "-eo", "pid"]]
