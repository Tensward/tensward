"""The built wheels: the Linux x86_64 one carries the census library, the pure one does not."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBE = (
    "import json; from tensward.census.library import census_home, find_library, installed; "
    "home = census_home(); "
    "print(json.dumps({'home': str(home), 'installed': installed(home), "
    "'cu12': str(find_library(home, 12)), 'cu13': str(find_library(home, 13))}))"
)


def _run(*command: str, env: dict[str, str] | None = None) -> str:
    done = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
    assert done.returncode == 0, done.stderr
    return done.stdout


def _install_and_probe(tmp_path: Path, platform_wheel: bool) -> dict[str, object]:
    env = {k: v for k, v in os.environ.items() if k != "TENSWARD_CENSUS_DIR"}
    env.pop("VIRTUAL_ENV", None)
    if platform_wheel:
        env["TENSWARD_PLATFORM_WHEEL"] = "1"
    else:
        env.pop("TENSWARD_PLATFORM_WHEEL", None)
    dist = tmp_path / "dist"
    _run("uv", "build", "--wheel", "-o", str(dist), str(ROOT), env=env)
    wheel = next(dist.glob("*.whl"))
    assert wheel.name.endswith("manylinux_2_28_x86_64.whl") == platform_wheel
    venv = tmp_path / "venv"
    _run("uv", "venv", "-q", str(venv), env=env)
    python = venv / "bin" / "python"
    _run("uv", "pip", "install", "-q", "--python", str(python), str(wheel), env=env)
    return json.loads(_run(str(python), "-I", "-c", PROBE, env=env))


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "linux", reason="the library is built for Linux")
def test_a_fresh_install_finds_the_census_without_any_env(tmp_path: Path) -> None:
    found = _install_and_probe(tmp_path, platform_wheel=True)
    assert found["installed"] is True
    assert str(found["home"]).endswith("site-packages/tensward/census/lib")
    assert "libtwcensus-v2-cu12.so" in str(found["cu12"])
    assert "libtwcensus-v2-cu13.so" in str(found["cu13"])


@pytest.mark.slow
def test_the_pure_wheel_says_the_census_is_not_installed(tmp_path: Path) -> None:
    found = _install_and_probe(tmp_path, platform_wheel=False)
    assert found["installed"] is False
    assert found["cu12"] == found["cu13"] == "None"
