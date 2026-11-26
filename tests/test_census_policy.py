import hashlib
import json
from types import SimpleNamespace

import pytest

from tensward.census import library as locator
from tensward.census import policy
from tensward.census.library import Cupti
from tensward.census.policy import CensusPlan, CensusSkip, plan_census
from tensward.engines.llamacpp.engine import LLAMACPP
from tensward.engines.vllm.engine import VLLM

OLD_CUPTI = b"an old cupti"
LOCAL = SimpleNamespace(kind="local", command=("llama-server",))
DOCKER = SimpleNamespace(kind="docker", image="vllm/vllm-openai:v0.30.0")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def home(tmp_path):
    folder = tmp_path / "census"
    folder.mkdir()
    entries = []
    for major, api_min in ((12, 22), (13, 130000)):
        body = f"library {major}".encode()
        (folder / f"libtwcensus-v2-cu{major}.so").write_bytes(body)
        entries.append(
            {
                "file": f"libtwcensus-v2-cu{major}.so",
                "cupti_major": major,
                "cupti_api_min": api_min,
                "glibc_min": "2.25",
                "sha256": sha(body),
            }
        )
    manifest = {
        "census": "2.1",
        "libraries": entries,
        "cupti": [],
        "cupti_files": [{"sha256": sha(OLD_CUPTI), "api": 21}],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


@pytest.fixture
def cuda(monkeypatch):
    monkeypatch.setattr(locator, "engine_cupti_major", lambda profile, command: 12)
    monkeypatch.setattr(
        locator, "local_cupti", lambda profile, command, major: Cupti("/cuda/lib", "x")
    )
    monkeypatch.setattr(locator, "image_id_of", lambda image: "sha256:abc")
    monkeypatch.setattr(
        locator, "image_cupti", lambda profile, image_id, major: Cupti("/image/cu13/lib", "x")
    )


def plan(home, **over):
    args = dict(
        engine=LLAMACPP,
        runtime=LOCAL,
        version="12246",
        platform="nvidia",
        backend="cuda",
        parallel=1,
        load_s=30.0,
        warmup_s=10.0,
        wanted=True,
        home=home,
    )
    return plan_census(**{**args, **over})


ROWS = {
    "12246": (dict(), True),
    "13001": (dict(version="13001"), False),
    "11669": (
        dict(version="11669"),
        "the GPU census is off for llama.cpp build 11669: census starts crash on it",
    ),
    "12000": (
        dict(version="12000"),
        "the GPU census is off for llama.cpp build 12000: census starts crash on it",
    ),
    "vulkan": (dict(backend="vulkan"), "the GPU census needs a CUDA build of llama.cpp"),
    "vllm": (dict(engine=VLLM, version="0.30.0", runtime=DOCKER), True),
    "untested vllm": (dict(engine=VLLM, version="0.31.0", runtime=DOCKER), False),
    "parallel": (dict(parallel=2), "the GPU census is not tested with several GPUs"),
    "apple": (dict(platform="apple"), "the GPU census needs an NVIDIA GPU"),
    "docker llama": (
        dict(runtime=DOCKER),
        "the GPU census covers vLLM (host and docker) and llama.cpp (host)",
    ),
    "slow": (
        dict(load_s=520.0),
        "a census start would take about 11 min here (start 520 s, warm-up 10 s, then a 1-2 min "
        "recording), over its 10-minute limit, so it was skipped",
    ),
    "off": (dict(wanted=False), "skipped (--no-census)"),
}


@pytest.mark.parametrize("row", ROWS)
def test_the_census_runs_only_where_it_is_proven(row, home, cuda):
    over, expected = ROWS[row]
    result = plan(home, **over)
    if isinstance(expected, str):
        assert isinstance(result, CensusSkip)
        assert result.reason == expected
        return
    assert isinstance(result, CensusPlan)
    assert result.verified is expected
    assert result.expected_s + policy.MARGIN_S <= policy.CENSUS_CEILING_S


def test_build_11669_gets_the_measured_advice(home, cuda):
    result = plan(home, version="11669")
    assert result.reason.endswith("census starts crash on it")
    assert "build 12246 ran 100 census starts on an L4" in result.advice
    assert result.advice.endswith("run tensward analyse again.")


def test_a_missing_cupti_names_where_it_looked_and_how_to_fix_it(home, cuda, monkeypatch):
    monkeypatch.setattr(locator, "local_cupti", lambda profile, command, major: None)
    result = plan(home)
    assert result.reason == "no CUPTI found"
    assert result.advice.startswith(
        "Install with: uv pip install --target ~/.local/share/tensward/cupti "
        "'nvidia-cuda-cupti-cu12'. "
    )
    assert "TENSWARD_CUPTI" in result.advice and "/usr/local/cuda" in result.advice
    assert "libcupti.so.12" in result.advice


def test_an_empty_home_is_not_installed(tmp_path, cuda):
    result = plan(tmp_path / "none")
    assert result == CensusSkip(
        "the GPU census is not installed", "the GPU census ships for Linux x86_64"
    )


def test_a_cuda_without_a_census_library_is_named(home, cuda, monkeypatch):
    monkeypatch.setattr(locator, "engine_cupti_major", lambda profile, command: 11)
    assert plan(home).reason == "no census library for CUDA 11"


def test_cuda_13_on_a_volta_gpu_is_not_supported(home, cuda, monkeypatch):
    monkeypatch.setattr(locator, "engine_cupti_major", lambda profile, command: 13)
    monkeypatch.setattr(policy, "compute_capability", lambda selected: 7.0)
    assert plan(home).reason.startswith("the GPU census is not supported on this GPU")
    monkeypatch.setattr(policy, "compute_capability", lambda selected: 8.9)
    assert isinstance(plan(home), CensusPlan)


def test_the_locator_refuses_a_wrong_checksum_and_a_known_old_cupti(home):
    library = locator.find_library(home, 12)
    assert library is not None and library.kind == "cu12" and library.cupti_api_min == 22
    assert locator.cupti_too_old(home, Cupti("/x", sha(OLD_CUPTI)), library)
    assert not locator.cupti_too_old(home, Cupti("/x", sha(b"newer")), library)
    (home / "libtwcensus-v2-cu12.so").write_bytes(b"changed")
    assert locator.find_library(home, 12) is None
    assert locator.find_library(home, 13) is not None


def test_the_engines_cupti_is_found_by_its_real_path(tmp_path, monkeypatch):
    toolkit = tmp_path / "cuda-12.9" / "extras" / "CUPTI" / "lib64"
    toolkit.mkdir(parents=True)
    (toolkit / "libcupti.so.12").write_bytes(OLD_CUPTI)
    (tmp_path / "cuda").symlink_to(tmp_path / "cuda-12.9")
    monkeypatch.delenv("TENSWARD_CUPTI_DIR", raising=False)
    monkeypatch.setenv("TENSWARD_CUPTI_DIR", str(tmp_path / "cuda" / "extras" / "CUPTI" / "lib64"))
    found = locator.local_cupti(LLAMACPP.census, ("no-such-server",), 12)
    assert found == Cupti(str(toolkit), sha(OLD_CUPTI))


def test_a_llamacpp_cuda_backend_beside_the_binary_names_the_major(tmp_path, monkeypatch):
    bin_dir = tmp_path / "llama"
    bin_dir.mkdir()
    for name in ("llama-server", "libggml-cuda.so"):
        (bin_dir / name).write_bytes(b"")
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    seen = []

    def ldd(args, **kwargs):
        seen.append(args[1])
        text = "libcudart.so.12 => /x" if args[1].endswith("libggml-cuda.so") else "libc.so.6"
        return SimpleNamespace(stdout=text, returncode=0)

    monkeypatch.setattr(locator.subprocess, "run", ldd)
    assert locator.engine_cupti_major(LLAMACPP.census, ("llama-server",)) == 12
    assert str(bin_dir / "libggml-cuda.so") in seen


def test_an_unknown_major_still_names_the_folders_looked_in(home, monkeypatch):
    monkeypatch.setattr(locator, "engine_cupti_major", lambda profile, command: None)
    monkeypatch.setattr(
        locator,
        "cupti_search_dirs",
        lambda profile, command, major: [__import__("pathlib").Path(f"/cuda-{major}/lib")],
    )
    result = plan(home)
    assert isinstance(result, CensusSkip)
    assert "/cuda-12/lib" in result.advice and "/cuda-13/lib" in result.advice
    assert "no folder" not in result.advice


def test_a_vllm_start_is_judged_without_its_one_time_jit_build():
    from pathlib import Path

    from tensward.engines.vllm.start_phases import one_time_s

    log = Path(__file__).parent.joinpath("fixtures", "vllm-jit-gap.log").read_text("utf-8")
    assert 105 <= one_time_s(log) <= 112
    cold = (
        "(EngineCore pid=1) INFO 11-18 08:53:22 [backends.py:393] "
        "Compiling a graph for compile range (1, 8192) takes 20.97 s\n"
    )
    assert one_time_s(cold) == 20.97


def test_a_slow_request_rate_makes_the_recording_two_slices(home, cuda) -> None:
    assert isinstance(plan(home, load_s=7.0, warmup_s=440.0), CensusPlan)
    skipped = plan(home, load_s=7.0, warmup_s=440.0, req_s=0.027)
    assert isinstance(skipped, CensusSkip) and "warm-up 440 s" in skipped.reason


def test_a_wrapper_script_and_a_pip_wheel_give_the_census_its_cupti(tmp_path, monkeypatch):
    """The H2H-SK box: llama-server on PATH is a bash wrapper that execs the release binary,
    and the only CUPTI is NVIDIA's pip wheel in a virtual environment."""
    release = tmp_path / "opt" / "llama" / "b12246"
    release.mkdir(parents=True)
    for name in ("llama-server", "libggml-cuda.so"):
        (release / name).write_bytes(b"\x7fELF")
        (release / name).chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    wrapper = bin_dir / "llama-server"
    wrapper.write_text(
        f"#!/bin/bash\nexport LD_LIBRARY_PATH={release}:/usr/local/cuda/lib64\n"
        f'exec {release}/llama-server "$@"\n'
    )
    wrapper.chmod(0o755)
    wheel = tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "nvidia" / "cuda_cupti"
    (wheel / "lib").mkdir(parents=True)
    (wheel / "lib" / "libcupti.so.12").write_bytes(OLD_CUPTI)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.delenv("TENSWARD_CUPTI_DIR", raising=False)
    monkeypatch.delenv("TENSWARD_CUPTI", raising=False)
    monkeypatch.setattr(locator, "VENV_ROOTS", (str(tmp_path / "venv"),))
    monkeypatch.setattr(locator, "TOOLKIT_DIRS", ())
    seen = []

    def run(args, **kwargs):
        seen.append(args[-1])
        if args[0] == "ldconfig":
            return SimpleNamespace(stdout="", returncode=0)
        text = "libcudart.so.12 => not found" if "libggml-cuda" in args[1] else "libc.so.6"
        return SimpleNamespace(stdout=text, returncode=0)

    monkeypatch.setattr(locator.subprocess, "run", run)
    assert locator.engine_cupti_major(LLAMACPP.census, ("llama-server",)) == 12
    assert str(release / "libggml-cuda.so") in seen
    found = locator.local_cupti(LLAMACPP.census, ("llama-server",), 12)
    assert found == Cupti(str(wheel / "lib"), sha(OLD_CUPTI))


def test_tensward_cupti_may_name_the_file(tmp_path, monkeypatch):
    folder = tmp_path / "cupti"
    folder.mkdir()
    (folder / "libcupti.so.12.9.79").write_bytes(OLD_CUPTI)
    monkeypatch.setenv("TENSWARD_CUPTI", str(folder / "libcupti.so.12.9.79"))
    monkeypatch.setattr(locator, "VENV_ROOTS", ())
    monkeypatch.setattr(locator, "TOOLKIT_DIRS", ())
    monkeypatch.setattr(locator, "_ldconfig", lambda major: [])
    found = locator.local_cupti(LLAMACPP.census, ("no-such-server",), 12)
    assert found == Cupti(str(folder), sha(OLD_CUPTI))
