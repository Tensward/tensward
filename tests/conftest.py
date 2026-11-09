"""Open-source tests run as if no closed-source analysis plugin were installed."""

import os
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from tensward.cli_options import set_json_output
from tensward.engines.vllm import caches
from tensward.extensions import DISABLE_ENV


@pytest.fixture(autouse=True)
def _restore_environ() -> Iterator[None]:
    """analyse puts the kept setup's engine environment in os.environ; no test may leak it."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture(autouse=True)
def _short_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tensward.capacity.PROBE_SECONDS", 0.8)


@pytest.fixture(autouse=True)
def _without_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DISABLE_ENV, "1")


@pytest.fixture(autouse=True)
def _no_gpu_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tensward.platforms.nvidia.SAMPLE_COMMAND", ("tensward-test-no-sampler",))


@pytest.fixture(autouse=True)
def _empty_compile_caches(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    home = str(tmp_path_factory.mktemp("compile-caches"))
    real = caches.cache_dirs

    def isolated(env: Mapping[str, str]) -> dict[str, Path]:
        return real({**env, "HOME": home, "XDG_CACHE_HOME": ""})

    monkeypatch.setattr(caches, "cache_dirs", isolated)


@pytest.fixture(autouse=True)
def _knowledge_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    monkeypatch.setenv("TENSWARD_KB_HOME", str(tmp_path_factory.mktemp("kb-home")))


@pytest.fixture(autouse=True)
def _cache_home(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path_factory.mktemp("cache-home")))


@pytest.fixture(autouse=True)
def _json_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TENSWARD_JSON", "1")
    set_json_output(True)
