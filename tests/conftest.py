"""Open-source tests run as if no closed-source analysis plugin were installed."""

from collections.abc import Mapping
from pathlib import Path

import pytest

from tensward.engines.vllm import caches
from tensward.extensions import DISABLE_ENV


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
