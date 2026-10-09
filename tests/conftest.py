"""Open-source tests run as if no closed-source analysis plugin were installed."""

import pytest

from tensward.extensions import DISABLE_ENV


@pytest.fixture(autouse=True)
def _short_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tensward.capacity.PROBE_SECONDS", 0.8)


@pytest.fixture(autouse=True)
def _without_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DISABLE_ENV, "1")
