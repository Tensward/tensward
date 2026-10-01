"""Open-source tests run as if no closed-source analysis plugin were installed."""

import pytest

from tensward.extensions import DISABLE_ENV


@pytest.fixture(autouse=True)
def _without_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DISABLE_ENV, "1")
