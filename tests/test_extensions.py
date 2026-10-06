"""A plugin built for another extension API is skipped with one line, never a traceback."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from tensward import extensions
from tensward.extensions import API_VERSION, Extender


def _entry(group: str, load: object, dist: str = "example-plugin") -> SimpleNamespace:
    return SimpleNamespace(group=group, load=load, dist=SimpleNamespace(name=dist))


def _broken() -> object:
    raise ImportError("cannot import name 'AnalyseFailure' from 'tensward.errors'")


@pytest.mark.parametrize(
    ("load", "said"),
    [
        (
            lambda: Extender(api_version=API_VERSION + 1, analyse=lambda *a: None),
            f"example-plugin needs extension API {API_VERSION + 1}; this Tensward has "
            f"{API_VERSION}; upgrade tensward",
        ),
        (
            _broken,
            "example-plugin could not be loaded (ImportError: cannot import name "
            "'AnalyseFailure' from 'tensward.errors'); upgrade example-plugin or tensward",
        ),
    ],
)
def test_a_mismatched_or_broken_analysis_plugin_is_skipped_with_one_line(
    load: object, said: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(extensions.DISABLE_ENV, raising=False)
    monkeypatch.setattr(extensions, "entry_points", lambda group: [_entry(group, load)])
    extensions._first_extender.cache_clear()
    assert extensions.load_extender() is None
    assert capsys.readouterr().err.splitlines() == [said]
    extensions._first_extender.cache_clear()


def test_a_command_plugin_for_another_api_is_skipped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    register = SimpleNamespace(API_VERSION=API_VERSION + 1)
    monkeypatch.delenv(extensions.DISABLE_ENV, raising=False)
    monkeypatch.setattr(extensions, "entry_points", lambda group: [_entry(group, lambda: register)])
    extensions.load_commands(argparse.ArgumentParser().add_subparsers())
    assert "needs extension API" in capsys.readouterr().err


def test_a_command_plugin_that_fails_to_register_is_skipped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def register(subcommands: object) -> None:
        raise KeyError("optimize")

    monkeypatch.delenv(extensions.DISABLE_ENV, raising=False)
    monkeypatch.setattr(extensions, "entry_points", lambda group: [_entry(group, lambda: register)])
    extensions.load_commands(argparse.ArgumentParser().add_subparsers())
    assert capsys.readouterr().err.splitlines() == [
        "example-plugin could not add its commands (KeyError: 'optimize')"
    ]
