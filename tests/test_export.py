from __future__ import annotations

from pathlib import Path

import pytest
from registration_fixtures import registered_project

from tensward.cli import main
from tensward.engines import ENGINES
from tensward.export import render
from tensward.runtime import RuntimeFailure

ARGV = ["/usr/bin/vllm", "serve", "/m", "--x", 'a$B "q" 50%']


def _render(fmt: str, argv: list[str] = ARGV, env: list[str] | None = None) -> str:
    return render(
        fmt,  # type: ignore[arg-type]
        argv=argv,
        engine=ENGINES["vllm"],
        version="0.30.0",
        model_path=Path("/m"),
        port=8000,
        env=env or [],
    )


def test_compose_keeps_special_characters_and_env_literal() -> None:
    text = _render("compose", env=["K=a$b"])
    assert '      - "a$$B \\"q\\" 50%"' in text
    assert '      - "K=a$$b"' in text
    assert '      - "127.0.0.1:8000:8000"' in text


def test_systemd_escapes_and_uses_absolute_path() -> None:
    text = _render("systemd", env=["K=50%"])
    assert 'ExecStart=/usr/bin/vllm serve /m --x "a$$B \\"q\\" 50%%" --host 127.0.0.1' in text
    assert 'Environment="K=50%%"' in text


def test_systemd_listens_on_loopback_unless_a_host_is_given() -> None:
    assert "--host 127.0.0.1" in _render("systemd")
    given = _render("systemd", [*ARGV, "--host", "0.0.0.0"])
    assert "--host 0.0.0.0" in given
    assert "127.0.0.1" not in given.split("ExecStart=")[1]


def test_helm_carries_env() -> None:
    assert '  - name: "K"\n    value: "v"' in _render("helm", env=["K=v"])


def test_an_export_before_any_run_pins_the_validated_vllm_image() -> None:
    # clean L4 vLLM leg: nothing kept, no run yet, and compose said vllm/vllm-openai:latest
    for fmt in ("compose", "helm"):
        text = render(
            fmt, argv=ARGV, engine=ENGINES["vllm"], version=None, model_path=Path("/m"), port=8000
        )
        assert "latest" not in text and "v0.30.0" in text


@pytest.mark.parametrize("fmt", ["compose", "systemd", "helm"])
def test_newline_is_refused(fmt: str) -> None:
    with pytest.raises(RuntimeFailure):
        _render(fmt, [*ARGV, "a\nb"])


def test_export_output_creates_missing_folders(tmp_path: Path) -> None:
    project = registered_project(tmp_path)
    target = tmp_path / "new" / "deeper" / "compose.yaml"
    assert (
        main(["export", "--format", "compose", "--project", str(project), "-o", str(target)]) == 0
    )
    assert "services" in target.read_text()
