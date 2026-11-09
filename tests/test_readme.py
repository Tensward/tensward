"""The commands in README.md and AGENT_SETUP.md parse with the real parser."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from tensward.cli import _bare_serve, build_parser

ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / "README.md").is_file() and (p / "AGENT_SETUP.md").is_file()
)
BASH_BLOCK = re.compile(r"```bash\n(.*?)```", re.DOTALL)


@pytest.mark.parametrize("name", ["README.md", "AGENT_SETUP.md"])
def test_every_readme_command_parses(name: str) -> None:
    text = (ROOT / name).read_text("utf-8")
    lines = [
        line.strip()
        for block in BASH_BLOCK.findall(text)
        for line in block.replace("\\\n", " ").splitlines()
        if line.strip().startswith("tensward ") and not line.split()[1].startswith("--")
    ]
    if name == "README.md":
        assert lines
    for line in lines:
        words = shlex.split(line, comments=True)
        build_parser().parse_args(_bare_serve(words[1:]))
