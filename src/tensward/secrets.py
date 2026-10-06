"""Blanking credentials in a command line before it is stored or shown."""

from __future__ import annotations

import re
from typing import Callable

ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
# A flag or variable whose name has one of these words (`--hf-token`, `HF_TOKEN`,
# `AWS_SECRET_ACCESS_KEY`) carries a secret. Whole words, so `--max-num-batched-tokens` is not one.
SECRET_WORDS = frozenset({"key", "apikey", "secret", "password", "passwd", "auth"})
_ENV_FLAGS = ("-e", "--env")


def is_secret(name: str) -> bool:
    """A name that carries a credential: ``--api-key``, ``HF_TOKEN``, ``--hf-token``. "token"
    counts only as the last word, because serving flags count tokens
    (``--long-prefill-token-threshold``)."""
    words = re.split(r"[-_]+", name.lower().strip("-"))
    return not SECRET_WORDS.isdisjoint(words) or words[-1] == "token"


def blank(assignment: str) -> str:
    """``NAME=value`` with the value blanked when NAME is a secret's name."""
    name, equals, _ = assignment.partition("=")
    return f"{name}=***" if equals and is_secret(name) else assignment


def without_secrets(
    tokens: list[str], *, canonical: Callable[[str], str], multi_value_flags: frozenset[str]
) -> list[str]:
    """The command with every secret value blanked: a secret-named flag's value (`--api-key`,
    `--hf-token`) and a secret-named variable's (`NAME=v`, `-e NAME=v`, `--env=NAME=v`)."""
    shown: list[str] = []
    hiding = 0  # how many following tokens are a secret flag's values
    env_next = False  # the next token is the value of a docker -e / --env
    for token in tokens:
        if hiding and not token.startswith("-"):
            shown.append("***")
            hiding -= 1
            continue
        hiding = 0
        name, equals, value = token.partition("=")
        if env_next:
            shown.append(blank(token))
        elif not token.startswith("-"):
            shown.append(blank(token) if ASSIGNMENT.match(token) else token)
        elif canonical(name) in _ENV_FLAGS:
            shown.append(f"{name}={blank(value)}" if equals else token)
        elif token.startswith("-e") and not token.startswith("--") and equals:  # -eNAME=value
            shown.append("-e" + blank(token[2:]))
        elif is_secret(name):
            shown.append(f"{name}=***" if equals else token)
            if not equals:
                hiding = 99 if canonical(name) in multi_value_flags else 1
        else:
            shown.append(token)
        env_next = token in _ENV_FLAGS
    return shown
