"""Prompt token ids kept per project, so a later step does not tokenize every prompt again."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx

from .engines import Engine
from .files import write_private
from .project import ResolvedProject
from .runtime import Runtime

CACHE_FILE = "token_ids.json"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
RENDERING_ARGS = ("chat-template", "tokenizer-mode", "tokenizer", "trust-remote-code")


def _file_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _rendering_args(engine_args: Sequence[str]) -> list[str]:
    found = []
    for argument in engine_args:
        name, _, value = argument.lstrip("-").partition("=")
        if name in RENDERING_ARGS:
            found.append(f"{name}={value}")
    return sorted(found)


class CachedTokens:
    """``engine`` with its prompt tokenization served from the project's cache on a key match;
    every other attribute is the engine's own."""

    def __init__(
        self,
        engine: Engine,
        runtime: Runtime,
        project_dir: Path,
        project: ResolvedProject,
        engine_args: Sequence[str],
    ) -> None:
        self._engine = engine
        self._path = project_dir / "cache" / CACHE_FILE
        root = project.artifact.path
        self._prefix = json.dumps(
            [
                project.artifact.metadata.fingerprint.value,
                [_file_digest(root / name) for name in TOKENIZER_FILES],
                engine.availability(runtime.kind, runtime.target).version,
                _rendering_args(engine_args),
            ]
        )
        self._entries: dict[str, list[int]] | None = None
        self._added = False
        if hasattr(engine, "prompt_token_ids"):
            self.prompt_token_ids = self._token_ids

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    def _known(self) -> dict[str, list[int]]:
        if self._entries is None:
            try:
                loaded = json.loads(self._path.read_text(encoding="utf-8"))
                self._entries = {k: list(v) for k, v in loaded.items()}
            except (OSError, ValueError, AttributeError, TypeError):
                self._entries = {}
        return self._entries

    async def _token_ids(
        self, client: httpx.AsyncClient, server_url: str, model: str, body: Mapping[str, Any]
    ) -> tuple[int, ...] | None:
        key = hashlib.sha256(
            f"{self._prefix}{json.dumps(body, sort_keys=True)}".encode()
        ).hexdigest()
        entries = self._known()
        if key in entries:
            return tuple(entries[key])
        tokens: tuple[int, ...] | None = await getattr(self._engine, "prompt_token_ids")(  # noqa: B009
            client, server_url, model, body
        )
        if tokens is not None:
            entries[key] = list(tokens)
            self._added = True
        return tokens

    async def count_prompt_tokens(
        self, client: httpx.AsyncClient, server_url: str, model: str, body: Mapping[str, Any]
    ) -> int | None:
        if not hasattr(self._engine, "prompt_token_ids"):
            return await self._engine.count_prompt_tokens(client, server_url, model, body)
        tokens = await self._token_ids(client, server_url, model, body)
        return None if tokens is None else len(tokens)

    def save(self) -> None:
        """Write what this run added."""
        if not self._added or self._entries is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        write_private(self._path, json.dumps(self._entries))
        self._added = False


def cached_tokens(
    engine: Engine,
    runtime: Runtime,
    project_dir: Path,
    project: ResolvedProject,
    engine_args: Sequence[str],
) -> CachedTokens:
    """``engine`` with ``prompt_token_ids`` and ``count_prompt_tokens`` served from the
    project's cache when the key matches; the returned object has ``save()``."""
    return CachedTokens(engine, runtime, project_dir, project, engine_args)
