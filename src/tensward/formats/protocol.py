"""What Tensward needs from a checkpoint format: recognise a directory, and register it."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..anatomy import ModelAnatomy
from ..artifacts import ArtifactEntry


class CheckpointFormat(Protocol):
    """One on-disk checkpoint layout."""

    name: str
    label: str  # the format as people write it

    def detect(self, root: Path) -> bool:
        """Whether the directory ``root`` looks like this format. Never raises."""
        ...

    def register(
        self,
        root: Path,
        *,
        engine_build: str,
        cache: Path | None = None,
        refresh: bool = False,
        trust_remote_code: bool = False,
    ) -> tuple[ArtifactEntry, ModelAnatomy]:
        """The record of the checkpoint at ``root`` and its anatomy (derived, not part of its
        identity), or a refusal saying why not. ``cache`` and ``refresh`` are those of
        :func:`tensward.artifacts.fingerprint_files`. ``trust_remote_code`` says the engine will
        run the checkpoint's own code, so that code is part of the identity."""
        ...
