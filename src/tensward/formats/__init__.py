"""Checkpoint formats: one module per on-disk layout behind :class:`CheckpointFormat`."""

from __future__ import annotations

from pathlib import Path

from .gguf import GGUF
from .hf_safetensors import HF_SAFETENSORS
from .protocol import CheckpointFormat

__all__ = ["FORMATS", "CheckpointFormat", "detect_format"]

FORMATS: tuple[CheckpointFormat, ...] = (HF_SAFETENSORS, GGUF)
"""In detection order."""


def detect_format(root: Path) -> CheckpointFormat:
    """The format of ``root``, a checkpoint directory or a GGUF file. One that no format
    recognises gets the first format, whose registration refuses it with the reason."""
    return next((fmt for fmt in FORMATS if fmt.detect(root)), FORMATS[0])
