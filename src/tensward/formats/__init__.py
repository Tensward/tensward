"""Checkpoint formats: one module per on-disk layout behind :class:`CheckpointFormat`."""

from __future__ import annotations

import os
from pathlib import Path

from ..errors import CHECKPOINT_INVENTORY_UNEXPECTED, PreflightError
from .hf_safetensors import HF_SAFETENSORS
from .protocol import CheckpointFormat

__all__ = ["FORMATS", "GGUF_NOT_YET", "CheckpointFormat", "detect_format"]

FORMATS: tuple[CheckpointFormat, ...] = (HF_SAFETENSORS,)
"""In detection order."""

GGUF_NOT_YET = (
    "GGUF checkpoints are supported with the llama.cpp engine, coming in a later Tensward "
    "release; use a safetensors checkpoint with vLLM for now"
)


def detect_format(root: Path) -> CheckpointFormat:
    """The format of the checkpoint directory ``root``. A directory holding GGUF files is
    refused; any other that no format recognises gets the first format, whose registration
    refuses it with the reason."""
    if _holds_gguf(root):
        raise PreflightError(CHECKPOINT_INVENTORY_UNEXPECTED, GGUF_NOT_YET)
    return next((fmt for fmt in FORMATS if fmt.detect(root)), FORMATS[0])


def _holds_gguf(root: Path) -> bool:
    try:
        with os.scandir(root) as entries:
            return any(entry.name.endswith(".gguf") for entry in entries)
    except OSError:
        return False
