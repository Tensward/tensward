"""Serving engines: the neutral interface and one module per engine."""

from __future__ import annotations

from .protocol import Engine
from .vllm import VLLM

ENGINES: dict[str, Engine] = {VLLM.name: VLLM}
