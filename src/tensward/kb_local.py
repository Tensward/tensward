"""The local knowledge layer: the user's own records and training rows, under the knowledge
home (``~/.tensward/kb``, or ``TENSWARD_KB_HOME``). Nothing here leaves the machine."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .extensions import DISABLE_ENV
from .facts import canonical_json
from .kb_schema import Record, record_from_json

KB_HOME_ENV = "TENSWARD_KB_HOME"


def kb_home() -> Path:
    """Where installed packs and the local layer live."""
    configured = os.environ.get(KB_HOME_ENV)
    return Path(configured).expanduser() if configured else Path.home() / ".tensward" / "kb"


def local_records() -> tuple[Record, ...]:
    """The records in ``local/records/*.json``, by file name; a file that is not a valid record
    is skipped with one line on stderr. Empty while plugins are disabled."""
    if os.environ.get(DISABLE_ENV):
        return ()
    found = []
    for path in sorted((kb_home() / "local" / "records").glob("*.json")):
        try:
            found.append(record_from_json(json.loads(path.read_bytes()), layer="local"))
        except (OSError, ValueError) as error:
            print(f"local knowledge record {path.name} skipped: {error}", file=sys.stderr)
    return tuple(found)


def append_row(row: Mapping[str, Any]) -> None:
    """Appends ``row`` to ``local/rows.jsonl`` as one canonical JSON line."""
    path = kb_home() / "local" / "rows.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "ab") as out:
        out.write(canonical_json(row) + b"\n")
