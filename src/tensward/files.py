"""Small shared helpers: run ids, JSON output, and strict JSON input."""

from __future__ import annotations

import json
import math
import os
import secrets
import time
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

from pydantic import BaseModel, ValidationError

from .errors import PreflightError, validation_summary

Model = TypeVar("Model", bound=BaseModel)


def new_run_id() -> str:
    """A sortable UTC timestamp plus a short random suffix, unique per run."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + secrets.token_hex(2)


def write_json(path: Path, payload: Any) -> None:
    """Write JSON to a private file, replacing any previous one atomically."""
    write_private(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_private(path: Path, text: str) -> None:
    """Write ``text`` to a file only its owner can read, replacing any previous one atomically."""
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), "utf-8")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError("duplicate key")
    return dict(pairs)


def _finite(text: str) -> float:
    value = float(text)  # 1e400 parses to infinity, which parse_constant never sees
    if not math.isfinite(value):
        raise ValueError("non-finite number")
    return value


def _reject_constant(text: str) -> Any:
    raise ValueError("non-finite number")


def load_strict_json(data: bytes, what: str, code: str) -> Any:
    """Decode UTF-8 JSON, refusing duplicate keys, NaN and infinity, which are ambiguous in a
    document that is hashed. ``what`` names the document in the refusal."""
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_constant,
            parse_float=_finite,
        )
    except (ValueError, RecursionError):
        raise PreflightError(code, f"{what} is not valid strict JSON") from None


def parse_document(model: type[Model], data: bytes, what: str, code: str) -> Model:
    """Validate a JSON document against ``model``. The strict pass comes first because pydantic
    alone accepts duplicate keys and NaN."""
    load_strict_json(data, what, code)
    try:
        return model.model_validate_json(data)
    except ValidationError as error:
        raise PreflightError(code, f"{what} is invalid: {validation_summary(error)}") from None
