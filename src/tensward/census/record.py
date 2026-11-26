"""The stored result of a census: ``census/census.json`` in the run directory."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..files import write_json
from .inefficiency import KINDS, LABELS, Inefficiency

Status = Literal["ok", "skipped", "no_output", "not_attached", "failed"]
STATUSES: tuple[Status, ...] = ("ok", "skipped", "no_output", "not_attached", "failed")
RECORD_NAME = "census.json"
RECORD_DIR = "census"


@dataclass(frozen=True, slots=True)
class CensusRecord:
    status: Status
    reason: str
    readings: Mapping[str, float | int | bool] = field(default_factory=dict)
    proof: Mapping[str, Any] = field(default_factory=dict)
    library: Mapping[str, Any] | None = None
    requests: int = 0
    span_s: float = 0.0
    wall_s: float = 0.0
    advice: str | None = None
    refresh_of: str | None = None
    inefficiency: tuple[Inefficiency, ...] | None = None
    out_tokens: int = 0


def inefficiency_json(record: CensusRecord) -> list[dict[str, Any]] | None:
    if record.inefficiency is None:
        return None
    return [asdict(found) | {"detail": dict(found.detail)} for found in record.inefficiency]


def census_block(found: list[dict[str, Any]], census_run: str | None) -> dict[str, Any]:
    """metrics.json's ``census``: what the settings cannot fix, and the run it was recorded in
    when not the run itself."""
    return {"inefficiency": found, **({"run": census_run} if census_run else {})}


def _inefficiency(raw: Any) -> tuple[Inefficiency, ...] | None:
    if not isinstance(raw, list):
        return None
    try:
        return tuple(
            Inefficiency(
                kind=row["kind"],
                share=float(row["share"]),
                floor=float(row["floor"]),
                detail={str(k): float(v) for k, v in (row.get("detail") or {}).items()},
                label=row.get("label") if row.get("label") in LABELS else None,
            )
            for row in raw
            if row.get("kind") in KINDS
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def record_path(run_dir: Path) -> Path:
    return Path(run_dir) / RECORD_DIR / RECORD_NAME


def write_record(run_dir: Path, record: CensusRecord) -> Path:
    path = record_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "status": record.status,
        "reason": record.reason,
        "readings": dict(record.readings),
        "proof": dict(record.proof),
        "library": dict(record.library) if record.library is not None else None,
        "requests": record.requests,
        "span_s": record.span_s,
        "wall_s": record.wall_s,
        "advice": record.advice,
        "refresh_of": record.refresh_of,
        "inefficiency": inefficiency_json(record),
        "out_tokens": record.out_tokens,
    }
    write_json(path, data)
    return path


def read_record(run_dir: Path) -> CensusRecord | None:
    """The stored record, or None when there is none or it is not one this version wrote."""
    try:
        data = json.loads(record_path(run_dir).read_text())
        status = data["status"]
        readings = data.get("readings") or {}
        if status not in STATUSES or not isinstance(readings, dict):
            return None
        return CensusRecord(
            status=status,
            reason=str(data.get("reason", "")),
            readings=readings,
            proof=data.get("proof") or {},
            library=data.get("library"),
            requests=int(data.get("requests", 0)),
            span_s=float(data.get("span_s", 0.0)),
            wall_s=float(data.get("wall_s", 0.0)),
            advice=data.get("advice"),
            refresh_of=data.get("refresh_of"),
            inefficiency=_inefficiency(data.get("inefficiency")),
            out_tokens=int(data.get("out_tokens", 0)),
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
