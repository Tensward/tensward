"""``init --like``: the new project's first run against the baseline of the project it was made
like, on the goal's metric."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import PreflightError
from .presets import read_choices
from .text import columns

FIELDS = {
    "throughput": ("request_throughput", "requests per second", ".2f"),
    "ttft": ("ttft_p95_ms", "TTFT p95 (ms)", ".0f"),
    "tpot": ("tpot_p95_ms", "TPOT p95 (ms)", ".1f"),
}
ENGINES = {"vllm": "vLLM", "llamacpp": "llama.cpp"}


def _read(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text("utf-8"))
    if not isinstance(document, dict):
        raise ValueError(path.name)
    return document


def _hardware(run: dict[str, Any]) -> tuple[str, ...]:
    gpus = sorted(str(gpu.get("name")) for gpu in run.get("gpus_identity") or ())
    return tuple(gpus) if gpus else (str((run.get("host") or {}).get("cpu_model")),)


def _label(project: dict[str, Any], run: dict[str, Any]) -> str:
    engine = ENGINES.get(str(run.get("engine")), str(run.get("engine")))
    files = project["artifact"]["files"]
    model = files.get("gguf") if isinstance(files, dict) else None
    return f"{engine}, {model or Path(project['artifact']['root']).name}"


def _baseline(project: Path) -> Path | None:
    """The project's first run, when it served requests."""
    runs = sorted(p for p in (project / "runs").iterdir() if p.is_dir())
    if not runs:
        return None
    try:
        return runs[0] if _read(runs[0] / "metrics.json").get("succeeded") else None
    except (OSError, ValueError):
        return None


def compared_with_previous(project_dir: Path, run_dir: Path) -> list[str]:
    """The lines to print after the first analysis of a project made with ``--like``: the goal
    metric of the old project's baseline and of this run, when both ran on the same hardware
    and the same workload. Nothing otherwise."""
    try:
        choices = read_choices(project_dir)
        if choices is None or choices.like is None or choices.goal not in FIELDS:
            return []
        old = Path(choices.like)
        old = old if (old / "project.json").is_file() else old / ".tensward"
        if (mine := _baseline(project_dir)) is None or mine.name != run_dir.name:
            return []
        if (base := _baseline(old)) is None:
            return []
        here, there = _read(project_dir / "project.json"), _read(old / "project.json")
        if here["workload_digest"] != there["workload_digest"]:
            return []
        run, before = _read(run_dir / "run.json"), _read(base / "run.json")
        if _hardware(run) != _hardware(before):
            return []
        field, name, spec = FIELDS[choices.goal]
        now, then = _read(run_dir / "metrics.json")[field], _read(base / "metrics.json")[field]
        if not now or not then:
            return []
        label = old.parent.name if old.name == ".tensward" else old.name
        moved = f"{round((now / then - 1) * 100):+d}%".replace("-", "−")
        table = columns(
            ("", "previous", "this setup", "change"),
            [(name, format(then, spec), format(now, spec), moved)],
        )
        previous = f"previous: {_label(there, before)}"
        this = f"this setup: {_label(here, run)}"
    except (PreflightError, OSError, ValueError, KeyError, TypeError, AttributeError):
        return []
    return [f"Compared with your previous setup ({label})", *table, previous, this]
