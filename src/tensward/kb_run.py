"""The knowledge files of a run: ``facts.json`` and the local training row of a measured step."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from . import __version__, kb_local
from .combine import parent_offer
from .context import AnalysisContext
from .engines import Engine
from .facts import FACTS_VERSION, Facts
from .files import write_json
from .history import recorded_env
from .kb_entries import change_key
from .kb_schema import TRAIN_METRICS, train_row
from .levers import LEVERS
from .runs import RUN_RECORD, WINDOW_RECORD
from .scoring import RuleScorer
from .settings import Settings

FACTS_RECORD = "facts.json"


def write_facts(run_dir: Path, facts: Facts, *, replace: bool = True) -> None:
    """Write the run's ``facts.json``: ``{facts_version, digest, facts}``. With ``replace`` off,
    a file already there stays."""
    path = run_dir / FACTS_RECORD
    if replace or not path.exists():
        write_json(
            path,
            {"facts_version": FACTS_VERSION, "digest": facts.digest(), "facts": facts.to_json()},
        )


def _read(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _scalar(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def _delta(
    engine: Engine, before: Settings, after: Settings
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The levers and the engine flags that differ, each as ``[before, after]``."""
    settings = {}
    for name in LEVERS:
        try:
            old, new = engine.lever_value(before, name), engine.lever_value(after, name)
        except ValueError:
            continue
        if old != new:
            settings[name] = [_scalar(old), _scalar(new)]
    flags = {
        flag: [_private(before.extra_args.get(flag)), _private(value)]
        for flag, value in after.extra_args.items()
        if before.extra_args.get(flag) != value
    }
    return settings, flags


def _private(value: Any) -> Any:
    """``value``, or ``"<set>"`` where it names a file (a path separator or a leading ``~``), so
    a local row never holds a path."""
    if isinstance(value, str) and ("/" in value or "\\" in value or value.startswith("~")):
        return "<set>"
    return value


def _outcome(
    window: Mapping[str, Any], metrics: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    change, band, beyond, direction = {}, {}, {}, {}
    for delta in window.get("deltas") or ():
        name = delta["metric"]
        if name not in TRAIN_METRICS:
            continue
        change[name], band[name] = delta["change"], delta["band"]
        beyond[name] = delta["verdict"] != "within_noise"
        direction[name] = "up" if delta["change"] > 0 else "down" if delta["change"] < 0 else "flat"
    sent = (metrics.get("succeeded") or 0) + (metrics.get("failed") or 0)
    outcome = {
        "started": True,
        "change": change,
        "noise_band": band,
        "failed_share": (metrics.get("failed") or 0) / sent if sent else 0.0,
    }
    return outcome, {"beyond_noise": beyond, "direction": direction}


def append_step_row(
    project: Path, run_dir: Path, parent: str, engine: Engine, context: AnalysisContext
) -> None:
    """Append the ``tw-train/1`` row of the step ``run_dir`` measured from ``parent`` to the
    local knowledge: the parent's facts, the change, what it did on the requests both sent, and
    the knowledge records that offered it. It holds no prompt, answer or path. Nothing is
    written when the parent has no ``facts.json`` or the step changed nothing."""
    parent_dir = project / "runs" / parent
    try:
        features = Facts.from_json(_read(parent_dir / FACTS_RECORD)["facts"])
        child, before = _read(run_dir / RUN_RECORD), _read(parent_dir / RUN_RECORD)
        window, metrics = _read(run_dir / WINDOW_RECORD), _read(run_dir / "metrics.json")
        was, now = Settings.from_json(before["settings"]), Settings.from_json(child["settings"])
    except (OSError, ValueError, KeyError, TypeError):
        return
    if not metrics.get("succeeded"):
        return
    env_before, env_now = recorded_env(before), recorded_env(child)
    settings, flags = _delta(engine, was, now)
    keys = [part for part in change_key(engine, was, now).split("+") if part]
    keys += sorted(env_now - env_before)
    if not keys:
        return
    change = (frozenset(child["engine_args"]) | env_now) - (
        frozenset(before["engine_args"]) | env_before
    )
    offer = parent_offer(parent_dir, change) or {}
    found = offer.get("kb") or {}
    action: dict[str, Any] = {
        "lever_keys": keys,
        "settings_delta": settings,
        "flags_delta": flags,
        "stack_size": len(offer.get("members") or ()) or 1,
    }
    if record := found.get("record") or (found.get("null") or {}).get("record"):
        action["records"] = [record]
    if offer.get("tier"):
        action["tier"] = offer["tier"]
    if found.get("score"):
        action |= {"score": found["score"], "scorer": RuleScorer.name}
    outcome, labels = _outcome(window, metrics)
    source = context.kb_matches.source if context.kb_matches is not None else {}
    provenance: dict[str, Any] = {
        "runs": {"parent": parent, "child": run_dir.name},
        "tensward": __version__,
    }
    if source.get("pack"):
        provenance |= {"pack": source["pack"], "content_digest": source["content_digest"]}
    try:
        kb_local.append_row(
            train_row(
                row_id=run_dir.name,
                source="local",
                features=features,
                action=action,
                outcome=outcome,
                labels=labels,
                provenance=provenance,
            )  # fmt: skip
        )
    except (OSError, ValueError, TypeError):
        return
