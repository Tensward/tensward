"""What the report says of the knowledge records behind an offer: the ``kb`` blocks of
``report.json``, the "Why" line of ``report.md`` and the lines of ``tensward report --kb``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

NO_PACK = "no knowledge pack installed"
METRIC_WORDS = {"throughput": "throughput", "ttft": "TTFT", "tpot": "TPOT"}
RANGES = ("<", "<=", ">", ">=")


def _shown(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ", ".join(_shown(item) for item in value)
    return f"{value:g}" if isinstance(value, float) else str(value)


def _term(term: Mapping[str, Any]) -> str:
    fact, op = term["fact"], term["op"]
    if op in ("known", "unknown"):
        return f"{fact} is {op}"
    seen = _shown(term["observed"])
    if op == "==":
        return f"{fact} = {seen}"
    if op in RANGES:
        return f"{fact} = {seen} {op} {_shown(term['value'])}"
    return f"{fact} = {seen} ({op} {_shown(term['value'])})"


def _percent(value: float) -> str:
    return f"{value * 100:+.0f}".replace("-", "\N{MINUS SIGN}")


def _moved(low: float, high: float) -> str:
    first, second = sorted((low, high), key=abs)
    if _percent(first) == _percent(second):
        return f"{_percent(first)}%"
    return f"{_percent(first)} to {_percent(second)}%"


def _measured(measured: Mapping[str, Any]) -> str:
    setups = measured["setups"]
    moves = ", ".join(
        f"{METRIC_WORDS.get(name, name)} {_moved(*found)}"
        for name, found in measured.items()
        if name not in ("setups", "pairs")
    )
    said = f"Measured on {setups} setup{'' if setups == 1 else 's'}"
    return f"{said}: {moves}." if moves else f"{said}."


def _name(record: str) -> str:
    return record.rpartition("/")[2].partition("@")[0]


def _origin(block: Mapping[str, Any]) -> str:
    if block.get("layer") == "local":
        return "your own record"
    return f"pack {block['pack']}" if block.get("pack") else "knowledge pack"


def why_line(kb: Mapping[str, Any]) -> str:
    """The one line under an offer that records explain: the record, where it came from, the
    terms that fired it and what was measured; for an offer a measured null demoted, the null."""
    null = kb.get("null")
    if null is not None:
        origin = _origin({"pack": kb.get("pack")})
        return (
            f"Why: KB record {_name(null['record'])} ({origin}) measured no gain from this "
            f"change: {null['reason']}."
        )
    fired = " and ".join(_term(term) for term in kb["fired_on"])
    parts = [f"Why: KB record {_name(kb['record'])} ({_origin(kb)}) fired because {fired}."]
    if kb.get("measured"):
        parts.append(_measured(kb["measured"]))
    if kb.get("direction_note"):
        parts.append(f"{kb['direction_note']}.")
    return " ".join(parts)


def source_summary(
    source: Mapping[str, Any], unknown: Sequence[tuple[str, Sequence[str]]]
) -> dict[str, Any]:
    """The top-level ``kb`` key of ``report.json``: ``{"pack": null}`` without a pack; with
    one, its version, digests, the records skipped and why. Records the run's facts could not
    decide, and a pack that failed to load, are named when there are any."""
    summary: dict[str, Any] = {"pack": source.get("pack")}
    if summary["pack"] is not None:
        summary |= {
            "content_digest": source.get("content_digest"),
            "facts_digest": source.get("facts_digest"),
        }
    if source.get("reason"):
        summary["reason"] = source["reason"]
    if source.get("skipped"):
        summary["skipped"] = dict(source["skipped"])
    if unknown:
        summary["unknown"] = [{"record": ref, "facts": list(facts)} for ref, facts in unknown]
    return summary


def kb_lines(document: Mapping[str, Any]) -> list[str]:
    """What ``tensward report --kb`` prints from a ``report.json``: the knowledge source, then
    every offer with the records that explain it (or that it comes from the playbook alone),
    and the records the run's facts could not decide."""
    kb = document.get("kb") or {"pack": None}
    if kb.get("pack") is None:
        why = f" ({kb['reason']})" if kb.get("reason") else ""
        lines = [f"{NO_PACK}{why}"]
    else:
        lines = [f"knowledge pack {kb['pack']} (content {kb.get('content_digest')})"]
    if kb.get("facts_digest"):
        lines.append(f"facts: {kb['facts_digest']}")
    if kb.get("skipped"):
        counted = ", ".join(f"{count} {why}" for why, count in sorted(kb["skipped"].items()))
        lines.append(f"records skipped: {counted}")
    offers = [
        (section["id"], block)
        for section in document.get("sections", ())
        for block in section.get("blocks", ())
        if block.get("kind") == "suggestion"
    ]
    for section, block in offers:
        where = "worth measuring" if section == "worth_measuring" else block["tier"]
        lines.append("")
        lines.append(f"{block['name']} ({where})")
        found = block.get("kb")
        if found is None:
            lines.append("  from the playbook")
            continue
        lines.append(f"  {why_line(found)}")
        if null := found.get("null"):
            lines += [f"  evidence: {item}" for item in null.get("evidence", ())]
            continue
        if found["unknown"]:
            lines.append(f"  not known here: {', '.join(found['unknown'])}")
        lines += [f"  evidence: {item}" for item in found["evidence"]]
    if unknown := kb.get("unknown"):
        lines += ["", "records the run's facts could not decide:"]
        lines += [f"  {item['record']}: needs {', '.join(item['facts'])}" for item in unknown]
    return lines
