"""Reading Prometheus text exposition."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

from .settings import EngineSignals

_NAME = re.compile(r"[A-Za-z_:][A-Za-z0-9_:]*")
_LABEL = re.compile(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*,?')
_ESCAPE = re.compile(r"\\(.)")
_VALUE = re.compile(r"\s+([-+]?(?:\d[\d.]*(?:[eE][-+]?\d+)?|Inf|NaN))")


def _unescape(match: re.Match[str]) -> str:
    return "\n" if match[1] == "n" else match[1]  # the exposition escapes \\, \" and \n


Sample = tuple[str, dict[str, str], float]  # name, labels, value


def samples(text: str) -> Iterator[Sample]:
    """(name, labels, value) of each sample line in a Prometheus exposition. Label values are
    quoted strings that may hold any character, including braces and escaped quotes; a line
    whose label set is not closed is skipped."""
    for line in text.splitlines():
        name = _NAME.match(line)
        if not name:
            continue
        position, labels = name.end(), {}
        if line.startswith("{", position):
            position += 1
            while not line.startswith("}", position):
                label = _LABEL.match(line, position)
                if not label:
                    break
                labels[label[1]] = _ESCAPE.sub(_unescape, label[2])
                position = label.end()
            if not line.startswith("}", position):
                continue
            position += 1
        value = _VALUE.match(line, position)
        if value:
            yield name[0], labels, float(value[1])


def values(parsed: Iterable[Sample]) -> dict[str, float]:
    """Each metric among ``parsed`` samples that has exactly one finite sample, by name."""
    found: dict[str, list[float]] = {}
    for name, _, value in parsed:
        if math.isfinite(value):
            found.setdefault(name, []).append(value)
    return {name: seen[0] for name, seen in found.items() if len(seen) == 1}


def labelled_value(parsed: Iterable[Sample], family: str, **labels: str) -> float | None:
    """The finite value of the one sample of ``family`` carrying these labels, else None."""
    found = [
        value
        for name, have, value in parsed
        if name == family
        and math.isfinite(value)
        and all(have.get(k) == v for k, v in labels.items())
    ]
    return found[0] if len(found) == 1 else None


def info_labels(parsed: Iterable[Sample], family: str) -> dict[str, str]:
    """The labels of an info gauge's one sample; empty if it is absent or there are several."""
    found = [labels for name, labels, _ in parsed if name == family]
    return found[0] if len(found) == 1 else {}


@dataclass(frozen=True, slots=True)
class Family:
    """Where an engine exports one signal: a metric family, optionally one labelled sample of
    it, or a label of an info gauge: its number, or whether it is set. A label that is absent
    or reads ``None`` (how a Python exporter writes a missing value) is unset."""

    name: str
    labels: Mapping[str, str] | None = None
    kind: Literal["value", "info_label", "info_flag"] = "value"
    label: str | None = None  # the info gauge's label that holds the number or the flag


SignalMap = Mapping[str, Family]  # an EngineSignals field name -> where the engine exports it


def read_signals(text: str, signals: SignalMap) -> EngineSignals:
    """One scrape read into engine-neutral signals; a field the map leaves out keeps its default."""
    parsed: Sequence[Sample] = list(samples(text))
    found = values(parsed)
    read: dict[str, Any] = {}
    for field, family in signals.items():
        if family.kind == "info_flag":
            read[field] = info_labels(parsed, family.name).get(family.label or "", "None") != "None"
        elif family.kind == "info_label":
            try:
                read[field] = float(info_labels(parsed, family.name)[family.label or ""])
            except (KeyError, ValueError):
                read[field] = None  # absent, or "None"
        elif family.labels:
            read[field] = labelled_value(parsed, family.name, **family.labels)
        else:
            read[field] = found.get(family.name)
    return EngineSignals(**read)
