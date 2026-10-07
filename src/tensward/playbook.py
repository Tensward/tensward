"""Playbooks: the setting changes an engine offers, and the bottlenecks each one addresses.

An :class:`Entry` is one change. ``applies`` reads the :class:`Situation` and says why the change
is worth trying (None when the run does not call for it; a signal that was not measured never
triggers it), and ``apply`` makes it. Whether a bottleneck crossed its threshold comes from the
situation's diagnosis, never from the thresholds themselves. ``addresses`` names the bottleneck
classes of :mod:`tensward.classify` it works on; ``evidence`` grades the published and measured
support for it ("strong", "ours" for Tensward's own measurements, "moderate" or "weak");
``costs`` says what it may cost. A :class:`Gate` rules the entry out for this workload or setup,
with the reason the report shows. Entries speak in engine-neutral :class:`Settings`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, Sequence

from .classify import Bottleneck, Diagnosis, Finding
from .measurement import Measurement
from .settings import Settings

MAX_CONTEXT_LEN_ROUNDING = 256
MAX_RUNGS = 4  # steps a numeric setting is walked in, the evidence-based target included

Evidence = Literal["strong", "ours", "moderate", "weak"]
EVIDENCE_ORDER: tuple[Evidence, ...] = ("strong", "ours", "moderate", "weak")


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkloadFacts:
    """What the registered workload says about itself."""

    prompts: tuple[str, ...]
    output_tokens: int
    context_limit: int  # the model's max_position_embeddings, from the checkpoint's config.json
    quantization: str | None = None  # the checkpoint's declared quantization, if any
    offers_tools: bool = False  # whether any prompt offers tools
    media_encoders: bool = False  # the checkpoint takes images
    sends_images: bool = False
    load: str = "the declared workload"  # the arrival policy the configuration declares
    model_type: str | None = None  # config.json's model_type
    structured: bool = False  # some request is sent under a structured-output constraint
    schema_with_tools: bool = False  # some request offers tools under such a constraint
    max_inflight: int | None = None  # the most requests the arrival policy keeps in flight
    encoder_gib: float = 0.0  # weights of the media encoders, which a text-only server skips


@dataclass(frozen=True, slots=True, kw_only=True)
class Situation:
    """What an entry decides from: the run, the workload, the settings it ran with, and the
    diagnosis. Entries read crossings from the diagnosis, never from thresholds."""

    measurement: Measurement
    facts: WorkloadFacts
    settings: Settings
    diagnosis: Diagnosis

    def finding(self, bottleneck: Bottleneck) -> Finding:
        return next(f for f in self.diagnosis.findings if f.bottleneck == bottleneck)

    def fired(self, bottleneck: Bottleneck) -> bool:
        return self.finding(bottleneck).state in ("warning", "critical")


class HeldBack(str):
    """The reason an entry is not offered although its signal calls for it: it is listed
    under "Not applicable here"."""


@dataclass(frozen=True, slots=True)
class Gate:
    reason: str  # why the entry does not apply here, as the report says it
    blocks: Callable[[WorkloadFacts, Settings], bool]


@dataclass(frozen=True, slots=True, kw_only=True)
class Entry:
    name: str
    addresses: frozenset[Bottleneck]
    applies: Callable[[Situation], str | None]
    apply: Callable[[Situation], Settings]
    evidence: Evidence
    costs: str
    gates: tuple[Gate, ...] = ()
    quality_risk: bool = False
    # For extensions that search settings; plain `analyse` ignores these three. ``helps``: the
    # directions it pushes ("throughput", "goodput", "req_s", "ttft" or "tpot"; empty is a
    # general fix). ``steps_on``: the numeric setting ``apply`` moves, walked there by
    # :func:`rungs`. ``start``: the value to walk from when the setting is unset.
    helps: frozenset[str] = frozenset()
    steps_on: str | None = None
    start: Callable[[Situation], float | None] | None = None
    # Offered only when ``applicable`` is asked for ``near``: a reason where ``applies`` says no.
    low_bar: Callable[[Situation], str | None] | None = None
    near: bool = False  # the low bar is its own signal near its gate, or a caution: it ranks last
    low_bar_costs: str | None = None  # what the low-bar offer may cost, where it differs
    # The line saying why this entry's change is blocked, for ``blocked_changes``.
    blocked: Callable[[Situation], str | None] | None = None
    # A reason when this entry removes what blocks another entry's change; it is then offered
    # first, for the bottlenecks in ``unblocks_for`` as well as its own.
    unblocks: Callable[[Situation], str | None] | None = None
    unblocks_for: frozenset[Bottleneck] = frozenset()
    # Where the reason comes from when it reads no measured signal ("your settings").
    basis: str = ""


Tier = Literal["try_first", "could_help"]


@dataclass(frozen=True, slots=True, kw_only=True)
class Suggestion:
    """One entry offered for one run. ``applicable`` sets ``tier`` to "try_first" when the
    entry's own signal called for it or it unblocks another change, and "could_help" for a low
    bar; ``suggest.suggestions`` makes the final call against the diagnosis and adds
    ``command`` and ``settings``. ``addresses``: the bottlenecks it is offered for."""

    entry: Entry
    reason: str
    tier: Tier
    addresses: frozenset[Bottleneck] = frozenset()
    near: bool = False
    cost: str | None = None  # what it may cost, where it differs from the entry's ``costs``
    basis: str = ""
    command: str | None = None
    settings: Settings | None = None


def blocked_changes(entries: Sequence[Entry], situation: Situation) -> dict[str, str]:
    """For each bottleneck, why an entry addressing it is blocked here (``Entry.blocked``)."""
    lines: dict[str, str] = {}
    for entry in entries:
        if entry.blocked and (line := entry.blocked(situation)):
            lines.update({name: line for name in entry.addresses})
    return lines


def ranked(found: Sequence[Suggestion]) -> list[Suggestion]:
    """``found`` by evidence grade; within a grade, in playbook order (cheaper changes first).
    Entries offered for a near signal come after the others."""
    return sorted(
        found,
        key=lambda suggestion: (
            suggestion.tier == "could_help" and suggestion.near,
            EVIDENCE_ORDER.index(suggestion.entry.evidence),
        ),
    )


def round_up_context(tokens: int) -> int:
    return -(-tokens // MAX_CONTEXT_LEN_ROUNDING) * MAX_CONTEXT_LEN_ROUNDING


def fitting_max_context_len(signals: Measurement, facts: WorkloadFacts) -> int | None:
    """The smallest multiple of 256 that fits the longest prompt plus its output.

    Never above the model's own context limit; None when no prompt is too long or when even
    the model's limit cannot hold the longest prompt.
    """
    if not signals.too_long or signals.max_prompt_tokens is None:
        return None
    longest = signals.max_prompt_tokens + facts.output_tokens
    if longest > facts.context_limit:
        return None
    return min(round_up_context(longest), facts.context_limit)


def _ladder(start: float, target: float) -> list[float]:
    """Values from ``start`` to ``target``: doubling or halving each time, or, for a fraction,
    halving the distance. Ends on ``target``; at most MAX_RUNGS, spread evenly."""
    if isinstance(target, float):
        steps = [round((start + target) / 2, 3), target]
    else:
        steps, value = [], start
        while True:
            value = value * 2 if target > start else value // 2
            if value >= target if target > start else value <= target:
                break
            steps.append(value)
        steps.append(target)
    steps = list(dict.fromkeys(step for step in steps if step != start))
    last = len(steps) - 1
    picks = sorted({round(i * last / (MAX_RUNGS - 1)) for i in range(MAX_RUNGS)})
    return [steps[i] for i in picks] if last >= MAX_RUNGS else steps


def rungs(
    entry: Entry,
    situation: Situation,
    *,
    prepare: Callable[[Settings], Settings | None] = lambda proposed: proposed,
) -> list[Settings]:
    """The settings ``entry`` proposes, easiest step first and the full change last.
    (Serves extensions that search settings; plain ``analyse`` applies the full change.)

    ``prepare`` adjusts each rung, and returns None for one that cannot be tried. That rung is
    dropped, and so is one that ``prepare`` leaves equal to ``settings``.

    An entry that moves a numeric setting proposes the steps between its current value and the
    target (8 -> 16 -> 32 -> 64 for concurrency), so a jump that overshoots what one point
    can tolerate still leaves a smaller gain to adopt.
    """
    settings = situation.settings
    target = entry.apply(situation)
    knob = entry.steps_on
    if knob is None:
        return _prepared([target], settings, prepare)
    start = getattr(settings, knob)
    if start is None and entry.start is not None:
        start = entry.start(situation)
    if start is None:
        return _prepared([target], settings, prepare)
    steps: list[Settings] = []
    for value in _ladder(start, getattr(target, knob)):
        change: dict[str, Any] = {knob: value}
        steps.append(replace(target, **change))
    return _prepared(steps, settings, prepare)


def _prepared(
    steps: list[Settings], settings: Settings, prepare: Callable[[Settings], Settings | None]
) -> list[Settings]:
    prepared = (prepare(step) for step in steps)
    return [step for step in prepared if step is not None and step != settings]
