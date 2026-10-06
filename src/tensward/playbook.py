"""Playbooks: the setting changes an engine offers, and the bottlenecks each one addresses.

An :class:`Entry` is one change. ``applies`` reads the measurement and says why the change is
worth trying (None when the signals do not call for it; a signal that was not measured never
triggers it), and ``apply`` makes it. ``addresses`` names the bottleneck classes of
:mod:`tensward.classify` it works on; ``evidence`` grades the published and measured support for
it ("strong", "ours" for Tensward's own measurements, "moderate" or "weak"); ``costs`` says what
it may cost. A :class:`Gate` rules the entry out for this workload or setup, with the reason the
report shows. Entries speak in engine-neutral :class:`Settings`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

from .engines.protocol import Settings
from .extensions import load_extender

if TYPE_CHECKING:
    from .measurement import Measurement
    from .project import ResolvedProject

MAX_CONTEXT_LEN_ROUNDING = 256
MAX_RUNGS = 4  # steps a numeric setting is walked in, the evidence-based target included
NEAR_THRESHOLD = 0.7  # a signal this share of its gate or more is near it

Evidence = Literal["strong", "ours", "moderate", "weak"]
EVIDENCE_ORDER: tuple[Evidence, ...] = ("strong", "ours", "moderate", "weak")


@dataclass(frozen=True, slots=True)
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
    structured: bool = False  # the workload declares structured output
    max_inflight: int | None = None  # the most requests the arrival policy keeps in flight
    encoder_gib: float = 0.0  # weights of the media encoders, which a text-only server skips

    @classmethod
    def of(cls, project: ResolvedProject) -> WorkloadFacts:
        metadata = project.artifact.metadata
        arrival = project.settings.workload.arrival
        load = f"{arrival.concurrency} concurrent clients"
        if arrival.kind != "closed_loop":
            load = f"{arrival.rate_rps:g} requests/s" + (
                f", at most {arrival.max_inflight} in flight" if arrival.kind == "capped" else ""
            )
        return cls(
            prompts=tuple(entry.text for entry in project.prompts),
            output_tokens=project.settings.workload.output_tokens,
            context_limit=metadata.context_limit,
            quantization=metadata.variants[0].label,
            offers_tools=any(entry.tools for entry in project.prompts),
            media_encoders="image" in project.anatomy.modalities,
            sends_images=any(entry.image_urls for entry in project.prompts),
            load=load,
            model_type=project.anatomy.model_type,
            structured=project.settings.workload.structured_output is not None,
            max_inflight=arrival.concurrency or arrival.max_inflight,
            encoder_gib=(project.anatomy.components.vision / 2**30)
            if project.anatomy.components
            else 0.0,
        )


class LowBar(str):
    """The reason of an entry offered below the bar of its own signal. ``near``: the signal is
    close to its gate but under it. ``cost``: the entry's cost where it differs from the
    entry's own."""

    near: bool
    cost: str | None

    def __new__(cls, text: str, *, near: bool = False, cost: str | None = None) -> LowBar:
        reason = super().__new__(cls, text)
        reason.near = near
        reason.cost = cost
        return reason

    def extended(self, note: str) -> LowBar:
        return LowBar(f"{self}; {note}", near=self.near, cost=self.cost)


class HeldBack(str):
    """The reason an entry is not offered although its signal calls for it: it is listed
    under "Not applicable here"."""


def is_near(value: float, gate: float) -> bool:
    """Whether ``value`` is at least NEAR_THRESHOLD of ``gate`` and below it."""
    return NEAR_THRESHOLD * gate <= value < gate


@dataclass(frozen=True, slots=True)
class Gate:
    reason: str  # why the entry does not apply here, as the report says it
    blocks: Callable[[WorkloadFacts, Settings], bool]


@dataclass(frozen=True, slots=True)
class Entry:
    name: str
    addresses: frozenset[str]
    applies: Callable[[Measurement, WorkloadFacts, Settings], str | None]
    apply: Callable[[Measurement, WorkloadFacts, Settings], Settings]
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
    start: Callable[[Measurement, Settings], float | None] | None = None
    # Offered only when ``applicable`` is asked for ``near``: a reason where ``applies`` says no.
    low_bar: Callable[[Measurement, WorkloadFacts, Settings], LowBar | None] | None = None
    # The line saying why this entry's change is blocked, for ``blocked_changes``.
    blocked: Callable[[Measurement, Settings], str | None] | None = None
    # Where the reason comes from when it reads no measured signal ("your settings").
    basis: str = ""


def applicable(
    entries: Sequence[Entry],
    signals: Measurement,
    facts: WorkloadFacts,
    settings: Settings,
    *,
    allow_quality_changes: bool,
    near: bool = False,
) -> tuple[list[tuple[Entry, str]], list[tuple[Entry, str]]]:
    """The entries the measurement calls for, each with its reason, and those of them a gate
    rules out here, each with the gate's reason. An installed analysis plugin's entries join
    ``entries``. With ``near``, an entry that declares a ``low_bar`` is also offered, with a
    :class:`LowBar` reason, where its signal is near its gate or a related change is blocked."""
    found, gated = [], []
    extender = load_extender()
    for entry in (*entries, *(extender.entries if extender else ())):
        if entry.quality_risk and not allow_quality_changes:
            continue
        reason = entry.applies(signals, facts, settings)
        if isinstance(reason, HeldBack):
            gated.append((entry, str(reason)))
            continue
        if not reason and near and entry.low_bar:
            reason = entry.low_bar(signals, facts, settings)
        if not reason:
            continue
        blocked = next((gate.reason for gate in entry.gates if gate.blocks(facts, settings)), None)
        if blocked:
            gated.append((entry, blocked))
        else:
            found.append((entry, reason))
    return found, gated


def blocked_changes(
    entries: Sequence[Entry], signals: Measurement, settings: Settings
) -> dict[str, str]:
    """For each bottleneck, why an entry addressing it is blocked here (``Entry.blocked``)."""
    lines: dict[str, str] = {}
    for entry in entries:
        if entry.blocked and (line := entry.blocked(signals, settings)):
            lines.update({name: line for name in entry.addresses})
    return lines


def ranked(found: Sequence[tuple[Entry, str]]) -> list[tuple[Entry, str]]:
    """``found`` by evidence grade; within a grade, in playbook order (cheaper changes first).
    Entries offered for a near signal come after the others."""
    return sorted(
        found,
        key=lambda pair: (
            isinstance(pair[1], LowBar) and pair[1].near,
            EVIDENCE_ORDER.index(pair[0].evidence),
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
    signals: Measurement,
    facts: WorkloadFacts,
    settings: Settings,
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
    target = entry.apply(signals, facts, settings)
    knob = entry.steps_on
    if knob is None:
        return _prepared([target], settings, prepare)
    start = getattr(settings, knob)
    if start is None and entry.start is not None:
        start = entry.start(signals, settings)
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
