"""The snapshots of a hybrid model in a report, and in the screen of a step that changes them."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from typing import Any

from ..engines.llamacpp.snapshots import Snapshots
from .model import Section, Text, item

EXPLANATION = (
    "This model is a hybrid: part of it keeps a running summary of the conversation instead of "
    "every word. The server can pick a conversation up again only from a saved copy of that "
    "summary (a snapshot). If no snapshot sits where the shared start (system prompt and tools) "
    "ends, those conversations may read the start again on every turn."
)


def lines(before: Snapshots, after: Snapshots | None = None) -> list[str]:
    """The explanation and the numbers of a run whose slots lost the shared start, and when
    ``after`` is the same load with another snapshot spacing, how many slots kept it there."""
    served = before.slots_served
    if before.shared_start_tokens is None or not before.lost:
        return []
    found = [
        "\n".join(textwrap.wrap(EXPLANATION, 80)),
        f"Shared start: your {served} conversations begin with the same "
        f"{before.shared_start_tokens} tokens (system prompt and tools).",
        f"Snapshots: {before.lost} of {served} conversations had none at the end of it and may "
        "read it again each turn.",
    ]
    if (
        after is not None
        and after.lost is not None
        and after.min_spacing is not None
        and after.min_spacing != before.min_spacing
    ):
        found.append(
            f"With --checkpoint-min-step {after.min_spacing}: "
            f"{after.slots_served - after.lost} of {after.slots_served} kept it."
        )
    return found


def section(facts: Mapping[str, Any] | None) -> Section | None:
    """The report section of a run whose log shows slots without a snapshot at the shared start."""
    if facts is None:
        return None
    start, lost = facts["run.shared_start_tokens"], facts["run.snapshot_lost"]
    slots = facts["run.slots"]
    if not isinstance(start, int) or not isinstance(lost, int) or not lost:
        return None
    if not isinstance(slots, int):
        return None
    shown = lines(Snapshots(start, lost, slots, None))
    return Section(
        id="snapshots",
        title="Snapshots",
        blocks=(
            Text(key="snapshots_explanation", text=EXPLANATION),
            item("snapshots_shared_start", shown[1]),
            item("snapshots_lost", shown[2]),
        ),
    )
