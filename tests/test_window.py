"""The window plan and the fallbacks of the comparison with a step's parent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensward import window

L4_RUN = Path(__file__).parent / "replays" / "l4-steps" / "20261014T164058Z-8255"


@pytest.mark.parametrize(
    ("rate", "clients", "configured", "predicted", "only", "requests", "seconds"),
    [
        (None, 16, 1032, None, None, 1032, None),  # no served run to take the rate from
        (3.37, 16, 200, None, None, 200, None),  # configured below the 60 s count
        (1.0, 64, 1032, None, None, 256, 120.0),  # 4 per client
        (3.37, 0, 1032, None, None, 203, 60.0),  # open loop: no wave
        (1.0, 16, 1032, None, 90.0, 122, 90.0),  # --window 90 although the rule wants longer
        (1.0, 16, 1032, None, None, 152, 120.0),  # a slow cell gets a longer window
        (0.2, 0, 1032, None, None, 120, 300.0),  # at least 120 requests
        (1.0, 16, 1032, 0.01, None, 152, 120.0),  # a 1% prediction buys no longer window
    ],
)
def test_the_plan(
    rate: float | None,
    clients: int,
    configured: int,
    predicted: float | None,
    only: float | None,
    requests: int,
    seconds: float | None,
) -> None:
    plan = window.plan(
        rate,
        clients=clients,
        configured=configured,
        metric="tpot_p95_ms",
        predicted=predicted,
        only=only,
    )
    assert (plan.requests, plan.seconds) == (requests, seconds) and plan.reason


def test_a_parent_with_failures_is_not_matched_and_the_fallback_says_so(tmp_path: Path) -> None:
    rows = [json.loads(line) for line in (L4_RUN / "requests.jsonl").read_text().splitlines()]
    for row in rows[::40]:
        row["outcome"] = "error"
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert window.compare_matched(L4_RUN, tmp_path, clients=16) == (
        "the parent had 2.5% failed requests"
    )
    step = window.figures(window.read_rows(L4_RUN) or [], clients=16, upto=236)
    parent = window.figures(window.read_rows(tmp_path) or [], clients=16)
    delta = window.compare_unmatched(step, parent, "tpot_p95_ms")
    assert delta is not None and not delta.matched
