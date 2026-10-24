"""How offers rank: the expected gain on the goal times the chance the answers are adopted."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .combine import ANSWER_RISK, NUMERIC, PRIOR_SHARE, answer_risk, goal_gain
from .matcher import Score

if TYPE_CHECKING:
    from .context import AnalysisContext
    from .facts import Facts
    from .playbook import Suggestion


class RuleScorer:
    """A single change gains its prediction, else PRIOR_SHARE of its prior; a stack gains its
    joint prediction plus PRIOR_SHARE of the priors of the members the prediction does not
    cover. Each member that may change the answers multiplies the gain by the chance they are
    adopted (``combine.answer_risk``)."""

    name = "rules/1"

    def __init__(self, goal: str, *, tools: bool = False) -> None:
        self.goal, self.tools = goal, tools

    def score(
        self, offer: Suggestion, facts: Facts | None, context: AnalysisContext | None
    ) -> Score:
        members = offer.members or (offer.entry,)
        predicted = goal_gain(offer.predicted, self.goal)
        covered: set[str] = set()
        if predicted is not None:
            covered = {m.name for m in members if m.name in NUMERIC or not offer.members}
        gain = (predicted or 0.0) + PRIOR_SHARE * sum(
            m.prior_gain.get(self.goal, 0.0) for m in members if m.name not in covered
        )
        value, adopt = gain, 1.0
        for member in members:
            if (risk := answer_risk(member, offer, tools=self.tools)) is not None:
                value *= 1 - ANSWER_RISK[risk]
                adopt *= 1 - ANSWER_RISK[risk]
        return Score(value=value, parts={"gain": gain, "p_adopt": adopt}, scorer=self.name)
