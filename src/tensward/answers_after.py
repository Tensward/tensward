"""The answers of a run before and after a change that can alter them."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .compare import HEALTH_MARGIN, Answer, Outcome, broken, compare_answers, share
from .context import current as analysis
from .errors import PreflightError
from .history import parent, past_runs
from .kept import answer_changes
from .runs import RunFile, RunMetrics, answers_in
from .settings import Settings

if TYPE_CHECKING:
    from .engines import Engine
    from .playbook import Situation


@dataclass(frozen=True, slots=True)
class AnswersAfter:
    before: str  # the run this one was made from
    changes: tuple[str, ...]  # the levers that differ and can change answers
    outcome: Outcome | None  # None: a side kept no answers, or shown earlier
    missing: tuple[str, ...] = ()  # the runs that kept no answers, when outcome is None
    shown_in: str | None = None  # the run whose report already showed these settings
    # per prompt: prompt_id, before, after, identical, changed, judged, similarity, broken
    pairs: tuple[Mapping[str, Any], ...] = ()
    # (broken answers, answers) before and after: empty or repeating themselves
    broken: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (0, 0))

    @property
    def broken_rose(self) -> bool:
        """Whether the share of broken answers rose by more than HEALTH_MARGIN."""
        (before, asked), (after, answered) = self.broken
        if not answered:
            return False
        return after / answered - (before / asked if asked else 0.0) > HEALTH_MARGIN

    def broken_words(self) -> str:
        """The broken count, a floor (an answer cut short before it repeats is not counted),
        and the answers' mean word overlap with the run they were made from."""
        (before, asked), (after, answered) = self.broken
        overlap = [p["similarity"] for p in self.pairs if p.get("similarity") is not None]
        shared = (
            f", and on average they share {fmean(overlap):.0%} of their words with that run's"
            if overlap
            else ""
        )
        return (
            f"at least {after} of {answered} answers repeat themselves or are empty, against "
            f"{before} of {asked} in run {self.before}{shared}"
        )


def before_after(project: Path | None, situation: Situation, engine: Engine) -> AnswersAfter | None:
    """This run's answers against those of the run it was made from, when the settings between
    them differ on a lever that can change answers; None when there is no such run."""
    context = analysis()
    run_dir = context.this_run()
    if project is None or run_dir is None:
        return None
    runs = past_runs(project, engine, before=context.reported_run)
    args = frozenset(context.engine_args)
    base = parent(args, runs)
    if base is None:
        return None
    changes = answer_changes(
        engine,
        Settings.from_json(base.settings),
        situation.settings,
        checkpoint_quantized=situation.facts.quantization is not None,
    )
    if not changes:
        return None
    earlier = next(
        (run for run in runs if run.served and run.engine_args == args
         and (project / "runs" / run.run_id / "quality.json").is_file()),
        None,
    )  # fmt: skip
    if earlier is not None:
        return AnswersAfter(base.run_id, changes, None, shown_in=earlier.run_id)
    before_dir = project / "runs" / base.run_id
    kept_none = tuple(
        name
        for name, kept in (
            (base.run_id, RunFile.read(before_dir).retained_responses),
            (run_dir.name, (run_dir / "responses.jsonl").is_file()),
        )
        if not kept
    )
    if kept_none:
        return AnswersAfter(base.run_id, changes, None, missing=kept_none)
    before, after = answers_in(before_dir), answers_in(run_dir)
    measured = situation.measurement
    outcome = compare_answers(
        before,
        after,
        {},
        json_answers=context.json_answers,
        failed=(
            _failed_share(before_dir),
            share(measured.failed, measured.succeeded + measured.failed),
        ),
    )
    pairs = tuple(
        {
            "prompt_id": prompt.prompt_id,
            "before": _said(before.get(prompt.prompt_id, ())),
            "after": _said(after.get(prompt.prompt_id, ())),
            "identical": prompt.identical,
            "changed": prompt.changed,
            "judged": prompt.noise is not None,
            "similarity": None if prompt.agreement is None else round(prompt.agreement, 3),
            "broken": {
                side: sum(map(broken, answers.get(prompt.prompt_id, ())))
                for side, answers in (("before", before), ("after", after))
            },
        }
        for prompt in outcome.prompts
    )
    counts = (_broken(before), _broken(after))
    return AnswersAfter(base.run_id, changes, outcome, pairs=pairs, broken=counts)


def _broken(side: Mapping[str, Sequence[Answer]]) -> tuple[int, int]:
    answers = [answer for given in side.values() for answer in given]
    return sum(map(broken, answers)), len(answers)


def _said(answers: Sequence[Answer]) -> list[str]:
    """Each answer's text, with its tool calls as ``name(arguments)`` lines after it."""
    return [
        "\n".join(
            [answer.text, *(f"{name}({args})" for name, args in answer.tool_calls or ())]
        ).strip()
        for answer in answers
    ]


def _failed_share(run_dir: Path) -> float | None:
    try:
        return RunMetrics.read(run_dir / "metrics.json").failed_share
    except PreflightError:
        return None


def counts(outcome: Outcome) -> dict[str, Any]:
    """How many prompts' answers are identical and how many differ, tool calls and text apart:
    a prompt with a tool call on either side counts under ``tool_calls`` by its calls (names and
    arguments), the others under ``text``; and the text's mean word overlap (1: the same
    words)."""
    answered = [p for p in outcome.prompts if p.identical is not None]
    calls = [p for p in answered if p.calls_identical is not None]
    text = [p for p in answered if p.calls_identical is None]
    overlap = [p.agreement for p in text if p.agreement is not None]
    return {
        "prompts": len(answered),
        "identical": sum(p.identical is True for p in answered),
        "differ": sum(p.identical is False for p in answered),
        "judged": sum(p.noise is not None for p in answered),
        "changed": sum(p.changed for p in answered),
        "tool_calls": {
            "prompts": len(calls),
            "identical": sum(p.calls_identical is True for p in calls),
        },
        "text": {
            "prompts": len(text),
            "differ": sum(p.identical is False for p in text),
            "similarity": round(fmean(overlap), 3) if overlap else None,
        },
        "unanswered": sum(p.identical is None for p in outcome.prompts),
    }


def quality_document(after: AnswersAfter, run_id: str) -> dict[str, Any]:
    """``quality.json``: every prompt with both runs' answers, and the counts. A pair's
    ``changed`` means more different than the earlier run's own repeats, so it is only
    meaningful where ``judged`` is true; ``identical`` compares the texts exactly."""
    (before, asked), (broke, answered) = after.broken
    summary = counts(after.outcome) if after.outcome is not None else {}
    return {
        "before": after.before,
        "after": run_id,
        "changes": list(after.changes),
        "summary": {
            **summary,
            "broken": {
                "before": [before, asked],
                "after": [broke, answered],
                "rose": after.broken_rose,
            },
        },
        "pairs": list(after.pairs),
    }


def broken_note(run_dir: Path) -> str | None:
    """From a run's ``quality.json``, a warning when its answers look broken against the run it
    was made from; None when they do not or there is no file."""
    try:
        document = json.loads((run_dir / "quality.json").read_text("utf-8"))
        found = document["summary"]["broken"]
        shown = AnswersAfter(
            document["before"],
            (),
            None,
            pairs=tuple(document.get("pairs") or ()),
            broken=(tuple(found["before"]), tuple(found["after"])),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return shown.broken_words() if shown.broken_rose else None


def broken_in_chain(project: Path, run: str) -> tuple[str, str] | None:
    """The nearest of ``run`` and the runs it was made from, one parent at a time, whose answers
    look broken against its own parent: (that run, the warning); None when none do."""
    from .project import load_project, project_engine

    engine = project_engine(load_project(project, verify_weights=False).record.current_setup)
    runs = past_runs(project, engine)
    by_id = {past.run_id: past for past in runs}
    name: str | None = run
    while name is not None:
        if (note := broken_note(project / "runs" / name)) is not None:
            return name, note
        here = by_id.get(name)
        older = [past for past in runs if past.run_id < name]
        base = parent(here.engine_args, older) if here is not None else None
        name = base.run_id if base is not None else None
    return None
