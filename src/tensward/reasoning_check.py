"""The reasoning check: when every answer of a step ran out of output tokens while thinking and
its reasoning differs from that of the run it was made from, Tensward replays the step's
reasoning through the current setup one token at a time and asks at sampled positions whether
the current setup would have written the same token. The current setup's own reasoning,
replayed the same way, is the yardstick: two greedy runs part after the first rounding flip,
but each token a sound change writes is still the one the current setup would write after the
same text, up to rounding. Only llama.cpp can be asked in 0.4.3.

The answer check does the same for finished answers that differ when no text rule passes the
step: each answer the step wrote that is none of the earlier run's is replayed from where it
parts from the closest of them, and every token from there must be one the current setup rates
within TIE of its own pick. The earlier run's copy, replayed the same way, guards the replay."""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal, Mapping, Sequence

import httpx

from .client import chat_body
from .compare import (
    Answer,
    Outcome,
    answers_pass,
    broken,
    compare_answers,
    counted_against,
    distinct,
    rounding,
    rounding_passes,
)
from .context import current as analysis
from .errors import AnalyseFailure
from .files import write_json
from .history import env_args, kept_run, past_runs, recorded_env, step_parent
from .kept import answer_changes
from .progress import PhaseStarted, emit
from .rejected import DEPARTS, LIKE_ROUNDING, REVIEW_MARK, WOULD_NOT_WRITE
from .runs import RunFile, answers_in, only_prompts, shared_prompts
from .settings import Settings
from .workload import ImagePart, WorkloadSpec, record_workload

if TYPE_CHECKING:
    from .engines import Engine
    from .kept import Kept
    from .project import ResolvedProject
    from .runtime import Runtime

FORMAT = 1  # bumped whenever a constant, a launch setting or the replay changes
PROMPTS = 12  # the first reasoned prompts that differ, in workload order
PROBES = 768  # per side, spread evenly over the prompts
MIN_PROBES = 256
MAX_TOKENS = 4096  # of each reasoning text
TOP = 20
FAR = math.log(10)  # a token this far below the top one is one the setup found unlikely
LEAD_PROBES = 4  # leading whitespace tokens the reasoning parser strips
PROMPT_PROBES = 64  # a prompt with this many probes is judged on its own too
SELF_AGREE, SELF_FAR = 0.97, 0.01
PASS_PROMPT, CHANGED_PROMPT = 0.90, 0.75
CHANGED_AGREE, CHANGED_FAR = 0.05, 0.02
Z = 2.33  # one-sided 99%
CHECK_FILE = "reasoning_check.json"
CHECK_DIR = "reasoning-check"
CACHE_FILE = "reasoning-check-aa.json"
# The check's only launch changes: no context checkpoints, so a probe is one forward and not a
# 4-token split plus two state copies on hybrid models; and probes of up to 127 tokens run the
# host-resident layers on the CPU, as decode does, instead of copying their weights to the GPU.
CHECK_FLAGS = {"--ctx-checkpoints": "0"}
CHECK_ENV = {"GGML_OP_OFFLOAD_MIN_BATCH": "128"}

ANSWER_FORMAT = 1
TIE = math.log(4)  # a token at least a quarter as likely as the current setup's own pick
ANSWER_TOKENS = 128  # checked per answer from where it parts
ANSWER_PROBES = 6144  # per step, both sides
# A step whose full replay needs more than ANSWER_OVER budgets by the answers' tokens cannot
# reach `same`; it only looks for a changed answer, within ANSWER_HUNT probes.
ANSWER_OVER = 2
ANSWER_HUNT = 1024
ANSWER_FILE = "answer_check.json"
TOO_MANY = "too many answers differ to check one by one"
REPLAY = "your current setup did not write its own answer again when replayed"
THOUGHT = "their reasoning differs, which this check does not cover yet"
SAME = "every answer is one the earlier run wrote too"

HOST_PLACED = ("host_layers", "host_weight_bytes")  # placement facts of layers in host memory
PROMPT_EVAL = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")

Verdict = Literal["same", "partial", "your_call", "changed", "not_run"]


class CheckFailed(Exception):
    """The check could not run to a result; ``str(error)`` is the reason in words."""


@dataclass(frozen=True, slots=True)
class Probe:
    position: int  # token index in the reasoning
    char: int  # where that token starts in the reasoning text
    token: int
    wrote: str  # the token's text
    logprob: float | None  # None: not in the current setup's top TOP
    top: int
    top_text: str
    top_logprob: float

    @property
    def agree(self) -> bool:
        return self.token == self.top

    @property
    def far(self) -> bool:
        return self.logprob is None or self.logprob < self.top_logprob - FAR


@dataclass(frozen=True, slots=True)
class Side:
    """One side's probes by prompt id."""

    probes: Mapping[str, Sequence[Probe]]

    @property
    def count(self) -> int:
        return sum(map(len, self.probes.values()))

    def share(self, prompt: str | None = None, *, far: bool = False) -> float:
        probes = self.probes[prompt] if prompt else [p for g in self.probes.values() for p in g]
        if not probes:
            return 0.0
        return sum(p.far if far else p.agree for p in probes) / len(probes)


def verdict(base: Side, step: Side, *, broken_answers: bool = False) -> tuple[Verdict, str | None]:
    """same / your_call / changed, or not_run with the reason, by the pre-registered rules."""
    if min(base.count, step.count) < MIN_PROBES:
        return "not_run", f"too little reasoning to check ({min(base.count, step.count)} tokens)"
    agree_b, far_b = base.share(), base.share(far=True)
    if agree_b < SELF_AGREE or far_b > SELF_FAR:
        return "not_run", (
            "your current setup did not reproduce its own reasoning when replayed "
            f"({agree_b:.0%} of tokens)"
        )
    agree_s, far_s = step.share(), step.share(far=True)
    margin = max(0.010, Z * math.sqrt(2 * agree_b * (1 - agree_b) / base.count))
    far_margin = max(0.005, Z * math.sqrt(2 * max(far_b, 1 / base.count) / base.count))
    judged = [step.share(p) for p, probes in step.probes.items() if len(probes) >= PROMPT_PROBES]
    if (
        broken_answers
        or agree_b - agree_s > CHANGED_AGREE
        or far_s - far_b > CHANGED_FAR
        or any(a < CHANGED_PROMPT for a in judged)
    ):
        return "changed", None
    if (
        agree_b - agree_s <= margin
        and far_s - far_b <= far_margin
        and all(a >= PASS_PROMPT for a in judged)
    ):
        return "same", None
    return "your_call", None


def positions(tokens: int, count: int) -> list[int]:
    """``count`` positions spread evenly over ``tokens``, or every one when there are fewer."""
    if tokens <= count:
        return list(range(tokens))
    return [math.floor((j + 0.5) * tokens / count) for j in range(count)]


def char_offsets(pieces: Sequence[bytes]) -> list[int]:
    """Where each token starts in the decoded text, in characters."""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    offsets, chars = [], 0
    for piece in pieces:
        offsets.append(chars)
        chars += len(decoder.decode(piece))
    return offsets


def _lead_ws(text: str) -> bool:
    return bool(text) and text.isspace()


@dataclass(frozen=True, slots=True)
class Prompt:
    """One prompt to check: its token ids up to the generation prompt, and the reasoning each
    side wrote."""

    prompt_id: str
    prefix: list[int]
    base: str
    step: str
    diverge: int | None = None  # where the two reasoning texts first differ, in characters


@dataclass(frozen=True, slots=True)
class _Parent:
    run_dir: Path
    settings: Settings
    env: frozenset[str]  # its engine environment, as NAME=VALUE
    version: str | None


class _Probing:
    """The current setup's server, asked one next-token question at a time."""

    def __init__(
        self,
        engine: Engine,
        client: httpx.AsyncClient,
        url: str,
        timeout_s: float,
        budget: int | None = None,
    ):
        self.engine, self.client, self.url, self.timeout_s = engine, client, url, timeout_s
        self.left = budget  # probes left; None: no limit
        self.out = TOO_MANY  # the reason when they run out
        self.leads: dict[tuple[int, ...], list[int]] = {}

    async def top(self, prefix: Sequence[int]) -> list[tuple[int, str, float]]:
        if self.left is not None:
            if self.left <= 0:
                raise CheckFailed(self.out)
            self.left -= 1
        found = await self.engine.next_token_logprobs(
            self.client, self.url, prefix, TOP, self.timeout_s
        )
        if not found:
            raise CheckFailed("the check's server did not answer a probe")
        return found

    async def lead(self, prefix: list[int]) -> tuple[list[int], list[tuple[int, str, float]]]:
        """The whitespace the model writes first, which the reasoning text lost, and the probe
        after it: that of position 0."""
        lead: list[int] = []
        found = await self.top(prefix)
        while _lead_ws(found[0][1]) and len(lead) < LEAD_PROBES:
            lead.append(found[0][0])
            found = await self.top(prefix + lead)
        return lead, found

    async def side(
        self,
        prefix: list[int],
        pieces: Sequence[tuple[int, bytes]],
        at: Sequence[int],
        first: list[tuple[int, str, float]] | None,
    ) -> list[Probe]:
        ids = [token for token, _ in pieces]
        offsets = char_offsets([piece for _, piece in pieces])
        probes = []
        for i in at:
            found = first if i == 0 and first is not None else await self.top(prefix + ids[:i])
            logprob = next((lp for token, _, lp in found if token == ids[i]), None)
            probes.append(Probe(i, offsets[i], ids[i], pieces[i][1].decode("utf-8", "replace"),
                                logprob, found[0][0], found[0][1], found[0][2]))  # fmt: skip
        return probes


async def replay(
    engine: Engine,
    client: httpx.AsyncClient,
    url: str,
    prompts: Sequence[Prompt],
    cached: Mapping[str, Any] | None,
    timeout_s: float,
) -> tuple[Side, Side, dict[str, Any]]:
    """Both sides' probes, the base side from ``cached`` (the A/A cache's prompts) when given;
    returns the base side's cache entries too."""
    probing = _Probing(engine, client, url, timeout_s)
    each = math.ceil(PROBES / len(prompts))
    texts = []
    for prompt in prompts:
        pair = []
        for text in (prompt.base, prompt.step):
            pieces = await engine.text_pieces(client, url, text)
            if pieces is None:
                raise CheckFailed("the check's server could not tokenize the reasoning")
            pair.append(pieces[:MAX_TOKENS])
        texts.append(pair)
    planned = [sum(min(len(pair[side]), each) for pair in texts) for side in (0, 1)]
    if min(planned) < MIN_PROBES:
        raise CheckFailed(f"too little reasoning to check ({min(planned)} tokens)")
    base: dict[str, list[Probe]] = {}
    step: dict[str, list[Probe]] = {}
    entries: dict[str, Any] = {}
    for prompt, (base_pieces, step_pieces) in zip(prompts, texts):
        held = (cached or {}).get(prompt.prompt_id)
        if held is not None:
            lead, first = list(held["lead"]), None
            base[prompt.prompt_id] = [Probe(*row) for row in held["probes"]]
        else:
            lead, first = await probing.lead(prompt.prefix)
            base[prompt.prompt_id] = await probing.side(
                prompt.prefix + lead, base_pieces, positions(len(base_pieces), each), first
            )
            first = None
        entries[prompt.prompt_id] = {
            "lead": lead,
            "probes": [list(_row(p)) for p in base[prompt.prompt_id]],
        }
        step[prompt.prompt_id] = await probing.side(
            prompt.prefix + lead, step_pieces, positions(len(step_pieces), each), first
        )
    return Side(base), Side(step), entries


def _row(probe: Probe) -> tuple[Any, ...]:
    return (probe.position, probe.char, probe.token, probe.wrote, probe.logprob, probe.top,
            probe.top_text, probe.top_logprob)  # fmt: skip


def run_if_needed(
    project: ResolvedProject,
    engine: Engine,
    runtime: Runtime,
    run_dir: Path,
    settings: Settings,
    ready_timeout_s: float,
) -> bool:
    """Run the checks this step needs, on one start of the current setup: the reasoning check
    when it has reasoned prompts whose reasoning differs from the run it was made from (written
    to ``reasoning_check.json``), the answer check when finished answers that differ are all
    that fail it (``answer_check.json``). A check that fails says why there and never fails the
    step. Returns whether Ctrl-C stopped it."""
    found = _differing(project, engine, run_dir, settings)
    if found is None:
        return False
    base, prompts, answers = found
    if engine.family != "llamacpp":
        if prompts:
            reason = f"the reasoning check is not available on {engine.label} yet"
            write_json(run_dir / CHECK_FILE, _not_run(reason))
        if answers:
            reason = f"the Fork check is not available on {engine.label} yet"
            write_json(run_dir / ANSWER_FILE, _not_run(reason, ANSWER_FORMAT))
        return False
    detail = (
        "Reasoning check: replaying the new setup's reasoning through your current setup"
        if prompts
        else "Fork check: replaying the answers that differ through your current setup"
    )
    emit(PhaseStarted(phase="compare", detail=detail))
    began = time.monotonic()
    extra = analysis().window_extra
    cut = extra.get("log_cut")
    interrupted = False
    try:
        documents = asyncio.run(
            _check(project, engine, runtime, run_dir, base, prompts, answers, ready_timeout_s)
        )
    except KeyboardInterrupt:
        documents, interrupted = _failed("interrupted"), True
    except AnalyseFailure as error:
        documents = _failed(
            "the check's server stopped during the check"
            if getattr(error, "ran", False)
            else "the check's server did not start"
        )
    finally:
        if cut is None:
            extra.pop("log_cut", None)
        else:
            extra["log_cut"] = cut
    change = _change(engine, base, settings, project, run_dir)
    seconds = time.monotonic() - began
    reasoning, answered = documents
    if prompts:
        reasoning = {**reasoning, "change": sorted(change)}
        reasoning["auto_keep"] = reasoning["verdict"] == "same" and engine.rounding_only(change)
        reasoning["seconds"] = round(seconds - answered.get("seconds", 0) if answers else seconds)
        write_json(run_dir / CHECK_FILE, reasoning)
    if answers:
        answered = {**answered, "change": sorted(change)}
        answered["auto_keep"] = answered["verdict"] == "same" and engine.answer_keep(change)
        answered["seconds"] = round(answered.get("seconds", seconds) if prompts else seconds)
        write_json(run_dir / ANSWER_FILE, answered)
    return interrupted


def _not_run(reason: str, format_: int = FORMAT) -> dict[str, Any]:
    return {"verdict": "not_run", "reason": reason, "format": format_}


def _failed(reason: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return _not_run(reason), _not_run(reason, ANSWER_FORMAT)


@dataclass(frozen=True, slots=True)
class Differs:
    """A finished prompt the answer check replays: the earlier run's distinct answers, and the
    step's that are none of them."""

    prompt_id: str
    base: tuple[Answer, ...]
    step: tuple[Answer, ...]


def _differing(
    project: ResolvedProject, engine: Engine, run_dir: Path, settings: Settings
) -> tuple[_Parent, list[tuple[int, Prompt]], list[tuple[int, Differs]]] | None:
    """The run this step was made from, and by workload index the prompts to check, at
    temperature 0 on both sides with no images: reasoned on both sides with reasoning not
    identical, the first PROMPTS in workload order (their ``prefix`` still empty); and the
    finished prompts the answer check replays. None when there is nothing to check."""
    from .project import workload_facts

    context = analysis()
    project_dir = run_dir.parent.parent
    if not context.engine_args or project.config.workload.temperature != 0:
        return None
    runs = past_runs(project_dir, engine, before=context.reported_run)
    args = frozenset(context.engine_args)
    found = step_parent(project_dir, runs, args, kept_run(project_dir, runs, context.kept))
    if found is None or not found.served:
        return None
    base_dir = project_dir / "runs" / found.run_id
    run_file = RunFile.read(base_dir)
    base = _Parent(
        base_dir,
        Settings.from_json(found.settings),
        recorded_env(run_file.model_dump(), engine.flag_rules),
        run_file.engine_version,
    )
    quantized = workload_facts(project).quantization is not None
    asked = context.try_step or answer_changes(
        engine, base.settings, settings, checkpoint_quantized=quantized
    )
    if not asked or run_file.temperature != 0 or not run_file.retained_responses:
        return None
    if not (run_dir / "responses.jsonl").is_file():
        return None
    shared = shared_prompts(base_dir, run_dir)
    before = only_prompts(answers_in(base_dir), shared)
    after = only_prompts(answers_in(run_dir), shared)
    outcome = compare_answers(before, after, {}, json_answers=context.json_answers, greedy=True)
    differ = {
        p.prompt_id: p.first_difference
        for p in outcome.prompts
        if p.reasoned and p.identical is False
    }
    against = set(_against(outcome, after))
    plain = [
        (index, entry)
        for index, entry in enumerate(project.prompts)
        if entry.messages is not None and not _has_images(entry.messages)
    ]
    prompts = [
        (index, Prompt(entry.id, [], before[entry.id][0].reasoning,
                       after[entry.id][0].reasoning, differ[entry.id]))
        for index, entry in plain
        if entry.id in differ
    ][:PROMPTS]  # fmt: skip
    answers = [
        (index, Differs(entry.id, *distinct(_finished(before[entry.id]),
                                            _finished(after[entry.id]))))
        for index, entry in plain
        if entry.id in against
    ]  # fmt: skip
    return (base, prompts, answers) if prompts or answers else None


def _finished(answers: Sequence[Answer]) -> list[Answer]:
    return [a for a in answers if not a.thinking]


def _against(outcome: Outcome, after: Mapping[str, Sequence[Answer]]) -> list[str]:
    """The finished prompts the answer check replays: those the text rules count against the
    step, when nothing else fails it (no broken answer, no prompt that only thought)."""
    if outcome.thought or any(broken(a) for given in after.values() for a in given):
        return []
    return counted_against(outcome.prompts)


def _has_images(messages: Sequence[Any]) -> bool:
    return any(
        isinstance(part, ImagePart)
        for message in messages
        if not isinstance(message.content, str)
        for part in message.content
    )


def _prompt_tokens(run_dir: Path) -> dict[str, int]:
    """Each prompt's token count as the server counted it in a successful request."""
    found: dict[str, int] = {}
    try:
        lines = (run_dir / "requests.jsonl").read_text("utf-8").splitlines()
    except OSError:
        return found
    for line in lines:
        row = json.loads(line)
        if row.get("outcome") == "success" and isinstance(row.get("prompt_tokens"), int):
            found.setdefault(row["prompt_id"], row["prompt_tokens"])
    return found


def _change(
    engine: Engine, base: _Parent, settings: Settings, project: ResolvedProject, run_dir: Path
) -> set[str]:
    """What the step changes from the run it was made from, as ``name=value`` words: engine
    arguments either way (without those the baseline's launch pinned) and engine variables."""
    from .kept import pinned_baseline

    pins = pinned_baseline(run_dir.parent.parent, project, engine)
    pinned = {word.partition("=")[0] for word in pins}
    words = {w for w in engine.engine_args_between(base.settings, settings) if w not in pins}
    words |= {
        w
        for w in engine.engine_args_between(settings, base.settings)
        if w.partition("=")[0] not in pinned
    }
    before = {f"{k}={v}" for k, v in base.settings.extra_env.items()} | base.env
    now = {f"{k}={v}" for k, v in settings.extra_env.items()} | _step_env(engine)
    return words | (before ^ now)


def _step_env(engine: Engine) -> frozenset[str]:
    return env_args(engine.inherited_env(dict(os.environ)), engine.flag_rules)


@contextlib.contextmanager
def _environment(engine: Engine, env: frozenset[str]) -> Iterator[None]:
    """The engine environment ``env`` of the run the step was made from, for the check's
    launch, in place of the step's own; the step's is put back after."""
    saved = {word.partition("=")[0]: os.environ[word.partition("=")[0]]
             for word in _step_env(engine)}  # fmt: skip
    try:
        for name in saved:
            del os.environ[name]
        os.environ.update(dict(word.split("=", 1) for word in env))
        yield
    finally:
        for word in env:
            os.environ.pop(word.partition("=")[0], None)
        os.environ.update(saved)


async def _check(
    project: ResolvedProject,
    engine: Engine,
    runtime: Runtime,
    run_dir: Path,
    base: _Parent,
    prompts: Sequence[tuple[int, Prompt]],
    answers: Sequence[tuple[int, Differs]],
    ready_timeout_s: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from .images import ImageSource
    from .measure import serving, spec_for

    settings = replace(
        base.settings,
        extra_args=engine.flag_rules.merge(base.settings.extra_args, CHECK_FLAGS),
        extra_env={**base.settings.extra_env, **CHECK_ENV},
    )
    record = run_dir / CHECK_DIR
    record.mkdir(mode=0o700, exist_ok=True)
    workload = project.config.workload
    sent = _prompt_tokens(base.run_dir)
    spec = spec_for(project, engine, settings, run_dir.name + "-check", None)
    reasoning: dict[str, Any] = {}
    answered: dict[str, Any] = {}
    with _environment(engine, base.env):
        async with serving(
            engine, runtime, spec, ready_timeout_s=ready_timeout_s, record_dirs=[record],
            phase="reasoning check" if prompts else "Fork check",
            images=ImageSource(project.images),
        ) as live:  # fmt: skip
            url = live.server.endpoint_url
            if prompts:
                try:
                    reasoning = await _reasoning(
                        engine, live.client, url, workload, base, prompts, sent, run_dir
                    )
                except CheckFailed as error:
                    reasoning = _not_run(str(error))
            if answers:
                began = time.monotonic()
                try:
                    answered = await check_answers(
                        engine, live.client, url, workload, answers, sent
                    )
                except CheckFailed as error:
                    answered = _not_run(str(error), ANSWER_FORMAT)
                answered["seconds"] = time.monotonic() - began
    return reasoning, answered


async def _body(engine: Engine, workload: WorkloadSpec, index: int) -> dict[str, Any]:
    chat = workload.chats[index]
    body = await chat_body(chat)
    if chat.tool_choice is not None:
        body["tool_choice"] = chat.tool_choice
    body.update(engine.request_extras(record_workload(workload, index)))
    return body


async def _reasoning(
    engine: Engine,
    client: httpx.AsyncClient,
    url: str,
    workload: WorkloadSpec,
    base: _Parent,
    prompts: Sequence[tuple[int, Prompt]],
    sent: Mapping[str, int],
    run_dir: Path,
) -> dict[str, Any]:
    built = []
    for index, prompt in prompts:
        ids = await engine.prompt_ids(client, url, await _body(engine, workload, index))
        if ids is not None and len(ids) == sent.get(prompt.prompt_id):
            built.append(replace(prompt, prefix=ids))
    if not built:
        raise CheckFailed("could not rebuild the prompt")
    key = _cache_key(base, [p.prompt_id for p in built])
    cached = _read_cache(base.run_dir, key)
    base_side, step_side, entries = await replay(
        engine, client, url, built, cached, workload.request_timeout_s
    )
    if cached is None:
        write_json(base.run_dir / CACHE_FILE, {"key": key, "prompts": entries})
    after = answers_in(run_dir)
    broke = any(broken(a) for p in built for a in after.get(p.prompt_id, ()))
    found, reason = verdict(base_side, step_side, broken_answers=broke)
    return _document(found, reason, base_side, step_side, built, cached is not None)


@dataclass(frozen=True, slots=True)
class Written:
    """An answer as the current setup reads it: the prompt and the answer's tokens in ``ids``,
    the answer from ``start``, and the text of those tokens whose text is known."""

    ids: list[int]
    start: int
    pieces: Mapping[int, str]


@dataclass(frozen=True, slots=True)
class Checked:
    """One answer of the step that is none of the earlier run's, replayed: ``explained`` when
    every token from where it parts is within TIE of the current setup's pick, ``changed`` when
    one is not, ``unchecked`` with the reason when the replay cannot tell."""

    prompt_id: str
    result: Literal["explained", "changed", "unchecked"]
    reason: str | None = None
    parted: int | None = None  # the answer token where it parts from the earlier run's
    tokens: int = 0  # its tokens checked
    cut: bool = False  # either side went on past ANSWER_TOKENS from there
    worst: Mapping[str, Any] | None = None  # its token furthest below the current setup's pick


def _near(found: Sequence[tuple[int, str, float]], token: int) -> float | None:
    """How far below the top token ``token`` is, in nats; None outside TIE or the top TOP."""
    logprob = next((lp for t, _, lp in found if t == token), None)
    if logprob is None or logprob < found[0][2] - TIE:
        return None
    return found[0][2] - logprob


async def check_answer(probing: _Probing, prompt_id: str, base: Written, step: Written) -> Checked:
    """The step's answer ``step`` against the earlier run's copy ``base`` it shares the longest
    start with, from the first token where they part: the copy is the guard, so every one of
    its tokens must be near the top, and so must every one of the answer's to be explained."""
    parted = next(
        (i for i, (a, b) in enumerate(zip(base.ids, step.ids)) if a != b),
        min(len(base.ids), len(step.ids)),
    )
    if parted < step.start:
        return Checked(prompt_id, "unchecked", "the prompt was rebuilt differently")
    sides = (base, step)
    ranges = [range(parted, min(len(side.ids), parted + ANSWER_TOKENS)) for side in sides]
    cut = any(len(side.ids) > parted + ANSWER_TOKENS for side in sides)
    if not any(ranges):
        return Checked(prompt_id, "unchecked", "the replay cannot see where they differ")
    first = await probing.top(base.ids[:parted])
    for at in ranges[0]:
        found = first if at == parted else await probing.top(base.ids[:at])
        if _near(found, base.ids[at]) is None:
            return Checked(prompt_id, "unchecked", REPLAY, parted - step.start)
    worst: dict[str, Any] | None = None
    for at in ranges[1]:
        found = first if at == parted else await probing.top(step.ids[:at])
        gap = _near(found, step.ids[at])
        if gap is None or worst is None or gap > worst["gap"]:
            listed = next((text for t, text, _ in found if t == step.ids[at]), None)
            worst = {
                "position": at - step.start,
                "wrote": step.pieces.get(at) if listed is None else listed,
                "writes": found[0][1],
                "gap": None if gap is None else round(gap, 3),
                "p": _ratio(found, step.ids[at]),
            }
        if gap is None:
            return Checked(prompt_id, "changed", None, parted - step.start, at - parted + 1, cut,
                           worst)  # fmt: skip
    return Checked(prompt_id, "explained", None, parted - step.start, len(ranges[1]), cut, worst)


def _ratio(found: Sequence[tuple[int, str, float]], token: int) -> float | None:
    """How likely ``token`` is against the top one, as the current setup rates them; None
    outside the top TOP."""
    logprob = next((lp for t, _, lp in found if t == token), None)
    return None if logprob is None else round(math.exp(logprob - found[0][2]), 4)


async def check_answers(
    engine: Engine,
    client: httpx.AsyncClient,
    url: str,
    workload: WorkloadSpec,
    answers: Sequence[tuple[int, Differs]],
    sent: Mapping[str, int],
) -> dict[str, Any]:
    """Replay each answer of ``answers`` through the current setup's server at ``url``, within
    ANSWER_PROBES probes, up to the first that reads changed: the verdict cannot change after
    it. When a full replay clearly cannot fit, only look for a changed answer, within
    ANSWER_HUNT probes. The answer check's document."""
    probing = _Probing(engine, client, url, workload.request_timeout_s, ANSWER_PROBES)
    end = await engine.end_of_turn(client, url)
    built = []
    for index, differs in answers:
        body = await _body(engine, workload, index)
        head = await engine.prompt_ids(client, url, body)
        built.append(
            (
                differs,
                body,
                head if head is not None and len(head) == sent.get(differs.prompt_id) else None,
            )
        )
    needed = await _planned(engine, client, url, [d for d, _, head in built if head is not None])
    if needed > ANSWER_OVER * ANSWER_PROBES:
        probing.left = ANSWER_HUNT
        probing.out = (
            f"too many answers differ to check in time: about {needed} probes needed (budget "
            f"{ANSWER_PROBES}); searched the first {ANSWER_HUNT} for a change, none found; "
            "judged as before"
        )
    checked: list[Checked] = []
    for n, (differs, body, head) in enumerate(built, 1):
        if head is None:
            checked += [Checked(differs.prompt_id, "unchecked", "could not rebuild the prompt")]
            continue
        if not differs.step:
            checked.append(Checked(differs.prompt_id, "unchecked", SAME))
        for i, answer in enumerate(differs.step, 1):
            checked.append(await _check_one(probing, body, head, end, differs, answer))
            if checked[-1].result == "changed":
                rest = [d.prompt_id for _, d in answers[n:]]
                if not rest and i == len(differs.step):
                    return _answer_document(checked)
                return _answer_document(checked, stopped=(n, len(answers), rest))
    return _answer_document(checked)


async def _planned(
    engine: Engine, client: httpx.AsyncClient, url: str, prompts: Sequence[Differs]
) -> int:
    """The probes a full replay of ``prompts`` takes, by the answers' tokens and no probe: from
    where each answer parts from the earlier run's copy it shares the most with, both sides up
    to ANSWER_TOKENS, the first probe shared; one for an answer on a different route than every
    copy, none for one whose reasoning differs. Leads and end tokens are left out."""
    tokens: dict[str, list[int] | None] = {}

    async def ids(answer: Answer) -> list[int] | None:
        text = answer.text.lstrip()
        if text not in tokens:
            found = await engine.text_pieces(client, url, text)
            tokens[text] = None if found is None else [token for token, _ in found]
        return tokens[text]

    total = 0
    for differs in prompts:
        for answer in differs.step:
            routes = [(_route(answer, copy), _route(copy, answer), copy) for copy in differs.base]
            if any("reasoning" in (mine, theirs) for mine, theirs, _ in routes):
                continue
            mine = await ids(answer)
            pairs = []
            for route, other, copy in routes:
                theirs = await ids(copy) if route == other != "tool" else None
                if mine is None or theirs is None:
                    continue
                shared = next(
                    (i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b),
                    min(len(mine), len(theirs)),
                )
                sides = [min(len(side) - shared, ANSWER_TOKENS) for side in (theirs, mine)]
                pairs.append((shared, -(sum(sides) - (1 if all(sides) else 0))))
            total += -max(pairs)[1] if pairs else 1
    return total


def _route(answer: Answer, other: Answer) -> Literal["text", "content", "tool", "reasoning"]:
    """How ``answer`` is rebuilt when paired with ``other``, by what it holds; "reasoning" when
    both reasoned and the reasoning differs, so they part there and the answer after it is not
    reached: such answers are not checked yet."""
    if answer.reasoning and other.reasoning and answer.reasoning != other.reasoning:
        return "reasoning"
    if answer.tool_calls:
        return "tool"
    return "content" if answer.reasoning else "text"


async def _check_one(
    probing: _Probing,
    body: Mapping[str, Any],
    head: list[int],
    end: int | None,
    differs: Differs,
    answer: Answer,
) -> Checked:
    """``answer`` against the earlier run's copy whose tokens share the longest start with it."""
    pairs = []
    built: dict[str, Written | None] = {}
    for copy in differs.base:
        route, other = _route(answer, copy), _route(copy, answer)
        if route == "reasoning" or other == "reasoning":
            return Checked(differs.prompt_id, "unchecked", THOUGHT)
        if route not in built:
            built[route] = await _written(probing, body, head, answer, route, end)
        mine = built[route]
        theirs = await _written(probing, body, head, copy, other, end)
        if mine is not None and theirs is not None:
            shared = next(
                (i for i, (a, b) in enumerate(zip(mine.ids, theirs.ids)) if a != b),
                min(len(mine.ids), len(theirs.ids)),
            )
            pairs.append((shared, theirs, mine))
    if not pairs:
        return Checked(differs.prompt_id, "unchecked", "could not rebuild the answer")
    _, theirs, mine = max(pairs, key=lambda pair: pair[0])
    return await check_answer(probing, differs.prompt_id, theirs, mine)


async def _written(
    probing: _Probing,
    body: Mapping[str, Any],
    head: list[int],
    answer: Answer,
    route: Literal["text", "content", "tool"],
    end: int | None,
) -> Written | None:
    """``answer`` as the current setup reads it after the prompt ``head`` (``body`` rendered),
    by ``route``; None when it cannot be rebuilt."""
    engine, client, url = probing.engine, probing.client, probing.url
    if route == "tool":
        message = {
            "role": "assistant",
            "content": answer.text,
            **({"reasoning_content": answer.reasoning} if answer.reasoning else {}),
            "tool_calls": [
                {"id": f"call_{i}", "type": "function",
                 "function": {"name": name, "arguments": arguments}}
                for i, (name, arguments) in enumerate(answer.tool_calls or ())
            ],
        }  # fmt: skip
        ids = await engine.assistant_ids(client, url, body, message, "tool")
        return None if ids is None else Written(ids, len(head), {})
    text, prefix = answer.text, head
    if route == "content":
        message = {"role": "assistant", "reasoning_content": answer.reasoning, "content": ""}
        found = await engine.assistant_ids(client, url, body, message, "content")
        if found is None:
            return None
        prefix = found
    ends = answer.finish_reason == "stop"
    pieces = await engine.text_pieces(client, url, text.lstrip())
    if pieces is None or (ends and end is None):
        return None
    lead = probing.leads.get(tuple(prefix))
    if lead is None:
        lead = probing.leads.setdefault(tuple(prefix), (await probing.lead(prefix))[0])
    at = len(prefix) + len(lead)
    return Written(
        [*prefix, *lead, *(token for token, _ in pieces), *([end] if ends and end else [])],
        len(head),
        {at + i: piece.decode("utf-8", "replace") for i, (_, piece) in enumerate(pieces)},
    )


def _answer_document(
    checked: Sequence[Checked], stopped: tuple[int, int, list[str]] | None = None
) -> dict[str, Any]:
    """The answer check's result: by prompt, explained when every answer is, changed when any
    is; the others stay with the text rules. ``stopped``: the prompt it read changed at, of how
    many, and those after it, not replayed."""
    by_prompt: dict[str, list[Checked]] = {}
    for row in checked:
        by_prompt.setdefault(row.prompt_id, []).append(row)
    changed = [p for p, rows in by_prompt.items() if any(r.result == "changed" for r in rows)]
    explained = [p for p, rows in by_prompt.items() if all(r.result == "explained" for r in rows)]
    left = [r for r in checked if r.result == "unchecked" and r.reason != SAME]
    unchecked = [p for p in by_prompt if p not in changed and p not in explained]
    unchecked = [p for p in unchecked if any(r.prompt_id == p for r in left)]
    reasons = [r.reason for r in left if r.reason]
    found = (
        "changed" if changed else ("partial" if unchecked else "same") if explained else "not_run"
    )
    return {
        "format": ANSWER_FORMAT,
        "verdict": found,
        "reason": reasons[0] if found in ("not_run", "partial") and reasons else None,
        "explained": explained,
        "changed": changed,
        "unchecked": unchecked,
        "tokens": ANSWER_TOKENS,
        **(
            {"stopped": {"at": stopped[0], "of": stopped[1], "not_replayed": stopped[2]}}
            if stopped
            else {}
        ),
        "answers": [
            {
                "prompt_id": r.prompt_id,
                "result": r.result,
                "reason": r.reason,
                "parted": r.parted,
                "tokens": r.tokens,
                "cut": r.cut,
                "worst": dict(r.worst) if r.worst else None,
            }
            for r in checked
        ],
    }


def _cache_key(base: _Parent, prompt_ids: Sequence[str]) -> str:
    text = json.dumps(
        [FORMAT, base.run_dir.name, repr(base.settings), sorted(base.env), base.version,
         list(prompt_ids), CHECK_FLAGS, CHECK_ENV],
    )  # fmt: skip
    return hashlib.sha256(text.encode()).hexdigest()


def _read_cache(base_dir: Path, key: str) -> Mapping[str, Any] | None:
    try:
        document = json.loads((base_dir / CACHE_FILE).read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict) or document.get("key") != key:
        return None
    prompts = document.get("prompts")
    return prompts if isinstance(prompts, dict) else None


def _document(
    found: Verdict,
    reason: str | None,
    base: Side,
    step: Side,
    prompts: Sequence[Prompt],
    cached: bool,
) -> dict[str, Any]:
    def departure(probe: Probe) -> tuple[bool, float]:
        return probe.logprob is not None, (probe.logprob or 0.0) - probe.top_logprob

    worst = sorted(
        (
            (prompt.prompt_id, probe, prompt.step)
            for prompt in prompts
            for probe in step.probes[prompt.prompt_id]
            if not probe.agree
        ),
        key=lambda found: departure(found[1]),
    )[:3]
    return {
        "format": FORMAT,
        "verdict": found,
        "reason": reason,
        "agree_b": round(base.share(), 4),
        "agree_s": round(step.share(), 4),
        "far_b": round(base.share(far=True), 4),
        "far_s": round(step.share(far=True), 4),
        "probes": [base.count, step.count],
        "tokens": MAX_TOKENS,
        "cached": cached,
        "prompts": [
            {
                "prompt_id": p.prompt_id,
                "probes": len(step.probes[p.prompt_id]),
                "agree_b": round(base.share(p.prompt_id), 4),
                "agree_s": round(step.share(p.prompt_id), 4),
                "far_s": round(step.share(p.prompt_id, far=True), 4),
                "diverge": p.diverge,
            }
            for p in prompts
        ],
        "worst": [
            {
                "prompt_id": prompt_id,
                "char": probe.char,
                "wrote": probe.wrote,
                "writes": probe.top_text,
                "p": round(math.exp(probe.top_logprob), 3),
                "before": text[max(0, probe.char - 80) : probe.char],
                "after": text[probe.char + len(probe.wrote) : probe.char + len(probe.wrote) + 80],
            }
            for prompt_id, probe, text in worst
        ],
    }


def read(run_dir: Path) -> dict[str, Any] | None:
    """A run's ``reasoning_check.json``; None when the check did not run for it."""
    try:
        found = json.loads((run_dir / CHECK_FILE).read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) and "verdict" in found else None


def _far_words(checked: Mapping[str, Any]) -> str:
    far = round(checked["far_s"] * checked["probes"][1])
    if not far:
        return "none was a token it found unlikely"
    noun = "was a token" if far == 1 else "were tokens"
    return f"{far} {noun} it found unlikely (1 in 10 or less)"


def summary(checked: Mapping[str, Any]) -> str:
    """What the check measured, in one sentence."""
    probes, prompts = checked["probes"][1], len(checked["prompts"])
    return (
        "Tensward replayed the new setup's reasoning through your current setup: "
        f"{probes} tokens across {prompts} prompt{'' if prompts == 1 else 's'}, up to token "
        f"{checked['tokens']}. Your current setup would have written the same token at "
        f"{checked['agree_s']:.1%} of them (its own reasoning, replayed the same way: "
        f"{checked['agree_b']:.1%}); {_far_words(checked)}."
    )


def departs(checked: Mapping[str, Any]) -> str:
    """Why a CHANGED step is held for the user."""
    far = checked["far_s"]
    unlikely = f"; {far:.0%} were tokens it found unlikely" if far >= 0.005 else ""
    return (
        f"The new setup's {DEPARTS}: the same token "
        f"at {checked['agree_s']:.0%} of {checked['probes'][1]} (its own: "
        f"{checked['agree_b']:.1%}){unlikely}."
    )


def lines(checked: Mapping[str, Any]) -> list[str]:
    """The try screen's lines on the check."""
    minutes = f"Reasoning check: {max(1, round(checked.get('seconds', 0) / 60))} min."
    found = checked["verdict"]
    if found == "not_run":
        return [f"Could not check the reasoning: {checked['reason']}.", minutes]
    if found == "changed":
        return [summary(checked), minutes]
    text = [summary(checked)]
    if found == "same" and not checked.get("auto_keep"):
        text.append(
            f"The check covers the first {checked['tokens']} reasoning tokens; this change can "
            "matter later in long answers."
        )
    elif found == "your_call":
        text.append(
            "Tensward cannot tell whether that changes the final answers. The 3 biggest "
            'departures are in report.md ("Answers before / after").'
        )
    return [*text, minutes]


def gate_words(checked: Mapping[str, Any]) -> str:
    """For a reasoned step the check did not pass: what the check found, as a clause."""
    found = checked["verdict"]
    if found == "same":
        return (
            f"the reasoning check found the same reasoning up to token {checked['tokens']}, "
            "but this change can matter later in long answers"
        )
    if found == "your_call":
        return (
            f"the reasoning check found the same token at {checked['agree_s']:.1%} "
            f"(its own: {checked['agree_b']:.1%}), too close to tell"
        )
    return f"the reasoning could not be checked ({checked['reason']})"


def row(checked: Mapping[str, Any]) -> tuple[str, str] | None:
    """The Answers row's text and mark when the check decides them; None to keep the counts."""
    if checked["verdict"] == "changed":
        return "changed", REVIEW_MARK
    if checked["verdict"] == "same":
        return ("same reasoning", "✔") if checked.get("auto_keep") else ("same so far", REVIEW_MARK)
    return None


def read_answers(run_dir: Path) -> dict[str, Any] | None:
    """A run's ``answer_check.json``; None when the answer check did not run for it."""
    try:
        found = json.loads((run_dir / ANSWER_FILE).read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) and "verdict" in found else None


def _count(count: int) -> str:
    return f"{count} answer{'' if count == 1 else 's'}"


def answer_row(answers: Mapping[str, Any]) -> tuple[str, str] | None:
    """The Answers row's text and mark when the answer check decides them; None to keep the
    counts."""
    checked = answers.get("answer_check") or {}
    counts = f"{answers['identical']}/{answers['total']} same"
    if checked.get("verdict") == "changed":
        if "stopped" in checked:
            return f"{counts}, changed", REVIEW_MARK
        return f"{counts}, {len(checked['changed'])} changed", REVIEW_MARK
    if answers_pass(answers) and rounding(answers):
        return f"{counts}, {len(checked['explained'])} rounding", "✔"
    if rounding_passes(answers) and not checked.get("auto_keep"):
        return f"{counts}, {answers['differ']} differ", REVIEW_MARK
    return None


def answer_lines(answers: Mapping[str, Any]) -> list[str]:
    """The try screen's lines on the answer check."""
    checked = answers["answer_check"]
    minutes = (
        [f"Fork check: {max(1, round(checked['seconds'] / 60))} min."]
        if "seconds" in checked
        else []
    )
    if checked["verdict"] == "not_run":
        return [
            f"Fork check: the answers that differ could not be checked one by one "
            f"({checked['reason']}); judged as before.",
            *minutes,
        ]
    if checked["verdict"] == "changed":
        return [f"{stopped_words(checked)}.", *minutes] if "stopped" in checked else minutes
    if checked["verdict"] == "partial":
        total = len(checked["explained"]) + len(checked["unchecked"])
        return [
            f"Fork check: {len(checked['explained'])} of {total} differing answers explained, "
            f"{len(checked['unchecked'])} not checked "
            f"({checked['reason']}); none counted as explained.",
            *minutes,
        ]
    text = []
    kept = set(rounding(answers))
    if explained := sum(a["prompt_id"] in kept for a in checked["answers"]):
        text.append(
            f"{_count(explained)} differ{'s' if explained == 1 else ''}. In "
            f"{'it' if explained == 1 else 'each'}, every word from the point where it parts is "
            "one your current setup rated almost as likely as its own pick. Parallel requests "
            "round differently, and two runs of the same setup do this too."
        )
        if any(a["cut"] for a in checked["answers"] if a["result"] == "explained"):
            text.append(
                f"Long answers are checked up to {checked['tokens']} tokens after they part."
            )
    if unchecked := [
        a for a in checked["answers"] if a["result"] == "unchecked" and a["reason"] != SAME
    ]:
        text.append(
            f"{_count(len(unchecked))} could not be checked one by one "
            f"({unchecked[0]['reason']}); judged as before."
        )
    return [*text, *minutes]


def answer_departs(checked: Mapping[str, Any]) -> str:
    """Why a step whose answers the answer check found changed is not kept, with the first one."""
    changed = [a for a in checked["answers"] if a["result"] == "changed"]
    count, first = len(changed), changed[0]
    worst = first["worst"]
    wrote = "a word" if worst["wrote"] is None else json.dumps(worst["wrote"], ensure_ascii=False)
    odds = f"1 in {round(1 / worst['p'])}" if worst["p"] else f"not in its top {TOP}"
    return (
        f"{_count(count)} use{'s' if count == 1 else ''} words {WOULD_NOT_WRITE} "
        f"({first['prompt_id']}: {wrote} where it writes "
        f"{json.dumps(worst['writes'], ensure_ascii=False)}, {odds})."
    )


def stopped_words(checked: Mapping[str, Any]) -> str:
    """Where an answer check that stopped at the first changed answer stopped."""
    stopped = checked["stopped"]
    return (
        f"The Fork check stopped at the first changed answer: changed at prompt "
        f"{stopped['at']} of the {stopped['of']} that differ; the rest not replayed"
    )


def rounding_words(change: str) -> str:
    """Why a step whose answers differ only by rounding is still the user's call, for a
    ``change`` of these flags."""
    return (
        f"{LIKE_ROUNDING}, but a change to {change} can matter more in long answers than these "
        "show."
    )


def parent_dir(
    project_dir: Path, engine: Engine, args: frozenset[str], kept: Kept | None
) -> Path | None:
    """The run a try of ``args`` is checked against, found as :func:`_differing` finds it."""
    runs = past_runs(project_dir, engine)
    found = step_parent(project_dir, runs, args, kept_run(project_dir, runs, kept))
    return project_dir / "runs" / found.run_id if found is not None and found.served else None


_Timed = tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], float, str]


def _timed(run_dir: Path) -> _Timed | None:
    """The run in ``run_dir``'s run.json, metrics.json, request rows, start-up seconds and
    server log; None when one is missing."""
    try:
        run = json.loads((run_dir / "run.json").read_text("utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text("utf-8"))
        rows = [json.loads(line) for line in (run_dir / "requests.jsonl").read_text().splitlines()]
        window = json.loads((run_dir / "window.json").read_text("utf-8"))
        ready = float(window["startup"]["ready_s"])
        log = (run_dir / "server.log").read_text("utf-8", "replace")
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return run, metrics, rows, ready, log


def _rates(
    metrics: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], log: str
) -> tuple[float, float] | None:
    """The run's best prefill rate (tokens a second) and its fastest decode step (seconds); None
    when they cannot tell, as on a setup with layers or experts in host memory, where a probe's
    cost is not measured yet."""
    facts = (metrics.get("resolved") or {}).get("facts") or {}
    placed = facts.get("placement") or {}
    if any(facts.get("host_expert_bytes") or placed.get(k) for k in HOST_PLACED):
        return None
    timed = [r["timings"] for r in rows if r.get("outcome") == "success" and r.get("timings")]
    rates = [t["prompt_n"] / t["prompt_ms"] * 1000 for t in timed if t.get("prompt_n", 0) >= 64]
    rates += [int(n) / float(ms) * 1000 for ms, n in PROMPT_EVAL.findall(log) if int(n) >= 64]
    decode = [t["predicted_ms"] / t["predicted_n"] / 1000 for t in timed if t.get("predicted_n")]
    return (max(rates), min(decode)) if rates and decode else None


def estimate(run_dir: Path) -> tuple[bool, float | None]:
    """Whether a try from the run in ``run_dir`` runs the check (every answer of some prompt
    ran out of output tokens while thinking, at temperature 0, on llama.cpp), and the seconds
    the check takes by that run's own uncontended timings; None when they cannot tell."""
    found = _timed(run_dir)
    if found is None:
        return False, None
    run, metrics, rows, ready, log = found
    cut = {
        prompt
        for prompt, given in answers_in(run_dir).items()
        if all(a.thinking and a.finish_reason == "length" for a in given)
    }
    if run.get("engine") != "llamacpp" or run.get("temperature") != 0 or not cut:
        return False, None
    timing = _rates(metrics, rows, log)
    thought = sorted(r.get("committed_output_tokens") or 0 for r in rows if r["prompt_id"] in cut)
    sent = sorted(r.get("prompt_tokens") or 0 for r in rows if r["prompt_id"] in cut)
    if timing is None or not thought:
        return True, None
    rate, step = timing
    prompts = min(PROMPTS, len(cut))
    tokens = min(MAX_TOKENS, thought[len(thought) // 2])
    probes = min(tokens, math.ceil(PROBES / prompts))
    sides = 1 if (run_dir / CACHE_FILE).is_file() else 2
    prefill = prompts * sent[len(sent) // 2] / rate
    probe = tokens / probes / rate + step
    return True, ready + sides * (prefill + prompts * probes * probe)


def answer_estimate(run_dir: Path) -> tuple[bool, float | None]:
    """Whether a try from the run in ``run_dir`` can run the answer check (at temperature 0, on
    llama.cpp), and the most seconds it takes when answers differ, by that run's own timings: a
    start, a prefill for each answer the probe budget covers and every probe; None when they
    cannot tell."""
    found = _timed(run_dir)
    if found is None:
        return False, None
    run, metrics, rows, ready, log = found
    if run.get("engine") != "llamacpp" or run.get("temperature") != 0:
        return False, None
    timing = _rates(metrics, rows, log)
    sent = sorted(r.get("prompt_tokens") or 0 for r in rows if r.get("outcome") == "success")
    if timing is None or not sent:
        return True, None
    rate, step = timing
    answers = ANSWER_PROBES // (2 * ANSWER_TOKENS)
    return True, ready + answers * sent[len(sent) // 2] / rate + ANSWER_PROBES * (step + 1 / rate)
