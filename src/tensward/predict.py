"""The memory-traffic model: a decode step reads the weights and each running sequence's KV
cache at the rate the run's own decode part reached, beside a prompt part. It predicts the TPOT
and request rate of a precision or concurrency change from the bytes that change removes or
adds. On a KV-bound run a precision change also grows the batch into the memory it frees.
Client TTFT follows TPOT where the tool-call parser holds the first content for some decode
steps or where a saturated closed loop holds the prompt inside decode steps, and loses its queued
part where the larger batch admits the waiting requests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from .answer_options import NATIVE_FP8
from .context import current as analysis
from .engines.vllm.quantization import ONLINE_FP8
from .playbook_common import kv_held_back
from .pressure import FULL, PROMPT_SHARE, TOOL_SHARE, WAITING, demand

if TYPE_CHECKING:
    from .measurement import Measurement
    from .playbook import Situation
    from .settings import Settings

KV_PRECISION = ("fp8-kv-cache", "fp8-kv-decode")
PRECISION = ("fp8-weights", *KV_PRECISION)
CAP_CHANGES = ("raise-concurrency", "kv-sized-concurrency", "lower-concurrency")
# with the prompt share unmeasured, only these get a number: their gain is the bytes alone
BOUNDED = ("fp8-weights", "fp8-kv-decode")
HELD_MIN = 4.0  # decode steps of parser hold from which client TTFT follows TPOT
SATURATED = 0.95  # mean running over the clients (or the cap) from which a prompt rides in steps
KV_ONLY_HOLD = 0.5  # share of the step change a prompt sees when only decode gets faster


@dataclass(frozen=True, slots=True)
class Prediction:
    tpot: float | None  # relative change of TPOT (-0.22 for -22%)
    req_s: float | None  # relative change of requests per second at the same cap
    text: str
    ttft: float | None = None  # relative change of client TTFT p50


@dataclass(frozen=True, slots=True)
class _Room:
    """A KV-bound run: requests wait for KV space while the cap leaves room."""

    running: float  # peak running requests
    tokens: float  # KV tokens per running request at that peak
    bytes_per_token: float  # of the engine's pool
    cap: int


def upper_share(m: Measurement, weights: float) -> float | None:
    """The prompt share's upper bound where the probe did not measure it: decode cannot read
    its bytes faster than the datasheet bandwidth. None unless the lower bound (prompt tokens at
    the datasheet prefill ceiling) brackets it below PROMPT_SHARE. ``weights``: the weight bytes
    a step reads."""
    c = m.ceilings
    if (
        c is None or m.step_ms is None or m.prompt_tokens_per_step is None
        or not c.prefill_ceiling_tok_s or not c.bandwidth_gbs
        or c.kv_bytes_per_sequence is None or c.avg_running_batch is None
    ):  # fmt: skip
        return None
    seconds = m.step_ms / 1000
    lower = m.prompt_tokens_per_step / c.prefill_ceiling_tok_s / seconds
    read = weights + c.kv_bytes_per_sequence * c.avg_running_batch
    upper = 1 - read / (c.bandwidth_gbs * 1e9 * seconds)
    if upper < lower or lower > PROMPT_SHARE / 2 or upper > PROMPT_SHARE:
        return None
    return upper


def held_steps(m: Measurement) -> float | None:
    """Decode steps the tool-call parser holds a streamed answer's first content for: the
    client's TTFT p50 over the engine's, in TPOT p50. None where few answers call a tool or the
    hold is under ``HELD_MIN`` steps."""
    calls = m.tool_calls
    if (
        calls is None or calls.produced_call is None or not m.succeeded
        or m.ttft_p50_ms is None or m.server_ttft_p50_ms is None or not m.tpot_p50_ms
        or calls.produced_call * calls.requests / m.succeeded < TOOL_SHARE
        or m.server_ttft_p50_ms >= m.ttft_p50_ms
    ):  # fmt: skip
        return None
    held = (m.ttft_p50_ms - m.server_ttft_p50_ms) / m.tpot_p50_ms
    return held if held >= HELD_MIN else None


def saturated(m: Measurement, cap: int | None) -> bool:
    """Whether the running requests fill the clients (or the cap) and none wait; False where a
    figure is unknown."""
    wanted = demand(m)
    if not wanted or m.mean_running is None or m.mean_waiting is None:
        return False
    return m.mean_waiting < WAITING and m.mean_running >= SATURATED * min(wanted, cap or wanted)


def step_held(m: Measurement, cap: int | None) -> float | None:
    """Decode steps a new request's prompt waits in: the engine's TTFT p50 in TPOT p50, where
    the running requests fill the clients (or the cap) and none wait. Its prompt is then
    computed inside decode steps, so its first token follows the step time. None off that
    state."""
    if not saturated(m, cap) or m.server_ttft_p50_ms is None or not m.tpot_p50_ms:
        return None
    return m.server_ttft_p50_ms / m.tpot_p50_ms


def _prompt_change(weights: float | None, total: float) -> float:
    """The relative change of a prompt computed inside decode steps, from the step's ``total``
    change and the part of it 8-bit ``weights`` make: that part counts in full where the GPU
    computes in fp8 (the prompt's matrix work gets faster too), the rest about half."""
    context = analysis()
    options = context.answer_options
    capability = options.capability if options is not None else context.gpu_capability
    if weights is None or capability is None or capability < NATIVE_FP8:
        return KV_ONLY_HOLD * total
    return weights + KV_ONLY_HOLD * (total - weights)


def _room(situation: Situation, cap: int | None) -> _Room | None:
    """The KV-bound route: KV memory short (``kv_capacity`` fired) with the peak of running
    requests under the cap (the settings', else the engine's ``cap``)."""
    m = situation.measurement
    cap = situation.settings.max_concurrent_requests or cap
    if (
        not situation.fired("kv_capacity") or cap is None or not m.peak_running
        or m.peak_running >= cap or not m.peak_kv_usage or not m.kv_capacity_tokens
        or not m.kv_available_gib or demand(m) is None
    ):  # fmt: skip
        return None
    return _Room(
        running=m.peak_running,
        tokens=m.peak_kv_usage * m.kv_capacity_tokens / m.peak_running,
        bytes_per_token=m.kv_available_gib * 2**30 / m.kv_capacity_tokens,
        cap=cap,
    )


@dataclass(frozen=True, slots=True)
class _Step:
    """This run's decode step: milliseconds, mean batch, the bytes it reads, its prompt part
    and the rate its decode part reached."""

    ms: float
    batch: float
    weights: float
    kv: float
    prompt: float
    rate: float
    label: str

    def change(self, batch: float, weights: float, kv: float, prompt: float) -> tuple[float, float]:
        """The relative TPOT and req/s of a step at ``batch`` reading these bytes."""
        after = prompt + (weights + kv) / self.rate
        return after / self.ms - 1, (batch / after) / (self.batch / self.ms) - 1


def _step(situation: Situation, *, linear_bytes: int | None, bounded: bool) -> _Step | None:
    """The run's step, None where the model does not hold. The ceilings count linear layers at
    16 bits even where the run served them as on-the-fly fp8, so those halve here."""
    m = situation.measurement
    c = m.ceilings
    if (
        c is None or not m.step_ms or c.weight_bytes_per_step is None
        or c.kv_bytes_per_sequence is None or not c.avg_running_batch
        or m.spec_acceptance_length is not None or c.moe_experts is not None
        or situation.diagnosis.primary == "load_limited"
    ):  # fmt: skip
        return None
    weights = float(c.weight_bytes_per_step)
    if situation.settings.quantization in ("fp8", *ONLINE_FP8):
        if linear_bytes is None:
            return None
        weights -= linear_bytes / 2
    share, label = m.prompt_work_share, ""
    if share is None:
        share = upper_share(m, weights) if bounded else None
        if share is None:
            return None
        label = f", prompt share at most {share:.0%}"
    if share > PROMPT_SHARE:
        return None
    step, batch = m.step_ms, c.avg_running_batch
    kv = c.kv_bytes_per_sequence * batch
    prompt = step * share
    return _Step(step, batch, weights, kv, prompt, (weights + kv) / (step - prompt), label)


def _precision(
    situation: Situation,
    step: _Step,
    *,
    linear_bytes: int | None,
    halve_kv: bool,
    cap: int | None,
    room: _Room | None,
) -> Prediction:
    """A precision change's prediction: at this batch, or on the KV-bound ``room`` at the
    batch the freed memory holds (``linear_bytes`` set: the weights halve). Client TTFT moves by
    the parser's held decode steps, and by the steps a saturated closed loop holds the prompt in
    or, on that route, by the queued part of the engine's TTFT (``queue_share`` is measured on
    the mean and applied to the p50: an approximation)."""
    m = situation.measurement
    batch, kv, prompt, queued = step.batch, step.kv, step.prompt, 0.0
    where, waits = "at this cap", "no request waits for KV space"
    if room is not None:
        pool = m.kv_capacity_tokens or 0.0
        if linear_bytes is not None:
            pool += linear_bytes / 2 / room.bytes_per_token
        if halve_kv:
            pool *= 2
        wanted = demand(m) or 0
        fits = math.floor(FULL * pool / room.tokens)
        running = max(room.running, min(wanted, cap or room.cap, fits))
        scale = running / room.running
        batch, kv, prompt = batch * scale, kv * scale, prompt * scale
        where = f"with about {running:.0f} running in the larger KV pool"
        if m.queue_share is not None and m.server_ttft_p50_ms is not None:
            queued = m.queue_share * m.server_ttft_p50_ms
            if running < wanted:
                queued *= (running - room.running) / (wanted - room.running)
                waits = "fewer requests wait for KV space"
    weights = step.weights - (linear_bytes / 2 if linear_bytes is not None else 0)
    tpot, req_s = step.change(batch, weights, kv / 2 if halve_kv else kv, prompt)
    held, riding = held_steps(m), step_held(m, cap)
    if riding is not None:
        queued = 0.0  # the engine's TTFT, its queued part included, follows the step
    by_weights = step.change(batch, weights, kv, prompt)[0] if linear_bytes is not None else None
    ttft, why = None, []
    if m.ttft_p50_ms and (held is not None or riding is not None or queued):
        decode = held * (m.tpot_p50_ms or 0.0) * tpot if held is not None else 0.0
        if riding is not None:
            decode += riding * (m.tpot_p50_ms or 0.0) * _prompt_change(by_weights, tpot)
        ttft = (decode - queued) / m.ttft_p50_ms
        if held is not None:
            why.append(f"the first content waits {held:.0f} decode steps for the tool-call parser")
        if riding is not None:
            why.append(f"with every client running, a prompt waits about {riding:.0f} decode steps")
        if queued:
            why.append(waits)
    return Prediction(
        tpot=tpot,
        req_s=req_s,
        text=f"predicted: TPOT {tpot:+.0%}, req/s {req_s:+.0%} {where} "
        f"(memory traffic of this run{step.label})"
        + (f", TTFT {ttft:+.0%} ({'; '.join(why)})" if ttft is not None else ""),
        ttft=ttft,
    )


def predict(
    situation: Situation,
    proposed: Settings,
    *,
    entry: str,
    linear_bytes: int | None,
    cap: int | None = None,
) -> Prediction | None:
    """The predicted change of moving this run to ``proposed`` by ``entry``; None where the
    model does not hold or the entry is not one it models. ``cap`` is the engine's concurrency
    cap where the settings leave it unset."""
    if entry in PRECISION:
        return predict_stack(situation, [(entry, proposed)], linear_bytes=linear_bytes, cap=cap)
    if entry not in CAP_CHANGES:
        return None
    step = _step(situation, linear_bytes=linear_bytes, bounded=False)
    if step is None:
        return None
    new_cap = proposed.max_concurrent_requests
    if new_cap is None:
        return None
    batch = min(new_cap, demand(situation.measurement) or step.batch)
    scale = batch / step.batch
    tpot, req_s = step.change(batch, step.weights, step.kv * scale, step.prompt * scale)
    return Prediction(
        tpot=tpot,
        req_s=req_s,
        text=f"predicted: TPOT {tpot:+.0%}, req/s {req_s:+.0%} at cap {new_cap} "
        f"(memory traffic of this run{step.label})",
    )


def predict_stack(
    situation: Situation,
    members: Sequence[tuple[str, Settings]],
    *,
    linear_bytes: int | None,
    cap: int | None,
) -> Prediction | None:
    """The joint prediction of changes measured in one step: the bytes every precision member
    removes, and on a KV-bound run the batch the memory they free holds. Members the model does
    not cover add nothing; None without a precision member or where the model does not hold."""
    names = {name for name, _ in members}
    if not names & set(PRECISION):
        return None
    weights, halve = "fp8-weights" in names, bool(names & set(KV_PRECISION))
    if weights and linear_bytes is None:
        return None
    room = _room(situation, cap)
    if halve and situation.fired("kv_capacity") and room is None:
        return None  # KV memory short with no room under the cap: the model does not see it
    bounded = bool(names & set(BOUNDED)) or room is not None
    step = _step(situation, linear_bytes=linear_bytes, bounded=bounded)
    if step is None:
        return None
    proposed = next(
        (s.max_concurrent_requests for _, s in members if s.max_concurrent_requests), None
    )
    return _precision(
        situation, step, linear_bytes=linear_bytes if weights else None, halve_kv=halve,
        cap=proposed or cap, room=room,
    )  # fmt: skip


def predict_pair(
    situation: Situation, *, linear_bytes: int | None, cap: int | None
) -> Prediction | None:
    """fp8 weights and fp8 KV in one step: on a KV-bound run by the batch the freed memory
    holds; under TPOT with no KV pressure by the step bytes alone; None off those routes."""
    if _room(situation, cap) is not None:
        kv = KV_PRECISION[0]
    elif analysis().goal == "tpot" and not kv_held_back(situation):
        kv = KV_PRECISION[1]
    else:
        return None
    pair = [(name, situation.settings) for name in ("fp8-weights", kv)]
    return predict_stack(situation, pair, linear_bytes=linear_bytes, cap=cap)
