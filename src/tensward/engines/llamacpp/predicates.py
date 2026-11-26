"""llama.cpp's playbook predicates: the conditions and changes its entries name as
"llamacpp:<name>". They read the run's start-up log, its placement block (regime, fit and
next steps) and its probes; none of them recomputes what ``placement`` already decided."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...classify import QUEUED_PREFILL
from ...context import current as analysis
from ...families import (
    MTP_HYBRID_SLOTS_FIXED_BUILD,
    drafts_mtp,
    host_experts_unsafe,
    mtp_slots_crash,
)
from ...fit import CHARS_PER_TOKEN
from ...formats.gguf import kv_bytes_per_token
from ...levers import Speculation
from ...measurement import Measurement, nearest_rank
from ...platforms.cpu import glibc_version, transparent_hugepages
from ...playbook import CENSUS_CONDITION, Gate, HeldBack, Saving
from ...playbook_common import (
    MIN_SHARED_PREFIX_TOKENS,
    MIN_SHARING_PROMPTS,
    declared_longest,
    facts_hold,
    sampled_memory,
    shared_lengths,
    trimmed_max_context_len,
)
from ...pressure import demand
from ...settings import Settings
from ...thresholds import PREFILL_TIME_SHARE
from .capabilities import DRAFT_MTP, NGRAM_MOD
from .command import DEFAULT_BATCH, Q8, apply_flag, known_build
from .flags import GLIBC_TUNABLES, OP_OFFLOAD_MIN_BATCH, build_number
from .placement import (
    DEVICE_MARGIN,
    ESTIMATE_LOW,
    HOST_RESERVE,
    LONG_PROMPT_TOKENS,
    MAJOR_FAULTS_TIGHT,
    MIB,
    MIN_POOL_TOKENS,
    POOL_ROUND,
    Q8_KV_SAVING,
    WIRED_LIMIT,
    DecodeModel,
    Fit2,
    Placement,
    _plan,
    active_share,
    device_kv,
    experts_to_host,
    usable_free,
)
from .probes import prefill_work
from .startup import Startup, read_startup

if TYPE_CHECKING:
    from ...playbook import Situation
    from ..protocol import Engine

SLOT_STEP = 2  # the slots ladder doubles
UBATCH_LADDER = (512, 2048, 4096, 8192)
# 8192 for experts on the host only: GLM-5.3-Flash Q2 gained x1.65-1.73 over 2048 on prompts over
# 2k (decode -1.6% to -4.2%); 16384 gained nothing over 8192. Other launches stop at 4096
UBATCH_MOST = 4096
# Offloaded dense Qwen3-32B: -ub 4096 added 4% prefill over 2048 but pushed layers off the card
# and cost decode 8-10%
DENSE_UBATCH_MOST = 2048
OFFLOADED = ("dense_offload", "expert_offload")
# K and V head sizes the CUDA flash-attention kernels take unconditionally
# (ggml-cuda/fattn.cu:575-620 at build 11669).
FLASH_HEAD_DIMS = frozenset({40, 64, 72, 80, 96, 112, 128, 256})
SWA_HIT_SHARE = 0.5  # cache hits under this share of the prefix the prompts share
ROOM_FLAGS = {
    "max_context_len": "--ctx-size",
    "max_concurrent_requests": "--parallel",
    "ubatch": "--ubatch-size",
    "kv_cache_precision": "--cache-type-k",
}


@lru_cache(maxsize=4)
def _log(path: str, mtime_ns: int) -> str:
    return Path(path).read_text("utf-8", errors="replace")


@dataclass(frozen=True, slots=True)
class View:
    """What an entry reads of the run: the start-up log, the placement block and the probes."""

    started: Startup
    block: Mapping[str, Any]
    probes: Mapping[str, Any]

    @property
    def regime(self) -> Mapping[str, Any]:
        return self.block.get("regime") or {}

    @property
    def placed(self) -> Mapping[str, Any]:
        return self.block.get("placement") or {}

    @property
    def fit(self) -> Mapping[str, Any]:
        return self.block.get("fit") or {}

    @property
    def environment(self) -> Mapping[str, Any]:
        return self.block.get("environment") or {}

    @property
    def kind(self) -> str | None:
        return self.regime.get("placement")

    @property
    def offloaded(self) -> bool:
        return self.kind in OFFLOADED

    def step(self, *levers: str) -> Mapping[str, Any] | None:
        return next((s for s in self.block.get("next_steps") or () if s["lever"] in levers), None)

    @property
    def room(self) -> int | None:
        """Device bytes the launch can still take: free memory less what it needs and the
        margin; on a unified-memory Mac also what the system memory has left."""
        fit = self.fit
        if not fit:
            return None
        room = fit["device_free"] - fit["device_need"]
        if _unified(self):
            if self.regime.get("host_memory_tight"):
                return 0
            room = min(room, fit["host_free"] - fit["device_need"] - fit["host_need"])
        return int(room)


def _unified(seen: View) -> bool:
    return seen.environment.get("platform") == "apple"


def _mac(seen: View) -> bool:
    return seen.environment.get("os_family") == "macos"


def view(situation: Situation) -> View | None:
    context = analysis()
    run_dir = context.this_run()
    if run_dir is None:
        return None
    path = run_dir / "server.log"
    try:
        text = _log(str(path), path.stat().st_mtime_ns)
    except OSError:
        return None
    return View(read_startup(text), context.placement or {}, context.probes or {})


def _longest_request(situation: Situation, seen: View) -> int:
    measured = seen.fit.get("need_tokens") or (
        (situation.measurement.max_prompt_tokens or 0) + situation.facts.output_tokens
    )
    return max(measured, declared_longest() or 0)


def _kv_token_bytes(seen: View, anatomy_layers: int) -> float:
    """Device bytes one more pool token costs."""
    if not seen.fit or not seen.fit["pool_tokens"]:
        return 0.0
    on_device = device_kv(Placement(**seen.placed), Fit2(**seen.fit), anatomy_layers)
    return float(on_device / seen.fit["pool_tokens"])


def _layers() -> int:
    anatomy = analysis().anatomy
    return (anatomy.layers if anatomy else 0) or 1


# --- slots ------------------------------------------------------------------------------


def _slots(situation: Situation, started: Startup) -> int | None:
    """The slots the run had: the setting, else what llama-server chose at start-up."""
    return situation.settings.max_concurrent_requests or started.slots


def _next_slots(
    situation: Situation,
    seen: View,
    cost: tuple[float, float] | None = None,
    pool: int | None = None,
) -> int | None:
    """The next rung of the slots ladder (doubling, never past the clients) that the pool can
    hold: a unified pool must keep room for each slot's longest request; a split pool grows
    with the slots, so the device must have room for the cells. ``cost`` and ``pool`` replace
    the device bytes per token, the free bytes and the pool size of the launch."""
    started = seen.started
    slots = _slots(situation, started)
    wanted = demand(situation.measurement)
    if not slots or not wanted or not started.slot_ctx or not started.ctx_total:
        return None
    target = min(slots * SLOT_STEP, wanted)
    if target <= slots:
        return None
    need = _longest_request(situation, seen)
    if started.kv_unified:
        target = min(target, (pool or started.ctx_total) // max(need, 1))
    else:
        per_token, room = cost or (_kv_token_bytes(seen, _layers()), seen.room)
        if per_token <= 0 or room is None:
            return None
        affordable = slots + int(room // (started.slot_ctx * per_token))
        target = min(target, affordable)
    return target if target > slots and started.slot_ctx >= need else None


def slots_applies(situation: Situation, engine: Engine) -> str | None:
    """More slots serve more requests at once and each stream generates a bit slower: not for
    a per-stream speed goal, nor for the first token of a single client."""
    seen = view(situation)
    if seen is None or not (slots := _slots(situation, seen.started)):
        return None
    m = situation.measurement
    full = (m.peak_waiting or 0) > 0 and (m.peak_running or 0) >= slots
    clients = demand(m) or 0
    goal = analysis().goal
    if goal == "tpot" or (goal == "ttft" and clients <= 1):
        return None
    if not (full or clients > slots) or (target := _next_slots(situation, seen)) is None:
        return None
    waiting = f"{m.peak_waiting:.0f} requests were deferred with all {slots} slots busy"
    return (
        (waiting if full else f"{clients} requests were in flight on {slots} slots")
        + f"; {target} slots fit the KV pool at {seen.started.slot_ctx} tokens each: more "
        "requests at once; each stream generates a bit slower"
    )


def with_more_slots(situation: Situation, engine: Engine) -> Settings:
    seen = view(situation)
    settings = situation.settings
    if seen is None or (target := _next_slots(situation, seen)) is None:
        return settings
    settings = dataclasses.replace(settings, max_concurrent_requests=target)
    if seen.started.kv_unified:
        extra = {**settings.extra_args, "--kv-unified": True}
        return dataclasses.replace(settings, extra_args=extra)
    return engine.with_lever(settings, "max_context_len", seen.started.slot_ctx)


def _trimmed_slots(situation: Situation, seen: View) -> tuple[int, int] | None:
    """For a split pool whose requests queued with every slot busy: the per-slot context the
    longest request needs (rounded up as fit-context does) and the slots that many cells of
    today's pool hold, doubled at most and never past the clients; None when that is no more
    slots than today, or than llamacpp-slots reaches at today's context."""
    started, m = seen.started, situation.measurement
    slots = _slots(situation, started)
    if started.kv_unified is not False or not slots or not started.slot_ctx:
        return None
    queued = situation.fired("queueing") and (m.peak_running or 0) >= slots
    if m.max_prompt_tokens is None or not queued:
        return None
    trimmed, wanted = trimmed_max_context_len(m, situation.facts), demand(m)
    if not wanted or trimmed >= started.slot_ctx:
        return None
    target = min(slots * SLOT_STEP, wanted, started.slot_ctx * slots // trimmed)
    if target <= slots or (_next_slots(situation, seen) or 0) >= target:
        return None  # llamacpp-slots reaches as many without the trim
    return trimmed, target


def trim_for_slots_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    if seen is None or (found := _trimmed_slots(situation, seen)) is None:
        return None
    (trimmed, target), m = found, situation.measurement
    slots = _slots(situation, seen.started)
    return (
        f"{m.peak_waiting:.0f} requests were deferred with all {slots} slots busy, and each slot "
        f"reserves {seen.started.slot_ctx} tokens while the longest request needs {trimmed}; "
        f"-c {trimmed * target} -np {target} spends no more KV cells on {target} slots; requests "
        f"longer than {trimmed} tokens would be refused"
    )


def with_trimmed_slots(situation: Situation, engine: Engine) -> Settings:
    seen = view(situation)
    if seen is None or (found := _trimmed_slots(situation, seen)) is None:
        return situation.settings
    trimmed, target = found
    settings = dataclasses.replace(situation.settings, max_concurrent_requests=target)
    return engine.with_lever(settings, "max_context_len", trimmed)


POOL_HEADROOM = 1.2  # the shared pool over the most context any measured run had in flight
# llama.cpp's compute buffer is the peak of the tensors live at once (ggml-alloc reuses memory
# between graph nodes), not a sum of parts. Under --kv-unified the pooled indexer reserves
# n_pool_max = cells / kpool x slots rows for every -ub token (b12246 llama-memory-hybrid-idx.cpp),
# so past some size that step is the peak and grows with cells x slots; below it the rest of the
# graph is, as today's buffer. Fit to two real b12246 starts of Qwen3.8-Flash-Next on an H100 at
# -ub 512: 8 slots in 137,728 cells 1,256 MiB and 16 slots in 180,224 cells (--n-cpu-moe 5)
# 3,078 MiB, against 1,008 MiB on 4 split slots of 32,768. Used without a shared-pool start in the
# project's logs; with one, its own bytes a cell per -ub token per slot over the same base.
POOL_COMPUTE_BYTES = 2.09  # a cell per -ub token per slot
POOL_COMPUTE_BASE = 130 * MIB
POOL_MOST_EXPERT_LAYERS = 3  # more blocks of experts on the host than this cost decode too much
# A shared pool keeps at least this much of the card allocatable at its predicted peak: the run's
# own allocatable free at its sampled peak less the pool's extra bytes. Without that reading, the
# fit's counted free less POOL_UNSEEN: the driver's reserve CUDA never allocates (480 MiB on an
# H100, 471 on an L4) and what CUDA takes after the start (about 200 MiB on the rc7 H100 smoke).
POOL_SPARE = 1024 * MIB
POOL_UNSEEN = 700 * MIB
# Research, not measured by Tensward (the kv-q8-one-slot row): with several slots at 16k to 32k of
# context a q8_0 cache decodes at x0.56 to x0.85 of f16's speed, and prefills 1 to 9% slower.
Q8_DECODE_SPEED = (0.56, 0.85)
Q8_PREFILL_SLOWER = (0.01, 0.09)
# Measured on the same model (by model_type), from the KB record's evidence: q8_0 against f16 at
# 16 slots in one shared pool of 163,840 on an H100, TPOT p50 115 against 107 to 112 ms
# (Qwen3.8-Flash-Next, box B of the H2H-SK head-to-head, 24 Nov).
Q8_DECODE_MEASURED = {
    "qwen4exp": ((0.93, 0.97), "Qwen3.8-Flash-Next at 16 slots in 163,840 tokens on an H100"),
}
STEP_ALONE_SHARE = 0.1  # the decode step alone: this low a share of the requests' token times
STEP_ALONE_MIN_TOKENS = 20  # requests with fewer output tokens than this give no token time


@dataclass(frozen=True, slots=True)
class Forecast:
    """A rough estimate of one shared-pool variant from this run's own numbers: the decode step
    at its streams, the prompt time a computed token, and requests a second; (low, high) pairs."""

    streams: int
    step_now: float  # decode ms a step today, at ``running`` streams
    running: float
    step: tuple[float, float]
    prefill_now: float  # prompt ms a computed token today
    prefill: tuple[float, float]
    req_now: float
    req: tuple[float, float]
    host_ms: float = 0.0  # decode ms a step the moved experts add
    copy_ms: tuple[float, float] = (0.0, 0.0)  # ms each prompt batch spends copying them


@dataclass(frozen=True, slots=True)
class SharedPool:
    slots: int
    pool: int
    peak: int
    longest: int
    extra: int  # device bytes over today's launch, after the moved experts are taken off
    cells: int  # device bytes of the cells over today's pool
    state: int  # device bytes of the added slots' recurrent state
    compute: int  # device bytes the compute buffer grows by
    measured_compute: bool  # the compute growth comes from a shared-pool start in this project
    clients: int
    bound: int = 0  # the most tokens the workload can have in flight on these slots
    estimated: int = 0  # prompts whose tokens come from their length
    q8: bool = False  # the pool is a q8_0 KV cache
    host_layers: int = 0  # blocks whose experts are on the host today (--n-cpu-moe)
    more: int = 0  # blocks whose experts the offer moves to the host as well
    freed: int = 0  # device bytes those experts free
    forecast: Forecast | None = None
    source: str = "bound"  # the term that sized the pool: bound, measured, held, longest or today
    sized: str = "workload"  # "workload": its worst case; "measured": the most measured + 20%
    held: int = 0  # the most the ``slots`` largest requests in flight held, at their output caps
    left: int | None = None  # device bytes still allocatable at the predicted peak, when known
    back: Spent | None = None  # the layers once the cells a smaller pool frees are spent
    saved_ms: float = 0.0  # decode milliseconds a token the moves back save, estimated
    alone_ms: float = 0.0  # of the moves back, what the launch's own room already buys


@lru_cache(maxsize=16)
def _rows_of(path: str, mtime_ns: int) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for line in Path(path).read_text("utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return tuple(rows)


def _rows(run: Path) -> tuple[Mapping[str, Any], ...]:
    path = run / "requests.jsonl"
    try:
        return _rows_of(str(path), path.stat().st_mtime_ns)
    except OSError:
        return ()


def _measured_runs() -> list[Path]:
    """This run and the project's runs before it."""
    context = analysis()
    here = context.this_run()
    if here is None:
        return []
    runs = context.project / "runs" if context.project is not None else None
    if runs is None or here.parent.resolve() != runs.resolve():
        return [here]
    return [*(p for p in sorted(runs.iterdir()) if p.is_dir() and p.name < here.name), here]


def _peak_tokens(situation: Situation) -> int:
    """The most prompt and output tokens in flight at once in this run or any earlier one."""
    peaks = [situation.measurement.peak_context_tokens or 0]
    for run in _measured_runs()[:-1]:
        try:
            value = json.loads((run / "metrics.json").read_text("utf-8")).get("peak_context_tokens")
        except (OSError, ValueError):
            continue
        if isinstance(value, int):
            peaks.append(value)
    return max(peaks)


def _session_needs(situation: Situation) -> tuple[tuple[int, bool], ...]:
    """The most cells each session of the workload file (each record outside a session) holds at
    once: its longest turn's prompt plus that turn's output cap, at most the model's context;
    longest first, each with whether its prompt was estimated. Prompt tokens are as a run of this
    project counted them, else from the text's length at the characters a token those runs show."""
    facts = situation.facts
    texts: tuple[str, ...] = getattr(facts, "prompts", ()) or ()
    if not texts:
        return ()
    counted: dict[int, int] = {}
    for run in _measured_runs():
        for row in _rows(run):
            index, tokens = row.get("prompt_index"), row.get("prompt_tokens")
            if isinstance(index, int) and 0 <= index < len(texts) and isinstance(tokens, int):
                if tokens > 0:
                    counted[index] = tokens
    chars, tokens = sum(len(texts[i]) for i in counted), sum(counted.values())
    per_token = chars / tokens if chars and tokens else CHARS_PER_TOKEN
    caps: tuple[int | None, ...] = getattr(facts, "max_outputs", ()) or ()
    sessions: tuple[str | None, ...] = getattr(facts, "sessions", ()) or ()
    limit = getattr(facts, "context_limit", 0) or 0
    output = getattr(facts, "output_tokens", 0) or 0
    most: dict[object, tuple[int, bool]] = {}
    for index, text in enumerate(texts):
        prompt = counted.get(index) or -(-len(text) // max(per_token, 1e-9))
        cap = (caps[index] if index < len(caps) else None) or output
        need = int(prompt + cap)
        need = min(need, limit) if limit else need
        key = sessions[index] if index < len(sessions) and sessions[index] else index
        if need > most.get(key, (0, False))[0]:
            most[key] = (need, index not in counted)
    return tuple(sorted(most.values(), reverse=True))


def _held_at_caps(situation: Situation, count: int) -> int:
    """The most the ``count`` largest requests in flight at once held in this run or an earlier
    one, each at its prompt plus its output cap, from when it was offered to when it ended."""
    facts = situation.facts
    caps: tuple[int | None, ...] = getattr(facts, "max_outputs", ()) or ()
    output = getattr(facts, "output_tokens", 0) or 0
    most = 0
    for run in _measured_runs():
        events: list[tuple[int, int, int]] = []
        needs: dict[int, int] = {}
        for at, row in enumerate(_rows(run)):
            index, prompt = row.get("prompt_index"), row.get("prompt_tokens")
            start, end = row.get("dispatch_ns"), row.get("terminal_ns")
            if not (isinstance(prompt, int) and isinstance(start, int) and isinstance(end, int)):
                continue
            known = isinstance(index, int) and 0 <= index < len(caps) and caps[index]
            needs[at] = prompt + (caps[index] if known else output)  # type: ignore[index,operator]
            events += [(start, 1, at), (end, 0, at)]  # at equal times an end comes first
        live: set[int] = set()
        for _, starts, at in sorted(events):
            if not starts:
                live.discard(at)
                continue
            live.add(at)
            most = max(most, sum(sorted((needs[i] for i in live), reverse=True)[:count]))
    return most


def _shared_starts(seen: View) -> list[tuple[int, int]]:
    """(pool cells x -ub x slots, device compute bytes) of every shared-pool start in this run's
    and the earlier runs' logs."""
    starts = [seen.started]
    for run in _measured_runs()[:-1]:
        path = run / "server.log"
        try:
            starts.append(read_startup(_log(str(path), path.stat().st_mtime_ns)))
        except OSError:
            continue
    return sorted(
        {
            (s.ctx_total * s.ubatch * s.slots, s.bytes_of("compute", "device"))
            for s in starts
            if s.kv_unified
            and s.ctx_total
            and s.ubatch
            and s.slots
            and s.bytes_of("compute", "device")
        }
    )


def _pooled_bytes(cells: int, slots: int, ubatch: int) -> float:
    """The compute buffer the pooled-indexer step of one shared pool needs."""
    return POOL_COMPUTE_BASE + POOL_COMPUTE_BYTES * cells * slots * ubatch


def _pool_compute(
    seen: View, pool: int, slots: int, points: list[tuple[int, int]]
) -> tuple[int, bool]:
    """The device compute buffer one shared pool of ``pool`` cells for ``slots`` needs at today's
    -ub, and whether a logged shared-pool start gave it, never under today's buffer: past the
    largest logged start, the line through the two farthest apart, else the largest start's own
    bytes a cell per -ub token per slot over the base; one start the same; else the pooled step
    of ``_pooled_bytes``."""
    started = seen.started
    ubatch = started.ubatch or UBATCH_LADDER[0]
    size = pool * ubatch * slots
    today = started.bytes_of("compute", "device") or 0
    if not points:
        return int(max(today, _pooled_bytes(pool, slots, ubatch))), False
    (low, low_bytes), (high, high_bytes) = points[0], points[-1]
    if high > low and high_bytes > low_bytes:
        slope = (high_bytes - low_bytes) / (high - low)
        return int(max(today, high_bytes + slope * max(size - high, 0))), True
    per = max(high_bytes - POOL_COMPUTE_BASE, 0) / high
    return int(max(today, high_bytes, POOL_COMPUTE_BASE + per * size)), True


def _host_room(seen: View) -> int | None:
    fit = seen.fit
    if not fit or _unified(seen):
        return None
    return int(fit["host_free"] - fit["host_need"] - HOST_RESERVE)


def _checkpoint_bytes(seen: View, added: int) -> int:
    """Host bytes the added slots' context checkpoints may take (llama.cpp keeps up to its
    --ctx-checkpoints of each recurrent slot's state)."""
    anatomy = analysis().anatomy
    state = (anatomy.state_bytes_per_seq or 0) if anatomy is not None else 0
    return state * (seen.started.checkpoints or 0) * added


def _movable_experts(seen: View) -> bool:
    """Whether more blocks' experts can go to the host with --n-cpu-moe: a mixture of experts
    whose blocks are all on the device and whose placement is a fixed command, not --fit's."""
    anatomy = analysis().anatomy
    if anatomy is None or not any(anatomy.expert_layer_bytes) or _unified(seen):
        return False
    if seen.placed.get("host_layers") or "tensor_overrides" in seen.started.fit_changed:
        return False
    return not host_experts_unsafe(anatomy, _build_number())


def _q8_pool(situation: Situation, seen: View) -> bool:
    """Whether the pool may be a q8_0 KV cache: f16 today, flash attention not off, the cache
    on the device and the build's --help lists the cache-type flags."""
    if (seen.started.type_k, seen.started.type_v) != ("f16", "f16"):
        return False
    if _q8_needs_flash_attn(situation) or seen.placed.get("kv_on_host"):
        return False
    return not not_in_help("--cache-type-k", "--cache-type-v")(situation)


@dataclass(frozen=True, slots=True)
class _Profile:
    """This run's per-request work: computed prompt tokens today and with a slot per session,
    output tokens, prompt ms a computed token, decode ms a step alone (with no prompt batch in
    it) and the streams decoding."""

    computed: float
    session_computed: float
    output: float
    prefill_ms: float
    step_ms: float
    running: float
    req_s: float


def _profile(situation: Situation) -> _Profile | None:
    """From this run's answered requests: a slot per session keeps each turn's context, so a
    follow-up turn computes only what its prompt adds to the turn before it (at most; the
    measured cache counts when it kept more)."""
    m = situation.measurement
    here = analysis().this_run()
    if here is None or not m.request_throughput or not m.tpot_p50_ms or not m.mean_running:
        return None
    rows = [
        row
        for row in _rows(here)
        if row.get("outcome") == "success"
        and isinstance(row.get("timings"), Mapping)
        and "prompt_ms" in row["timings"]
        and isinstance(row.get("prompt_tokens"), int)
    ]
    computed = sum(row["timings"].get("prompt_n", 0) for row in rows)
    if not rows or not computed:
        return None
    sessions: tuple[str | None, ...] = getattr(situation.facts, "sessions", ()) or ()
    before: dict[str, Mapping[str, Any]] = {}
    ideal = 0
    for row in sorted(rows, key=lambda row: row.get("dispatch_ns", 0)):
        index = row.get("prompt_index", -1)
        key = sessions[index] if 0 <= index < len(sessions) else None
        prompt, now = row["prompt_tokens"], row["timings"].get("prompt_n", 0)
        last = before.get(key) if key else None
        if last is not None and last.get("prompt_index", -1) < index:
            kept = last["prompt_tokens"] + (last.get("committed_output_tokens") or 0)
            now = min(now, max(prompt - kept, 0))
        ideal += now
        if key:
            before[key] = row
    prompt_ms = sum(row["timings"]["prompt_ms"] for row in rows)
    output = sum(row["timings"].get("predicted_n", 0) for row in rows)
    return _Profile(
        computed / len(rows), ideal / len(rows), output / len(rows), prompt_ms / computed,
        _step_alone(rows) or m.tpot_p50_ms, m.mean_running, m.request_throughput,
    )  # fmt: skip


def _step_alone(rows: list[Mapping[str, Any]]) -> float | None:
    """The decode step with no other slot's prompt batch in it: the low end of the requests' own
    token times (a request's TPOT counts the prompt batches that ran between its tokens)."""
    times = sorted(
        float(row["timings"]["predicted_per_token_ms"])
        for row in rows
        if (row["timings"].get("predicted_n") or 0) >= STEP_ALONE_MIN_TOKENS
        and isinstance(row["timings"].get("predicted_per_token_ms"), (int, float))
    )
    if len(times) < 1 / STEP_ALONE_SHARE:
        return None
    return times[int(len(times) * STEP_ALONE_SHARE)]


def _q8_speed() -> tuple[tuple[float, float], str | None]:
    """q8_0's decode speed against f16: measured on this model when the KB has it (with where),
    else the research range."""
    anatomy = analysis().anatomy
    found = Q8_DECODE_MEASURED.get(getattr(anatomy, "model_type", None) or "")
    return (found[0], found[1]) if found else (Q8_DECODE_SPEED, None)


def _forecast(situation: Situation, seen: View, found: SharedPool) -> Forecast | None:
    """The placement model's decode model at the variant's streams from today's step alone (more
    experts read a step, the moved experts read from host memory) and each prompt batch copying
    the moved experts to the card: at the link rate the model uses (low) up to all of today's
    batch time spread over today's host experts (high); a q8_0 cache at this model's measured
    speed, else the research range; with fewer slots than clients, the session-cache gain from
    none (low) to the share of the clients the added slots serve (high); requests a second from
    this run's work per request, scaled."""
    anatomy, block = analysis().anatomy, seen.block.get("decode_model")
    profile = _profile(situation)
    if profile is None or anatomy is None or not block or not seen.placed:
        return None
    try:
        model, p = DecodeModel(**block), Placement(**seen.placed)
    except TypeError:  # a run recorded with other fields
        return None
    today = _slots(situation, seen.started) or 1
    streams = min(found.slots, found.clients)
    running = min(float(streams), profile.running * streams / today)
    grows = active_share(anatomy, round(running)) - active_share(anatomy, round(profile.running))
    experts = sum(anatomy.expert_layer_bytes)
    on_host = p.host_expert_bytes
    step = profile.step_ms + grows * (
        (experts - on_host) / (model.bw_dev_gbs * 1e6) + on_host / (model.bw_host_gbs * 1e6)
    )
    read = found.freed * active_share(anatomy, round(running))
    host_ms = read / (model.bw_host_gbs * 1e6) - read / (model.bw_dev_gbs * 1e6)
    step += host_ms
    ubatch = seen.started.ubatch or UBATCH_LADDER[0]
    copy = (0.0, 0.0)
    if found.freed:
        rate = seen.block.get("pcie_gbs") or _stall_rate(situation, model, ubatch)
        low = found.freed / (rate * 1e6) if rate else None
        high = profile.prefill_ms * ubatch * found.freed / on_host if on_host else low
        if low is None or high is None:
            low = high = low or high or 0.0
        copy = (low, max(low, high))
    prefill = tuple(profile.prefill_ms + c / ubatch for c in copy)
    steps = (step, step)
    if found.q8:
        speed = _q8_speed()[0]
        steps = (step / speed[1], step / speed[0])
        prefill = tuple(x * (1 + s) for x, s in zip(prefill, Q8_PREFILL_SLOWER, strict=True))
    share = min((streams - today) / max(found.clients - today, 1), 1.0)
    gain = profile.session_computed - profile.computed
    computed = profile.computed + gain * share
    none = computed if share >= 1.0 else profile.computed

    def per_request(prompt_ms: float, step_ms: float, at: float, tokens: float) -> float:
        return tokens * prompt_ms + profile.output * step_ms / at

    now = per_request(profile.prefill_ms, profile.step_ms, profile.running, profile.computed)
    fast = per_request(prefill[0], steps[0], running, computed)
    slow = per_request(prefill[1], steps[1], running, none)
    return Forecast(
        streams, profile.step_ms, profile.running, steps, profile.prefill_ms,
        (prefill[0], prefill[1]), profile.req_s,
        (profile.req_s * now / slow, profile.req_s * now / fast), host_ms, copy,
    )  # fmt: skip


def _stall_rate(situation: Situation, model: DecodeModel, ubatch: int) -> float | None:
    """The link rate the decode model's prompt-streaming term was made with (a run recorded
    before the placement block kept it): its ubatches over its decode steps and the stall."""
    here, running = analysis().this_run(), situation.measurement.mean_running
    if here is None or not model.stall_ms_per_byte or not running:
        return None
    work = prefill_work(_rows(here), ubatch, running)
    if work is None or not work.steps:
        return None
    return work.ubatches / (model.stall_ms_per_byte * 1e6 * work.steps)


def _variants(situation: Situation, seen: View) -> list[SharedPool]:
    """One pool shared by a slot per client, for requests that queued for a slot, sized to hold
    the most tokens the workload can have in flight (the sessions with the longest turns, one a
    slot, prompt plus output cap) and never under 20% over the most measured in flight, today's
    pool or the longest request, which may use the whole pool. When the device has no room for
    the cells, the slots' state and the larger compute buffer, the ways to make room, each
    priced: a q8_0 KV cache, then (a mixture of experts) the fewest more blocks' experts on the
    host, with a q8_0 cache when they alone do not pay, then fewer slots than clients in a
    smaller pool; and the pool sized from what the runs measured (20% over the most in flight, or
    over the largest requests in flight at their output caps when that is larger) when it is
    smaller, alone or paid the same ways. Ordered by the predicted requests a second at the slow
    end, the way that needs no room alone; empty when the clients are no more than the slots, or
    nothing fits."""
    started, m = seen.started, situation.measurement
    slots, clients = _slots(situation, started), demand(m) or 0
    peak = _peak_tokens(situation)
    if not slots or not started.ctx_total or not peak:
        return []
    if clients <= slots:
        freeing = _freeing_pool(situation, seen, slots, clients, peak)
        return [freeing] if freeing is not None else []
    waited = (m.peak_waiting or 0) > 0 and (m.peak_running or 0) >= slots
    if not (waited or situation.fired("queueing")):
        return []
    longest = _longest_request(situation, seen)
    room, host, points = seen.room, _host_room(seen), _shared_starts(seen)
    usable = _usable(seen)
    budget = _pool_budget(room, usable)
    needs = _session_needs(situation)
    limit = getattr(situation.facts, "context_limit", 0) or 0
    anatomy = analysis().anatomy
    state_bytes = (anatomy.state_bytes_per_seq or 0) if anatomy is not None else 0
    per_token = _kv_token_bytes(seen, _layers())
    held = 0

    def build(count: int, q8: bool, sized: str = "workload") -> SharedPool:
        bound = sum(need for need, _ in needs[:count])
        bound = min(bound, limit * count) if limit else bound
        estimated = sum(guessed for _, guessed in needs[:count])
        measured = round(peak * POOL_HEADROOM * count / clients)
        terms = {"bound": bound, "measured": measured}
        if sized == "measured":
            nonlocal held
            held = held or _held_at_caps(situation, clients)
            terms = {"measured": measured, "held": round(held * POOL_HEADROOM)}
        terms |= {"longest": longest, "today": started.ctx_total or 0}
        source = max(terms, key=lambda name: terms[name])
        wanted = terms[source]
        pool = -(-wanted // POOL_ROUND) * POOL_ROUND
        cell = per_token * (1 - Q8_KV_SAVING) if q8 else per_token
        cells = int(pool * cell - (started.ctx_total or 0) * per_token)
        compute, measured_compute = _pool_compute(seen, pool, count, points)
        grows = max(compute - started.bytes_of("compute", "device"), 0)
        state = state_bytes * max(count - slots, 0)
        return SharedPool(
            count, pool, peak, longest, cells + state + grows, cells, state, grows,
            measured_compute, clients, bound, estimated if bound else 0, q8, source=source,
            sized=sized, held=held if sized == "measured" else 0,
        )  # fmt: skip

    def fits(found: SharedPool) -> bool:
        if found.extra > 0 and (budget is None or found.extra > budget):
            return False
        host_bytes = _checkpoint_bytes(seen, found.slots - slots) + found.freed
        return host is None or host_bytes <= 0 or host_bytes <= host

    def moved(base: SharedPool) -> SharedPool | None:
        if budget is None or anatomy is None or not _movable_experts(seen):
            return None
        layers = int(seen.placed.get("host_expert_layers") or 0)
        if base.extra <= budget:
            return None  # the device has room: it is the host that does not
        found = experts_to_host(
            anatomy,
            layers,
            base.extra - budget,
            POOL_MOST_EXPERT_LAYERS,
        )
        if found is None:
            return None
        paid = dataclasses.replace(
            base, extra=base.extra - found[1], host_layers=layers, more=found[0], freed=found[1]
        )
        return paid if fits(paid) else None

    def paid(sized: str) -> list[SharedPool]:
        whole = build(clients, False, sized)
        if fits(whole):
            return [whole]
        ways: list[SharedPool] = []
        q8 = build(clients, True, sized) if _q8_pool(situation, seen) else None
        if q8 is not None and fits(q8):
            ways.append(q8)
        experts = moved(whole) or (moved(q8) if q8 is not None and q8 not in ways else None)
        return [*ways, experts] if experts is not None else ways

    whole = build(clients, False)
    if fits(whole):
        left = None if usable is None else usable - whole.extra
        return [dataclasses.replace(whole, forecast=_forecast(situation, seen, whole), left=left)]
    ways = paid("workload")
    if build(clients, False, "measured").pool < whole.pool:
        ways += paid("measured")
    for count in range(clients - 1, slots, -1):
        if fits(fewer := build(count, False)):
            ways.append(fewer)
            break
    ways = [
        dataclasses.replace(
            w,
            forecast=_forecast(situation, seen, w),
            left=None if usable is None else usable - w.extra,
        )
        for w in ways
    ]
    if all(w.forecast is not None for w in ways):
        ways.sort(key=lambda w: -w.forecast.req[0])  # type: ignore[union-attr]
    return ways


def _freeing_pool(
    situation: Situation, seen: View, slots: int, clients: int, peak: int
) -> SharedPool | None:
    """A slot a client and nothing queued, but the split slots reserve more cells than the
    workload holds in flight while experts sit on the host: one shared pool for the same slots,
    sized from the workload's longest sessions (else 20% over the most measured in flight or the
    largest requests in flight at their output caps), never under the longest request, whose
    freed cells the placement model spends on bringing expert layers back to the card."""
    started = seen.started
    if started.kv_unified or seen.kind != "expert_offload" or seen.placed.get("host_layers"):
        return None
    if slots < 2 or not seen.placed.get("host_expert_layers") or not started.ctx_total:
        return None
    needs = _session_needs(situation)
    limit = getattr(situation.facts, "context_limit", 0) or 0
    bound = sum(need for need, _ in needs[:slots])
    bound = min(bound, limit * slots) if limit else bound
    longest = _longest_request(situation, seen)
    measured = round(peak * POOL_HEADROOM)
    held = _held_at_caps(situation, slots)
    sized, terms = "workload", {"bound": bound, "measured": measured, "longest": longest}
    if not bound or bound >= started.ctx_total:
        terms = {"measured": measured, "held": round(held * POOL_HEADROOM), "longest": longest}
        sized = "measured"
    source = max(terms, key=lambda name: terms[name])
    pool = -(-terms[source] // POOL_ROUND) * POOL_ROUND
    per_token = _kv_token_bytes(seen, _layers())
    if pool >= started.ctx_total or per_token <= 0:
        return None
    freed = int((started.ctx_total - pool) * per_token)
    compute, measured_compute = _pool_compute(seen, pool, slots, _shared_starts(seen))
    grows = max(compute - started.bytes_of("compute", "device"), 0)
    # the moves back keep POOL_SPARE allocatable, as the queueing pool does
    budget, room = _pool_budget(seen.room, _usable(seen)), seen.room or 0
    short = min((budget if budget is not None else room) - room, 0)
    spent = _spend(seen, freed - grows + short, pool / started.ctx_total)
    if spent is None or spent[1].host_layers != spent[0].host_layers:
        return None
    base, back = spent
    if back.host_expert_layers >= base.host_expert_layers:
        return None
    usable = _usable(seen)
    return SharedPool(
        slots, pool, peak, longest, grows - freed + back.used - base.used, -freed, 0, grows,
        measured_compute, clients, bound, source=source, sized=sized, held=held,
        host_layers=int(seen.placed["host_expert_layers"]), back=back,
        saved_ms=back.saved_ms - base.saved_ms, alone_ms=base.saved_ms,
        left=None if usable is None else usable + freed - grows - (back.used - base.used),
    )  # fmt: skip


def _freeing_words(situation: Situation, found: SharedPool) -> str:
    """The shared pool offered for the cells it frees, with the expert layers they bring back."""
    started = view(situation).started  # type: ignore[union-attr]
    back = found.back
    assert back is not None
    moved = found.host_layers - back.host_expert_layers
    tpot = situation.measurement.tpot_p50_ms
    alone = found.alone_ms
    total = found.saved_ms + alone
    saved = f"about {total:.1f} ms a token less decode"
    if alone:
        saved += f", {found.saved_ms:.1f} of them from the freed cells"
    saved += (
        f" (TPOT p50 {tpot:.1f} ms today; the placement model's estimate, not measured)"
        if tpot
        else " (the placement model's estimate, not measured)"
    )
    source = {
        "bound": f"the workload's {found.slots} longest sessions, prompt plus output cap",
        "measured": f"the most measured in flight, {found.peak:,}, plus {POOL_HEADROOM - 1:.0%}",
        "held": f"the {found.slots} largest requests measured in flight at their output caps, "
        f"plus {POOL_HEADROOM - 1:.0%}",
        "longest": f"the longest request, {found.longest} tokens",
    }[found.source]
    text = (
        f"shared pool of {found.pool:,} tokens ({source}) for the same {found.slots} slots, with "
        f"the experts of {moved} more layer{'s' * (moved != 1)} on the GPU (--n-cpu-moe "
        f"{found.host_layers} to {back.host_expert_layers}): nothing waited for a slot, but "
        f"{found.slots} split slots of {started.slot_ctx:,} tokens reserve "
        f"{started.ctx_total:,} cells; one pool of {found.pool:,} frees about "
        f"{-found.cells / MIB:,.0f} MiB of KV cache"
    )
    if found.compute:
        text += f" less {found.compute / MIB:,.0f} MiB more compute buffer"
    text += (
        f", which the placement model spends on those experts: {saved}; -np {found.slots} "
        f"--kv-unified -c {found.pool}"
    )
    if found.sized == "measured":
        text += f"; {_risk(found)}"
    return (
        f"{text}; any one request may still use up to {found.pool} tokens (the longest must "
        f"accept {found.longest})"
    )


def _usable(seen: View) -> int | None:
    """Device bytes the run left allocatable at its sampled peak (the CUDA total less the peak),
    when both were read."""
    usable = (seen.block.get("memory") or {}).get("usable_mib")
    return None if usable is None else int(usable * MIB)


def _pool_budget(room: int | None, usable: int | None) -> int | None:
    """The most device bytes a shared pool may add so that at least POOL_SPARE stays
    allocatable at its peak: from the run's own allocatable free when read, else from the fit's
    counted free less what it does not see."""
    if room is None:
        return None
    if usable is None:
        return room + DEVICE_MARGIN - POOL_UNSEEN - POOL_SPARE
    return min(room, usable - POOL_SPARE)


def _pool_settings(situation: Situation, engine: Engine, found: SharedPool) -> Settings:
    settings = apply_flag(situation.settings, "--kv-unified", True)
    if found.q8:
        settings = engine.with_lever(settings, "kv_cache_precision", "8bit")
    if found.more:
        settings = apply_flag(settings, "--n-cpu-moe", found.host_layers + found.more)
    if found.back is not None:
        settings = apply_flag(settings, "--n-cpu-moe", found.back.host_expert_layers)
    return dataclasses.replace(
        settings, max_concurrent_requests=found.slots, max_context_len=found.pool
    )


def _open_variants(situation: Situation, engine: Engine, seen: View) -> list[SharedPool]:
    """``_variants`` without those a run of the project already measured or a rejection
    refuses, so that after a pool try the next way in the order is the command."""
    from ...history import is_env, past_runs
    from ...rejected import refused_by

    context = analysis()
    here = context.this_run()
    measured = {
        run.engine_args
        for run in (past_runs(context.project, engine) if context.project is not None else ())
        if here is None or run.run_id != here.name
    }  # every run of the registered snapshot, later ones too, as the report's history reads them
    args = frozenset(context.engine_args)
    env = frozenset(a for a in args if is_env(a, context.flag_rules))
    ways = []
    for found in _variants(situation, seen):
        change = engine.engine_args_between(
            situation.settings, _pool_settings(situation, engine, found)
        )
        moved = {a.partition("=")[0] for a in change}
        held = (a for a in context.held if a.partition("=")[0] not in moved)
        command = env | frozenset((*held, *change))  # as the offer's command carries them
        if command in measured or refused_by(context.rejected, command, args, context.flag_rules):
            continue
        ways.append(found)
    return ways


def _pool_memory(found: SharedPool) -> str:
    compute = (
        "from this project's shared-pool start"
        if found.measured_compute
        else f"estimated at {POOL_COMPUTE_BYTES} bytes a cell per -ub token per slot over "
        f"{POOL_COMPUTE_BASE // MIB} MiB, never under today's buffer"
    )
    parts = [
        f"{words} {size / MIB:,.0f} MiB"
        for words, size in (
            ("the cells", found.cells),
            ("the added slots' state", found.state),
            (f"the compute buffer ({compute})", found.compute),
        )
        if size > 0
    ]
    total = found.cells + found.state + found.compute
    text = f"it needs about {total / MIB:,.0f} MiB more GPU memory: {', '.join(parts)}"
    spare = f"Tensward keeps at least {POOL_SPARE / MIB:,.0f} MiB"
    if found.left is not None:
        return (
            f"{text}; at its peak about {found.left / MIB:,.0f} MiB of the card stays "
            f"allocatable ({spare})"
        )
    return (
        f"{text}; sized so about {POOL_SPARE / MIB:,.0f} MiB stays allocatable at its peak, "
        f"counting {POOL_UNSEEN / MIB:,.0f} MiB the driver and CUDA hold beyond the fit's margin"
    )


def _way(found: SharedPool) -> str:
    """How a variant makes room, in a few words."""
    said = []
    if found.q8:
        said.append("a q8_0 KV cache (LOSSY: it changes answers)")
    if found.more:
        moved = found.host_layers + found.more
        said.append(
            f"{found.more} more expert {'layer' if found.more == 1 else 'layers'} in host memory "
            f"(--n-cpu-moe {found.host_layers} to {moved})"
        )
    if found.slots < found.clients and said:
        said.append(f"{found.slots} slots for the {found.clients} clients")
    return " and ".join(said)


def _fewer(found: SharedPool) -> str:
    return f"{found.slots} slots for the {found.clients} clients in {found.pool:,} tokens"


def _price(found: SharedPool) -> str:
    """The forecast of a variant, with no semicolon: it is part of the reason's first clause."""
    f = found.forecast
    if f is None:
        return "not estimated here (no decode model or request timings in this run)"
    return (
        f"about {_span(f.req, 2)} req/s (now {f.req_now:.2f}), decode "
        f"{_span(f.step, 0)} ms a step at {f.streams} streams (now {f.step_now:.0f} at "
        f"{f.running:.0f}), prompts {_span(f.prefill, 2)} ms a computed token (now "
        f"{f.prefill_now:.2f})"
    )


def _span(pair: tuple[float, float], digits: int) -> str:
    low, high = (f"{x:,.{digits}f}" for x in pair)
    return low if low == high else f"{low} to {high}"


def _size_words(found: SharedPool) -> str:
    measured = round(found.peak * POOL_HEADROOM * found.slots / found.clients)
    sessions = f"the {found.slots} sessions with the longest turns"
    estimated = (
        f", {found.estimated} prompts not yet sent counted from their length"
        if found.estimated
        else ""
    )
    bound = (
        f"the workload can hold {found.bound:,} tokens in flight ({sessions}, each prompt plus its "
        f"output cap{estimated})"
        if found.bound
        else "the workload's own bound is not known"
    )
    larger = {
        "bound": "the workload's most",
        "measured": "the measured peak",
        "held": "the largest requests measured in flight",
        "longest": "the longest request",
        "today": "today's pool",
    }[found.source]
    scaled = (
        f", scaled to {found.slots} of {found.clients} clients"
        if found.slots < found.clients
        else ""
    )
    if found.sized == "measured":
        return (
            f"the runs measured at most {found.peak:,} in flight (+{POOL_HEADROOM - 1:.0%}"
            f"{scaled}: {measured:,}) and the {found.slots} largest requests in flight together "
            f"at most {found.held:,} at their output caps (+{POOL_HEADROOM - 1:.0%}: "
            f"{round(found.held * POOL_HEADROOM):,}), so the pool is {larger}; {bound}"
        )
    return (
        f"{bound}, the runs measured at most {found.peak:,} (+{POOL_HEADROOM - 1:.0%}"
        f"{scaled}: {measured:,}), so the pool is {larger}"
    )


def _risk(found: SharedPool) -> str:
    """What a pool sized from the measurements risks, plainly."""
    worst = f" ({found.bound:,} tokens)" if found.bound else ""
    return (
        f"RISK: the pool is below the workload's worst case{worst}, so a burst of longer "
        "requests than the runs measured fills it and llama.cpp fails every request running in "
        "it ('Context size has been exceeded')"
    )


def shared_pool_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    if seen is None or analysis().goal == "tpot":
        return None
    ways = _open_variants(situation, engine, seen)
    if not ways:
        return None
    found, started = ways[0], seen.started
    if found.back is not None:
        return _freeing_words(situation, found)
    today = (
        f"one pool of {started.ctx_total} tokens"
        if started.kv_unified
        else f"{started.slot_ctx} tokens reserved for each"
    )
    slots = _slots(situation, started)
    who = "every client" if found.slots == found.clients else f"{found.slots} clients"
    way = _way(found)
    for_whom = (
        "every client"
        if found.slots == found.clients
        else f"{found.slots} of the {found.clients} clients (as many as the GPU memory holds)"
    )
    source = {
        "bound": f"the workload's {found.slots} longest sessions, prompt plus output cap",
        "measured": f"the most measured in flight plus {POOL_HEADROOM - 1:.0%}",
        "held": f"the {found.slots} largest requests measured in flight at their output caps, "
        f"plus {POOL_HEADROOM - 1:.0%}",
        "longest": f"the longest request, {found.longest} tokens",
        "today": "today's pool",
    }[found.source]
    text = (
        f"shared pool of {found.pool:,} tokens ({source}) for {for_whom}"
        f"{f', paid with {way}' if way else ''}: {_price(found)}; "
        f"{found.clients} clients shared {slots} slots ({today}), so requests waited for a slot; "
        f"{_size_words(found)}: -np {found.slots} --kv-unified -c {found.pool}"
        f"{' with a q8_0 KV cache' if found.q8 else ''} gives {who} a slot in one shared pool; "
        f"{_pool_memory(found)}"
    )
    if found.q8:
        (low, high), where = _q8_speed()
        measured = (
            f"measured on this model by an outside agent ({where}): it decodes at x{low} to "
            f"x{high} of f16's speed"
            if where
            else "research, not measured by Tensward: with several slots at 16k to 32k it "
            f"decodes at x{low} to x{high} of f16's speed"
        )
        prefill = " and prefills 1 to 9% slower" + (" (research)" if where else "")
        text += f"; the q8_0 cache is half the size and changes answers; {measured}{prefill}"
    if found.more:
        text += f"; {_expert_cost(found)}"
    if found.slots < found.clients:
        text += f"; {found.clients - found.slots} clients still wait for a slot"
    rest_ways = ways[1:]
    if found.sized == "measured":
        text += f"; {_risk(found)}"
        safer = [w for w in rest_ways if w.sized == "workload"]
        rest_ways = [w for w in rest_ways if w.sized == "measured"]
        if safer:
            listed = "; ".join(
                f"{f'{_way(w)} in {w.pool:,} tokens' if _way(w) else _fewer(w)}: {_price(w)}"
                for w in safer
            )
            text += f"; the safer ways, each pool sized to hold the workload's worst case: {listed}"
    if rest_ways:
        rest = "; ".join(f"{_variant_words(w)}: {_price(w)}" for w in rest_ways)
        text += f"; the other ways to make room, priced the same way: {rest}"
    return (
        f"{text}; any one request may still use up to {found.pool} tokens (the longest must "
        f"accept {found.longest}); each stream generates a bit slower"
    )


def _variant_words(found: SharedPool) -> str:
    """A variant in the list of the other ways: how it makes room, and its pool when that was
    sized from the measurements (below the workload's worst case)."""
    way = _way(found) or _fewer(found)
    if found.sized == "measured":
        return f"{way} in {found.pool:,} tokens (sized from the runs, below the worst case)"
    return way


def _expert_cost(found: SharedPool) -> str:
    """What moving the experts costs, from the forecast's decode and prompt terms."""
    f, freed = found.forecast, found.freed / MIB
    layers = "layer frees" if found.more == 1 else "layers free"
    head = f"the {found.more} more expert {layers} about {freed:,.0f} MiB"
    if f is None:
        return f"{head}; decode reads them from host memory and every prompt batch copies them"
    return (
        f"{head}; decode reads them from host memory (+{f.host_ms:.0f} ms a step at {f.streams} "
        f"streams) and every prompt batch copies them to the GPU (+{_span(f.copy_ms, 0)} ms a "
        f"batch: from the link rate up to all of today's batch time spread over today's host "
        "experts)"
    )


def with_shared_pool(situation: Situation, engine: Engine) -> Settings:
    seen = view(situation)
    ways = [] if seen is None else _open_variants(situation, engine, seen)
    return _pool_settings(situation, engine, ways[0]) if ways else situation.settings


# --- KV cache ---------------------------------------------------------------------------


def _prompt_p95(situation: Situation) -> int:
    lengths = [float(tokens) for tokens, _ in analysis().prompt_blocks.values()]
    return (
        int(nearest_rank(lengths, 0.95))
        if lengths
        else situation.measurement.max_prompt_tokens or 0
    )


def kv_unified_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    if seen is None or seen.started.kv_unified is not False:
        return None
    started, p95 = seen.started, _prompt_p95(situation)
    if not started.slot_ctx or not started.ctx_total or p95 <= started.slot_ctx:
        return None
    if p95 + situation.facts.output_tokens > started.ctx_total:
        return None
    return (
        f"prompts of {p95} tokens (95th percentile) exceed the {started.slot_ctx} tokens each "
        f"of the {started.slots} slots holds, while the pool holds {started.ctx_total}"
    )


def with_kv_unified(situation: Situation, engine: Engine) -> Settings:
    seen = view(situation)
    settings = apply_flag(situation.settings, "--kv-unified", True)
    if seen is not None and seen.started.ctx_total:
        settings = dataclasses.replace(settings, max_context_len=seen.started.ctx_total)
    return settings


def _larger_pool(
    situation: Situation, seen: View, cost: tuple[float, float] | None = None
) -> int | None:
    started = seen.started
    per_token, room = cost or (_kv_token_bytes(seen, _layers()), seen.room)
    if not started.ctx_total or per_token <= 0 or room is None:
        return None
    grown = min(started.ctx_total * 2, started.ctx_total + int(room // per_token))
    grown -= grown % POOL_ROUND
    return grown if grown >= started.ctx_total + POOL_ROUND else None


def kv_pool_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    if seen is None or not situation.fired("kv_capacity"):
        return None
    grown = _larger_pool(situation, seen)
    if grown is None:
        return None
    return (
        f"{situation.finding('kv_capacity').evidence}; the device has room for a pool of "
        f"{grown} tokens, the launch has {seen.started.ctx_total}"
    )


def with_larger_pool(situation: Situation, engine: Engine) -> Settings:
    seen = view(situation)
    grown = None if seen is None else _larger_pool(situation, seen)
    if grown is None:
        return situation.settings
    return dataclasses.replace(situation.settings, max_context_len=grown)


def _q8_cost(seen: View) -> tuple[float, float] | None:
    """Device bytes one pool token costs at q8_0, and the free bytes once the f16 cache that
    the launch holds is halved."""
    per_token = _kv_token_bytes(seen, _layers())
    if per_token <= 0 or seen.room is None or not seen.started.ctx_total:
        return None
    if _unified(seen) and seen.regime.get("host_memory_tight"):
        return None
    freed = per_token * seen.started.ctx_total * Q8_KV_SAVING
    return per_token * (1 - Q8_KV_SAVING), seen.room + freed


def _q8_growth(situation: Situation, seen: View) -> tuple[int | None, int | None]:
    """The pool (unified) and the slot count the bytes a q8_0 cache frees can hold."""
    cost = _q8_cost(seen)
    if cost is None:
        return None, None
    pool = _larger_pool(situation, seen, cost) if seen.started.kv_unified else None
    return pool, _next_slots(situation, seen, cost, pool)


def _beyond_f16(situation: Situation, seen: View, pool: int | None, slots: int | None) -> bool:
    """Whether the q8_0 pool or slot count is above what the same device room gives in f16."""
    f16_pool = _larger_pool(situation, seen) if seen.started.kv_unified else None
    f16_slots = _next_slots(situation, seen, None, f16_pool)
    return bool(
        (pool and pool > (f16_pool or seen.started.ctx_total or 0))
        or (slots and slots > (f16_slots or seen.started.slots or 0))
    )


def kv_q8_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    if seen is None or (seen.started.type_k, seen.started.type_v) != ("f16", "f16"):
        return None
    if not situation.fired("kv_capacity"):
        return None
    pool, slots = _q8_growth(situation, seen)
    if pool is None and slots is None:
        return None
    if not _beyond_f16(situation, seen, pool, slots):
        return None
    gain = f"{slots} slots" if slots else f"a pool of {pool} tokens"
    return (
        f"{situation.finding('kv_capacity').evidence}; a q8_0 KV cache is about half the size "
        f"and leaves room for {gain}"
    )


def with_kv_q8(situation: Situation, engine: Engine) -> Settings:
    settings = engine.with_lever(situation.settings, "kv_cache_precision", "8bit")
    seen = view(situation)
    if seen is None:
        return settings
    pool, slots = _q8_growth(situation, seen)
    if pool is not None:
        settings = dataclasses.replace(settings, max_context_len=pool)
    if slots is not None:
        settings = dataclasses.replace(settings, max_concurrent_requests=slots)
        if not seen.started.kv_unified:
            settings = engine.with_lever(settings, "max_context_len", seen.started.slot_ctx)
    return settings


def flash_attn_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    if seen is None or not (state := seen.started.flash_attn):
        return None
    anatomy = analysis().anatomy
    dims = {group.head_dim for group in anatomy.attention} if anatomy else set()
    if not state.startswith("off") or "not supported" in state or not dims:
        return None
    if seen.started.backend == "cuda" and not dims <= FLASH_HEAD_DIMS:
        return None
    return f"flash attention ran {state}; the attention head size ({max(dims)}) is supported"


FLASH_ATTN_GAIN = 0.5  # a rank weight, never printed: throughput rose 51-73% on an L4 with it on


def flash_attn_leads(situation: Situation, engine: Engine) -> Saving | None:
    return Saving(FLASH_ATTN_GAIN) if flash_attn_applies(situation, engine) else None


def with_flash_attn_on(situation: Situation, engine: Engine) -> Settings:
    return apply_flag(situation.settings, "--flash-attn", "on")


def swa_full_applies(situation: Situation, engine: Engine) -> str | None:
    anatomy, m, facts = analysis().anatomy, situation.measurement, situation.facts
    if anatomy is None or not any(g.kind == "sliding" for g in anatomy.attention):
        return None
    if situation.settings.prefix_caching is False or not facts.prompts:
        return None
    shared = shared_lengths(facts.prompts)
    sharing = [n for n in shared if n >= MIN_SHARED_PREFIX_TOKENS]
    if len(sharing) < MIN_SHARING_PROMPTS * len(shared):
        return None
    words = [len(p.split()) for p in facts.prompts]
    share = sum(shared) / max(sum(words), 1)
    hits = m.prefix_cache_hit_rate
    if hits is None or hits >= SWA_HIT_SHARE * share:
        return None
    return (
        f"the prompts share about {share:.0%} of their words but the cache served only "
        f"{hits:.0%} of the prompt tokens; a sliding-window cache drops the shared prefix "
        "unless --swa-full keeps it"
    )


SWA_FULL_GAIN = 0.17  # a rank weight, never printed: TPOT p95 -17% on Gemma 4 E4B, the lesser gain


def swa_full_leads(situation: Situation, engine: Engine) -> Saving | None:
    """Its own signal, a prompt cache that drops the shared prefix, holds whatever limit the
    run names: it leads Try first for every goal, as each of them measured better."""
    return Saving(SWA_FULL_GAIN) if swa_full_applies(situation, engine) else None


def with_swa_full(situation: Situation, engine: Engine) -> Settings:
    return apply_flag(situation.settings, "--swa-full", True)


# --- ubatch -----------------------------------------------------------------------------


def _ubatch_growth(situation: Situation, seen: View) -> tuple[int, int] | None:
    """The next -ub rung and the device bytes its larger compute buffer adds (0 when the log
    shows no device buffer)."""
    current = seen.started.ubatch or UBATCH_LADDER[0]
    larger = next((rung for rung in _ubatch_rungs(seen) if rung > current), None)
    return None if larger is None else _growth_at(situation, seen, larger)


def _ubatch_rungs(seen: View) -> tuple[int, ...]:
    most = {"dense_offload": DENSE_UBATCH_MOST, "expert_offload": UBATCH_LADDER[-1]}
    return tuple(rung for rung in UBATCH_LADDER if rung <= most.get(seen.kind or "", UBATCH_MOST))


def _growth_at(situation: Situation, seen: View, larger: int) -> tuple[int, int] | None:
    """The device bytes the compute buffer adds at ``-ub larger``: what a probe start read at
    that rung, else today's buffer scaled with the rung (6.8x too high on GLM-5.3-Flash, right
    on Qwen3.5-35B-A3B and dense Qwen3-32B)."""
    started = seen.started
    current = started.ubatch or UBATCH_LADDER[0]
    batch = situation.settings.prefill_batch_tokens or DEFAULT_BATCH
    if (started.ctx_total or 0) < max(batch, larger):
        return None
    buffer = started.bytes_of("compute", "device")
    if not buffer:
        return larger, 0
    measured = (seen.probes.get("ubatch_buffers") or {}).get(str(larger))
    if isinstance(measured, int) and measured > 0:
        return larger, max(measured - buffer, 0)
    return larger, buffer * larger // current - buffer


def _fit_headroom(situation: Situation, seen: View) -> int:
    """Device bytes llama.cpp's fit can still take from a context it chose itself (no --ctx-size
    given, and the log says fit cut it): the KV of the pool above what the longest measured
    request needs on each of the run's slots."""
    started = seen.started
    if situation.settings.max_context_len or "ctx_size" not in started.fit_changed:
        return 0
    if not started.ctx_total:
        return 0
    floor = _pool_for(_longest_request(situation, seen), _slots(situation, started) or 1)
    return max(int((started.ctx_total - floor) * _kv_token_bytes(seen, _layers())), 0)


def _next_ubatch(situation: Situation, seen: View, freed: int = 0) -> int | None:
    """The next -ub rung, when its compute buffer fits the room, with ``freed`` more bytes and
    what fit can take from a context it chose."""
    if (growth := _ubatch_growth(situation, seen)) is None:
        return None
    larger, extra = growth
    if not extra:
        return larger
    if (room := seen.room) is None or extra > room + freed + _fit_headroom(situation, seen):
        return None
    return larger


def _fit_shrink(situation: Situation, seen: View) -> str:
    """The cost of a larger -ub whose compute buffer fit takes from its own context."""
    growth, room = _ubatch_growth(situation, seen), seen.room
    if growth is None or room is None or growth[1] <= room or not _fit_headroom(situation, seen):
        return ""
    tokens = -(-(growth[1] - room) // _kv_token_bytes(seen, _layers()))
    tokens = int(-(-tokens // POOL_ROUND) * POOL_ROUND)
    return (
        f"; the cost: no --ctx-size is set, so fit picks a context about {tokens:,} tokens "
        f"smaller than today's {seen.started.ctx_total:,} to make room"
    )


def _ubatch_step(situation: Situation, freed: int = 0) -> tuple[View, int] | None:
    """The run's view and the next -ub rung, when one fits (with ``freed`` more device bytes)
    and no probe ruled it out."""
    seen = view(situation)
    if seen is None or (larger := _next_ubatch(situation, seen, freed)) is None:
        return None
    pcie = seen.probes.get("pcie") or {}
    if pcie.get("reason", "").startswith("the larger ubatch did not shorten"):
        return None
    if _unified(seen) and not seen.offloaded:
        return None
    return seen, larger


def raise_ubatch_applies(situation: Situation, engine: Engine, freed: int = 0) -> str | None:
    if (step := _ubatch_step(situation, freed)) is None:
        return None
    seen, larger = step
    cost = _fit_shrink(situation, seen)
    if situation.fired("prefill"):
        why = situation.finding("prefill").evidence
    elif (
        analysis().goal == "ttft"
        and situation.finding("prefill").evidence == QUEUED_PREFILL
        and (_compute_share(situation.measurement) or 0.0) >= PREFILL_TIME_SHARE
        and (long := _long_prompts(situation, seen, larger))
    ):
        return f"{long}; a shorter prefill frees each slot sooner, so the queue falls too{cost}"
    elif _unified(seen):
        return None
    elif seen.kind == "expert_offload":
        why = "the experts run from host memory, and a larger -ub reuses each streamed weight"
    elif seen.offloaded and (situation.measurement.max_prompt_tokens or 0) >= LONG_PROMPT_TOKENS:
        why = "prompts of 2,048 tokens or more stream the host weights once per ubatch"
    else:
        return None
    current = seen.started.ubatch or UBATCH_LADDER[0]
    return (
        f"{why}; -ub {current} to {larger}{_microbatches(situation, seen, current, larger)}{cost}"
    )


def _microbatches(situation: Situation, seen: View, current: int, larger: int) -> str:
    """What fewer micro-batches save on the longest prompts: each one uploads the experts in
    host memory once, at this run's measured copy rate (GLM-5.3-Flash Q2 0.97 s predicted, 0.975 s
    measured; Q4 at 8192 15.4 s against 16.3 s on a 16k prompt; Qwen3.5-35B 0.71 s against
    0.69 s); empty without a measured rate."""
    if seen.kind != "expert_offload" or not (host := seen.placed.get("host_expert_bytes")):
        return ""
    facts: Mapping[str, object] = analysis().kb_facts or {}
    pcie = seen.probes.get("pcie") or {}
    rate = facts.get("census.h2d_gb_per_s") or pcie.get("gbs_effective") or pcie.get("stored_gbs")
    if not isinstance(rate, (int, float)) or rate <= 0:
        return ""
    prompt = _prompt_p95(situation)
    computed = int(prompt * (1 - (situation.measurement.prefix_cache_hit_rate or 0.0)))
    each = host / (rate * 1e9)
    now, then = -(-computed // current), -(-computed // larger)
    if now <= then:
        return ""
    return (
        f"; each micro-batch uploads the {host / 1e9:.1f} GB of experts in host memory at the "
        f"measured {rate:.1f} GB/s, about {each:.2f} s: a {computed:,}-token prompt (p95 after "
        f"the prompt cache) takes {now} at -ub {current} and {then} at -ub {larger}, about "
        f"{(now - then) * each:.1f} s less (predicted)"
    )


def _compute_share(m: Measurement) -> float | None:
    """The share of a typical answer's server time spent computing its prompt: TTFT less its
    queued share, against that plus the decode of the mean answer; None without the rates."""
    if m.ttft_p50_ms is None or m.queue_share is None or not m.tpot_p50_ms:
        return None
    if not m.request_throughput or m.output_throughput is None:
        return None
    prompt = m.ttft_p50_ms * (1 - m.queue_share)
    decode = m.tpot_p50_ms * max(m.output_throughput / m.request_throughput - 1, 0.0)
    return prompt / (prompt + decode) if prompt + decode > 0 else None


def _long_prompts(situation: Situation, seen: View, larger: int) -> str | None:
    """Prompts that need more than one micro-batch after the prompt cache."""
    current, prompt = seen.started.ubatch or UBATCH_LADDER[0], _prompt_p95(situation)
    computed = int(prompt * (1 - (situation.measurement.prefix_cache_hit_rate or 0.0)))
    if computed <= current:
        return None
    return (
        f"the longest prompts (p95 {prompt:,} tokens, about {computed:,} computed after the "
        f"prompt cache) are read {current} tokens at a time; -ub {current} to {larger}"
    )


def raise_ubatch_side(situation: Situation, engine: Engine, freed: int = 0) -> str | None:
    """Prompts that need more than one micro-batch after the prompt cache: a larger -ub reads
    them in fewer steps."""
    if (step := _ubatch_step(situation, freed)) is None:
        return None
    seen, larger = step
    long = _long_prompts(situation, seen, larger)
    return f"{long}{_fit_shrink(situation, seen)}" if long else None


def with_larger_ubatch(situation: Situation, engine: Engine, freed: int = 0) -> Settings:
    seen = view(situation)
    settings = situation.settings
    if seen is None or (larger := _next_ubatch(situation, seen, freed)) is None:
        return settings
    settings = apply_flag(settings, "--ubatch-size", larger)
    if (settings.prefill_batch_tokens or DEFAULT_BATCH) < larger:
        settings = apply_flag(settings, "--batch-size", larger)
    return settings


# --- speculation ------------------------------------------------------------------------

SPEC_MAX_RUNNING = 4  # requests running at once above which drafts cost more than they save


def _ngram_room(situation: Situation, engine: Engine) -> str | None:
    """Why draft-free n-gram speculation has room: few requests run at once, none waited, the KV
    cache is not the limit, and nothing speculates yet; None otherwise."""
    m = situation.measurement
    if engine.lever_value(situation.settings, "speculation") is not None:
        return None
    if situation.facts.structured or m.tpot_p50_ms is None or m.peak_running is None:
        return None
    if m.peak_running > SPEC_MAX_RUNNING or m.peak_waiting or situation.fired("kv_capacity"):
        return None
    return (
        f"at most {m.peak_running:.0f} requests ran at once and none waited, so each decode step "
        "has room to check drafted tokens; --spec-type ngram-mod drafts them from text already in "
        "the context (the prompt and the answer so far) with no draft model, so it helps when "
        "answers copy their input, such as tool-call arguments, and the first token comes later"
    )


def ngram_speculation_applies(situation: Situation, engine: Engine) -> str | None:
    return _ngram_room(situation, engine) if analysis().goal == "tpot" else None


def with_ngram_speculation(situation: Situation, engine: Engine) -> Settings:
    return engine.with_lever(situation.settings, "speculation", Speculation(NGRAM_MOD))


MTP_DRAFT_TOKENS = 2  # tokens drafted a step: on an L4, Qwen3.8-27B's TPOT fell more at 2 than at 1
# Tokens drafted a step on tool turns at up to MTP_TOOLS_MAX_SLOTS requests at once: on an L4,
# Qwen3.8-27B's agent TPOT at 4 was 0.65 to 0.84 of that at 2 over 2 to 4 slots, b12246.
MTP_TOOLS_DRAFT_TOKENS = 4
MTP_TOOLS_MAX_SLOTS = 4
MTP_MIDDLE_DRAFT_TOKENS = 3  # with the memory for it and no tools: x2.33 pooled against x2.07 at 2
# One more drafted token a step when at least this share of the drafted ones was accepted: on an
# L4, Qwen3.8-27B accepted 0.76 at 2 a step and drafting 3 gave 30.6 against 26.4 tok/s.
MTP_DEEPER_ACCEPTED = 0.6
MTP_DEEPER_GAIN = 0.137  # a rank weight, never printed: TPOT 1 - 26.4/30.6 on the same L4 runs
MTP_GAIN = 0.396  # a rank weight, never printed: TPOT p50 -39.6% on Qwen3.8-27B, one L4, agent c3
# Card memory the draft head took at its peak on Qwen3.8-27B (hybrid), one L4, one slot: about
# 550 MiB plus 150 MiB a drafted token (0.85 GB at 2, 1.15 GB at 4, 1.75 GB at 8); the estimate
# when the GGUF's blocks or the start-up log are not read.
MTP_HEAD_BYTES = 550 * MIB
MTP_STEP_BYTES = 150 * MIB
MTP_HIDDEN_ROWS = 2048  # rows of target hidden states the head keeps on the card (-b's default)


def _mtp_draft_bytes(seen: View, depth: int = MTP_DRAFT_TOKENS) -> int | None:
    """Device bytes the draft head takes at ``depth`` tokens a step, from its parts on b12246
    (L27): the MTP blocks' weights, which llama.cpp loads only for draft-mtp; a KV cache as long
    as the target's for each MTP layer; a compute buffer no larger than the target's; the target's
    hidden states for a batch in f32; and on a hybrid model one more recurrent state a slot for
    each drafted token (n_rs_seq). Without the blocks' sizes, at least ``MTP_HEAD_BYTES`` plus
    ``MTP_STEP_BYTES`` a drafted token. None without the pool or the anatomy."""
    anatomy = analysis().anatomy
    full = sum(g.layers for g in anatomy.attention if g.window is None) if anatomy else 0
    pool = seen.started.kv_cells
    if anatomy is None or not anatomy.mtp_layers or not full or not pool:
        return None
    k, v = seen.started.type_k or "f16", seen.started.type_v or "f16"
    layer = kv_bytes_per_token(anatomy, k, v) // full
    context = anatomy.mtp_layers * layer * pool + seen.started.bytes_of("compute", "device")
    blocks = anatomy.layer_bytes[-anatomy.mtp_layers :]
    if len(blocks) != anatomy.mtp_layers or not all(blocks) or not anatomy.hidden:
        return max(context, MTP_HEAD_BYTES) + MTP_STEP_BYTES * depth
    slots = seen.started.slots or 1
    states = anatomy.state_bytes_per_seq * slots * depth  # 598.5 MiB a depth step on L27
    parts = sum(blocks) + context + anatomy.hidden * 4 * MTP_HIDDEN_ROWS + states
    return max(parts, MTP_HEAD_BYTES + MTP_STEP_BYTES * depth)


def _mtp_left(situation: Situation, seen: View) -> int | None:
    """The device bytes the MTP room check sees left: the fit room, and the memory still
    allocatable at the sampled peak less the fit margin, whichever is smaller."""
    left = seen.room
    if (free := _sampled_free(situation, seen)) is not None:
        free -= DEVICE_MARGIN
        left = free if left is None else min(left, free)
    return left


def _sampled_free(situation: Situation, seen: View) -> int | None:
    """Device bytes still allocatable at the run's sampled peak: the CUDA total the log reports
    less the peak, else the sampled total less the peak (which counts the driver's reserve)."""
    peak, total = sampled_memory(analysis().this_run(), situation.measurement)
    if peak is None or total is None:
        return None
    usable = usable_free(peak, seen.started.device_total_bytes)
    return usable if usable is not None else int((total - peak) * MIB)


def _mtp_depth(situation: Situation, seen: View | None = None) -> int:
    """Tokens the first MTP offer drafts a step. On a run with layers on the host, 2: the head's
    memory comes out of the target's layers. Otherwise, at a known few requests at once, 4 on tool
    turns and 3 elsewhere, each only when the card has the memory for it; 2 in every other case.
    Unknown memory keeps the measured rule: 4 on tool turns, else 2."""
    facts = situation.facts
    few = facts.max_inflight is not None and facts.max_inflight <= MTP_TOOLS_MAX_SLOTS
    wanted = MTP_TOOLS_DRAFT_TOKENS if facts.offers_tools else MTP_MIDDLE_DRAFT_TOKENS
    if not few or (seen is not None and seen.offloaded):
        return MTP_DRAFT_TOKENS
    left = None if seen is None else _mtp_left(situation, seen)
    if left is None or seen is None:
        return MTP_TOOLS_DRAFT_TOKENS if facts.offers_tools else MTP_DRAFT_TOKENS
    for depth in range(wanted, MTP_DRAFT_TOKENS, -1):
        if (draft := _mtp_draft_bytes(seen, depth)) is not None and draft <= left:
            return depth
    return MTP_DRAFT_TOKENS


def _mtp_room(situation: Situation, engine: Engine, freed: int = 0) -> str | None:
    """Why the model's own next-token-prediction layers can draft: the GGUF has them, nothing
    speculates yet, and a context the user sized leaves room for the draft head, by the fit
    estimate and in the device memory the run left free less the fit margin; None otherwise, and
    when the GGUF does not say; with ``freed`` more device bytes."""
    anatomy = analysis().anatomy
    layers = anatomy.mtp_layers if anatomy is not None else None
    if not layers or engine.lever_value(situation.settings, "speculation") is not None:
        return None
    if situation.measurement.spec_acceptance_length is not None:
        return None
    seen = view(situation)
    depth = _mtp_depth(situation, seen)
    draft = None if seen is None else _mtp_draft_bytes(seen, depth)
    room = None if seen is None else seen.room
    pinned = seen is not None and seen.offloaded  # try pins the layers: the head must fit beside
    if (situation.settings.max_context_len or pinned) and draft and seen is not None:
        if room is not None and room + freed < draft:
            return None
        free = _sampled_free(situation, seen)
        if free is not None and free - DEVICE_MARGIN + freed < draft:
            return None
    memory = f"about {draft / 2**20:,.0f} MiB (estimated)" if draft else "device memory"
    held, source = (
        (f"{layers} next-token-prediction layers", "them")
        if layers > 1
        else ("a next-token-prediction layer", "it")
    )
    text = (
        f"the model file carries {held}; --spec-type draft-mtp drafts {depth} tokens "
        f"a step from {source} and the model checks them in one pass, so a step can yield more "
        f"than one token, with no draft model; the draft head takes {memory} of card memory "
        "and the first token comes later"
    )
    slots = seen.started.slots if seen is not None else None
    if (slots or situation.facts.max_inflight or 1) > 1:
        text += (
            "; with several requests at once it gains less and reads prompts slower: on "
            "Qwen3.8-27B at depth 4 decode was x2.25 with 1 client against x1.66 with 4, and "
            "prompts read 15% slower with 4 (research, not measured by Tensward)"
        )
    if seen is not None and seen.offloaded:
        text += (
            "; with layers on the host that memory comes out of the target's layers: measured "
            "on Qwen3.8-27B with 12 and 16 GB of card, giving the head layers still paid on "
            "coding (x1.19, x1.40) but about broke even on chat at 12 GB (x1.03; research, not "
            "measured by Tensward); a block draft that takes layers is never offered"
        )
    return text


def _current_mtp(situation: Situation, engine: Engine) -> int | None:
    """Tokens the setup drafts a step from the model's own layers; None when it does not."""
    current = engine.lever_value(situation.settings, "speculation")
    if not isinstance(current, Speculation) or current.method != DRAFT_MTP:
        return None
    return current.draft_tokens or MTP_DRAFT_TOKENS


def _deeper_mtp(situation: Situation, engine: Engine) -> str | None:
    """Why drafting one more token a step can pay: the setup drafts from the model's own layers,
    this run measured most of the drafted tokens accepted, the depth stays at most 4 and, when
    the card memory is known, the larger head fits; None otherwise."""
    depth = _current_mtp(situation, engine)
    if depth is None or depth >= MTP_TOOLS_DRAFT_TOKENS:
        return None  # depths past 4 were never measured for general traffic
    length = situation.measurement.spec_acceptance_length
    if length is None or (accepted := (length - 1) / depth) < MTP_DEEPER_ACCEPTED:
        return None
    seen = view(situation)
    if seen is not None and (left := _mtp_left(situation, seen)) is not None:
        if left < MTP_STEP_BYTES:
            return None
    return (
        f"{accepted:.0%} of the {depth} tokens drafted a step were accepted; "
        f"--spec-draft-n-max {depth + 1} drafts one more and the model still checks "
        "them in one pass"
    )


def _mtp_offer(situation: Situation, engine: Engine) -> str | None:
    return _mtp_room(situation, engine) or _deeper_mtp(situation, engine)


def mtp_speculation_applies(situation: Situation, engine: Engine) -> str | None:
    return _mtp_offer(situation, engine) if analysis().goal in ("tpot", "throughput") else None


def mtp_speculation_side(situation: Situation, engine: Engine) -> str | None:
    return _mtp_offer(situation, engine)


def mtp_speculation_leads(situation: Situation, engine: Engine) -> Saving | None:
    if not mtp_speculation_applies(situation, engine):
        return None
    return Saving(MTP_DEEPER_GAIN if _deeper_mtp(situation, engine) else MTP_GAIN)


def with_mtp_speculation(situation: Situation, engine: Engine) -> Settings:
    if _deeper_mtp(situation, engine) and (depth := _current_mtp(situation, engine)):
        tokens = depth + 1
    else:
        tokens = _mtp_depth(situation, view(situation))
    return engine.with_lever(situation.settings, "speculation", Speculation(DRAFT_MTP, tokens))


# --- placement --------------------------------------------------------------------------


def _saves(step: Mapping[str, Any]) -> str:
    saved = step["saved_ms_per_token"]
    if saved is None:
        return f"{step['why']}; saving not predicted"
    return f"{step['why']}; about {saved:.2f} ms per token"


def _total_layers(seen: View) -> int | None:
    return seen.started.total_layers


def _step_share(situation: Situation, step: Mapping[str, Any] | None) -> Saving | None:
    tpot = situation.measurement.tpot_p50_ms
    saved = None if step is None else step["saved_ms_per_token"]
    if not (saved and tpot and step):
        return None
    return Saving(saved / tpot, step["label"] == "lower bound", step["label"] == "upper estimate")


def offload_less_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    step = None if seen is None else seen.step("host_layers", "host_expert_layers")
    if step is None or isinstance(step["value"], Mapping):
        return None
    return _saves(step)


def offload_less_leads(situation: Situation, engine: Engine) -> Saving | None:
    seen = view(situation)
    step = None if seen is None else seen.step("host_layers", "host_expert_layers")
    if step is None or isinstance(step["value"], Mapping):
        return None
    return _step_share(situation, step)


def with_fewer_host_layers(situation: Situation, engine: Engine) -> Settings:
    seen = view(situation)
    step = None if seen is None else seen.step("host_layers", "host_expert_layers")
    if seen is None or step is None or isinstance(step["value"], Mapping):
        return situation.settings
    if step["lever"] == "host_expert_layers":
        return apply_flag(situation.settings, "--n-cpu-moe", step["value"])
    total = _total_layers(seen)
    return (
        situation.settings
        if total is None
        else apply_flag(situation.settings, "--n-gpu-layers", total - step["value"])
    )


def _expert_step(seen: View) -> Mapping[str, Any] | None:
    step = seen.step("host_expert_layers")
    return step if step is not None and isinstance(step["value"], Mapping) else None


def offload_experts_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    step = None if seen is None else _expert_step(seen)
    if step is None:
        return None
    return _saves(step)


def offload_experts_leads(situation: Situation, engine: Engine) -> Saving | None:
    seen = view(situation)
    return _step_share(situation, None if seen is None else _expert_step(seen))


def with_host_experts(situation: Situation, engine: Engine) -> Settings:
    seen = view(situation)
    step = None if seen is None else _expert_step(seen)
    if seen is None or step is None or (total := _total_layers(seen)) is None:
        return situation.settings
    settings = apply_flag(situation.settings, "--n-gpu-layers", total)
    return apply_flag(settings, "--n-cpu-moe", step["value"]["host_expert_layers"])


def offload_ffn_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    step = None if seen is None else seen.step("host_ffn_layers")
    return None if step is None else step["why"]


def with_host_ffn(situation: Situation, engine: Engine) -> Settings:
    seen = view(situation)
    step = None if seen is None else seen.step("host_ffn_layers")
    if seen is None or step is None or (total := _total_layers(seen)) is None:
        return situation.settings
    settings = apply_flag(situation.settings, "--n-gpu-layers", total)
    return apply_flag(settings, "--n-cpu-ffn", step["value"])


def kv_host_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    if seen is not None and _unified(seen):
        return None
    step = None if seen is None else seen.step("kv_on_host")
    return None if step is None else _saves(step)


def with_kv_on_host(situation: Situation, engine: Engine) -> Settings:
    return apply_flag(situation.settings, "--kv-offload", False)


def _pool_for(need: int, slots: int) -> int:
    """The KV pool that holds ``slots`` requests of ``need`` tokens each, rounded as a cut pool
    is."""
    return max(-(-need * slots // POOL_ROUND) * POOL_ROUND, MIN_POOL_TOKENS)


def _room_step(seen: View) -> Mapping[str, Any] | None:
    """The first room step; on a discrete card never the q8_0 KV cache, which
    llamacpp-kv-q8-for-layers offers with one slot."""
    unfit = _unified(seen) and not seen.fit.get("fits", True)
    levers = ROOM_FLAGS if _unified(seen) else [k for k in ROOM_FLAGS if k != "kv_cache_precision"]
    return seen.step(*levers) if seen.offloaded or unfit else None


def _gives_up(seen: View, step: Mapping[str, Any]) -> str:
    """What a step that cuts the slots or the context takes from a launch that does not fit this
    Mac's Metal working set, and the other way out."""
    fit = seen.fit
    slots, pool, need = fit.get("slots"), fit.get("pool_tokens"), fit.get("need_tokens")
    if step["lever"] == "max_concurrent_requests" and slots and pool and need:
        cut = _pool_for(need, step["value"])
        gone = (
            f"concurrency ({step['value']} request{'s' * (step['value'] != 1)} at a time instead "
            f"of {slots}) and context (a KV pool of {max(cut, DEFAULT_BATCH)} tokens instead of "
            f"{pool})"
        )
    elif step["lever"] == "max_context_len" and pool:
        gone = f"context (a KV pool of {step['value']} tokens instead of {pool})"
    else:
        return ""
    limit = seen.step(WIRED_LIMIT)
    raised = f"={limit['value']}" if limit else ""
    return (
        f"; this is the largest setup that fits this Mac's Metal working set, and it gives up "
        f"{gone}; requests beyond that wait or are refused; raising the working set "
        f"(sudo sysctl iogpu.wired_limit_mb{raised}) keeps today's setup instead"
    )


def _cuts_window(situation: Situation, seen: View, step: Mapping[str, Any]) -> bool:
    """Whether a room step leaves a request less context than it must be able to hold."""
    if step["lever"] != "max_context_len":
        return False
    slots = 1 if seen.started.kv_unified else _slots(situation, seen.started) or 1
    return int(step["value"]) // slots < _longest_request(situation, seen)


def buy_room_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    step = None if seen is None else _room_step(seen)
    if step is None or seen is None or _cuts_window(situation, seen, step):
        return None
    unfit = _unified(seen) and not seen.fit.get("fits", True)
    return _saves(step) + (_gives_up(seen, step) if unfit else "")


def with_room_step(situation: Situation, engine: Engine) -> Settings:
    seen = view(situation)
    step = None if seen is None else _room_step(seen)
    settings = situation.settings
    if seen is None or step is None:
        return settings
    lever, value = step["lever"], step["value"]
    if step.get("cpu_moe") is not None:
        settings = apply_flag(settings, "--n-cpu-moe", step["cpu_moe"])
    if lever == "kv_cache_precision":
        return engine.with_lever(settings, lever, value)
    if lever == "ubatch":
        return apply_flag(settings, "--ubatch-size", value)
    if lever == "max_concurrent_requests":
        settings = dataclasses.replace(settings, max_concurrent_requests=value)
        return dataclasses.replace(
            settings, max_context_len=_pool_for(_longest_request(situation, seen), value)
        )
    return dataclasses.replace(settings, max_context_len=value)


@dataclass(frozen=True, slots=True)
class Spent:
    """Where the layers stand once a device room is spent on moving them back."""

    host_layers: int
    host_expert_layers: int
    saved_ms: float  # decode milliseconds a token the moves save, estimated
    used: int  # device bytes the moves take


def _spend(seen: View, extra: int, kv_share: float = 1.0) -> tuple[Spent, Spent] | None:
    """The placement model's moves over the launch's own room, and over that room with ``extra``
    more bytes and each layer's KV cache at ``kv_share`` of today's; None without the placement
    block, its fit or its decode model."""
    anatomy, model = analysis().anatomy, seen.block.get("decode_model")
    if anatomy is None or not anatomy.layers or not model or not seen.fit or not seen.placed:
        return None
    p, fit, decode = Placement(**seen.placed), Fit2(**seen.fit), DecodeModel(**model)
    room, kv_layer = fit.device_free - fit.device_need, fit.kv_bytes // anatomy.layers

    def spent(bytes_: int, kv: int) -> Spent:
        steps, saved, left = _plan(anatomy, p, decode, bytes_, kv, "possible")
        values = {step.lever: step.value for step in steps}
        return Spent(
            values.get("host_layers", p.host_layers),
            values.get("host_expert_layers", p.host_expert_layers),
            saved,
            bytes_ - left,
        )

    return spent(room, kv_layer), spent(room + extra, round(kv_layer * kv_share))


def _brought_back(base: Spent, more: Spent) -> int:
    return base.host_layers - more.host_layers + base.host_expert_layers - more.host_expert_layers


def _moved_words(seen: View, base: Spent, more: Spent) -> str:
    layers = base.host_layers - more.host_layers
    experts = base.host_expert_layers - more.host_expert_layers
    said = []
    if layers:
        said.append(f"{layers} more layer{'s' * (layers != 1)} on the GPU")
    if experts:
        said.append(f"the experts of {experts} more layer{'s' * (experts != 1)} on the GPU")
    return " and ".join(said)


def _with_spent(settings: Settings, seen: View, spent: Spent) -> Settings:
    """``settings`` with the explicit placement ``spent`` describes: every try launch pins the
    placement, so the layers a change makes room for are named."""
    p = Placement(**seen.placed)
    if spent.host_expert_layers != p.host_expert_layers:
        settings = apply_flag(settings, "--n-cpu-moe", spent.host_expert_layers)
    if spent.host_layers != p.host_layers and (total := _total_layers(seen)) is not None:
        settings = apply_flag(settings, "--n-gpu-layers", total - spent.host_layers)
    return settings


# --- the fit margin ----------------------------------------------------------------------

FIT_TARGET = 256 * MIB  # the tighter margin offered against llama.cpp's default 1024 MiB
FIT_LEFT_MIN = 128 * MIB  # less card memory than this left after the move: not offered


def _fit_target(situation: Situation, seen: View) -> tuple[Spent, Spent, int] | None:
    """For a run whose layers llama.cpp's fit placed with the default margin and that keeps
    layers or experts on the host: the placement the 768 MiB a 256 MiB margin releases brings
    back, and the card memory left after it (sampled at the peak, else the fit room plus the
    margin less the 0.3 GiB the estimate leaves out); None when it brings nothing back, leaves
    under ``FIT_LEFT_MIN``, or offload-less already moves layers."""
    extra = situation.settings.extra_args
    if _unified(seen) or not seen.offloaded or _no_host_experts(situation):
        return None
    if not {"n_gpu_layers", "tensor_overrides"} & set(seen.started.fit_changed):
        return None
    if "--n-gpu-layers" in extra or "--n-cpu-moe" in extra:
        return None
    if str(extra.get("--fit-target", DEVICE_MARGIN // MIB)) != str(DEVICE_MARGIN // MIB):
        return None
    if seen.step("host_layers", "host_expert_layers") is not None:
        return None
    if (spent := _spend(seen, DEVICE_MARGIN - FIT_TARGET)) is None:
        return None
    base, tight = spent
    if _brought_back(base, tight) <= 0 or seen.room is None:
        return None
    free = _sampled_free(situation, seen)
    if free is None:
        free = seen.room + DEVICE_MARGIN - ESTIMATE_LOW
    left = free - (tight.used - max(base.used, 0))
    return (base, tight, left) if left >= FIT_LEFT_MIN else None


def _left_words(situation: Situation, seen: View) -> str:
    if _sampled_free(situation, seen) is not None:
        return "as sampled at this run's peak"
    return "counting the 0.3 GiB the engine's estimate leaves out"


def fit_target_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    if seen is None or (found := _fit_target(situation, seen)) is None:
        return None
    base, tight, left = found
    saved = tight.saved_ms - base.saved_ms
    return (
        f"llama.cpp's fit keeps 1,024 MiB of the card free by its own estimate; with "
        f"--fit-target 256 it places {_moved_words(seen, base, tight)} (about {saved:.1f} ms per "
        f"token saved, estimated); card memory left at the peak: about {left / MIB:,.0f} MiB, "
        f"{_left_words(situation, seen)}; Tensward measures that placement as explicit flags, "
        "and in your own command --fit-target 256 lets llama.cpp choose it"
    )


def with_fit_target(situation: Situation, engine: Engine) -> Settings:
    seen = view(situation)
    if seen is None or (found := _fit_target(situation, seen)) is None:
        return situation.settings
    return _with_spent(situation.settings, seen, found[1])


# --- KV q8_0 for layers --------------------------------------------------------------------

Q8_FREE_GQA = 4  # query heads per KV head with which one-slot q8_0 decoded like f16


def _gqa(anatomy: Any) -> int | None:
    """Query heads per KV head of the full-attention layers, when the GGUF says and it is one
    figure."""
    full = {g.kv_heads for g in anatomy.attention if g.window is None and g.kv_heads}
    if not anatomy.q_heads or len(full) != 1:
        return None
    heads = full.pop()
    return anatomy.q_heads // heads if anatomy.q_heads % heads == 0 else None


def _q8_layers(situation: Situation, engine: Engine, seen: View) -> tuple[Spent, Spent] | None:
    """For an offloaded run on a discrete card that serves one request at a time with an f16 KV
    cache and flash attention, on a model with 4 query heads per KV head: the placement the bytes
    a q8_0 cache frees bring back; None when they bring nothing back or kv-q8 (more slots)
    applies."""
    started, anatomy = seen.started, analysis().anatomy
    if _unified(seen) or not seen.offloaded or anatomy is None:
        return None
    if (started.type_k, started.type_v) != ("f16", "f16") or started.flash_attn != "on":
        return None
    inflight = situation.facts.max_inflight
    wanted = demand(situation.measurement) if inflight is None else inflight
    if wanted is None or wanted > 1 or _gqa(anatomy) != Q8_FREE_GQA:
        return None
    if situation.fired("kv_capacity") and kv_q8_applies(situation, engine):
        return None
    if not seen.fit or not seen.placed:
        return None
    kv = device_kv(Placement(**seen.placed), Fit2(**seen.fit), anatomy.layers or 1)
    if seen.step("host_layers", "host_expert_layers") is not None:
        return None  # offload-less moves layers with the room there is first
    spent = _spend(seen, round(kv * Q8_KV_SAVING), 1 - Q8_KV_SAVING)
    if spent is None or _brought_back(*spent) <= 0:
        return None
    return spent


def kv_q8_layers_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    if seen is None or (spent := _q8_layers(situation, engine, seen)) is None:
        return None
    base, more = spent
    kv = device_kv(Placement(**seen.placed), Fit2(**seen.fit), _layers())
    return (
        f"a q8_0 KV cache frees about {kv * Q8_KV_SAVING / MIB:,.0f} MiB, room for "
        f"{_moved_words(seen, base, more)} (about {more.saved_ms - base.saved_ms:.1f} ms per "
        "token saved, estimated); the requests come one at a time, so it runs one slot (-np 1), "
        "where this model's attention (4 query heads per KV head) read a q8_0 cache as fast as "
        "f16 in Tensward's research"
    )


def with_kv_q8_layers(situation: Situation, engine: Engine) -> Settings:
    seen = view(situation)
    if seen is None or (spent := _q8_layers(situation, engine, seen)) is None:
        return situation.settings
    settings = engine.with_lever(situation.settings, "kv_cache_precision", "8bit")
    settings = dataclasses.replace(settings, max_concurrent_requests=1)
    return _with_spent(settings, seen, spent[1])


@dataclass(frozen=True, slots=True)
class ContextTrade:
    """A change refused only for device room, and the smaller context that makes room for it."""

    entry: str  # the refused playbook entry
    change: str  # its flags
    needs: int  # device bytes it needs
    room: int  # device bytes the launch left, after the margin
    slots: int
    pool: int  # the --ctx-size the user reserved
    cut: int  # the --ctx-size that frees the room, never below the measured requests
    freed: int  # device bytes the cut frees
    longest: int  # tokens of the longest measured request, prompt and answer
    why: str  # the refused entry's own reason


def _context_trade(situation: Situation, engine: Engine) -> ContextTrade | None:
    """For a launch whose context the user sized and that runs on the device: a larger -ub that
    helps the goal and is refused only for device room, with --ctx-size cut by the tokens whose
    KV frees that room, rounded as a cut pool is, and never below the longest measured request
    on each slot (sized as buy-gpu-room cuts a pool)."""
    seen = view(situation)
    pool = situation.settings.max_context_len
    if seen is None or not pool or seen.offloaded or (room := seen.room) is None:
        return None
    growth = _ubatch_growth(situation, seen)
    per_token = _kv_token_bytes(seen, _layers())
    if growth is None or per_token <= 0 or _next_ubatch(situation, seen) is not None:
        return None
    larger, extra = growth
    slots = _slots(situation, seen.started) or 1
    longest = _longest_request(situation, seen)
    short = int(-(-(extra - room) // per_token))
    cut = max(_pool_for(longest, slots), (pool - short) // POOL_ROUND * POOL_ROUND)
    if cut >= pool:
        return None
    freed = int((pool - cut) * per_token)
    if why := raise_ubatch_applies(situation, engine, freed):
        return ContextTrade(
            "raise-ubatch", f"-ub {larger}", extra, max(room, 0), slots, pool, cut, freed,
            longest, why,
        )  # fmt: skip
    return None


ANY_ROOM = 2**62  # freed bytes that stand for "however much it takes"
FREE_ROOM = "free it by lowering --ctx-size or --parallel"


def _free_room(seen: View) -> str:
    """The ways to free device memory for this run: on an offloaded run, a smaller --ctx-size or
    fewer --parallel slots free KV memory only when the requests leave the context unused;
    otherwise moving weights to the host frees room, and slows decode."""
    smaller = "free it with a smaller --ctx-size if your requests need less, or with "
    if seen.kind == "expert_offload":
        return f"{smaller}more layers' experts on the host (raise --n-cpu-moe), which slows decode"
    if seen.kind == "dense_offload":
        return f"{smaller}fewer layers on the GPU (lower --n-gpu-layers), which slows decode"
    return FREE_ROOM


def _short(needs: int, left: int, seen: View) -> str:
    return (
        f"needs about {needs / MIB:,.0f} MiB more GPU memory and {max(left, 0) / MIB:,.0f} MiB is "
        f"left after the {DEVICE_MARGIN // MIB:,} MiB margin; {_free_room(seen)}"
    )


def raise_ubatch_blocked(situation: Situation, engine: Engine) -> str | None:
    """The line for a larger -ub whose signal holds but whose compute buffer the device has no
    room for."""
    seen = view(situation)
    if seen is None or (growth := _ubatch_growth(situation, seen)) is None:
        return None
    if _next_ubatch(situation, seen) is not None or (room := seen.room) is None:
        return None
    if not raise_ubatch_applies(situation, engine, ANY_ROOM):
        return None
    return f"-ub {growth[0]} would help but {_short(growth[1], room, seen)}"


TRADE_MOST_LAYERS = 3  # more blocks of experts on the host than this cost decode too much
RAISE_UBATCH_EXPERT_TRADE = (
    'engine.build >= 12246 and engine.build < 13000 and run.placement == "expert_offload" '
    "and census.graphs_off and census.host_passes_per_request >= 4 "
    'and census.copy_wait_share >= 0.20 and memory.topology == "discrete"'
)


@dataclass(frozen=True, slots=True)
class UbatchTrade:
    current: int  # today's -ub
    rung: int  # the -ub the trade sets
    needs: int  # device bytes the larger compute buffer adds
    room: int  # device bytes the launch left, after the margin
    host_layers: int  # blocks whose experts are on the host today (--n-cpu-moe)
    more: int  # blocks whose experts the trade moves to the host as well
    freed: int  # device bytes those experts free


def _ubatch_trade(situation: Situation) -> UbatchTrade | None:
    """The largest -ub rung for experts on the host whose larger compute buffer the experts of
    the fewest more blocks (at most three) pay for, when the next rung does not fit as it is;
    the start fits by construction: the freed bytes are at least the bytes the buffer adds."""
    seen = view(situation)
    anatomy = analysis().anatomy
    if seen is None or anatomy is None or seen.kind != "expert_offload":
        return None
    if seen.placed.get("host_layers") or (room := seen.room) is None:
        return None
    if _ubatch_growth(situation, seen) is None or _next_ubatch(situation, seen) is not None:
        return None
    current = seen.started.ubatch or UBATCH_LADDER[0]
    host_layers = int(seen.placed.get("host_expert_layers") or 0)
    for rung in sorted((r for r in _ubatch_rungs(seen) if r > current), reverse=True):
        if (growth := _growth_at(situation, seen, rung)) is None:
            continue
        found = experts_to_host(anatomy, host_layers, growth[1] - room, TRADE_MOST_LAYERS)
        if found is not None:
            return UbatchTrade(current, rung, growth[1], room, host_layers, *found)
    return None


def _decode_cost(more: int, freed: int) -> str:
    """What decode pays: the share of a decode step the moved experts add, estimated from the
    host bytes a step reads and the host's share of the step."""
    facts, anatomy = analysis().kb_facts, analysis().anatomy
    moved = (
        f"decode reads the experts of {more} more {'layer' if more == 1 else 'layers'} from "
        "host memory, so TPOT rises"
    )
    if facts is None or anatomy is None:
        return f"{moved} (not estimated here)"
    host_bytes, host_share = facts["run.host_bytes_per_token"], facts["run.host_time_share"]
    if not isinstance(host_bytes, (int, float)) or not isinstance(host_share, (int, float)):
        return f"{moved} (not estimated here)"
    running = facts["run.mean_running"]
    at = max(1, round(running)) if isinstance(running, (int, float)) else 1
    read = freed * active_share(anatomy, at)
    if not host_bytes:
        return f"{moved} (not estimated here)"
    return f"{moved}: about +{read / host_bytes * host_share:.0%} (estimated, not measured)"


def raise_ubatch_expert_trade_applies(situation: Situation, engine: Engine) -> str | None:
    """The census saw the host experts copied to the GPU many times a request while the GPU
    waited on the copies, and -ub 2048 has no device room: buy the room with a few more
    blocks' experts on the host."""
    anatomy = analysis().anatomy
    if anatomy is None or host_experts_unsafe(anatomy, _build_number()):
        return None
    if not facts_hold(RAISE_UBATCH_EXPERT_TRADE) or not raise_ubatch_blocked(situation, engine):
        return None
    if (trade := _ubatch_trade(situation)) is None or not _computed_long(situation):
        return None
    passes, wait = _reading("census.host_passes_per_request"), _reading("census.copy_wait_share")
    moved = trade.host_layers + trade.more
    return (
        f"the GPU census saw the experts in host memory copied to the GPU {passes:.1f} times "
        f"per request, and copies with no kernel running filled {wait:.0%} of the time: each "
        f"micro-batch of the prompt streams them again; -ub {trade.current} to {trade.rung} "
        f"streams them {trade.rung // trade.current} times less often, but its compute buffer "
        "needs about "
        f"{trade.needs / MIB:,.0f} MiB more GPU memory and {max(trade.room, 0) / MIB:,.0f} MiB "
        f"is left after the {DEVICE_MARGIN // MIB:,} MiB margin; --n-cpu-moe {trade.host_layers} "
        f"to {moved} frees about {trade.freed / MIB:,.0f} MiB; the cost: "
        f"{_decode_cost(trade.more, trade.freed)}"
        f"{_pinned_carry(situation, trade) or ''}"
        f"{_trade_microbatches(situation, trade)}"
    )


def _trade_microbatches(situation: Situation, trade: UbatchTrade) -> str:
    seen = view(situation)
    return "" if seen is None else _microbatches(situation, seen, trade.current, trade.rung)


def _computed_long(situation: Situation) -> bool:
    """Whether the prompts still need more than one micro-batch after the prompt cache: a larger
    -ub has nothing to remove otherwise (L35: 115 tokens computed a turn, 0 of 150 over 512)."""
    seen = view(situation)
    larger = UBATCH_LADDER[1]
    return seen is not None and _long_prompts(situation, seen, larger) is not None


def _mapped(situation: Situation, seen: View | None) -> bool:
    """Whether the host weights are read through mmap and no --load-mode is set."""
    if seen is None or "--load-mode" in situation.settings.extra_args:
        return False
    return seen.started.mmap_with_overrides or seen.started.load_mode == "mmap"


def _pinned_carry(situation: Situation, trade: UbatchTrade) -> str | None:
    """The trade's words for the --load-mode none it carries on a mapped file (every research
    win of the trade had it pinned), when the host RAM allows the lock the offer of its own
    asks for, with the experts the trade moves counted; None when it is not carried."""
    seen = view(situation)
    if seen is None or not _mapped(situation, seen):
        return None
    host = (seen.placed.get("host_weight_bytes") or 0) + trade.freed
    price = _lock_price(seen, host, "--load-mode none")
    if isinstance(price, HeldBack):
        return None
    locks = f": {price[0].lower()}{price[1:]}" if price else ""
    return f"; it also loads the host experts with --load-mode none (pinned){locks}"


def with_ubatch_trade(situation: Situation, engine: Engine) -> Settings:
    if (trade := _ubatch_trade(situation)) is None:
        return situation.settings
    settings = apply_flag(situation.settings, "--ubatch-size", trade.rung)
    if (settings.prefill_batch_tokens or DEFAULT_BATCH) < trade.rung:
        settings = apply_flag(settings, "--batch-size", trade.rung)
    if _pinned_carry(situation, trade) is not None:
        settings = apply_flag(settings, "--load-mode", "none")
    return apply_flag(settings, "--n-cpu-moe", trade.host_layers + trade.more)


OFFLOAD_UBATCH_COPIES = (
    'engine.build >= 12246 and engine.build < 13000 and run.placement == "expert_offload" '
    'and memory.topology == "discrete" and census.graphs_off '
    "and census.host_passes_per_request >= 0.3 and census.busy_share <= 0.5"
)
OFFLOAD_UBATCH_RESEARCH = (
    "research, not measured by Tensward: on Qwen3.5-35B-A3B Q4_K_M offloaded to one L4 at a 12 GB "
    "budget (b12603, one stream) each micro-batch paid a fixed 0.69 s of expert uploads plus 0.33 "
    "ms per token, and a cold 16k prompt took 67.5 s at the defaults, 29.1 s with --load-mode none "
    "and 11.7 s with -ub 2048 on top, decode unchanged"
)


UNMEASURED_LINK_SHARE = 0.73  # of the link's theoretical rate (GLM2 H100: 23 of 31.5 GB/s)
UPLOADS_IN_PROMPT = 0.3  # share of the long prompts' time the expert uploads must explain
OFFLOAD_UBATCH_UNCENSUSED = (
    'engine.build >= 12246 and engine.build < 13000 and run.placement == "expert_offload" '
    'and memory.topology == "discrete"'
)


def _census_ran() -> bool:
    facts: Mapping[str, object] = analysis().kb_facts or {}
    return isinstance(facts.get("census.busy_share"), (int, float))


def _uploads_in_prompt_time(situation: Situation) -> str | None:
    """Without a census: the share of the long prompts' server time that one upload of the host
    experts per micro-batch explains, at the link rate this run knows, when it is at least
    ``UPLOADS_IN_PROMPT`` (GLM-5.3-Flash Q2, one H100: 14 micro-batches of 512 on a 7,066-token
    prompt, 1.2 s each at 23 GB/s, 35% of its 46.0 s; the census saw 1.09 passes a request at
    25% busy)."""
    seen = view(situation)
    if seen is None or not facts_hold(OFFLOAD_UBATCH_UNCENSUSED):
        return None
    point = seen.probes.get("prefill") or {}
    host = seen.placed.get("host_expert_bytes")
    tokens, ms, ubatch = point.get("prompt_tokens"), point.get("prompt_ms"), point.get("ubatch")
    if not (host and tokens and ms and ubatch):
        return None
    link = _ubatch_link(seen)
    if link is None:
        return None
    gbs, source = link
    batches = -(-int(tokens) // int(ubatch))
    uploads = batches * host / (gbs * 1e9) * 1e3
    if uploads < UPLOADS_IN_PROMPT * ms:
        return None
    return (
        f"a {int(tokens):,}-token prompt took {ms / 1e3:.1f} s in {batches} micro-batches, and "
        f"uploading the {host / 1e9:.1f} GB of host experts once per micro-batch at "
        f"{gbs:.1f} GB/s ({source}) explains {uploads / ms:.0%} of it"
    )


def _ubatch_link(seen: View) -> tuple[float, str] | None:
    pcie = seen.probes.get("pcie") or {}
    if pcie.get("gbs_effective"):
        return float(pcie["gbs_effective"]), "measured"
    if pcie.get("stored_gbs"):
        return float(pcie["stored_gbs"]), "measured by an earlier run on this host"
    if pcie.get("theoretical_gbs"):
        share = UNMEASURED_LINK_SHARE
        return share * pcie["theoretical_gbs"], f"{share:.0%} of the link's theoretical rate"
    return None


def offload_ubatch_copies_applies(situation: Situation, engine: Engine) -> str | None:
    """The census saw the experts in host memory copied to the GPU for a large share of each
    request while the GPU sat mostly idle, and -ub 2048 has no device room: the same trade as
    ``raise_ubatch_expert_trade_applies``, on readings that agent turns reach."""
    anatomy = analysis().anatomy
    if anatomy is None or host_experts_unsafe(anatomy, _build_number()):
        return None
    if facts_hold(RAISE_UBATCH_EXPERT_TRADE):
        return None
    timed = None if _census_ran() else _uploads_in_prompt_time(situation)
    if timed is None and not facts_hold(OFFLOAD_UBATCH_COPIES):
        return None
    if not raise_ubatch_blocked(situation, engine) or (trade := _ubatch_trade(situation)) is None:
        return None
    if not _computed_long(situation):
        return None
    if timed is not None:
        seen_by = (
            f"no GPU census ran; the server's own prompt timings say {timed}: each micro-batch "
            f"of {trade.current} prompt tokens uploads the experts in host memory again"
        )
    else:
        passes, busy = _reading("census.host_passes_per_request"), _reading("census.busy_share")
        h2d = _reading("census.h2d_bytes_per_request")
        seen_by = (
            f"the GPU census saw {h2d / 1e9:.1f} GB copied from host to GPU per request, "
            f"{passes:.2f} times the experts kept in host memory, while the GPU ran kernels only "
            f"{busy:.0%} of the time: each micro-batch of {trade.current} prompt tokens uploads "
            "those experts again"
        )
    moved = trade.host_layers + trade.more
    return (
        f"{seen_by}; -ub {trade.rung} uploads them for {trade.rung // trade.current} times as "
        "many tokens, but its "
        f"compute buffer needs about {trade.needs / MIB:,.0f} MiB more GPU memory and "
        f"{max(trade.room, 0) / MIB:,.0f} MiB is left after the {DEVICE_MARGIN // MIB:,} MiB "
        f"margin; --n-cpu-moe {trade.host_layers} to {moved} frees about "
        f"{trade.freed / MIB:,.0f} MiB; the cost: {_decode_cost(trade.more, trade.freed)}"
        f"{_pinned_carry(situation, trade) or ''}{_trade_microbatches(situation, trade)}; "
        f"{OFFLOAD_UBATCH_RESEARCH}"
    )


def mtp_speculation_blocked(situation: Situation, engine: Engine) -> str | None:
    """The line for the model's own MTP drafts where only the device room refuses them."""
    if analysis().goal not in ("tpot", "throughput") or _mtp_offer(situation, engine):
        return None
    if _mtp_hybrid_slots_crash(situation) or _mtp_host_experts(situation):
        return None
    seen = view(situation)
    if seen is None or not _mtp_room(situation, engine, ANY_ROOM):
        return None
    left = _mtp_left(situation, seen)
    draft = _mtp_draft_bytes(seen, _mtp_depth(situation, seen))
    if left is None or draft is None:
        return None
    return (
        f"the model's MTP drafts (--spec-type draft-mtp) would help but {_short(draft, left, seen)}"
    )


def context_for_room_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    if seen is None or (t := _context_trade(situation, engine)) is None:
        return None
    if seen.started.kv_unified:
        reserved, limit = "a KV pool any one request can fill", t.cut
    else:
        reserved, limit = f"{t.slots} slots of {t.pool // t.slots:,} tokens", t.cut // t.slots
    return (
        f"{t.why}; `{t.entry}` needs about {t.needs / MIB:,.0f} MiB more GPU memory at today's "
        f"context and only {t.room / MIB:,.0f} MiB is left after the {DEVICE_MARGIN // MIB:,} MiB "
        f"margin; --ctx-size {t.pool} reserves {reserved}, while the longest measured request "
        f"is {t.longest:,} tokens (prompt and answer); --ctx-size {t.cut} frees about "
        f"{t.freed / MIB:,.0f} MiB, so {t.change} fits; the cost: a request longer than the "
        f"measured ones may not fit, and none past {limit:,} tokens does; documented, not "
        "measured: the -ub change is measured, this trade is not"
    )


def with_context_for_room(situation: Situation, engine: Engine) -> Settings:
    if (t := _context_trade(situation, engine)) is None:
        return situation.settings
    settings = dataclasses.replace(situation.settings, max_context_len=t.cut)
    return with_larger_ubatch(dataclasses.replace(situation, settings=settings), engine, t.freed)


def _env_set(situation: Situation, name: str) -> bool:
    """Whether the run sets the engine variable ``name``: in its settings, or inherited, as a
    kept step's environment is (the analysis' arguments carry it as NAME=VALUE)."""
    if name in situation.settings.extra_env:
        return True
    return any(arg.partition("=")[0] == name for arg in analysis().engine_args)


OP_OFFLOAD_THRESHOLD = 256  # prompt tokens per batch from which host experts are copied over
# Host expert layers from which the copies matter: 17 gained req/s +11.6%, 7 only TTFT -8%.
OP_OFFLOAD_MIN_HOST_EXPERT_LAYERS = 12


def op_offload_threshold_applies(situation: Situation, engine: Engine) -> str | None:
    """Experts of at least ``OP_OFFLOAD_MIN_HOST_EXPERT_LAYERS`` layers on the host and prompts
    that compute few tokens after the prompt cache: each such batch copies the host experts to
    the device, where the CPU would compute it sooner."""
    seen = view(situation)
    settings = situation.settings
    if seen is None or seen.kind != "expert_offload" or _unified(seen):
        return None
    if (seen.placed.get("host_expert_layers") or 0) < OP_OFFLOAD_MIN_HOST_EXPERT_LAYERS:
        return None
    if _env_set(situation, OP_OFFLOAD_MIN_BATCH) or "--op-offload" in settings.extra_args:
        return None
    prompt = _prompt_p95(situation)
    computed = int(prompt * (1 - (situation.measurement.prefix_cache_hit_rate or 0.0)))
    if not computed or computed >= OP_OFFLOAD_THRESHOLD:
        return None
    return (
        f"the experts run from host memory and prompts compute about {computed:,} tokens each "
        f"after the prompt cache (p95), so each prompt batch copies the host experts to the GPU; "
        f"{OP_OFFLOAD_MIN_BATCH}={OP_OFFLOAD_THRESHOLD} computes batches under "
        f"{OP_OFFLOAD_THRESHOLD} tokens on the CPU instead and keeps the copy for long prompts"
    )


def with_op_offload_threshold(situation: Situation, engine: Engine) -> Settings:
    settings = situation.settings
    env = {**settings.extra_env, OP_OFFLOAD_MIN_BATCH: str(OP_OFFLOAD_THRESHOLD)}
    return dataclasses.replace(settings, extra_env=env)


def no_op_offload_applies(situation: Situation, engine: Engine) -> str | None:
    """Never offered until a leg measures how much of the host memory bandwidth a CPU reaches
    on a ubatch of prompt tokens, the figure the PCIe rate has to be compared with."""
    return None


def with_no_op_offload(situation: Situation, engine: Engine) -> Settings:
    return apply_flag(situation.settings, "--op-offload", False)


# --- host -------------------------------------------------------------------------------


def pin_weights_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    if seen is None or not seen.placed.get("host_weight_bytes"):
        return None
    faults = seen.block.get("major_faults")
    taken = faults["end"] - faults["start"] if faults else 0
    if taken > MAJOR_FAULTS_TIGHT:
        return f"{taken} pages of the weights were read back from disk while decoding"
    if seen.regime.get("host_memory_tight"):
        return "host memory is tight, so the weights it holds may be paged out and read back"
    return None


def with_load_mode(situation: Situation, engine: Engine) -> Settings:
    seen = view(situation)
    if seen is None:
        return situation.settings
    weights_gib = seen.placed["host_weight_bytes"] / 2**30
    memlock = seen.environment.get("memlock_gib")
    mode = "mlock" if memlock is None or memlock >= weights_gib else "none"
    return apply_flag(situation.settings, "--load-mode", mode)


HUGETLB = "glibc.malloc.hugetlb=1"
HUGETLB_FAULTS_PER_REQUEST = 5000
"""Minor page faults per request from which the server's host buffers are offered huge pages:
at 4,600 to 4,800 (GLM-4-9B returning users, L4) they changed nothing."""
HUGETLB_BUILD = 12246  # the build the faults and the gain were measured on
HUGETLB_BUILD_BELOW = 13000  # the record's engine_version ends here
HUGETLB_GLIBC = (2, 35)  # glibc.malloc.hugetlb exists from this version


def hugetlb_state_save_applies(situation: Situation, engine: Engine) -> str | None:
    """The server faulted in fresh host pages for every request, the cost of the slot state it
    saves and restores in host memory, and none of its memory is in huge pages."""
    seen = view(situation)
    if seen is None or seen.environment.get("platform") != "nvidia" or _mac(seen):
        return None
    if _env_set(situation, GLIBC_TUNABLES):
        return None
    found = known_build()
    if found is None or not HUGETLB_BUILD <= found[0].build < HUGETLB_BUILD_BELOW:
        return None
    libc = glibc_version()
    if libc is None or libc < HUGETLB_GLIBC or transparent_hugepages() == "never":
        return None
    faults, requests = seen.block.get("host_faults") or {}, situation.measurement.succeeded
    if not requests or faults.get("anon_hugepages_kib") != 0:
        return None
    per_request = faults["minor_faults"] / requests
    if per_request < HUGETLB_FAULTS_PER_REQUEST:
        return None
    return (
        f"the server took {per_request:,.0f} minor page faults per request and holds no huge "
        f"pages: the slot state it saves in host memory is faulted in again for each request; "
        f"{GLIBC_TUNABLES}={HUGETLB} backs those buffers with 2 MiB pages"
    )


def with_hugetlb(situation: Situation, engine: Engine) -> Settings:
    settings = situation.settings
    return dataclasses.replace(settings, extra_env={**settings.extra_env, GLIBC_TUNABLES: HUGETLB})


STATE_SAVE_OFF = {"--cache-ram": "0", "--ctx-checkpoints": "0"}
QWEN38_27B = (
    'model.kind == "hybrid" and model.family == "qwen35" '
    "and model.active_params_b >= 26 and model.active_params_b <= 28"
)
STATE_SAVE_OFF_TOOLS = (
    f'engine.build == 12246 and {QWEN38_27B} and gpu.arch == "hopper" and run.slots == 16 '
    "and workload.tools and workload.prompt_tokens_max <= 800"
)
STATE_SAVE_OFF_SHORT = (
    f'engine.build == 12246 and {QWEN38_27B} and gpu.arch == "ada" and run.slots == 8 '
    "and not workload.tools and workload.prompt_tokens_max <= 128 "
    "and workload.output_tokens_p50 <= 4"
)


def _state_saved(settings: Settings) -> bool:
    """Whether the prompt cache and the checkpoints are at llama.cpp's defaults."""
    return not any(flag in settings.extra_args for flag in STATE_SAVE_OFF)


CENSUS_BUILDS = "engine.build >= 12246 and engine.build < 13000"
FULL_GPU_ADA_HOPPER = 'run.placement == "full_gpu" and gpu.arch in ["ada", "hopper"]'
STATE_SAVE_OFF_GPU_WAITS = (
    f"{CENSUS_BUILDS} and anatomy.linear_attention and {FULL_GPU_ADA_HOPPER} "
    "and workload.prompt_tokens_max <= 2048 and workload.clients >= 8 "
    "and census.graphs_off and census.busy_share < 0.70"
)


def _reading(name: str) -> float:
    """A census reading of this run's facts, for a reason whose condition already read it."""
    facts = analysis().kb_facts
    value = facts[name] if facts is not None else None
    return float(value) if isinstance(value, (int, float)) else 0.0


def _rereads(situation: Situation) -> str:
    """What the state saves off cost: every returning request computes its whole prompt again,
    the cached part included, at this run's measured prompt rate."""
    facts = analysis().kb_facts
    longest = facts.get("workload.prompt_tokens_max") if facts is not None else None
    measurement = getattr(situation, "measurement", None)
    rate = getattr(measurement, "prompt_capacity_tok_s", None)
    cost = "every returning request re-reads its whole prompt"
    if not isinstance(longest, (int, float)) or not longest:
        return cost
    if not isinstance(rate, (int, float)) or rate <= 0:
        return f"{cost} (up to {longest:,.0f} tokens)"
    return f"{cost}: about {longest / rate:.1f} s ({longest:,.0f} tokens at {rate:,.0f} a second)"


def _similarity_off(settings: Settings) -> bool:
    return str(settings.extra_args.get("--slot-prompt-similarity")) in ("0", "0.0")


def state_save_off_gpu_waits_applies(situation: Situation, engine: Engine) -> str | None:
    """The census saw the GPU idle most of the window while a linear-attention model saves
    every request's state at the defaults. Never with --slot-prompt-similarity 0, which already
    sends a returning conversation to any free slot."""
    if not _state_saved(situation.settings) or _similarity_off(situation.settings):
        return None
    if not facts_hold(STATE_SAVE_OFF_GPU_WAITS):
        return None
    return (
        f"the GPU census found the GPU running kernels only "
        f"{_reading('census.busy_share'):.0%} of the time; the rest it waited while the server "
        "saved every request's state to host memory (a prompt-cache entry and context "
        "checkpoints); --cache-ram 0 --ctx-checkpoints 0 stops the saves, and then "
        f"{_rereads(situation)}"
    )


def _census_saves_off(situation: Situation, engine: Engine) -> bool:
    """Whether the census entry makes the same change here: then a pinned entry steps aside,
    so one offer carries what the census read. Not for a TTFT goal, which the census entry
    never gets (its first token comes later on returning conversations)."""
    return analysis().goal != "ttft" and bool(state_save_off_gpu_waits_applies(situation, engine))


def state_save_off_tools_applies(situation: Situation, engine: Engine) -> str | None:
    if not _state_saved(situation.settings) or not facts_hold(STATE_SAVE_OFF_TOOLS):
        return None
    if _census_saves_off(situation, engine):
        return None
    return (
        "this hybrid model's server saves every request's state to host memory, a prompt-cache "
        "entry and context checkpoints, while the GPU waits; --cache-ram 0 --ctx-checkpoints 0 "
        "stops the saves, and each request then reads its whole prompt again"
    )


def state_save_off_short_applies(situation: Situation, engine: Engine) -> str | None:
    if not _state_saved(situation.settings) or not facts_hold(STATE_SAVE_OFF_SHORT):
        return None
    if _census_saves_off(situation, engine):
        return None
    return (
        "this hybrid model's server saves every request's state to host memory, a prompt-cache "
        "entry and context checkpoints, while the GPU waits, and with answers of a few tokens "
        "the saves are most of each request; --cache-ram 0 --ctx-checkpoints 0 stops them"
    )


def with_state_save_off(situation: Situation, engine: Engine) -> Settings:
    settings = situation.settings
    for flag, value in STATE_SAVE_OFF.items():
        settings = apply_flag(settings, flag, value)
    return settings


BACKEND_SAMPLING = "--backend-sampling"
CHAT_32_SLOTS_HOPPER = (
    'engine.build == 12246 and gpu.arch == "hopper" and run.slots == 32 '
    "and not workload.tools and workload.output_tokens_p50 >= 128 "
    "and workload.prompt_tokens_max <= 128"
)
BACKEND_SAMPLING_GEMMA4 = (
    f'{CHAT_32_SLOTS_HOPPER} and model.family == "gemma4" '
    "and model.params_b >= 7 and model.params_b <= 8"
)
BACKEND_SAMPLING_HYBRID = f"{CHAT_32_SLOTS_HOPPER} and {QWEN38_27B}"
SAMPLED_ON_HOST = (
    "every decode step copies the logits of each running request to the host and samples them "
    "on the CPU while the GPU waits; --backend-sampling samples on the GPU"
)


BACKEND_SAMPLING_GPU_WAITS = (
    f"{CENSUS_BUILDS} and {FULL_GPU_ADA_HOPPER} and not workload.tools "
    "and workload.output_tokens_p50 >= 64 and census.graphs_off and census.busy_share < 0.80 "
    "and census.logits_on_host_share >= 0.9 and census.logits_on_host_share <= 1.5"
)


def backend_sampling_gpu_waits_applies(situation: Situation, engine: Engine) -> str | None:
    """The census saw the GPU idle and about one row of logits per output token copied to the
    host. A linear-attention model whose state saves are still on gets the saves offered first:
    they were the larger stall, and the census after that keep decides this one."""
    settings = situation.settings
    if BACKEND_SAMPLING in settings.extra_args:
        return None
    if not _saves_off(settings) and not facts_hold("not anatomy.linear_attention"):
        return None
    if not facts_hold(BACKEND_SAMPLING_GPU_WAITS):
        return None
    return (
        f"the GPU census found the GPU running kernels only "
        f"{_reading('census.busy_share'):.0%} of the time, and "
        f"{_reading('census.logits_on_host_share'):.2f} rows of logits per output token were "
        "copied to the host: every decode step samples on the CPU while the GPU waits; "
        "--backend-sampling samples on the GPU"
    )


def backend_sampling_gemma4_applies(situation: Situation, engine: Engine) -> str | None:
    settings = situation.settings
    if BACKEND_SAMPLING in settings.extra_args or not _state_saved(settings):
        return None
    if not facts_hold(BACKEND_SAMPLING_GEMMA4):
        return None
    return None if backend_sampling_gpu_waits_applies(situation, engine) else SAMPLED_ON_HOST


def _saves_off(settings: Settings) -> bool:
    """Whether the prompt cache and the checkpoints are both off."""
    return all(settings.extra_args.get(flag) == value for flag, value in STATE_SAVE_OFF.items())


def backend_sampling_hybrid_applies(situation: Situation, engine: Engine) -> str | None:
    """With the state saves already off, as measured: they were the larger stall."""
    settings = situation.settings
    if BACKEND_SAMPLING in settings.extra_args or not _saves_off(settings):
        return None
    if not facts_hold(BACKEND_SAMPLING_HYBRID):
        return None
    return None if backend_sampling_gpu_waits_applies(situation, engine) else SAMPLED_ON_HOST


def with_backend_sampling(situation: Situation, engine: Engine) -> Settings:
    return apply_flag(situation.settings, BACKEND_SAMPLING, True)


PINNED_FIRST_SHARE = 0.5  # pageable share of host-to-GPU bytes above which pinning goes first


def pageable_h2d() -> str | None:
    """The census's own reading of the host-to-GPU copies, when most come from pageable memory:
    pinned host buffers copy them faster (L35: 97% pageable at 6.5 GB/s, 1% at 11.9 once
    pinned; pinned loading gave +28 to +42% in three measured runs)."""
    facts = analysis().kb_facts
    share = facts.get("census.pageable_h2d_share") if facts is not None else None
    if not isinstance(share, (int, float)) or share < PINNED_FIRST_SHARE:
        return None
    rate = facts.get("census.h2d_gb_per_s") if facts is not None else None
    at = f" at {rate:.1f} GB/s" if isinstance(rate, (int, float)) else ""
    return (
        f"the GPU census saw {share:.0%} of the host-to-GPU bytes copied from pageable memory"
        f"{at}, which pinned memory copies faster"
    )


LOAD_MODE_RESERVE_GIB = 8
"""The host RAM a lock must leave of what the run measured available while its server ran."""


def load_mode_none_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    if seen is None or not seen.offloaded or _unified(seen):
        return None
    if not seen.placed.get("host_weight_bytes") or "--load-mode" in situation.settings.extra_args:
        return None
    started = seen.started
    if not (started.mmap_with_overrides or started.load_mode == "mmap"):
        return None
    reason = (
        "llama.cpp reads the CPU-offloaded experts through mmap; loading them into memory once "
        "is faster"
    )
    if (pageable := pageable_h2d()) is not None:
        reason = f"{pageable}; {reason}"
    price = _lock_price(seen, seen.placed["host_weight_bytes"], "--load-mode none")
    if price is None or isinstance(price, HeldBack):
        return price or reason
    return f"{reason}. {price}, and a warm start takes about 7 s longer"


def _lock_price(seen: View, host_bytes: float, what: str) -> str | HeldBack | None:
    """What locking ``host_bytes`` of host RAM leaves; held back when that is under the
    reserve, None when the run did not read the RAM."""
    host = host_bytes / 2**30
    env = seen.environment
    total, free = env.get("mem_gib"), env.get("mem_available_gib")
    if total is None or free is None:
        return None
    limit, used = env.get("cgroup_mem_gib"), env.get("cgroup_mem_used_gib")
    if limit is not None and limit < total:
        total, free = limit, min(free, limit - (used or 0.0))
    reserve = LOAD_MODE_RESERVE_GIB
    if free - host < reserve:
        return HeldBack(
            f"{what} would lock {host:.1f} GiB of host RAM, leaving about "
            f"{free - host:.1f} GiB of {total:.0f} GiB; it needs {reserve:.1f} GiB left over"
        )
    return (
        f"It locks {host:.1f} GiB of host RAM while the server runs, leaving about "
        f"{free - host:.1f} GiB"
    )


def with_load_mode_none(situation: Situation, engine: Engine) -> Settings:
    return apply_flag(situation.settings, "--load-mode", "none")


CACHE_SLOTS = 1  # WB1 measured one slot; four clients with a small cache measured worse
CACHE_BUILDS = range(12257, 13000)  # WB1's build b12257 up to the record's range
# Q8_0 experts at the default CPU share 0 (one L4, b12257, one slot): a 17,900 MiB cache beat the
# fit by 29-36%; 11,328 MiB tied it (x0.99) and 7,296 MiB was slower than no cache (x0.86)
CACHE_MIN_MIB_8BIT = 17_900
EIGHT_BIT_EXPERTS = frozenset({"Q8_0", "F16", "BF16", "F32"})
# The cache pays when the experts it still uploads per token cost less than the host part of
# today's time per token. Misses per token = MoE layers x experts per token x a x (1 - c)^p, with
# c the cached share of all experts, on a curve fitted per model family: GLM-5.3-Flash Q2 (four
# cache sizes) and Qwen3.5-35B-A3B (CM1), whose routing is less flat. Another family is
# predicted with both, and is unsure where they disagree
CACHE_CURVE_GLM = (0.48, 1.6)
CACHE_CURVE_QWEN = (0.28, 1.9)
CACHE_FAMILY_CURVES = {
    "glm5-next": ("GLM-5.3-Flash", CACHE_CURVE_GLM),
    "qwen35moe": ("Qwen3.5-35B-A3B", CACHE_CURVE_QWEN),
}
CACHE_PAYS = 0.9  # the predicted upload must stay under this share of the host part
CACHE_MISS_COST = 1.13  # a miss costs this much more than its bare copy (WB1)
HOST_SHARE_MIDDLE = 0.5  # of the decode model's upper bound, where no probe measured the host
CACHE_PINNED = 0.1  # a census pageable share under which its copy rate is the pinned rate
CACHE_LINK_SHARE = 0.73  # of the link's theoretical rate, where no pinned rate was measured
# Device memory the cache holds past its logged size: GLM-5.3-Flash Q2 at -ub 2048 used 455 MiB
# above its logged buffers with a 69,480 MiB cache and 315 MiB without one
CACHE_UNSEEN = 140 * MIB
CACHE_MEASURED = (
    "research, not measured by Tensward: against the launch without a cache x1.64 requests per "
    "second on GLM-5.3-Flash Q2 (one H100, PCIe gen4, b12603), x1.09 to x1.20 on Qwen3.5-35B-A3B "
    "Q3_K_M and Q4_K_M at a 16 GB budget (L4 and H100), and x0.6 to x0.85 on GLM-5.3-Flash Q4, "
    "where the prediction is over the host part; one slot only; answers are expected to differ "
    "from today's launch (30-60% identical at one client in the small-cache runs)"
)


def _loaded_experts(situation: Situation) -> tuple[int, ...]:
    """The routed-expert bytes of each block llama.cpp loads: the next-token blocks only when
    the launch drafts with them (GLM-5.3-Flash Q2's MTP block holds 2,502 MiB the log never
    places)."""
    anatomy = analysis().anatomy
    if anatomy is None:
        return ()
    blocks = anatomy.expert_layer_bytes
    if anatomy.mtp_layers and not drafts_mtp(situation.settings.extra_args):
        blocks = blocks[: len(blocks) - anatomy.mtp_layers]
    return tuple(blocks)


def _cache_bytes(situation: Situation, seen: View) -> int | None:
    """The cache --cpu-moe leaves room for: the device memory the launch left allocatable
    (the fit's free less the margin, at most what the run's peak left less the margin), plus
    the loaded experts it keeps on the device, less what the cache holds past its size. The
    compute buffer is today's: a --cpu-moe start at the same -ub logs the same one (2,069 MiB at
    -ub 2048 with 16 and with all of GLM-5.3-Flash Q2's expert layers on the host)."""
    experts, room = sum(_loaded_experts(situation)), seen.room
    if not experts or room is None or room < 0 or seen.placed.get("host_layers"):
        return None
    if (usable := _usable(seen)) is not None:
        room = min(room, usable - DEVICE_MARGIN)
    on_device = max(experts - seen.placed.get("host_expert_bytes", 0), 0)
    size = min(room + on_device - CACHE_UNSEEN, experts)
    return size if size > 0 else None


def _cache_link(seen: View) -> tuple[float, str] | None:
    """The rate the cache's misses cross the link at, in GB/s, and where it comes from: the
    census's copies when they were pinned, else a share of the link's theoretical rate."""
    facts: Mapping[str, object] = analysis().kb_facts or {}
    rate, pageable = facts.get("census.h2d_gb_per_s"), facts.get("census.pageable_h2d_share")
    if isinstance(rate, (int, float)) and rate > 0 and isinstance(pageable, (int, float)):
        if pageable < CACHE_PINNED:
            return float(rate), "the census's pinned copies"
    pcie = seen.probes.get("pcie") or {}
    theoretical = pcie.get("theoretical_gbs") or facts.get("gpu.pcie_gbs_theoretical")
    if isinstance(theoretical, (int, float)) and theoretical > 0:
        gen, width = pcie.get("gen"), pcie.get("width")
        link = f"gen{gen} x{width} " if gen and width else ""
        return (
            CACHE_LINK_SHARE * theoretical,
            f"{CACHE_LINK_SHARE:.0%} of the {link}link's theoretical {theoretical:.1f} GB/s",
        )
    return None


def _project_host_rate() -> float | None:
    """The host bandwidth a probe launch measured in an earlier run of this project (the same
    model on the same box), the latest one."""
    project = analysis().project
    runs = sorted((project / "runs").glob("*/run.json")) if project is not None else []
    for path in reversed(runs):
        try:
            host = (json.loads(path.read_text("utf-8")).get("probes") or {}).get("host") or {}
        except (OSError, ValueError):
            continue
        rate = host.get("mem_bw_gbs")
        if host.get("method") == "launch" and isinstance(rate, (int, float)) and rate > 0:
            return float(rate)
    return None


def _estimated_host_share(seen: View, read: int, tpot: float) -> tuple[float | None, str]:
    """The host's share of the time per token when this run measured none: at the host rate an
    earlier probe of this project measured, else half the decode model's share. The decode
    model gives the host all the time the device reads leave, an upper bound: on the 35B Q8_0
    one-slot L4 boxes it read 69-73% where the probes read 27-42%."""
    if (rate := _project_host_rate()) is not None:
        share = min(read / (rate * 1e6) / tpot, 1.0)
        return (
            share,
            f", estimated at the {rate:.1f} GB/s an earlier probe of this project measured",
        )
    rate = (seen.block.get("decode_model") or {}).get("bw_host_gbs")
    if not isinstance(rate, (int, float)) or rate <= 0:
        return None, ""
    bound = min(read / (rate * 1e6) / tpot, 1.0)
    return bound * HOST_SHARE_MIDDLE, (
        f", estimated: no host probe, so half the {bound:.0%} the decode model's {rate:.1f} GB/s "
        "host rate allows at most"
    )


@dataclass(frozen=True, slots=True)
class CacheCall:
    verdict: str  # "pays", "unsure" or "no"
    words: str


def _cache_call(situation: Situation, seen: View, size: int) -> CacheCall | str:
    """Whether the cache's predicted upload per token pays against the host part of today's
    time per token, on the model family's miss curve, else on both; or why it cannot be
    predicted here."""
    anatomy = analysis().anatomy
    facts: Mapping[str, object] = analysis().kb_facts or {}
    layers = [b for b in _loaded_experts(situation) if b]
    count, used = getattr(anatomy, "experts", None), getattr(anatomy, "experts_used", None)
    if not layers or not count or not used:
        return "the model's expert counts are unknown, so the upload per token is not predicted"
    tpot, share = situation.measurement.tpot_p50_ms, facts.get("run.host_time_share")
    estimated = ""
    read = seen.placed.get("host_bytes_per_token")
    if tpot and read and not isinstance(share, (int, float)):
        # the measured share's own basis: TPOT p95 (situation.py)
        step = situation.measurement.tpot_p95_ms or tpot
        share, estimated = _estimated_host_share(seen, read, step)
    if not tpot or not isinstance(share, (int, float)):
        return (
            "this run did not measure the host's share of the time per token, so the cache's "
            "upload per token cannot be compared with it"
        )
    if (link := _cache_link(seen)) is None:
        return "no rate of the host link is known, so the upload per token is not predicted"
    gbs, source = link
    cached = size / sum(layers)
    each = sum(layers) / (count * len(layers))
    left = max(1.0 - cached, 0.0)
    host = tpot * share

    def predict(curve: tuple[float, float]) -> tuple[float, float]:
        mib = len(layers) * used * curve[0] * left ** curve[1] * each / MIB
        return mib, mib * MIB / (gbs * 1e6) * CACHE_MISS_COST

    against = (
        f"at {gbs:.1f} GB/s ({source}), against {host:.1f} ms, the host part of today's time per "
        f"token (TPOT p50 {tpot:.1f} ms x host share {share:.0%}{estimated}); it pays under "
        f"{CACHE_PAYS:.0%} of that"
    )
    family = CACHE_FAMILY_CURVES.get(getattr(anatomy, "model_type", None) or "")
    if family is not None:
        mib, ms = predict(family[1])
        words = (
            f"with {cached:.0%} of the experts cached the miss curve fitted on {family[0]}, this "
            f"model's family, predicts {mib:,.0f} MiB uploaded a token, {ms:.1f} ms {against}"
        )
        return CacheCall("pays" if ms < CACHE_PAYS * host else "no", words)
    (glm_mib, glm_ms), (qwen_mib, qwen_ms) = predict(CACHE_CURVE_GLM), predict(CACHE_CURVE_QWEN)
    pays = [ms < CACHE_PAYS * host for ms in (glm_ms, qwen_ms)]
    words = (
        f"with {cached:.0%} of the experts cached the miss curve fitted on GLM-5.3-Flash predicts "
        f"{glm_mib:,.0f} MiB uploaded a token, {glm_ms:.1f} ms, and the one fitted on "
        f"Qwen3.5-35B-A3B {qwen_mib:,.0f} MiB, {qwen_ms:.1f} ms, {against}"
    )
    verdict = "pays" if all(pays) else "unsure" if any(pays) else "no"
    if verdict == "unsure":
        words = (
            "unsure: no miss curve is fitted on this model's family and the two fitted ones "
            f"disagree here: {words}"
        )
    return CacheCall(verdict, words)


def _expert_cache(situation: Situation) -> tuple[bool, str] | HeldBack | None:
    """(whether the prediction pays for sure, the reason) for the expert cache, a HeldBack with
    the numbers when it does not pay or the RAM cannot hold the pinned file, None where the
    cache is not for this launch."""
    seen = view(situation)
    if seen is None or seen.kind != "expert_offload" or _unified(seen):
        return None
    if build_number(situation.measurement.engine_version) not in CACHE_BUILDS:
        return None
    if _slots(situation, seen.started) != CACHE_SLOTS:
        return None
    extra = situation.settings.extra_args
    if any(
        flag in extra for flag in ("--moe-cache-mib", "--moe-cache-cpu-share", "--override-tensor")
    ):
        return None
    total, size = _total_layers(seen), _cache_bytes(situation, seen)
    if total is None or size is None:
        return None
    anatomy = analysis().anatomy
    types = getattr(anatomy, "precisions", {}).get("routed_experts", ())
    if size < CACHE_MIN_MIB_8BIT * MIB and (not types or EIGHT_BIT_EXPERTS & set(types)):
        return HeldBack(
            f"an expert cache of {size // MIB:,} MiB is under the {CACHE_MIN_MIB_8BIT:,} MiB "
            "that 8-bit experts needed to beat the launch without one (on one L4 11,328 MiB "
            "tied it and 7,296 MiB was slower)"
        )
    call = _cache_call(situation, seen, size)
    if isinstance(call, str):
        return HeldBack(f"the expert cache is not offered: {call}")
    if call.verdict == "no":
        return HeldBack(f"an expert cache of {size // MIB:,} MiB would not pay: {call.words}")
    buffer = seen.started.bytes_of("compute", "device")
    ubatch = seen.started.ubatch or UBATCH_LADDER[0]
    logged = f" ({buffer // MIB:,} MiB, as the server logged it)" if buffer else ""
    reason = (
        f"one slot: every expert on the host (--n-cpu-moe {total}, as --cpu-moe, with --load-mode "
        f"none) and an expert cache of {size // MIB:,} MiB in the GPU memory left after the "
        f"{DEVICE_MARGIN // MIB:,} MiB margin, the compute buffer at -ub {ubatch}{logged} and "
        f"{CACHE_UNSEEN // MIB} MiB the cache holds past its size; {call.words}; {CACHE_MEASURED}"
    )
    # --load-mode none with every expert on the host pins the whole file (33.2 GB for Q8_0)
    host = seen.placed["host_weight_bytes"]
    pinned = host if extra.get("--load-mode") in ("none", "mlock") else 0
    whole = seen.placed.get("device_weight_bytes", 0) + host
    price = _lock_price(seen, whole - pinned, "the whole model file pinned by --load-mode none")
    if isinstance(price, HeldBack):
        return price
    return call.verdict == "pays", reason if price is None else f"{reason}. {price}"


def expert_cache_applies(situation: Situation, engine: Engine) -> str | None:
    found = _expert_cache(situation)
    if found is None or isinstance(found, HeldBack):
        return found
    return found[1] if found[0] else None


def expert_cache_low_bar(situation: Situation, engine: Engine) -> str | None:
    """The two fitted miss curves disagree: a could-help that says it is unsure."""
    found = _expert_cache(situation)
    if found is None or isinstance(found, HeldBack) or found[0]:
        return None
    return found[1]


def with_expert_cache(situation: Situation, engine: Engine) -> Settings:
    seen = view(situation)
    size = None if seen is None else _cache_bytes(situation, seen)
    if seen is None or size is None or (total := _total_layers(seen)) is None:
        return situation.settings
    settings = apply_flag(situation.settings, "--n-gpu-layers", total)
    settings = apply_flag(settings, "--n-cpu-moe", total)
    settings = apply_flag(settings, "--moe-cache-mib", size // MIB)
    return apply_flag(settings, "--load-mode", "none")


CACHE_SHARE_MIB = 4096  # R0500M's cache with four clients (Qwen3.5-35B-A3B Q8_0, one L4)
CACHE_SHARE_HIGH = "0.95"  # the share of cache misses the CPU computes instead of uploading
CACHE_SHARE_ZERO = "0"
# 8-bit and 6-bit experts won with share 0.95 on the L4 and the H100; 3 to 5-bit ones mostly with 0
CACHE_SHARE_TYPES = frozenset({"Q8_0", "Q6_K", "F16", "BF16"})
CACHE_SHARE_BUILDS = range(12246, 13000)  # b12246 lists --moe-cache-cpu-share
CACHE_FLAGS = ("--moe-cache-mib", "--moe-cache-cpu-share", "--override-tensor")
CACHE_SHARE_RESEARCH = (
    "research, not measured by Tensward: on this model and card with four clients (b12246) "
    "--moe-cache-mib 4096 --moe-cache-cpu-share 0.95 --load-mode none cut TPOT p50 20-23% and "
    "raised tok/s 28-31%; with one stream the best share switched once along the precision "
    "ladder (8-bit 0.95, 3- and 4-bit 0) and a wrong choice cost up to 66%"
)


@dataclass(frozen=True, slots=True)
class CacheLayout:
    host_layers: int  # blocks whose experts are on the host today (--n-cpu-moe)
    more: int  # blocks whose experts move to the host to make room for the cache
    freed: int  # device bytes those experts free


def _cache_situation(situation: Situation) -> View | None:
    """The run's view when an expert cache can be offered at all: experts on the host of a
    discrete NVIDIA card, on a build with the cache's share flag."""
    seen = view(situation)
    if seen is None or seen.kind != "expert_offload" or _unified(seen):
        return None
    if seen.environment.get("platform") != "nvidia":
        return None
    if build_number(situation.measurement.engine_version) not in CACHE_SHARE_BUILDS:
        return None
    return seen


def _cache_share_layout(situation: Situation) -> tuple[View, CacheLayout] | None:
    """More than one slot, 8 or 6-bit experts and no cache yet: the experts of the fewest more
    blocks whose bytes make room for a 4,096 MiB cache beside what the launch left."""
    seen = _cache_situation(situation)
    anatomy = analysis().anatomy
    if seen is None or anatomy is None or (_slots(situation, seen.started) or 1) <= CACHE_SLOTS:
        return None
    if any(flag in situation.settings.extra_args for flag in CACHE_FLAGS):
        return None
    if seen.placed.get("host_layers") or (room := seen.room) is None:
        return None
    types = set(getattr(anatomy, "precisions", {}).get("routed_experts", ()))
    if not types or not types <= CACHE_SHARE_TYPES or (total := _total_layers(seen)) is None:
        return None
    host_layers = int(seen.placed.get("host_expert_layers") or 0)
    needs = CACHE_SHARE_MIB * MIB - max(room, 0)
    if needs <= 0:
        return seen, CacheLayout(host_layers, 0, 0)
    found = experts_to_host(anatomy, host_layers, needs, total - host_layers)
    return None if found is None else (seen, CacheLayout(host_layers, *found))


def expert_cache_share_applies(situation: Situation, engine: Engine) -> str | HeldBack | None:
    if (found := _cache_share_layout(situation)) is None:
        return None
    seen, layout = found
    pinned = situation.settings.extra_args.get("--load-mode") in ("none", "mlock")
    host = (0 if pinned else seen.placed["host_weight_bytes"]) + layout.freed
    price = _lock_price(seen, host, "the experts pinned by --load-mode none")
    if isinstance(price, HeldBack):
        return price
    moved = (
        f"--n-cpu-moe {layout.host_layers} to {layout.host_layers + layout.more} frees about "
        f"{layout.freed / MIB:,.0f} MiB for it, so decode reads those experts from the cache or "
        "the CPU; "
        if layout.more
        else ""
    )
    reason = (
        f"the experts are 8 or 6-bit and run from host memory at {_slots(situation, seen.started)} "
        f"slots: an expert cache of {CACHE_SHARE_MIB:,} MiB on the GPU with "
        f"--moe-cache-cpu-share {CACHE_SHARE_HIGH} keeps the busiest experts on the card and "
        f"computes 95% of the misses on the CPU instead of uploading them; {moved}"
        f"{CACHE_SHARE_RESEARCH}"
    )
    return reason if price is None else f"{reason}. {price}"


def with_expert_cache_share(situation: Situation, engine: Engine) -> Settings:
    if (found := _cache_share_layout(situation)) is None:
        return situation.settings
    _, layout = found
    settings = situation.settings
    if layout.more:
        settings = apply_flag(settings, "--n-cpu-moe", layout.host_layers + layout.more)
    settings = apply_flag(settings, "--moe-cache-mib", CACHE_SHARE_MIB)
    settings = apply_flag(settings, "--moe-cache-cpu-share", CACHE_SHARE_HIGH)
    return apply_flag(settings, "--load-mode", "none")


def _other_share(situation: Situation) -> str | None:
    """The cache share worth measuring against the one set: 0.95 for share 0 at any slot count,
    0 for 0.95 at one slot only (share 0 with small caches lost 24-80% at four and eight)."""
    extra = situation.settings.extra_args
    seen = _cache_situation(situation)
    if seen is None or "--moe-cache-mib" not in extra or "--override-tensor" in extra:
        return None
    try:
        share = float(extra.get("--moe-cache-cpu-share") or 0)
    except (TypeError, ValueError):
        return None
    if share == 0:
        return CACHE_SHARE_HIGH
    if share == float(CACHE_SHARE_HIGH) and (_slots(situation, seen.started) or 1) <= CACHE_SLOTS:
        return CACHE_SHARE_ZERO
    return None


def expert_cache_other_share_applies(situation: Situation, engine: Engine) -> str | None:
    if (other := _other_share(situation)) is None:
        return None
    now = situation.settings.extra_args.get("--moe-cache-cpu-share") or CACHE_SHARE_ZERO
    what = (
        "computes 95% of the cache misses on the CPU instead of uploading them, which wins on "
        "8 and 6-bit experts"
        if other == CACHE_SHARE_HIGH
        else "uploads every cache miss and computes it on the GPU, which wins on 3 to 5-bit experts"
    )
    return (
        f"the expert cache runs with --moe-cache-cpu-share {now}; share {other} {what}; which "
        f"one is faster depends on the file and the card, so both are measured; "
        f"{CACHE_SHARE_RESEARCH}"
    )


def with_other_share(situation: Situation, engine: Engine) -> Settings:
    if (other := _other_share(situation)) is None:
        return situation.settings
    return apply_flag(situation.settings, "--moe-cache-cpu-share", other)


def _mac_threads(seen: View) -> tuple[int, int] | None:
    """The generation and prompt thread counts worth trying on a Mac: the performance cores
    plus the efficiency cores less two, and all of them. An M4 (4P+6E) decoded fastest at 8
    of 10, and a prompt ran about 30 % faster at 10."""
    env = seen.environment
    p_cores, e_cores = env.get("p_cores"), env.get("e_cores")
    if not p_cores:
        return None
    return max(p_cores, p_cores + (e_cores or 0) - 2), p_cores + (e_cores or 0)


def _measured_m4(seen: View) -> bool:
    return (seen.environment.get("p_cores"), seen.environment.get("e_cores")) == (4, 6)


def _threads(seen: View) -> tuple[int | None, int | None, int | None]:
    return seen.started.threads, seen.started.threads_batch, seen.environment.get("usable_cpus")


def threads_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    if seen is None:
        return None
    threads, _, usable = _threads(seen)
    if _mac(seen):
        wanted = _mac_threads(seen)
        if wanted is None or not threads or seen.regime.get("compute") == "gpu":
            return None
        if threads == wanted[0]:
            return None
        basis = (
            "measured on an M4 with 4 performance and 6 efficiency cores"
            if _measured_m4(seen)
            else "measure it"
        )
        return f"llama-server runs {threads} threads; {wanted[0]} is worth trying ({basis})"
    if not threads or not usable:
        return None
    if threads > usable:
        return f"llama-server runs {threads} threads on {usable} usable CPUs"
    return None


def threads_smt_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    if (
        seen is None
        or _mac(seen)
        or seen.regime.get("compute") == "gpu"
        or not seen.environment.get("smt")
    ):
        return None
    threads, _, usable = _threads(seen)
    if threads and usable and threads < usable:
        return f"llama-server runs {threads} threads of the {usable} usable CPUs the host gives it"
    return None


def with_threads(situation: Situation, engine: Engine) -> Settings:
    seen = view(situation)
    if seen is not None and _mac(seen) and (wanted := _mac_threads(seen)) is not None:
        return apply_flag(situation.settings, "--threads", wanted[0])
    usable = None if seen is None else _threads(seen)[2]
    return situation.settings if not usable else apply_flag(situation.settings, "--threads", usable)


def prior_off_mac(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    return "measured off the Mac" if seen is not None and _mac(seen) else None


def threads_batch_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    if seen is None or seen.regime.get("compute") == "gpu":
        return None
    _, batch, usable = _threads(seen)
    if _mac(seen):
        wanted = _mac_threads(seen)
        usable = wanted[1] if wanted else None
    if not batch or not usable or batch >= usable:
        return None
    what = "cores" if _mac(seen) else "usable CPUs"
    return f"prompt processing runs {batch} threads of the {usable} {what}"


def with_threads_batch(situation: Situation, engine: Engine) -> Settings:
    seen = view(situation)
    usable = None if seen is None else _threads(seen)[2]
    if seen is not None and _mac(seen) and (wanted := _mac_threads(seen)) is not None:
        usable = wanted[1]
    if not usable:
        return situation.settings
    return apply_flag(situation.settings, "--threads-batch", usable)


# --- hold-backs -------------------------------------------------------------------------


def not_in_help(*flags: str) -> Callable[[Situation], str | None]:
    """A hold-back for an entry whose flags the binary's ``--help`` does not list."""

    def hold(situation: Situation) -> str | None:
        found = known_build()
        missing = found[1].missing(flags) if found is not None else []
        return f"not in this build's --help: {', '.join(missing)}" if missing else None

    return hold


def _room_help(situation: Situation) -> str | None:
    seen = view(situation)
    step = None if seen is None else _room_step(seen)
    return None if step is None else not_in_help(ROOM_FLAGS[step["lever"]])(situation)


def _q8_needs_flash_attn(situation: Situation) -> str | None:
    seen = view(situation)
    if seen is not None and (seen.started.flash_attn or "").startswith("off"):
        return "a q8_0 V cache needs flash attention, which is off in this launch"
    return None


def _kv_host_with_q8(situation: Situation) -> str | None:
    if situation.settings.kv_cache_dtype == Q8:
        return "the KV cache on the host is never combined with a q8_0 KV cache"
    return None


def _no_draft_mtp(situation: Situation) -> str | None:
    found = known_build()
    if found is None or found[1].lists("--spec-type", DRAFT_MTP):
        return None
    return f"this build's --help lists no {DRAFT_MTP} for --spec-type"


def _mtp_hybrid_slots_crash(situation: Situation) -> str | None:
    anatomy = analysis().anatomy
    if anatomy is None or not anatomy.linear_attention:
        return None
    build = _build_number()
    if build is not None and build >= MTP_HYBRID_SLOTS_FIXED_BUILD:
        return None
    seen = view(situation)
    settings = situation.settings
    asked = str(settings.extra_args.get("--parallel") or "")
    counts = (
        seen.started.slots if seen is not None else None,
        situation.facts.max_inflight if seen is None or seen.started.slots is None else None,
        settings.max_concurrent_requests,
        int(asked) if asked.isdigit() else None,
    )
    if max((c for c in counts if c is not None), default=1) <= 1:
        return None
    fixed = f"b{MTP_HYBRID_SLOTS_FIXED_BUILD}"
    gain = (
        "MTP drafting would raise throughput here (req/s +45% measured on Qwen3.8-27B, one L4, "
        "3 agent clients), but this llama.cpp build can crash with it at more than one slot "
        f"(#31656, fixed in {fixed})"
    )
    if _from_ollama():
        return f"{gain}: it needs an Ollama release on llama.cpp {fixed} or later"
    return f"{gain}: upgrade llama.cpp to {fixed} or later, then run tensward analyse again"


def _from_ollama() -> bool:
    from ...presets import read_choices

    project = analysis().project
    choices = read_choices(project) if project is not None else None
    return choices is not None and choices.ollama is not None


def _drafting_slots_crash(situation: Situation) -> str | None:
    """More slots for a setup that already drafts with draft-mtp, where #31656 aborts it."""
    if not mtp_slots_crash(analysis().anatomy, situation.settings.extra_args, 2, _build_number()):
        return None
    fixed = f"b{MTP_HYBRID_SLOTS_FIXED_BUILD}"
    return (
        f"your setup drafts with --spec-type draft-mtp, and llama.cpp builds before {fixed} can "
        f"crash with it on hybrid models with several slots (issue #31656); update to {fixed} "
        "or later"
    )


def first(*holds: Callable[[Situation], str | None]) -> Callable[[Situation], str | None]:
    def hold(situation: Situation) -> str | None:
        return next((why for check in holds if (why := check(situation))), None)

    return hold


def _build_number() -> int | None:
    found = known_build()
    return found[0].build if found and found[0] else None


def _no_host_experts(situation: Situation) -> str | None:
    anatomy = analysis().anatomy
    if anatomy is not None and host_experts_unsafe(anatomy, _build_number()):
        return "this model gives wrong output with experts on the host (llama.cpp #32710)"
    return None


def _experts_on_host(seen: View) -> bool:
    if seen.placed.get("host_expert_layers") or seen.kind == "expert_offload":
        return True
    anatomy = analysis().anatomy
    on_host = seen.placed.get("host_layers") or seen.placed.get("host_weight_bytes")
    return bool(on_host and anatomy is not None and any(anatomy.expert_layer_bytes))


def _mtp_host_experts(situation: Situation) -> str | None:
    """Drafting with the model's own head is held back while experts run on the host: each
    drafted token is checked against experts the CPU computes."""
    seen = view(situation)
    if seen is None or not _experts_on_host(seen):
        return None
    return (
        "MTP drafting is held back with experts on the host: it was slower there, decode -6% at "
        "draft depth 1 and -18% at depth 2 (GLM-5.3-Flash, one L4, experts on the CPU; research, "
        "not measured by Tensward); it is offered when every expert is on the card"
    )


HOLD_BACKS: dict[str, Callable[[Situation], str | None]] = {
    "llamacpp-slots": first(not_in_help("--parallel"), _drafting_slots_crash),
    "llamacpp-kv-unified": not_in_help("--kv-unified"),
    "llamacpp-shared-pool": first(
        not_in_help("--parallel", "--kv-unified", "--ctx-size"), _drafting_slots_crash
    ),
    "raise-ubatch": not_in_help("--ubatch-size", "--batch-size"),
    "llamacpp-kv-pool": not_in_help("--ctx-size"),
    "fit-context": not_in_help("--ctx-size"),
    "kv-q8": first(not_in_help("--cache-type-k", "--cache-type-v"), _q8_needs_flash_attn),
    "llamacpp-kv-q8-for-layers": not_in_help("--cache-type-k", "--n-gpu-layers", "--n-cpu-moe"),
    "llamacpp-fit-target-256": first(
        _no_host_experts, not_in_help("--n-gpu-layers", "--n-cpu-moe", "--fit-target")
    ),
    "flash-attn": not_in_help("--flash-attn"),
    "swa-full": not_in_help("--swa-full"),
    "offload-less": not_in_help("--n-gpu-layers", "--n-cpu-moe"),
    "buy-gpu-room": first(_no_host_experts, _room_help),
    "llamacpp-raise-ubatch-expert-trade": first(
        _no_host_experts, not_in_help("--ubatch-size", "--batch-size", "--n-cpu-moe")
    ),
    "offload-experts": first(_no_host_experts, not_in_help("--n-cpu-moe", "--n-gpu-layers")),
    "offload-ffn": not_in_help("--n-cpu-ffn", "--n-gpu-layers"),
    "kv-on-host": first(not_in_help("--kv-offload"), _kv_host_with_q8),
    "no-op-offload": not_in_help("--op-offload"),
    "pin-host-weights": first(_no_host_experts, not_in_help("--load-mode")),
    "load-mode-none": first(_no_host_experts, not_in_help("--load-mode")),
    "expert-cache-to-free-vram": first(
        _no_host_experts, not_in_help("--moe-cache-mib", "--n-cpu-moe", "--load-mode")
    ),
    "cpu-threads": not_in_help("--threads"),
    "cpu-threads-smt": not_in_help("--threads"),
    "cpu-threads-batch": not_in_help("--threads-batch"),
    "enable-tool-calling": not_in_help("--jinja"),
    "llamacpp-mtp-speculation": first(
        not_in_help("--spec-type", "--spec-draft-n-max"),
        _no_draft_mtp,
        _mtp_host_experts,
        _mtp_hybrid_slots_crash,
    ),
}

for _reads, _condition in (
    (state_save_off_gpu_waits_applies, STATE_SAVE_OFF_GPU_WAITS),
    (backend_sampling_gpu_waits_applies, BACKEND_SAMPLING_GPU_WAITS),
    (raise_ubatch_expert_trade_applies, RAISE_UBATCH_EXPERT_TRADE),
    (offload_ubatch_copies_applies, OFFLOAD_UBATCH_COPIES),
):
    setattr(_reads, CENSUS_CONDITION, _condition)

PREDICATES: dict[str, Callable[..., Any] | Gate] = {
    f"llamacpp:{predicate.__name__}": predicate
    for predicate in (
        slots_applies,
        with_more_slots,
        kv_unified_applies,
        with_kv_unified,
        kv_pool_applies,
        with_larger_pool,
        kv_q8_applies,
        with_kv_q8,
        flash_attn_applies,
        with_flash_attn_on,
        swa_full_applies,
        with_swa_full,
        raise_ubatch_applies,
        with_larger_ubatch,
        offload_less_applies,
        with_fewer_host_layers,
        offload_experts_applies,
        with_host_experts,
        offload_ffn_applies,
        with_host_ffn,
        kv_host_applies,
        with_kv_on_host,
        buy_room_applies,
        with_room_step,
        fit_target_applies,
        with_fit_target,
        kv_q8_layers_applies,
        with_kv_q8_layers,
        context_for_room_applies,
        with_context_for_room,
        raise_ubatch_blocked,
        raise_ubatch_expert_trade_applies,
        offload_ubatch_copies_applies,
        with_ubatch_trade,
        mtp_speculation_blocked,
        trim_for_slots_applies,
        with_trimmed_slots,
        shared_pool_applies,
        with_shared_pool,
        no_op_offload_applies,
        with_no_op_offload,
        pin_weights_applies,
        with_load_mode,
        load_mode_none_applies,
        with_load_mode_none,
        expert_cache_applies,
        expert_cache_low_bar,
        with_expert_cache,
        expert_cache_share_applies,
        with_expert_cache_share,
        expert_cache_other_share_applies,
        with_other_share,
        threads_applies,
        threads_smt_applies,
        with_threads,
        threads_batch_applies,
        with_threads_batch,
        prior_off_mac,
        ngram_speculation_applies,
        with_ngram_speculation,
        mtp_speculation_applies,
        with_mtp_speculation,
        op_offload_threshold_applies,
        with_op_offload_threshold,
        hugetlb_state_save_applies,
        with_hugetlb,
        state_save_off_tools_applies,
        state_save_off_short_applies,
        state_save_off_gpu_waits_applies,
        with_state_save_off,
        backend_sampling_gemma4_applies,
        backend_sampling_hybrid_applies,
        backend_sampling_gpu_waits_applies,
        with_backend_sampling,
    )
}
PREDICATES.update(
    {
        "llamacpp:leads:flash-attn": flash_attn_leads,
        "llamacpp:leads:swa-full": swa_full_leads,
        "llamacpp:leads:offload-less": offload_less_leads,
        "llamacpp:leads:offload-experts": offload_experts_leads,
        "llamacpp:side:raise-ubatch": raise_ubatch_side,
        "llamacpp:side:llamacpp-mtp-speculation": mtp_speculation_side,
        "llamacpp:leads:llamacpp-mtp-speculation": mtp_speculation_leads,
    }
)
