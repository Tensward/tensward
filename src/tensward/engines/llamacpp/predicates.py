"""llama.cpp's playbook predicates: the conditions and changes its entries name as
"llamacpp:<name>". They read the run's start-up log, its placement block (regime, fit and
next steps) and its probes; none of them recomputes what ``placement`` already decided."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...context import current as analysis
from ...measurement import nearest_rank
from ...playbook import Gate, Saving
from ...playbook_common import MIN_SHARED_PREFIX_TOKENS, MIN_SHARING_PROMPTS, shared_lengths
from ...pressure import demand
from ...settings import Settings
from .command import DEFAULT_BATCH, Q8, apply_flag, known_build
from .placement import (
    LONG_PROMPT_TOKENS,
    MAJOR_FAULTS_TIGHT,
    MIN_POOL_TOKENS,
    POOL_ROUND,
    Q8_KV_SAVING,
    Fit2,
    Placement,
    device_kv,
)
from .startup import Startup, read_startup

if TYPE_CHECKING:
    from ...playbook import Situation
    from ..protocol import Engine

SLOT_STEP = 2  # the slots ladder doubles
UBATCH_LADDER = (512, 2048, 4096)
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
        margin."""
        fit = self.fit
        return fit["device_free"] - fit["device_need"] if fit else None


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
    return seen.fit.get("need_tokens") or (
        (situation.measurement.max_prompt_tokens or 0) + situation.facts.output_tokens
    )


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
    slots, started = situation.settings.max_concurrent_requests, seen.started
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
    seen = view(situation)
    m, slots = situation.measurement, situation.settings.max_concurrent_requests
    if seen is None or not slots:
        return None
    full = (m.peak_waiting or 0) > 0 and (m.peak_running or 0) >= slots
    clients = demand(m) or 0
    if not (full or clients > slots) or (target := _next_slots(situation, seen)) is None:
        return None
    waiting = f"{m.peak_waiting:.0f} requests were deferred with all {slots} slots busy"
    return (
        waiting if full else f"{clients} requests were in flight on {slots} slots"
    ) + f"; {target} slots fit the KV pool at {seen.started.slot_ctx} tokens each"


def with_more_slots(situation: Situation, engine: Engine) -> Settings:
    seen = view(situation)
    settings = situation.settings
    if seen is None or (target := _next_slots(situation, seen)) is None:
        return settings
    settings = dataclasses.replace(settings, max_concurrent_requests=target)
    if not seen.started.kv_unified:
        settings = engine.with_lever(settings, "max_context_len", seen.started.slot_ctx)
    return settings


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


def with_swa_full(situation: Situation, engine: Engine) -> Settings:
    return apply_flag(situation.settings, "--swa-full", True)


# --- ubatch -----------------------------------------------------------------------------


def _next_ubatch(situation: Situation, seen: View) -> int | None:
    started = seen.started
    current = started.ubatch or UBATCH_LADDER[0]
    larger = next((rung for rung in UBATCH_LADDER if rung > current), None)
    if larger is None:
        return None
    batch = situation.settings.prefill_batch_tokens or DEFAULT_BATCH
    if (started.ctx_total or 0) < max(batch, larger):
        return None
    buffer = started.bytes_of("compute", "device")
    if buffer:
        room = seen.room
        if room is None or buffer * larger // current - buffer > room:
            return None
    return larger


def raise_ubatch_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    if seen is None or (larger := _next_ubatch(situation, seen)) is None:
        return None
    pcie = seen.probes.get("pcie") or {}
    if pcie.get("reason", "").startswith("the larger ubatch did not shorten"):
        return None
    if situation.fired("prefill"):
        why = situation.finding("prefill").evidence
    elif seen.kind == "expert_offload":
        why = "the experts run from host memory, and a larger -ub reuses each streamed weight"
    elif seen.offloaded and (situation.measurement.max_prompt_tokens or 0) >= LONG_PROMPT_TOKENS:
        why = "prompts of 2,048 tokens or more stream the host weights once per ubatch"
    else:
        return None
    return f"{why}; -ub {seen.started.ubatch} to {larger}"


def with_larger_ubatch(situation: Situation, engine: Engine) -> Settings:
    seen = view(situation)
    settings = situation.settings
    if seen is None or (larger := _next_ubatch(situation, seen)) is None:
        return settings
    settings = apply_flag(settings, "--ubatch-size", larger)
    if (settings.prefill_batch_tokens or DEFAULT_BATCH) < larger:
        settings = apply_flag(settings, "--batch-size", larger)
    return settings


# --- placement --------------------------------------------------------------------------


def _saves(step: Mapping[str, Any]) -> str:
    return f"{step['why']}; about {step['saved_ms_per_token']:.2f} ms per token"


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
    step = None if seen is None else seen.step("kv_on_host")
    return (
        None
        if step is None
        else f"{step['why']}; about {step['saved_ms_per_token']:.2f} ms per token"
    )


def with_kv_on_host(situation: Situation, engine: Engine) -> Settings:
    return apply_flag(situation.settings, "--kv-offload", False)


def _room_step(seen: View) -> Mapping[str, Any] | None:
    return seen.step(*ROOM_FLAGS) if seen.offloaded else None


def buy_room_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    step = None if seen is None else _room_step(seen)
    return (
        None
        if step is None
        else f"{step['why']}; about {step['saved_ms_per_token']:.2f} ms per token"
    )


def with_room_step(situation: Situation, engine: Engine) -> Settings:
    seen = view(situation)
    step = None if seen is None else _room_step(seen)
    settings = situation.settings
    if seen is None or step is None:
        return settings
    lever, value = step["lever"], step["value"]
    if lever == "kv_cache_precision":
        return engine.with_lever(settings, lever, value)
    if lever == "ubatch":
        return apply_flag(settings, "--ubatch-size", value)
    if lever == "max_concurrent_requests":
        settings = dataclasses.replace(settings, max_concurrent_requests=value)
        need = seen.fit.get("need_tokens") or 0
        cut = max(-(-need * value // POOL_ROUND) * POOL_ROUND, MIN_POOL_TOKENS)
        return dataclasses.replace(settings, max_context_len=cut)
    return dataclasses.replace(settings, max_context_len=value)


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


def _threads(seen: View) -> tuple[int | None, int | None, int | None]:
    return seen.started.threads, seen.started.threads_batch, seen.environment.get("usable_cpus")


def threads_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    if seen is None:
        return None
    threads, _, usable = _threads(seen)
    if not threads or not usable:
        return None
    if threads > usable:
        return f"llama-server runs {threads} threads on {usable} usable CPUs"
    return None


def threads_smt_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    if seen is None or seen.regime.get("compute") == "gpu" or not seen.environment.get("smt"):
        return None
    threads, _, usable = _threads(seen)
    if threads and usable and threads < usable:
        return f"llama-server runs {threads} threads of the {usable} usable CPUs the host gives it"
    return None


def with_threads(situation: Situation, engine: Engine) -> Settings:
    seen = view(situation)
    usable = None if seen is None else _threads(seen)[2]
    return situation.settings if not usable else apply_flag(situation.settings, "--threads", usable)


def threads_batch_applies(situation: Situation, engine: Engine) -> str | None:
    seen = view(situation)
    if seen is None or seen.regime.get("compute") == "gpu":
        return None
    _, batch, usable = _threads(seen)
    if not batch or not usable or batch >= usable:
        return None
    return f"prompt processing runs {batch} threads of the {usable} usable CPUs"


def with_threads_batch(situation: Situation, engine: Engine) -> Settings:
    seen = view(situation)
    usable = None if seen is None else _threads(seen)[2]
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


def first(*holds: Callable[[Situation], str | None]) -> Callable[[Situation], str | None]:
    def hold(situation: Situation) -> str | None:
        return next((why for check in holds if (why := check(situation))), None)

    return hold


HOLD_BACKS: dict[str, Callable[[Situation], str | None]] = {
    "llamacpp-slots": not_in_help("--parallel"),
    "llamacpp-kv-unified": not_in_help("--kv-unified"),
    "raise-ubatch": not_in_help("--ubatch-size", "--batch-size"),
    "llamacpp-kv-pool": not_in_help("--ctx-size"),
    "fit-context": not_in_help("--ctx-size"),
    "kv-q8": first(not_in_help("--cache-type-k", "--cache-type-v"), _q8_needs_flash_attn),
    "flash-attn": not_in_help("--flash-attn"),
    "swa-full": not_in_help("--swa-full"),
    "offload-less": not_in_help("--n-gpu-layers", "--n-cpu-moe"),
    "buy-gpu-room": _room_help,
    "offload-experts": not_in_help("--n-cpu-moe", "--n-gpu-layers"),
    "offload-ffn": not_in_help("--n-cpu-ffn", "--n-gpu-layers"),
    "kv-on-host": first(not_in_help("--kv-offload"), _kv_host_with_q8),
    "no-op-offload": not_in_help("--op-offload"),
    "pin-host-weights": not_in_help("--load-mode"),
    "cpu-threads": not_in_help("--threads"),
    "cpu-threads-smt": not_in_help("--threads"),
    "cpu-threads-batch": not_in_help("--threads-batch"),
    "enable-tool-calling": not_in_help("--jinja"),
}

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
        no_op_offload_applies,
        with_no_op_offload,
        pin_weights_applies,
        with_load_mode,
        threads_applies,
        threads_smt_applies,
        with_threads,
        threads_batch_applies,
        with_threads_batch,
    )
}
PREDICATES.update(
    {
        "llamacpp:leads:flash-attn": flash_attn_leads,
        "llamacpp:leads:offload-less": offload_less_leads,
        "llamacpp:leads:offload-experts": offload_experts_leads,
    }
)
