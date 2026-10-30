"""What a llama-server command that pins every layer on the device is told before launch, when
the model cannot fit the device."""

from __future__ import annotations

from collections.abc import Callable

from ...anatomy import ModelAnatomy
from ...capabilities import Resolved
from ...platforms import MIB, hardware_spec
from ...platforms.cpu import HostFacts
from ...playbook import Entry
from ...settings import Settings
from .command import DEFAULT_SLOTS, DEFAULT_UBATCH, Q8, apply_flag, split_kv
from .placement import (
    DEVICE_MARGIN,
    GIB,
    DecodeModel,
    ExpertSplit,
    Placement,
    active_share,
    expert_split,
    logged_kv_bytes,
    predict_decode_ms,
)

CUDA_CONTEXT_BYTES = 256 * MIB  # device memory a CUDA context takes before any weight loads
DEFAULT_FIT_CTX = 4096  # --fit-ctx: the smallest context --fit cuts to
UNIFIED_MEMORY = "GGML_CUDA_ENABLE_UNIFIED_MEMORY"
_OFF = ("off", "false", "0", "no")


def moe_layers(settings: Settings) -> int:
    """The leading layers ``--n-cpu-moe`` keeps the experts of on the host."""
    count = str(settings.extra_args.get("--n-cpu-moe") or 0)
    return int(count) if count.isdigit() else 0


def _host_placement(settings: Settings) -> bool:
    """Whether the command moves a tensor to the host other than by ``--n-cpu-moe N``, which is
    checked against the device."""
    extra = settings.extra_args
    ffn = str(extra.get("--n-cpu-ffn") or 0)
    moe = str(extra.get("--n-cpu-moe") or 0)
    return (
        not ffn.isdigit()
        or int(ffn) > 0
        or not moe.isdigit()
        or bool(extra.get("--cpu-moe"))
        or bool(extra.get("--override-tensor"))
    )


def fit_runs(settings: Settings) -> bool:
    return str(settings.extra_args.get("--fit", "on")).lower() not in _OFF


def pins_every_layer(settings: Settings, layers: int) -> bool:
    """Whether the command puts all ``layers`` blocks and the output layer on the device and
    moves no tensor to the host: ``--fit off`` without ``-ngl`` (``auto``, or -1, then means
    all), or ``-ngl`` of ``all``, below -1, or above the layer count, which stops ``--fit``
    (common/fit.cpp:463)."""
    if _host_placement(settings):
        return False
    given = str(settings.extra_args.get("--n-gpu-layers", "auto"))
    if given in ("auto", "-1"):
        return not fit_runs(settings)
    if given == "all":
        return True
    return given.lstrip("-").isdigit() and (int(given) < -1 or int(given) > layers)


def _gib(size: float) -> str:
    return f"{size / GIB:.1f} GiB"


def _decode_step(
    anatomy: ModelAnatomy, split: ExpertSplit, bw_dev_gbs: float, bw_host_gbs: float, slots: int
) -> float:
    active = active_share(anatomy)
    resident = sum(anatomy.expert_layer_bytes) - split.moved_bytes
    device = split.device_weights - split.moved_bytes
    placed = Placement(
        device_weight_bytes=device,
        host_weight_bytes=split.moved_bytes,
        host_expert_bytes=split.moved_bytes,
        host_layers=0,
        host_expert_layers=split.layers or 0,
        kv_on_host=False,
        device_bytes_per_token=device - round(resident * (1 - active)),
        host_bytes_per_token=round(split.moved_bytes * active),
    )
    return predict_decode_ms(DecodeModel(bw_dev_gbs, bw_host_gbs, 0.0, slots, False), placed)


def _pool_tokens(
    settings: Settings, context_limit: int | None, slots: int
) -> tuple[int | None, bool]:
    """(the KV pool the launch asks for, whether ``--fit`` cut it): ``-c``, else the model's
    context; ``--fit`` first cuts an unset context to ``--fit-ctx`` for each stream of a split
    cache, unless the model's context is no larger, then stops at ``-ngl``."""
    if settings.max_context_len:
        return settings.max_context_len, False
    if not fit_runs(settings) or not context_limit:
        return context_limit, False
    cut = int(str(settings.extra_args.get("--fit-ctx") or DEFAULT_FIT_CTX))
    if context_limit <= cut:
        return context_limit, False
    return cut * (slots if split_kv(settings) else 1), True


def fit_advice(
    anatomy: ModelAnatomy,
    settings: Settings,
    device_name: str,
    device_total: int,
    host: HostFacts,
    host_gbs: float | None,
    context_limit: int | None,
    command: Callable[[Settings], str] | None,
    entry: Entry | None,
    baseline: Resolved | None = None,
) -> tuple[str, bool] | None:
    """(text, whether the launch certainly fails) for a command that pins every layer on a
    device that cannot hold them with the KV cache, or that keeps the experts of too few layers
    on the host (``--n-cpu-moe N``); None when it fits or is not pinned. ``entry`` is the
    ``offload-experts`` playbook entry. ``baseline`` is what the project's baseline run
    resolved: its logged KV bytes and the device memory it found free replace the formula's
    and the device's total."""
    slots = settings.max_concurrent_requests or DEFAULT_SLOTS
    pool, cut = _pool_tokens(settings, context_limit, slots)
    if (
        not pool
        or not anatomy.layers
        or settings.extra_env.get(UNIFIED_MEMORY) == "1"
        or not pins_every_layer(settings, anatomy.layers)
    ):
        return None
    extra = settings.extra_args
    default = Q8 if settings.kv_cache_dtype == Q8 else "f16"
    type_k = str(extra.get("--cache-type-k", default))
    type_v = str(extra.get("--cache-type-v", default))
    free = baseline.facts.get("device_free_bytes") if baseline is not None else None
    device_total = min(device_total, free) if free else device_total
    split = expert_split(
        anatomy, device_total, host, pool_tokens=pool, type_k=type_k, type_v=type_v,
        ubatch=int(str(extra.get("--ubatch-size") or DEFAULT_UBATCH)),
        kv_on_device=extra.get("--kv-offload") is not False,
        kv_logged=logged_kv_bytes(anatomy, baseline, pool, type_k, type_v),
    )  # fmt: skip
    if split.device_need <= device_total:
        return None
    if given := moe_layers(settings):
        need = split.device_need - sum(anatomy.expert_layer_bytes[:given])
        if need - DEVICE_MARGIN <= device_total:
            return None  # a tight fit still starts; only a launch that cannot is refused
        return (
            f"`-ngl all --n-cpu-moe {given}` would not fit the device: the weights, the KV cache "
            f"for {pool} tokens and the compute buffer need {_gib(need - DEVICE_MARGIN)} of its "
            f"{_gib(device_total)}"
            + (
                f"; the smallest --n-cpu-moe that fits with a 1 GiB margin is {split.layers}."
                if split.layers is not None
                else ", whatever the experts of how many layers stay on the host."
            ),
            True,
        )
    fails = split.device_weights + CUDA_CONTEXT_BYTES > device_total
    kv = (
        f"the KV cache {_gib(split.kv_bytes)} for {pool} tokens"
        if split.kv_bytes
        else "the KV cache is in host memory"
    )
    other = split.device_need - split.device_weights - split.kv_bytes
    lead = (
        f"The model {'does' if fails else 'is estimated to'} not fit the GPU with every layer on "
        f"it: the weights on the device take {_gib(split.device_weights)}, {kv}, and about "
        f"{_gib(other)} go to the compute buffer and margin, against the GPU's "
        f"{_gib(device_total)}."
    )
    if not anatomy.expert_layer_bytes:
        text = (
            f"{lead} Register the command again without -ngl and --fit off "
            "(`tensward init --current '<your command without -ngl and --fit off>'`) so that "
            "llama.cpp's --fit keeps some layers in host memory; tensward analyse then measures "
            "the split and offers offload-less and offload-ffn."
        )
        return text, fails
    if split.layers is None:
        what = (
            f"the host has {host.mem_available_gib:.1f} GiB of memory available, too little for "
            "the experts that would have to stay there"
            if split.host_short
            else "the rest does not fit even with the experts of every layer in host memory"
        )
        return f"{lead} No --n-cpu-moe value helps: {what}.", fails
    n = split.layers
    text = (
        f"{lead} Keep the experts of the first {n} layers in host memory with "
        f"`-ngl all --n-cpu-moe {n}`: the rest fits with that KV cache for "
        f"{slots} {'slot' if slots == 1 else 'slots'}."
    )
    if cut:
        text += f" The command sets the context to {pool} tokens, the size --fit would cut it to."
    if entry is not None:
        text += (
            " Experts in host memory move fewer bytes per token than a dense -ngl split of the "
            "same device memory, because attention and the shared weights stay on the device; "
            f"measured in the {entry.name} playbook entry: {entry.prior}. Cost: {entry.costs}."
        )
    text += (
        f" Each token then reads about {_gib(split.moved_bytes * active_share(anatomy))} of "
        "experts from host memory"
    )
    spec = hardware_spec(device_name)
    if host_gbs and spec:
        step = _decode_step(anatomy, split, spec.bandwidth_gbs, host_gbs, slots)
        text += f"; predicted decode step about {step:.1f} ms a token (possible)."
    else:
        text += (
            "; the host memory bandwidth is not measured yet, and the run of the offered command "
            "measures it."
        )
    text += (
        " Answers may change; if the current command does not start, the run of the offered "
        "command is the first with answers to compare against."
    )
    if command is None:
        return text + " tensward analyse will offer the command to try.", fails
    proposed = apply_flag(apply_flag(settings, "--n-gpu-layers", "all"), "--n-cpu-moe", n)
    if cut:
        proposed = apply_flag(proposed, "--ctx-size", pool)
    return f"{text} Try: `{command(proposed)}`", fails
