"""Will the model fit this machine's GPU? An estimate made at ``init`` without starting the
engine: weights from the checkpoint's anatomy, engine overhead from measurement (see the
constants), KV cache from the anatomy's per-layer KV function. Advice only, never a refusal.

Judged against the GPU's total memory times the engine's memory fraction, so a production
server running on the same GPU does not turn the verdict into "does not fit"; the memory it
uses is reported separately, since ``analyse`` needs the GPU free.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .anatomy import Components, ModelAnatomy
from .ceilings import Unavailable, gpu_spec
from .engines.protocol import Settings
from .platforms import MIB, Device

# Until validation measures the estimate's error, a shortfall within this share of usable
# memory is "tight", not "does not fit".
MARGIN = 0.05
OTHER_PROCESSES_SHARE = 0.05  # memory in use above this share of the GPU is reported
CHARS_PER_TOKEN = 4  # the token count of a prompt is estimated from its length
GIB = 2**30

Verdict = Literal["fits", "tight", "likely does not fit", "not checked"]


@dataclass(frozen=True, slots=True)
class Fit:
    verdict: Verdict
    reason: str
    gpu: str | None = None
    total_bytes: int = 0
    in_use_bytes: int = 0
    usable_bytes: int = 0
    weights_bytes: int = 0
    overhead_bytes: int = 0
    kv_available_bytes: int = 0
    kv_needed_bytes: int | None = None
    max_concurrency: int | None = None
    max_context_tokens: int | None = None
    capacity_tokens: int | None = None

    def summary(self) -> dict[str, object]:
        """What ``init`` and ``inspect`` print: byte figures in GiB, to 0.1."""
        if self.verdict == "not checked":
            return {"verdict": self.verdict, "reason": self.reason, "gpu": self.gpu}

        def gib(size: int | None) -> float | None:
            return None if size is None else round(size / GIB, 1)

        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "gpu": self.gpu,
            "total_gib": gib(self.total_bytes),
            "in_use_gib": gib(self.in_use_bytes),
            "usable_gib": gib(self.usable_bytes),
            "weights_gib": gib(self.weights_bytes),
            "overhead_gib": gib(self.overhead_bytes),
            "kv_available_gib": gib(self.kv_available_bytes),
            "kv_needed_gib": gib(self.kv_needed_bytes),
            "max_concurrency": self.max_concurrency,
            "max_context_tokens": self.max_context_tokens,
            "capacity_tokens": self.capacity_tokens,
        }


def _largest_context(anatomy: ModelAnatomy, dtype: str, in_flight: int, room: int, cap: int) -> int:
    """The longest context up to ``cap`` whose per-request KV reservation fits in ``room``."""
    low, high = 0, cap
    while low < high:
        middle = (low + high + 1) // 2
        if (anatomy.kv_bytes(middle, dtype, in_flight) or 0) <= room:
            low = middle
        else:
            high = middle - 1
    return low


def _loads_encoders(anatomy: ModelAnatomy, settings: Settings) -> bool:
    """Whether the engine loads the vision and audio towers. vLLM v0.30 skips a tower once every
    modality it serves has a limit of 0, and ``--language-model-only`` sets them all to 0."""
    if settings.media_inputs is False:
        return False
    limits = settings.media_limits or {}
    return any(limits.get(kind) != 0 for kind in anatomy.modalities if kind != "text")


def estimate_fit(
    anatomy: ModelAnatomy,
    settings: Settings,
    *,
    concurrency: int | None,
    avg_tokens: int,
    counts_images: bool = False,
    gpus: tuple[Device, ...],
    selected: tuple[str, ...],
    default_fraction: float,
    context_limit: int | None = None,
    parallel: int = 1,
    in_flight_tokens: int = 0,
    overhead_bytes: int,
) -> Fit:
    parts = anatomy.components
    if parts is None or not anatomy.attention:
        return Fit(
            "not checked",
            "the checkpoint's layout is not modelled: " + "; ".join(anatomy.unavailable),
        )
    if parallel > 1:
        return Fit("not checked", "tensor or pipeline parallelism is not modelled")
    if not gpus:
        return Fit("not checked", "no supported accelerator is visible")
    try:
        index, name, _ = gpu_spec(tuple((g.name, g.total_bytes // MIB) for g in gpus), selected)
    except Unavailable as error:
        return Fit("not checked", str(error))
    gpu = gpus[index]
    usable = int(gpu.total_bytes * (settings.kv_memory_fraction or default_fraction))
    skipped = () if _loads_encoders(anatomy, settings) else ("vision", "audio")
    weights = sum(getattr(parts, name) for name in Components.__slots__ if name not in skipped)
    available = usable - weights - overhead_bytes
    base = Fit(
        "fits", "", name, gpu.total_bytes, gpu.used_bytes, usable, weights,
        overhead_bytes, available,
    )  # fmt: skip
    note = ""
    if gpu.used_bytes > OTHER_PROCESSES_SHARE * gpu.total_bytes:
        note = (
            f"; {gpu.used_bytes / GIB:.1f} GiB of the GPU is in use by other processes (for "
            "example your production server); `analyse` needs the GPU free"
        )
    if available < -MARGIN * usable:
        short = (weights + overhead_bytes - usable) / GIB
        return replace(
            base,
            verdict="likely does not fit",
            reason=(
                f"the weights and the engine's overhead need {short:.1f} GiB more than the engine "
                "may use; try a lower-precision checkpoint, a smaller model, or a GPU with more "
                "memory" + note
            ),
        )
    dtype = settings.kv_cache_dtype
    context = settings.max_context_len or context_limit
    per_sequence = anatomy.kv_bytes(min(avg_tokens, context or avg_tokens), dtype) or 1
    room = max(available, 0)
    max_context = capacity = None
    if context:
        reserved = anatomy.kv_bytes(context, dtype, in_flight_tokens) or 1
        max_context = _largest_context(anatomy, dtype, in_flight_tokens, room, context)
        capacity = int(room / reserved * context)
    sequences = concurrency or 1
    images = " including images at their maximum token count" if counts_images else ""
    known = replace(
        base,
        kv_needed_bytes=per_sequence * sequences,
        max_concurrency=room // per_sequence,
        max_context_tokens=max_context,
        capacity_tokens=capacity,
    )
    if available <= 0:
        reason = f"weights and engine overhead leave {available / GIB:.1f} GiB for the KV cache"
    elif max_context is not None and context and max_context < context:
        reason = f"the engine refuses to start with max_context_len {context}; {max_context} fits"
    elif per_sequence * sequences > available:
        reason = (
            f"the KV cache holds {known.max_concurrency} of the {sequences} declared concurrent "
            f"requests of about {avg_tokens} tokens{images} (estimated at {CHARS_PER_TOKEN} "
            "characters per token)"
        )
    else:
        checked = f"{sequences} concurrent requests" if concurrency else "one sequence"
        suffix = "" if concurrency else " (the workload declares no concurrency)"
        return replace(
            known,
            reason=f"the KV cache holds {checked} of about {avg_tokens} tokens{images}"
            f"{suffix}{note}",
        )
    return replace(known, verdict="tight", reason=reason + note)
