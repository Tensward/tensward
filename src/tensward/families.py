"""Model families that need something from the engine beyond a generic checkpoint: the first
engine release that serves them. None of them is validated (``anatomy.VALIDATED``) until a
measured run says so."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .anatomy import ModelAnatomy


@dataclass(frozen=True, slots=True)
class Family:
    name: str
    llamacpp_build: int  # the first llama.cpp release that loads the architecture
    vllm: str  # the first vLLM version that serves it
    vllm_released: bool
    ollama: str  # Ollama's support, in words


FAMILIES = {
    # llama.cpp PR #32694 (inkling2_loop, first in b12600); vLLM recipe: 0.35.0, unreleased on
    # 20 Nov 2026; Ollama v0.40.17 serves it on Apple silicon only.
    "inkling2_loop": Family(
        name="Inkling 2 Mini",
        llamacpp_build=12600,
        vllm="0.35.0",
        vllm_released=False,
        ollama="Ollama serves Inkling 2 Mini only on Apple silicon (0.40.17; PC support is not "
        "released yet), and Tensward does not support Ollama on a Mac yet",
    ),
}


def engine_note(
    arch: str | None, engine: str, build: int | None = None, *, ollama: str | None = None
) -> str | None:
    """What the engine needs for this model family, in plain words; None for a generic one."""
    family = FAMILIES.get(arch or "")
    if family is None:
        return None
    unvalidated = f"{family.name} is not validated with Tensward yet"
    if ollama is not None:
        return f"note: {family.ollama}. {unvalidated}."
    if engine == "vllm":
        from .engines.vllm.command import VLLM_VERSION

        released = "" if family.vllm_released else ", which is not released yet"
        return (
            f"note: this model needs vLLM {family.vllm} or later{released}; Tensward is "
            f"validated on vLLM {VLLM_VERSION}. {unvalidated}."
        )
    needed = f"this model needs llama.cpp b{family.llamacpp_build} or later"
    untested = " (Tensward has not been tested on these builds yet)"
    if build is not None and build < family.llamacpp_build:
        return (
            f"warning: {needed}; this llama-server is build {build}, which cannot load it. "
            f"Install b{family.llamacpp_build} or later{untested} and run tensward init again."
        )
    return f"note: {needed}{untested}. {unvalidated}."


# llama.cpp #32710: Inkling 2 Mini outputs wrong text, at a normal speed, when a low-bit expert
# file has its expert weights on the host and op offload computes them (measured on b12603 with
# IQ4_NL; Q8_0 was correct, as was --no-op-offload). A bisect found micro-batches up to 40 tokens
# coherent and garbage from 41; the guard stays at 32, where op offload starts in the source.
# The issue is open; set the constant to the first fixed build once one exists.
HOST_EXPERT_ARCH = "inkling2_loop"
HOST_EXPERT_FIXED_BUILD: int | None = None
OP_OFFLOAD_MIN_BATCH = 32
SAFE_EXPERT_TYPES = frozenset({"Q8_0", "F16", "BF16"})
HOST_EXPERT_WARNING = (
    "Inkling 2 Mini gives garbage output with experts on the CPU (llama.cpp #32710) when the "
    "expert weights are not Q8_0, F16 or BF16: measured on b12603; no fixed build yet. The fault "
    "was measured on IQ4_NL and Q8_0 was measured coherent; the other low-bit types were not "
    "measured and are treated like IQ4_NL. Keep all weights on the GPU, or add --no-op-offload "
    "or a micro-batch of 32 or less (-ub 32): both were measured to give coherent text, which "
    "differs from the all-on-card text (not identical), and cost prompt-processing speed (how "
    "much is not measured). The fault was not run with --n-cpu-moe"
)
REASONING_BUDGET_WARNING = (
    "--reasoning-budget 0 makes Inkling 2 Mini loop on empty thinking blocks until the output "
    'cap (3 of 4 prompts, b12603, IQ4_NL); use "chat_template_kwargs": {"reasoning_effort": '
    "...} in the request or the server flag --reasoning-effort instead"
)
UBATCH_ADVICE = (
    "Inkling 2 Mini's compute buffer grows with the context: on a 24 GB card -ub 256 loaded "
    "64k tokens (-ub 512, the default, loaded 32k; one slot), at the same prompt speed within "
    "the spread"
)


# llama.cpp before this build aborts in post_decode seq_rm ("failed to remove sequence") when
# the model's own drafts run on a hybrid model with several busy slots (issue #31656; fixed by
# #31669, commit d35dc4d9, first in b12676); L27 aborted after 27 min on b12246 at -np 4.
MTP_HYBRID_SLOTS_FIXED_BUILD = 12676
MTP_SLOTS_WARNING = (
    f"your command drafts with --spec-type draft-mtp on a hybrid model with several slots: "
    f"llama.cpp builds before b{MTP_HYBRID_SLOTS_FIXED_BUILD} can crash with that (issue "
    f"#31656); update to b{MTP_HYBRID_SLOTS_FIXED_BUILD} or later, or run it with -np 1"
)


def drafts_mtp(extra_args: Mapping[str, str | bool | None]) -> bool:
    return "draft-mtp" in str(extra_args.get("--spec-type") or "").split(",")


def mtp_slots_crash(
    anatomy: ModelAnatomy | None, extra_args: Mapping[str, str | bool | None], slots: int | None,
    build: int | None,
) -> bool:  # fmt: skip
    """Whether a command that drafts with draft-mtp on several slots meets #31656 here."""
    if anatomy is None or not anatomy.linear_attention or not drafts_mtp(extra_args):
        return False
    if build is not None and build >= MTP_HYBRID_SLOTS_FIXED_BUILD:
        return False
    return (slots or 1) > 1


def is_inkling(anatomy: ModelAnatomy | None) -> bool:
    return anatomy is not None and anatomy.model_type == HOST_EXPERT_ARCH


def reasoning_budget_set(extra_args: Mapping[str, str | bool | None]) -> bool:
    return str(extra_args.get("--reasoning-budget")) == "0"


def ubatch_advice(
    anatomy: ModelAnatomy | None, extra_args: Mapping[str, str | bool | None]
) -> str | None:
    """The -ub advice for Inkling 2 Mini, unless the command already sets 256 or less."""
    if not is_inkling(anatomy):
        return None
    size = str(extra_args.get("--ubatch-size") or "")
    return None if size.isdigit() and int(size) <= 256 else UBATCH_ADVICE


def host_experts_unsafe(anatomy: ModelAnatomy | None, build: int | None = None) -> bool:
    """Whether expert weights on the host give wrong output here: Inkling 2 Mini with an expert
    type other than Q8_0, F16 and BF16, on any build until a fixed one is named."""
    if anatomy is None or anatomy.model_type != HOST_EXPERT_ARCH:
        return False
    fixed = HOST_EXPERT_FIXED_BUILD
    if fixed is not None and build is not None and build >= fixed:
        return False
    return any(t not in SAFE_EXPERT_TYPES for t in anatomy.precisions.get("routed_experts", ()))


def _tensors_to_cpu(value: str) -> bool:
    return any(
        "exps" in pattern and target.strip().upper() == "CPU"
        for pattern, _, target in (part.rpartition("=") for part in value.split(","))
    )


def host_expert_flags(
    extra_args: Mapping[str, str | bool | None], layers: int | None = None
) -> bool:
    """Whether a llama-server command both keeps expert tensors on the host (``--cpu-moe``, a
    ``--n-cpu-moe`` above 0, an ``--override-tensor`` that sends them to ``CPU``, or ``-ngl``
    below the layer count) and sends them through op offload (no ``--no-op-offload``, a
    micro-batch above 32)."""
    moe = str(extra_args.get("--n-cpu-moe") or 0)
    ngl = str(extra_args.get("--n-gpu-layers"))
    on_host = (
        bool(extra_args.get("--cpu-moe"))
        or not moe.isdigit()
        or int(moe) > 0
        or _tensors_to_cpu(str(extra_args.get("--override-tensor") or ""))
        or (ngl.isdigit() and int(ngl) < (layers if layers else 1))
    )
    if not on_host or extra_args.get("--op-offload") is False:
        return False
    batches = [str(extra_args.get(flag) or "") for flag in ("--ubatch-size", "--batch-size")]
    sizes = [int(b) for b in batches if b.isdigit()]
    return not sizes or min(sizes) > OP_OFFLOAD_MIN_BATCH
