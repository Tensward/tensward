"""llama-server's flags as Tensward reads and turns them, from ``common/arg.cpp`` at build 11669
(commit 620d2225c), and what a build's own ``--help`` admits."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

PIN_BUILD = 11669
MULTI_GPU_REASON = "Tensward 0.4 measures one GPU"


@dataclass(frozen=True, slots=True)
class Flag:
    canonical: str
    aliases: tuple[str, ...]
    takes_value: bool
    env: str | None
    negations: tuple[str, ...] = ()

    @property
    def spellings(self) -> tuple[str, ...]:
        return (self.canonical, *self.aliases, *self.negations)


def _value(canonical: str, env: str | None = None, *aliases: str) -> Flag:
    return Flag(canonical, aliases, True, env)


def _switch(
    canonical: str, env: str | None = None, *aliases: str, off: tuple[str, ...] = ()
) -> Flag:
    return Flag(canonical, aliases, False, env, off)


_TABLE: tuple[Flag, ...] = (
    _value("--model", "LLAMA_ARG_MODEL", "-m"),
    _value("--model-url", "LLAMA_ARG_MODEL_URL", "-mu"),
    _value("--docker-repo", "LLAMA_ARG_DOCKER_REPO", "-dr"),
    _value("--hf-repo", "LLAMA_ARG_HF_REPO", "-hf", "-hfr"),
    _value("--hf-file", "LLAMA_ARG_HF_FILE", "-hff"),
    _value("--hf-token", "HF_TOKEN", "-hft"),
    _value("--mmproj", "LLAMA_ARG_MMPROJ", "-mm"),
    _value("--mmproj-url", "LLAMA_ARG_MMPROJ_URL", "-mmu"),
    _value("--lora"),
    _value("--lora-scaled"),
    _value("--ctx-size", "LLAMA_ARG_CTX_SIZE", "-c"),
    _value("--parallel", "LLAMA_ARG_N_PARALLEL", "-np"),
    _switch("--kv-unified", "LLAMA_ARG_KV_UNIFIED", "-kvu", off=("-no-kvu", "--no-kv-unified")),
    _value("--kv-unified-per-slot", "LLAMA_ARG_KV_UNIFIED_PER_SLOT"),
    _value("--batch-size", "LLAMA_ARG_BATCH", "-b"),
    _value("--ubatch-size", "LLAMA_ARG_UBATCH", "-ub"),
    _value("--n-gpu-layers", "LLAMA_ARG_N_GPU_LAYERS", "-ngl", "--gpu-layers"),
    _value("--n-cpu-moe", "LLAMA_ARG_N_CPU_MOE", "-ncmoe"),
    _switch("--cpu-moe", "LLAMA_ARG_CPU_MOE", "-cmoe"),
    _value("--n-cpu-ffn", "LLAMA_ARG_N_CPU_FFN", "-ncffn"),
    _value("--moe-cache-mib", "LLAMA_ARG_MOE_CACHE_MIB"),
    _value("--override-tensor", "LLAMA_ARG_OVERRIDE_TENSOR", "-ot"),
    _switch("--kv-offload", "LLAMA_ARG_KV_OFFLOAD", "-kvo", off=("-nkvo", "--no-kv-offload")),
    _switch("--op-offload", None, off=("--no-op-offload",)),
    _switch("--repack", "LLAMA_ARG_REPACK", off=("-nr", "--no-repack")),
    _switch("--no-host", "LLAMA_ARG_NO_HOST"),
    _value("--load-mode", "LLAMA_ARG_LOAD_MODE", "-lm"),
    _value("--lazy-mode", "LLAMA_ARG_LAZY_MODE", "-lzm"),
    _value("--threads", "LLAMA_ARG_THREADS", "-t"),
    _value("--threads-batch", None, "-tb"),
    _value("--threads-http", "LLAMA_ARG_THREADS_HTTP"),
    _value("--flash-attn", "LLAMA_ARG_FLASH_ATTN", "-fa"),
    _value("--cache-type-k", "LLAMA_ARG_CACHE_TYPE_K", "-ctk"),
    _value("--cache-type-v", "LLAMA_ARG_CACHE_TYPE_V", "-ctv"),
    _switch("--swa-full", "LLAMA_ARG_SWA_FULL"),
    _switch("--cache-prompt", "LLAMA_ARG_CACHE_PROMPT", off=("--no-cache-prompt",)),
    _value("--cache-reuse", "LLAMA_ARG_CACHE_REUSE"),
    _value("--cache-ram", "LLAMA_ARG_CACHE_RAM", "-cram"),
    _switch("--cache-idle-slots", "LLAMA_ARG_CACHE_IDLE_SLOTS", off=("--no-cache-idle-slots",)),
    _value("--ctx-checkpoints", "LLAMA_ARG_CTX_CHECKPOINTS", "-ctxcp", "--swa-checkpoints"),
    _value("--checkpoint-min-step", "LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT", "-cms"),
    _switch("--jinja", "LLAMA_ARG_JINJA", off=("--no-jinja",)),
    _value("--chat-template", "LLAMA_ARG_CHAT_TEMPLATE"),
    _value("--chat-template-file", "LLAMA_ARG_CHAT_TEMPLATE_FILE"),
    _value("--chat-template-kwargs", "LLAMA_ARG_CHAT_TEMPLATE_KWARGS"),
    _value("--reasoning-format", "LLAMA_ARG_THINK"),
    _value("--reasoning", "LLAMA_ARG_REASONING", "-rea"),
    _value("--reasoning-effort", "LLAMA_ARG_REASONING_EFFORT"),
    _value("--reasoning-budget", "LLAMA_ARG_THINK_BUDGET"),
    _value("--reasoning-budget-message", "LLAMA_ARG_THINK_BUDGET_MESSAGE"),
    _switch(
        "--reasoning-preserve", "LLAMA_ARG_REASONING_PRESERVE", off=("--no-reasoning-preserve",)
    ),
    _switch(
        "--cont-batching", "LLAMA_ARG_CONT_BATCHING", "-cb", off=("-nocb", "--no-cont-batching")
    ),
    _value("--sleep-idle-seconds"),
    _switch("--context-shift", "LLAMA_ARG_CONTEXT_SHIFT", off=("--no-context-shift",)),
    _value("--override-kv"),
    _value("--fit", "LLAMA_ARG_FIT", "-fit"),
    _value("--fit-ctx", "LLAMA_ARG_FIT_CTX", "-fitc"),
    _value("--fit-target", "LLAMA_ARG_FIT_TARGET", "-fitt"),
    _value("--host", "LLAMA_ARG_HOST"),
    _value("--port", "LLAMA_ARG_PORT"),
    _value("--alias", "LLAMA_ARG_ALIAS", "-a"),
    _value("--api-key", "LLAMA_API_KEY"),
    _value("--api-key-file", "LLAMA_ARG_API_KEY_FILE"),
    _value("--ssl-key-file", "LLAMA_ARG_SSL_KEY_FILE"),
    _value("--ssl-cert-file", "LLAMA_ARG_SSL_CERT_FILE"),
    _switch("--metrics", "LLAMA_ARG_ENDPOINT_METRICS"),
    _switch("--slots", "LLAMA_ARG_ENDPOINT_SLOTS", off=("--no-slots",)),
    _value("--numa", "LLAMA_ARG_NUMA"),
    _value("--device", "LLAMA_ARG_DEVICE", "-dev"),
    _value("--split-mode", "LLAMA_ARG_SPLIT_MODE", "-sm"),
    _value("--tensor-split", "LLAMA_ARG_TENSOR_SPLIT", "-ts"),
    _value("--main-gpu", "LLAMA_ARG_MAIN_GPU", "-mg"),
    _value("--rpc", "LLAMA_ARG_RPC"),
    _value("--spec-type", "LLAMA_ARG_SPEC_TYPE"),
    _value("--spec-draft-n-max", "LLAMA_ARG_SPEC_DRAFT_N_MAX"),
    _value("--spec-draft-n-min", "LLAMA_ARG_SPEC_DRAFT_N_MIN"),
    _value("--spec-draft-p-min", "LLAMA_ARG_SPEC_DRAFT_P_MIN", "--draft-p-min"),
    _value("--spec-synth-len"),
    _value("--spec-synth-rates"),
    _value("--spec-draft-model", "LLAMA_ARG_SPEC_DRAFT_MODEL", "-md", "--model-draft"),
    _value("--defrag-thold", "LLAMA_ARG_DEFRAG_THOLD", "-dt"),
    _value("--verbosity", "LLAMA_ARG_LOG_VERBOSITY", "-lv", "--log-verbosity"),
    _switch("--log-disable"),
    _value("--log-file", "LLAMA_ARG_LOG_FILE"),
    _switch("--log-prefix", "LLAMA_ARG_LOG_PREFIX", off=("--no-log-prefix",)),
    _switch("--log-timestamps", "LLAMA_ARG_LOG_TIMESTAMPS", off=("--no-log-timestamps",)),
    _switch("--log-jsonl", "LLAMA_ARG_LOG_JSONL", off=("--no-log-jsonl",)),
    _value("--log-colors", "LLAMA_ARG_LOG_COLORS"),
    _switch("--verbose", None, "-v", "--log-verbose"),
)

FLAGS: Mapping[str, Flag] = {flag.canonical: flag for flag in _TABLE}
"""Canonical spelling -> flag, for every flag Tensward reads, turns, pins or refuses."""

_BY_SPELLING: Mapping[str, Flag] = {s: flag for flag in _TABLE for s in flag.spellings}
_ENV_FLAGS: Mapping[str, Flag] = {flag.env: flag for flag in _TABLE if flag.env}

REMOVED: Mapping[str, str] = {
    "--no-mmap": "removed; use --load-mode",
    "--mmap": "removed; use --load-mode",
    "--mlock": "removed; use --load-mode",
    "--direct-io": "removed; use --load-mode",
    "--draft-max": "removed; use --spec-draft-n-max",
    "--draft-min": "removed; use --spec-draft-n-min",
    "--spec-ngram-size-n": "removed; use the --spec-ngram-*-size-n options",
    "--spec-ngram-size-m": "removed; use the --spec-ngram-*-size-m options",
}

NO_OP: frozenset[str] = frozenset({"-dt", "--defrag-thold"})

SYNTHETIC_REASON = "diagnostic synthetic acceptance, a research tool that never runs in a timed run"

NEVER: Mapping[str, str] = {
    "--context-shift": "it silently drops the oldest tokens of a long conversation",
    "--override-kv": "it changes the model's metadata, and with it the answers",
    "--spec-synth-len": SYNTHETIC_REASON,
    "--spec-synth-rates": SYNTHETIC_REASON,
    "--split-mode": MULTI_GPU_REASON,
    "--tensor-split": MULTI_GPU_REASON,
    "--main-gpu": MULTI_GPU_REASON,
    "--rpc": MULTI_GPU_REASON,
    "--device": MULTI_GPU_REASON,
}
"""Flags Tensward never turns for a project; the multi-GPU ones it also refuses in a command."""

REFUSED_SOURCES = frozenset({"--hf-repo", "--model-url", "--docker-repo"})
"""Flags that name a model to download instead of a local file."""

SECRETS = frozenset({"--api-key", "--api-key-file", "--ssl-key-file", "--hf-token"})

PINNED = frozenset({
    "--host", "--port", "--alias", "--metrics", "--slots", "--verbosity",
})  # fmt: skip
"""Set by Tensward on every launch."""

FIT_FLAGS = frozenset({"--fit", "--fit-ctx", "--fit-target"})
"""Kept from the user's command for the baseline launch and replaced by ``--fit off`` on every
launch pinned to a baseline."""

LOG_FLAGS = frozenset({
    "--log-disable", "--log-file", "--log-prefix", "--log-timestamps", "--log-jsonl",
    "--log-colors", "--verbose",
})  # fmt: skip
"""Flags that change the server log's shape, which the log reader needs as it is."""

LOG_ENV = ("LLAMA_LOG_", "LLAMA_ARG_LOG_", "LLAMA_ARG_NO_LOG_")

ADMITTED: frozenset[str] = frozenset({
    "--batch-size", "--ubatch-size", "--n-gpu-layers", "--n-cpu-moe", "--cpu-moe", "--n-cpu-ffn",
    "--override-tensor", "--kv-offload", "--op-offload", "--load-mode", "--threads",
    "--threads-batch", "--flash-attn", "--cache-type-k", "--cache-type-v", "--swa-full",
    "--cache-prompt", "--cache-reuse", "--cache-ram", "--kv-unified", "--ctx-size", "--parallel",
    "--cont-batching", "--numa", "--ctx-checkpoints", "--moe-cache-mib", "--repack",
})  # fmt: skip
"""Flags a knowledge record may change on a build."""

ENV_ADMITTED: frozenset[str] = frozenset({
    "GGML_CUDA_DISABLE_GRAPHS", "GGML_CUDA_GRAPH_OPT", "GGML_CUDA_PDL",
})  # fmt: skip


def flag_for(spelling: str) -> Flag | None:
    """The flag a command-line spelling names; ``--foo_bar`` reads as ``--foo-bar``, as
    llama-server does."""
    if spelling.startswith("--"):
        spelling = "--" + spelling[2:].replace("_", "-")
    return _BY_SPELLING.get(spelling)


def dashed(name: str) -> str:
    """``name`` as a command-line spelling: an engine-argument key may be bare (``ubatch-size``,
    ``ngl``, ``no-op-offload``)."""
    if name.startswith("-"):
        return name
    return f"-{name}" if flag_for(f"--{name}") is None and flag_for(f"-{name}") else f"--{name}"


def resolve(name: str) -> Flag | None:
    return flag_for(dashed(name))


def flag_for_env(name: str) -> tuple[Flag, bool] | None:
    """The flag an environment variable sets, and whether it sets it (``LLAMA_ARG_NO_X`` unsets
    the switch ``LLAMA_ARG_X``)."""
    if (flag := _ENV_FLAGS.get(name)) is not None:
        return flag, True
    if name.startswith("LLAMA_ARG_NO_") and (
        flag := _ENV_FLAGS.get("LLAMA_ARG_" + name.removeprefix("LLAMA_ARG_NO_"))
    ):
        return flag, False
    return None


def canonical_flags(flags: Mapping[str, Any]) -> dict[str, Any]:
    """``flags`` with each spelling folded into its canonical long name, in sorted order; a
    switch's off spelling becomes the canonical name set to False."""
    found: dict[str, Any] = {}
    for spelling, value in flags.items():
        known = resolve(spelling)
        if known is None:
            found[dashed(spelling)] = value
            continue
        off = dashed(spelling)
        off = "--" + off[2:].replace("_", "-") if off.startswith("--") else off
        found[known.canonical] = False if off in known.negations else value
    return dict(sorted(found.items()))


def merge(extra_args: Mapping[str, Any], flags: Mapping[str, Any]) -> dict[str, Any]:
    """``extra_args`` with ``flags`` written in."""
    return {**canonical_flags(extra_args), **canonical_flags(flags)}


def canonical_args(args: Iterable[str]) -> frozenset[str]:
    """Engine arguments as ``name=value`` words (``no-name`` for a switch turned off), two
    spellings of one change as one."""
    flags: dict[str, Any] = {}
    for text in args:
        name, given, value = text.partition("=")
        flags[dashed(name)] = value if given else True
    words = set()
    for flag, value in canonical_flags(flags).items():
        bare = flag.lstrip("-")
        words.add(bare if value is True else f"no-{bare}" if value is False else f"{bare}={value}")
    return frozenset(words)


def build_number(version: str | None) -> int | None:
    """The build a version names: ``11669``, ``b11669`` or ``build 11669``."""
    found = re.search(r"\d{3,}", version or "")
    return int(found[0]) if found else None


def allowed(version: str | None, flag: str) -> bool:
    """Whether a record may change ``flag`` on build ``version``; never for an unknown build."""
    build = build_number(version)
    known = flag_for(flag)
    return (
        build is not None
        and build >= PIN_BUILD
        and known is not None
        and (known.canonical in ADMITTED)
    )


def env_allowed(version: str | None, name: str) -> bool:
    build = build_number(version)
    return build is not None and build >= PIN_BUILD and name in ENV_ADMITTED


_HEADER_END = re.compile(r"\s{2,}(?=[^-\s])")
_REMOVED_TEXT = "has been removed"


@dataclass(frozen=True, slots=True)
class HelpManifest:
    flags: frozenset[str]  # every spelling the binary's --help lists

    @classmethod
    def from_help(cls, text: str) -> HelpManifest:
        """The spellings a build's ``--help`` lists; an option whose help says it has been
        removed is a stub and is left out."""
        entries: list[tuple[str, list[str]]] = []
        for line in text.splitlines():
            if line.startswith("-") and not line.startswith("--- "):
                header, *rest = _HEADER_END.split(line, maxsplit=1)
                entries.append((header, rest))
            elif entries and line.strip():
                entries[-1][1].append(line)
        found: set[str] = set()
        for header, rest in entries:
            if _REMOVED_TEXT in " ".join(rest):
                continue
            for piece in header.split(","):
                words = piece.split()
                if words and words[0].startswith("-"):
                    found.add(words[0])
        return cls(frozenset(found))

    def has(self, flag: str) -> bool:
        """Whether the build lists ``flag`` under any of its spellings."""
        known = flag_for(flag)
        names = known.spellings if known is not None else (flag,)
        return any(name in self.flags for name in names)

    def missing(self, flags: Iterable[str]) -> list[str]:
        return sorted(flag for flag in flags if not self.has(flag))


DIRECTED = frozenset(
    {"--ubatch-size", "--n-gpu-layers", "--n-cpu-ffn", "--threads", "--threads-batch"}
)
UNSET_VALUES = {"--ubatch-size": 512}  # -ub default
OFFLOAD_STEPS = frozenset({"--n-cpu-moe", "--n-cpu-ffn"})


def key_flags(
    before: Mapping[str, Any], after: Mapping[str, Any], sets: Mapping[str, Any]
) -> dict[str, Any]:
    """The changed flags (canonical) as a lever key reads them: a numeric flag by whether it
    went up, down or was newly set. ``--n-cpu-moe`` is carried by its lever, and
    ``--n-gpu-layers`` is the companion of an offload step, so neither names a step by itself."""
    old, new = canonical_flags(before), canonical_flags(after)
    changed = {flag: value for flag, value in new.items() if old.get(flag) != value}
    companion = bool(OFFLOAD_STEPS & changed.keys())
    found: dict[str, Any] = {}
    for flag, value in changed.items():
        if (
            flag == "--n-cpu-moe"
            and "host_expert_layers" in sets
            or (flag == "--n-gpu-layers" and companion)
        ):
            continue
        if flag in DIRECTED and str(value).isdigit():
            found[flag] = _direction(old.get(flag, UNSET_VALUES.get(flag)), value)
        else:
            found[flag] = value
    return found


def _direction(was: Any, value: Any) -> str:
    if was is None or not str(was).isdigit():
        return "set"
    return "up" if int(str(value)) > int(str(was)) else "down"


class _LlamaCppFlagRules:
    engine = "llamacpp"
    version = str(PIN_BUILD)
    env_names = ENV_ADMITTED
    pool_shrinking: Mapping[str, str] = {
        "ctx-size": "max_context_len",
    }
    pool_defaults: Mapping[str, Any] = {}
    pool_bytes_flag: str | None = None
    canonical = staticmethod(canonical_flags)
    allowed = staticmethod(allowed)
    env_allowed = staticmethod(env_allowed)
    merge = staticmethod(merge)
    canonical_args = staticmethod(canonical_args)


LLAMACPP_FLAG_RULES = _LlamaCppFlagRules()
