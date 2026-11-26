"""A kept step's settings in Ollama's own words, for a project imported from Ollama.

Ollama 0.40.16 starts llama-server itself (llm/llama_server.go): ``-c`` is ``num_ctx`` times
OLLAMA_NUM_PARALLEL, ``-np`` is OLLAMA_NUM_PARALLEL, ``-ngl`` is ``num_gpu``, ``-b`` and ``-ub``
are both ``num_batch``, ``-t`` is ``num_thread``, ``--cache-type-k`` and ``--cache-type-v`` are
both OLLAMA_KV_CACHE_TYPE, ``--flash-attn`` follows OLLAMA_FLASH_ATTENTION and ``use_mmap false``
starts it with ``--load-mode none`` (appendLoadModeArgs)."""

from __future__ import annotations

from collections.abc import Iterable

SPACING_VARIABLE = "LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT"
SERVER = "for ollama serve (sudo systemctl edit ollama)"
MODELFILE = "in the model's Modelfile"
NO_SETTING = "this change needs llama.cpp; Ollama has no setting for it"
# llama.cpp placement flags Ollama cannot pass: num_gpu without them puts every layer's experts
# on the card (O35, Qwen3.5-35B-A3B Q8_0 on an L4: num_gpu 41 alone ran out of memory at start)
UNPLACED = ("n-cpu-moe", "cpu-moe", "override-tensor")


def _split(engine_args: Iterable[str]) -> dict[str, str]:
    found = {}
    for argument in engine_args:
        name, _, value = argument.lstrip("-").partition("=")
        found[name] = value
    return found


def _batch(batch: str | None, ubatch: str | None) -> tuple[str | None, str | None]:
    """The Modelfile line for a kept batch size, or what has no Ollama spelling."""
    if batch is None and ubatch is None:
        return None, None
    if batch == "2048" and ubatch is None:
        return f"  PARAMETER num_batch 0   {MODELFILE}", None
    if batch is not None and batch == ubatch:
        return f"  PARAMETER num_batch {batch}   {MODELFILE}", None
    shown = " ".join(
        f"--{name} {size}" for name, size in (("batch-size", batch), ("ubatch-size", ubatch))
        if size is not None
    )  # fmt: skip
    return None, f"{shown} (num_batch sets -b and -ub to one value)"


def lines(version: str, engine_args: Iterable[str], parallel: int | None = None) -> list[str]:
    """How to apply the kept ``engine_args`` to Ollama ``version``, each in Ollama's words or
    named as having none; ``parallel``: the slots of the setup they were kept on, for a context
    size kept without ``--parallel``. Empty when nothing was kept."""
    args = _split(engine_args)
    if args.pop(SPACING_VARIABLE, None) == "0":
        args["checkpoint-min-step"] = "0"
    env: list[str] = []
    params: list[str] = []
    notes: list[str] = []
    missing: list[str] = []
    slots = args.pop("parallel", None)
    if slots is not None:
        env.append(f'  Environment="OLLAMA_NUM_PARALLEL={slots}"   {SERVER}')
    if (context := args.pop("ctx-size", None)) is not None:
        count = int(slots) if slots and slots.isdigit() else parallel or 1
        per_slot = int(context) // count if context.isdigit() and count > 0 else 0
        if per_slot > 0:
            params.append(f"  PARAMETER num_ctx {per_slot}   {MODELFILE}")
            notes.append(
                f"  (num_ctx is per request: Ollama starts its runner with num_ctx x {count} "
                f"slots, --ctx-size {per_slot * count})"
            )
        else:
            missing.append(f"--ctx-size {context}")
    unplaced = [name for name in UNPLACED if name in args]
    if (layers := args.pop("n-gpu-layers", None)) is not None and unplaced:
        missing.append(
            f"--n-gpu-layers {layers} with --{unplaced[0]} {args[unplaced[0]]}".strip()
            + " (one placement; num_gpu alone would put every expert on the card, which may not"
            " fit: keep your current num_gpu in Ollama)"
        )
        args.pop(unplaced[0])
    elif layers is not None:
        if layers.isdigit():
            params.append(f"  PARAMETER num_gpu {layers}   {MODELFILE}")
        else:
            notes.append(f"  --n-gpu-layers {layers} is Ollama's default: leave num_gpu unset")
    line, unmatched = _batch(args.pop("batch-size", None), args.pop("ubatch-size", None))
    params += [line] if line else []
    missing += [unmatched] if unmatched else []
    if (threads := args.pop("threads", None)) is not None:
        params.append(f"  PARAMETER num_thread {threads}   {MODELFILE}")
    if (flash := args.pop("flash-attn", None)) is not None:
        if flash in ("on", "off"):
            env.append(f'  Environment="OLLAMA_FLASH_ATTENTION={int(flash == "on")}"   {SERVER}')
        else:
            notes.append(f"  --flash-attn {flash} is Ollama's default: leave it unset")
    k, v = args.pop("cache-type-k", None), args.pop("cache-type-v", None)
    if k is not None and k == v:
        env.append(f'  Environment="OLLAMA_KV_CACHE_TYPE={k}"   {SERVER}')
    elif k is not None or v is not None:
        shown = " ".join(f"--cache-type-{n} {t}" for n, t in (("k", k), ("v", v)) if t)
        missing.append(f"{shown} (OLLAMA_KV_CACHE_TYPE sets K and V to one type)")
    if (load := args.pop("load-mode", None)) == "none":
        params.append(f"  PARAMETER use_mmap false   {MODELFILE}")
    elif load is not None:
        missing.append(f"--load-mode {load}")
    if (step := args.pop("checkpoint-min-step", None)) == "0":
        env.append(
            f'  Environment="{SPACING_VARIABLE}=0"   {SERVER}; '
            "the runner's log then says min spacing = 0"
        )
    elif step is not None:
        missing.append(f"--checkpoint-min-step {step}")
    for name, value in args.items():
        if not name.islower():
            found_env = f"{name}={value}"
            missing.append(f"{found_env} (a llama-server variable, not measured through Ollama)")
        else:
            missing.append(f"--{name} {value}".strip() if value else f"--{name}")
    if not (env or params or notes or missing):
        return []
    found = [f"In Ollama {version}:", *env, *params, *notes]
    found += [f"  {item}: {NO_SETTING}" for item in missing]
    if params:
        found.append("Then: ollama create <model> -f Modelfile   (it starts FROM <model>)")
    if env:
        found.append("Then: sudo systemctl restart ollama")
    return found
