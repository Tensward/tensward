"""The compile caches of a vLLM launch: where they are and how much each held before and after
the server came up, so a run records whether it started warm."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ...context import AnalysisContext
from ...progress import PhaseStarted, ProgressEvent, ServerReady
from ...runtime import ENGINE_CACHES, Runtime, engine_cache_root

NAMES = ("vllm", "triton", "inductor", "flashinfer")
COMPILE_CACHE = "torch_compile_cache"


def cache_dirs(env: Mapping[str, str]) -> dict[str, Path]:
    """The directories vLLM 0.30 resolves from ``env``: its cache root, Triton's, Inductor's
    (inside the vLLM root unless ``TORCHINDUCTOR_CACHE_DIR`` moves it) and FlashInfer's."""
    home = Path(env.get("HOME", "~")).expanduser()
    cache = Path(env.get("XDG_CACHE_HOME") or home / ".cache")
    vllm = Path(env.get("VLLM_CACHE_ROOT") or cache / "vllm")
    return {
        "vllm": vllm,
        "triton": Path(env.get("TRITON_CACHE_DIR") or home / ".triton" / "cache"),
        "inductor": Path(env.get("TORCHINDUCTOR_CACHE_DIR") or vllm / COMPILE_CACHE),
        "flashinfer": Path(env.get("FLASHINFER_WORKSPACE_BASE") or home) / ".cache" / "flashinfer",
    }


def mounted_dirs(root: Path) -> dict[str, Path]:
    """The same four directories for a container whose caches ``root`` keeps on the host."""
    return {
        **{name: root / name for name in ENGINE_CACHES},
        "inductor": root / "vllm" / COMPILE_CACHE,
    }


def sizes(dirs: Mapping[str, Path]) -> dict[str, int]:
    return {
        name: sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else 0
        for name, path in dirs.items()
    }


def _image_id(image: str) -> str | None:
    try:
        done = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None if done.returncode == 0 else None


def _runtime_dirs(runtime: Runtime, env: Mapping[str, str]) -> dict[str, Path] | None:
    if runtime.kind == "local":
        return cache_dirs(env)
    image_id = _image_id(runtime.target)
    return None if image_id is None else mounted_dirs(engine_cache_root(image_id))


def observer(
    context: AnalysisContext, runtime: Runtime, env: Mapping[str, str]
) -> Callable[[ProgressEvent], None]:
    """Records ``caches`` in ``context.window_extra`` from the launch of the first server to
    its readiness."""
    before: dict[str, int] | None = None
    dirs: dict[str, Path] | None = None

    def observe(event: ProgressEvent) -> None:
        nonlocal before, dirs
        if isinstance(event, PhaseStarted) and event.phase == "launch" and before is None:
            dirs = _runtime_dirs(runtime, env)
            before = None if dirs is None else sizes(dirs)
        elif isinstance(event, ServerReady) and dirs is not None and before is not None:
            context.window_extra["caches"] = recorded(before, sizes(dirs))
            dirs = None

    return observe


def recorded(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, Any]:
    return {
        name: {
            "before": before[name],
            "after": after[name],
            "warm": after[name] == before[name] and before[name] > 0,
        }
        for name in NAMES
    }
