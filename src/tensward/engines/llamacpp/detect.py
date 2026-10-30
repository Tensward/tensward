"""What tells Tensward which engine a user runs: the serve command, a running endpoint, the
checkpoint's file type and the environment. Each source gives votes of a strength; the engine
choice is made by :func:`tensward.environment.resolve_engine`."""

from __future__ import annotations

import ipaddress
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from .command import is_container_run, launch_tokens, runs_llama_server

Strength = Literal["decisive", "strong", "medium", "weak"]

LOOKALIKE_REFUSAL = (
    "built on llama.cpp but not llama-server; its flags differ; Tensward 0.4 supports llama-server"
)
ENDPOINT_TIMEOUT_S = 2.0
ENDPOINT_MAX_BYTES = 1_000_000
NEEDS_COMMAND = (
    "Needs the command that started it (--current or --current-file); Tensward relaunches it "
    "to measure."
)

_LLAMA_ONLY_FLAGS = frozenset({
    "-ngl", "--n-gpu-layers", "-np", "--parallel", "--n-cpu-moe", "-ot", "--override-tensor",
    "-ctk", "--cache-type-k", "--jinja", "--fit",
})  # fmt: skip
_VLLM_ONLY_FLAGS = frozenset({"--gpu-memory-utilization", "--max-num-seqs", "--max-model-len"})
_LOOKALIKE_PROGRAMS = {
    "ollama": "Ollama",
    "lms": "LM Studio",
    "llamafile": "llamafile",
    "koboldcpp": "koboldcpp",
}
_LOOKALIKE_IMAGE = re.compile(r"(^|/)ollama/ollama\b")
_VLLM_IMAGE = re.compile(r"(^|/)vllm/vllm-openai")
_BUILD_INFO = re.compile(r"^b(\d+)")
LM_STUDIO_OWNER = "organization_owner"


@dataclass(frozen=True, slots=True)
class Evidence:
    """One piece of evidence about the engine. ``engine`` is None for a look-alike (``lookalike``
    names it) or when the source says nothing about the engine."""

    source: Literal["engine_flag", "command", "endpoint", "model", "env"]
    engine: str | None
    lookalike: str | None
    strength: Strength
    text: str


@dataclass(frozen=True, slots=True)
class EndpointFacts:
    evidence: list[Evidence]
    model_path: str | None
    build: int | None


def _command(engine: str | None, strength: Strength, text: str) -> Evidence:
    return Evidence("command", engine, None, strength, text)


def command_evidence(text: str) -> list[Evidence]:
    try:
        _, tokens = launch_tokens(text)
    except ValueError:
        return []
    if not tokens:
        return []
    program = os.path.basename(tokens[0])
    if is_container_run(tokens):
        image = next((t for t in tokens[1:] if _LOOKALIKE_IMAGE.search(t)), None)
        if image is not None:
            return [Evidence("command", None, "Ollama", "strong", f"docker image {image}")]
        if any(_VLLM_IMAGE.search(t) for t in tokens[1:]):
            return [_command("vllm", "strong", "serve command")]
    elif program in _LOOKALIKE_PROGRAMS or program.startswith("koboldcpp"):
        name = _LOOKALIKE_PROGRAMS.get(program, "koboldcpp")
        return [Evidence("command", None, name, "strong", f"serve command {program}")]
    elif "llama_cpp.server" in tokens:
        return [Evidence("command", None, "llama-cpp-python", "strong", "serve command")]
    if runs_llama_server(text):
        return [_command("llamacpp", "strong", "serve command")]
    if program == "vllm" or any(
        word == "-m" and later.startswith("vllm.entrypoints")
        for word, later in zip(tokens, tokens[1:], strict=False)
    ):
        return [_command("vllm", "strong", "serve command")]
    flags = {token.partition("=")[0] for token in tokens if token.startswith("-")}
    found = []
    if flags & _LLAMA_ONLY_FLAGS:
        found.append(_command("llamacpp", "medium", "serve command flags"))
    if flags & _VLLM_ONLY_FLAGS:
        found.append(_command("vllm", "medium", "serve command flags"))
    return found


def model_evidence(path: Path) -> list[Evidence]:
    """A GGUF file or a directory holding one: weak, as vLLM can load GGUF too."""
    try:
        gguf = path.suffix.lower() == ".gguf" or (
            path.is_dir() and any(child.suffix.lower() == ".gguf" for child in path.iterdir())
        )
    except OSError:
        return []
    if not gguf:
        return []
    return [Evidence("model", "llamacpp", None, "weak", "a GGUF file")]


def env_evidence(environ: Mapping[str, str], current_file: str | None) -> list[Evidence]:
    """``LLAMA_ARG_*`` or ``VLLM_*`` names in the text of the file that holds the serve command (a
    shell script, a unit file, a compose file) are a medium vote. The same names in the process
    environment are weak: shells export them for other servers too."""
    found = []
    for engine, prefix in (("llamacpp", "LLAMA_ARG_"), ("vllm", "VLLM_")):
        if current_file is not None and re.search(rf"\b{prefix}[A-Z0-9_]+", current_file):
            found.append(Evidence("env", engine, None, "medium", f"{prefix}* variables"))
        elif any(name.startswith(prefix) for name in environ):
            found.append(Evidence("env", engine, None, "weak", f"{prefix}* variables"))
    return found


def _get(client: httpx.Client, url: str, headers: Mapping[str, str]) -> httpx.Response | None:
    try:
        with client.stream(
            "GET", url, headers=dict(headers), timeout=ENDPOINT_TIMEOUT_S
        ) as response:
            body = b""
            for chunk in response.iter_bytes():
                body += chunk
                if len(body) > ENDPOINT_MAX_BYTES:
                    return httpx.Response(response.status_code)
            return httpx.Response(response.status_code, content=body)
    except httpx.HTTPError:
        return None


def _json(response: httpx.Response | None) -> Any:
    if response is None or response.status_code != 200:
        return None
    try:
        return json.loads(response.content)
    except ValueError:
        return None


def sends_key(url: str) -> bool:
    """Whether ``LLAMA_API_KEY`` may be sent to ``url``: only over https or to this machine."""
    parts = urlsplit(url)
    if parts.scheme == "https" or parts.hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(parts.hostname or "").is_loopback
    except ValueError:
        return False


def probe_endpoint(url: str, client: httpx.Client) -> EndpointFacts | None:
    """What the server at ``url`` says about itself, from ``GET /health``, ``/v1/models`` and
    ``/props`` (and ``/api/version`` when those do not read as llama-server). Nothing is sent
    that loads it. None when nothing answers. ``LLAMA_API_KEY`` is sent as the bearer key when
    ``sends_key(url)``."""
    base = url.rstrip("/")
    key = os.environ.get("LLAMA_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key and sends_key(url) else {}
    health = _get(client, f"{base}/health", {})
    if health is None:
        return None
    models_response = _get(client, f"{base}/v1/models", headers)
    props_response = _get(client, f"{base}/props", headers)
    models, props = _json(models_response), _json(props_response)
    owner = None
    listed = models.get("data") if isinstance(models, dict) else None
    first = listed[0] if isinstance(listed, list) and listed else None
    if isinstance(first, dict):
        owner = first.get("owned_by")
    build_info = props.get("build_info") if isinstance(props, dict) else None
    build = _BUILD_INFO.match(build_info) if isinstance(build_info, str) else None
    model_path = props.get("model_path") if isinstance(props, dict) else None
    evidence: list[Evidence] = []

    def add(
        engine: str | None, strength: Strength, text: str, lookalike: str | None = None
    ) -> None:
        evidence.append(Evidence("endpoint", engine, lookalike, strength, text))

    if owner == "llamacpp" and build:
        add("llamacpp", "strong", "endpoint /props + /v1/models owned_by=llamacpp")
    elif owner == "llamacpp":
        add("llamacpp", "medium", "endpoint /v1/models owned_by=llamacpp")
    elif build:
        add("llamacpp", "medium", "endpoint /props build_info")
    elif owner == "vllm":
        add("vllm", "medium", "endpoint /v1/models owned_by=vllm")
    elif owner == LM_STUDIO_OWNER:
        add(None, "medium", "endpoint /v1/models", lookalike="LM Studio")
    elif {r.status_code for r in (models_response, props_response) if r is not None} == {401}:
        says_llama = _json(health) == {"status": "ok"}
        add("llamacpp" if says_llama else None, "medium", "endpoint requires an API key")
    if not any(item.engine or item.lookalike for item in evidence):
        version = _json(_get(client, f"{base}/api/version", {}))
        if isinstance(version, dict) and "version" in version:
            add(None, "strong", "endpoint /api/version", lookalike="Ollama")
    return EndpointFacts(
        evidence, model_path if isinstance(model_path, str) else None,
        int(build[1]) if build else None,
    )  # fmt: skip


def build_mismatch(running: int | None, binary: int | None) -> str | None:
    if running is None or binary is None or running == binary:
        return None
    return (
        f"warning: the running server is build {running}, the command's binary is build "
        f"{binary}; Tensward measures {binary}"
    )
