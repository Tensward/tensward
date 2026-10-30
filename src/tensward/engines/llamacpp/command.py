"""llama-server command lines: reading a customer's launch, writing Tensward's, and asking a
binary which build it is and which flags it lists."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...probes import PROBE_TIMEOUT_S, executable
from ...secrets import ASSIGNMENT, is_secret, without_secrets
from ...settings import ParsedSetup, Settings
from .flags import (
    FIT_FLAGS,
    FLAGS,
    LOG_ENV,
    LOG_FLAGS,
    MULTI_GPU_REASON,
    NO_OP,
    PINNED,
    REFUSED_SOURCES,
    REMOVED,
    SECRETS,
    HelpManifest,
    flag_for,
    flag_for_env,
)

LLAMA_SERVER = "llama-server"
LOG_VERBOSITY = "4"
ENGINE_ENV_PREFIX = "GGML_"
DEFAULT_BATCH = 2048
DEFAULT_UBATCH = 512
DEFAULT_SLOTS = 4  # slots when -np is unset (server.cpp:168-172)
OPTIONAL_VALUE = frozenset({"--flash-attn", "--fit", "--log-colors", "--reasoning"})
Q8 = "q8_0"
_KV_TYPES = ("--cache-type-k", "--cache-type-v")
_INTEGER_FIELDS = {
    "--ctx-size": "max_context_len",
    "--parallel": "max_concurrent_requests",
    "--batch-size": "prefill_batch_tokens",
}
_NUMBER = re.compile(r"-?[0-9.]+")
_CONTAINER_CLIS = ("docker", "podman")
_LLAMA_IMAGE = re.compile(r"ghcr\.io/ggml-org/llama\.cpp:(server|full)")
_VERSION = re.compile(r"version: (\S+) \(build (\d+), commit (\w+)\)")
_BACKENDS = re.compile(r"\b(CUDA|Metal|Vulkan|HIP|SYCL)\b")
_SECRET_NOTE = (
    "{flag} carries a secret, which Tensward never stores or puts on a command line; "
    "Tensward measures with a key of its own"
)
LAUNCHER_HELP = "expected a `llama-server ...` command"


@dataclass(frozen=True, slots=True)
class BuildIdentity:
    build: int
    commit: str
    version: str
    backends: tuple[str, ...]


def build_identity(version_text: str) -> BuildIdentity | None:
    """The build a ``llama-server --version`` output names; None when it names none."""
    found = _VERSION.search(version_text)
    if found is None:
        return None
    backends = tuple(dict.fromkeys(_BACKENDS.findall(version_text)))
    return BuildIdentity(int(found[2]), found[3], found[1], backends)


def launch_tokens(text: str) -> tuple[dict[str, str], list[str]]:
    tokens = shlex.split(re.sub(r"\\\r?\n", " ", text))
    env: dict[str, str] = {}
    while tokens and ASSIGNMENT.match(tokens[0]):
        name, _, value = tokens.pop(0).partition("=")
        env[name] = value
    return env, tokens[1:] if tokens[:1] == ["sudo"] else tokens


def is_container_run(tokens: Sequence[str]) -> bool:
    return bool(tokens) and os.path.basename(tokens[0]) in _CONTAINER_CLIS


def runs_llama_server(command: str) -> bool:
    """Whether ``command`` launches llama-server, directly or as a container of llama.cpp's
    server image."""
    try:
        _, tokens = launch_tokens(command)
    except ValueError:
        return False
    if is_container_run(tokens):
        return _LLAMA_IMAGE.search(command) is not None
    return (
        bool(tokens)
        and re.fullmatch(rf"{LLAMA_SERVER}(\.exe)?", os.path.basename(tokens[0])) is not None
    )


def _resolve_executable(first: str) -> str | None:
    found = first if os.sep in first else shutil.which(first)
    return os.path.abspath(found) if found else None


def _read_flags(arguments: list[str]) -> list[tuple[str, str | None]]:
    """(spelling, value) pairs; a value is taken from ``=`` or from the next word when that is
    not a flag. Which flags take one is decided by the caller from the flag table."""
    pairs: list[tuple[str, str | None]] = []
    position = 0
    while position < len(arguments):
        token = arguments[position]
        position += 1
        if not token.startswith("-") or _NUMBER.fullmatch(token):
            raise ValueError(f"unexpected argument {token!r}: cannot tell which flag it is for")
        name, equals, value = token.partition("=")
        given: str | None = value if equals else None
        known = flag_for(name)
        takes = known.takes_value if known is not None else None
        nxt = arguments[position] if position < len(arguments) else None
        is_value = nxt is not None and (not nxt.startswith("-") or _NUMBER.fullmatch(nxt))
        if given is None and is_value and takes is not False:
            optional = known is not None and known.canonical in OPTIONAL_VALUE
            if takes or optional or known is None:
                given = nxt
                position += 1
        if given is None and takes and known is not None and known.canonical not in OPTIONAL_VALUE:
            raise ValueError(f"{name} needs a value")
        pairs.append((name, given))
    return pairs


def read_flag(name: str, value: str | None) -> tuple[str, Any]:
    """A flag as ``(canonical name, value)``: a switch is True or False (its off spelling or
    ``=false``), a flag with a value is its text; a flag the table does not know keeps its
    spelling."""
    known = flag_for(name)
    if known is None:
        flag = "--" + name[2:].replace("_", "-") if name.startswith("--") else name
        return flag, True if value is None else value
    if known.takes_value:
        return known.canonical, True if value is None else value
    off = name in known.negations or name.replace("_", "-") in known.negations
    enabled = value is None or value.lower() not in ("false", "0", "off")
    return known.canonical, enabled != off


def _integer(flag: str, value: Any) -> int:
    try:
        return int(str(value))
    except ValueError:
        raise ValueError(f"{flag} needs an integer value, not {value!r}") from None


def _unfold_kv_types(settings: Settings) -> Settings:
    if settings.kv_cache_dtype == "auto":
        return settings
    extra = {**settings.extra_args, **dict.fromkeys(_KV_TYPES, settings.kv_cache_dtype)}
    return dataclasses.replace(settings, kv_cache_dtype="auto", extra_args=extra)


def _fold_kv_types(settings: Settings) -> Settings:
    extra = dict(settings.extra_args)
    keys, values = _KV_TYPES, {extra.get(k) for k in _KV_TYPES}
    if all(k in extra for k in keys) and len(values) == 1:
        (kind,) = values
        if kind in (Q8, "f16"):
            for key in keys:
                del extra[key]
            return dataclasses.replace(
                settings, kv_cache_dtype=Q8 if kind == Q8 else "auto", extra_args=extra
            )
    return settings


def apply_flag(settings: Settings, flag: str, value: Any) -> Settings:
    """``settings`` with the flag ``flag`` (canonical) set to ``value``."""
    if flag in _INTEGER_FIELDS:
        number = _integer(flag, value)
        change: dict[str, Any] = {_INTEGER_FIELDS[flag]: number if number > 0 else None}
        return dataclasses.replace(settings, **change)
    if flag == "--cache-prompt":
        return dataclasses.replace(settings, prefix_caching=bool(value))
    if flag == "--jinja":
        extra = {k: v for k, v in settings.extra_args.items() if k != flag}
        if value:
            return dataclasses.replace(settings, tool_calling=True, extra_args=extra)
        return dataclasses.replace(settings, tool_calling=False, extra_args={**extra, flag: False})
    if flag in _KV_TYPES:
        settings = _unfold_kv_types(settings)
    settings = dataclasses.replace(settings, extra_args={**settings.extra_args, flag: value})
    return _fold_kv_types(settings) if flag in _KV_TYPES else settings


def parse_setup(text: str, environ: Mapping[str, str] | None = None) -> ParsedSetup:
    """A customer's ``llama-server`` command as settings. ``environ`` is environment the command
    runs with beyond its own ``VAR=value`` prefix; a flag wins over the variable that sets it."""
    env, tokens = launch_tokens(text)
    env = {**(environ or {}), **env}
    if is_container_run(tokens):
        raise ValueError(
            "run llama-server as a local process: Tensward 0.4.0 does not start llama.cpp "
            "containers"
        )
    if not tokens or not runs_llama_server(text):
        raise ValueError(LAUNCHER_HELP)
    notes: list[str] = []
    shown = shlex.join(
        without_secrets(
            [_secret_spelling(word) for word in shlex.split(re.sub(r"\\\r?\n", " ", text))],
            canonical=lambda name: name,
            multi_value_flags=frozenset(),
        )
    )
    # --jinja is on by default from b11669; only --no-jinja turns it off
    settings, model = Settings(tool_calling=True), None
    given: set[str] = set()
    for name, value in _read_flags(tokens[1:]):
        flag, read = read_flag(name, value)
        given.add(flag)
        spelling = "--" + name[2:].replace("_", "-") if name.startswith("--") else name
        if spelling in REMOVED:
            notes.append(f"dropped {spelling}: {REMOVED[spelling]}")
        elif flag in NO_OP or spelling in NO_OP:
            notes.append(f"dropped {spelling}: it has no effect in current builds")
        elif flag == "--model":
            model = str(read)
        elif flag in REFUSED_SOURCES:
            raise ValueError(f"{name} downloads a model; pass the local file with -m")
        elif flag in SECRETS or is_secret(flag):
            notes.append("dropped " + _SECRET_NOTE.format(flag=flag))
        elif flag in PINNED:
            notes.append(f"dropped {spelling}: Tensward sets it")
        elif flag in LOG_FLAGS:
            notes.append(f"dropped {spelling}: it changes the server log Tensward reads")
        elif flag in ("--mmproj", "--mmproj-url"):
            notes.append(f"dropped {spelling}: Tensward measures the text model only")
        elif refuse := _multi_gpu(flag, read):
            raise ValueError(f"{name} {refuse}")
        else:
            settings = apply_flag(settings, flag, read)
    kept: dict[str, str] = {}
    ignored: list[str] = []
    for name, value in env.items():
        found = flag_for_env(name)
        if name.startswith(LOG_ENV):
            notes.append(f"dropped {name}: it changes the server log Tensward reads")
        elif name == "LLAMA_API_KEY":
            notes.append(f"dropped {name}: Tensward sets it")
        elif found is not None:
            known, enabled = found
            if known.canonical in REFUSED_SOURCES:
                raise ValueError(f"{name} downloads a model; pass the local file with -m")
            if known.canonical in given | PINNED | SECRETS | LOG_FLAGS:
                continue
            if known.canonical == "--model":
                model = model or value
            elif refuse := _multi_gpu(known.canonical, value):
                raise ValueError(f"{name} {refuse}")
            else:
                on = enabled and value.lower() not in ("0", "false", "off", "disabled")
                settings = apply_flag(settings, known.canonical, value if known.takes_value else on)
        elif name.startswith(ENGINE_ENV_PREFIX) and not is_secret(name):
            kept[name] = value
        else:
            ignored.append(name)
    if ignored:
        names = ", ".join(sorted(ignored))
        notes.append(f"ignored environment (not engine settings, or credentials): {names}")
    unified = settings.max_concurrent_requests is None or (
        settings.extra_args.get("--kv-unified") is True
    )
    extra = {"--fit": "on", **settings.extra_args, "--kv-unified": unified}
    settings = dataclasses.replace(settings, extra_env=kept, extra_args=extra)
    settings, mapped = _size_pool_per_slot(
        settings, pinned_context="--ctx-size" in given or "LLAMA_ARG_CTX_SIZE" in env
    )
    notes.extend(mapped)
    return ParsedSetup(
        settings, model, None, tuple(notes), shown, executable=_resolve_executable(tokens[0])
    )


def _size_pool_per_slot(settings: Settings, *, pinned_context: bool) -> tuple[Settings, list[str]]:
    """``--kv-unified-per-slot N`` sizes the pool to slots times N when ``-c`` is not given
    (server.cpp:175-184); Tensward pins the pool, so the product becomes ``-c``."""
    extra = dict(settings.extra_args)
    per_slot = extra.pop("--kv-unified-per-slot", None)
    if per_slot is None:
        return settings, []
    settings = dataclasses.replace(settings, extra_args=extra)
    if pinned_context:
        return settings, ["dropped --kv-unified-per-slot: -c is given, which it does not override"]
    slots = settings.max_concurrent_requests or DEFAULT_SLOTS
    pool = _integer("--kv-unified-per-slot", per_slot) * slots
    note = f"mapped --kv-unified-per-slot {per_slot} to -c {pool} ({slots} slots)"
    return dataclasses.replace(settings, max_context_len=pool), [note]


def _multi_gpu(flag: str, value: Any) -> str | None:
    """Why a flag with this value is refused: it spreads the model over several devices."""
    if flag in ("--tensor-split", "--main-gpu", "--rpc"):
        return f"is refused: {MULTI_GPU_REASON}"
    if flag == "--split-mode" and str(value) not in ("none", "layer"):
        return f"is refused: {MULTI_GPU_REASON}"
    if flag == "--device" and "," in str(value):
        return f"is refused: {MULTI_GPU_REASON}"
    return None


def _flag_words(flag: str, value: Any) -> list[str]:
    known = FLAGS.get(flag)
    if value is False:
        return [known.negations[0]] if known is not None and known.negations else []
    return [flag] if value is None or value is True else [flag, str(value)]


def _secret_spelling(word: str) -> str:
    """``word`` with a flag that carries a secret spelled in full (``-hft`` as ``--hf-token``),
    so the secrets helper, which goes by names, blanks its value."""
    name, equals, value = word.partition("=")
    known = flag_for(name) if word.startswith("-") else None
    if known is not None and known.canonical in SECRETS:
        return known.canonical + equals + value
    return word


def split_kv(settings: Settings) -> bool:
    """Whether a launch with ``settings`` splits the KV cache per slot: the setting when it is
    given, else what llama-server does at build 11669 (an explicit ``-np`` turns the unified
    cache off, server.cpp:168-172)."""
    given = settings.extra_args.get("--kv-unified")
    if isinstance(given, bool):
        return not given and settings.max_concurrent_requests is not None
    return settings.max_concurrent_requests is not None


def launch_argv(
    settings: Settings,
    *,
    model: str,
    served_model_name: str,
    host: str,
    port: int,
    serving: bool = False,
) -> list[str]:
    argv = ["-m", model, "--host", host, "--port", str(port), "-a", served_model_name]
    if not serving:
        argv += ["--metrics", "--slots", "-lv", LOG_VERBOSITY]
    return argv + settings_argv(settings)


def settings_argv(settings: Settings) -> list[str]:
    """The words ``settings`` adds to a launch beyond the measurement pins: the resolved
    placement, ``--fit off`` and the KV layout, so the server runs what was measured."""
    argv: list[str] = []
    if "--fit" not in settings.extra_args:
        argv += ["--fit", "off"]
    for flag, field in _INTEGER_FIELDS.items():
        if (value := getattr(settings, field)) is not None:
            argv += [flag, str(value)]
    if settings.kv_cache_dtype != "auto":
        argv += [_KV_TYPES[0], settings.kv_cache_dtype, _KV_TYPES[1], settings.kv_cache_dtype]
    if settings.prefix_caching is not None:
        argv.append("--cache-prompt" if settings.prefix_caching else "--no-cache-prompt")
    if settings.tool_calling:
        argv.append("--jinja")
    argv.append("-no-kvu" if split_kv(settings) else "-kvu")
    for flag, value in settings.extra_args.items():
        if flag not in PINNED and flag not in LOG_FLAGS and flag not in ("--model", "--kv-unified"):
            argv += _flag_words(flag, value)
    return argv


def production_command(text: str, settings: Settings) -> str:
    """The customer's own command with ``settings``' flags written in over theirs. Their
    executable, model, ``--host``, ``--port``, alias and every flag ``settings`` does not set
    stay as they wrote them; Tensward's measurement pins are never added."""
    env, tokens = launch_tokens(text)
    added = settings_argv(settings)
    replaced = {
        known.canonical for word in added if word.startswith("-") and (known := flag_for(word))
    }
    kept: list[str] = []
    for name, value in _read_flags(tokens[1:]):
        known = flag_for(name)
        if known is not None and (
            known.canonical in replaced or (known.canonical in FIT_FLAGS and "--fit" in replaced)
        ):
            continue
        kept += [name] if value is None else [name, value]
    prefix = [f"{key}={value}" for key, value in env.items()]
    return shlex.join([*prefix, tokens[0], *kept, *added])


def _digest(path: str, argv: Sequence[str]) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1 << 20):
            sha.update(chunk)
    sha.update("\0".join(argv).encode())
    return sha.hexdigest()


def _run(argv: Sequence[str]) -> str | None:
    try:
        done = subprocess.run(  # noqa: S603 - list argv, no shell
            list(argv), capture_output=True, text=True, stdin=subprocess.DEVNULL,
            timeout=PROBE_TIMEOUT_S, check=False,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return (done.stdout + done.stderr) if done.returncode == 0 else None


def _cache_dir() -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "tensward" / "llamacpp"


_BuildKey = tuple[str, int, int]
_known: dict[_BuildKey, tuple[BuildIdentity, HelpManifest]] = {}
_active: ContextVar[_BuildKey | None] = ContextVar("llamacpp_build", default=None)


def _build_key(command: str) -> _BuildKey | None:
    found = executable(command)
    if found is None:
        return None
    try:
        stat = os.stat(found)
    except OSError:
        return None
    return (found, stat.st_mtime_ns, stat.st_size)


def known_build(command: str | None = None) -> tuple[BuildIdentity, HelpManifest] | None:
    """The build and flags of the binary ``command`` runs, once :func:`probe_build` has read
    them; without a command, of the binary the current context last probed. Binaries are told
    apart by path, modification time and size, so two projects never read each other's."""
    key = _build_key(command) if command is not None else _active.get()
    return _known.get(key) if key is not None else None


def use_build(command: str) -> None:
    """Make ``command``'s binary the one :func:`known_build` answers for in this context."""
    key = _build_key(command)
    if key is not None and key in _known:
        _active.set(key)
    else:
        probe_build(command)


def probe_build(command: str) -> tuple[BuildIdentity, HelpManifest] | None:
    """The build and the flags of the binary ``command`` runs, from its ``--version`` and
    ``--help``. Both are asked once per binary and argument list and kept under the user cache
    directory; None when the command cannot be run or names no build."""
    found = executable(command)
    if found is None:
        return None
    argv = shlex.split(command)
    cache = _cache_dir() / f"{_digest(found, argv)}.json"
    try:
        stored = json.loads(cache.read_text("utf-8"))
        version_text, help_text = stored["version"], stored["help"]
    except (OSError, ValueError, KeyError):
        version_text, help_text = _run([*argv, "--version"]), _run([*argv, "--help"])
        if version_text is None or help_text is None:
            return None
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"version": version_text, "help": help_text}), "utf-8")
        except OSError:
            pass
    identity = build_identity(version_text)
    if identity is None:
        return None
    found_build = (identity, HelpManifest.from_help(help_text))
    if (key := _build_key(command)) is not None:
        _known[key] = found_build
        _active.set(key)
    return found_build
