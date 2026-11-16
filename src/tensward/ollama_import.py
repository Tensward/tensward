"""An Ollama model's files: where they are, whether they can be read, and the link or copy
into ``~/models``. Names, digests and manifests come from a store another process wrote, so
each is checked before a path or a printed command is built from it."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import hash_minutes, open_regular, read_chunks
from .errors import CHECKPOINT_INVENTORY_UNSAFE, PreflightError
from .formats.gguf import has_chat_template, read_gguf
from .text import printable

DIGEST = re.compile(r"^sha256[:-]([0-9a-fA-F]{64})$")
NAME_PART = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,79}$")
BLOB_NAME = re.compile(r"^sha256-[0-9a-fA-F]{64}$")
FROM_LINE = re.compile(r"^FROM[ \t]+(\S.*)$", re.MULTILINE)
MODEL_LAYER = "application/vnd.ollama.image.model"
PROJECTOR_LAYER = "application/vnd.ollama.image.projector"
MANIFEST_LIST = "application/vnd.ollama.manifest.list.v2+json"
RUNNERS = ("llamacpp", "ggml")
MAX_MANIFEST_BYTES = 4 * 2**20
DISK_MARGIN = 2**30
PROJECTOR_NOTE = "The vision part isn't imported; Tensward tunes the text model."
NOT_VISIBLE = "Ollama's model files aren't visible from this user."
STORE_ROOTS = ("/usr/share/ollama/.ollama/models", "~/.ollama/models")
CONTAINER_ROOT = "/root/.ollama/models"
CONTAINER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_name(name: str) -> tuple[str, str, str, str] | None:
    """``(host, namespace, model, tag)`` of an Ollama name by its own rules, None when any part
    is not allowed."""
    body, _, tag = name.rpartition(":") if ":" in name.rsplit("/", 1)[-1] else (name, "", "latest")
    parts = body.split("/")
    if not 1 <= len(parts) <= 3 or not tag:
        return None
    host, namespace, model = (
        ("registry.ollama.ai", "library", *parts)
        if len(parts) == 1
        else ("registry.ollama.ai", *parts)
        if len(parts) == 2
        else parts
    )
    if not all(
        NAME_PART.match(part) and ".." not in part for part in (host, namespace, model, tag)
    ):
        return None
    return host, namespace, model, tag


def blob_name(digest: str) -> str | None:
    """``sha256-<hex>`` for a valid digest in either spelling."""
    found = DIGEST.match(digest)
    return f"sha256-{found[1]}" if found else None


def slug(name: str) -> str | None:
    text = re.sub(r"[^A-Za-z0-9._-]", "-", name)[:100].strip("-.")
    return text or None


def from_digest(modelfile: str) -> tuple[str, str] | None:
    """The directory and blob name of the first ``FROM`` line that names a blob."""
    for found in FROM_LINE.finditer(modelfile):
        path = found[1].strip()
        name = blob_name(os.path.basename(path))
        if name is not None:
            return os.path.dirname(path), name
    return None


def _read_small(path: Path) -> bytes | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_MANIFEST_BYTES:
            return None
        return os.read(descriptor, info.st_size)
    finally:
        os.close(descriptor)


def _manifest_file(path: Path, blobs: Path) -> bytes | None:
    """A manifest is a symlink to a blob, or a plain file in the older layout."""
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if stat.S_ISLNK(info.st_mode):
        try:
            link = os.readlink(path)
        except OSError:
            return None
        real = Path(os.path.realpath(path))
        if not BLOB_NAME.match(os.path.basename(link)) or real.parent != Path(
            os.path.realpath(blobs)
        ):
            return None
        return _read_small(real)
    return _read_small(path) if stat.S_ISREG(info.st_mode) else None


def _layers(document: Any) -> list[tuple[str, str]] | None:
    """``(mediaType, blob name)`` for each layer; None when the document is not a manifest."""
    if not isinstance(document, dict) or not isinstance(document.get("layers"), list):
        return None
    found = []
    for layer in document["layers"]:
        if not isinstance(layer, dict) or not isinstance(layer.get("mediaType"), str):
            return None
        digest = layer.get("digest")
        name = blob_name(digest) if isinstance(digest, str) else None
        if name is None:
            return None
        found.append((layer["mediaType"], name))
    return found


def read_manifest(root: Path, name: str) -> list[tuple[str, str]] | None:
    """The layers of ``name``'s manifest under ``root``, None when it can't be read or isn't one."""
    parts = parse_name(name)
    if parts is None:
        return None
    host, namespace, model, tag = parts
    blobs = root / "blobs"
    current = "ollama.com" if host == "registry.ollama.ai" else host
    for folder in ("manifests-v2", "manifests"):
        names = (current if folder == "manifests-v2" else host, namespace, model, tag)
        raw = _manifest_file(root / folder / Path(*names), blobs)
        if raw is None:
            continue
        try:
            document = json.loads(raw)
        except ValueError:
            continue
        if isinstance(document, dict) and document.get("mediaType") == MANIFEST_LIST:
            document = _runner_child(document, blobs)
        if (layers := _layers(document)) is not None:
            return layers
    return None


def _runner_child(document: dict[str, Any], blobs: Path) -> Any:
    children = document.get("manifests")
    if not isinstance(children, list):
        return None
    for runner in RUNNERS:
        for child in children:
            if not isinstance(child, dict) or child.get("runner") != runner:
                continue
            digest = child.get("digest")
            name = blob_name(digest) if isinstance(digest, str) else None
            raw = _read_small(blobs / name) if name else None
            try:
                return json.loads(raw) if raw else None
            except ValueError:
                return None
    return None


@dataclass(frozen=True, slots=True)
class Located:
    """The files of one model: blob names in shard order, the store they are in and, for a
    Docker store this user can't read, the container."""

    blobs: tuple[str, ...]
    root: Path | None
    container: str | None
    container_root: str | None
    notes: tuple[str, ...] = ()

    def host_path(self, name: str) -> Path | None:
        return self.root / "blobs" / name if self.root is not None else None

    def container_path(self, name: str) -> str:
        return f"{self.container_root}/blobs/{name}"


def locate(
    model: str,
    show: dict[str, Any],
    *,
    root: Path | None,
    container: str | None = None,
    container_root: str | None = None,
) -> Located | str:
    """Where ``model``'s files are, or the sentence that says why Tensward can't import it.
    ``root`` is the store as this user sees it (None for a store only a container can read)."""
    label = printable(model)
    modelfile = show.get("modelfile")
    source = from_digest(modelfile) if isinstance(modelfile, str) else None
    if parse_name(model) is None or source is None:
        return f"{label} has a name or a file Tensward won't copy from. {NOT_VISIBLE}"
    directory, main = source
    if container_root:
        expected = f"{container_root}/blobs"
    elif root is not None:
        expected = str(root / "blobs")
    else:
        return NOT_VISIBLE
    if directory != expected:
        return NOT_VISIBLE
    layers = read_manifest(root, model) if root is not None else None
    notes: list[str] = []
    blobs = [main]
    if layers is not None:
        models = [name for kind, name in layers if kind == MODEL_LAYER]
        if any(kind == "application/vnd.ollama.image.adapter" for kind, _ in layers):
            return (
                f"{label} has a fine-tune adapter. Tensward can't import it, and the base model "
                "alone would answer differently."
            )
        if any(kind == "application/vnd.ollama.image.tensor" for kind, _ in layers):
            return (
                f"{label} is stored as safetensors, so llama.cpp can't load it. To tune it, "
                "download its .gguf from Hugging Face."
            )
        if any(kind == PROJECTOR_LAYER for kind, _ in layers):
            notes.append(PROJECTOR_NOTE)
        if not models or models[0] != main:
            return NOT_VISIBLE
        blobs = models
    else:
        info = show.get("model_info")
        count = info.get("split.count") if isinstance(info, dict) else None
        if isinstance(count, int) and count > 1:
            return (
                f"{label} is split into {count} files and Tensward can't read the store's list. "
                "Make the store readable, or download the GGUF from Hugging Face."
            )
    return Located(tuple(blobs), root, container, container_root, tuple(notes))


def shown_path(path: Path) -> str:
    """``path`` with the home folder as ``~``, for reading."""
    home = str(Path.home())
    text = str(path)
    return "~" + text[len(home) :] if text == home or text.startswith(home + os.sep) else text


@dataclass(frozen=True, slots=True)
class Target:
    """Where the model goes: one file, or the shards of a split model in a folder of their own."""

    files: tuple[Path, ...]

    @property
    def first(self) -> Path:
        return self.files[0]


def target_for(model: str, count: int, home: Path | None = None) -> Target | None:
    name = slug(model)
    if name is None:
        return None
    folder = (home or Path.home()) / "models"
    if count == 1:
        return Target((folder / f"{name}.gguf",))
    return Target(
        tuple(folder / name / f"{name}-{n:05d}-of-{count:05d}.gguf" for n in range(1, count + 1))
    )


def nearest_existing(path: Path) -> Path:
    while path != path.parent and not os.path.exists(path):
        path = path.parent
    return path


def source_kind(path: Path, target: Path) -> str:
    """``link`` (readable, ours, on the target's device), ``copy`` (readable otherwise),
    ``unreadable`` or ``unsafe`` (not a plain file)."""
    try:
        with open_regular(path) as (descriptor, _):
            info = os.fstat(descriptor)
    except PreflightError as error:
        return "unsafe" if error.code == CHECKPOINT_INVENTORY_UNSAFE else "unreadable"
    except OSError:
        return "unreadable"
    same_device = info.st_dev == os.stat(nearest_existing(target.parent)).st_dev
    return "link" if info.st_uid == os.getuid() and same_device else "copy"


def target_state(blob: Path | None, target: Path) -> str:
    """``absent``, ``same`` (same file as the blob), ``maybe`` (same size as the blob, or the
    blob can't be examined: judged when the user continues), ``different`` or ``symlink``."""
    try:
        existing = os.lstat(target)
    except FileNotFoundError:
        return "absent"
    except OSError:
        return "different"
    if stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode):
        return "symlink"
    try:
        seen = os.stat(blob) if blob is not None else None
    except OSError:
        seen = None
    if seen is None:
        return "maybe"
    if os.path.samestat(existing, seen):
        return "same"
    return "maybe" if existing.st_size == seen.st_size else "different"


def matches_digest(path: Path, blob: str) -> bool:
    """Whether the SHA-256 of ``path`` is the one in the blob's name."""
    digest = hashlib.sha256()
    try:
        with open_regular(path) as (descriptor, size):
            for chunk in read_chunks(descriptor, size, path.name):
                digest.update(chunk)
    except (PreflightError, OSError):
        return False
    return digest.hexdigest() == blob.removeprefix("sha256-").lower()


def gguf_problem(path: Path, label: str) -> str | None:
    """Why the file at ``path`` can't be used by llama.cpp, from its header and template."""
    try:
        read_gguf(path)
        chat = has_chat_template(path)
    except (PreflightError, OSError, ValueError):
        return f"{label} is not a GGUF file llama.cpp can read. Download its .gguf instead."
    if not chat:
        return (
            "This Ollama file has no chat template, so llama.cpp can't chat with it. "
            "Download the .gguf from Hugging Face instead."
        )
    return None


def free_problem(need: int, target: Target) -> str | None:
    """The sentence when ``need`` bytes (and a margin) don't fit beside ``target``."""
    folder = nearest_existing(target.first.parent)
    try:
        free = shutil.disk_usage(folder).free
    except OSError:
        return None
    if free >= need + DISK_MARGIN:
        return None
    sentence = (
        f"This copy needs {need / 2**30:.1f} GiB but only {free / 2**30:.1f} GiB is free in "
        f"{shown_path(target.first.parent)}. Free some space and run tensward init again."
    )
    leftover = [f for f in target.files if os.path.lexists(f"{f}.part")]
    if leftover:
        sentence += (
            f" A leftover {shown_path(Path(f'{leftover[0]}.part'))} from an earlier copy "
            "can be deleted."
        )
    return sentence


def link_or_copy(blob: Path, target: Path, *, link: bool) -> str:
    """``linked`` or ``copied``. Nothing already at ``target`` is ever replaced: a hard link or
    a finished temporary file is linked to the name, which fails if the name is taken."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with open_regular(blob) as (source, _):
        if link:
            try:
                os.link(blob, target, follow_symlinks=False)
                return "linked"
            except FileExistsError:
                raise
            except OSError:
                pass
        descriptor, name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temp = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as stream, os.fdopen(os.dup(source), "rb") as reader:
                shutil.copyfileobj(reader, stream, 2**20)
            try:
                os.link(temp, target)
            except FileExistsError:
                raise
            except OSError:
                _create_exclusive(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        return "copied"


def _create_exclusive(temp: Path, target: Path) -> None:
    """The copy for a filesystem without hard links."""
    descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with open(temp, "rb") as source, os.fdopen(descriptor, "wb") as stream:
            shutil.copyfileobj(source, stream, 2**20)
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def copy_chain(located: Located, target: Target, kind: str, *, sudo: bool = False) -> list[str]:
    """The commands the user runs for each file when Tensward can't: ``ln``, a plain ``cp``
    (readable but not linkable), ``sudo cp`` (the store is not readable) or ``docker cp``. A copy
    goes to ``.part`` and only a finished one gets the model's name; ``cp`` never follows a
    link in the store."""
    lines = []
    for blob, file in zip(located.blobs, target.files, strict=True):
        part = shlex.quote(f"{file}.part")
        path = located.host_path(blob)
        steps = [f"mkdir -p {shlex.quote(str(file.parent))}"]
        if kind == "ln":
            steps.append(f"ln {shlex.quote(str(path))} {shlex.quote(str(file))}")
        else:
            if kind == "docker":
                source = shlex.quote(f"{located.container}:{located.container_path(blob)}")
                steps.append(f"docker cp {source} {part}")
            else:
                prefix = "sudo cp" if sudo else "cp"
                steps.append(
                    f"{prefix} --no-dereference --remove-destination {shlex.quote(str(path))} "
                    f"{part}"
                )
            if sudo:
                steps.append(f'sudo chown -h "$USER:" {part}')
            steps.append(f"mv -n {part} {shlex.quote(str(file))}")
        lines.append(" && ".join(steps))
    return lines


def check_minutes(size: int) -> str:
    low, high = hash_minutes(size)
    return f"Checking the file ({size / 2**30:.1f} GiB, up to ~{low}-{high} min)..."
