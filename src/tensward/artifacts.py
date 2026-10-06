"""What identifies a checkpoint: its files, what they declare, and the SHA-256 over their bytes.

These records are stored in ``project.json``, so their shape is part of the on-disk format.
"""

from __future__ import annotations

import contextvars
import errno
import hashlib
import json
import os
import stat
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Iterator, Literal

from pydantic import Field, StringConstraints

from .contracts import DigestHex, Identifier, PositiveInt, StrictModel
from .errors import CHECKPOINT_CHANGED, CHECKPOINT_INVENTORY_UNSAFE, PreflightError
from .files import load_strict_json, write_private
from .progress import Note, emit

AbsolutePath = Annotated[str, StringConstraints(pattern=r"^/[^\x00]*$")]
WeightPrecision = Literal["bf16", "fp16", "fp8", "int8", "int4"]
ActivationDtype = Literal["bfloat16", "float16"]

CHUNK_BYTES = 1024 * 1024
SLOW_HASH_S = 5.0  # after this long, say that the weights are being hashed


class ArtifactVariant(StrictModel):
    """One weight and activation precision the checkpoint provides.

    ``quantization_method`` is what the checkpoint declares ("none" when unquantized).
    ``group_size`` is None for per-tensor or per-channel scales; ``activation_scheme`` is None when
    activations stay in the model dtype.
    """

    weight_precision: WeightPrecision
    activation_dtype: ActivationDtype
    quantization_method: str
    weight_bits: int | None = None
    group_size: int | None = None
    activation_scheme: Literal["static", "dynamic"] | None = None

    @property
    def label(self) -> str | None:
        """A short description of the quantization, e.g. ``awq int4 group 128``."""
        if self.quantization_method == "none":
            return None
        parts = [self.quantization_method, self.weight_precision]
        if self.group_size is not None:
            parts.append(f"group {self.group_size}")
        if self.activation_scheme is not None:
            parts.append(f"{self.activation_scheme} activations")
        return " ".join(parts)


class ThinkingSupport(StrictModel):
    """Always unsupported today; kept because it is part of the stored record."""

    supported: bool
    profile_ids: tuple[Identifier, ...] = ()


class ArtifactFingerprint(StrictModel):
    algorithm: Literal["sha256"] = "sha256"
    value: DigestHex


class ArtifactMetadata(StrictModel):
    engine_name: str
    engine_build: str
    context_limit: PositiveInt
    weight_bytes: PositiveInt  # tensor bytes, from the weight files' headers
    architectures: tuple[str, ...] = Field(min_length=1)
    variants: tuple[ArtifactVariant, ...] = Field(min_length=1)
    thinking: ThinkingSupport
    fingerprint: ArtifactFingerprint


class ArtifactFiles(StrictModel):
    """The single-file parts: ``auxiliary`` lists the tokenizer companions, the chat template
    and the shard index that are present."""

    config: str
    generation_config: str
    tokenizer: str
    auxiliary: tuple[str, ...] = ()


class ArtifactWeights(StrictModel):
    shard_count: PositiveInt
    shards: tuple[str, ...] = Field(min_length=1)


class ArtifactEntry(StrictModel):
    """One registered checkpoint: where it is, which files make it up, and what they declare."""

    root: AbsolutePath
    files: ArtifactFiles
    weights: ArtifactWeights
    metadata: ArtifactMetadata

    @property
    def path(self) -> Path:
        return Path(self.root)


def declared_files(files: ArtifactFiles, weights: ArtifactWeights) -> list[tuple[str, str]]:
    """The ``(role, name)`` pairs in fingerprint order: the three documents, the auxiliary
    files sorted, then the shards in their declared order (which is semantic)."""
    return [
        ("config", files.config),
        ("generation_config", files.generation_config),
        ("tokenizer", files.tokenizer),
        *(("auxiliary", name) for name in sorted(files.auxiliary)),
        *(("shard", name) for name in weights.shards),
    ]


SYMLINK_REFUSAL = (
    "{name} is a symlink; download the model with `hf download <repo> --local-dir <dir>` "
    "to get plain files"
)


def locate(path: Path) -> Path:
    """Where ``path``'s content is. A symlink inside a Hugging Face cache snapshot that points
    into the same cache repository directory (``snapshots/<rev>/x -> ../../blobs/<hash>``) is
    followed once; any other symlink is returned unchanged, and so refused when opened."""
    if not path.is_symlink():
        return path
    repository = next((p for p in path.parents if p.name.startswith("models--")), None)
    if repository is None:
        return path
    target = Path(os.path.realpath(path))
    return target if target.is_relative_to(os.path.realpath(repository)) else path


@contextmanager
def open_regular(path: Path) -> Iterator[tuple[int, int]]:
    """Yield ``(descriptor, size)`` of a regular file, opened without following a symlink (a
    link could point out of the checkpoint, except inside a Hugging Face cache repository) and
    without blocking on a FIFO."""
    path = locate(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise PreflightError(
                CHECKPOINT_INVENTORY_UNSAFE, SYMLINK_REFUSAL.format(name=path.name)
            ) from None
        raise PreflightError(CHECKPOINT_CHANGED, f"{path.name} could not be read") from None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise PreflightError(CHECKPOINT_INVENTORY_UNSAFE, f"{path.name} is not a regular file")
        yield descriptor, info.st_size
    finally:
        os.close(descriptor)


def read_chunks(descriptor: int, size: int, name: str) -> Iterator[bytes]:
    """The first ``size`` bytes of an open file; a file that ends early was changed."""
    remaining = size
    while remaining:
        chunk = os.read(descriptor, min(CHUNK_BYTES, remaining))
        if not chunk:
            raise PreflightError(CHECKPOINT_CHANGED, f"{name} shrank while it was read")
        yield chunk
        remaining -= len(chunk)


def fingerprint_files(
    root: Path, declared: list[tuple[str, str]], *, cache: Path | None = None, refresh: bool = False
) -> str:
    """SHA-256 over each file framed as ``file\\0 role \\0 name \\0 size(8 bytes) bytes``.

    Names are relative, so the same checkpoint has one fingerprint wherever it lives.

    With ``cache`` (a private file), a checkpoint whose files have the same (inode, size,
    mtime_ns) as when last hashed is not hashed again, so a changed file is still noticed
    unless it kept its size and mtime. ``refresh`` hashes anyway and rewrites the cache.
    """
    stamp = _stamp(root, declared)
    if cache is not None and not refresh and stamp is not None:
        try:
            known = load_strict_json(cache.read_bytes(), "the weights cache", CHECKPOINT_CHANGED)
            if known["stamp"] == stamp:
                return str(known["fingerprint"])
        except (OSError, PreflightError, KeyError, TypeError):
            pass  # no usable cache: hash
    hasher = hashlib.sha256()
    try:
        total: int | None = sum(os.stat(locate(root / name)).st_size for _, name in declared)
    except OSError:
        total = None
    notice = threading.Timer(
        SLOW_HASH_S, contextvars.copy_context().run, [_announce, _hash_notice(total)]
    )
    notice.daemon = True
    notice.start()
    try:
        for role, name in declared:
            with open_regular(root / name) as (descriptor, size):
                hasher.update(b"file\x00%b\x00%b\x00" % (role.encode(), name.encode()))
                hasher.update(size.to_bytes(8, "big"))
                for chunk in read_chunks(descriptor, size, name):
                    hasher.update(chunk)
    finally:
        notice.cancel()
    fingerprint = hasher.hexdigest()
    if cache is not None and stamp is not None and stamp == _stamp(root, declared):
        write_private(cache, json.dumps({"stamp": stamp, "fingerprint": fingerprint}))
    return fingerprint


def _announce(text: str) -> None:
    emit(Note(text=text))


def _hash_notice(total_bytes: int | None) -> str:
    """What hashing will take, from the measured rate of 5 GB in about 1-2 minutes; the size is
    None when a file could not be examined."""
    if total_bytes is None:
        return "hashing the model weights"
    gigabytes = total_bytes / 1e9
    low, high = max(1, round(gigabytes / 5)), max(2, round(gigabytes * 2 / 5))
    return f"hashing {gigabytes:.1f} GB of model weights (about {low}-{high} min)"


def _stamp(root: Path, declared: list[tuple[str, str]]) -> list[Any] | None:
    """Where each declared file is and what it looks like; None if one cannot be examined."""
    try:
        stats = [(name, os.stat(locate(root / name))) for _, name in declared]
    except OSError:
        return None
    return [
        str(root),
        *([name, info.st_ino, info.st_size, info.st_mtime_ns] for name, info in stats),
    ]
