"""GGUF checkpoints: one model file, or a split set, that llama.cpp loads.

The model file holds the tokenizer and the chat template; nothing else is part of the identity.
A file given as ``--model`` is the model: its split set comes from its own name, and other GGUF
files in the folder never matter. A folder is accepted when it holds exactly one model (a
multimodal projector, ``mmproj-*``, is not one).

Nothing is executed, imported or downloaded: only the header, the metadata and the tensor
directory are read, never the weights. Files are opened without following symlinks, bar a
Hugging Face cache snapshot's own links.
"""

from __future__ import annotations

import io
import math
import os
import re
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..anatomy import (
    BLOCK_KV_BYTES,
    AttentionGroup,
    Components,
    ModelAnatomy,
    Moe,
)
from ..artifacts import (
    ArtifactEntry,
    ArtifactFingerprint,
    ArtifactMetadata,
    ArtifactVariant,
    ArtifactWeights,
    GgufFiles,
    ThinkingSupport,
    declared_files,
    fingerprint_files,
    open_regular,
)
from ..errors import (
    CHECKPOINT_INVENTORY_UNEXPECTED,
    CHECKPOINT_LAYOUT_INVALID,
    CHECKPOINT_PRECISION_UNSUPPORTED,
    PreflightError,
    not_found_message,
)
from ..progress import Note, emit

SUFFIX = ".gguf"
PROJECTOR_PREFIX = "mmproj"
MAX_TENSORS = 1 << 20
MAX_KV_PAIRS = 1 << 16
MAX_STRING_BYTES = 1 << 24
MAX_ARRAY_KEPT = 1 << 13  # longer arrays (token tables) are skipped
MAX_ALIGNMENT = 1 << 20
MAX_DIMS = 4
DEFAULT_ALIGNMENT = 32
SPLIT_NAME = re.compile(r"^(?P<stem>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$")
ARCHITECTURE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,63}$")

# Type ids, names, block sizes and block bytes of ggml/src/ggml.c `type_traits` and
# ggml/src/ggml-common.h at llama.cpp b11669 (620d2225).
GGML_TYPES: Mapping[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
    40: ("NVFP4", 64, 36),
    41: ("Q1_0", 128, 18),
    42: ("Q2_0", 64, 18),
}

# llama.h `llama_ftype`; the GUESSED flag (1024) is not part of the name.
FILE_TYPES: Mapping[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
    11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
    17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
    23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
    29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
    38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0", 41: "Q2_0",
}  # fmt: skip
FILE_TYPE_GUESSED = 1024

_SCALARS: Mapping[int, str] = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q",
    12: "<d",
}  # fmt: skip
_STRING, _ARRAY = 8, 9
_SKIPPED = object()

ROLES = ("embedding", "output", "attention", "ffn", "routed_experts", "shared_experts", "other")
_BLOCK = re.compile(r"^blk\.(?P<layer>\d+)\.(?P<rest>.+)$")


@dataclass(frozen=True, slots=True)
class GgufTensor:
    name: str
    ggml_type: str
    shape: tuple[int, ...]
    nbytes: int


@dataclass(frozen=True, slots=True)
class GgufFile:
    path: Path
    version: int
    metadata: Mapping[str, Any]
    tensors: tuple[GgufTensor, ...]
    data_offset: int


def _invalid(name: str, what: str) -> PreflightError:
    return PreflightError(CHECKPOINT_LAYOUT_INVALID, f"{name} {what}")


class _Cursor:
    def __init__(self, stream: io.BufferedReader, size: int, name: str) -> None:
        self.stream, self.size, self.name = stream, size, name

    def take(self, count: int) -> bytes:
        data = self.stream.read(count)
        if len(data) != count:
            raise _invalid(self.name, "ends before its header does")
        return data

    def unpack(self, fmt: str) -> Any:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def length(self, limit: int = MAX_STRING_BYTES) -> int:
        count = self.unpack("<Q")
        if count > limit or self.stream.tell() + count > self.size:
            raise _invalid(self.name, "has a string or array that does not fit")
        return int(count)

    def string(self) -> str:
        data = self.take(self.length())
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            raise _invalid(self.name, "has a string that is not UTF-8") from None

    def skip(self, count: int) -> None:
        if self.stream.tell() + count > self.size:
            raise _invalid(self.name, "has an array that does not fit")
        self.stream.seek(count, os.SEEK_CUR)


def _value(cursor: _Cursor, kind: int) -> Any:
    if kind in _SCALARS:
        return cursor.unpack(_SCALARS[kind])
    if kind == _STRING:
        return cursor.string()
    if kind != _ARRAY:
        raise _invalid(cursor.name, f"has a metadata value of unknown type {kind}")
    element = cursor.unpack("<I")
    count = cursor.unpack("<Q")
    if element == _STRING:
        if count * 8 > cursor.size:
            raise _invalid(cursor.name, "has an array that does not fit")
        for _ in range(count):
            cursor.skip(cursor.length(cursor.size))
        return _SKIPPED
    if element not in _SCALARS:
        raise _invalid(cursor.name, f"has an array of unsupported type {element}")
    width = struct.calcsize(_SCALARS[element])
    if count > MAX_ARRAY_KEPT:
        cursor.skip(count * width)
        return _SKIPPED
    layout = f"<{count}{_SCALARS[element][1]}"
    return list(struct.unpack(layout, cursor.take(count * width)))


def read_gguf(path: Path) -> GgufFile:
    """The header, metadata and tensor directory of one GGUF file. Arrays of strings and long
    arrays (the vocabulary) are left out of the metadata."""
    with open_regular(path) as (descriptor, size), os.fdopen(os.dup(descriptor), "rb") as stream:
        return _parse(_Cursor(stream, size, path.name), path)


def _parse(cursor: _Cursor, path: Path) -> GgufFile:
    name = cursor.name
    if cursor.take(4) != b"GGUF":
        raise _invalid(name, "is not a GGUF file")
    version = cursor.unpack("<I")
    if version not in (2, 3):
        raise _invalid(name, f"is GGUF version {version}; versions 2 and 3 are supported")
    tensor_count, pair_count = cursor.unpack("<Q"), cursor.unpack("<Q")
    if not 0 < tensor_count <= MAX_TENSORS or pair_count > MAX_KV_PAIRS:
        raise _invalid(name, "declares an implausible number of tensors or metadata pairs")
    metadata: dict[str, Any] = {}
    for _ in range(pair_count):
        key = cursor.string()
        value = _value(cursor, cursor.unpack("<I"))
        if key in metadata:
            raise _invalid(name, f"repeats the metadata key {key[:64]!r}")
        if value is not _SKIPPED:
            metadata[key] = value
    alignment = metadata.get("general.alignment", DEFAULT_ALIGNMENT)
    if not (isinstance(alignment, int) and 0 < alignment <= MAX_ALIGNMENT):
        raise _invalid(name, "declares an invalid alignment")
    if alignment & (alignment - 1):
        raise _invalid(name, "declares an alignment that is not a power of two")
    infos: list[tuple[str, str, tuple[int, ...], int, int]] = []
    seen: set[str] = set()
    for _ in range(tensor_count):
        tensor = cursor.string()
        dims = cursor.unpack("<I")
        if dims > MAX_DIMS or tensor in seen:
            raise _invalid(name, "has a malformed tensor directory")
        seen.add(tensor)
        shape = struct.unpack(f"<{dims}Q", cursor.take(8 * dims))
        type_id, offset = cursor.unpack("<I"), cursor.unpack("<Q")
        if type_id not in GGML_TYPES:
            raise PreflightError(
                CHECKPOINT_PRECISION_UNSUPPORTED,
                f"{name} has a tensor of unsupported ggml type {type_id}",
            )
        kind, block, block_bytes = GGML_TYPES[type_id]
        if not all(shape) or (shape[0] if shape else 1) % block:
            raise _invalid(name, f"has a tensor whose shape does not fit its type {kind}")
        infos.append((tensor, kind, shape, math.prod(shape) // block * block_bytes, offset))
    data_offset = -(-cursor.stream.tell() // alignment) * alignment
    position = end = 0
    for tensor, _, _, nbytes, offset in infos:
        if offset != position:
            raise _invalid(name, f"has a tensor ({tensor[:64]}) at an unexpected offset")
        end = offset + nbytes
        position += -(-nbytes // alignment) * alignment
    if data_offset + end > cursor.size:
        raise _invalid(name, "is shorter than its tensor directory says")
    tensors = tuple(GgufTensor(n, kind, shape, nbytes) for n, kind, shape, nbytes, _ in infos)
    return GgufFile(path, version, metadata, tensors, data_offset)


# --- the files -----------------------------------------------------------------------


def _shard_set(path: Path) -> tuple[Path, ...]:
    match = SPLIT_NAME.match(path.name)
    if match is None:
        return (path,)
    count = int(match["count"])
    shards = tuple(
        path.with_name(f"{match['stem']}-{number:05d}-of-{count:05d}{SUFFIX}")
        for number in range(1, count + 1)
    )
    missing = next((shard for shard in shards if not os.path.lexists(shard)), None)
    if missing is not None:
        raise PreflightError(
            CHECKPOINT_LAYOUT_INVALID, f"the split model is missing {missing.name}"
        )
    return shards


def _gguf_names(root: Path) -> list[str]:
    with os.scandir(root) as entries:
        return sorted(e.name for e in entries if e.name.endswith(SUFFIX) and e.is_file())


def _candidates(root: Path) -> list[str]:
    return [n for n in _gguf_names(root) if not n.startswith(PROJECTOR_PREFIX)]


def model_files(root: Path) -> tuple[Path, ...]:
    """The files of the model ``root`` names: a GGUF file (its split set comes from its own name)
    or a folder that holds exactly one model. Refuses what is ambiguous or incomplete."""
    if root.is_dir():
        names = _candidates(root)
        models: dict[tuple[str, ...], Path] = {}
        for name in names:
            match = SPLIT_NAME.match(name)
            key = (match["stem"], match["count"]) if match else (name,)
            models.setdefault(key, root / name)
        if not models:
            raise PreflightError(CHECKPOINT_LAYOUT_INVALID, "the folder holds no GGUF model file")
        if len(models) > 1:
            raise PreflightError(
                CHECKPOINT_INVENTORY_UNEXPECTED,
                f"{root} holds more than one model file ({', '.join(names)}); pass the one you "
                "want to serve as --model <file>",
            )
        return _shard_set(next(iter(models.values())))
    if not root.name.endswith(SUFFIX):
        raise PreflightError(
            CHECKPOINT_LAYOUT_INVALID, f"{root.name} is not a GGUF file or a checkpoint folder"
        )
    return _shard_set(root)


# --- registration --------------------------------------------------------------------


class Gguf:
    """The llama.cpp layout: a GGUF file, or the shards of a split one."""

    name = "gguf"
    label = "GGUF"

    def detect(self, root: Path) -> bool:
        try:
            if root.is_dir():
                return bool(_candidates(root))
            return root.name.endswith(SUFFIX)
        except OSError:
            return False

    def register(
        self,
        root: Path,
        *,
        engine_build: str,
        engine_name: str,
        cache: Path | None = None,
        refresh: bool = False,
        trust_remote_code: bool = False,
    ) -> tuple[ArtifactEntry, ModelAnatomy]:
        if not root.exists():
            raise PreflightError(
                CHECKPOINT_LAYOUT_INVALID, not_found_message("the GGUF model", root)
            )
        paths = model_files(root)
        directory = paths[0].parent
        if root.is_dir() and len(_gguf_names(root)) > len(_candidates(root)):
            emit(Note(text="ignoring the multimodal projector; Tensward serves the text model"))
        files = tuple(read_gguf(path) for path in paths)
        metadata = _joined_metadata(files)
        tensors = tuple(t for f in files for t in f.tensors)
        if len({t.name for t in tensors}) != len(tensors):
            raise PreflightError(
                CHECKPOINT_LAYOUT_INVALID, "the shards of the split model repeat a tensor"
            )
        arch = _architecture(metadata, paths[0].name)
        anatomy = _anatomy(metadata, arch, tensors, paths[0].name)
        gguf = GgufFiles(gguf=paths[0].name)
        weights = ArtifactWeights(shard_count=len(paths), shards=tuple(p.name for p in paths))
        fingerprint = fingerprint_files(
            directory, declared_files(gguf, weights), cache=cache, refresh=refresh
        )
        bits = anatomy.bits_per_weight or 0.0
        entry = ArtifactEntry(
            root=str(directory),
            files=gguf,
            weights=weights,
            metadata=ArtifactMetadata(
                engine_name=engine_name,
                engine_build=engine_build,
                context_limit=_positive(metadata, f"{arch}.context_length", paths[0].name),
                weight_bytes=sum(t.nbytes for t in tensors),
                architectures=(arch,),
                variants=(
                    ArtifactVariant(
                        weight_precision="gguf",
                        activation_dtype="engine",
                        quantization_method=f"gguf:{_file_type(metadata)}",
                        weight_bits=round(bits),
                    ),
                ),
                thinking=ThinkingSupport(supported=False),
                fingerprint=ArtifactFingerprint(value=fingerprint),
            ),
        )
        return entry, anatomy


GGUF = Gguf()


def has_chat_template(path: Path) -> bool:
    """Whether the model file carries a chat template."""
    return bool(read_gguf(path).metadata.get("tokenizer.chat_template"))


def _joined_metadata(files: tuple[GgufFile, ...]) -> Mapping[str, Any]:
    """The first shard's metadata, after checking that every shard says where it belongs."""
    for index, shard in enumerate(files):
        number, count = shard.metadata.get("split.no"), shard.metadata.get("split.count")
        if len(files) > 1 and (number != index or count != len(files)):
            raise _invalid(shard.path.name, "does not say it is that shard of the split model")
    return files[0].metadata


def _architecture(metadata: Mapping[str, Any], name: str) -> str:
    arch = metadata.get("general.architecture")
    if not isinstance(arch, str) or not ARCHITECTURE_NAME.match(arch):
        raise _invalid(name, "declares no valid general.architecture")
    return arch


def _positive(metadata: Mapping[str, Any], key: str, name: str) -> int:
    value = metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise _invalid(name, f"declares no positive {key}")
    return value


def _file_type(metadata: Mapping[str, Any]) -> str:
    declared = metadata.get("general.file_type")
    if not isinstance(declared, int) or isinstance(declared, bool):
        return "unknown"
    number = declared & ~FILE_TYPE_GUESSED
    return FILE_TYPES.get(number, f"ftype{number}")


# --- anatomy -------------------------------------------------------------------------


def _role(name: str) -> str:
    if name.startswith(("token_embd.", "per_layer_token_embd.")):
        return "embedding"
    if name.startswith("output."):
        return "output"
    block = _BLOCK.match(name)
    if block is None:
        return "other"
    rest = block["rest"]
    if rest.startswith("attn_"):
        return "attention"
    if rest.startswith("ffn_") and "_exps" in rest:
        return "routed_experts"
    if rest.startswith("ffn_") and "_shexp" in rest:
        return "shared_experts"
    if rest.startswith("ffn_"):
        return "ffn"
    return "other"


def _per_layer(value: Any, layers: int) -> list[int | None]:
    """A setting that is one number or one per layer; None where it is not declared (0 in a
    per-layer list is declared: that layer has none)."""
    if isinstance(value, list) and len(value) == layers:
        return [v if isinstance(v, int) and v >= 0 else None for v in value]
    return [_count(value)] * layers


def _anatomy(
    metadata: Mapping[str, Any], arch: str, tensors: tuple[GgufTensor, ...], file_name: str
) -> ModelAnatomy:
    layers = _positive(metadata, f"{arch}.block_count", file_name)
    hidden = _positive(metadata, f"{arch}.embedding_length", file_name)
    by_role: Counter[str] = Counter()
    layer_bytes, expert_bytes = [0] * layers, [0] * layers
    kinds: Counter[str] = Counter()
    kinds_by_role: dict[str, set[str]] = {}
    elements: Counter[str] = Counter()
    for tensor in tensors:
        role = _role(tensor.name)
        by_role[role] += tensor.nbytes
        kinds[tensor.ggml_type] += tensor.nbytes
        kinds_by_role.setdefault(role, set()).add(tensor.ggml_type)
        elements[role] += math.prod(tensor.shape)
        block = _BLOCK.match(tensor.name)
        if block is not None and int(block["layer"]) < layers:
            layer_bytes[int(block["layer"])] += tensor.nbytes
            if role == "routed_experts":
                expert_bytes[int(block["layer"])] += tensor.nbytes
    total = sum(by_role.values())

    def parts(figures: Mapping[str, int]) -> Components:
        return Components(
            text_dense=figures["attention"] + figures["ffn"] + figures["other"],
            embedding=figures["embedding"],
            lm_head=figures["output"],
            routed_experts=figures["routed_experts"],
            shared_experts=figures["shared_experts"],
            vision=0,
            audio=0,
            unclassified=0,
        )

    reasons: list[str] = []
    attention, swa, window, shared = _attention(metadata, arch, layers, hidden, tensors, reasons)
    experts = _count(metadata.get(f"{arch}.expert_count"))
    used = _count(metadata.get(f"{arch}.expert_used_count"))
    routed_layers = sum(1 for b in expert_bytes if b)
    moe = None
    if experts and experts > 1 and used and routed_layers:
        moe = Moe(
            experts, used, routed_layers, by_role["routed_experts"] // (routed_layers * experts)
        )
    embedding = next((t for t in tensors if t.name == "token_embd.weight"), None)
    vocab = embedding.shape[-1] if embedding is not None and len(embedding.shape) == 2 else None
    quantization = f"gguf:{_file_type(metadata)}"
    return ModelAnatomy(
        model_type=arch,
        validated=False,
        hidden=hidden,
        layers=layers,
        vocab=vocab,
        tied_embeddings=by_role["output"] == 0,
        token_embedding_bytes=0 if embedding is None else embedding.nbytes,
        components=parts({role: by_role[role] for role in ROLES}),
        params=parts({role: elements[role] for role in ROLES}),
        precisions={role: tuple(sorted(found)) for role, found in sorted(kinds_by_role.items())},
        attention=attention,
        moe=moe,
        vision=None,
        modalities=("text",),
        unavailable=tuple(reasons),
        kv_shared_layers=shared,
        mixed_head_dims=len({g.head_dim for g in attention}) > 1,
        architecture=arch,
        dtype=max(kinds, key=lambda kind: kinds[kind]),
        quantization=quantization,
        weight_bits=round(8 * total / sum(elements.values())),
        bits_per_weight=8 * total / sum(elements.values()),
        weight_bytes_by_role={role: by_role[role] for role in ROLES},
        layer_bytes=tuple(layer_bytes),
        expert_layer_bytes=tuple(expert_bytes),
        sliding_window=window,
        swa_layers=swa,
        experts=experts,
        experts_used=used,
    )


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _attention(
    metadata: Mapping[str, Any],
    arch: str,
    layers: int,
    hidden: int,
    tensors: tuple[GgufTensor, ...],
    reasons: list[str],
) -> tuple[tuple[AttentionGroup, ...], frozenset[int], int | None, int]:
    """The attention groups (a group is the layers that cache the same shape), the sliding
    layers, the declared window and the trailing layers that reuse earlier layers' cache."""
    if any(t.name.startswith(("blk.0.ssm_", "blk.0.time_mix")) for t in tensors):
        reasons.append("layers with a state for each sequence are not modelled, so no KV is priced")
        return (), frozenset(), None, 0
    declared = _per_layer(metadata.get(f"{arch}.attention.head_count"), layers)
    heads = [h for h in declared if h is not None]
    if len(heads) != layers:
        reasons.append("the GGUF declares no attention head count")
        return (), frozenset(), None, 0
    kv_heads = _per_layer(metadata.get(f"{arch}.attention.head_count_kv"), layers)
    key_len = _count(metadata.get(f"{arch}.attention.key_length"))
    value_len = _count(metadata.get(f"{arch}.attention.value_length"))
    swa_key = _count(metadata.get(f"{arch}.attention.key_length_swa")) or key_len
    swa_value = _count(metadata.get(f"{arch}.attention.value_length_swa")) or value_len
    window = _count(metadata.get(f"{arch}.attention.sliding_window"))
    pattern = metadata.get(f"{arch}.attention.sliding_window_pattern")
    sliding: frozenset[int] = frozenset()
    if window and isinstance(pattern, list) and len(pattern) == layers:
        sliding = frozenset(i for i, flag in enumerate(pattern) if flag)
    if window and not sliding:
        reasons.append("a sliding window is declared without a per-layer pattern; all layers count")
    shared = _count(metadata.get(f"{arch}.attention.shared_kv_layers")) or 0
    groups: Counter[tuple[str, int, int, int, int | None]] = Counter()
    for index in range(layers - shared):
        head_count = heads[index]
        declared_kv = kv_heads[index]
        kv = head_count if declared_kv is None else declared_kv
        if not (head_count and kv):
            continue  # a layer declared with no heads (recurrent) keeps no KV
        is_swa = index in sliding
        width = hidden // head_count
        k = (swa_key if is_swa else key_len) or width
        v = (swa_value if is_swa else value_len) or width
        groups["sliding" if is_swa else "full", kv, k, v, window if is_swa else None] += 1
    attention = tuple(
        AttentionGroup(kind, count, kv, k, win, False, v)  # type: ignore[arg-type]
        for (kind, kv, k, v, win), count in groups.items()
    )
    return attention, sliding, window, shared


# --- KV cache ------------------------------------------------------------------------


def _row_bytes(kv_type: str, elements: int) -> int:
    width, block = BLOCK_KV_BYTES.get(kv_type, (2, 1))
    return elements * width // block


def kv_bytes_per_token(anatomy: ModelAnatomy, type_k: str, type_v: str) -> int:
    """Bytes the cache adds per token of context in the layers that attend to all of it:
    per layer ``n_head_kv * (k_len * b(type_k) + v_len * b(type_v))``. Sliding layers hold a
    window, not a token each; see :func:`swa_kv_bytes`."""
    return sum(
        group.layers
        * group.kv_heads
        * (
            _row_bytes(type_k, group.head_dim)
            + _row_bytes(type_v, group.value_head_dim or group.head_dim)
        )
        for group in anatomy.attention
        if group.window is None
    )


def swa_kv_bytes(anatomy: ModelAnatomy, type_k: str, type_v: str, context: int) -> int:
    """Bytes the sliding layers hold for a sequence of ``context`` tokens: each keeps at most its
    window."""
    return sum(
        group.layers
        * min(group.window, context)
        * group.kv_heads
        * (
            _row_bytes(type_k, group.head_dim)
            + _row_bytes(type_v, group.value_head_dim or group.head_dim)
        )
        for group in anatomy.attention
        if group.window is not None
    )
