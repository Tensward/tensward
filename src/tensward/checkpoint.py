"""Registering a local checkpoint: check it is a shape Tensward supports and identify it.

Supported: a text or image+text safetensors checkpoint with ``config.json``,
``generation_config.json``, ``tokenizer.json`` and ``tokenizer_config.json`` (and, for images, a
``processor_config.json`` or ``preprocessor_config.json``), unquantized BF16 or FP16, or AWQ, GPTQ,
compressed-tensors or FP8 as ``config.json`` declares, either as one ``model.safetensors`` or
as a shard index plus exactly the shards it names. Any other file, symlink or directory is
refused, because an identity that ignores a file the engine would load is worse than none.

Nothing is executed, imported or downloaded: files are hashed as bytes, and safetensors files
are inspected through their JSON header only. Files are opened without following symlinks so
a link cannot lead out of the checkpoint.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Iterator, Literal, Mapping

from .anatomy import ModelAnatomy, Tensors, build_anatomy
from .artifacts import (
    ActivationDtype,
    ArtifactEntry,
    ArtifactFiles,
    ArtifactFingerprint,
    ArtifactMetadata,
    ArtifactVariant,
    ArtifactWeights,
    ThinkingSupport,
    declared_files,
    fingerprint_files,
    open_regular,
    read_chunks,
)
from .errors import (
    CHECKPOINT_CHANGED,
    CHECKPOINT_INVENTORY_UNEXPECTED,
    CHECKPOINT_INVENTORY_UNSAFE,
    CHECKPOINT_LAYOUT_INVALID,
    CHECKPOINT_PRECISION_UNSUPPORTED,
    CHECKPOINT_UNSUPPORTED,
    PreflightError,
    not_found_message,
)
from .files import load_strict_json

MAX_DOCUMENT_BYTES = 64 * 1024 * 1024  # real tokenizer.json files reach tens of MB
MAX_HEADER_BYTES = 16 * 1024 * 1024

CONFIG = "config.json"
GENERATION_CONFIG = "generation_config.json"
TOKENIZER = "tokenizer.json"
TOKENIZER_CONFIG = "tokenizer_config.json"
SHARD_INDEX = "model.safetensors.index.json"
SINGLE_WEIGHTS = "model.safetensors"
SHARD_NAME = re.compile(r"^model-(?P<ordinal>[0-9]{5})-of-(?P<total>[0-9]{5})\.safetensors$")
REQUIRED_FILES = (CONFIG, GENERATION_CONFIG, TOKENIZER, TOKENIZER_CONFIG)
PROCESSOR_FILES = ("processor_config.json", "preprocessor_config.json")
# Older quantizers write their parameters beside config.json; engines read them from there.
QUANT_SIDE_FILES = ("quantize_config.json", "quant_config.json")
OPTIONAL_FILES = (
    "special_tokens_map.json",
    *QUANT_SIDE_FILES,
    *PROCESSOR_FILES,
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.jinja",
)
ACCEPTED_FILES = frozenset((*REQUIRED_FILES, *OPTIONAL_FILES, SHARD_INDEX, SINGLE_WEIGHTS))
IGNORED = frozenset(  # documentation and VCS entries: never read, never part of the identity
    ("README.md", "LICENSE", "LICENSE.txt", "LICENSE.md", "NOTICE", "NOTICE.txt")
    + (".gitattributes", ".git", ".cache")
    + ("quant_log.csv",)  # quantizer logs (GPTQModel), never read by an engine
)

# Quantizer metadata (GPTQModel's staging paths): never loaded, so its paths are not references.
QUANTIZER_META = ("quantization_config", "meta")
# A declared file reference in a tokenizer document must name one of these checkpoint files.
TOKENIZER_FILE_FIELDS: Mapping[str, str] = {
    "tokenizer_file": TOKENIZER,
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "special_tokens_map_file": "special_tokens_map.json",
    "added_tokens_file": "added_tokens.json",
    "chat_template_file": "chat_template.jinja",
}
# Maps in tokenizer.json whose keys are tokens, not settings (a token can be named "_file").
TOKEN_MAPS = (("model", "vocab"), ("post_processor", "special_tokens"))
PRECISIONS: dict[tuple[str, int], Literal["fp8", "int8", "int4"]] = {
    ("float", 8): "fp8",
    ("int", 8): "int8",
    ("int", 4): "int4",
}
ACTIVATION_TENSOR: dict[ActivationDtype, str] = {"bfloat16": "BF16", "float16": "F16"}
SAFETENSORS_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "I8": 1, "I32": 4, "I64": 8}
SUPPORTED_QUANT_METHODS = ("awq", "gptq", "compressed-tensors", "fp8")
ARCHITECTURE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,127}$")
SHOWN_TOKEN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# name -> (inode, size, mtime): taken before and after registering to catch a concurrent edit
Snapshot = dict[str, tuple[int, int, int]]


@dataclass(frozen=True, slots=True)
class Quantization:
    method: str
    weight_precision: Literal["fp8", "int8", "int4"]
    weight_bits: int
    group_size: int | None  # None for per-tensor or per-channel scales
    activation_scheme: Literal["static", "dynamic"] | None


def register_checkpoint(
    root: Path, *, engine_build: str, cache: Path | None = None, refresh: bool = False
) -> tuple[ArtifactEntry, ModelAnatomy]:
    """The record of the checkpoint directory ``root`` and its anatomy (derived, not part of its
    identity), or a refusal saying why not. ``cache`` and ``refresh`` are those of
    :func:`fingerprint_files`."""
    if not root.is_dir():
        raise PreflightError(
            CHECKPOINT_LAYOUT_INVALID, not_found_message("the checkpoint directory", root)
        )
    before = _scan(root)
    missing = [name for name in REQUIRED_FILES if name not in before]
    if missing:
        raise PreflightError(
            CHECKPOINT_LAYOUT_INVALID, f"the checkpoint is missing {', '.join(missing)}"
        )
    shards = _shard_names(root, before)

    _read_object(root, GENERATION_CONFIG)
    config = _read_object(root, CONFIG)
    _check_settings(config, CONFIG, before)
    side_documents = {name: _read_json(root, name) for name in QUANT_SIDE_FILES if name in before}
    quantization = _quantization(config.get("quantization_config"), side_documents)
    activation_dtype = _activation_dtype(config)
    for name in (TOKENIZER, TOKENIZER_CONFIG):
        _check_settings(_read_object(root, name), name, before)
    # Quantized checkpoints keep scales and embeddings in many dtypes; unquantized weights are
    # all in the one dtype the config declares.
    tensor_dtypes = (
        SAFETENSORS_DTYPE_BYTES.keys() if quantization else {ACTIVATION_TENSOR[activation_dtype]}
    )
    tensors: dict[str, tuple[str, tuple[int, ...], int]] = {}
    for shard in shards:
        found = _weight_tensors(root, shard, tensor_dtypes)
        if tensors.keys() & found.keys():
            raise PreflightError(
                CHECKPOINT_LAYOUT_INVALID, f"{shard} repeats a tensor another weight file has"
            )
        tensors.update(found)
    weight_bytes = sum(size for _, _, size in tensors.values())
    processors = {}
    for name in PROCESSOR_FILES:
        if name in before:
            processors[name] = _read_object(root, name)
            _check_settings(processors[name], name, before)

    files = ArtifactFiles(
        config=CONFIG,
        generation_config=GENERATION_CONFIG,
        tokenizer=TOKENIZER,
        auxiliary=tuple(
            sorted(
                name for name in (*OPTIONAL_FILES, TOKENIZER_CONFIG, SHARD_INDEX) if name in before
            )
        ),
    )
    weights = ArtifactWeights(shard_count=len(shards), shards=tuple(shards))
    fingerprint = fingerprint_files(
        root, declared_files(files, weights), cache=cache, refresh=refresh
    )
    if _scan(root) != before:
        raise PreflightError(CHECKPOINT_CHANGED, "the checkpoint changed while it was registered")
    entry = ArtifactEntry(
        root=str(root),
        files=files,
        weights=weights,
        metadata=ArtifactMetadata(
            engine_name="vllm",
            engine_build=engine_build,
            context_limit=_context_limit(config),
            weight_bytes=weight_bytes,
            architectures=_architectures(config),
            variants=(_variant(quantization, activation_dtype),),
            thinking=ThinkingSupport(supported=False),
            fingerprint=ArtifactFingerprint(value=fingerprint),
        ),
    )
    bits = quantization.weight_bits if quantization else None
    return entry, build_anatomy(config, next(iter(processors.values()), None), tensors, bits)


def _variant(
    quantization: Quantization | None, activation_dtype: ActivationDtype
) -> ArtifactVariant:
    if quantization is None:
        return ArtifactVariant(
            weight_precision="bf16" if activation_dtype == "bfloat16" else "fp16",
            activation_dtype=activation_dtype,
            quantization_method="none",
        )
    return ArtifactVariant(
        weight_precision=quantization.weight_precision,
        activation_dtype=activation_dtype,
        quantization_method=quantization.method,
        weight_bits=quantization.weight_bits,
        group_size=quantization.group_size,
        activation_scheme=quantization.activation_scheme,
    )


# --- the directory -------------------------------------------------------------------


def _scan(root: Path) -> Snapshot:
    """The checkpoint's top-level runtime files; refuses a symlink, a directory or any file
    that is not part of the supported shape."""
    snapshot: Snapshot = {}
    with os.scandir(root) as entries:
        for entry in entries:
            name = entry.name
            if name in IGNORED:
                continue
            is_file = entry.is_file(follow_symlinks=False)
            if entry.is_symlink() or not (is_file or entry.is_dir(follow_symlinks=False)):
                raise PreflightError(
                    CHECKPOINT_INVENTORY_UNSAFE, f"{name} is a symlink or a special file"
                )
            if not is_file or (name not in ACCEPTED_FILES and not SHARD_NAME.match(name)):
                raise PreflightError(
                    CHECKPOINT_INVENTORY_UNEXPECTED, f"{name} is not part of a supported checkpoint"
                )
            info = entry.stat(follow_symlinks=False)
            snapshot[name] = (info.st_ino, info.st_size, info.st_mtime_ns)
    return snapshot


def _shard_names(root: Path, present: Snapshot) -> list[str]:
    """The weight files: ``model.safetensors``, or the shards the index names (all of them)."""
    weight_files = sorted(n for n in present if n == SINGLE_WEIGHTS or SHARD_NAME.match(n))
    if SHARD_INDEX not in present:
        if weight_files != [SINGLE_WEIGHTS]:
            raise PreflightError(
                CHECKPOINT_LAYOUT_INVALID,
                f"the checkpoint has neither {SINGLE_WEIGHTS} nor a shard index",
            )
        return weight_files
    weight_map = _read_object(root, SHARD_INDEX).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise PreflightError(CHECKPOINT_LAYOUT_INVALID, "the shard index has no weight_map")
    shards = sorted({str(shard) for shard in weight_map.values()})
    matches = [SHARD_NAME.match(shard) for shard in shards]
    ordinals = sorted(int(m["ordinal"]) for m in matches if m)
    consistent = all(m and int(m["total"]) == len(shards) for m in matches)
    if shards != weight_files or not consistent or ordinals != list(range(1, len(shards) + 1)):
        raise PreflightError(
            CHECKPOINT_LAYOUT_INVALID,
            "the shard index does not match the weight files present (names, count or order)",
        )
    return shards


# --- documents -------------------------------------------------------------------------


def _read_json(root: Path, name: str) -> Any:
    with open_regular(root / name) as (descriptor, size):
        if size > MAX_DOCUMENT_BYTES:
            raise PreflightError(CHECKPOINT_LAYOUT_INVALID, f"{name} is too large")
        data = b"".join(read_chunks(descriptor, size, name))
    return load_strict_json(data, name, CHECKPOINT_LAYOUT_INVALID)


def _read_object(root: Path, name: str) -> dict[str, Any]:
    document = _read_json(root, name)
    if not isinstance(document, dict):
        raise PreflightError(CHECKPOINT_LAYOUT_INVALID, f"{name} is not a JSON object")
    return document


def _shown(value: Any) -> str:
    """A declared value that is safe to print: a short plain token, else a marker."""
    text = str(value)
    return text if SHOWN_TOKEN.match(text) else "<invalid>"


def _settings(
    document: Any, token_maps: tuple[tuple[str, str], ...]
) -> Iterator[tuple[tuple[str, ...], str, Any]]:
    """Every ``(path, key, value)`` in a JSON document, except the keys of ``token_maps``.
    Iterative, so a deeply nested document cannot exhaust the interpreter."""
    stack: list[tuple[tuple[str, ...], Any]] = [((), document)]
    while stack:
        path, node = stack.pop()
        if isinstance(node, list):
            stack.extend((path, item) for item in node)
        elif isinstance(node, dict):
            for key, value in node.items():
                if path not in token_maps:
                    yield path, key, value
                stack.append((path + (key,), value))


def _check_settings(document: dict[str, Any], name: str, present: Snapshot) -> None:
    """Refuse custom code and file references to anywhere but the checkpoint's own tokenizer
    files, wherever in the document they are declared."""
    is_model = name == CONFIG
    token_maps = TOKEN_MAPS if name == TOKENIZER else ()
    for path, key, value in _settings(document, token_maps):
        where = f"{name} setting {_shown(key)}"
        if key == "auto_map" and value is not None:
            raise PreflightError(CHECKPOINT_UNSUPPORTED, f"{where} declares custom code")
        if key == "trust_remote_code" and value is not None and value is not False:
            raise PreflightError(CHECKPOINT_UNSUPPORTED, f"{where} asks to run remote code")
        if is_model and key == "quantization_config" and path and value is not None:
            raise PreflightError(CHECKPOINT_UNSUPPORTED, f"{where} is a nested quantization")
        if key == "_name_or_path":  # descriptive, never loaded
            if value is not None and not isinstance(value, str):
                raise PreflightError(CHECKPOINT_LAYOUT_INVALID, f"{where} is not a string")
        elif key.endswith(("_file", "_path")) and value is not None and path[:2] != QUANTIZER_META:
            if is_model or TOKENIZER_FILE_FIELDS.get(key) != value or value not in present:
                raise PreflightError(
                    CHECKPOINT_UNSUPPORTED, f"{where} references a file outside the checkpoint"
                )


def _context_limit(config: dict[str, Any]) -> int:
    text = config.get("text_config")
    limit = config.get("max_position_embeddings")
    if limit is None and isinstance(text, dict):  # multimodal configs nest the text model's
        limit = text.get("max_position_embeddings")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise PreflightError(
            CHECKPOINT_LAYOUT_INVALID, "config.json declares no positive max_position_embeddings"
        )
    return limit


def _architectures(config: dict[str, Any]) -> tuple[str, ...]:
    names = config.get("architectures")
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(n, str) and ARCHITECTURE_NAME.match(n) for n in names)
        or len(set(names)) != len(names)
    ):
        raise PreflightError(
            CHECKPOINT_LAYOUT_INVALID, "config.json declares no valid, distinct architectures"
        )
    return tuple(names)


def _activation_dtype(config: dict[str, Any]) -> ActivationDtype:
    allowed = tuple(ACTIVATION_TENSOR)
    declared = [config[key] for key in ("dtype", "torch_dtype") if config.get(key) is not None]
    if not declared or any(value != declared[0] or value not in allowed for value in declared):
        raise PreflightError(
            CHECKPOINT_PRECISION_UNSUPPORTED,
            f"config.json must declare the weight dtype {' or '.join(allowed)}",
        )
    dtype: ActivationDtype = declared[0]
    return dtype


def _quantization(declared: Any, side_documents: Mapping[str, Any]) -> Quantization | None:
    """Read ``quantization_config`` (older quantizers' side files fill in what it omits)."""
    if declared is None:
        if side_documents:
            raise PreflightError(
                CHECKPOINT_UNSUPPORTED, "quantization is declared beside config.json, not in it"
            )
        return None
    method = declared.get("quant_method") if isinstance(declared, dict) else None
    if not isinstance(method, str):
        raise PreflightError(CHECKPOINT_LAYOUT_INVALID, "quantization_config has no quant_method")
    if method not in SUPPORTED_QUANT_METHODS:
        raise PreflightError(
            CHECKPOINT_UNSUPPORTED,
            f"quantization method {_shown(method)!r} "
            f"(supported: {', '.join(SUPPORTED_QUANT_METHODS)})",
        )
    merged: dict[str, Any] = {}
    for document in side_documents.values():
        if isinstance(document, dict):
            merged.update(document)
    merged.update(declared)
    bits, weight_type, group_size, scheme = _quantization_details(method, merged)
    if group_size == -1:  # per-channel
        group_size = None
    precision = PRECISIONS.get((weight_type, bits)) if isinstance(bits, int) else None
    if precision is None or (method == "awq" and bits != 4):  # engines only run 4-bit AWQ
        raise PreflightError(
            CHECKPOINT_PRECISION_UNSUPPORTED,
            f"{method} weights of type {_shown(weight_type)} with {_shown(bits)} bits",
        )
    if not (group_size is None or isinstance(group_size, int) and group_size > 0) or (
        scheme not in (None, "static", "dynamic")
    ):
        raise PreflightError(
            CHECKPOINT_LAYOUT_INVALID, f"quantization_config for {method} is invalid"
        )
    return Quantization(method, precision, bits, group_size, scheme)


def _quantization_details(method: str, merged: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    """``(bits, weight type, group size, activation scheme)`` as the method spells them."""
    if method in ("awq", "gptq"):
        bits = merged.get("bits", merged.get("w_bit"))
        return bits, "int", merged.get("group_size", merged.get("q_group_size")), None
    if method == "fp8":
        return 8, "float", None, merged.get("activation_scheme", "dynamic")
    groups = merged.get("config_groups")
    group = next(iter(groups.values())) if isinstance(groups, dict) and len(groups) == 1 else {}
    weights = group.get("weights") if isinstance(group, dict) else None
    activations = group.get("input_activations") if isinstance(group, dict) else None
    if not isinstance(weights, dict) or not isinstance(activations, dict | None):
        raise PreflightError(
            CHECKPOINT_UNSUPPORTED,
            "compressed-tensors needs exactly one config group that declares weights",
        )
    scheme = None if activations is None else "dynamic" if activations.get("dynamic") else "static"
    return weights.get("num_bits"), weights.get("type"), weights.get("group_size"), scheme


# --- weights ---------------------------------------------------------------------------


def _weight_tensors(root: Path, name: str, allowed: Collection[str]) -> Tensors:
    """Validate one safetensors header and return each tensor's (dtype, shape, bytes).

    The header is 8 bytes of length plus JSON. The tensors must tile the payload without gap
    or overlap and the file must be exactly as long as the header says; no tensor is read.
    """
    with open_regular(root / name) as (descriptor, size):
        length = int.from_bytes(b"".join(read_chunks(descriptor, min(size, 8), name)), "little")
        if size < 8 or not 1 <= length <= MAX_HEADER_BYTES or 8 + length > size:
            raise PreflightError(
                CHECKPOINT_LAYOUT_INVALID, f"{name} has no valid safetensors header"
            )
        header = b"".join(read_chunks(descriptor, length, name))
    tensors = load_strict_json(header, f"the header of {name}", CHECKPOINT_LAYOUT_INVALID)
    if not isinstance(tensors, dict) or not isinstance(tensors.pop("__metadata__", {}), dict):
        raise PreflightError(CHECKPOINT_LAYOUT_INVALID, f"the header of {name} is malformed")
    ranges = {key: _tensor_range(entry, allowed, name) for key, entry in tensors.items()}
    end = 0
    for start, stop in sorted(ranges.values()):
        if start != end:
            raise PreflightError(
                CHECKPOINT_LAYOUT_INVALID, f"the tensors of {name} leave a gap or overlap"
            )
        end = stop
    if end == 0 or size != 8 + length + end:
        raise PreflightError(
            CHECKPOINT_LAYOUT_INVALID, f"{name} is not the size its header declares"
        )
    return {
        key: (tensors[key]["dtype"], tuple(tensors[key]["shape"]), stop - start)
        for key, (start, stop) in ranges.items()
    }


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _tensor_range(entry: Any, allowed: Collection[str], name: str) -> tuple[int, int]:
    """The ``(start, end)`` data offsets of one header entry, after checking it is well formed."""
    malformed = PreflightError(CHECKPOINT_LAYOUT_INVALID, f"a tensor entry in {name} is malformed")
    if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
        raise malformed
    dtype, shape, offsets = entry["dtype"], entry["shape"], entry["data_offsets"]
    if not isinstance(dtype, str):
        raise malformed
    if dtype not in allowed:
        raise PreflightError(
            CHECKPOINT_PRECISION_UNSUPPORTED,
            f"{name} has a tensor of unsupported dtype {_shown(dtype)}",
        )
    if not (isinstance(shape, list) and shape and all(map(_is_count, shape))):
        raise malformed
    if not (isinstance(offsets, list) and len(offsets) == 2 and all(map(_is_count, offsets))):
        raise malformed
    start, stop = offsets
    if stop - start != SAFETENSORS_DTYPE_BYTES[dtype] * math.prod(shape):
        raise malformed
    return start, stop
