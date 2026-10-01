"""A registered project: one private directory holding ``project.json``.

The record binds a checkpoint directory, a serving configuration and a workload JSONL to the
identities of their bytes, and stores where they are, never their contents. Loading a project
derives every identity again from the sources and compares: a changed input, or a record
edited to claim something else, is refused. That is self-consistency, not authenticity; a
caller holding a trusted ``snapshot_id`` can pass it as ``expected_snapshot_id`` (used by
extensions that hand a project between processes).

The directory is ``0700``, the record ``0600``, and a record is published atomically and never
replaces an existing one, so a retry cannot silently rebind a project to different inputs.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from .artifacts import AbsolutePath, ArtifactEntry
from .checkpoint import register_checkpoint
from .contracts import DigestHex, Identifier, StrictModel
from .engines import Engine
from .engines.protocol import ParsedSetup, Settings
from .errors import (
    PROJECT_BUSY,
    PROJECT_CONFIG_UNSUPPORTED,
    PROJECT_INPUTS_CHANGED,
    PROJECT_INPUTS_INVALID,
    PROJECT_LAYOUT_INVALID,
    PROJECT_NOT_REGISTERED,
    PROJECT_PUBLICATION_FAILED,
    PROJECT_RECORD_INVALID,
    PROJECT_SNAPSHOT_MISMATCH,
    PROJECT_STATE_UNSAFE,
    PreflightError,
    not_found_message,
)
from .files import parse_document
from .inputs import (
    MAX_INPUT_BYTES,
    PromptEntry,
    ServingConfig,
    load_prompt_entries,
    load_serving_config,
    workload_digest,
)

PROJECT_DOCUMENT = "project.json"
LOCK_FILE = ".registration.lock"
WEIGHTS_CACHE = ".weights-verified.json"  # private state: when the weights were last hashed
SNAPSHOT_DIGEST_DOMAIN = b"tensward:registration-snapshot:1\x00"


class CurrentSetup(StrictModel):
    """What the customer runs today: the baseline every improvement is measured against.

    ``settings`` are engine-neutral :class:`Settings` as JSON. A ``command`` setup was imported
    from the customer's own command line (secrets blanked in ``text``); a ``config`` setup is
    the serving fields of the configuration document; a ``defaults`` setup declares none, so
    the engine's out-of-the-box behaviour is the baseline.
    """

    source: Literal["command", "config", "defaults"]
    settings: dict[str, Any]
    text: str | None = None
    image: str | None = None
    gpus: tuple[str, ...] = ()  # the devices the command ran on, if it selected any
    notes: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return {
            "command": "imported from --current",
            "config": "declared in config",
            "defaults": "engine defaults (no current setup provided)",
        }[self.source]

    @property
    def engine_settings(self) -> Settings:
        return Settings.from_json(self.settings)


_LEGACY_POLICY_FIELDS = frozenset(
    {"input_retention", "retain_responses", "resource_policy_state", "execution_authorization"}
)


class ProjectRecord(StrictModel):
    """The committed ``project.json``.

    ``snapshot_id`` binds the artifact fingerprint, the two digests and the current setup, so
    it is the same on any machine and wherever the inputs happen to live.
    """

    schema_version: Literal["1"] = "1"
    project_id: Identifier
    snapshot_id: DigestHex
    artifact: ArtifactEntry
    config_path: AbsolutePath
    prompts_path: AbsolutePath
    config_digest: DigestHex
    workload_digest: DigestHex
    current_setup: CurrentSetup


@dataclass(frozen=True, slots=True)
class ResolvedProject:
    """A verified project: its record, its serving configuration and its prompts."""

    record: ProjectRecord
    settings: ServingConfig
    prompts: tuple[PromptEntry, ...]

    @property
    def artifact(self) -> ArtifactEntry:
        return self.record.artifact


@dataclass(frozen=True, slots=True)
class _Derived:
    """A project freshly derived from its sources: what a record would say right now."""

    record: ProjectRecord
    settings: ServingConfig
    prompts: tuple[PromptEntry, ...]


# --- the operations --------------------------------------------------------------------


def init_project(
    project: Path,
    *,
    model: Path,
    config: Path,
    prompts: Path,
    engine: Engine,
    current: str | None = None,
) -> ProjectRecord:
    """Register a project, or return the identical existing registration.

    ``current`` is the customer's own engine command line; without it the baseline is the
    serving fields of ``config``. An existing record is verified, never replaced, and a
    request that differs from it is refused.
    """
    project = project.expanduser()
    sources = (
        _resolve(model, "the model directory", directory=True),
        _resolve(config, "the serving configuration", directory=False),
        _resolve(prompts, "the workload document", directory=False),
    )
    parsed = _parse_current(engine, current)
    _check_layout(project, *sources)
    _prepare_directory(project)
    with _locked(project):
        existing = _read_record(project)
        derived = _derive(*sources, parsed, cache=project / WEIGHTS_CACHE)
        if existing is None:
            _publish(project, derived.record)
            return derived.record
        _require_self_consistent(existing)
        bound = (existing.artifact.path, Path(existing.config_path), Path(existing.prompts_path))
        if bound != sources:
            raise PreflightError(
                PROJECT_INPUTS_CHANGED, "the project is registered with different inputs"
            )
        _require_same_identity(existing, derived)
        if existing.current_setup != derived.record.current_setup:
            raise PreflightError(
                PROJECT_INPUTS_CHANGED, "the project is registered with a different current setup"
            )
        return existing


def load_project(
    project: Path, *, expected_snapshot_id: str | None = None, verify_weights: bool = False
) -> ResolvedProject:
    """Verify a registered project against its sources and return it.

    Weights are hashed only when a file's (inode, size, mtime_ns) differs from when they were
    last hashed; ``verify_weights`` hashes them anyway. A file edited in place that keeps its
    size and mtime is therefore noticed only with ``verify_weights``.
    """
    project = project.expanduser()
    record = _read_record(project)
    if record is None:
        raise PreflightError(PROJECT_NOT_REGISTERED, "no project is registered at this location")
    _require_self_consistent(record)
    derived = _derive(
        record.artifact.path,
        Path(record.config_path),
        Path(record.prompts_path),
        cache=project / WEIGHTS_CACHE,
        refresh=verify_weights,
    )
    _require_same_identity(record, derived)
    if expected_snapshot_id is not None and expected_snapshot_id != record.snapshot_id:
        raise PreflightError(
            PROJECT_SNAPSHOT_MISMATCH, "the project snapshot is not the expected one"
        )
    return ResolvedProject(record=record, settings=derived.settings, prompts=derived.prompts)


def registered_setup(project: Path) -> CurrentSetup | None:
    """The project's current setup (its ``--current`` command: image, GPUs), if it is registered."""
    record = _read_record(project.expanduser())
    return None if record is None else record.current_setup


def project_summary(record: ProjectRecord) -> dict[str, object]:
    """The record as printed by ``init`` and ``inspect``: identities and the current setup, never
    an input's path or a prompt."""
    current = record.current_setup
    return {
        "current_setup": {
            "source": current.label,
            "command": current.text,
            "image": current.image,
            "gpus": list(current.gpus),
            "settings": current.settings,
            "dropped_or_ignored": list(current.notes),
        },
        "schema_version": record.schema_version,
        "project_id": record.project_id,
        "snapshot_id": record.snapshot_id,
        "artifact_fingerprint": record.artifact.metadata.fingerprint.value,
        "weights": record.artifact.metadata.variants[0].model_dump(mode="json"),
        "config_digest": record.config_digest,
        "workload_digest": record.workload_digest,
        "registration_state": "registered",
    }


# --- deriving and comparing identities -----------------------------------------------


def _derive(
    model_root: Path,
    config_path: Path,
    prompts_path: Path,
    parsed: ParsedSetup | None = None,
    *,
    cache: Path | None = None,
    refresh: bool = True,
) -> _Derived:
    """Read the sources and work out the project they describe right now."""
    prompts = load_prompt_entries(prompts_path)
    settings, config_digest = load_serving_config(config_path, prompts)
    entry = register_checkpoint(
        model_root, engine_build=settings.engine_build, cache=cache, refresh=refresh
    )
    _check_case_against_checkpoint(settings, entry)
    if settings.workload.api == "chat":
        _require_chat_template(model_root)
    digest = workload_digest(prompts)
    current = _current_setup(parsed, settings, entry)
    record = ProjectRecord(
        project_id=uuid.uuid4().hex,
        snapshot_id=_snapshot_id(entry, config_digest, digest, current),
        artifact=entry,
        config_path=str(config_path),
        prompts_path=str(prompts_path),
        config_digest=config_digest,
        workload_digest=digest,
        current_setup=current,
    )
    return _Derived(record, settings, prompts)


def _snapshot_id(
    entry: ArtifactEntry, config_digest: str, workload_digest: str, current: CurrentSetup
) -> str:
    setup = current.model_dump(mode="json")
    if not current.gpus:  # recorded only when present, so identities made before it stay valid
        del setup["gpus"]
    payload = {
        "schema_version": "1",
        "artifact_fingerprint": entry.metadata.fingerprint.value,
        "config_digest": config_digest,
        "workload_digest": workload_digest,
        "current_setup": setup,
    }
    canonical = json.dumps(
        payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(SNAPSHOT_DIGEST_DOMAIN + canonical.encode()).hexdigest()


def _require_self_consistent(record: ProjectRecord) -> None:
    """Refuse a record whose snapshot id does not follow from its own fields."""
    expected = _snapshot_id(
        record.artifact, record.config_digest, record.workload_digest, record.current_setup
    )
    if expected != record.snapshot_id:
        raise PreflightError(
            PROJECT_RECORD_INVALID, "the project record is not consistent with its own identity"
        )


def _require_same_identity(stored: ProjectRecord, fresh: _Derived) -> None:
    """Refuse when the sources no longer produce the stored identity."""
    now = fresh.record
    if stored.artifact.metadata.fingerprint != now.artifact.metadata.fingerprint:
        raise PreflightError(PROJECT_INPUTS_CHANGED, "the checkpoint changed since registration")
    if stored.artifact != now.artifact:  # same bytes, different claims: the record was edited
        raise PreflightError(
            PROJECT_RECORD_INVALID, "the recorded checkpoint facts do not match its files"
        )
    if stored.config_digest != now.config_digest:
        raise PreflightError(
            PROJECT_INPUTS_CHANGED, "the serving configuration changed since registration"
        )
    if stored.workload_digest != now.workload_digest:
        raise PreflightError(
            PROJECT_INPUTS_CHANGED, "the workload document changed since registration"
        )


def _check_case_against_checkpoint(settings: ServingConfig, entry: ArtifactEntry) -> None:
    """Refuse serving settings the checkpoint does not provide."""
    case, metadata = settings.case, entry.metadata
    provided = {(v.weight_precision, v.activation_dtype) for v in metadata.variants}
    if (case.weight_precision, case.activation_dtype) not in provided:
        raise PreflightError(
            PROJECT_CONFIG_UNSUPPORTED,
            "the declared weight precision and activation dtype are not what the checkpoint "
            "provides",
        )
    if case.max_model_len is not None and case.max_model_len > metadata.context_limit:
        raise PreflightError(
            PROJECT_CONFIG_UNSUPPORTED, "max_model_len exceeds the checkpoint's context limit"
        )


def _require_chat_template(model_root: Path) -> None:
    """Refuse a chat workload for a checkpoint that ships no chat template."""
    try:
        tokenizer_config = json.loads((model_root / "tokenizer_config.json").read_text("utf-8"))
        has_template = bool(tokenizer_config.get("chat_template"))
    except (OSError, ValueError, AttributeError):
        has_template = False
    if not has_template and not (model_root / "chat_template.jinja").is_file():
        raise PreflightError(
            PROJECT_INPUTS_INVALID,
            "the workload uses the chat api but the checkpoint has no chat template "
            "(chat_template in tokenizer_config.json or a chat_template.jinja file); "
            "use an instruct checkpoint or the completions api",
        )


# --- the baseline ------------------------------------------------------------------------


def _parse_current(engine: Engine, text: str | None) -> ParsedSetup | None:
    if text is None:
        return None
    try:
        return engine.parse_setup(text)
    except ValueError as error:
        raise PreflightError(PROJECT_INPUTS_INVALID, f"--current: {error}") from None


def _current_setup(
    parsed: ParsedSetup | None, settings: ServingConfig, entry: ArtifactEntry
) -> CurrentSetup:
    """The baseline: the customer's command if given, else the serving fields the
    configuration declares, else the engine's own defaults."""
    case = settings.case
    declared: dict[str, Any] = {
        "max_concurrent_requests": case.max_num_seqs,
        "max_context_len": case.max_model_len,
        "kv_memory_fraction": case.gpu_memory_utilization,
        "prefill_batch_tokens": case.max_num_batched_tokens,
        "kv_cache_dtype": case.kv_cache_dtype,
        "prefix_caching": case.prefix_cache,
    }
    declared = {name: value for name, value in declared.items() if value is not None}
    if parsed is None:
        if not declared:
            # weight_precision and activation_dtype describe the checkpoint, and tool_calling
            # the workload: with no serving field declared the engine's defaults run.
            return CurrentSetup(
                source="defaults", settings=asdict(Settings(tool_calling=case.tool_calling))
            )
        engine_settings = Settings(
            dtype=case.activation_dtype, tool_calling=case.tool_calling, **declared
        )
        return CurrentSetup(source="config", settings=asdict(engine_settings))
    method = entry.metadata.variants[0].quantization_method
    asked = parsed.settings.quantization
    if asked and re.split(r"[_-]", asked)[0] != re.split(r"[_-]", method)[0]:
        raise PreflightError(
            PROJECT_CONFIG_UNSUPPORTED,
            f"--current quantizes with {asked!r}, the registered checkpoint is {method!r}",
        )
    notes = parsed.notes
    if conflicts := [
        f"{name} (configuration {value}; your command "
        + ("does not set it" if (own := getattr(parsed.settings, name)) is None else f"sets {own}")
        + ")"
        for name, value in declared.items()
        if getattr(parsed.settings, name) != value
    ]:
        notes += (
            "your --current command wins; these serving fields of the configuration are "
            f"ignored: {', '.join(conflicts)}",
        )
    served = re.split(r"[/\\]", parsed.model or "")[-1]
    if served and served.lower() != entry.path.name.lower():
        notes += (
            f"your command serves {parsed.model}; Tensward measures the registered checkpoint "
            f"{entry.path.name}",
        )
    return CurrentSetup(
        source="command",
        settings=asdict(parsed.settings),
        text=parsed.text,
        image=parsed.image,
        gpus=parsed.gpus,
        notes=notes,
    )


# --- the filesystem --------------------------------------------------------------------


def _resolve(path: Path, what: str, *, directory: bool) -> Path:
    """The canonical location of a declared input. A file may not be a symlink."""
    try:
        if not directory and path.is_symlink():
            raise OSError
        resolved = path.resolve(strict=True)
    except FileNotFoundError:
        raise PreflightError(PROJECT_INPUTS_INVALID, not_found_message(what, path)) from None
    except (OSError, ValueError, RuntimeError):
        raise PreflightError(PROJECT_INPUTS_INVALID, f"{what} is missing or unreadable") from None
    if not (resolved.is_dir() if directory else resolved.is_file()):
        kind = "a directory" if directory else "a regular file"
        raise PreflightError(PROJECT_INPUTS_INVALID, f"{what} is not {kind}")
    return resolved


def _check_layout(project: Path, model_root: Path, config_path: Path, prompts_path: Path) -> None:
    """Refuse a project directory inside the checkpoint (it would change the checkpoint's own
    file list) or a declared input that is one of the project's own files."""
    resolved = Path(os.path.realpath(project))
    if resolved.is_relative_to(model_root):
        raise PreflightError(
            PROJECT_LAYOUT_INVALID, "the project directory must not be inside the checkpoint"
        )
    if {config_path, prompts_path} & {resolved / PROJECT_DOCUMENT, resolved / LOCK_FILE}:
        raise PreflightError(
            PROJECT_LAYOUT_INVALID, "a declared input is one of the project's own files"
        )


def _is_private(path: Path, kind: Callable[[int], bool]) -> bool:
    """Whether ``path`` (not followed if it is a symlink) is of ``kind``, owned by this user and
    closed to group and others. Root may also use what another user owns: it can read
    everything anyway, and the check exists to keep other non-root users out."""
    info = path.lstat()
    mine = os.geteuid() in (0, info.st_uid)
    return kind(info.st_mode) and mine and not info.st_mode & 0o077


def hand_back_to_owner(project: Path) -> None:
    """Give the files root created in a project that another user owns to that user, so the
    user can keep using the project after a ``sudo`` run. Does nothing for anyone else."""
    try:
        owner = project.lstat()
    except OSError:
        return
    if os.geteuid() != 0 or owner.st_uid == 0:
        return
    # fwalk works on directory descriptors and never follows a link, and each entry is changed
    # relative to its parent's descriptor, so another user swapping an entry for a symlink
    # mid-walk cannot make root change a file outside the project.
    for _, directories, files, parent in os.fwalk(project, follow_symlinks=False):
        for name in (*directories, *files):
            if os.stat(name, dir_fd=parent, follow_symlinks=False).st_uid == 0:
                os.chown(name, owner.st_uid, owner.st_gid, dir_fd=parent, follow_symlinks=False)


def _prepare_directory(project: Path) -> None:
    """Create the project directory if needed. An existing one is never chmodded, only checked."""
    try:
        project.mkdir(mode=0o700, parents=True, exist_ok=True)
        private = _is_private(project, stat.S_ISDIR)
    except OSError:
        raise PreflightError(PROJECT_STATE_UNSAFE, "the project directory cannot be used") from None
    if not private:
        raise PreflightError(
            PROJECT_STATE_UNSAFE,
            "the project directory must be a real directory that only you can access (0700)",
        )


@contextmanager
def _locked(project: Path) -> Iterator[None]:
    """Hold the project's lock, so two registrations cannot interleave."""
    descriptor = os.open(project / LOCK_FILE, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PreflightError(
                PROJECT_BUSY, "another registration holds this project's lock"
            ) from None
        yield
    finally:
        os.close(descriptor)


def _read_record(project: Path) -> ProjectRecord | None:
    """The committed record, or None when the project has none."""
    document = project / PROJECT_DOCUMENT
    try:
        private = _is_private(project, stat.S_ISDIR) and _is_private(document, stat.S_ISREG)
    except FileNotFoundError:
        return None
    if not private:
        raise PreflightError(
            PROJECT_STATE_UNSAFE,
            "the project directory (0700) or document (0600) is not private to you",
        )
    if document.stat().st_size > MAX_INPUT_BYTES:
        raise PreflightError(PROJECT_RECORD_INVALID, "the project document is too large")
    data = document.read_bytes()
    if any(f'"{name}"'.encode() in data for name in _LEGACY_POLICY_FIELDS):
        # Earlier versions wrote four fixed policy fields that no identity ever covered.
        stripped = json.loads(data)
        data = json.dumps(
            {k: v for k, v in stripped.items() if k not in _LEGACY_POLICY_FIELDS}
        ).encode()
    return parse_document(ProjectRecord, data, "the project document", PROJECT_RECORD_INVALID)


def _publish(project: Path, record: ProjectRecord) -> None:
    """Write the record to a private temporary file and link it into place. Linking fails
    instead of replacing a record that appeared meanwhile."""
    text = json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
    temporary = project / f".incomplete-project-{uuid.uuid4().hex}"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(text + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary, project / PROJECT_DOCUMENT)
        directory = os.open(project, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)  # so the new name survives a crash
        finally:
            os.close(directory)
    except OSError:
        raise PreflightError(
            PROJECT_PUBLICATION_FAILED, "the project record could not be published"
        ) from None
    finally:
        temporary.unlink(missing_ok=True)
