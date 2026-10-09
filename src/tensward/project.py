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
from dataclasses import MISSING, asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Sequence

from .anatomy import ModelAnatomy
from .artifacts import AbsolutePath, ArtifactEntry
from .contracts import DigestHex, Identifier, StrictModel
from .engines import ENGINES, Engine
from .environment import EngineChoice, engine_choice, resolve_engine, supported_launchers
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
    AnalyseFailure,
    PreflightError,
    not_found_message,
)
from .files import parse_document
from .fit import CHARS_PER_TOKEN, Fit, estimate_fit
from .formats import detect_format
from .images import PromptImages, load_prompt_images
from .inputs import (
    MAX_INPUT_BYTES,
    PromptEntry,
    ServingConfig,
    load_prompt_entries,
    load_serving_config,
    refuse_named_tool_under_config,
    workload_digest,
)
from .platforms import detect_devices, detect_platform
from .playbook import WorkloadFacts
from .settings import ParsedSetup, Settings
from .text import canonical_json
from .workload import ArrivalSpec, WorkloadSpec

PROJECT_DOCUMENT = "project.json"
LOCK_FILE = ".registration.lock"
WEIGHTS_CACHE = ".weights-verified.json"  # private state: when the weights were last hashed
IMAGES_CACHE = "images-cache.json"  # likewise for the workload's images
LEGACY_ENGINE = "vllm"  # the engine of every project registered before engines were recorded
SNAPSHOT_DIGEST_DOMAIN = b"tensward:registration-snapshot:1\x00"
# The settings every snapshot id has carried since schema 1. A later field joins an id only
# when it is set away from its default, so the ids of projects that leave it alone stay valid.
SNAPSHOT_V1_SETTINGS = frozenset(
    {
        "dtype",
        "max_concurrent_requests",
        "max_context_len",
        "kv_memory_fraction",
        "prefill_batch_tokens",
        "kv_cache_dtype",
        "prefix_caching",
        "quantization",
        "cuda_graphs",
        "tool_calling",
        "tool_parser",
        "async_scheduling",
        "api_server_count",
        "extra_args",
        "extra_env",
    }
)
_SETTINGS_DEFAULTS: dict[str, Any] = {
    f.name: f.default_factory() if f.default_factory is not MISSING else f.default
    for f in fields(Settings)
}


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
    engine: str = LEGACY_ENGINE

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


class ImageRecord(StrictModel):
    """One workload image as registered: where it is below the prompts file, and its bytes."""

    url: str
    sha256: DigestHex
    size: int


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
    images: tuple[ImageRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class ResolvedProject:
    """A verified project: its record, its serving configuration, its prompts, their images
    and its checkpoint's anatomy."""

    record: ProjectRecord
    config: ServingConfig
    prompts: tuple[PromptEntry, ...]
    images: PromptImages
    anatomy: ModelAnatomy

    @property
    def artifact(self) -> ArtifactEntry:
        return self.record.artifact


# --- the operations --------------------------------------------------------------------


def init_project(
    project: Path,
    *,
    model: Path,
    config: Path,
    prompts: Path,
    engine: str | None = None,
    current: str | None = None,
) -> ResolvedProject:
    """Register a project, or return the identical existing registration, resolved.

    ``current`` is the customer's own engine command line; without it the baseline is the
    serving fields of ``config``. ``engine`` is ``--engine``; the engine recorded is resolved by
    :func:`resolve_engine`. An existing record is verified, never replaced, and a
    request that differs from it is refused.
    """
    project = project.expanduser()
    sources = (
        _resolve(model, "the model directory", directory=True),
        _resolve(config, "the serving configuration", directory=False),
        _resolve(prompts, "the workload document", directory=False),
    )
    found = detect_platform()
    choice = resolve_engine(
        requested=engine,
        current=current,
        fmt=detect_format(sources[0]),
        platform=found[0] if found else None,
    )
    parsed = _parse_current(choice, current)
    _check_layout(project, *sources)
    _prepare_directory(project)
    with _locked(project):
        existing = _read_record(project)
        derived = _derive(
            *sources,
            parsed,
            project=project,
            engine=choice.engine.name,
            trust_remote_code=(
                parsed is not None and choice.engine.runs_remote_code(parsed.settings)
            ),
        )
        if parsed is not None:
            _require_same_quantization(choice.engine, parsed, derived.record.artifact)
        if existing is None:
            refuse_named_tool_under_config(derived.config)
            _publish(project, derived.record)
            return derived
        _require_self_consistent(existing)
        bound = (existing.artifact.path, Path(existing.config_path), Path(existing.prompts_path))
        if bound != sources:
            raise PreflightError(
                PROJECT_INPUTS_CHANGED, "the project is registered with different inputs"
            )
        _require_same_identity(existing, derived)
        unnoted = {"notes": ()}  # advisory text, derived from the command and the configuration
        if existing.current_setup.model_copy(update=unnoted) != (
            derived.record.current_setup.model_copy(update=unnoted)
        ):
            raise PreflightError(
                PROJECT_INPUTS_CHANGED, "the project is registered with a different current setup"
            )
        return replace(derived, record=existing)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProjectFacts(WorkloadFacts):
    """The workload's facts, with the positions of the prompts that offer tools."""

    tool_prompts: frozenset[int] = frozenset()


def workload_facts(project: ResolvedProject) -> ProjectFacts:
    """What the registered workload says about itself."""
    metadata = project.artifact.metadata
    arrival = project.config.workload.arrival
    load = f"{arrival.concurrency} concurrent clients"
    if arrival.kind != "closed_loop":
        load = f"{arrival.rate_rps:g} requests/s" + (
            f", at most {arrival.max_inflight} in flight" if arrival.kind == "capped" else ""
        )
    declared = project.config.workload.structured_output is not None
    return ProjectFacts(
        tool_prompts=frozenset(i for i, entry in enumerate(project.prompts) if entry.tools),
        prompts=tuple(entry.text for entry in project.prompts),
        output_tokens=project.config.workload.output_tokens,
        context_limit=metadata.context_limit,
        quantization=metadata.variants[0].label,
        offers_tools=any(entry.tools for entry in project.prompts),
        media_encoders="image" in project.anatomy.modalities,
        sends_images=any(entry.image_urls for entry in project.prompts),
        load=load,
        model_type=project.anatomy.model_type,
        structured=declared or any(entry.constrained for entry in project.prompts),
        schema_with_tools=any(
            entry.tools and (declared or entry.constrained) for entry in project.prompts
        ),
        max_inflight=arrival.concurrency or arrival.max_inflight,
        encoder_gib=(project.anatomy.components.vision / 2**30)
        if project.anatomy.components
        else 0.0,
    )


def json_answer_ids(project: ResolvedProject) -> frozenset[str]:
    """The prompts whose answers must parse as JSON: all of them under the configuration's
    ``json`` or ``json_object`` constraint, else those whose record declares a response_format."""
    declared = project.config.workload.structured_output or {}
    if "json" in declared or "json_object" in declared:
        return frozenset(entry.id for entry in project.prompts)
    return frozenset(entry.id for entry in project.prompts if entry.constrained)


def with_workload(
    project: ResolvedProject,
    *,
    requests: Sequence[PromptEntry] | None = None,
    concurrency: int | None = None,
) -> ResolvedProject:
    """The project offering exactly ``requests`` (one each, in order) and/or keeping
    ``concurrency`` requests in flight; what is left out stays as registered.

    The changed workload is validated as a registered one is: no requests, records of the
    other api, or a concurrency on an arrival that does not take one raise ``ValueError``.
    """
    workload = project.config.workload
    changes: dict[str, object] = {}
    prompts = project.prompts if requests is None else tuple(requests)
    if requests is not None:
        changes = {
            "prompts": tuple(entry.prompt for entry in prompts if entry.prompt is not None),
            "chats": tuple(entry.chat for entry in prompts if entry.messages is not None),
            "request_count": len(prompts),
        }
    if concurrency is not None:
        arrival = {**dict(workload.arrival), "concurrency": concurrency}
        changes["arrival"] = ArrivalSpec.model_validate(arrival)
    spec = WorkloadSpec.model_validate({**dict(workload), **changes})
    return replace(project, prompts=prompts, config=replace(project.config, workload=spec))


def settings_for(project: ResolvedProject, engine: Engine, engine_args: Sequence[str]) -> Settings:
    """The customer's current setup as settings, with ``KEY=VALUE`` engine flags applied."""
    settings = project.record.current_setup.engine_settings
    facts = workload_facts(project)
    if settings.tool_parser is None and (facts.offers_tools or settings.tool_calling):
        settings = replace(settings, tool_parser=engine.default_tool_parser(project.artifact.path))
    return apply_engine_args(engine, settings, engine_args)


def tool_calling_warning(
    project: ResolvedProject, engine: Engine, settings: Settings
) -> str | None:
    """The warning ``init`` and ``analyse`` print when prompts offer tools that ``settings``
    cannot serve: the server would refuse those requests or ignore the tools."""
    offering = sum(bool(entry.tools) for entry in project.prompts)
    source = project.record.current_setup.source
    if not offering or (fix := engine.tool_calling_fix(settings, source)) is None:
        return None
    return f"warning: {offering} of {len(project.prompts)} prompts offer tools; {fix}"


def apply_engine_args(engine: Engine, settings: Settings, engine_args: Sequence[str]) -> Settings:
    """``settings`` with each ``KEY=VALUE`` engine flag applied in turn."""
    trusted = engine.runs_remote_code(settings)
    for text in engine_args:
        try:
            settings = engine.with_engine_arg(settings, text)
        except ValueError as error:
            raise AnalyseFailure(str(error)) from None
    if engine.runs_remote_code(settings) and not trusted:
        raise AnalyseFailure(
            "trust-remote-code changes which code the model runs, so it is part of the model's "
            "identity: register the project again with it in --current"
        )
    return settings


def load_project(
    project: Path,
    *,
    expected_snapshot_id: str | None = None,
    verify_weights: bool = False,
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
        project=project,
        refresh=verify_weights,
        engine=record.current_setup.engine,
        trust_remote_code=project_engine(record.current_setup).runs_remote_code(
            record.current_setup.engine_settings
        ),
    )
    _require_same_identity(record, derived)
    if expected_snapshot_id is not None and expected_snapshot_id != record.snapshot_id:
        raise PreflightError(
            PROJECT_SNAPSHOT_MISMATCH, "the project snapshot is not the expected one"
        )
    return replace(derived, record=record)


def registered_setup(project: Path) -> CurrentSetup | None:
    """The project's current setup (its ``--current`` command: image, GPUs), if it is registered."""
    record = _read_record(project.expanduser())
    return None if record is None else record.current_setup


def project_engine(setup: CurrentSetup) -> Engine:
    """The engine the project records; refused when this Tensward version does not have it."""
    engine = ENGINES.get(setup.engine)
    if engine is None:
        raise PreflightError(
            PROJECT_RECORD_INVALID,
            "the project records an engine this Tensward version does not have; upgrade Tensward",
        )
    return engine


def project_environment(record: ProjectRecord) -> dict[str, object]:
    """Where the project runs here: the detected platform, the recorded engine and why it
    serves the project, and the checkpoint's format."""
    setup = record.current_setup
    engine = project_engine(setup)
    fmt = detect_format(record.artifact.path)
    found = detect_platform()
    platform = found[0] if found else None
    return {
        "platform": platform.name if platform else None,
        "engine": engine.name,
        "engine_choice": engine_choice(engine, setup.text, fmt, platform),
        "format": fmt.name,
    }


def project_summary(
    record: ProjectRecord, anatomy: ModelAnatomy, fit: Fit, environment: dict[str, object]
) -> dict[str, object]:
    """The project as printed by ``init`` and ``inspect``: identities, the current setup, the
    environment it runs in, what the checkpoint is made of and whether it fits this GPU, never
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
        "anatomy": anatomy.summary(),
        "fit": fit.summary(),
        "environment": environment,
    }


def project_fit(resolved: ResolvedProject, engine: Engine, settings: Settings | None = None) -> Fit:
    """Will the registered model fit this machine's GPU at ``settings`` (default: the current
    setup)? The workload declares the concurrency; an open arrival declares none."""
    record, workload = resolved.record, resolved.config.workload
    settings = settings or record.current_setup.engine_settings
    arrival = workload.arrival
    concurrency = arrival.concurrency if arrival.kind == "closed_loop" else arrival.max_inflight
    prompt_chars = [len(entry.text) for entry in resolved.prompts]
    vision = resolved.anatomy.vision
    takes_images = "image" in resolved.anatomy.modalities
    image_tokens = round(
        sum(len(entry.image_urls) for entry in resolved.prompts)
        / len(resolved.prompts)
        * ((vision and vision.max_tokens_per_image) or 0)
    )
    return estimate_fit(
        resolved.anatomy,
        settings,
        concurrency=concurrency or settings.max_concurrent_requests,
        avg_tokens=sum(prompt_chars) // len(prompt_chars) // CHARS_PER_TOKEN
        + image_tokens
        + workload.output_tokens,
        counts_images=image_tokens > 0,
        gpus=detect_devices(),
        selected=record.current_setup.gpus,
        default_fraction=engine.default_kv_memory_fraction,
        context_limit=record.artifact.metadata.context_limit,
        parallel=engine.parallel_degree(settings),
        in_flight_tokens=engine.kv_in_flight_tokens(settings, takes_images),
        overhead_bytes=engine.memory_overhead_bytes,
        loads_encoders=engine.loads_encoders(settings, resolved.anatomy.modalities),
    )


# --- deriving and comparing identities -----------------------------------------------


def _derive(
    model_root: Path,
    config_path: Path,
    prompts_path: Path,
    parsed: ParsedSetup | None = None,
    *,
    project: Path | None = None,
    refresh: bool = True,
    engine: str = LEGACY_ENGINE,
    trust_remote_code: bool = False,
) -> ResolvedProject:
    """Read the sources and work out the project they describe right now. With ``project``,
    the weights and images are hashed again only if their files changed.

    The checkpoint's Python files are part of its identity when the current setup lets the
    engine run them."""
    prompts = load_prompt_entries(prompts_path)
    settings, config_digest = load_serving_config(config_path, prompts)
    images = load_prompt_images(
        prompts_path,
        prompts,
        cache=project and project / IMAGES_CACHE,
        refresh=refresh,
    )
    entry, anatomy = detect_format(model_root).register(
        model_root,
        engine_build=settings.engine_build,
        engine_name=engine,
        cache=project and project / WEIGHTS_CACHE,
        refresh=refresh,
        trust_remote_code=trust_remote_code,
    )
    _check_case_against_checkpoint(settings, entry)
    if settings.workload.api == "chat":
        _require_chat_template(model_root)
    if images.images and "image" not in anatomy.modalities:
        raise PreflightError(
            PROJECT_CONFIG_UNSUPPORTED,
            "the workload sends images but the checkpoint takes no images",
        )
    digest = workload_digest(prompts, images)
    current = _current_setup(parsed, settings, entry, engine)
    record = ProjectRecord(
        project_id=uuid.uuid4().hex,
        snapshot_id=_snapshot_id(entry, config_digest, digest, current),
        artifact=entry,
        config_path=str(config_path),
        prompts_path=str(prompts_path),
        config_digest=config_digest,
        workload_digest=digest,
        current_setup=current,
        images=tuple(
            ImageRecord(url=i.url, sha256=i.sha256, size=i.size) for i in images.images.values()
        ),
    )
    return ResolvedProject(record, settings, prompts, images, anatomy)


def _snapshot_id(
    entry: ArtifactEntry, config_digest: str, workload_digest: str, current: CurrentSetup
) -> str:
    setup = current.model_dump(mode="json")
    if setup["engine"] == LEGACY_ENGINE:  # likewise: projects before 0.2.0 all ran vLLM
        del setup["engine"]
    if not current.gpus:  # recorded only when present, so identities made before it stay valid
        del setup["gpus"]
    setup["settings"] = {
        key: value
        for key, value in setup["settings"].items()
        if key in SNAPSHOT_V1_SETTINGS
        or key not in _SETTINGS_DEFAULTS
        or value != _SETTINGS_DEFAULTS[key]
    }
    payload = {
        "schema_version": "1",
        "artifact_fingerprint": entry.metadata.fingerprint.value,
        "config_digest": config_digest,
        "workload_digest": workload_digest,
        "current_setup": setup,
    }
    return hashlib.sha256(SNAPSHOT_DIGEST_DOMAIN + canonical_json(payload)).hexdigest()


def _require_self_consistent(record: ProjectRecord) -> None:
    """Refuse a record whose snapshot id does not follow from its own fields."""
    expected = _snapshot_id(
        record.artifact, record.config_digest, record.workload_digest, record.current_setup
    )
    if expected != record.snapshot_id:
        raise PreflightError(
            PROJECT_RECORD_INVALID, "the project record is not consistent with its own identity"
        )


def _require_same_identity(stored: ProjectRecord, fresh: ResolvedProject) -> None:
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
        current = {image.url: image for image in now.images}
        changed = next((i.url for i in stored.images if current.get(i.url) != i), None)
        raise PreflightError(
            PROJECT_INPUTS_CHANGED,
            "the workload document changed since registration"
            if changed is None or not now.images
            else f"workload image {changed} changed since registration",
        )


def _check_case_against_checkpoint(settings: ServingConfig, entry: ArtifactEntry) -> None:
    """Refuse serving settings the checkpoint does not provide."""
    case, metadata = settings.case, entry.metadata
    provided = {(v.weight_precision, v.activation_dtype) for v in metadata.variants}
    if (case.weight_precision, case.activation_dtype) not in provided:
        # Both sides are validated enum values, safe to show.
        offered = " or ".join(sorted(f"{p} with {d}" for p, d in provided))
        raise PreflightError(
            PROJECT_CONFIG_UNSUPPORTED,
            f"the configuration declares weight_precision {case.weight_precision} with "
            f"activation_dtype {case.activation_dtype}; the checkpoint provides {offered}",
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


def _parse_current(choice: EngineChoice, text: str | None) -> ParsedSetup | None:
    if text is None:
        return None
    try:
        return choice.engine.parse_setup(text)
    except ValueError as error:
        if choice.named:
            raise PreflightError(PROJECT_INPUTS_INVALID, f"--current: {error}") from None
        raise PreflightError(
            PROJECT_INPUTS_INVALID,
            f"Tensward can't tell which engine this command runs ({error}); supported: "
            f"{supported_launchers()}",
        ) from None


def _require_same_quantization(engine: Engine, parsed: ParsedSetup, entry: ArtifactEntry) -> None:
    """Refuse a command that quantizes the model with another method than the checkpoint's. The
    engine's aliases of one method (``auto_gptq`` for ``gptq``) are the same method."""
    method = entry.metadata.variants[0].quantization_method
    asked = parsed.settings.quantization
    if asked and engine.quantization_family(asked) != engine.quantization_family(method):
        raise PreflightError(
            PROJECT_CONFIG_UNSUPPORTED,
            f"--current quantizes with {asked!r}, the registered checkpoint is {method!r}",
        )


def _current_setup(
    parsed: ParsedSetup | None, settings: ServingConfig, entry: ArtifactEntry, engine: str
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
                source="defaults",
                settings=asdict(Settings(tool_calling=case.tool_calling)),
                engine=engine,
            )
        engine_settings = Settings(
            dtype=case.activation_dtype, tool_calling=case.tool_calling, **declared
        )
        return CurrentSetup(source="config", settings=asdict(engine_settings), engine=engine)
    notes = parsed.notes
    if case.tool_calling:  # false is also the default, so only true is a declared value
        declared["tool_calling"] = True
    if conflicts := [
        f"{name} (configuration {value}"
        + ("" if (own := getattr(parsed.settings, name)) is None else f"; your command sets {own}")
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
        engine=engine,
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
    document = record.model_dump(mode="json")
    if record.current_setup.engine == LEGACY_ENGINE:  # as before 0.2.0, so 0.1.x can read it
        del document["current_setup"]["engine"]
    text = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
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
