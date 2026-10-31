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
import random
import re
import stat
import sys
import uuid
from contextlib import contextmanager
from dataclasses import MISSING, asdict, dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Sequence

from .anatomy import ModelAnatomy, freed_bytes
from .artifacts import AbsolutePath, ArtifactEntry, GgufFiles
from .ceilings import Unavailable, gpu_spec
from .context import current
from .contracts import DigestHex, Identifier, StrictModel
from .engines import ENGINES, Engine
from .engines.llamacpp.detect import (
    NEEDS_COMMAND,
    Evidence,
    command_evidence,
    env_evidence,
    model_evidence,
)
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
from .formats.gguf import has_chat_template, kv_bytes_per_token
from .images import PromptImages, load_prompt_images
from .inputs import (
    MAX_INPUT_BYTES,
    DocumentWorkload,
    PromptEntry,
    PromptFile,
    ServingConfig,
    load_serving_config,
    read_serving_document,
    refuse_named_tool_under_config,
    workload_digest,
)
from .kept import Kept
from .platforms import MIB, detect_devices, detect_platform
from .platforms.cpu import host_facts
from .playbook import WorkloadFacts
from .settings import ParsedSetup, Settings
from .text import canonical_json
from .workload import ArrivalSpec, WorkloadSpec

if TYPE_CHECKING:
    from .runtime import Runtime

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
    executable: str | None = None  # the program the command runs, as an absolute path

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
    choice: EngineChoice | None = None

    @property
    def artifact(self) -> ArtifactEntry:
        return self.record.artifact


class NeedsCommand(PreflightError):
    """``init --endpoint`` found the engine but has no command to relaunch it with."""

    def __init__(self, choice: EngineChoice) -> None:
        super().__init__(PROJECT_INPUTS_INVALID, NEEDS_COMMAND)
        self.choice = choice


# --- the operations --------------------------------------------------------------------


def init_project(
    project: Path,
    *,
    model: Path,
    config: Path,
    prompts: Path,
    engine: str | None = None,
    current: str | None = None,
    endpoint: Sequence[Evidence] | None = None,
    current_is_file: bool = False,
    check: Callable[[ResolvedProject], None] | None = None,
) -> ResolvedProject:
    """Register a project, or return the identical existing registration, resolved.

    ``current`` is the customer's own engine command line; without it the baseline is the
    serving fields of ``config``. ``engine`` is ``--engine``; the engine recorded is resolved by
    :func:`resolve_engine` from ``--engine`` and the evidence of the command, the checkpoint, the
    environment and ``endpoint`` (what ``--endpoint`` found; None without it, which then needs
    ``current``). An existing record is verified, never replaced, and a
    request that differs from it is refused. ``check`` sees a new registration before it is
    written, and refuses it by raising.
    """
    project = project.expanduser()
    sources = (
        _resolve_model(model),
        _resolve(config, "the serving configuration", directory=False),
        _resolve(prompts, "the workload document", directory=False),
    )
    found = detect_platform()
    choice = resolve_engine(
        requested=engine,
        current=current,
        fmt=detect_format(sources[0]),
        platform=found[0] if found else None,
        evidence=[
            *command_evidence(current or ""),
            *model_evidence(sources[0]),
            *env_evidence(os.environ, current if current_is_file else None),
            *(endpoint or ()),
        ],
        interactive=sys.stdin.isatty() and sys.stderr.isatty(),
    )
    if endpoint is not None and current is None:
        raise NeedsCommand(choice)
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
            if check is not None:
                check(derived)
            _publish(project, derived.record)
            return replace(derived, choice=choice)
        _require_self_consistent(existing)
        bound = (
            _bound_model(existing.artifact, sources[0]),
            Path(existing.config_path),
            Path(existing.prompts_path),
        )
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
        return replace(derived, record=existing, choice=choice)


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


def linear_weight_bytes(project: ResolvedProject) -> int | None:
    """The bytes of the checkpoint's linear weights that an fp8 quantization would shrink, or
    None when its anatomy has no component sizes."""
    freed = freed_bytes(project.anatomy)
    return None if freed is None else freed["quantizable"]


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


# The last project derived from its sources, with the stamps of those sources. One analysis loads
# the same project twice; the second load reuses the first unless a source moved.
_DERIVED: list[tuple[tuple[Any, ...], ResolvedProject]] = []


def _stamps(root: Path) -> tuple[tuple[str, int, int], ...]:
    """Path, size and mtime of the file at ``root``, or of every file under it."""
    paths = [root] if root.is_file() else [Path(d, n) for d, _, ns in os.walk(root) for n in ns]
    found = []
    for path in paths:
        try:
            info = path.stat()
        except OSError:
            continue
        found.append((str(path), info.st_size, info.st_mtime_ns))
    return tuple(sorted(found))


def _model_stamps(entry: ArtifactEntry) -> tuple[tuple[str, int, int], ...]:
    """The stamps of the checkpoint: every shard of a GGUF model, else the directory's files."""
    if isinstance(entry.files, GgufFiles):
        return tuple(
            sorted(s for name in entry.weights.shards for s in _stamps(Path(entry.root, name)))
        )
    return _stamps(entry.path)


def _derived_key(project: Path, record: ProjectRecord, verify_weights: bool) -> tuple[Any, ...]:
    return (
        project,
        verify_weights,
        record.snapshot_id,
        _model_stamps(record.artifact),
        _stamps(Path(record.config_path)),
        _stamps(Path(record.prompts_path)),
    )


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
    key = _derived_key(project, record, verify_weights)
    derived = _DERIVED[0][1] if _DERIVED and _DERIVED[0][0] == key else None
    if derived is None:
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
        _DERIVED[:] = [] if derived.images.images else [(key, derived)]
    _require_same_identity(record, derived)
    if expected_snapshot_id is not None and expected_snapshot_id != record.snapshot_id:
        raise PreflightError(
            PROJECT_SNAPSHOT_MISMATCH, "the project snapshot is not the expected one"
        )
    resolved = replace(derived, record=record)
    plan = current().window_plan
    if plan is None or plan["requests"] == resolved.config.workload.request_count:
        return resolved
    workload = resolved.config.workload.model_copy(update={"request_count": plan["requests"]})
    return replace(resolved, config=replace(resolved.config, workload=workload))


def registered_setup(project: Path) -> CurrentSetup | None:
    """The project's current setup (its ``--current`` command: image, GPUs), if it is registered."""
    record = _read_record(project.expanduser())
    return None if record is None else record.current_setup


def registered_env(project: Path) -> dict[str, str]:
    """The environment the registered setup sets for the engine; empty when none is registered."""
    current = registered_setup(project)
    return dict(current.settings.get("extra_env") or {}) if current is not None else {}


def declared_workload(project: Path) -> DocumentWorkload | None:
    """The workload the registered configuration declares; None when it cannot be read."""
    try:
        record = _read_record(project.expanduser())
        return None if record is None else read_serving_document(Path(record.config_path)).workload
    except PreflightError:
        return None


def project_slos(project: Path) -> dict[str, float]:
    """The latency limits the registered configuration declares in ``slos``; empty when it
    declares none or nothing is registered here."""
    record = _read_record(project.expanduser())
    if record is None:
        return {}
    return dict(read_serving_document(Path(record.config_path)).slos or {})


SAMPLE_ABOVE = 10_000  # a workload of at most this many records is sent whole, in file order


@dataclass(frozen=True, slots=True)
class WorkloadSample:
    """The records a run sends from a workload larger than it needs: ``records`` of the file's
    ``of``, drawn with ``seed``, so every run of the project sends the same ones."""

    seed: int
    records: int
    of: int

    def positions(self) -> list[int]:
        return sorted(random.Random(self.seed).sample(range(self.of), self.records))


def sample_of(records: int, needed: int, workload_digest: str) -> WorkloadSample | None:
    if records <= SAMPLE_ABOVE or records <= needed:
        return None
    return WorkloadSample(seed=int(workload_digest[:16], 16), records=needed, of=records)


def workload_sample(project: Path) -> WorkloadSample | None:
    """The sample a run of the registered project sends, decided as ``_derive`` decides it; the
    records are counted, not parsed."""
    record = _read_record(project.expanduser())
    if record is None:
        return None
    document = read_serving_document(Path(record.config_path))
    needed = document.workload.request_count + document.warmup_requests
    records = len(PromptFile(Path(record.prompts_path)))
    return sample_of(records, needed, record.workload_digest)


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
    record: ProjectRecord,
    anatomy: ModelAnatomy,
    fit: Fit,
    environment: dict[str, object],
    *,
    kept: Kept | None = None,
) -> dict[str, object]:
    """The project as printed by ``init`` and ``inspect``: identities, the current setup, the
    kept setup, the environment it runs in, what the checkpoint is made of and whether it fits
    this GPU, never an input's path or a prompt."""
    current = record.current_setup
    held = (
        {
            "kept": {
                "run": kept.run,
                "engine_args": list(kept.engine_args),
                **({"env": list(kept.env)} if kept.env else {}),
            }
        }
        if kept
        else {}
    )
    return {
        **held,
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
        "anatomy": _anatomy_summary(anatomy),
        "fit": fit.summary(),
        "environment": environment,
    }


def _anatomy_summary(anatomy: ModelAnatomy) -> dict[str, object]:
    """The anatomy as printed; a GGUF model adds what its tensor bytes and metadata give."""
    summary = anatomy.summary()
    if anatomy.bits_per_weight is not None:
        summary |= {
            "layers": anatomy.layers,
            "bits_per_weight": round(anatomy.bits_per_weight, 2),
            "weight_bytes_by_role": dict(anatomy.weight_bytes_by_role),
            "sliding_window": anatomy.sliding_window,
            "swa_layers": len(anatomy.swa_layers),
            "experts": anatomy.experts,
            "experts_used": anatomy.experts_used,
            "kv_bytes_per_token": {
                kv: kv_bytes_per_token(anatomy, kv, kv) for kv in ("f16", "q8_0")
            },
        }
    return summary


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


def project_fit_advice(
    resolved: ResolvedProject,
    engine: Engine,
    settings: Settings,
    runtime: Runtime | None = None,
    project: Path | None = None,
) -> tuple[str, bool] | None:
    """What the engine says before launch about ``settings`` that cannot fit the GPU: (text,
    whether the launch certainly fails). The command to try is written only with ``project``,
    inside an analysis. None without a single NVIDIA GPU, or when nothing is wrong."""
    from .suggest import suggestion_command  # suggest imports the playbook and the engines

    record = resolved.record
    if runtime is None:
        found = detect_platform()
        platform, devices = (None, ()) if found is None else found
    else:
        platform, devices = runtime.platform, runtime.platform.detect()
    selected = (runtime and runtime.gpus) or record.current_setup.gpus
    if platform is None or platform.name != "nvidia" or (len(devices) > 1 and not selected):
        return None
    try:
        index, name, _ = gpu_spec(tuple((d.name, d.total_bytes // MIB) for d in devices), selected)
    except Unavailable:
        return None
    current = record.current_setup.engine_settings
    write = (
        None
        if project is None
        else lambda proposed: suggestion_command(engine, project, current, proposed)
    )
    baseline = None
    if project is not None:
        from .runs import baseline_resolved  # runs imports the engines

        baseline = baseline_resolved(project, resolved)
    return engine.fit_advice(
        resolved.anatomy, settings, name, devices[index].total_bytes, host_facts(), runtime,
        record.artifact.metadata.context_limit, write, baseline,
    )  # fmt: skip


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
    file = PromptFile(prompts_path)
    images = load_prompt_images(
        prompts_path,
        file.records(),
        cache=project and project / IMAGES_CACHE,
        refresh=refresh,
    )
    digest = workload_digest(file.records(), images)
    document = read_serving_document(config_path)
    needed = document.workload.request_count + document.warmup_requests
    sample = sample_of(len(file), needed, digest)
    prompts = tuple(file.records(None if sample is None else sample.positions()))
    settings, config_digest = load_serving_config(config_path, prompts)
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
        _require_chat_template(model_root, entry)
    if images.images and "image" not in anatomy.modalities:
        raise PreflightError(
            PROJECT_CONFIG_UNSUPPORTED,
            "the workload sends images but the checkpoint takes no images",
        )
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
    if current.executable is None:
        del setup["executable"]
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
    gguf = isinstance(entry.files, GgufFiles)
    provided = {(v.weight_precision, v.activation_dtype) for v in metadata.variants}
    if (case.weight_precision, case.activation_dtype) not in provided:
        # Both sides are validated enum values, safe to show.
        offered = " or ".join(sorted(f"{p} with {d}" for p, d in provided))
        fix = (
            "a GGUF checkpoint needs weight_precision gguf with activation_dtype engine"
            if gguf
            else f"the checkpoint provides {offered}"
        )
        raise PreflightError(
            PROJECT_CONFIG_UNSUPPORTED,
            f"the configuration declares weight_precision {case.weight_precision} with "
            f"activation_dtype {case.activation_dtype}; {fix}",
        )
    if case.max_model_len is not None and case.max_model_len > metadata.context_limit:
        raise PreflightError(
            PROJECT_CONFIG_UNSUPPORTED, "max_model_len exceeds the checkpoint's context limit"
        )
    if not gguf and case.kv_cache_dtype == "q8_0":
        raise PreflightError(
            PROJECT_CONFIG_UNSUPPORTED, "kv_cache_dtype q8_0 is a llama.cpp setting"
        )
    if gguf and case.gpu_memory_utilization is not None:
        raise PreflightError(
            PROJECT_CONFIG_UNSUPPORTED,
            "llama.cpp sizes memory with -c and -ngl; remove gpu_memory_utilization",
        )
    if gguf and case.kv_cache_dtype not in (None, "auto", "q8_0"):
        raise PreflightError(
            PROJECT_CONFIG_UNSUPPORTED,
            f"kv_cache_dtype {case.kv_cache_dtype} is not a llama.cpp setting; "
            "use auto (f16) or q8_0",
        )


def _require_chat_template(model_root: Path, entry: ArtifactEntry) -> None:
    """Refuse a chat workload for a checkpoint that ships no chat template."""
    if isinstance(entry.files, GgufFiles):
        if not has_chat_template(entry.path):
            raise PreflightError(
                PROJECT_INPUTS_INVALID,
                "the workload uses the chat api but the GGUF has no chat template "
                "(tokenizer.chat_template); use an instruct model or the completions api",
            )
        return
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


def _as_command_sets(engine: str, command: Settings, name: str, value: Any) -> Settings:
    """``command`` with the configuration's ``value`` of a field written as the engine writes
    it, so a per-slot context compares with the command's own sizing."""
    if name != "max_context_len":
        return replace(command, **{name: value})
    return ENGINES[engine].with_lever(command, name, value)


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
        context = declared.pop("max_context_len", None)
        engine_settings = Settings(
            dtype=case.activation_dtype, tool_calling=case.tool_calling, **declared
        )
        if context is not None:  # the configuration's context is per slot; the engine sizes it
            sized = ENGINES[engine].with_lever
            engine_settings = sized(engine_settings, "max_context_len", context)
        return CurrentSetup(source="config", settings=asdict(engine_settings), engine=engine)
    notes = parsed.notes
    if case.tool_calling:  # false is also the default, so only true is a declared value
        declared["tool_calling"] = True
    if conflicts := [
        f"{name} (configuration {value}"
        + ("" if (own := getattr(parsed.settings, name)) is None else f"; your command sets {own}")
        + ")"
        for name, value in declared.items()
        if getattr(_as_command_sets(engine, parsed.settings, name, value), name)
        != getattr(parsed.settings, name)
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
        executable=parsed.executable,
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


def _resolve_model(path: Path) -> Path:
    """The canonical location of ``--model``: a checkpoint directory, or a GGUF file (resolved
    by its folder, so a Hugging Face cache link keeps its ``.gguf`` name)."""
    if path.is_dir() or not path.name.endswith(".gguf"):
        return _resolve(path, "the model directory", directory=True)
    resolved = _resolve(path.parent, "the model folder", directory=True) / path.name
    if not os.path.lexists(resolved):
        raise PreflightError(PROJECT_INPUTS_INVALID, not_found_message("the GGUF file", path))
    return resolved


def _bound_model(entry: ArtifactEntry, given: Path) -> Path:
    """The registered model as ``--model`` named it: the folder, or the file (any shard of a
    split model names the same set)."""
    if given.is_dir() or given.name not in entry.weights.shards:
        return Path(entry.root)
    return Path(entry.root) / given.name


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
    if record.current_setup.executable is None:
        del document["current_setup"]["executable"]
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
