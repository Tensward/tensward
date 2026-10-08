"""What runs here: the engine a project records and why, and what each run ran on.

Engines, checkpoint formats and platforms are independent registries; this module is where
they meet. It never imports ``project``.
"""

from __future__ import annotations

import os
import platform as host_platform
import re
import shlex
from dataclasses import asdict, dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .artifacts import ArtifactEntry
from .engines import ENGINES, Engine
from .errors import (
    ENGINE_UNAVAILABLE,
    GPU_MISMATCH,
    PROJECT_CONFIG_UNSUPPORTED,
    PROJECT_INPUTS_INVALID,
    PreflightError,
)
from .fit import GIB
from .formats import FORMATS, CheckpointFormat, detect_format
from .platforms import PLATFORMS, Platform, detect_devices, detect_platform
from .settings import Availability

if TYPE_CHECKING:
    from .runtime import Runtime

TORCH_FAMILY = ("torch", "torchaudio", "torchvision")
CUDA_TAG = re.compile(r"""__version__\s*=\s*['"][^'"]*\+(cu\d+)""")

RUNTIME_WORDS = {"docker": "docker image", "local": "local command"}

PROC = Path("/proc")


@dataclass(frozen=True, slots=True)
class EngineChoice:
    """The engine ``init`` records. ``named``: ``--engine`` named it or ``--current`` runs it,
    so a command it cannot read is refused as that command's fault, not as a wrong guess."""

    engine: Engine
    named: bool


def compatible_engines(fmt: str, platform: str | None) -> tuple[Engine, ...]:
    """The engines, in preference order, that serve ``fmt`` on ``platform`` (on any platform
    when none was detected)."""
    return tuple(
        engine
        for engine in ENGINES.values()
        if fmt in engine.formats and (platform is None or platform in engine.platforms)
    )


def resolve_engine(
    *,
    requested: str | None,
    current: str | None,
    fmt: CheckpointFormat,
    platform: Platform | None,
) -> EngineChoice:
    """The engine ``init`` records: ``--engine``, else the one the ``--current`` command
    launches, else the first in preference order that serves ``fmt`` on ``platform``. Refuses
    an ``--engine`` that conflicts with ``--current``, and an engine that cannot serve this
    checkpoint here."""
    recognised = [e for e in ENGINES.values() if current is not None and e.recognizes(current)]
    if requested is not None:
        engine = ENGINES[requested]
        if recognised and engine not in recognised:
            raise PreflightError(
                PROJECT_INPUTS_INVALID,
                f"--engine {requested} conflicts with --current, which runs "
                f"{recognised[0].name}; give only one of them",
            )
    else:
        candidates = recognised or list(
            compatible_engines(fmt.name, platform.name if platform else None)
        )
        if not candidates:
            raise PreflightError(
                PROJECT_CONFIG_UNSUPPORTED,
                f"no engine serves a {fmt.label} checkpoint{_on(platform)}; {_supported()}",
            )
        engine = candidates[0]
    if fmt.name not in engine.formats or (
        platform is not None and platform.name not in engine.platforms
    ):
        raise PreflightError(
            PROJECT_CONFIG_UNSUPPORTED,
            f"{engine.name} does not serve a {fmt.label} checkpoint{_on(platform)}; {_supported()}",
        )
    return EngineChoice(engine, named=requested is not None or bool(recognised))


def engine_choice(
    engine: Engine, current: str | None, fmt: CheckpointFormat, platform: Platform | None
) -> str:
    """Why the project's ``engine`` runs it, derived the same way by ``init`` and
    ``inspect``."""
    if current is not None and engine.recognizes(current):
        return "from your --current command"
    candidates = compatible_engines(fmt.name, platform.name if platform else None)
    if not candidates or candidates[0] is not engine:
        return "from --engine"
    where = _on(platform) or "; no supported accelerator detected"
    others = ", ".join(other.name for other in candidates[1:])
    return f"chosen for a {fmt.label} checkpoint{where}" + (
        f"; also possible: {others}" if others else ""
    )


def supported_launchers() -> str:
    """Each engine and the commands it recognises, for a refusal."""
    return "; ".join(f"{engine.name} ({engine.launchers})" for engine in ENGINES.values())


def _on(platform: Platform | None) -> str:
    return f" on {platform.label}" if platform is not None else ""


def _supported() -> str:
    labels = {f.name: f.label for f in FORMATS} | {p.name: p.label for p in PLATFORMS}

    def words(names: frozenset[str]) -> str:
        return " or ".join(labels.get(name, name) for name in sorted(names))

    return "supported: " + "; ".join(
        f"{e.name} serves {words(e.formats)} on {words(e.platforms)}" for e in ENGINES.values()
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Host:
    """The machine a run ran on, beside its GPUs. ``overcommit_memory`` is Linux's
    ``vm.overcommit_memory`` (0 heuristic, 1 always, 2 never)."""

    cpu_model: str | None
    physical_cores: int | None
    logical_cores: int | None
    ram_bytes: int | None
    swap_bytes: int | None
    overcommit_memory: int | None


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except (OSError, UnicodeDecodeError):
        return None


def _meminfo_bytes(meminfo: str, key: str) -> int | None:
    found = re.search(rf"^{key}:\s+(\d+) kB$", meminfo, re.MULTILINE)
    return int(found[1]) * 1024 if found else None


def describe_host() -> Host:
    """From /proc/cpuinfo ("model name", distinct "physical id"/"core id" pairs),
    /proc/meminfo ("MemTotal", "SwapTotal") and /proc/sys/vm/overcommit_memory; elsewhere
    platform.processor() and os.cpu_count(). A value that cannot be read is None."""
    cpuinfo = _read(PROC / "cpuinfo") or ""
    meminfo = _read(PROC / "meminfo") or ""
    overcommit = (_read(PROC / "sys/vm/overcommit_memory") or "").strip()
    model = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
    cores = set()
    for entry in cpuinfo.split("\n\n"):
        socket = re.search(r"^physical id\s*:\s*(\d+)$", entry, re.MULTILINE)
        core = re.search(r"^core id\s*:\s*(\d+)$", entry, re.MULTILINE)
        if socket and core:
            cores.add((socket[1], core[1]))
    return Host(
        cpu_model=model[1].strip() if model else host_platform.processor() or None,
        physical_cores=len(cores) or None,
        logical_cores=os.cpu_count(),
        ram_bytes=_meminfo_bytes(meminfo, "MemTotal"),
        swap_bytes=_meminfo_bytes(meminfo, "SwapTotal"),
        overcommit_memory=int(overcommit) if overcommit.isdigit() else None,
    )


@dataclass(frozen=True, slots=True)
class RunEnvironment:
    """What a run ran on, read once before it starts: its report's first line and its run.json.
    ``identities`` are those of every device the run could see; ``host`` is the machine."""

    engine: str
    engine_version: str | None
    platform: str | None
    format: str
    identities: tuple[dict[str, Any], ...]
    ran: str
    host: Host


def describe_run(
    engine: Engine,
    runtime: Runtime,
    artifact: ArtifactEntry,
    availability: Availability | None = None,
) -> RunEnvironment:
    """What a run of ``artifact`` with ``engine`` on ``runtime`` will run on. The line names the
    devices the run selected (the first without a selection) and, for a local command, only its
    executable."""
    found = availability or engine.availability(runtime.kind, runtime.target)
    platform = runtime.platform
    devices = platform.select(detect_devices(), runtime.gpus)[0]
    identities = tuple(platform.identity(device) for device in devices)
    shown = [
        platform.describe(device, identity)
        for device, identity in zip(devices, identities, strict=True)
    ]
    if not runtime.gpus:
        shown = shown[:1]
    served_by = (
        runtime.target
        if runtime.kind == "docker"
        else os.path.basename(shlex.split(runtime.target)[0])
    )
    fmt = detect_format(artifact.path)
    variant = artifact.metadata.variants[0]
    ran = (
        f"{engine.label} {found.version or 'version unknown'} "
        f"({RUNTIME_WORDS[runtime.kind]} {served_by}) on "
        f"{', '.join(shown) or 'no detected accelerator'}; "
        f"checkpoint: {fmt.label}, {variant.label or variant.weight_precision}"
    )
    return RunEnvironment(
        engine.name,
        found.version,
        platform.name if devices else None,
        fmt.name,
        identities,
        ran,
        describe_host(),
    )


def with_logged_version(engine: Engine, environment: RunEnvironment, log: str) -> RunEnvironment:
    """``environment`` with the version the server log names, which is what ran; unchanged when
    the log names none."""
    found = re.search(engine.version_log, log)
    if not found or found[1] == environment.engine_version:
        return environment
    before = f"{engine.label} {environment.engine_version or 'version unknown'} "
    return replace(
        environment,
        engine_version=found[1],
        ran=environment.ran.replace(before, f"{engine.label} {found[1]} ", 1),
    )


def require_available(engine: Engine, runtime: Runtime) -> Availability:
    """The engine's availability for ``runtime``; refused when it is not there."""
    found = engine.availability(runtime.kind, runtime.target)
    if not found.available:
        raise PreflightError(
            ENGINE_UNAVAILABLE,
            f"{engine.label} is not available as the {RUNTIME_WORDS[runtime.kind]} "
            f"{runtime.target}; {found.how_to_get}",
        )
    return found


def unavailable_warning(engine: Engine, command: str | None, image: str | None) -> str | None:
    """A warning for ``init`` when the engine is not here the way the project will run it: as
    the ``--current`` command's own image or local command, or, without one, as either."""
    local = shlex.join(engine.local_command)
    if command is None:
        targets = {"docker": engine.default_image, "local": local}
    else:
        targets = {"docker": image} if image else {"local": local}
    found = {kind: engine.availability(kind, target) for kind, target in targets.items()}
    if any(check.available for check in found.values()):
        return None
    options = "; or ".join(
        f"as the {RUNTIME_WORDS[kind]} {targets[kind]}: {check.how_to_get}"
        for kind, check in found.items()
    )
    return f"warning: {engine.label} is not available, {options}"


def torch_cuda_builds() -> dict[str, str]:
    """The CUDA tag (``cu130``) each installed torch-family package was built for, read from its
    ``version.py`` without importing it. Packages without a tag are left out."""
    builds = {}
    for name in TORCH_FAMILY:
        try:
            files = metadata.files(name) or []
            version_file = next(f for f in files if f.parts == (name, "version.py"))
            found = CUDA_TAG.search(version_file.read_text())
        except (metadata.PackageNotFoundError, StopIteration, OSError, UnicodeDecodeError):
            continue
        if found:
            builds[name] = found.group(1)
    return builds


def environment_report(
    *,
    engines: Sequence[str],
    runtimes: Sequence[str],
    image: str | None,
    local_command: str | None,
) -> dict[str, Any]:
    """What ``tensward env`` reports: the detected platform and its devices, each engine's
    availability per runtime, the registered formats, and the combinations that work here."""
    warnings = [warning for name in engines for warning in ENGINES[name].host_warnings()]
    rows: list[dict[str, Any]] = []
    for name in engines:
        engine = ENGINES[name]
        targets = {
            "docker": image or engine.default_image,
            "local": local_command or shlex.join(engine.local_command),
        }
        checks = {
            kind: {"target": targets[kind], **asdict(engine.availability(kind, targets[kind]))}
            for kind in runtimes
        }
        rows.append(
            {"name": engine.name, "label": engine.label, "formats": sorted(engine.formats),
             "platforms": sorted(engine.platforms), **checks}
        )  # fmt: skip
    formats = [{"name": fmt.name, "label": fmt.label} for fmt in FORMATS]
    found = detect_platform()
    if found is None:
        return {"platform": None, "engines": rows, "formats": formats, "combinations": [],
                "warnings": warnings}  # fmt: skip
    platform, devices = found
    devices_here = []
    for device in devices:
        identity = platform.identity(device)
        devices_here.append(
            {**identity, "total_bytes": device.total_bytes, "used_bytes": device.used_bytes,
             "description": platform.describe(device, identity)}
        )  # fmt: skip
    combinations = [
        {"engine": row["name"], "format": fmt["name"], "platform": platform.name,
         "runtimes": usable}
        for row in rows
        if platform.name in row["platforms"]
        for fmt in formats
        if fmt["name"] in row["formats"]
        if (usable := [kind for kind in runtimes if row[kind]["available"]])
    ]  # fmt: skip
    return {
        "platform": {"name": platform.name, "label": platform.label, "devices": devices_here},
        "engines": rows,
        "formats": formats,
        "combinations": combinations,
        "warnings": warnings,
    }


def render_environment(report: Mapping[str, Any]) -> list[str]:
    """``tensward env`` as text."""
    platform = report["platform"]
    if platform is None:
        lines = ["platform: no supported accelerator detected"]
    else:
        devices = platform["devices"]
        lines = [f"platform: {platform['label']}, {len(devices)} device(s)"]
        lines += [
            f"  {d['index']}: {d['description']}, {d['total_bytes'] / GIB:.1f} GiB" for d in devices
        ]
    for engine in report["engines"]:
        lines.append(f"engine {engine['name']} ({engine['label']}):")
        for kind, words in RUNTIME_WORDS.items():
            if (check := engine.get(kind)) is not None:
                state = (
                    f"available, version {check['version'] or 'unknown'}"
                    if check["available"]
                    else f"not available; {check['how_to_get']}"
                )
                lines.append(f"  {words} {check['target']}: {state}")
    labels = {fmt["name"]: fmt["label"] for fmt in report["formats"]}
    lines.append("formats: " + ", ".join(f"{label} ({name})" for name, label in labels.items()))
    works = [
        f"{c['engine']} + {labels[c['format']]} on {platform['label']} ({', '.join(c['runtimes'])})"
        for c in report["combinations"]
    ]
    lines.append("works here: " + ("; ".join(works) or "nothing yet"))
    return lines + list(report.get("warnings", []))


def _bare_name(platform: Platform, name: str) -> str:
    return name.strip().casefold().removeprefix(platform.label.casefold() + " ")


def require_gpus(
    platform: Platform, selection: tuple[str, ...] | None, name: str | None, driver: str | None
) -> None:
    """Refuse a machine whose selected GPUs (all of them without a selection) are not the card
    ``name`` or do not run ``driver`` (a dotted prefix of its version also matches)."""
    gpus = detect_devices()
    if not gpus:
        raise PreflightError(GPU_MISMATCH, "no supported accelerator detected")
    if why := platform.selection_refusal(selection or ()):
        raise PreflightError(GPU_MISMATCH, f"{why} with --require-gpu or --require-driver")
    matched, missing = platform.select(gpus, selection)
    if missing:
        listed = ", ".join(gpu.index for gpu in gpus)
        raise PreflightError(
            GPU_MISMATCH, f"GPU {missing[0]} selected, but the GPUs here are {listed}"
        )
    for gpu in matched:
        actual = f"GPU {gpu.index} is {gpu.name} with driver {gpu.driver}, but"
        if name is not None and _bare_name(platform, gpu.name) != _bare_name(platform, name):
            raise PreflightError(GPU_MISMATCH, f"{actual} --require-gpu wants {name}")
        if driver is not None and not (
            gpu.driver is not None and (gpu.driver + ".").startswith(driver + ".")
        ):
            raise PreflightError(GPU_MISMATCH, f"{actual} --require-driver wants {driver}")
