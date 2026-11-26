"""Finding the census library that matches an engine, and the CUPTI it must run against."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..engines.protocol import CensusProfile
from ..runtime import engine_cache_root

CENSUS_DIR_ENV = "TENSWARD_CENSUS_DIR"
CUPTI_DIR_ENV = "TENSWARD_CUPTI_DIR"
CUPTI_ENV = "TENSWARD_CUPTI"
CUPTI_HOME = Path("~/.local/share/tensward/cupti")
MANIFEST = "manifest.json"
PROBE_TIMEOUT_S = 60.0
SITE_PACKAGES = ("lib/python*/site-packages", "lib/python*/dist-packages")
WHEEL_DIRS = ("nvidia/cu{major}/lib", "nvidia/cuda_cupti/lib", "torch/lib")
TOOLKIT_DIRS = (
    "/usr/local/cuda*/extras/CUPTI/lib64",
    "/usr/local/cuda*/targets/*-linux/lib",
    "/usr/local/cuda*/lib64",
    "/opt/cuda*/targets/*-linux/lib",
    "/usr/lib/*-linux-gnu",
    "/usr/local/cuda*/nsight-systems-*/target-linux-x64",
    "/opt/nvidia/nsight-systems/*/target-linux-x64",
    "/usr/local/cuda*/nsight-compute-*/target/linux-desktop-glibc_*-x64",
    "/opt/nvidia/nsight-compute/*/target/linux-desktop-glibc_*-x64",
)
VENV_ROOTS = (
    "/opt/*",
    "/opt/*/*",
    "/opt/*/.venv",
    "/usr/local",
    "/usr",
    "~/.venv",
    "~/venv",
    "~/*/.venv",
    "~/*",
    "~/.local",
    "~/.local/share/uv/tools/*",
    "~/.local/pipx/venvs/*",
)
LAUNCHERS = frozenset({"env", "nice", "taskset", "numactl", "sudo", "bash", "sh", "stdbuf"})
SCRIPT_PATHS = re.compile(r"(?<![\w.~-])(/[\w./+-]+)")
CUDA_LIBS = re.compile(r"libcu(?:dart|blas)\.so\.(\d+)")


@dataclass(frozen=True, slots=True)
class CensusLibrary:
    path: Path
    major: int
    cupti_api_min: int
    kind: str


@dataclass(frozen=True, slots=True)
class Cupti:
    """A CUPTI library: its folder as the engine sees it and the checksum of the file."""

    folder: str
    sha256: str | None


def census_home() -> Path:
    given = os.environ.get(CENSUS_DIR_ENV)
    return Path(given) if given else Path(__file__).parent / "lib"


def _manifest(home: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((Path(home) / MANIFEST).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("libraries"), list) else None


def installed(home: Path) -> bool:
    return _manifest(home) is not None


def _sha256(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def find_library(home: Path, major: int) -> CensusLibrary | None:
    """The library for CUDA ``major`` that the manifest lists, when its file is there and its
    checksum is the listed one."""
    manifest = _manifest(home)
    if manifest is None:
        return None
    for entry in manifest["libraries"]:
        try:
            if entry["cupti_major"] != major:
                continue
            path = Path(home) / entry["file"]
            if _sha256(path) != entry["sha256"]:
                return None
            kind = re.search(r"-(cu\d+)\.so$", entry["file"])
            return CensusLibrary(
                path, major, int(entry["cupti_api_min"]), kind.group(1) if kind else f"cu{major}"
            )
        except (KeyError, TypeError, ValueError):
            return None
    return None


def cupti_too_old(home: Path, cupti: Cupti, library: CensusLibrary) -> bool:
    """Whether the manifest lists this CUPTI file with an API older than the library needs. A
    file it does not list is not refused."""
    manifest = _manifest(home)
    if manifest is None or cupti.sha256 is None:
        return False
    for entry in manifest.get("cupti_files") or ():
        try:
            if entry["sha256"] == cupti.sha256:
                return int(entry["api"]) < library.cupti_api_min
        except (KeyError, TypeError, ValueError):
            continue
    return False


def _executable(command: tuple[str, ...]) -> Path | None:
    found = shutil.which(command[0]) if command else None
    return Path(found) if found else None


def _binary(executable: Path) -> Path:
    """The program a wrapper script starts (the last absolute path it names that is an
    executable file), or the executable itself."""
    try:
        with executable.open("rb") as handle:
            if handle.read(2) != b"#!":
                return executable
        text = executable.read_text(errors="replace")
    except OSError:
        return executable
    named = [
        Path(path)
        for line in text.splitlines()[1:]
        if not line.lstrip().startswith("#")
        for path in SCRIPT_PATHS.findall(line)
    ]
    started = [
        p
        for p in named
        if p.is_file()
        and os.access(p, os.X_OK)
        and p != executable
        and p.name not in LAUNCHERS
        and not p.suffix
    ]
    return started[0] if started else executable


def _site_folders(executable: Path | None) -> list[Path]:
    if executable is None:
        return []
    root = executable.parent.parent
    return [
        Path(found) for pattern in SITE_PACKAGES for found in sorted(glob.glob(str(root / pattern)))
    ]


def _expand(template: str, major: int) -> list[str]:
    return sorted(glob.glob(template.replace("{major}", str(major))))


def _given() -> list[Path]:
    folders = []
    if given := os.environ.get(CUPTI_ENV):
        path = Path(given).expanduser()
        folders.append(path.parent if path.is_file() else path)
    if given := os.environ.get(CUPTI_DIR_ENV):
        folders.append(Path(given).expanduser())
    return folders


def _wheel_folders(major: int) -> list[Path]:
    """CUPTI of NVIDIA's pip wheels (and torch's copy) in any Python environment on the box,
    Tensward's own included."""
    roots = [Path(sys.prefix), CUPTI_HOME.expanduser()]
    for pattern in VENV_ROOTS:
        roots += [Path(found) for found in sorted(glob.glob(os.path.expanduser(pattern)))]
    folders: list[Path] = []
    for root in dict.fromkeys(roots):
        sites = [root] if root == CUPTI_HOME.expanduser() else []
        sites += [Path(f) for pattern in SITE_PACKAGES for f in glob.glob(str(root / pattern))]
        for site in sorted(sites):
            for template in WHEEL_DIRS:
                folder = site / template.replace("{major}", str(major))
                if folder.is_dir():
                    folders.append(folder)
    return folders


def _ldconfig(major: int) -> list[Path]:
    try:
        done = subprocess.run(
            [shutil.which("ldconfig") or "/sbin/ldconfig", "-p"],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    name = f"libcupti.so.{major}"
    return [
        Path(line.rsplit("=>", 1)[1].strip()).parent
        for line in done.stdout.splitlines()
        if "=>" in line and line.split()[0] == name
    ]


def cupti_search_dirs(profile: CensusProfile, command: tuple[str, ...], major: int) -> list[Path]:
    """Every folder looked in for ``libcupti.so.<major>``, the first asked for first: the
    user's TENSWARD_CUPTI / TENSWARD_CUPTI_DIR, the engine's own CUDA, the CUDA toolkits and
    the Nsight tools, the linker cache, then NVIDIA's pip wheels in any environment."""
    executable = _executable(command)
    folders = _given()
    for template in profile.cupti_candidates:
        if template.startswith("/"):
            found = _expand(template, major)
            if not found and "*" not in template:
                found = [template.replace("{major}", str(major))]
            folders += [Path(path) for path in found]
        else:
            for site in _site_folders(executable):
                folders += [Path(found) for found in _expand(str(site / template), major)]
    if executable is not None:
        folders.append(executable.parent)
        if (binary := _binary(executable)) != executable:
            folders.append(binary.parent)
    if any(t.startswith("/") for t in profile.cupti_candidates):
        folders += [Path(path) for t in TOOLKIT_DIRS for path in sorted(glob.glob(t))]
        folders += _ldconfig(major)
    folders += _wheel_folders(major)
    return list(dict.fromkeys(folders))


def local_cupti(profile: CensusProfile, command: tuple[str, ...], major: int) -> Cupti | None:
    """The CUPTI library of the engine's own CUDA, symlinks resolved; ``TENSWARD_CUPTI_DIR``
    is looked in first."""
    name = f"libcupti.so.{major}"
    for folder in cupti_search_dirs(profile, command, major):
        file = folder / name
        found = file if file.is_file() else _versioned(folder, name)
        if found is not None:
            real = found.resolve()
            return Cupti(str(real.parent), _sha256(real))
    return None


def _versioned(folder: Path, name: str) -> Path | None:
    """``libcupti.so.12.9`` and the like where a folder has no ``libcupti.so.12`` link."""
    found = sorted(
        folder.glob(f"{name}.*") if folder.is_dir() else (),
        key=lambda path: [int(part) for part in re.findall(r"\d+", path.name[len(name) :])],
    )
    return found[-1] if found else None


def cupti_install(major: int | None) -> str:
    """The command that puts CUPTI where Tensward looks, for the engine's CUDA major."""
    package = "nvidia-cuda-cupti==13.*" if major == 13 else "nvidia-cuda-cupti-cu12"
    command = f"uv pip install --target {CUPTI_HOME} '{package}'"
    return command if major else f"{command} (CUDA 13: 'nvidia-cuda-cupti==13.*')"


def engine_cupti_major(profile: CensusProfile, command: tuple[str, ...]) -> int | None:
    """The CUDA major of the engine's own libraries: from ``ldd`` of a llama.cpp binary and of
    the ``libggml-cuda*.so`` beside it (release builds load CUDA there), from the CUPTI folder of a
    vLLM environment; None when neither says."""
    executable = _executable(command)
    if executable is None:
        return None
    if any(t.startswith("/") for t in profile.cupti_candidates):
        majors: list[int] = []
        binary = _binary(executable)
        beside = {executable.resolve().parent, binary.resolve().parent}
        targets = [binary, *sorted(p for d in beside for p in d.glob("libggml-cuda*.so"))]
        for target in targets:
            try:
                done = subprocess.run(
                    ["ldd", str(target)],
                    capture_output=True,
                    text=True,
                    timeout=PROBE_TIMEOUT_S,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            majors += [int(m) for m in CUDA_LIBS.findall(done.stdout)]
        if majors:
            return max(majors)
        templates = [*profile.cupti_candidates, str(executable.parent), str(binary.parent)]
        present = [
            m
            for m in (12, 13)
            if any(
                (Path(folder) / f"libcupti.so.{m}").is_file()
                for template in templates
                if template.startswith("/")
                for folder in _expand(template, m) or [template.replace("{major}", str(m))]
            )
        ]
        return present[0] if len(present) == 1 else None
    for major in (13, 12):
        if any(
            (folder / f"libcupti.so.{major}").is_file()
            for folder in cupti_search_dirs(profile, command, major)
        ):
            return major
    return None


def image_id_of(image: str) -> str | None:
    try:
        done = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None if done.returncode == 0 else None


def image_cupti(profile: CensusProfile, image_id: str, major: int) -> Cupti | None:
    """The CUPTI library inside an image: one ``docker run --rm --entrypoint /bin/sh``, the
    answer kept next to the image's compile caches."""
    cache = engine_cache_root(image_id) / f"census-cupti-{major}.json"
    try:
        known = json.loads(cache.read_text())
        return Cupti(known["folder"], known.get("sha256")) if known else None
    except (OSError, ValueError, KeyError, TypeError):
        pass
    paths = " ".join(
        f'-path "*/{t.replace("{major}", str(major)).lstrip("/")}/libcupti.so.{major}" -o'
        for t in profile.cupti_candidates
    )
    script = (
        f"f=$(find /usr /opt \\( {paths} -false \\) -print -quit 2>/dev/null); "
        '[ -n "$f" ] && dirname "$f" && sha256sum "$f"'
    )
    try:
        done = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "/bin/sh", image_id, "-c", script],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = done.stdout.split()
    found = Cupti(lines[0], lines[1]) if done.returncode == 0 and len(lines) >= 2 else None
    if done.returncode not in (0, 1):
        return None
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps({"folder": found.folder, "sha256": found.sha256} if found else None)
        )
    except OSError:
        pass
    return found
