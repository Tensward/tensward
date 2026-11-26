"""Where the GPU census runs and, where it does not, why not in the words the user reads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from ..engines.protocol import CensusProfile
from ..platforms.nvidia import compute_capability
from . import library as locator
from .library import CensusLibrary, Cupti

IDLE_S = 2.0
UNPAUSE_WAIT_S = 1.5
CENSUS_SLICE_S = 60.0
FLUSH_WAIT_S = 7.0
CENSUS_CEILING_S = 600.0
EXPECTED_STOP_S = 7.0
SETTLE_S = 10.0
PROBE_S = 10.0
MARGIN_S = 30.0

CENSUS_ENV = {
    "TWCENSUS_FLUSH": "ondemand",
    "TWCENSUS_API": "0",
    "TWCENSUS_SYNC": "1",
    "TWCENSUS_KERNELS": "1",
    "TWCENSUS_MEMCPY": "1",
}
DOCKER_START_S = 5.0
NOT_WANTED = "skipped (--no-census)"
NOT_COVERED = "the GPU census covers vLLM (host and docker) and llama.cpp (host)"
NOT_NVIDIA = "the GPU census needs an NVIDIA GPU"
NOT_INSTALLED = "the GPU census is not installed"
# CUDA 13 dropped Maxwell, Pascal and Volta, so its CUPTI records nothing on them.
CUDA13_MIN_CAPABILITY = 7.5
INSTALL = "Install with: "
INSTALL_END = ". "
FOLDERS_SHOWN = 6
UNSHOWN = (NOT_WANTED, NOT_COVERED, NOT_NVIDIA, NOT_INSTALLED)  # skips a report does not mention


@dataclass(frozen=True, slots=True)
class CensusPlan:
    profile: CensusProfile
    library: CensusLibrary
    cupti: Cupti
    major: int
    docker: bool
    verified: bool
    expected_s: float


@dataclass(frozen=True, slots=True)
class CensusSkip:
    reason: str
    advice: str | None = None


def expected_census_s(
    load_s: float, warmup_s: float, docker: bool, req_s: float | None = None
) -> float:
    """How long a census start takes: the start and warm-up measured on this setup, then the
    fixed sequence. The recording is the time its 16 requests need at the measured rate, from
    one slice to two (where it is cut)."""
    recording = (
        min(2 * CENSUS_SLICE_S, max(CENSUS_SLICE_S, 16 / req_s)) if req_s else CENSUS_SLICE_S
    )
    sequence = (
        IDLE_S + UNPAUSE_WAIT_S + recording + IDLE_S + FLUSH_WAIT_S + EXPECTED_STOP_S + SETTLE_S
    )
    return load_s + warmup_s + sequence + (DOCKER_START_S if docker else 0.0)


def _version(text: str | None) -> Version | None:
    try:
        return Version((text or "").removeprefix("b"))
    except InvalidVersion:
        return None


def _in(version: Version | None, spec: str) -> bool:
    try:
        return version is not None and SpecifierSet(spec).contains(version, prereleases=True)
    except InvalidSpecifier:
        return False


def _no_cupti(engine: Any, major: int | None, docker: bool, folders: list[Path]) -> CensusSkip:
    library = "libcupti.so" if major is None else f"libcupti.so.{major}"
    if docker:
        return CensusSkip(
            f"no CUPTI found in the {engine.label} image",
            f"The census reads the GPU through NVIDIA's CUPTI. Tensward found no {library} in "
            "the image. Use an image that has it, then run tensward analyse again.",
        )
    shown = ", ".join(str(folder) for folder in folders[:FOLDERS_SHOWN]) or "no folder"
    more = len(folders) - FOLDERS_SHOWN
    where = f"{shown} and {more} more" if more > 0 else shown
    return CensusSkip(
        "no CUPTI found",
        f"{INSTALL}{locator.cupti_install(major)}{INSTALL_END}The census reads the GPU through "
        f"NVIDIA's CUPTI. Tensward looked for {library} in {where}, the linker cache and "
        "NVIDIA's pip wheels in every Python environment it found. Or point TENSWARD_CUPTI at "
        "the file, then run tensward analyse again.",
    )


def plan_census(
    engine: Any,
    runtime: Any,
    *,
    version: str | None,
    platform: Any,
    backend: str,
    parallel: int,
    load_s: float,
    warmup_s: float,
    wanted: bool,
    home: Path,
    built_once: bool = False,
    req_s: float | None = None,
) -> CensusPlan | CensusSkip:
    """The census start to run, or why none runs. With ``built_once`` the measured start held
    one-time builds a census start finds cached, so its estimate does not skip it: the hard stop
    inside CENSUS_CEILING_S ends a census start that still runs long."""
    if not wanted:
        return CensusSkip(NOT_WANTED)
    profile: CensusProfile | None = engine.census
    docker = runtime.kind == "docker"
    if profile is None or (docker and profile.docker is None):
        return CensusSkip(NOT_COVERED)
    if getattr(platform, "name", platform) != "nvidia":
        return CensusSkip(NOT_NVIDIA)
    if backend != "cuda":
        return CensusSkip(f"the GPU census needs a CUDA build of {engine.label}")
    if parallel > 1:
        return CensusSkip("the GPU census is not tested with several GPUs")
    parsed = _version(version)
    if profile.known_crash and _in(parsed, profile.known_crash):
        return CensusSkip(
            f"the GPU census is off for {engine.label} {engine.version_prefix}{version}: "
            "census starts crash on it",
            profile.crash_advice,
        )
    if not locator.installed(home):
        return CensusSkip(NOT_INSTALLED, "the GPU census ships for Linux x86_64")
    if docker:
        major: int | None = profile.docker
        image_id = locator.image_id_of(runtime.image)
        cupti = (
            None
            if image_id is None or major is None
            else locator.image_cupti(profile, image_id, major)
        )
        command: tuple[str, ...] = ()
    else:
        command = tuple(runtime.command)
        major = locator.engine_cupti_major(profile, command)
        cupti = None if major is None else locator.local_cupti(profile, command, major)
    if major is None:
        folders = (
            []
            if docker
            else [
                folder
                for guess in (12, 13)
                for folder in locator.cupti_search_dirs(profile, command, guess)
            ]
        )
        return _no_cupti(engine, None, docker, list(dict.fromkeys(folders)))
    if major >= 13 and (cc := compute_capability(getattr(runtime, "gpus", None) or None)):
        if cc < CUDA13_MIN_CAPABILITY:
            return CensusSkip(
                f"the GPU census is not supported on this GPU (compute capability {cc}) with "
                f"CUDA {major}",
                "CUDA 13 dropped Pascal and Volta GPUs; use a CUDA 12 build of the engine for the "
                "census.",
            )
    library = locator.find_library(home, major)
    if library is None:
        return CensusSkip(f"no census library for CUDA {major}")
    if cupti is None:
        folders = [] if docker else locator.cupti_search_dirs(profile, command, major)
        return _no_cupti(engine, major, docker, folders)
    if locator.cupti_too_old(home, cupti, library):
        return CensusSkip("the engine's CUPTI is older than the census build needs")
    expected = expected_census_s(load_s, warmup_s, docker, req_s)
    if expected + MARGIN_S > CENSUS_CEILING_S and not built_once:
        return CensusSkip(
            f"a census start would take about {round((expected + MARGIN_S) / 60)} min here "
            f"(start {load_s:.0f} s, warm-up {warmup_s:.0f} s, then a 1-2 min recording), "
            "over its 10-minute limit, so it was skipped",
            "fewer or shorter warm-up requests (warmup_requests in config.json) bring it "
            "under the limit",
        )
    verified = any(_in(parsed, spec) for spec in profile.tested)
    return CensusPlan(profile, library, cupti, major, docker, verified, expected)
