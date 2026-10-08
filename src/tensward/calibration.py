"""Calibration profiles: the threshold levels for one engine on one kind of hardware, and which
of them were calibrated there. Selection: the engine's default when the platform, GPU or version
is unknown; else an extension's profile naming the GPU; else the public one; else none
(uncalibrated). Past the default, a profile is taken only for the engine versions it names."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from functools import cache
from importlib.resources import files
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    from .engines.protocol import Engine

PROFILES_FILE = "data/calibration.toml"


@dataclass(frozen=True, slots=True, kw_only=True)
class CalibrationProfile:
    id: str
    engine: str
    platform: str  # Platform.name: "nvidia", "apple", "cpu"
    devices: tuple[str, ...]  # the devices it was calibrated on
    versions: str  # engine versions it holds for: "*" (every version) or ">=0.30,<0.31"
    placement: str = "device"  # where the weights are: "device", "offload" or "host"
    # Threshold name -> (warning, critical)
    levels: Mapping[str, tuple[float, float]] = field(hash=False)
    calibrated: frozenset[str]  # threshold names calibrated in this profile
    calibrated_on: str


UNCALIBRATED = CalibrationProfile(
    id="uncalibrated", engine="", platform="", devices=(), versions="*", levels={},
    calibrated=frozenset(), calibrated_on="",
)  # fmt: skip
"""No matching profile: the default levels (thresholds.py), every threshold uncalibrated."""


def _profile(table: Mapping[str, Any]) -> CalibrationProfile:
    return CalibrationProfile(
        id=table["id"],
        engine=table["engine"],
        platform=table["platform"],
        devices=tuple(table["devices"]),
        versions=table["versions"],
        placement=table.get("placement", "device"),
        levels={name: (float(w), float(c)) for name, (w, c) in table["levels"].items()},
        calibrated=frozenset(table["calibrated"]),
        calibrated_on=table["calibrated_on"],
    )


@cache
def public_profiles() -> dict[str, CalibrationProfile]:
    """The profiles shipped in ``data/calibration.toml``, by id."""
    document = tomllib.loads(files("tensward").joinpath(PROFILES_FILE).read_text("utf-8"))
    return {profile.id: profile for profile in map(_profile, document["profile"])}


def on_devices(gpu: str, devices: Sequence[str]) -> bool:
    """Whether the device name ``gpu`` ("NVIDIA L4") is one of ``devices`` ("L4")."""
    return bool(set(devices) & set(re.split(r"[^A-Za-z0-9]+", gpu)))


def _holds_for(profile: CalibrationProfile, version: str | None) -> bool:
    """Whether ``profile`` holds for engine ``version``: its ``versions`` is "*" or a PEP 440
    specifier set (">=0.30,<0.31") the version satisfies, or the version is unknown. An
    unparseable specifier or version holds for none."""
    if profile.versions == "*" or version is None:
        return True
    try:
        return SpecifierSet(profile.versions).contains(Version(version), prereleases=True)
    except (InvalidSpecifier, InvalidVersion):
        return False


def profile_for(
    engine: Engine,
    platform: str | None,
    gpu: str | None,
    version: str | None,
    placement: str = "device",
) -> CalibrationProfile | None:
    """The profile a run is judged with: the engine's default when the platform, GPU or version
    is unknown (never for a run off the device or on a CPU), else an installed extension's
    profile naming the GPU, else the public one for this engine (or the engine's default
    profile, so an adapter's subclass keeps its family's profile), platform and placement. Past
    the default, a profile is taken only where its ``versions`` holds for ``version``."""
    unknown = platform is None or gpu is None or version is None
    if unknown and engine.default_profile and placement == "device" and platform != "cpu":
        return public_profiles()[engine.default_profile]
    from .extensions import load_extender

    extender = load_extender()
    for profile in extender.profiles if extender else ():
        if (profile.engine, profile.platform) == (engine.name, platform) and gpu:
            if on_devices(gpu, profile.devices) and _holds_for(profile, version):
                return profile
    return next(
        (
            profile
            for profile in public_profiles().values()
            if (profile.platform, profile.placement) == (platform, placement)
            and (profile.engine == engine.name or profile.id == engine.default_profile)
            and _holds_for(profile, version)
        ),
        None,
    )
