"""The serve-source format: a menu of named settings that ``tensward serve start --from``
serves. Any extension may write one as ``<project>/optimize/<id>/packages.json``."""

from __future__ import annotations

from typing import Literal

from pydantic import JsonValue

from .contracts import Identifier, StrictModel

PACKAGES_FILE = "packages.json"


class Package(StrictModel):
    name: Identifier
    settings: dict[str, JsonValue]  # a Settings as JSON (Settings.from_json reads it)
    requires_review: bool  # its changes can affect output quality
    confirmed: bool  # it beat the current setup again when measured a second time


class PackagesFile(StrictModel):
    schema_version: Literal["1"]
    recommended: str | None
    packages: tuple[Package, ...]
