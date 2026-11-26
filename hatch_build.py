"""The Linux x86_64 wheel carries the census library; every other wheel stays pure Python."""

import os
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

PLATFORM_WHEEL_ENV = "TENSWARD_PLATFORM_WHEEL"
PLATFORM_TAG = "py3-none-manylinux_2_28_x86_64"
LIBRARY_DIR = "src/tensward/census/lib"


class CensusLibraryHook(BuildHookInterface):  # type: ignore[misc]
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel" or not os.environ.get(PLATFORM_WHEEL_ENV):
            return
        folder = Path(self.root) / LIBRARY_DIR
        if not (folder / "manifest.json").is_file():
            raise RuntimeError(f"{PLATFORM_WHEEL_ENV} is set but {folder} has no manifest.json")
        build_data["pure_python"] = False
        build_data["tag"] = PLATFORM_TAG
        build_data["force_include"][str(folder)] = "tensward/census/lib"
