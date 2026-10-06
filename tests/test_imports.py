"""Every module imports on its own, so an import cycle fails here rather than for a user."""

from __future__ import annotations

import pkgutil
import subprocess
import sys

import tensward

MODULES = sorted(m.name for m in pkgutil.walk_packages(tensward.__path__, "tensward."))
# One interpreter: before each import every tensward module is forgotten, so each module is
# imported first, as a fresh interpreter would; pydantic and httpx stay loaded, which is cheap.
FRESH_IMPORTS = """
import importlib, sys
for name in sys.argv[1:]:
    for loaded in [m for m in sys.modules if m == "tensward" or m.startswith("tensward.")]:
        del sys.modules[loaded]
    importlib.import_module(name)
"""


def test_each_module_imports_first() -> None:
    subprocess.run([sys.executable, "-c", FRESH_IMPORTS, *MODULES], check=True)
