"""Copied into a test's site dir as ``sitecustomize``: process detection sees only the processes
started under the test (``TW_TEST_ROOT_PID``), not other servers running on the machine."""

import importlib.abc
import importlib.util
import os
import sys

ROOT = int(os.environ.get("TW_TEST_ROOT_PID", "0"))


def _own(rows):
    parent = {pid: ppid for pid, ppid, _ in rows}

    def under_root(pid):
        while pid > 1:
            if pid == ROOT:
                return True
            pid = parent.get(pid, 0)
        return False

    return [row for row in rows if under_root(row[0])]


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "tensward.detect":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            everything = module._processes
            module._processes = lambda: _own(everything())

        spec.loader.exec_module = exec_module
        return spec


if ROOT:
    sys.meta_path.insert(0, _Finder())
