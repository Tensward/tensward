"""How long a llama-server start took to load the model, and what kind of start it was."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .startup import MIB

CLASSES = ("warm", "cold_disk", "cold_dio")
WARM_BYTES_PER_S = 2e9  # page-cache reads run well above this, a cold NVMe read below it
_STAMP = re.compile(r"^(\d+)\.(\d{2})\.(\d{3})\.(\d{3}) ")
_LOADED = re.compile(r"srv\s+llama_server: model loaded")
_FILE_SIZE = re.compile(r"print_info: file size\s+= ([\d.]+) (GiB|MiB)")
_LOAD_MODE = re.compile(r"load_tensors: .*\(load_mode = (\w+)\)")
DIRECT = "dio"


@dataclass(frozen=True, slots=True)
class Load:
    seconds: float  # launch to the "model loaded" line
    file_bytes: int | None

    @property
    def mb_per_s(self) -> float | None:
        if self.file_bytes is None or self.seconds <= 0:
            return None
        return self.file_bytes / 1e6 / self.seconds


def _seconds(line: str) -> float | None:
    found = _STAMP.match(line)
    if found is None:
        return None
    minutes, seconds, millis, micros = (int(g) for g in found.groups())
    return minutes * 60 + seconds + millis / 1e3 + micros / 1e6


def load(log: str) -> Load | None:
    """The load time and the model file's size; None when the log shows no loaded model."""
    seconds, size = None, None
    for line in log.splitlines():
        stamp = _seconds(line)
        if stamp is not None and _LOADED.search(line):
            seconds = stamp
        if size is None and (found := _FILE_SIZE.search(line)):
            size = int(float(found[1]) * (2**30 if found[2] == "GiB" else MIB))
    return None if seconds is None else Load(seconds, size)


def load_mode(log: str) -> str | None:
    found = next((m for line in log.splitlines() if (m := _LOAD_MODE.search(line))), None)
    return found[1] if found else None


def start_class(log: str) -> str | None:
    """``cold_dio`` for a direct-I/O load, else ``warm`` when the weights came in faster than
    ``WARM_BYTES_PER_S``, else ``cold_disk``; None when the log cannot say."""
    if load_mode(log) == DIRECT:
        return "cold_dio"
    loaded = load(log)
    if loaded is None or loaded.file_bytes is None:
        return None
    return "warm" if loaded.seconds < loaded.file_bytes / WARM_BYTES_PER_S else "cold_disk"
