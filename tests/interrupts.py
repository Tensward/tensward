"""Interrupting a command the way a user does: a signal while it waits for the engine."""

from __future__ import annotations

import os
import signal
import threading
import time
from pathlib import Path


def signal_once_present(path_glob: tuple[Path, str], sig: signal.Signals) -> None:
    """Send ``sig`` to this process shortly after a file matching the glob appears."""

    def wait_then_send() -> None:
        directory, pattern = path_glob
        for _ in range(200):
            if next(directory.glob(pattern), None) is not None:
                time.sleep(0.3)
                os.kill(os.getpid(), sig)
                return
            time.sleep(0.05)

    threading.Thread(target=wait_then_send, daemon=True).start()
