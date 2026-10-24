"""A stand-in for an engine launcher: ``launcher.py [--env NAME=VALUE]... -- COMMAND...``.

It appends its argv and pid to ``LAUNCHER_RECORD`` and constructs the child's environment
from the pairs plus a minimal PATH and locale. With ``LAUNCHER_BUSY_MARKER``, it stays as
the child's parent and marks a 20-second busy period on stop. A separately owned
``--busy-keeper FILE`` process releases that marker; the next launch waits for it.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def keep_busy(marker: Path) -> None:
    print("ready", flush=True)
    while True:
        if marker.exists() and time.monotonic() >= json.loads(marker.read_text())["release_at"]:
            marker.unlink()
        time.sleep(0.05)


def launch(arguments: list[str]) -> int:
    pairs = []
    while arguments and arguments[0] != "--":
        assert arguments.pop(0) == "--env"
        pairs.append(arguments.pop(0))
    command = arguments[1:]
    env = {"PATH": os.defpath, "LANG": "C.UTF-8", **dict(pair.split("=", 1) for pair in pairs)}
    busy = os.environ.get("LAUNCHER_BUSY_MARKER")
    waited = None
    previous = None
    if busy:
        marker = Path(busy)
        if marker.exists():
            previous = json.loads(marker.read_text())
            waited = time.monotonic()
            while marker.exists():
                time.sleep(0.05)
        child = subprocess.Popen(command, env=env)
        stopping = 0

        def stop(sig: int, _frame: object) -> None:
            nonlocal stopping
            stopping = sig
            child.send_signal(sig)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
    with open(os.environ["LAUNCHER_RECORD"], "a") as record:
        record.write(
            json.dumps(
                {
                    "env": pairs,
                    "command": command,
                    "pid": os.getpid(),
                    "child_pid": child.pid if busy else os.getpid(),
                    "wait_started_at": waited,
                    "busy": previous,
                    "launched_at": time.monotonic(),
                }
            )
            + "\n"
        )
    if not busy:
        os.execvpe(command[0], command, env)
    try:
        result = child.wait()
        if stopping:
            stopped = time.monotonic()
            temporary = marker.with_suffix(".tmp")
            temporary.write_text(json.dumps({"stopped_at": stopped, "release_at": stopped + 20}))
            temporary.replace(marker)
            return 128 + stopping
        return result
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


if __name__ == "__main__":
    if sys.argv[1:2] == ["--busy-keeper"]:
        keep_busy(Path(sys.argv[2]))
    else:
        sys.exit(launch(sys.argv[1:]))
