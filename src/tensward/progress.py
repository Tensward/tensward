"""Phase lines for the terminal: what a long command is doing, with the seconds since it began.

They go to standard error, so the command's own output stays clean for pipes and scripts.
"""

from __future__ import annotations

import sys
import time
from typing import Callable

_START = time.monotonic()


def say(text: str) -> None:
    print(f"[{time.monotonic() - _START:5.0f}s] {text}", file=sys.stderr, flush=True)


def quarters(label: str, total: int) -> Callable[[int], None]:
    """A callback for "this many requests are done" that says so at 25, 50, 75 and 100%."""
    marks = {total * quarter // 4: quarter * 25 for quarter in (1, 2, 3, 4)}

    def done(count: int) -> None:
        if count in marks:
            say(f"{label}: {count}/{total} requests ({marks[count]}%)")

    return done
