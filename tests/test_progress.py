"""The status line a long wait shows: in place on a terminal, plain lines elsewhere."""

from __future__ import annotations

import io

from tensward.progress import PhaseStarted, RequestsDone, ServerLoading, Status


class Screen(io.TextIOWrapper):
    def __init__(self, tty: bool) -> None:
        self.written = io.BytesIO()
        super().__init__(self.written, encoding="utf-8", newline="")
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty

    def getvalue(self) -> str:
        self.flush()
        return self.written.getvalue().decode()


def test_a_wait_spins_in_place_on_a_terminal_and_prints_plain_lines_elsewhere() -> None:
    loading = ServerLoading(phase="launch", waited_s=0.0, expected_s=110.0, engine="llama.cpp",
                            model_bytes=17_400_000_000)  # fmt: skip
    tty, pipe = Screen(tty=True), Screen(tty=False)
    for screen in (tty, pipe):
        status = Status(screen, environ={"TERM": "xterm"})
        status.take(loading)
        began = status.wait.began if status.wait else 0.0
        status.tick(began + 1)
        status.tick(began + 42)
        status.take(PhaseStarted(phase="measure", detail="measuring: 160 requests"))
        status.take(RequestsDone(phase="measure", done=37, total=160))
        status.tick(began + 200)
        status.close()
    frames = tty.getvalue().split("\r\x1b[2K")
    assert frames[1].endswith(
        "Starting llama.cpp and loading the model (17.4 GB)…  \x1b[2m0:42 of about 2 min\x1b[22m"
    )
    assert "Sending requests… ▕████░░░░░░░░░░░░▏ 37/160  " in frames[-2]
    assert tty.getvalue().endswith("\x1b[?25h")
    lines = pipe.getvalue().splitlines()
    assert "\r" not in pipe.getvalue() and "\x1b" not in pipe.getvalue()
    assert lines[0].endswith("Starting llama.cpp and loading the model (17.4 GB), about 2 min")
    assert lines[1].endswith("loading the model (17.4 GB)... 0:42 of about 2 min")
    assert "] Sending requests (37 of 160)... 3:" in lines[-1]


def test_no_color_and_a_dumb_terminal_keep_the_line_plain() -> None:
    plain = Screen(tty=True)
    status = Status(plain, environ={"TERM": "xterm", "NO_COLOR": "1"})
    status.take(PhaseStarted(phase="compare"))
    status.tick((status.wait.began if status.wait else 0.0) + 3)
    assert "Comparing the answers…  0:03" in plain.getvalue() and "\x1b[2m" not in plain.getvalue()
    assert not Status(Screen(tty=True), environ={"TERM": "dumb"}).spin
