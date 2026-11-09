import asyncio
from pathlib import Path
from types import SimpleNamespace

from tensward import measure
from tensward.measure import Progress, _watch_server


class BrokenLogServer:
    def __init__(self) -> None:
        self.stopped = False

    def log_text(self) -> str:
        raise OSError("docker logs failed")

    def stop(self) -> None:
        self.stopped = True


def test_a_log_that_cannot_be_read_does_not_end_the_stall_stop(tmp_path: Path):
    server = BrokenLogServer()
    progress = Progress()
    progress.at = progress.started = 0.0
    stalled = asyncio.Event()
    asyncio.run(_watch_server(server, "key", [tmp_path], progress, 1.0, stalled))  # type: ignore[arg-type]
    assert stalled.is_set() and server.stopped


def test_the_log_is_not_rewritten_when_it_has_not_grown(tmp_path: Path):
    server = SimpleNamespace(log_text=lambda: "line\n")
    assert measure._save_log(server, "key", [tmp_path]) == 5  # type: ignore[arg-type]
    (tmp_path / "server.log").unlink()
    assert measure._save_log(server, "key", [tmp_path], 5) == 5  # type: ignore[arg-type]
    assert not (tmp_path / "server.log").exists()
