import asyncio
import json
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


def test_the_pace_keeps_its_rule_where_130_requests_fit_and_the_floor_only_where_not():
    from tensward.context import AnalysisContext, active

    def paced(projected: float, **context: object) -> Progress:
        progress = Progress()
        with active(AnalysisContext(**context)):  # type: ignore[arg-type]
            measure._pace(progress, projected, 160)
        return progress

    assert paced(50 * 60, pace_floor=13).cap == 130
    assert paced(50 * 60).cap == 130
    much_slower = paced(200 * 60, try_step=True)
    assert much_slower.slow == 200 * 60 and much_slower.cap is None
    started = paced(200 * 60, try_step=True, unstarted_parent=True, pace_floor=13)
    assert started.cap == measure.SLOW_SAMPLE
    assert paced(200 * 60, pace_floor=13).cap == 24
    long = paced(160 * 310, pace_floor=13)
    assert long.cap == 13 and long.slow is None
    refused = paced(160 * 15 * 60 * 2, pace_floor=13)
    assert refused.slow is not None and refused.floor_requests == 13


def test_a_step_gets_the_floor_only_from_a_parent_cut_to_it(tmp_path: Path):
    from tensward.cli import _pace_floor

    arrival = SimpleNamespace(concurrency=1, max_inflight=None)
    project = SimpleNamespace(config=SimpleNamespace(workload=SimpleNamespace(arrival=arrival)))
    runs = [SimpleNamespace(run_id="0001-baseline", engine_args=frozenset())]
    (tmp_path / "0001-baseline").mkdir()
    window = tmp_path / "0001-baseline" / "window.json"
    window.write_text("{}")

    def floor(parent: str | None, *args: str) -> int | None:
        return _pace_floor(project, tmp_path, runs, parent, args)  # type: ignore[arg-type]

    assert floor(None) == floor("0001-baseline") == 14
    assert floor("0001-baseline", "ubatch-size=2048") is None
    window.write_text(json.dumps({"floor_cut": {"requests": 37}}))
    assert floor("0001-baseline", "ubatch-size=2048") == 37
