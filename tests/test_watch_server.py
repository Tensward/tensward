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

    def is_running(self) -> bool:
        return not self.stopped


def test_a_log_that_cannot_be_read_does_not_end_the_stall_stop(tmp_path: Path):
    server = BrokenLogServer()
    progress = Progress()
    progress.at = progress.started = 0.0
    stalled = asyncio.Event()
    asyncio.run(_watch_server(server, "key", [tmp_path], progress, 1.0, stalled))  # type: ignore[arg-type]
    assert stalled.is_set() and server.stopped and not progress.exited


def test_a_server_that_aborts_mid_run_is_named_as_crashed_alive_or_not(tmp_path: Path):
    """L27 try-01: llama.cpp aborted, stayed alive writing its core, and try said nothing
    streamed back in 300 s."""
    log = (
        "slot update\n/__w/llama.cpp/common/common.cpp:1634: failed to remove sequence 1\n\n"
        "/opt/llama/libggml-base.so.0(+0x16a8b)[0x77c544d16a8b]\n"
        "/opt/llama/libggml-base.so.0(ggml_abort+0x15b)[0x77c544d170eb]\n"
    )
    for running in (False, True):
        server = SimpleNamespace(
            log_text=lambda: log, is_running=lambda: running, stop=lambda: None
        )
        progress = Progress()
        progress.at = progress.started = 1e12  # nothing idle yet
        stalled = asyncio.Event()
        watch = _watch_server(server, "key", [tmp_path], progress, 300.0, stalled, aborts=True)  # type: ignore[arg-type]
        asyncio.run(watch)
        assert stalled.is_set() and progress.exited
        failure = measure._stalled(server, "key", 300.0, progress)  # type: ignore[arg-type]
        assert str(failure).startswith(
            "the engine crashed: /__w/llama.cpp/common/common.cpp:1634: failed to remove "
            "sequence 1\n"
        )


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


def test_on_apple_the_warm_up_pace_says_how_long_the_requests_take():
    """X1 (Mac): the estimate said about 6 min and the analysis took 27."""
    workload = SimpleNamespace(arrival=SimpleNamespace(concurrency=4), request_count=63)
    project = SimpleNamespace(config=SimpleNamespace(warmup_requests=4, workload=workload))
    assert measure.warm_pace(project, 120.0) == (  # type: ignore[arg-type]
        "Warm-up took 120 s; at that pace the measured requests take about 16 min more"
    )
