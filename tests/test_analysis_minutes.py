from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from registration_fixtures import registered_project

from tensward.project import declared_workload
from tensward.wizard import analysis_minutes, may_relaunch, run_minutes, workload_plan

GB = 10**9


def _runtime(topology: str, total: int) -> SimpleNamespace:
    platform = SimpleNamespace(
        memory_topology=topology,
        detect=lambda: None,
        select=lambda *_: ([SimpleNamespace(total_bytes=total)], None),
    )
    return SimpleNamespace(platform=platform, gpus=None)


def test_every_screen_gets_the_declared_workloads_minutes(tmp_path: Path) -> None:
    project = registered_project(tmp_path)
    workload = declared_workload(project)
    engine = SimpleNamespace(family="vllm")
    minutes = analysis_minutes(project, 16 * GB, engine, _runtime("discrete", 24 * GB))  # type: ignore[arg-type]
    assert minutes == run_minutes(16 * GB, **workload_plan(workload))


def test_a_mac_run_of_weights_that_do_not_fit_counts_its_second_launch() -> None:
    engine = SimpleNamespace(family="llamacpp")
    assert may_relaunch(engine, _runtime("unified", 16 * GB), 20 * GB)  # type: ignore[arg-type]
    assert not may_relaunch(engine, _runtime("unified", 64 * GB), 20 * GB)  # type: ignore[arg-type]
    assert not may_relaunch(engine, _runtime("host", 16 * GB), 20 * GB)  # type: ignore[arg-type]
    assert not may_relaunch(SimpleNamespace(family="vllm"), _runtime("discrete", GB), 20 * GB)  # type: ignore[arg-type]


def test_serve_names_only_the_flags_the_kept_setup_tuned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensward import kept as kept_module
    from tensward.kept import Kept

    project = registered_project(tmp_path)
    monkeypatch.setattr(kept_module, "pinned_baseline", lambda *_: frozenset({"n-gpu-layers=41"}))
    setup = Kept("0001", ("n-gpu-layers=41", "threads=16"), (), "registered", "snap")
    assert kept_module.tuned_words(project, setup) == "--threads 16"
