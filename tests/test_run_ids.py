import json
import re
from pathlib import Path

import pytest

from tensward.analyse import tried_args
from tensward.errors import PreflightError
from tensward.files import new_run_id
from tensward.reuse import measured_at
from tensward.runids import resolve_run


def _run(runs: Path, name: str, **record: object) -> Path:
    directory = runs / name
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(json.dumps(record))
    (directory / "metrics.json").write_text("{}")
    return directory


def test_run_ids_sort_resolve_and_leave_legacy_projects_alone(tmp_path: Path) -> None:
    runs = tmp_path / "numbered" / "runs"
    assert new_run_id(runs) == "0001-baseline"
    _run(runs, "0001-baseline", engine_args=[])
    assert new_run_id(runs, change=["max-num-seqs=48"]) == "0002-max-num-seqs-48"
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{4}", new_run_id())
    _run(runs, "0002-max-num-seqs-48", engine_args=["max-num-seqs=48"])
    project = runs.parent
    assert resolve_run(project, "last") == "0002-max-num-seqs-48"
    assert resolve_run(project, "baseline") == "0001-baseline"
    assert resolve_run(project, "2") == "0002-max-num-seqs-48"
    with pytest.raises(PreflightError):
        resolve_run(project, "9")

    legacy = tmp_path / "legacy"
    old = _run(legacy / "runs", "20261005T073854Z-ea08")
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{4}", new_run_id(legacy / "runs"))
    assert measured_at(old) == 1791185934
    assert resolve_run(legacy, "last") == "20261005T073854Z-ea08"


def test_run_id_names_the_change_tried_not_the_kept_setup(tmp_path: Path) -> None:
    kept = ["n-gpu-layers=all", "n-cpu-moe=17"]
    (tmp_path / "project.json").write_text(json.dumps({"snapshot_id": "s"}))
    (tmp_path / "kept.json").write_text(json.dumps({
        "run": "0001-n-gpu-layers-all", "engine_args": kept, "answer_changes": [],
        "previous": "registered", "snapshot_id": "s",
    }))  # fmt: skip
    runs = tmp_path / "runs"
    _run(runs, "0001-n-gpu-layers-all", engine_args=kept)
    assert new_run_id(runs, change=tried_args(tmp_path, [*kept, "threads=16"])) == "0002-threads-16"
    assert new_run_id(runs, change=tried_args(tmp_path, kept)) == "0002-n-gpu-layers-all"


def test_run_id_leaves_out_the_flags_the_baseline_launch_pinned(tmp_path: Path) -> None:
    pinned = {"n-gpu-layers=41", "n-cpu-moe=18"}
    runs = tmp_path / "runs"
    _run(runs, "0001-baseline", engine_args=[])
    args = ["n-gpu-layers=41", "n-cpu-moe=18", "threads=16"]
    assert new_run_id(runs, change=tried_args(tmp_path, args, pinned)) == "0002-threads-16"
    assert new_run_id(runs, change=tried_args(tmp_path, args[:2], pinned)) == "0002-n-gpu-layers-41"
