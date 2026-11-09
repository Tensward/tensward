"""``tensward try``: when a step is kept, and which offers are left."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from registration_fixtures import registered_project

from tensward.cli_try import NothingToTry, _unmatched, judge_step, offers


def step(*, recommended: bool = True, broken: int = 0, calls: tuple[int, int] | None = None):
    return {
        "metric": "req_s",
        "before": 4.0,
        "after": 5.0,
        "change": 0.25,
        "band": 0.03,
        "recommended": recommended,
        "answers": {"identical": 40, "differ": 0, "broken": broken, "total": 40},
        "tool_calls": None
        if calls is None
        else {
            "before": {"requests": 40, "calling": 40, "valid_json": calls[0] / 40},
            "after": {"requests": 40, "calling": 40, "valid_json": calls[1] / 40},
        },
    }


def test_try_keeps_only_a_better_change_with_passing_answers(tmp_path: Path) -> None:
    assert judge_step(step(), tools_on=False).keep
    assert judge_step(None, tools_on=False).gate == "no comparison"
    assert judge_step(step(recommended=False), tools_on=False).gate == ("better")
    assert judge_step(step(broken=1), tools_on=False).gate == "answers"
    assert judge_step(step(calls=(40, 36)), tools_on=True).gate == ("tool calls")
    assert judge_step(step(calls=(40, 40)), tools_on=True).keep
    # qwen38 27B Q4, try parallel=8: valid JSON 15/16 -> 12/14 is 1 more failing call out of
    # 14 calls, too few to judge: kept, with the manual-check note
    real = {
        "before": {"requests": 160, "calling": 16, "valid_json": 0.9375},
        "after": {"requests": 130, "calling": 14, "valid_json": 0.8571428571428571},
    }
    verdict = judge_step({**step(), "tool_calls": real}, tools_on=True)
    assert verdict.keep
    assert "only 14 calls, too few to judge; check them yourself" in verdict.rows[-1][2]

    # GO-run2, qwen3.5 35B-A3B Q8, try threads=16: 19 of 64 answers identical is not kept
    threads = {
        **step(),
        "before": 0.2033,
        "after": 0.2118,
        "change": 0.0505,
        "band": 0.0416,
        "answers": {"identical": 19, "differ": 45, "broken": 0, "total": 64},
    }
    refused = judge_step(threads, tools_on=False)
    assert refused.gate == "answers" and not refused.keep

    with pytest.raises(NothingToTry, match="run tensward analyse first"):
        offers(registered_project(tmp_path))

    project = tmp_path / "project"
    snapshot = json.loads((project / "project.json").read_text())["snapshot_id"]
    for name, args, command in (
        ("0001-baseline", [], "--engine-arg max-num-seqs=4"),
        ("0002-max-num-seqs-4", ["max-num-seqs=4"], "--engine-arg max-num-seqs=4 --engine-arg x=1"),
    ):
        run = project / "runs" / name
        run.mkdir(parents=True)
        document = {"schema_version": "1", "snapshot_id": snapshot, "engine_args": args}
        document |= {"settings": {}, "retained_responses": True, "temperature": 0.0}
        document |= {"runtime": "local", "gpus": None, "image": None}
        (run / "run.json").write_text(json.dumps(document))
        (run / "metrics.json").write_text("{}")
        block = {"kind": "suggestion", "name": name, "reason": f"because {name}"}
        block |= {"command": f"tensward analyse {command}", "changes_answers": False}
        block |= {"engine_args": command.split()[1::2], "engine_env": []}
        report = {"sections": [{"id": "s", "blocks": [block]}]}
        (run / "report.json").write_text(json.dumps(report))
    found = offers(project)
    assert [offer.name for offer in found] == ["0001-baseline"]


def test_a_setup_that_does_not_start_prints_a_line_and_is_not_offered_again(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tensward.cli_try import Offer, _start_failed
    from tensward.rejected import read_rejected

    project = registered_project(tmp_path)
    capsys.readouterr()
    offer = Offer(1, "fewer-sequences", ("max-num-seqs=8",), (), "why", None, True, first=True)
    oom = "the server exited before it was ready: the GPU ran out of memory (`Tried to allocate`)"
    code = _start_failed(
        project, True, offer, frozenset(), "--max-num-seqs 8", "0001-max-num-seqs-8",
        f"{oom}\n(APIServer pid=1) Traceback (most recent call last):", False,
    )  # fmt: skip
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert code == 1 and result["kept"] is False and result["error"] == oom
    assert "Traceback" not in captured.out + captured.err
    (rejection,) = read_rejected(project)
    assert rejection.change == ("max-num-seqs=8",)


@pytest.mark.parametrize(
    ("cause", "said"),
    [
        (  # clean Ubuntu 24.04, L4, vLLM 0.33: torch inductor found no C compiler
            "torch._inductor.exc.InductorError: RuntimeError: Failed to find C compiler. Please "
            "specify via CC environment variable or set triton.knobs.build.impl.",
            "this machine has no C compiler, which vLLM needs to start. "
            "Fix: sudo apt install -y build-essential, then run the command again.",
        ),
        (  # clean Ubuntu 24.04, L4, vLLM 0.30: FlashInfer's JIT found no nvcc
            "RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't "
            "exist",
            "this machine has no CUDA compiler (nvcc), which vLLM needs to start. Fix: install "
            "NVIDIA's CUDA toolkit (see the README's install section), then run the command "
            "again.",
        ),
        (  # clean Ubuntu 24.04, L4, vLLM 0.30: FlashInfer's JIT found no ninja
            "FileNotFoundError: [Errno 2] No such file or directory: 'ninja'",
            "this machine has no ninja build tool, which vLLM needs to start. "
            "Fix: sudo apt install -y ninja-build, then run the command again.",
        ),
    ],
)
def test_a_missing_compiler_names_the_fix_and_keeps_the_change_on_offer(
    cause: str, said: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tensward.cli_try import Offer, _start_failed
    from tensward.engines.vllm.engine import VLLM
    from tensward.rejected import read_rejected
    from tensward.runtime import exit_reason

    log = (
        f"(EngineCore pid=9) ERROR 11-07 13:57:25 [core.py:1584] {cause}\n"
        "(APIServer pid=1) Traceback (most recent call last):\n"
        "RuntimeError: Engine core initialization failed. See root cause above.\n"
    )
    reason = exit_reason(log, VLLM)
    assert reason == said
    project = registered_project(tmp_path)
    capsys.readouterr()
    offer = Offer(1, "shorter-context", ("max-model-len=21379",), (), "why", None, True, first=True)
    error = f"the setup with these engine arguments did not run: {reason}"
    _start_failed(project, False, offer, frozenset(), "--max-model-len 21379", "", error, False)
    err = capsys.readouterr().err
    assert err.startswith(f"Not tested: T{reason[1:]}\nNothing changed.")
    assert "Traceback" not in err and "reject --clear" not in err
    assert not read_rejected(project)


def test_try_after_a_report_with_no_change_shows_the_menu_and_succeeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensward.cli import main

    monkeypatch.delenv("TENSWARD_JSON")

    # clean L4 llama.cpp leg: the baseline's report offered nothing for throughput
    project = registered_project(tmp_path)
    snapshot = json.loads((project / "project.json").read_text())["snapshot_id"]
    run = project / "runs" / "0001-baseline"
    run.mkdir(parents=True)
    document = {"schema_version": "1", "snapshot_id": snapshot, "engine_args": []}
    document |= {"settings": {}, "retained_responses": True, "temperature": 0.0}
    document |= {"runtime": "local", "gpus": None, "image": None}
    (run / "run.json").write_text(json.dumps(document))
    (run / "metrics.json").write_text("{}")
    (run / "report.json").write_text(json.dumps({"sections": []}))
    capsys.readouterr()
    assert main(["try", "--project", str(project), "--yes"]) == 0
    assert capsys.readouterr().out == (
        "Nothing to try: the last run found no change worth testing for your goal.\n"
        "Next:\n"
        "  tensward serve        serve your current setup as it is\n"
    )


def test_a_first_run_with_few_tool_calls_is_not_passed(tmp_path: Path) -> None:
    from tensward.cli_try import _first_verdict

    tool_calls = {"requests": 130, "calling": 2, "valid_json": 1.0, "known_tool": 1.0}
    (tmp_path / "metrics.json").write_text(
        json.dumps({"succeeded": 130, "failed": 0, "tool_calls": tool_calls}), "utf-8"
    )
    for name in ("requests.jsonl", "responses.jsonl"):
        (tmp_path / name).write_text("", "utf-8")
    _, _, line, mark = _first_verdict(tmp_path)[0].rows[-1]
    assert "of 2 calls" in line and "too few to judge" in line and mark == ""


def test_a_refusal_before_launch_is_recorded_too(tmp_path: Path) -> None:
    from tensward.cli_try import Offer, _start_failed
    from tensward.rejected import read_rejected

    project = registered_project(tmp_path)
    offer = Offer(1, "fewer-sequences", ("max-num-seqs=8",), (), "why", None, True, first=True)
    _start_failed(
        project, True, offer, frozenset(), "--max-num-seqs 8", "", "cannot start: x", False
    )
    (rejection,) = read_rejected(project)
    assert rejection.change == ("max-num-seqs=8",) and rejection.run.startswith("unstarted-")


def test_a_run_that_stopped_midway_says_it_did_not_finish_and_ids_never_collide(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tensward.cli_try import Offer, _start_failed
    from tensward.rejected import read_rejected

    project = registered_project(tmp_path)
    offer = Offer(1, "fewer-sequences", ("max-num-seqs=8",), (), "why", None, True, first=True)
    other = Offer(1, "more-sequences", ("max-num-seqs=64",), (), "why", None, True, first=True)
    capsys.readouterr()
    _start_failed(project, False, offer, frozenset(), "--max-num-seqs 8", "", "too slow", True)
    assert "did not finish: too slow" in capsys.readouterr().err
    _start_failed(project, False, other, frozenset(), "--max-num-seqs 64", "", "x", False)
    assert "would not start: x" in capsys.readouterr().err
    assert len({r.run for r in read_rejected(project)}) == 2


def test_every_step_ends_in_the_same_next_menu() -> None:
    from tensward.cli import _bare_serve
    from tensward.cli_try import next_choices, next_lines

    reworded = next_choices("--x 1", reworded=True, after_keep="--y 2")
    assert reworded == [
        ("tensward try --keep", "keep it, then try --y 2"),
        ("tensward try", "skip it and try --x 1"),
        ("tensward serve", "stop tuning and serve what's kept now"),
    ]
    last = dict(next_choices(None, reworded=True))["tensward try --keep"]
    assert last == "keep it; nothing else to try after it"
    kept = [c for c, _ in next_choices("--x 1", kept=True)]
    assert kept == ["tensward try", "tensward undo", "tensward serve"]
    assert [c for c, _ in next_choices("--x 1")] == ["tensward try", "tensward serve"]
    done = next_lines(next_choices(None, kept=True), more=False)
    assert done[0] == "No more changes to try." and "tensward try " not in "".join(done)
    assert _bare_serve(["serve", "--project", "p"]) == [
        "serve", "start", "--from", "current", "--project", "p",
    ]  # fmt: skip
    assert _bare_serve(["serve", "stop"]) == ["serve", "stop"]


TTFT_LOSS = (
    "Tensward does not recommend keeping this change: a loss: TTFT p95 after the ramp +72% ±34%."
)


def test_a_not_better_step_says_in_plain_words_what_moved_and_whether_it_is_real() -> None:
    from tensward.cli_try import _gate_text

    # GO-run2, qwen3.5 35B-A3B Q8, try threads=16: throughput +6% within noise, TTFT +72% worse
    threads = {**step(recommended=False), "before": 0.186, "after": 0.193, "change": 0.052}
    threads |= {"band": 0.0558, "reason": TTFT_LOSS}
    verdict = judge_step(threads, tools_on=False)
    nothing = (
        "Nothing changed significantly: differences this small also happen between two "
        "identical runs."
    )
    assert verdict.lines[0] == nothing
    assert _gate_text("better", threads) == (
        "No significant change in throughput, but the wait for the first word got 72% worse."
    )
    held_floor = judge_step({**threads, "recommended": True, "reason": None}, tools_on=False)
    assert (
        held_floor.lines[0] == "Requests/s: no significant change; it held your floor and limits."
    )
    real = judge_step({**step(), "band": 0.04}, tools_on=False)
    assert real.lines[0] == "Requests/s: +25%: a real gain."
    within = {**threads, "reason": None}
    assert _gate_text("better", within) == nothing
    for text in (verdict.lines[0], _gate_text("better", threads)):
        assert "±" not in text and not {"ramp", "req_s", "closed_loop", "band"} & set(text.split())


def test_reworded_answers_are_not_marked_failed_and_say_it_is_the_users_call() -> None:
    from tensward.cli_try import _gate_text

    reworded = {
        **step(),
        "answers": {"identical": 18, "differ": 46, "broken": 0, "total": 64},
    }
    verdict = judge_step(reworded, tools_on=False)
    assert verdict.gate == "answers"
    assert verdict.rows[0] == ("Answers", "", "18 of 64 identical", "your call")
    assert (
        _gate_text("answers", reworded) == "46 of 64 answers are worded differently (none wrong)."
    )
    broken = judge_step(step(broken=3), tools_on=False)
    assert broken.rows[0] == ("Answers", "", "3 of 40 broken", "✘")


def test_a_try_names_only_the_change_and_says_once_what_it_sits_on() -> None:
    from tensward.cli_try import Offer, _on_top, _words

    offer = Offer(
        1, "threads", ("n-gpu-layers=41", "n-cpu-moe=18", "threads=16"), (), "why", None, True
    )
    held = frozenset({"n-gpu-layers=41", "n-cpu-moe=18"})
    assert _words(offer.engine_args, offer.engine_env, held) == "--threads 16"
    assert (
        _on_top(offer, held) == " (on top of your current setup: --n-gpu-layers 41 --n-cpu-moe 18)"
    )
    assert _on_top(offer, frozenset()) == ""


def test_flags_are_spelled_as_typed_never_as_key_value() -> None:
    from tensward.kept import flag_words

    assert (
        flag_words(["threads-batch=16", "no-mmap", "jinja"])
        == "--threads-batch 16 --no-mmap --jinja"
    )
    assert flag_words(["a=1"], ["CUDA_VISIBLE_DEVICES=0"]) == "--a 1 CUDA_VISIBLE_DEVICES=0"
    assert flag_words(["CUDA_VISIBLE_DEVICES=0"]) == "CUDA_VISIBLE_DEVICES=0"


def test_the_announced_time_follows_the_pace_of_the_last_run(tmp_path: Path) -> None:
    from tensward.cli_try import Offer, expected_minutes

    project = registered_project(tmp_path)
    run = project / "runs" / "0001-baseline"
    run.mkdir(parents=True)
    # GO-run2: 160 requests at 0.18 req/s
    rows = [
        {"request_id": f"w:{i}", "dispatch_ns": int(i * 5.4e9), "outcome": "success",
         "first_content_ns": int(i * 5.4e9 + 3e9), "terminal_ns": int(i * 5.4e9 + 21e9),
         "committed_output_tokens": 100}
        for i in range(160)
    ]  # fmt: skip
    (run / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "run.json").write_text("{}")
    (run / "metrics.json").write_text("{}")
    (run / "window.json").write_text(json.dumps({"startup": {"ready_s": 16.0}}))
    offer = Offer(1, "x", ("a=1",), (), "why", None, False)
    # the registered workload sends 10 requests: 16 s to start + (10 + 8 warm-up) x 5.4 s
    assert expected_minutes(project, offer, "throughput", 10**9, fallback=29) == 2
    assert expected_minutes(tmp_path / "nowhere", offer, "throughput", 10**9, fallback=29) == 29


def test_the_announced_time_keeps_the_measured_start_when_no_window_can_be_planned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensward import history
    from tensward.cli_try import Offer, expected_minutes

    project = registered_project(tmp_path)
    run = project / "runs" / "0001-baseline"
    run.mkdir(parents=True)
    rows = [
        {"request_id": f"w:{i}", "dispatch_ns": int(i * 5.4e9), "outcome": "success",
         "first_content_ns": int(i * 5.4e9 + 3e9), "terminal_ns": int(i * 5.4e9 + 21e9),
         "committed_output_tokens": 100}
        for i in range(160)
    ]  # fmt: skip
    (run / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for name in ("run.json", "metrics.json"):
        (run / name).write_text("{}")
    (run / "window.json").write_text(json.dumps({"startup": {"ready_s": 8.0}}))

    def unplannable(*args: object, **kwargs: object) -> None:
        raise KeyError("plan")

    monkeypatch.setattr(history, "planned_window", unplannable)
    offer = Offer(1, "x", ("a=1",), (), "why", None, False)
    # GO-run2, 36 GB of weights loaded in 8 s at 5.4 s a request: the cold-load rule (276 s)
    # announced 22 min, the run's own start gives (8 + 168 x 5.4) / 60
    assert expected_minutes(project, offer, "throughput", 36 * 10**9, fallback=29) == 16


def _kept(project: Path, run: str, args: list[str], history: list[dict]) -> None:
    snapshot = json.loads((project / "project.json").read_text())["snapshot_id"]
    document = {"run": run, "engine_args": args, "answer_changes": [], "previous": "registered"}
    document |= {"snapshot_id": snapshot, "kept_history": history}
    (project / "kept.json").write_text(json.dumps(document))


def test_undo_says_what_was_undone_in_flags_and_ends_in_a_menu(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from argparse import Namespace

    from tensward.cli_try import run_undo

    project = registered_project(tmp_path)
    _kept(project, "0003-threads-batch-16", ["threads-batch=16"], [])
    capsys.readouterr()
    assert run_undo(Namespace(project=project)) == 0
    out = capsys.readouterr().out
    assert out.startswith("Undone: --threads-batch 16. Back to your starting setup.\n")
    assert "keeps its settings until you run tensward serve again" in out
    assert "Next:" in out and "tensward serve" in out and "--engine-arg" not in out


def test_undo_to_a_starting_setup_that_cannot_start_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from argparse import Namespace

    from tensward import cli_try
    from tensward.cli_try import Offer, run_undo

    project = registered_project(tmp_path)
    _kept(project, "0001-n-gpu-layers-all", ["n-gpu-layers=all"], [])
    fit = Offer(1, "fit", ("n-gpu-layers=all",), (), "why", None, True, first=True)
    monkeypatch.setattr(cli_try, "offers", lambda _project: [fit])
    capsys.readouterr()
    assert run_undo(Namespace(project=project)) == 0
    assert (
        "Back to your starting setup (your starting setup does not fit this GPU)."
        in capsys.readouterr().out
    )


def test_runs_prints_a_table_and_keeps_the_json_under_the_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from argparse import Namespace

    from tensward.cli_options import JSON_OUTPUT
    from tensward.cli_try import run_runs
    from tensward.rejected import reject_failed_start

    project = registered_project(tmp_path)
    snapshot = json.loads((project / "project.json").read_text())["snapshot_id"]
    for name, args, parent in (
        ("0001-baseline", [], None),
        ("0002-threads-16", ["threads=16"], "0001-baseline"),
    ):
        run = project / "runs" / name
        run.mkdir(parents=True)
        document = {"schema_version": "1", "snapshot_id": snapshot, "engine_args": args}
        document |= {"settings": {}, "retained_responses": True, "temperature": 0.0}
        document |= {"runtime": "local", "gpus": None, "image": None}
        (run / "run.json").write_text(json.dumps(document))
        (run / "window.json").write_text(json.dumps({"parent": parent}))
        if parent:
            data = {
                "kind": "data",
                "key": "step",
                "metric": "req_s",
                "change": 0.06,
                "recommended": False,
            }
            data |= {"answers": {"identical": 19, "differ": 45, "broken": 0, "total": 64}}
            (run / "report.json").write_text(
                json.dumps({"sections": [{"id": "s", "blocks": [data]}]})
            )
    reject_failed_start(project, "0002-threads-16", {"threads=16"}, "slow")
    JSON_OUTPUT.set(False)
    capsys.readouterr()
    assert run_runs(Namespace(project=project, json=False)) == 0
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("0002"))
    assert "--threads 16" in row and "not kept" in row and "throughput +6%" in row
    assert "45 of 64 reworded" in row and "Next:" in out and "reverted" not in out
    assert run_runs(Namespace(project=project, json=True)) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [r["state"] for r in listed] == ["", "not kept"]


def test_the_measured_answers_gate_a_step_whatever_the_record_says() -> None:
    # leg A, qwen3.8 27B int4: first-word p95 36.5 s -> 0.31 s, but 10 of 96 answers identical
    leg_a = {
        **step(),
        "metric": "ttft_p95_ms",
        "before": 36500.0,
        "after": 310.0,
        "change": -0.99,
        "answers": {"identical": 10, "differ": 86, "broken": 0, "total": 96},
    }
    verdict = judge_step(leg_a, tools_on=False)
    assert verdict.gate == "answers" and not verdict.keep
    within = {**leg_a, "answers": {**leg_a["answers"], "judged": 86, "changed": 0}}
    assert judge_step(within, tools_on=False).keep


def test_a_first_run_says_the_model_did_not_run_before_and_tables_its_numbers(
    tmp_path: Path,
) -> None:
    from tensward.cli_try import _first_verdict, _result_lines

    # README run: 30.1 tok/s, first word p50 2644 ms, p95 3673 ms
    metrics = {
        "succeeded": 160,
        "failed": 0,
        "output_throughput": 30.1,
        "ttft_p50_ms": 2644.0,
        "ttft_p95_ms": 3673.0,
    }
    run = tmp_path / "0001"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps(metrics), "utf-8")
    for name in ("requests.jsonl", "responses.jsonl"):
        (run / name).write_text("", "utf-8")
    verdict = _first_verdict(run)[0]
    assert verdict.lines == [
        "Your model did not run before; now it runs. Answers checked on their own."
    ]
    lines = _result_lines(tmp_path, "0001", None, verdict)
    assert lines[2:5] == [
        "Output tok/s                 30.1",
        "First word, typical          2.64 s",
        "First word, slowest 1 in 20  3.67 s",
    ]


def test_the_last_line_of_a_finished_session_names_the_speed_and_serve(tmp_path: Path) -> None:
    from tensward.cli_try import next_lines, wrap_up
    from tensward.kept import Kept

    run = tmp_path / "runs" / "0001-x"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(
        json.dumps({"succeeded": 1, "failed": 0, "output_throughput": 30.1}), "utf-8"
    )
    kept = Kept("0001-x", ("a=1",), (), "registered", "s")
    said = wrap_up(tmp_path, kept)
    assert said == "Your model now runs at 30.1 tok/s; nothing else helped."
    shown = next_lines([("tensward serve", "x")], more=False, wrap_up=said)
    assert shown[0] == said and sum("Next:" in line for line in shown) == 1
    assert wrap_up(tmp_path, None) is None
    # READMEV: a kept 30.1 tok/s setup, then a change with a real gain and reworded answers
    assert wrap_up(tmp_path, kept, pending=True) == (
        "Your model now runs at 30.1 tok/s. The last change is faster but rewords answers: keep "
        "it (tensward try --keep) or serve without it (tensward serve)."
    )


def test_the_step_figures_are_those_of_metrics_json(tmp_path: Path) -> None:
    from tensward.measurement import Measurement
    from tensward.report.build import _metrics_figures

    # README run: window figures 0.2385 / 0.2454, metrics.json 0.2359 / 0.2465
    run = tmp_path / "runs" / "0001-a"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(
        json.dumps({"request_throughput": 0.2359, "ttft_p50_ms": 2600.0}), "utf-8"
    )
    own = Measurement(succeeded=130, failed=0, request_throughput=0.2465)
    assert _metrics_figures(tmp_path, "0001-a", own, "req_s", 0.2385, 0.2454) == (
        0.2359,
        0.2465,
        None,
    )


def test_try_tables_the_run_before_and_after_with_the_verdict_under_it(
    tmp_path: Path,
) -> None:
    from tensward.cli_try import _result_lines

    real = {
        "before": {"requests": 160, "calling": 82, "valid_json": 1.0},
        "after": {"requests": 160, "calling": 82, "valid_json": 1.0},
    }
    verdict = judge_step({**step(), "tool_calls": real}, tools_on=True)
    runs = tmp_path / "runs"
    for name, busy, rate, parent in (("0001", 40, 4.0, None), ("0002", 70, 5.0, "0001")):
        (runs / name).mkdir(parents=True)
        metrics = {
            "request_throughput": rate,
            "diagnosis": {"requests": 5, "primary": None, "gpu_state": {"utilization": busy}},
        }
        (runs / name / "metrics.json").write_text(json.dumps(metrics))
        (runs / name / "window.json").write_text(json.dumps({"parent": parent}))
    lines = _result_lines(runs, "0002", {**step(), "tool_calls": real}, verdict)
    assert lines == [
        "            before           after                change",
        "─" * 57,
        "Requests/s  4                5                    +25%",
        "GPU active  40% of the time  70% of the time      +30 pts",
        "Answers                      all 40 identical     ✔",
        "Tool calls                   82 calls, all valid  ✔",
        "",
        "Requests/s: +25%: a real gain.",
        "No limit found.",
    ]


def test_a_real_gain_refused_only_for_reworded_answers_has_the_answers_gate(tmp_path: Path) -> None:
    from tensward.cli_try import wrap_up

    reworded = {
        **step(recommended=False),
        "verdict": "better",
        "reason": None,
        "answers": {"identical": 9, "differ": 87, "broken": 0, "total": 96},
    }
    assert judge_step(reworded, tools_on=False).gate == "answers"
    slower = {**reworded, "reason": "a loss: output tokens per second −12% ±3%"}
    assert judge_step(slower, tools_on=False).gate == "better"
    assert wrap_up(tmp_path, None, pending=True) == (
        "The last change is faster but rewords answers: keep it (tensward try --keep) or serve "
        "without it (tensward serve)."
    )


def test_still_loading_is_reported_at_most_once_a_minute() -> None:
    from tensward.runtime import STILL_LOADING_EVERY_S

    assert STILL_LOADING_EVERY_S >= 60


def test_a_change_for_another_goal_is_kept_only_if_it_helps_and_the_goal_holds() -> None:
    from tensward.cli_try import _gate_text
    from tensward.history import Compared
    from tensward.report.build import _side_reason
    from tensward.window import Delta

    def compared(ttft: str, tpot: str) -> Compared:
        moves = {"better": -0.2, "worse": 0.06, "within_noise": 0.01}
        return Compared(base="0001", deltas={
            "ttft_p95_ms": Delta("ttft_p95_ms", moves[ttft], 0.05, ttft, True),  # type: ignore[arg-type]
            "tpot_p95_ms": Delta("tpot_p95_ms", moves[tpot], 0.02, tpot, True),  # type: ignore[arg-type]
        })  # fmt: skip

    assert _side_reason(compared("better", "within_noise"), "ttft_p95_ms", "tpot_p95_ms") is None
    assert _side_reason(compared("within_noise", "within_noise"), "ttft_p95_ms", "tpot_p95_ms")
    lost = _side_reason(compared("better", "worse"), "ttft_p95_ms", "tpot_p95_ms")
    assert lost == "a loss: TPOT p95 +6% ±2%"

    side = {**step(recommended=False), "metric": "ttft_p95_ms", "change": -0.2, "band": 0.05}
    side |= {"verdict": "better", "reason": f"Tensward does not recommend keeping it: {lost}."}
    side["goal"] = {"metric": "tpot_p95_ms", "change": 0.06, "band": 0.02, "verdict": "worse"}
    verdict = judge_step(side, tools_on=False)
    assert verdict.gate == "better"
    assert _gate_text("better", side) == ("it made time per token worse (+6%).")
    assert "Per token, slowest 1 in 20: +6% slower: a real change." in verdict.lines


def test_a_step_not_compared_says_why_in_words(tmp_path: Path) -> None:
    window = tmp_path / "window.json"
    window.write_text(json.dumps({"matched": {"parent": "0001-baseline", "removed": 0.25}}))
    assert _unmatched(tmp_path) == (
        "It answered 25% more of the prompts than run 0001-baseline, which failed them, so the "
        "two runs did different work and their speed is not compared."
    )
    why = "too few requests answered in both runs to pair"
    window.write_text(json.dumps({"matched": {"parent": "0001-baseline", "why": why}}))
    assert _unmatched(tmp_path) == f"It was not compared with run 0001-baseline: {why}."
    window.write_text(json.dumps({"matched": {"parent": "0001-baseline", "upto": 64}}))
    assert _unmatched(tmp_path) is None
