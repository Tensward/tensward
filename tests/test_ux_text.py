"""The plain-words parts of ``init`` and ``analyse``: what they say, derived from one source."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tensward.cli import build_parser
from tensward.context import AnalysisContext
from tensward.detect import FIT_CONSEQUENCE, EngineFound, Hardware, Machine, fit_consequence
from tensward.errors import PreflightError
from tensward.presets import Choices, _records
from tensward.report.model import SuggestionBlock, change_words
from tensward.report.summary import (
    health_line,
    limit_line,
    next_line,
    result_line,
    tool_line,
    tool_warning,
)
from tensward.wizard import (
    GOAL_OPTIONS,
    Asker,
    InitPlan,
    equivalent_lines,
    found_models,
    plan_init,
    run_seconds,
)


def test_a_run_that_offered_tools_and_rarely_got_a_call_says_so_plainly() -> None:
    assert tool_warning(160, 2) == (
        "Only 2 of 160 requests called a tool, so tool-call quality cannot be checked; "
        "your real traffic may differ."
    )
    assert "None of the 160 requests called a tool" in str(tool_warning(160, 0))
    assert tool_warning(160, 90) is None and tool_warning(0, 0) is None


def test_the_result_line_and_summary_use_plain_words() -> None:
    assert result_line(160, 0, 2, "throughput") == (
        "Result: your server answered every request; 2 changes could still help."
    )
    assert health_line(160, 0, 0.18) == (
        "Is the server healthy: yes. All 160 requests were answered, 0.18 requests per second."
    )


def test_an_offer_shows_its_flag_and_not_the_record_name() -> None:
    command = (
        "tensward analyse --project p --engine-arg n-gpu-layers=41 --engine-arg threads=16 "
        "--engine-env GGML_X=1"
    )
    assert change_words(command, {"n-gpu-layers=41"}) == "--threads 16 GGML_X=1"
    assert "change" in SuggestionBlock.__dataclass_fields__


def test_a_second_launch_replaces_the_first_as_what_testing_a_change_takes() -> None:
    context = AnalysisContext(launched_s=58.0, load_s=16.0, measured_s=934.0)
    context.relaunched()
    context.launched_s, context.load_s, context.measured_s = 934.0, 8.0, 1887.0
    assert context.test_s == 953.0


def test_the_expected_minutes_grow_with_the_model_the_requests_and_a_second_start() -> None:
    gib = 2**30
    small = run_seconds(8 * gib, concurrency=4)
    assert run_seconds(37 * gib, concurrency=4) > small
    assert run_seconds(8 * gib, concurrency=2) > small
    assert run_seconds(37 * gib, relaunch=True) > 1.7 * run_seconds(37 * gib)
    assert math.ceil(run_seconds(37 * gib, relaunch=True) / 60) > 25


def test_agent_prompts_over_the_sample_tools_often_end_in_a_request_that_needs_a_tool() -> None:
    choices = Choices("agent", None, "default", False, 4, "throughput", "command")
    records = _records(choices)
    last = [r["messages"][-1]["content"].lower() for r in records]
    needing = [t for t in last if any(w in t for w in TOOL_WORDS)]
    assert len(needing) >= len(records) * 0.4
    assert all(r.get("tools") for r in records)


TOOL_WORDS = ("weather", "calculat", "search", "read the file", "readme", "query")


def test_a_choice_names_its_default_in_the_question(tmp_path: Path) -> None:
    said: list[str] = []
    ask = Asker(read=lambda prompt: "", write=said.append)
    ask.choice("Goal:", GOAL_OPTIONS, 3)
    ask.choice("Workload:", ("chat", "agent (multi-turn)"), 1)
    assert said[0] == "Goal [throughput]:" and "Workload [chat]:" in said


def test_the_repeat_command_is_on_short_lines_a_shell_joins() -> None:
    choices = Choices("agent", None, "default", False, 4, "throughput", "command")
    plan = InitPlan(
        Path("/m/model.gguf"), "llamacpp", "llama-server -m /m/model.gguf", choices, None
    )
    lines = equivalent_lines(plan)
    assert len(lines) == 3 and all(len(line) < 100 for line in lines[:2])
    assert lines[0].endswith("\\") and lines[-1].lstrip().startswith("--current")


def test_a_model_too_big_for_the_gpu_says_what_happens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gib = 2**30
    monkeypatch.setattr("tensward.detect._weights", lambda model: (35 * gib, "q8_0"))
    gpu = Hardware("nvidia", (("NVIDIA L4", 22 * gib),), None, True, None, None, None)
    roomy = Hardware("nvidia", (("NVIDIA H100", 80 * gib),), None, True, None, None, None)
    assert fit_consequence(tmp_path, gpu) == FIT_CONSEQUENCE
    assert fit_consequence(tmp_path, roomy) is None


def test_the_summary_names_the_real_limit_and_one_next_step() -> None:
    assert limit_line("load_limited", True, None).startswith(
        "What limits it: nothing in the server"
    )
    assert limit_line("load_limited", True, None, 44) == (
        "What limits it: part of the model is read from CPU memory, 44% of each step."
    )
    assert next_line("--threads 16", "throughput", 601) == (
        "Next: `tensward try` tests --threads 16 (~11 min)."
    )


def test_the_summary_shows_requests_that_called_a_tool_and_how_many_calls_were_valid() -> None:
    assert tool_line(160, 82, 1.0) == "Tools: 82 of 160 requests called a tool; all calls valid."
    assert tool_line(160, 82, 0.9) == "Tools: 82 of 160 requests called a tool; 90% of calls valid."
    assert tool_line(160, 2, 1.0) is None  # the few-calls warning says it


def test_a_model_path_missing_its_leading_slash_is_offered_and_found_models_are_numbered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"
    (models / "Qwen3-8B-GGUF").mkdir(parents=True)
    gguf = models / "Qwen3-8B-GGUF" / "Qwen3-8B-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")
    (models / "notes.txt").write_text("")
    assert found_models([models]) == [gguf]

    said: list[str] = []
    answers = iter([str(gguf).lstrip("/"), ""])
    ask = Asker(read=lambda prompt: next(answers), write=said.append)
    assert ask.path("Model:", lambda path: None if path.exists() else "missing") == gguf
    assert said == ["missing", f"Did you mean {gguf}?"]
    monkeypatch.setenv("MODELS", str(models))
    typed = iter(["$MODELS/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"])
    assert Asker(read=lambda prompt: next(typed)).path("Model:", lambda path: None) == gguf
    assert Asker(read=lambda prompt: "1").path("Model:", lambda p: "no", numbered=[gguf]) == gguf


def test_a_command_for_an_engine_that_is_not_installed_says_so() -> None:
    gpu = Hardware("nvidia", (("NVIDIA L4", 24 * 10**9),), "580.95", True, None, None, None)
    vllm = EngineFound("vllm", "0.30.0", "vllm", ())
    machine = Machine((vllm,), gpu, ())
    command = "llama-server -m /models/Qwen3-8B-Q4_K_M.gguf"
    arguments = build_parser().parse_args(["init", "--current", command])
    with pytest.raises(PreflightError) as refused:
        plan_init(arguments, machine, None, current=command)
    assert refused.value.message == (
        "Your command runs llama.cpp (llama-server), which is not installed here. "
        "Install it: see 'Don't have an engine yet?' in the README."
    )


def test_a_model_whose_engine_is_missing_or_wrong_is_refused_before_any_question(
    tmp_path: Path,
) -> None:
    gpu = Hardware("nvidia", (("NVIDIA L4", 24 * 10**9),), "580.95", True, None, None, None)
    gguf = tmp_path / "Qwen3-8B-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF")
    folder = tmp_path / "Qwen3-8B"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    only_vllm = Machine((EngineFound("vllm", "0.30.0", "vllm", ()),), gpu, ())
    only_llama = Machine((EngineFound("llamacpp", "b11669", "llama-server", ()),), gpu, ())
    cases = [
        (only_vllm, ["--model", str(gguf)], None,
         "Your model is a GGUF file, which runs on llama.cpp; llama.cpp is not installed here. "
         "Install it: see 'Don't have an engine yet?' in the README."),
        (only_llama, ["--model", str(folder)], None,
         "Your model is a safetensors folder, which runs on vLLM; vLLM is not installed here. "
         "Install it: see 'Don't have an engine yet?' in the README."),
        (only_vllm, [], f"vllm serve {gguf}",
         "Your command runs vLLM, which cannot load your model: it is a GGUF file, which runs "
         "on llama.cpp."),
    ]  # fmt: skip
    for machine, flags, command, said in cases:
        given = ["--current", command] if command else []
        arguments = build_parser().parse_args(["init", *flags, *given])
        with pytest.raises(PreflightError) as refused:
            plan_init(arguments, machine, None, current=command)
        assert refused.value.message == said


def test_columns_align_wrap_inside_80_columns_and_fall_back_to_a_plain_rule() -> None:
    from tensward.text import columns

    long = "44 of 64 identical, 20 reworded (within the earlier run's own repeats)"
    rows = [("Requests/s", "0.237", "0.321", "+33%"), ("Answers", "", long, "✔")]
    lines = columns(("", "before", "after", "change"), rows, plain=True)
    assert lines[1] == "-" * max(map(len, lines))
    assert all(len(line) <= 80 for line in lines)
    at = lines[0].index("after")
    assert lines[2][at:].startswith("0.321") and lines[3][at:].startswith("44 of 64")
    assert lines[4].startswith(" " * at) and lines[4].strip()
    change = lines[0].index("change")
    assert lines[2][change:] == "+33%" and lines[3][change:] == "✔"
    narrow = columns(("", "x"), [("a", "word " * 30)], width=40, plain=False)
    assert narrow[1] == "─" * 40 and all(len(line) <= 40 for line in narrow)
