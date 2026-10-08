"""End-to-end ``tensward analyse`` against a fake vLLM server, plus the docker argv contract."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import shlex
import shutil
import signal
import stat
import sys
from pathlib import Path
from typing import Any, NoReturn

import pytest
from image_fixtures import write_images
from interrupts import signal_once_present
from registration_fixtures import (
    QUANTIZED_CHECKPOINTS,
    REGISTRATION_CONFIG,
    SAFETENSORS_NAME,
    make_gemma4_checkpoint,
    make_image_workload,
    make_registration_inputs,
    safetensors_from_tensors,
    write_config,
    write_prompts,
)
from test_playbook import situation

from tensward import progress
from tensward.ceilings import gpu_spec
from tensward.cli import main
from tensward.client import RequestPlan, run_workload
from tensward.engines import ENGINES
from tensward.engines.vllm import VLLM
from tensward.engines.vllm.capabilities import speculative_config
from tensward.engines.vllm.predicates import NGRAM
from tensward.errors import AnalyseFailure
from tensward.extensions import API_VERSION, Extender
from tensward.images import ImageSource
from tensward.measurement import Measurement
from tensward.platforms import Device as GpuInfo
from tensward.playbook import WorkloadFacts, ranked
from tensward.profiling.trace import read_trace
from tensward.progress import ProgressEvent
from tensward.project import apply_engine_args
from tensward.report.build import FEEDBACK_LINE
from tensward.runtime import process_start_time
from tensward.settings import Settings
from tensward.suggest import applicable
from tensward.workload import (
    ArrivalSpec,
    ChatMessage,
    ChatRequest,
    ImagePart,
    ImageUrl,
    WorkloadSpec,
)

NGRAM_CONFIG = speculative_config(NGRAM)
PLAYBOOK = VLLM.playbook()
SHORT_WINDOW = (
    "the measurement window was only {} s; throughput is noisy; raise request_count to about {}"
)


def checks_of(checks: list[str]) -> list[str]:
    """The run's checks, with the short-window line's length masked."""
    return [
        re.sub(r"about \d+", "about {}", re.sub(r"only [\d.]+ s", "only {} s", check))
        for check in checks
    ]


def sections_of(run_dir: Path) -> dict[str, list[str]]:
    """report.json as section id -> the keys of its blocks, in order."""
    report = json.loads((run_dir / "report.json").read_text())
    return {s["id"]: [b["key"] for b in s["blocks"]] for s in report["sections"]}


FAKE_SERVER = Path(__file__).resolve().parent / "fake_vllm_server.py"
SYNTHETIC_TRACE = Path(__file__).resolve().parent / "synthetic_trace.json"


def test_analyse_runs_end_to_end_against_the_fake_server(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "FAKE_QUANT_KERNEL_LINE", "INFO Using MarlinLinearKernel for AutoAWQMarlinLinearMethod"
    )
    monkeypatch.setattr("tensward.ceilings.detect_gpus", lambda: (("NVIDIA L4", 23034),))
    monkeypatch.setattr(
        "tensward.project.detect_devices", lambda: (GpuInfo("NVIDIA L4", 23034 * 2**20, 0),)
    )
    # no Nsight Compute
    monkeypatch.setattr("tensward.profiling.run.shutil.which", lambda name: None)
    model, config, _ = make_registration_inputs(tmp_path, "awq")
    # Give the tiny checkpoint real dimensions so the hardware ceilings can be computed.
    document = json.loads((model / "config.json").read_text())
    dims = {"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4}
    dims |= {"num_key_value_heads": 2, "intermediate_size": 128, "vocab_size": 100}
    (model / "config.json").write_text(json.dumps({**document, **dims}))
    tensors = {**QUANTIZED_CHECKPOINTS["awq"][2], "model.embed_tokens.weight": ("F16", (100, 64))}
    (model / SAFETENSORS_NAME).write_bytes(safetensors_from_tensors(tensors))
    # The third prompt is longer than the 2048-token context, so the server refuses it. The
    # other two share a 300-token prefix, which prefix caching would serve from cache.
    prefix = "shared " * 300
    prompts = write_prompts(
        tmp_path / "long.jsonl",
        [
            {"id": "shared-a", "prompt": prefix + "alpha " + " ".join(f"a{i}" for i in range(300))},
            {"id": "shared-b", "prompt": prefix + "beta " + " ".join(f"b{i}" for i in range(300))},
            {"id": "too-long", "prompt": "word " * 3000},
        ],
    )
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    capsys.readouterr()

    command = shlex.join([sys.executable, str(FAKE_SERVER)])
    code = main(["analyse", "--project", str(project), "--runtime", "local", "--counters",
                 "--engine-arg", "no-async-scheduling", "--slo-ttft-ms", "5000",
                 "--engine-arg", 'compilation-config={"cudagraph_capture_sizes": [1, 2, 3]}',
                 "--local-command", command, "--ready-timeout", "30"])  # fmt: skip
    assert code == 0
    captured = capsys.readouterr()
    out = captured.out
    assert "current setup + overrides (no-async-scheduling, compilation-config=" in out
    phases = [line for line in captured.err.splitlines() if line.startswith("[")]
    for phase in ("measurement: launching vllm", "waiting for the model to load", "warmup: ",
                  "measuring: 10/10 requests (100%)", "trace: profiling", "measurement: stopping",
                  "counters: profiling"):  # fmt: skip
        assert any(phase in line for line in phases), phase
    assert "3 of 10 requests failed" in out and "TTFT p95" in out and "GPU busy" in out
    assert out.startswith("Ran: vLLM ")
    assert out.index("TPOT p95") < out.index("Bottleneck: ") < out.index("## What to try next")
    assert out.index("## What to try next") < out.index("run directory:")

    (run_dir,) = (project / "runs").iterdir()
    names = {path.name for path in run_dir.iterdir()}
    assert {"serve.json", "requests.jsonl", "responses.jsonl", "metrics_before.prom",
            "metrics_after.prom", "metrics.json", "report.md",
            "trace"} <= names  # fmt: skip
    assert len(list((run_dir / "trace").glob("*.pt.trace.json.gz"))) == 1  # the raw trace is kept
    # prefix-caching is offered and the server returned every prompt's token ids
    lines = (run_dir / "prompt_blocks.jsonl").read_text().splitlines()
    blocks = [json.loads(line) for line in lines]
    assert [row["prompt_id"] for row in blocks] == ["shared-a", "shared-b", "too-long"]
    assert all(len(row["blocks"]) == row["tokens"] // row["block_tokens"] for row in blocks)
    assert "expected:" not in out  # a fake-server run is too short for an estimate
    assert "Using MarlinLinearKernel" in (run_dir / "server.log").read_text()  # the whole log

    serve = json.loads((run_dir / "serve.json").read_text())
    assert "api_key" not in json.dumps(serve)
    responses = [
        json.loads(line) for line in (run_dir / "responses.jsonl").read_text().splitlines()
    ]
    assert len(responses) == 10
    served = [row for row in responses if row["prompt_id"] != "too-long"]
    assert len(served) == 7
    assert all(row["text"].startswith("tok0 tok1") and row["truncated"] for row in served)
    assert {row["prompt_id"] for row in responses} == {"shared-a", "shared-b", "too-long"}

    requests = [json.loads(line) for line in (run_dir / "requests.jsonl").read_text().splitlines()]
    failed = [row for row in requests if row["outcome"] != "success"]
    assert {row["prompt_id"] for row in failed} == {"too-long"} and len(failed) == 3
    assert all(row["http_status"] == 400 for row in failed)
    assert all("maximum context length is 2048" in row["error"] for row in failed)
    assert not any("http_status" in row for row in requests if row["outcome"] == "success")
    summary = (run_dir / "report.md").read_text()
    assert summary.startswith("# Tensward analysis")
    ran = next(line for line in summary.splitlines() if line.startswith("Ran: "))
    assert ran.startswith("Ran: vLLM version unknown (local command python")
    assert ran.endswith("; checkpoint: safetensors, awq int4 group 128")
    assert "\nBottleneck: " in summary and "\n## What to try next\n" in summary
    assert summary.endswith(f"\n{FEEDBACK_LINE}\n")
    assert not re.search(r"\b(?:[A-Z]{2,4}-\d+|A\d+\.\d+)\b", summary)  # no private catalog IDs
    assert json.loads((run_dir / "run.json").read_text())["format"] == "hf-safetensors"
    sections = sections_of(run_dir)
    assert list(sections) == [
        "header", "diagnosis", "next_steps", "setup", "requests", "performance", "engine",
        "defaults", "ceilings", "quantization", "context", "checks", "trace", "counters",
        "answers", "feedback",
    ]  # fmt: skip
    assert {"source", "settings", "measured", "fails_requests"} <= set(sections["setup"])
    assert sections["requests"][-3:] == ["repeated_prompts", "failures", "failure"]
    assert {"failed", "ceiling", "gpu_busy"} <= set(sections["header"])
    assert {"startup_wave", "goodput", "slo_attainment"} <= set(sections["performance"])
    assert "kv_capacity" in sections["engine"] and "gate.fit" in sections["diagnosis"]
    assert "cant_tell" in sections["diagnosis"]
    assert "for.host_overhead" not in sections["next_steps"]  # nothing to change: no empty group
    assert {"fit-context", "prefix-caching", "not_applicable", "serve_hint"} <= set(
        sections["next_steps"]
    )
    assert sections["context"] == ["too_long", "longest"]  # 3032 tokens fit the model's 4096
    assert sections["trace"] == ["caveat", "window", "gaps", "analysis_hint"]
    assert sections["counters"] == ["unavailable"]  # --counters implies --trace; no ncu here
    assert "async-scheduling" not in sections["next_steps"]  # no analysis plugin
    report = json.loads((run_dir / "report.json").read_text())
    steps = next(s for s in report["sections"] if s["id"] == "next_steps")["blocks"]
    commands = [b["command"] for b in steps if b["kind"] == "suggestion" and b["command"]]
    assert commands and all(c.startswith("tensward analyse --project ") for c in commands)
    assert all("--engine-arg " in c for c in commands)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    diagnosis = metrics["diagnosis"]
    assert [f["bottleneck"] for f in diagnosis["findings"]] == [
        "queueing", "kv_capacity", "prefill", "prefill_stalls_decode", "decode_bandwidth",
        "gpu_compute", "host_overhead", "frontend_cpu", "long_context", "speculation",
    ]  # fmt: skip
    assert diagnosis["gates"][-1]["state"] == "critical"  # three requests did not fit
    assert diagnosis["requests"] == 7  # 10 requests, 3 too long
    states = {f["bottleneck"]: f["state"] for f in diagnosis["findings"]}
    assert states["gpu_compute"] == "cant_tell"  # no ncu here
    assert "queue_share" in diagnosis["thresholds"]
    (gated,) = [row for row in diagnosis["not_applicable"] if row["entry"] == "ngram-speculation"]
    assert gated["why"].startswith("the pinned CUDA graph sizes (max 3)")
    assert metrics["kv_capacity_tokens"] == 40416 and metrics["kv_capacity_estimate_tokens"]
    assert metrics["quant_kernels"] == [
        {"name": "MarlinLinearKernel", "layer": "AutoAWQMarlinLinearMethod", "slow": False}
    ]
    assert metrics["too_long"] == ["too-long"] and metrics["max_prompt_tokens"] == 3000
    # Total counts prompt tokens too; goodput counts only requests inside the SLO (5 s TTFT here).
    assert metrics["total_throughput"] > metrics["output_throughput"] > 0
    assert 0 < metrics["goodput"] <= metrics["request_throughput"]
    assert metrics["slo_attainment"] == pytest.approx(0.7)  # the 3 refused requests miss it
    assert metrics["startup_wave"]["requests"] == 1
    ceilings = metrics["ceilings"]
    assert ceilings["gpu"] == "L4" and ceilings["unavailable"] is None
    assert (ceilings["bandwidth_gbs"], ceilings["bandwidth_source"]) == (300.0, "datasheet")
    assert ceilings["tensor_tflops"] == 121
    assert ceilings["decode_ceiling_batch1_tok_s"] > 0 and ceilings["prefill_ceiling_tok_s"] > 0
    assert ceilings["measured_decode_tok_s"] > 0 and ceilings["measured_prefill_tok_s"] > 0
    assert gpu_spec((("NVIDIA Foo", 1),), None) == (0, "NVIDIA Foo", None)
    trace = metrics["trace"]  # a separate launch; the fake server serves the synthetic trace
    assert trace["status"] == "ok" and trace["kernels"] == 80
    assert trace["gpu_busy_share"] == pytest.approx(0.676, abs=5e-4)
    assert trace["gap_count"] == 3 and trace["idle_share"] == pytest.approx(0.217, abs=5e-4)
    assert trace["analysis"] is None
    assert metrics["counters"]["status"] == "unavailable" and metrics["counters"]["kernels"] == []
    assert metrics["counters"]["note"] == "ncu not found"
    assert "report.json" in names and "evidence.json" not in names
    assert checks_of(metrics["checks"]) == [SHORT_WINDOW]  # every vLLM 0.30 metric is read

    with pytest.raises(ProcessLookupError):
        os.kill(serve["identity"]["pid"], 0)


WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city", "days"],
        },
    },
}
SUPPORT_CHAT = [
    {"role": "system", "content": "You are a support agent for Acme. Be concise."},
    {"role": "user", "content": "My order has not arrived."},
    {"role": "assistant", "content": "Sorry to hear that. What is the order number?"},
    {"role": "user", "content": "It is 1234, placed last week."},
]


def _chat_project(tmp_path: Path, name: str, *, tool_calling: bool, current: bool = False) -> Path:
    """Register a chat workload (one plain chat, one offering a tool) on a Qwen2-family model;
    with ``current``, from a plain ``vllm serve`` command."""
    model, _, _ = make_registration_inputs(tmp_path / name)
    document = json.loads((model / "config.json").read_text())
    (model / "config.json").write_text(json.dumps({**document, "model_type": "qwen2"}))
    config = {
        "case": {**REGISTRATION_CONFIG["case"], "tool_calling": tool_calling},
        "workload": {**REGISTRATION_CONFIG["workload"], "api": "chat"},
    }
    prompts = write_prompts(
        tmp_path / name / "chat.jsonl",
        [
            {"id": "support", "messages": SUPPORT_CHAT},
            {
                "id": "weather",
                "messages": [{"role": "user", "content": "Weather in Paris for 3 days?"}],
                "tools": [WEATHER],
                "tool_choice": "auto",
                "max_tokens": 64,
            },
        ],
    )
    project = tmp_path / name / "project"
    serving = write_config(tmp_path / name / "chat-serving.json", {**REGISTRATION_CONFIG, **config})
    imported = ["--current", f"vllm serve {model}"] if current else []
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(serving), "--prompts", str(prompts), *imported]) == 0  # fmt: skip
    return project


ANSWER_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}}
ANSWER_FORMAT = {"type": "json_schema", "json_schema": {"name": "answer", "schema": ANSWER_SCHEMA}}


def _structured_project(tmp_path: Path, name: str) -> Path:
    """A chat workload whose first two records declare their own JSON schema; the third offers a
    tool and no schema; the fourth only thinks. ``ramble`` asks for something the schema cannot
    hold."""
    model, _, _ = make_registration_inputs(tmp_path / name)
    document = json.loads((model / "config.json").read_text())
    (model / "config.json").write_text(json.dumps({**document, "model_type": "qwen2"}))
    prompts = write_prompts(
        tmp_path / name / "structured.jsonl",
        [
            {"id": "facts", "messages": [{"role": "user", "content": "Name one fact."}],
             "response_format": ANSWER_FORMAT, "chat_template_kwargs": {"enable_thinking": False}},
            {"id": "ramble", "messages": [{"role": "user", "content": "ramble about the sea"}],
             "response_format": ANSWER_FORMAT, "max_tokens": 32},
            {"id": "support", "messages": SUPPORT_CHAT, "tools": [WEATHER]},
            {"id": "think", "messages": [{"role": "user", "content": "Think about the sea."}],
             "chat_template_kwargs": {"enable_thinking": True}, "max_tokens": 8},
        ],
    )  # fmt: skip
    config = {
        **REGISTRATION_CONFIG,
        "case": {**REGISTRATION_CONFIG["case"], "tool_calling": True},
        "workload": {**REGISTRATION_CONFIG["workload"], "api": "chat"},
    }
    serving = write_config(tmp_path / name / "serving.json", config)
    project = tmp_path / name / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(serving), "--prompts", str(prompts)]) == 0  # fmt: skip
    return project


@dataclasses.dataclass(frozen=True)
class Analysed:
    """One ``tensward analyse`` invocation: its exit code, standard error and the run it wrote."""

    code: int
    report: str
    err: str
    out: str
    run_dir: Path | None
    metrics: dict | None
    responses: list[dict] | None


def _analyse(project: Path, capsys: pytest.CaptureFixture[str], *cli_words: str) -> Analysed:
    command = shlex.join([sys.executable, str(FAKE_SERVER)])
    runs = project / "runs"
    before = set(runs.iterdir()) if runs.exists() else set()
    code = main(["analyse", "--project", str(project), "--runtime", "local",
                 "--local-command", command, "--ready-timeout", "30", *cli_words])  # fmt: skip
    captured = capsys.readouterr()
    written = set(runs.iterdir()) - before if runs.exists() else set()
    if not written:
        return Analysed(code, "", captured.err, captured.out, None, None, None)
    (run_dir,) = written
    return Analysed(
        code,
        (run_dir / "report.md").read_text(),
        captured.err,
        captured.out,
        run_dir,
        json.loads((run_dir / "metrics.json").read_text()),
        [json.loads(line) for line in (run_dir / "responses.jsonl").read_text().splitlines()],
    )


def test_records_carry_their_own_structured_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = _analyse(_structured_project(tmp_path, "shaped"), capsys)
    assert run.code == 0 and run.metrics["failed"] == 0
    for row in run.responses:
        if row["prompt_id"] == "facts":
            assert json.loads(row["text"])["answer"].startswith("tok0")
        elif row["prompt_id"] == "support":
            assert row["tool_calls"]
    assert {row["prompt_id"] for row in run.responses} >= {"facts", "support"}
    assert "structured output with tools" not in run.report
    requests = (run.run_dir / "requests.jsonl").read_text().splitlines()
    thinking = [row for row in map(json.loads, requests) if row["prompt_id"] == "think"]
    assert thinking and all(row["first_content_ns"] is not None for row in thinking)


COMPACT = 'structured-outputs-config={"backend": "xgrammar", "disable_any_whitespace": true}'


def test_a_whitespace_loop_in_the_answers_suggests_compact_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _structured_project(tmp_path, "loop")
    base = _analyse(project, capsys)
    stats = base.metrics["structured_answers"]
    # 10 requests over 4 records: facts 3, ramble 3; support and think are not judged
    assert stats["answers"] == 6
    assert (
        stats["invalid_json"]
        == stats["runaway_whitespace"]
        == stats["cut_short"]
        == pytest.approx(3 / 6)
    )
    assert stats["runaway_prompts"] == ["ramble"]
    assert "## Structured output (quality, not performance)" in base.report
    assert "`compact-json`" in base.report and "(from your answers)" in base.report
    assert f"--engine-arg '{COMPACT}'" in base.report  # the suggested command, as printed

    # the fake starts only with a backend
    compact = _analyse(project, capsys, "--engine-arg", COMPACT)
    assert compact.code == 0
    assert compact.metrics["structured_answers"]["invalid_json"] == 0.0
    assert "`compact-json`" not in compact.report
    assert "invalid JSON" in compact.report  # the comparison with the current setup


def test_analyse_measures_the_imported_current_setup_and_labels_overrides(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    current = f"vllm serve {model} --max-model-len 2048 --swap-space 4 -e VLLM_TEST_FLAG=1"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts),
                 "--current", current]) == 0  # fmt: skip
    command = shlex.join([sys.executable, str(FAKE_SERVER)])

    events: list[ProgressEvent] = []
    with progress.sink(events.append):
        assert main(["analyse", "--project", str(project), "--runtime", "local",
                     "--engine-arg", "max-num-seqs=3",
                     "--local-command", command, "--ready-timeout", "30"]) == 0  # fmt: skip
    firsts = list(dict.fromkeys(type(event).__name__ for event in events))
    assert firsts == [
        "PhaseStarted", "ServerLoading", "ServerReady", "WindowOpened", "RequestsDone",
        "RunWritten",
    ]  # fmt: skip

    (run_dir,) = (project / "runs").iterdir()
    report = (run_dir / "report.md").read_text()
    assert "## Your current setup + overrides (max-num-seqs=3)" in report
    assert "- source: imported from --current" in report
    assert "max_concurrent_requests=3, max_context_len=2048, --swap-space 4" in report
    # --current wins over the configuration's serving fields, and the report says which differ.
    assert ("- note: your --current command wins; these serving fields of the configuration "
            "are ignored: max_concurrent_requests (configuration 1)") in report  # fmt: skip
    assert "\n- prefix_caching: on (for generative models)\n" in report  # unset, so the default
    assert "- prefix-cache hit rate: " in report
    argv = json.loads((run_dir / "serve.json").read_text())["identity"]["argv"]
    assert argv[argv.index("--swap-space") + 1] == "4"  # every other flag is reproduced verbatim


def test_a_local_command_given_as_a_path_finds_its_neighbours_and_a_failed_start_says_why(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    for name, body in {
        "ninja": "",
        "serve": 'echo "(APIServer pid=7) ValidationError: 1 validation error for ModelConfig"; '
        'echo "(APIServer pid=7)   Value error, no $(command -v ninja)"; echo "later noise"; '
        'echo "RuntimeError: Engine core initialization failed. See root cause above."; exit 1',
    }.items():
        (bin_dir / name).write_text(f"#!/bin/sh\n{body}\n")
        (bin_dir / name).chmod(0o755)
    capsys.readouterr()

    assert main(["analyse", "--project", str(project), "--runtime", "local",
                 "--local-command", str(bin_dir / "serve")]) == 1  # fmt: skip

    failure = capsys.readouterr().err
    assert (
        "the server exited before it was ready: ValidationError: 1 validation error for "
        f"ModelConfig: Value error, no {bin_dir}/ninja"
    ) in failure


def test_analyse_suggests_serving_only_the_text_model_and_leaves_the_baseline_alone(
    tmp_path: Path,
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path, "compressed-tensors-int4")
    make_gemma4_checkpoint(model)
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    command = shlex.join([sys.executable, str(FAKE_SERVER)])

    assert main(["analyse", "--project", str(project), "--runtime", "local",
                 "--local-command", command, "--ready-timeout", "30"]) == 0  # fmt: skip

    (run_dir,) = (project / "runs").iterdir()
    report = (run_dir / "report.md").read_text()
    assert "`no-media-encoders`" in report and "--engine-arg language-model-only" in report
    argv = json.loads((run_dir / "serve.json").read_text())["identity"]["argv"]
    assert "--language-model-only" not in argv  # the baseline runs as the customer's command says


def test_analyse_sends_images_as_data_urls_and_counts_their_tokens(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, _ = make_registration_inputs(tmp_path, "compressed-tensors-int4", api="chat")
    make_gemma4_checkpoint(model)
    prompts = make_image_workload(tmp_path)
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip

    result = _analyse(project, capsys)  # the fake engine refuses any image that is not a data URL

    assert result.code == 0
    lines = (result.run_dir / "requests.jsonl").read_text().splitlines()
    assert {json.loads(line)["outcome"] for line in lines} == {"success"}
    assert result.metrics["max_prompt_tokens"] >= 256  # an image counts as the engine counted it


def test_mixed_workload_splits_and_keeps_media_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, _ = make_registration_inputs(tmp_path, "compressed-tensors-int4", api="chat")
    make_gemma4_checkpoint(model)
    prompts = make_image_workload(tmp_path)
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip

    result = _analyse(project, capsys)

    assert result.code == 0
    split = result.metrics["image_split"]
    assert split["with_images"]["requests"] > 0 and split["without_images"]["requests"] > 0
    section = result.report.split("## Requests with and without images")[1].split("\n## ")[0]
    assert "- with images: " in section and "- without images: " in section
    assert section.count("TTFT p95") == 2
    assert "`no-media-encoders`" not in result.report


def test_an_image_changed_after_registration_is_not_sent(tmp_path: Path) -> None:
    images = write_images(tmp_path, count=1, size=1000)
    (tmp_path / "0.png").write_bytes(bytes([9]) * 1000)  # same size, other bytes
    chat = ChatRequest(
        messages=(
            ChatMessage(
                role="user", content=(ImagePart(type="image_url", image_url=ImageUrl(url="0.png")),)
            ),
        )
    )
    workload = WorkloadSpec(
        api="chat",
        chats=(chat,),
        output_tokens=8,
        request_count=1,
        request_timeout_s=5.0,
        arrival=ArrivalSpec(kind="closed_loop", concurrency=1),
        temperature=0.0,
        top_p=1.0,
    )

    class NothingIsSent:
        def stream(self, plan: RequestPlan) -> NoReturn:
            raise AssertionError("a request with changed image bytes was sent")

    (record,) = asyncio.run(
        run_workload(
            workload, run_id="r", model="m", transport=NothingIsSent(), images=ImageSource(images)
        )
    )

    assert record.outcome == "error"
    assert record.error == "workload image 0.png changed since registration"


def test_secrets_in_the_current_command_reach_no_file_output_or_server_argv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    current = (f"docker run -e HF_TOKEN=hf_SECRET1 --env=OTHER_API_KEY=hf_SECRET2 img:1 "
               f"--model {model} --hf-token hf_SECRET3 --max-model-len 2048")  # fmt: skip
    base = ["--project", str(project)]
    assert main(["init", *base, "--model", str(model), "--config", str(config),
                 "--prompts", str(prompts), "--current", current]) == 0  # fmt: skip
    assert main(["inspect", *base]) == 0
    command = shlex.join([sys.executable, str(FAKE_SERVER)])
    assert main(["analyse", *base, "--runtime", "local", "--image", "img:1",
                 "--local-command", command, "--ready-timeout", "30"]) == 0  # fmt: skip

    output = capsys.readouterr()
    everything = (
        output.out
        + output.err
        + "".join(path.read_text(errors="replace") for path in project.rglob("*") if path.is_file())
    )
    assert "hf_SECRET" not in everything
    (run_dir,) = (project / "runs").iterdir()
    assert "--hf-token" not in json.loads((run_dir / "serve.json").read_text())["identity"]["argv"]


def test_analyse_interrupted_while_the_model_loads_leaves_no_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_LOAD_SECONDS", "60")
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    signal_once_present((project / "runs", "*/serve.json"), signal.SIGTERM)
    command = shlex.join([sys.executable, str(FAKE_SERVER)])

    assert main(["analyse", "--project", str(project), "--runtime", "local",
                 "--local-command", command, "--ready-timeout", "120"]) == 130  # fmt: skip

    (serve_json,) = (project / "runs").glob("*/serve.json")
    pid = json.loads(serve_json.read_text())["identity"]["pid"]
    assert process_start_time(pid) is None  # the engine is gone


def test_a_run_interrupted_after_the_launch_keeps_its_evidence_and_is_never_a_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _chat_project(tmp_path, "project", tool_calling=True)

    def interrupted(*args: object, **kwargs: object) -> NoReturn:
        raise KeyboardInterrupt

    command = shlex.join([sys.executable, str(FAKE_SERVER)])
    with monkeypatch.context() as patch:
        patch.setattr("tensward.analyse.write_derived", interrupted)
        assert main(["analyse", "--project", str(project), "--runtime", "local",
                     "--local-command", command, "--ready-timeout", "30"]) == 130  # fmt: skip
    capsys.readouterr()
    (broken,) = (project / "runs").iterdir()
    assert (broken / "requests.jsonl").is_file() and not (broken / "metrics.json").exists()

    baseline = _analyse(project, capsys)
    after = _analyse(project, capsys, "--engine-arg", "max-num-seqs=3")
    assert f"## What changed vs your current setup (run {baseline.run_dir.name})" in after.report


def test_an_interrupted_or_failing_profile_still_writes_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip

    def raises(error: BaseException) -> Any:
        def call(*args: object, **kwargs: object) -> NoReturn:
            raise error

        return call

    with monkeypatch.context() as patch:
        patch.setattr("tensward.profiling.run.read_trace", raises(KeyboardInterrupt()))
        interrupted = _analyse(project, capsys, "--counters")
    assert interrupted.code == 130 and interrupted.metrics is not None
    assert interrupted.metrics["trace"]["status"] == "unavailable"
    assert interrupted.metrics["counters"]["note"].startswith("interrupted")
    assert "trace unavailable: interrupted" in interrupted.report

    plugin = Extender(api_version=API_VERSION, analyse=raises(KeyError("gap")))
    with monkeypatch.context() as patch:
        patch.setattr("tensward.profiling.run.load_extender", lambda: plugin)
        failed = _analyse(project, capsys, "--trace")
    assert failed.code == 0 and failed.metrics is not None
    assert failed.metrics["trace"]["status"] == "ok"
    assert "the analysis plugin failed and was skipped (KeyError: 'gap')" in failed.err


def test_analyse_measures_chat_and_tool_calls_and_blocks_tools_that_the_server_cannot_serve(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    off = _analyse(_chat_project(tmp_path, "off", tool_calling=False), capsys)
    assert off.code == 0
    assert "## Checks" in off.report and "tool calling is not enabled" in off.report
    assert "`enable-tool-calling`" in off.report  # the Qwen2 parser (hermes) is known
    assert "HTTP 400" in off.report
    assert '"auto" tool choice requires --enable-auto-tool-choice' in off.report
    assert off.metrics["tool_calls"] is None  # every request that offered tools was refused
    assert off.metrics["ttft_p50_ms"] is not None  # the plain chat requests still stream
    warning = (
        "warning: 1 of 2 prompts offer tools; "
        "the workload offers tools but tool calling is not enabled"
    )
    assert off.err.count(warning) == 2  # at init and before the analyse launch

    # --current overrides the configuration's tool_calling: the fix is to the command
    imported = _chat_project(tmp_path, "imported", tool_calling=True, current=True)
    err = capsys.readouterr().err
    assert "add `--enable-auto-tool-choice --tool-call-parser hermes` to your --current" in err
    notes = json.loads((imported / "project.json").read_text())["current_setup"]["notes"]
    root = tmp_path / "imported"
    again = ["init", "--project", str(imported), "--model", str(root / "model"),
             "--config", str(root / "chat-serving.json"), "--prompts", str(root / "chat.jsonl"),
             "--current", f"vllm serve {root / 'model'}"]  # fmt: skip
    assert main(again) == 0  # the same init again: notes are not the identity
    capsys.readouterr()
    assert any("tool_calling (configuration True; your command sets False)" in n for n in notes)

    on = _analyse(_chat_project(tmp_path, "on", tool_calling=True), capsys)
    assert on.code == 0 and "prompts offer tools;" not in on.err
    report, responses = on.report, on.responses
    assert checks_of(on.metrics["checks"]) == [SHORT_WINDOW] and "enable-tool-calling" not in report
    assert on.metrics["failed"] == 0 and on.metrics["ttft_p50_ms"] is not None
    assert on.metrics["tool_calls"] == {
        "requests": 5, "calling": 5, "produced_call": 1.0,
        "valid_json": 1.0, "known_tool": 1.0, "schema_valid": 1.0,
    }  # fmt: skip
    assert "- produced a tool call: 100%" in report
    assert "prompts that should not call a tool count as misses" in report
    weather = next(row for row in responses if row["prompt_id"] == "weather")
    assert weather["finish_reason"] == "tool_calls"
    assert weather["tool_calls"] == [
        {"name": "get_weather", "arguments": '{"city": "x", "days": 1}'}
    ]
    support = next(row for row in responses if row["prompt_id"] == "support")
    assert support["text"].startswith("tok0 tok1") and support["tool_calls"] == []


def test_a_change_is_compared_with_the_current_setup_and_its_answers_kept(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _chat_project(tmp_path, "project", tool_calling=True)
    change = ("--engine-arg", "kv-cache-dtype=fp8_e4m3")
    first = _analyse(project, capsys, *change)
    assert first.code == 0 and "No run of your current setup to compare with" in first.report

    assert _analyse(project, capsys).code == 0  # the current setup
    (project / "runs" / "broken").mkdir()
    (project / "runs" / "broken" / "run.json").write_text("{not json")
    after = _analyse(project, capsys, *change)

    assert after.code == 0 and "## Compared with your current setup" in after.report
    assert "Answers changed:" in after.report and "alt0" in after.report
    report = after.report
    assert report.index("## What changed vs your current setup") < report.index("## Diagnosis")
    assert "Change: --engine-arg kv-cache-dtype=fp8_e4m3" in report
    assert "- output " in report and " tok/s (" in report and "- TTFT p95 " in report
    assert "- answers: Answers changed" in report
    assert "- bottleneck: " in report and " → " in report
    assert "One run each: repeat both runs before trusting a difference of a few percent." in report
    assert "## What changed vs your current setup" in after.out
    answers = next((project / "compare").glob("*/answers.md"))
    assert stat.S_IMODE(answers.stat().st_mode) == 0o600 and "alt0" in answers.read_text()
    assert re.search(r"Tool calls identical for \d+ of \d+ prompts that call tools", report)

    other = _analyse(project, capsys, "--engine-arg", "kv-cache-dtype=fp8_e5m2")
    assert main(["compare", "--project", str(project), after.run_dir.name, other.run_dir.name]) == 0
    assert f"## Run {other.run_dir.name} compared with run {after.run_dir.name}" in (
        capsys.readouterr().out
    )


@pytest.mark.parametrize("recorded", ["run_0_3_3", "run_0_3_4"])
def test_a_run_from_an_earlier_release_is_still_a_baseline(
    recorded: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    old = project / "runs" / "20261005T000000Z-0330"
    shutil.copytree(Path(__file__).parent / recorded, old)
    record = json.loads((old / "run.json").read_text())
    record["snapshot_id"] = json.loads(capsys.readouterr().out)["snapshot_id"]
    (old / "run.json").write_text(json.dumps(record))
    after = _analyse(project, capsys, "--engine-arg", "max-num-seqs=3")
    assert after.code == 0
    assert f"## What changed vs your current setup (run {old.name})" in after.report
    assert main(["compare", "--project", str(project), old.name, after.run_dir.name]) == 0


def answers_jsonl_from(responses: list[dict]) -> str:
    """Recorded production answers: the text and tool calls, without a finish reason."""
    rows = ({k: row[k] for k in ("prompt_id", "text", "tool_calls")} for row in responses)
    return "\n".join(json.dumps(row) for row in rows) + "\n"


def test_require_equal_gates_changes_and_require_gpu_refuses_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _chat_project(tmp_path, "project", tool_calling=True)
    nothing = _analyse(
        project, capsys, "--engine-arg", "kv-cache-dtype=fp8_e4m3", "--require-equal"
    )
    assert nothing.code == 4 and "No run of your current setup" in nothing.report

    _analyse(project, capsys)
    same = _analyse(project, capsys, "--engine-arg", "enable-prefix-caching", "--require-equal")
    drift = _analyse(project, capsys, "--engine-arg", "kv-cache-dtype=fp8_e4m3", "--require-equal")
    assert same.code == 0 and "All answers identical to" in same.report
    assert "reproduced its own answers for" in same.report
    assert drift.code == 4 and "first difference at character" in drift.report

    recorded = tmp_path / "recorded.jsonl"
    recorded.write_text(answers_jsonl_from(same.responses))  # without finish_reason
    replay = _analyse(project, capsys, "--baseline-answers", str(recorded), "--require-equal")
    assert replay.code == 0 and "not applicable" in replay.report
    assert "cover 2 of 2 prompts" in replay.report
    assert "## What changed vs your recorded answers" in replay.report
    assert "like-for-like" not in replay.report
    changes = replay.report.split("## What changed vs your recorded answers")[1].split("\n## ")[0]
    assert " tok/s" not in changes
    assert "- bottleneck:" not in replay.report

    a10 = GpuInfo("NVIDIA A10", 24 * 2**30, 0, index="0", uuid="GPU-1", driver="580.95.05")
    monkeypatch.setattr("tensward.environment.detect_devices", lambda: (a10,))
    refused = _analyse(project, capsys, "--require-gpu", "A10G")
    assert refused.code == 2 and "gpu_mismatch" in refused.err and refused.run_dir is None
    assert "NVIDIA A10" in refused.err and "580.95.05" in refused.err


def test_init_refuses_chat_workloads_the_checkpoint_or_declared_api_cannot_take(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _chat_project(tmp_path, "ok", tool_calling=False)
    model = tmp_path / "ok" / "model"
    serving, prompts = tmp_path / "ok" / "chat-serving.json", tmp_path / "ok" / "chat.jsonl"
    capsys.readouterr()

    def init_message(project_name: str, prompts: Path) -> str:
        code = main(["init", "--project", str(tmp_path / project_name), "--model", str(model),
                     "--config", str(serving), "--prompts", str(prompts)])  # fmt: skip
        assert code != 0
        return json.loads(capsys.readouterr().err)["message"]

    write_prompts(tmp_path / "plain.jsonl", [{"id": "a", "prompt": "hello"}])
    assert "every record must match the declared api" in init_message(
        "p1", tmp_path / "plain.jsonl"
    )
    write_prompts(tmp_path / "bad.jsonl", [{"id": "a", "messages": SUPPORT_CHAT[:3]}])
    assert "must end with a user message" in init_message("p2", tmp_path / "bad.jsonl")
    bad_tool = {**WEATHER, "function": {"name": "f", "parameters": {"type": "string"}}}
    write_prompts(
        tmp_path / "tool.jsonl",
        [{"id": "a", "messages": SUPPORT_CHAT[-1:], "tools": [bad_tool]}],
    )
    assert '"type": "object"' in init_message("p3", tmp_path / "tool.jsonl")
    (model / "tokenizer_config.json").write_text("{}")
    (model / "chat_template.jinja").unlink()
    assert "no chat template" in init_message("p4", prompts)


def test_ngram_speculation_under_structured_output_is_offered_last_with_a_warning() -> None:
    measurement = Measurement(1, 0, tpot_p50_ms=10.0)
    unique = tuple(" ".join(f"w{i}x{j}" for j in range(300)) for i in range(8))
    facts = WorkloadFacts(prompts=unique, output_tokens=128, context_limit=4096, structured=True)
    found, _ = applicable(
        PLAYBOOK, situation(measurement, facts, Settings()), allow_quality_changes=False, near=True
    )
    offered = next(s for s in found if s.entry.name == "ngram-speculation")
    assert offered.tier == "could_help" and offered.near
    assert "grammar-constrained" in offered.reason and "--require-equal" in (offered.cost or "")


@pytest.mark.parametrize(
    ("inflight", "seqs", "extra", "expected"),
    [
        (12, None, {}, 16),
        (None, 100, {}, 128),
        (12, 8, {}, 8),
        (None, None, {}, None),
        (None, 300, {}, None),
        (3, None, {"--speculative-config": NGRAM_CONFIG}, 16),
        (12, None, {"--max-cudagraph-capture-size": "64"}, None),
    ],
)
def test_cuda_graphs_are_captured_only_up_to_the_concurrency_cap(
    inflight: int | None, seqs: int | None, extra: dict[str, str], expected: int | None
) -> None:
    facts = WorkloadFacts(
        prompts=("p",), output_tokens=128, context_limit=4096, max_inflight=inflight
    )
    settings = Settings(max_concurrent_requests=seqs, cuda_graphs=False, extra_args=extra)
    (entry,) = [e for e in PLAYBOOK if e.name == "cuda-graphs"]
    applied = entry.apply(situation(Measurement(1, 0), facts, settings))
    assert applied.cuda_graphs
    flag = json.loads(applied.extra_args.get("--compilation-config", "{}"))
    assert flag.get("max_cudagraph_capture_size") == expected


def _graph_case(**changes: Any) -> tuple[Measurement, WorkloadFacts, Settings]:
    """The hybrid 27B INT4 model on a 22.06 GiB card (vLLM's own total), eager: 0.53 GiB of KV for
    4,551 tokens."""
    fields = dict(
        kv_available_gib=0.53, kv_capacity_tokens=4551, max_context_len=4096,
        gpu_memory_gib=22.06, max_prompt_tokens=500, hybrid_cache=False, kv_blocks=None,
        peak_running=None,
    )  # fmt: skip
    facts = dict(max_inflight=16, media_encoders=True, encoder_gib=0.87)
    settings = dict(cuda_graphs=False, kv_memory_fraction=0.92)
    for target in (fields, facts, settings):
        target.update({k: changes.pop(k) for k in list(changes) if k in target})
    return (
        Measurement(1, 0, **fields),
        WorkloadFacts(prompts=("p",), output_tokens=55, context_limit=8192, **facts),
        Settings(**settings),
    )


def _graphs(measurement: Measurement, facts: WorkloadFacts, settings: Settings) -> Any:
    found, held = applicable(
        PLAYBOOK, situation(measurement, facts, settings), allow_quality_changes=False
    )
    return dict((s.entry.name, s.reason) for s in found).get("cuda-graphs"), dict(
        (e.name, why) for e, why in held
    )


def test_cuda_graphs_on_a_near_full_card_come_with_the_memory_they_need() -> None:
    (entry,) = [e for e in PLAYBOOK if e.name == "cuda-graphs"]
    case = _graph_case()
    reason, _ = _graphs(*case)
    assert "`no-media-encoders` frees" in reason
    applied = entry.apply(situation(*case))
    assert applied.cuda_graphs and applied.media_inputs is False
    assert "--compilation-config" in applied.extra_args
    found, _ = applicable(PLAYBOOK, situation(*case), allow_quality_changes=False)
    assert ranked(found)[0].entry.name == "cuda-graphs"
    engine = ENGINES["vllm"]
    assert engine.engine_args_between(case[2], engine.consistent(case[2], applied)[0]) == [
        "no-enforce-eager",
        "language-model-only",
        'compilation-config={"max_cudagraph_capture_size": 16}',
    ]
    small_tower = _graph_case(encoder_gib=0.01, kv_memory_fraction=0.95)
    assert "`no-media-encoders`" not in _graphs(*small_tower)[0]

    text_only = _graph_case(media_encoders=False)
    assert "`more-kv-memory` frees" in _graphs(*text_only)[0]
    assert entry.apply(situation(*text_only)).kv_memory_fraction == 0.95

    trimmed = _graph_case(media_encoders=False, kv_memory_fraction=0.95)
    assert "`trim-max-context-len` lowers" in _graphs(*trimmed)[0]
    assert entry.apply(situation(*trimmed)).max_context_len == 768

    stuck = _graph_case(media_encoders=False, kv_memory_fraction=0.95, max_prompt_tokens=3500)
    reason, held = _graphs(*stuck)
    assert reason is None and "no change that frees memory applies" in held["cuda-graphs"]


def test_cuda_graphs_on_a_hybrid_cache_limit_the_sequences_to_the_blocks_left() -> None:
    (entry,) = [e for e in PLAYBOOK if e.name == "cuda-graphs"]
    engine = ENGINES["vllm"]
    case = _graph_case(hybrid_cache=True, kv_blocks=10, peak_running=2)
    reason, _ = _graphs(*case)
    assert "the Try sets it to 5" in reason and "estimate, not measured" in reason
    applied = entry.apply(situation(*case))
    assert applied.max_concurrent_requests == 5
    assert engine.engine_args_between(case[2], engine.consistent(case[2], applied)[0]) == [
        "max-num-seqs=5",
        "no-enforce-eager",
        "language-model-only",
        'compilation-config={"max_cudagraph_capture_size": 8}',
    ]

    busy = _graph_case(hybrid_cache=True, kv_blocks=10, peak_running=6)
    reason, held = _graphs(*busy)
    assert reason is None
    assert "the run reached 6 running requests" in held["cuda-graphs"]

    plain = _graph_case(kv_blocks=14283, peak_running=2)
    assert entry.apply(situation(*plain)).max_concurrent_requests is None


def _hybrid_kv_case(**changes: Any) -> tuple[Measurement, WorkloadFacts, Settings]:
    """v034d: graphs on, cap 4 and capture 4, 17 blocks of 1568 tokens in 0.82 GiB, 4 running at
    100%, the longest prompt 427 tokens plus 128 output."""
    fields = dict(
        kv_available_gib=0.82, kv_capacity_tokens=7736, max_context_len=4096, gpu_memory_gib=22.06,
        max_prompt_tokens=427, hybrid_cache=True, kv_blocks=17, kv_block_tokens=1568,
        peak_running=4, peak_kv_usage=1.0, peak_waiting=28,
    )  # fmt: skip
    setup = dict(
        max_concurrent_requests=4, kv_memory_fraction=0.92, cuda_graphs=True,
        extra_args={"--compilation-config": '{"max_cudagraph_capture_size": 4}'},
    )  # fmt: skip
    for target in (fields, setup):
        target.update({k: changes.pop(k) for k in list(changes) if k in target})
    facts = WorkloadFacts(prompts=("p",), output_tokens=128, context_limit=4096)
    return Measurement(160, 0, **fields), facts, Settings(**setup)


def _levers(case: tuple[Measurement, WorkloadFacts, Settings]) -> tuple[dict, dict]:
    found, held = applicable(PLAYBOOK, situation(*case), allow_quality_changes=True)
    return {s.entry.name: s for s in found}, {entry.name: why for entry, why in held}


def test_kv_levers_on_a_hybrid_cache_count_whole_blocks() -> None:
    engine = ENGINES["vllm"]
    case = _hybrid_kv_case()
    offered, held = _levers(case)
    proposed, _ = engine.consistent(
        case[2], offered["more-kv-memory"].entry.apply(situation(*case))
    )
    assert sorted(engine.engine_args_between(case[2], proposed)) == [
        'compilation-config={"max_cudagraph_capture_size": 8}',
        "gpu-memory-utilization=0.95",
        "max-num-seqs=7",
    ]
    assert "raises max_concurrent_requests from 4 to 7" in offered["more-kv-memory"].reason
    assert "state page" in held["trim-max-context-len"]
    assert "state page" in held["fp8-kv-cache"]  # 555 tokens fit in one 1568-token block

    offered, held = _levers(_hybrid_kv_case(max_prompt_tokens=3000))
    assert "longer than 1568 tokens" in offered["fp8-kv-cache"].reason

    offered, held = _levers(_hybrid_kv_case(kv_memory_fraction=0.95))  # the client's setup
    assert "more-kv-memory" not in offered
    assert "already 0.95" in held["more-kv-memory"] and "state dtype" in held["more-kv-memory"]

    unset = _hybrid_kv_case(kv_memory_fraction=None)  # vLLM 0.30's default share is 0.92
    assert offered_cap(unset) == 7

    for unmeasured in (_hybrid_kv_case(peak_running=None), _hybrid_kv_case(kv_blocks=None)):
        assert offered_cap(unmeasured) == 4

    offered, held = _levers(_hybrid_kv_case(hybrid_cache=False))
    assert "fp8-kv-cache" in offered
    assert "whatever max_context_len is" in held["trim-max-context-len"]


def offered_cap(case: tuple[Measurement, WorkloadFacts, Settings]) -> int | None:
    more = _levers(case)[0]["more-kv-memory"]
    return more.entry.apply(situation(*case)).max_concurrent_requests


def test_cuda_graphs_stay_plain_where_the_card_has_room() -> None:
    (entry,) = [e for e in PLAYBOOK if e.name == "cuda-graphs"]
    roomy = _graph_case(
        kv_available_gib=3.37, kv_capacity_tokens=47786, max_context_len=8192,
        kv_memory_fraction=0.95,
    )  # fmt: skip
    reason, held = _graphs(*roomy)
    assert reason and "combined" not in reason and not held
    assert entry.apply(situation(*roomy)).media_inputs is None


def test_a_raised_concurrency_widens_the_graph_bound_of_the_cuda_graphs_suggestion() -> None:
    facts = WorkloadFacts(prompts=("p",), output_tokens=128, context_limit=4096, max_inflight=12)
    (entry,) = [e for e in PLAYBOOK if e.name == "cuda-graphs"]
    before = entry.apply(situation(Measurement(1, 0), facts, Settings(max_concurrent_requests=16)))
    after, note = ENGINES["vllm"].consistent(
        before, dataclasses.replace(before, max_concurrent_requests=64)
    )
    assert json.loads(after.extra_args["--compilation-config"])["max_cudagraph_capture_size"] >= 64
    assert note

    drafting = dataclasses.replace(
        before, extra_args={**before.extra_args, "--speculative-config": NGRAM_CONFIG}
    )
    widened, note = ENGINES["vllm"].consistent(before, drafting)
    assert note and ENGINES["vllm"].unstartable(before, widened) is None
    assert ENGINES["vllm"].unstartable(before, drafting) is not None


def test_analyse_with_a_bad_image_is_a_one_line_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip

    code = main(["analyse", "--project", str(project), "--image=-x"])

    err = capsys.readouterr().err
    assert code == 1 and err.strip().splitlines()[-1].endswith(
        "'-x' is not a docker image reference"
    )
    assert "Traceback" not in err


def test_analyse_without_a_current_setup_measures_the_engine_defaults(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The config declares no serving fields, so nothing is passed to the engine (the fake
    defaults to 256 sequences, room for 9 in its KV cache) and the entries fall back to what
    the engine reported."""
    model, _, prompts = make_registration_inputs(tmp_path)
    case = {"weight_precision": "bf16", "activation_dtype": "bfloat16"}
    workload = {**REGISTRATION_CONFIG["workload"], "request_count": 32}
    workload["arrival"] = {"kind": "closed_loop", "concurrency": 16}
    config = write_config(
        tmp_path / "defaults.json", {**REGISTRATION_CONFIG, "case": case, "workload": workload}
    )
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip

    result = _analyse(project, capsys)

    assert result.code == 0
    assert "- source: engine defaults (no current setup provided)" in result.report
    assert "- settings: the engine's defaults" in result.report
    report = result.report
    assert "`more-kv-memory`: " in report and "kv_memory_fraction is the engine default" in report
    metrics = json.loads((result.run_dir / "metrics.json").read_text())
    assert metrics["max_context_len"] == 4096  # what the engine chose, as it reported it
    argv = json.loads((result.run_dir / "serve.json").read_text())["identity"]["argv"]
    assert not {"--max-num-seqs", "--max-model-len", "--gpu-memory-utilization"} & set(argv)


def test_trust_remote_code_can_only_come_from_the_registered_setup() -> None:
    engine = ENGINES["vllm"]
    with pytest.raises(AnalyseFailure, match="register the project again"):
        apply_engine_args(engine, Settings(), ["trust-remote-code"])
    trusted = engine.parse_setup("vllm serve /m --trust-remote-code").settings
    assert apply_engine_args(engine, trusted, ["max-num-seqs=4"]).max_concurrent_requests == 4


def test_trace_steps_are_the_busiest_worker_s_step_annotations(tmp_path: Path) -> None:
    def annotation(pid: int, name: str, ts: float) -> dict:
        return {"ph": "X", "cat": "user_annotation", "name": name, "pid": pid, "tid": 1,
                "ts": ts, "dur": 5.0}  # fmt: skip

    steps = [
        "execute_96_context_1(sq64sk64sqsq4096sqsk4096)_generation_32(sq32sk9000sqsq32sqsk9000)",
        "execute_context_0(0)_generation_33(33)",
        "execute_context_1(64)_generation_32(32)",
    ]
    kernel = {"ph": "X", "cat": "kernel", "name": "gemm", "pid": 0, "tid": 7, "ts": 0.0}
    events = [{**kernel, "dur": 1.0}]
    for worker in (10, 11):  # tensor parallel: both workers annotate every step
        events += [annotation(worker, name, 100.0 * i) for i, name in enumerate(steps)]
    empty = [  # steps that scheduled no token run no forward pass
        "execute_0_context_0(sq0sk0sqsq0sqsk0)_generation_0(sq0sk0sqsq0sqsk0)",
        "execute_context_0(0)_generation_0(0)",
    ]
    events += [annotation(10, name, 1000.0 + i) for i, name in enumerate(empty)]
    events += [annotation(12, "execute_model", 0.0)]
    events += [annotation(12, "gpu_model_runner: forward", 1.0)]
    path = tmp_path / "worker.pt.trace.json"
    path.write_text(json.dumps({"traceEvents": events}))

    tracing = VLLM.tracing
    assert tracing is not None
    assert read_trace([path], tracing.step_scope, step_pattern=tracing.step_pattern).steps == 3
    assert read_trace([path], tracing.step_scope).steps is None


def test_a_start_up_wave_that_outlasts_the_run_is_measured_over_the_whole_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two clients. The start-up wave offers prompts 4 and 0: the long answer keeps decoding
    while the other start-up request and the four short declared requests come and go through
    the other client's slot. The fake server's KV cache holds one sequence, so each request
    admitted beside the long one counts a preemption. The long answer keeps the whole-run
    window above the 2 s short-window line (about 2.3 s)."""
    model, _, _ = make_registration_inputs(tmp_path)
    document = json.loads((model / "config.json").read_text())
    (model / "config.json").write_text(json.dumps({**document, "model_type": "qwen2"}))
    rows = [
        {"id": f"short-{i}", "messages": [{"role": "user", "content": f"Name a colour {i}."}],
         "max_tokens": 2}
        for i in range(4)
    ]  # fmt: skip
    rows.append(
        {"id": "long", "messages": [{"role": "user", "content": "Tell a story."}],
         "max_tokens": 1500}
    )  # fmt: skip
    prompts = write_prompts(tmp_path / "outlast.jsonl", rows)
    case = {**REGISTRATION_CONFIG["case"], "max_num_seqs": 2, "gpu_memory_utilization": 0.1}
    workload = {**REGISTRATION_CONFIG["workload"], "api": "chat", "request_count": 4}
    workload["arrival"] = {"kind": "closed_loop", "concurrency": 2}
    config = write_config(
        tmp_path / "outlast.json", {**REGISTRATION_CONFIG, "case": case, "workload": workload}
    )
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip

    result = _analyse(project, capsys)

    assert result.code == 0
    assert "the start-up wave had not finished when the last request was sent" in result.err
    metrics = result.metrics
    assert metrics["window"]["kind"] == "whole_run"
    assert "the start-up wave outlasted the last request" in " ".join(metrics["checks"])
    assert metrics["preemptions"] >= 4  # every admission beside the long answer
    assert metrics["peak_kv_usage"] == 1.0 and metrics["peak_running"] >= 1  # polled to the end
    assert metrics["frontend_cpu_cores"] > 0
    kv = next(f for f in metrics["diagnosis"]["findings"] if f["bottleneck"] == "kv_capacity")
    assert kv["state"] == "critical"
    assert kv["evidence"].endswith("over 6 requests, the start-up wave's 2 included")
    assert "outside these figures, inside the engine's" in result.report
