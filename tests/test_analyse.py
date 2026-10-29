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
import time
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest
from image_fixtures import write_images
from interrupts import signal_once_present
from pydantic import ValidationError
from registration_fixtures import (
    QUANTIZED_CHECKPOINTS,
    REGISTRATION_CONFIG,
    SAFETENSORS_NAME,
    case_for,
    make_checkpoint,
    make_gemma4_checkpoint,
    make_image_workload,
    make_mxfp4_checkpoint,
    make_registration_inputs,
    probe_registration,
    registered_project,
    safetensors_from_tensors,
    write_config,
    write_prompts,
)
from test_playbook import situation

from tensward import capacity, inputs, progress, window
from tensward import project as project_module
from tensward.ceilings import gpu_spec
from tensward.classify import Diagnosis
from tensward.cli import main
from tensward.client import RequestPlan, run_workload
from tensward.context import AnalysisContext, active
from tensward.engines import ENGINES
from tensward.engines.vllm import VLLM
from tensward.engines.vllm.capabilities import speculative_config
from tensward.engines.vllm.predicates import NGRAM
from tensward.errors import AnalyseFailure, PreflightError
from tensward.extensions import API_VERSION, Extender
from tensward.history import command_args
from tensward.images import ImageSource
from tensward.inputs import PromptEntry
from tensward.measurement import Measurement
from tensward.platforms import Device as GpuInfo
from tensward.playbook import Suggestion, WorkloadFacts, ranked
from tensward.predict import Prediction
from tensward.profiling.trace import read_trace
from tensward.progress import ProgressEvent
from tensward.project import apply_engine_args, load_project, sample_of
from tensward.rebuild import _from_json, rebuild
from tensward.report.build import FEEDBACK_LINE
from tensward.runtime import LocalProcessRuntime, RunningServer, ServeSpec, process_start_time
from tensward.settings import Availability, Settings
from tensward.suggest import applicable
from tensward.token_cache import cached_tokens
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
FAKE_SMI = Path(__file__).resolve().parent / "fake_nvidia_smi.py"
SYNTHETIC_TRACE = Path(__file__).resolve().parent / "synthetic_trace.json"


def test_analyse_runs_end_to_end_against_the_fake_server(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "FAKE_QUANT_KERNEL_LINE", "INFO Using MarlinLinearKernel for AutoAWQMarlinLinearMethod"
    )
    monkeypatch.setattr("tensward.ceilings.detect_gpus", lambda: (("NVIDIA L4", 23034),))
    for module in ("project", "measure"):
        monkeypatch.setattr(
            f"tensward.{module}.detect_devices", lambda: (GpuInfo("NVIDIA L4", 23034 * 2**20, 0),)
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
    assert "current setup (declared in config) + this run's overrides: no-async-scheduling, " in out
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
    metrics = json.loads((run_dir / "metrics.json").read_text())
    recorded = json.loads((run_dir / "run.json").read_text())
    assert recorded["step_budget_tokens"] == metrics["step_budget_tokens"] == 2048
    assert recorded["compile_cache"] == metrics["compile_cache"] == "warm"
    # The registration config serves at max_num_seqs 1: one lane of
    # about 600 prompt tokens cannot fill a 2048-token step, so no R and no share.
    assert recorded["prompt_capacity"]["source"] == "cannot-fill"
    probe = "prompt capacity: not measured (the prompts are too short to fill an engine step)"
    assert probe in (run_dir / "report.md").read_text()
    # None fields are left out of metrics.json (LEFT_OUT_AT_DEFAULT), so read them with .get
    assert metrics.get("prompt_capacity_tok_s") is None and metrics.get("prompt_work_share") is None
    assert metrics["scheduled_tokens_per_step"] > 0 and metrics["step_ms"] > 0
    assert metrics["mean_waiting"] is not None and metrics["mean_kv_usage"] is not None
    assert metrics["mean_in_flight"] > 0 and metrics["server_ttft_p50_ms"] > 0
    # A dedicated project for the probe: cap 8, four system-first multi-turn
    # chat records of about 600 words with tools; probed on the first run, cached on the second.
    probe_project = probe_registration(tmp_path, model)
    capacities = []
    for _ in range(2):
        assert main(["analyse", "--project", str(probe_project), "--runtime", "local",
                     "--local-command", command, "--ready-timeout", "30"]) == 0  # fmt: skip
        newest = max((probe_project / "runs").iterdir())
        capacities.append(json.loads((newest / "run.json").read_text())["prompt_capacity"])
    first, second = capacities
    assert first["source"] == "probed" and first["completions"] >= 20 and first["tok_s"] > 0
    assert first["median_prompt_tokens"] >= 0.8 * first["target_tokens"]
    assert second["source"] == "cached" and second["tok_s"] == first["tok_s"]
    # Registered with --current "vllm serve <model>": no dtype and no cap set, key still known
    current = probe_registration(tmp_path / "current", model, current="vllm serve {model}")
    assert main(["analyse", "--project", str(current), "--runtime", "local",
                 "--local-command", command, "--ready-timeout", "30"]) == 0  # fmt: skip
    newest = max((current / "runs").iterdir())
    assert (
        json.loads((newest / "run.json").read_text())["prompt_capacity"]["source"]
        != "unknown-setup"
    )

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
    assert all(
        row["first_chunk_ns"] <= row["first_content_ns"]
        for row in requests
        if row["outcome"] == "success"
    )
    summary = (run_dir / "report.md").read_text()
    assert summary.startswith("# Tensward analysis")
    ran = next(line for line in summary.splitlines() if line.startswith("Ran: "))
    assert ran.startswith("Ran: vLLM 0.30.0 (local command python")
    assert ran.endswith("; checkpoint: safetensors, awq int4 group 128")
    assert "\nBottleneck: " in summary and "\n## What to try next\n" in summary
    assert summary.endswith(f"\n{FEEDBACK_LINE}\n")
    assert not re.search(r"\b(?:[A-Z]{2,4}-\d+|A\d+\.\d+)\b", summary)  # no private catalog IDs
    assert json.loads((run_dir / "run.json").read_text())["format"] == "hf-safetensors"
    sections = sections_of(run_dir)
    assert list(sections) == [
        "header", "diagnosis", "next_steps", "setup", "requests", "performance", "engine",
        "defaults", "ceilings", "quantization", "context", "checks", "trace", "counters",
        "answers", "keep", "feedback",
    ]  # fmt: skip
    assert {"source", "settings", "measured", "fails_requests"} <= set(sections["setup"])
    assert sections["requests"][-3:] == ["repeated_prompts", "failures", "failure"]
    assert {"failed", "ceiling", "gpu_busy"} <= set(sections["header"])
    assert {"startup_wave", "goodput", "slo_attainment"} <= set(sections["performance"])
    assert "kv_capacity" in sections["engine"] and "gate.fit" in sections["diagnosis"]
    assert "cant_tell" in sections["diagnosis"]
    keys = sections["next_steps"]
    if "for.host_overhead" in keys:  # never an empty group: async scheduling is off here
        assert keys[keys.index("for.host_overhead") + 1] == "async-scheduling"
    assert {"fit-context", "prefix-caching", "not_applicable", "serve_hint"} <= set(
        sections["next_steps"]
    )
    assert sections["context"] == ["too_long", "longest"]  # 3032 tokens fit the model's 4096
    assert sections["trace"] == ["caveat", "window", "gaps", "analysis_hint"]
    assert sections["counters"] == ["unavailable"]  # --counters implies --trace; no ncu here
    report = json.loads((run_dir / "report.json").read_text())
    steps = next(s for s in report["sections"] if s["id"] == "next_steps")["blocks"]
    async_steps = [
        b for b in steps if b["kind"] == "suggestion" and b["name"] == "async-scheduling"
    ]
    assert [b["evidence"] for b in async_steps] == ["moderate"]  # the built-in entry, no plugin's
    commands = [b["command"] for b in steps if b["kind"] == "suggestion" and b["command"]]
    assert commands and all(c.startswith("tensward analyse --project ") for c in commands)
    assert all("--engine-arg " in c for c in commands)
    for block in steps:
        if block["kind"] == "suggestion":
            seconds = (block["test_seconds"], block["start_seconds"])
            assert all(isinstance(v, float) and v > 0 for v in seconds) == bool(block["command"])
    diagnosis = metrics["diagnosis"]
    assert [f["bottleneck"] for f in diagnosis["findings"]] == [
        "load_limited", "queueing", "kv_capacity", "prefill", "step_budget",
        "prefill_stalls_decode", "decode_bandwidth", "gpu_compute", "gpu_saturated",
        "host_overhead", "frontend_cpu", "long_context", "speculation",
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


ASKED = PromptEntry(id="a", prompt="word")
ANSWERED = PromptEntry(
    id="a",
    messages=({"role": "user", "content": "word"}, {"role": "assistant", "content": "word"}),
)


@pytest.mark.parametrize(
    ("entry", "tokens", "counted", "source"),
    [
        (ASKED, 600, None, "cannot-fill"),  # 4 lanes of 600 tokens fill one budget, not 1.5
        (ANSWERED, 1000, None, "not-user-turn"),  # the last turn cannot be resized
        (ASKED, 1000, 1300, "long-prompts"),  # the engine counted over 1.2 times the target
    ],
)
def test_the_probe_refuses_a_setup_it_cannot_measure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: PromptEntry, tokens: int,
    counted: int | None, source: str,
) -> None:  # fmt: skip
    key = capacity.Key("m", "gpu", 2048, 4, 1, "auto", True, "none", True, "0.30.0", "auto")
    monkeypatch.setattr(capacity, "PROBE_SECONDS", 0.2)

    async def send(*_: object) -> tuple[int, int]:
        assert counted is not None, "no probe request is sent"
        return counted, 0

    async def computed() -> None:
        return None

    result = asyncio.run(
        capacity.probe(
            project=tmp_path,
            key=key,
            entries=[entry],
            prompt_tokens=[tokens],
            words_per_token=1.0,
            send=send,
            computed=computed,
            run_id="r",
        )  # fmt: skip
    )
    assert (result.source, result.tok_s) == (source, None)


def _probe(
    project: Path, key: capacity.Key, entry: PromptEntry, counted: int, tokens: int = 1000,
    computed: tuple[float, float] | None = None,
) -> tuple[capacity.CapacityResult, int]:  # fmt: skip
    """One probe of ``entry`` (counted by the client as ``tokens``); the result and the number of
    requests it sent."""
    sent: list[int] = []
    scrapes = iter(computed or (None, None))

    async def send(*_: object) -> tuple[int, int]:
        sent.append(1)
        return counted, 0

    async def scrape() -> float | None:
        return next(scrapes)

    result = asyncio.run(
        capacity.probe(
            project=project,
            key=key,
            entries=[entry],
            prompt_tokens=[tokens],
            words_per_token=1.0,
            send=send,
            computed=scrape,
            run_id="r",
        )  # fmt: skip
    )
    return result, len(sent)


def test_a_prompt_shaped_failure_is_reused_whatever_the_serving_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(capacity, "PROBE_SECONDS", 0.2)
    key = capacity.Key("m", "gpu", 2048, 4, 1, "auto", True, "none", True, "0.30.0", "auto")
    first, probed = _probe(tmp_path, key, ASKED, counted=1300)
    assert (first.source, probed > 0) == ("long-prompts", True)
    other = dataclasses.replace(key, cap=8, kv_cache_dtype="fp8", quantization="fp8")
    again, sent = _probe(tmp_path, other, ASKED, counted=1300)
    assert (again.source, again.tok_s, again.key, sent) == ("long-prompts", None, other, 0)
    _, sent = _probe(tmp_path, other, PromptEntry(id="b", prompt="word"), counted=1300)
    assert sent > 0
    _, sent = _probe(tmp_path, dataclasses.replace(key, model="n"), ASKED, counted=1300)
    assert sent > 0


def test_a_cached_prefix_failure_is_reused_only_on_the_whole_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(capacity, "PROBE_SECONDS", 0.2)
    key = capacity.Key("m", "gpu", 2048, 4, 1, "auto", True, "none", True, "0.30.0", "auto")
    first, _ = _probe(tmp_path, key, ASKED, counted=1000, computed=(0.0, 1.0))
    assert first.source == "cached-prefix"
    again, sent = _probe(tmp_path, key, ASKED, counted=1000)
    assert (again.source, sent) == ("cached-prefix", 0)
    _, sent = _probe(tmp_path, dataclasses.replace(key, prefix_caching=False), ASKED, counted=1300)
    assert sent > 0


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


def _chat_project(
    tmp_path: Path,
    name: str,
    *,
    tool_calling: bool,
    current: str | None = None,
    refused: bool = False,
    **workload: Any,
) -> Path:
    """Register a chat workload (one plain chat, one offering a tool) on a Qwen2-family model;
    with ``current``, from that ``--current`` command (``{model}`` is the checkpoint), which
    ``init`` may refuse (``refused``); ``workload`` overrides its fields."""
    model, _, _ = make_registration_inputs(tmp_path / name)
    document = json.loads((model / "config.json").read_text())
    (model / "config.json").write_text(json.dumps({**document, "model_type": "qwen2"}))
    config = {
        "case": {**REGISTRATION_CONFIG["case"], "tool_calling": tool_calling},
        "workload": {**REGISTRATION_CONFIG["workload"], "api": "chat", **workload},
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
    imported = ["--current", current.format(model=model)] if current else []
    code = main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(serving), "--prompts", str(prompts), *imported])  # fmt: skip
    assert (code != 0) is refused
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


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.is_file() else []


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
        [json.loads(line) for line in _lines(run_dir / "responses.jsonl")],
    )


SATURATED_ROW = "99,69.5,70.00,1230,1590,Active,Not Active,Not Active"


def test_analyse_samples_the_gpu_over_the_measured_window(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tensward.platforms.nvidia.SAMPLE_COMMAND", (sys.executable, str(FAKE_SMI)))
    monkeypatch.setenv("FAKE_SMI_ROW", SATURATED_ROW)
    monkeypatch.setenv("FAKE_SMI_OLD", "1")  # the sampler falls back to the older field names
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    capsys.readouterr()
    run = _analyse(project, capsys)
    assert run.code == 0
    state = run.metrics["diagnosis"]["gpu_state"]
    assert state["unavailable"] is None and state["samples"] >= 5 and state["index"] == 0
    shown = (state["utilization"], state["power_limit_w"], state["sm_max_mhz"])
    assert shown == (99.0, 70.0, 1590.0)
    assert state["power_capped_share"] == 1.0 and state["thermal_share"] == 0.0
    memory = json.loads((run.run_dir / "window.json").read_text())["memory"]
    assert memory == {"peak_used_mib": 1024.0, "total_mib": 24576.0}
    findings = {f["bottleneck"]: f for f in run.metrics["diagnosis"]["findings"]}
    assert findings["gpu_saturated"]["state"] == "critical"
    assert "at its power limit" in findings["gpu_saturated"]["evidence"]


# --- answer options ---

FAKE_VLLM = {
    "vllm/__init__.py": "__version__ = '0.30.0'\n",
    "vllm/plugins/__init__.py": "def load_general_plugins():\n    pass\n",
    "vllm/config/__init__.py": "",
    "vllm/config/cache.py": (
        "from typing import Literal\nCacheDType = Literal['auto', 'float16', 'fp8']\n"
    ),
    "vllm/model_executor/__init__.py": "",
    "vllm/model_executor/layers/__init__.py": "",
    "vllm/model_executor/layers/quantization/__init__.py": (
        "from typing import Literal\n"
        "QuantizationMethods = Literal['fp8', 'awq', 'gptq', 'compressed-tensors', 'odd']\n"
        "QUANTIZATION_METHODS = ['fp8', 'awq', 'gptq', 'compressed-tensors', 'odd']\n"
        "FLOORS = {'fp8': 75, 'awq': 75, 'gptq': 60, 'compressed-tensors': 70}\n"
        "class _Config:\n"
        "    def __init__(self, name):\n        self.name = name\n"
        "    def get_min_capability(self):\n        return FLOORS[self.name]\n"
        "def get_quantization_config(name):\n    return _Config(name)\n"
    ),
}  # fmt: skip


def test_analyse_reads_what_the_engine_can_run_without_loading_a_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, text in FAKE_VLLM.items():
        (tmp_path / "fake" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "fake" / name).write_text(text)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "fake"))
    monkeypatch.setattr("tensward.platforms.nvidia.CAPABILITY_COMMAND",
                        (sys.executable, "-c", "print('0, 8.6, NVIDIA A10G')"))  # fmt: skip
    project = registered_project(tmp_path)
    done = _analyse(project, capsys)
    assert done.run_dir is not None
    assert "reading which quantization methods the engine supports" in done.err
    options = json.loads((done.run_dir / "run.json").read_text())["answer_options"]
    assert options["read"] is True and options["capability"] == 8.6
    assert options["on_the_fly"] == {"fp8": "weight-only (W8A16)"} and options["kv_cache"] == []
    assert options["unjudged"] == ["odd"]
    monkeypatch.delenv("PYTHONPATH")
    plain = _analyse(project, capsys)
    assert plain.run_dir is not None
    failed = json.loads((plain.run_dir / "run.json").read_text())["answer_options"]
    assert failed["read"] is False and "vllm" in failed["why"]


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


def _analyse_counting_tokens(project: Path, *extra: str) -> list[int | None]:
    command = shlex.join([sys.executable, str(FAKE_SERVER)])
    assert main(["analyse", "--project", str(project), "--runtime", "local", *extra,
                 "--local-command", command, "--ready-timeout", "30"]) == 0  # fmt: skip
    run_dir = sorted((project / "runs").iterdir())[-1]
    rows = (run_dir / "requests.jsonl").read_text().splitlines()
    return [json.loads(row)["prompt_tokens"] for row in rows]


def test_a_second_step_counts_no_prompt_tokens_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "tok.log"
    monkeypatch.setenv("FAKE_TOKENIZE_LOG", str(log))
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    first = _analyse_counting_tokens(project)
    counted = len(log.read_text().splitlines())
    assert counted == len(prompts.read_text().splitlines())
    second = _analyse_counting_tokens(project, "--engine-arg", "max-num-seqs=8")
    assert len(log.read_text().splitlines()) == counted
    assert first == second
    assert stat.S_IMODE((project / "cache" / "token_ids.json").stat().st_mode) == 0o600


def test_a_changed_tokenizer_misses_the_token_cache(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer.json").write_text("{}")
    artifact = SimpleNamespace(
        path=model, metadata=SimpleNamespace(fingerprint=SimpleNamespace(value="f"))
    )
    runtime = SimpleNamespace(kind="local", target="vllm")
    asked: list[int] = []

    class Tokenizing:
        def availability(self, kind: str, target: str) -> Availability:
            return Availability(True, "0.30.0", "")

        async def prompt_token_ids(self, *_: object) -> tuple[int, ...]:
            asked.append(1)
            return (1, 2, 3)

    async def count() -> None:
        cache = cached_tokens(
            Tokenizing(), runtime, tmp_path, SimpleNamespace(artifact=artifact), []
        )
        assert await cache.count_prompt_tokens(None, "", "m", {"prompt": "a b c"}) == 3
        cache.save()

    asyncio.run(count())
    asyncio.run(count())
    assert len(asked) == 1
    (model / "tokenizer.json").write_text('{"changed": true}')
    asyncio.run(count())
    assert len(asked) == 2


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
        "Note", "PhaseStarted", "ServerLoading", "ServerReady", "WindowOpened", "RequestsDone",
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


def test_an_explicit_kv_size_reaches_the_server_and_its_memory_lines_are_recorded(
    tmp_path: Path,
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    command = shlex.join([sys.executable, str(FAKE_SERVER)])

    assert main(["analyse", "--project", str(project), "--runtime", "local",
                 "--engine-arg", "kv-cache-memory-bytes=2770000000",
                 "--local-command", command, "--ready-timeout", "30"]) == 0  # fmt: skip

    (run_dir,) = (project / "runs").iterdir()
    log = (run_dir / "server.log").read_text()
    assert "GPU KV cache size: 100,000 tokens" in log and "Actual usage is" in log
    started = json.loads((run_dir / "window.json").read_text())["startup"]
    assert started["kv_pool_tokens"] == 100_000
    assert started["memory_usage"]["peak_activation_gib"] == 0.82
    assert json.loads((run_dir / "window.json").read_text())["memory"].keys() == {
        "peak_used_mib", "total_mib",
    }  # fmt: skip


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


def test_the_text_only_offer_carries_a_kv_size_and_its_run_keeps_a_normal_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path, "compressed-tensors-int4")
    make_gemma4_checkpoint(model)
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    command = shlex.join([sys.executable, str(FAKE_SERVER)])
    monkeypatch.setenv("FAKE_MEDIA_PROFILE", "1")  # a text-only start without a size profiles media

    assert main(["analyse", "--project", str(project), "--runtime", "local",
                 "--local-command", command, "--ready-timeout", "30"]) == 0  # fmt: skip

    (baseline,) = (project / "runs").iterdir()
    argv = json.loads((baseline / "serve.json").read_text())["identity"]["argv"]
    assert "--language-model-only" not in argv  # the baseline runs as the customer's command says
    (offer,) = [
        block for section in json.loads((baseline / "report.json").read_text())["sections"]
        for block in section.get("blocks", ()) if block.get("name") == "no-media-encoders"
    ]  # fmt: skip
    args = sorted(command_args(offer["command"]))
    assert args[0].startswith("kv-cache-memory-bytes=") and args[1:] == ["language-model-only"]

    assert main(["analyse", "--project", str(project), "--runtime", "local",
                 *(word for arg in args for word in ("--engine-arg", arg)),
                 "--local-command", command, "--ready-timeout", "30"]) == 0  # fmt: skip

    step = max((project / "runs").iterdir())
    pool = json.loads((step / "window.json").read_text())["startup"]["kv_pool_tokens"]
    baseline_pool = json.loads((baseline / "window.json").read_text())["startup"]["kv_pool_tokens"]
    assert pool == pytest.approx(baseline_pool, rel=0.02)  # the flag alone would get 9,668


def test_a_stopped_step_is_offered_again_with_a_size_and_not_the_same_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tensward.history import stopped_runs

    model, config, prompts = make_registration_inputs(tmp_path, "compressed-tensors-int4")
    make_gemma4_checkpoint(model)
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    command = shlex.join([sys.executable, str(FAKE_SERVER)])
    hangs = tmp_path / "hangs"
    hangs.write_text("#!/bin/sh\necho 'INFO Available KV cache memory: 0.01 GiB'\nexec sleep 60\n")
    hangs.chmod(0o755)
    monkeypatch.setenv("FAKE_MEDIA_PROFILE", "1")
    monkeypatch.setenv("FAKE_TOKEN_DELAY_S", "0.005")
    monkeypatch.setattr("tensward.platforms.nvidia.SAMPLE_COMMAND", (sys.executable, str(FAKE_SMI)))
    monkeypatch.setenv("FAKE_SMI_ROW", SATURATED_ROW)
    monkeypatch.setenv("FAKE_SMI_MEMORY", "23452,24576")  # the sampled peak caps the size
    run = ["analyse", "--project", str(project), "--runtime", "local", "--ready-timeout", "60"]
    assert main([*run, "--local-command", command]) == 0
    (baseline,) = (project / "runs").iterdir()
    assert json.loads((baseline / "metrics.json").read_text())["peak_kv_usage"] > 0

    def offer(run_dir: Path) -> str:
        (block,) = [
            b for section in json.loads((run_dir / "report.json").read_text())["sections"]
            for b in section.get("blocks", ()) if b.get("name") == "no-media-encoders"
        ]  # fmt: skip
        return str(block["command"])

    first = offer(baseline)
    args = [word for arg in sorted(command_args(first)) for word in ("--engine-arg", arg)]
    assert main([*run, "--local-command", str(hangs), *args]) != 0
    assert [stopped for _, stopped, _ in stopped_runs(project)] == [command_args(first)]
    capsys.readouterr()

    assert main(["report", baseline.name, "--project", str(project)]) == 0
    again = offer(baseline)
    assert again != first and command_args(again) != command_args(first)
    assert any(a.startswith("kv-cache-memory-bytes=") for a in command_args(again))

    assert main([*run, "--local-command", command]) == 0
    rerun = max(p for p in (project / "runs").iterdir() if (p / "run.json").exists())
    analysed = offer(rerun)
    assert main(["report", rerun.name, "--project", str(project)]) == 0
    assert analysed == offer(rerun) and command_args(analysed) == command_args(again)


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
    command = shlex.join(["env", "HF_TOKEN=hf_SECRET4", sys.executable, str(FAKE_SERVER)])
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
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
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

    # --current overrides the configuration's tool_calling: init refuses, names the fix and leaves
    # no registration; a project registered without that check is refused by analyse as well
    current = "vllm serve {model}"
    refused_init = _chat_project(
        tmp_path, "refused", tool_calling=True, current=current, refused=True
    )
    fix = "add `--enable-auto-tool-choice --tool-call-parser hermes` to your --current"
    assert fix in capsys.readouterr().err and not (refused_init / "project.json").exists()
    with monkeypatch.context() as patch:
        patch.setattr("tensward.cli._tool_call_refusal", lambda *args: None)
        imported = _chat_project(tmp_path, "imported", tool_calling=True, current=current)
    capsys.readouterr()
    notes = json.loads((imported / "project.json").read_text())["current_setup"]["notes"]
    assert any("tool_calling (configuration True; your command sets False)" in n for n in notes)
    root = tmp_path / "imported"
    again = ["init", "--project", str(imported), "--model", str(root / "model"),
             "--config", str(root / "chat-serving.json"), "--prompts", str(root / "chat.jsonl"),
             "--current", f"vllm serve {root / 'model'}"]  # fmt: skip
    assert main(again) != 0  # the notes are not the identity: it is the tools that are refused
    assert fix in capsys.readouterr().err
    refused = _analyse(imported, capsys)
    assert refused.code != 0 and fix in refused.err and not (imported / "runs").exists()
    assert main(["inspect", "--project", str(imported)]) == 0
    assert "tool calling is not enabled" in capsys.readouterr().err
    allowed = "vllm serve {model} --enable-auto-tool-choice --tool-call-parser hermes"
    served = _chat_project(tmp_path, "served", tool_calling=True, current=allowed)
    assert _analyse(served, capsys).code == 0

    on = _analyse(_chat_project(tmp_path, "on", tool_calling=True), capsys)
    assert on.code == 0 and "prompts offer tools;" not in on.err
    assert "the tool-call parser holds output" not in on.report
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

    monkeypatch.setenv("FAKE_TOOL_CALL_HOLD_S", "0.6")  # a long hold also ends the short window
    held = _analyse(_chat_project(tmp_path, "held", tool_calling=True), capsys)
    assert "the tool-call parser holds output until a call is complete" in held.report


TOOL_CONVERSATION: list[dict[str, Any]] = [
    {"role": "system", "content": "You are the support agent of a parcel carrier."},
    {"role": "user", "content": "Where is parcel X1? It was due yesterday."},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "call1", "type": "function",
     "function": {"name": "track_parcel", "arguments": "{\"tracking_id\": \"X1\"}"}}]},
    {"role": "tool", "tool_call_id": "call1",
     "content": "Parcel X1 is held at the Leeds depot because its label is damaged."},
]  # fmt: skip


def test_analyse_sends_tool_conversations_through_the_chat_template(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A probe-style setup (a known GPU, cap 8, a closed loop of 8, records of about 600 words)
    # gets the sealed probe past "unknown-setup" and "cannot-fill" to the records themselves.
    monkeypatch.setattr("tensward.ceilings.detect_gpus", lambda: (("NVIDIA L4", 23034),))
    for module in ("project", "measure"):
        monkeypatch.setattr(
            f"tensward.{module}.detect_devices", lambda: (GpuInfo("NVIDIA L4", 23034 * 2**20, 0),)
        )
    model, _, _ = make_registration_inputs(tmp_path)
    chat = {
        **REGISTRATION_CONFIG,
        "case": {**REGISTRATION_CONFIG["case"], "max_num_seqs": 8},
        "workload": {**REGISTRATION_CONFIG["workload"], "api": "chat",
                     "arrival": {"kind": "closed_loop", "concurrency": 8}},
    }  # fmt: skip
    config = write_config(tmp_path / "chat.json", chat)
    handbook = {"role": "system", "content": " ".join(f"h{i}" for i in range(600))}
    answered = [handbook, *TOOL_CONVERSATION[1:]]  # ends on the tool's result
    follow_up = [*answered, {"role": "assistant", "content": "It is at the Leeds depot."},
                 {"role": "user", "content": "Can I collect it today?"}]  # fmt: skip
    prompts = write_prompts(tmp_path / "tools.jsonl", [
        {"id": "answered", "messages": answered},
        {"id": "follow-up", "messages": follow_up},
    ])  # fmt: skip
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    capsys.readouterr()
    run = _analyse(project, capsys)
    assert run.code == 0 and run.metrics["failed"] == 0  # the server took every tool message
    rows = (run.run_dir / "requests.jsonl").read_text().splitlines()
    assert {json.loads(row)["prompt_id"] for row in rows} == {"answered", "follow-up"}
    probe = "prompt capacity: not measured (a conversation does not end with a user turn)"
    assert probe in run.report


@pytest.mark.parametrize(
    ("messages", "error"),
    [
        (TOOL_CONVERSATION[:3], "must end with a user message or a tool result"),
        ([*TOOL_CONVERSATION[:2], {**TOOL_CONVERSATION[3], "tool_call_id": "call9"}],
         "answers no earlier tool call"),
        ([TOOL_CONVERSATION[1], {**TOOL_CONVERSATION[2], "role": "user"},
          TOOL_CONVERSATION[3]], "only an assistant message has tool_calls"),
        ([*TOOL_CONVERSATION[:2], {**TOOL_CONVERSATION[2], "tool_calls": [{"id": "call1",
          "type": "function", "function": {"name": "f", "arguments": "{"}}]}],
         "tool call arguments must be JSON text"),
    ],
)  # fmt: skip
def test_a_tool_conversation_the_engine_would_refuse_is_refused(messages: list, error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        PromptEntry.model_validate_json(json.dumps({"id": "x", "messages": messages})).chat


def test_a_change_is_compared_with_the_current_setup_and_its_answers_kept(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _chat_project(tmp_path, "project", tool_calling=True)
    change = ("--engine-arg", "kv-cache-dtype=fp8_e4m3")
    first = _analyse(project, capsys, *change)
    assert first.code == 0 and "No run of your current setup to compare with" in first.report

    current = _analyse(project, capsys)  # the current setup
    assert current.code == 0
    (project / "runs" / "broken").mkdir()
    (project / "runs" / "broken" / "run.json").write_text("{not json")
    after = _analyse(project, capsys, *change)

    assert after.code == 0 and "## Compared with your current setup" in after.report
    assert "Answers changed:" in after.report and "alt0" in after.report
    report = after.report
    assert report.index("## What changed vs your current setup") < report.index("## Diagnosis")
    assert "Change: --engine-arg kv-cache-dtype=fp8_e4m3" in report
    assert "- output " in report and " tok/s (" in report and "- TTFT p95 " in report
    assert "- answers changed: " in report
    assert "- bottleneck: " in report and " → " in report
    assert "One run each: repeat both runs before trusting a difference of a few percent." in report
    assert "## What changed vs your current setup" in after.out
    answers = next((project / "compare").glob("*/answers.md"))
    assert stat.S_IMODE(answers.stat().st_mode) == 0o600 and "alt0" in answers.read_text()
    assert re.search(r"Tool calls identical for \d+ of \d+ prompts that call tools", report)
    assert "expected (before this change):" not in report and "compile cache" not in report

    record_path = current.run_dir / "run.json"
    record = json.loads(record_path.read_text())
    record_path.write_text(json.dumps({**record, "compile_cache": "cold"}))
    assert main(["compare", "--project", str(project), current.run_dir.name,
                 after.run_dir.name]) == 0  # fmt: skip
    shown = capsys.readouterr().out
    assert "cold compile cache" in shown and "expected (before this change):" not in shown
    block = {"kind": "suggestion", "command": f"tensward analyse {' '.join(change)}",
             "expected": {"low": 1.5, "high": 2.0}}  # fmt: skip
    report_path = current.run_dir / "report.json"
    original = report_path.read_text()
    report_path.write_text(json.dumps({"sections": [{"blocks": [block]}]}))
    assert main(["compare", "--project", str(project), current.run_dir.name,
                 after.run_dir.name]) == 0  # fmt: skip
    shown = capsys.readouterr().out
    report_path.write_text(original)
    assert re.search(r"- expected \(before this change\): throughput .*; measured: [x+−]", shown)

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
    assert "compile cache" not in after.report
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


KV_WITH_TOWERS = round(0.53 * 2**30) + round(0.87 * 2**30)  # the pool and the encoder weights freed


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
        f"kv-cache-memory-bytes={KV_WITH_TOWERS}",
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
        f"kv-cache-memory-bytes={KV_WITH_TOWERS}",
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
    assert "`explicit-kv-size`: " in report and "the engine's start log says" in report
    assert "`more-kv-memory`: " not in report  # the engine's figure gives more than gmu 0.95
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
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two clients. The start-up wave offers prompts 4 and 0: the long answer keeps decoding
    while the other start-up request and the four short declared requests come and go through
    the other client's slot. The fake server's KV cache holds one sequence, so each request
    admitted beside the long one counts a preemption. The long answer keeps the whole-run
    window above the 2 s short-window line (about 2.3 s). The first run sends its four
    declared requests: no answer check needs more here."""
    monkeypatch.setattr(window, "JUDGE_REQUESTS", 4)
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


@pytest.mark.parametrize(
    ("records", "needed", "sampled"),
    [(10_000, 50, None), (10_001, 50, 50), (12_000, 20_000, None), (61_000, 3040, 3040)],
)
def test_only_a_file_over_ten_thousand_records_larger_than_the_run_is_sampled(
    records: int, needed: int, sampled: int | None
) -> None:
    sample = sample_of(records, needed, "ab" * 32)
    assert (sample.records if sample else None) == sampled


def test_a_workload_over_ten_thousand_records_sends_the_same_sample_every_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, _ = make_registration_inputs(tmp_path)
    rows = [{"id": f"q{i}", "prompt": f"question {i} about parcels"} for i in range(10_050)]
    prompts = write_prompts(tmp_path / "large.jsonl", rows)
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    capsys.readouterr()
    runs = [_analyse(project, capsys), _analyse(project, capsys)]
    samples = [
        json.loads((run.run_dir / "run.json").read_text())["workload_sample"] for run in runs
    ]
    assert samples[0] == samples[1] and (samples[0]["records"], samples[0]["of"]) == (13, 10_050)
    sent = [
        {
            json.loads(line)["prompt_id"]
            for line in (run.run_dir / "requests.jsonl").read_text().splitlines()
        }
        for run in runs
    ]
    assert sent[0] == sent[1] and len(sent[0]) <= 13
    assert _window(runs[0])["plan"]["requests"] == len(_sent(runs[0])) == window.JUDGE_REQUESTS


# --- capacity ---


def test_capacity_measures_client_steps_within_the_declared_limits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tensward.ceilings.detect_gpus", lambda: (("NVIDIA L4", 23034),))
    monkeypatch.setattr("tensward.checks.SHORT_WINDOW_S", 60.0)  # every step's window is short
    project = registered_project(tmp_path)  # closed loop, 1 client: steps of 1 and 2 clients
    refused = _analyse(project, capsys, "--capacity")
    assert refused.code != 0 and refused.run_dir is None
    assert "needs a declared limit" in refused.err
    loose = _analyse(project, capsys, "--capacity", "--slo-tpot-p95", "100000")
    assert loose.code == 0 and loose.run_dir is None  # nothing written under runs/
    assert "every step was within your limits: the capacity is at least" in loose.out
    assert "requests reused a prompt already sent on this launch" in loose.out
    (summary,) = (project / "capacity").glob("*/capacity.json")
    document = json.loads(summary.read_text())
    assert [s["clients"] for s in document["steps"]] == [1, 2]
    # 2 prompts; 3 warm-up requests, then each step's 32 measured and one per client to start
    assert document["sent"] == 3 + (32 + 1) + (32 + 2) and document["reused"] == 70 - 2
    assert document["prefix_caching"] is False
    steps = summary.parent / "steps"
    assert all((steps / f"{c:04d}-clients" / "report.json").is_file() for c in (1, 2))
    probed = json.loads((steps / "0001-clients" / "run.json").read_text())["prompt_capacity"]
    cache = [{"key": probed["key"], "tok_s": 1234.0, "run_id": "earlier"}]
    (project / "prompt_capacity.json").write_text(json.dumps(cache))
    tight = _analyse(project, capsys, "--capacity", "--slo-ttft-p95", "0.001")
    assert "no step was within your limits; the smallest, 1 client, had TTFT p95" in tight.out
    (newest,) = [p for p in (project / "capacity").glob("*/steps") if p.parent != summary.parent]
    reused = json.loads((newest / "0001-clients" / "run.json").read_text())["prompt_capacity"]
    assert reused["source"] == "cached" and reused["tok_s"] == 1234.0


def test_a_goal_and_its_limit_lead_the_report_and_the_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, _, prompts = make_registration_inputs(tmp_path)
    config = write_config(
        tmp_path / "slos.json", {**REGISTRATION_CONFIG, "slos": {"ttft_p95_ms": 900.0}}
    )
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    capsys.readouterr()
    first = _analyse(project, capsys, "--goal", "ttft")
    assert "goal: time to first token (TTFT p95)" in first.out
    assert "limits: TTFT <= 900 ms per request" in first.out
    goal = json.loads((first.run_dir / "report.json").read_text())["goal"]
    assert goal["goal"] == "ttft" and goal["limits"] == {"ttft_ms": 900.0}
    assert "expected:" not in first.report
    second = _analyse(
        project, capsys, "--goal", "ttft", "--engine-arg", "max-num-batched-tokens=512"
    )
    changed = second.out[second.out.index("## What changed") :]
    assert changed.index("TTFT p95") < changed.index("- requests")
    assert (
        main(["compare", "--project", str(project), first.run_dir.name, second.run_dir.name]) == 0
    )
    out = capsys.readouterr().out
    assert out.index("TTFT p95 (ms)") < out.index("requests per second")
    # under --goal tpot the opt-in async-scheduling change is in Try first with a command
    third = _analyse(project, capsys, "--goal", "tpot", "--engine-arg", "no-async-scheduling")
    shown = [
        b for s in json.loads((third.run_dir / "report.json").read_text())["sections"]
        for b in s["blocks"]
        if b.get("kind") == "suggestion" and b["tier"] == "try_first" and b.get("command")
    ]  # fmt: skip
    assert "async-scheduling" in {b["name"] for b in shown}
    assert all(re.fullmatch(r"about \d+ min on this setup \(.*\)", b["test"]) for b in shown)
    goal = json.loads((second.run_dir / "report.json").read_text())["goal"]
    assert set(goal) >= {"goal", "limits", "plan", "check", "stop"}


def test_a_run_s_own_answers_are_a_baseline_and_failed_rows_are_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = registered_project(tmp_path)
    after = _analyse(project, capsys)
    assert after.run_dir is not None
    own = after.run_dir / "responses.jsonl"
    against = _analyse(project, capsys, "--baseline-answers", str(own))
    assert against.code == 0 and "Comparison with your current setup skipped" not in against.report
    failed = tmp_path / "with-failed.jsonl"
    first = json.loads(own.read_text().splitlines()[0])
    row = {**first, "request_id": "x:999999", "text": "", "tool_calls": [], "finish_reason": None}
    failed.write_text(own.read_text() + json.dumps(row) + "\n")
    again = _analyse(project, capsys, "--baseline-answers", str(failed))
    assert again.code == 0
    covers = re.compile(r"The recorded answers cover \d+ of \d+ prompts")
    assert covers.search(again.report)[0] == covers.search(against.report)[0]
    outcomes = [
        json.loads(path.read_text())["outcome"]
        for path in sorted((project / "compare").glob("*/compare.json"))
    ]
    assert outcomes[0] == outcomes[1]


# --- tensward set ---


def test_set_keeps_a_run_and_analyse_starts_from_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = registered_project(tmp_path)
    base = _analyse(project, capsys)
    assert "This run measured your current setup" in base.report
    changed = _analyse(project, capsys, "--engine-arg", "max-num-seqs=4")
    assert changed.run_dir is not None
    run = changed.run_dir.name
    assert f"tensward set last --project {project}`" in changed.report
    assert "--accept-answer-changes" not in changed.report.split("To keep this change")[-1]
    assert main(["set", run, "--project", str(project)]) == 0
    out = capsys.readouterr().out
    assert out.splitlines() == [
        f"kept the settings of run {run}: tensward analyse starts from them",
        "before: the registered setup (no engine arguments)",
        "after:  --engine-arg max-num-seqs=4",
        f"undo:   tensward set registered --project {project}",
        f"serve:  tensward serve start --project {project} --from current",
    ]
    again = _analyse(project, capsys)
    assert again.run_dir is not None
    assert json.loads((again.run_dir / "run.json").read_text())["engine_args"] == ["max-num-seqs=4"]
    assert f"starting from the kept setup (run {run})" in again.err + again.out
    assert f"This run measured the kept setup (run {run})" in again.report
    other = _analyse(project, capsys, "--engine-arg", "max-num-batched-tokens=512")
    assert "leaves out the kept setup's max-num-seqs=4" in other.err + other.out
    assert f"`tensward set registered --project {project}` undoes it" not in other.report
    assert f"`tensward set {run} --project {project}` undoes it" in other.report
    bigger = _analyse(project, capsys, "--engine-arg", "max-num-seqs=8")
    assert "replaces the kept setup's max-num-seqs=4" in bigger.err + bigger.out
    assert main(["set", run, "--project", str(project)]) == 0
    assert capsys.readouterr().out == f"run {run} is already the kept setup\n"
    assert main(["inspect", "--project", str(project)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["kept"] == {"run": run, "engine_args": ["max-num-seqs=4"]}
    assert base.run_dir is not None
    assert main(["set", base.run_dir.name, "--project", str(project)]) == 0
    out = capsys.readouterr().out
    assert "so keeping it is the same as tensward set registered" in out
    assert f"undo:   tensward set {run} --project {project}" in out
    assert not (project / "kept.json").exists()


def test_set_refuses_a_change_that_can_affect_answers_without_the_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = registered_project(tmp_path)
    _analyse(project, capsys)
    risky = _analyse(project, capsys, "--engine-arg", "kv-cache-dtype=fp8_e4m3")
    assert risky.run_dir is not None
    run = risky.run_dir.name
    typed = "the KV-cache precision (kv-cache-dtype=fp8_e4m3), which can change answers"
    assert f"tensward set last --project {project} --accept-answer-changes" in risky.report
    assert f"this run changes {typed}" in risky.report
    assert main(["set", "last", "--project", str(project)]) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"not kept: run {run} changes {typed}")
    assert f"tensward set {run} --project {project} --accept-answer-changes" in err
    assert not (project / "kept.json").exists()
    assert main(["set", "last", "--project", str(project), "--accept-unreviewed"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"kept the settings of run {run} (the latest run): ")
    assert "you accepted that with --accept-answer-changes" in out
    kept = json.loads((project / "kept.json").read_text())
    assert kept["answer_changes"] == ["kv_cache_precision"]
    assert stat.S_IMODE((project / "kept.json").stat().st_mode) == 0o600
    assert main(["set", "registered", "--project", str(project)]) == 0
    undo = f"undo:   tensward set {run} --project {project} --accept-answer-changes"
    assert undo in capsys.readouterr().out.splitlines()


def test_set_refuses_a_run_that_cannot_be_the_project_s_setup(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = registered_project(tmp_path)
    made = _analyse(project, capsys, "--engine-arg", "max-num-seqs=4")
    assert made.run_dir is not None
    record = json.loads((made.run_dir / "run.json").read_text())
    metrics = json.loads((made.run_dir / "metrics.json").read_text())
    for name, run, served in (("unserved", record, {**metrics, "succeeded": 0}),
                              ("other", {**record, "snapshot_id": "0" * 64}, metrics)):  # fmt: skip
        (project / "runs" / name).mkdir()
        (project / "runs" / name / "run.json").write_text(json.dumps(run))
        (project / "runs" / name / "metrics.json").write_text(json.dumps(served))
    for run, refusal in (
        ("nope", "no run nope in "),
        ("../runs", "no run ../runs in "),
        ("unserved", "run unserved served no request"),
        ("other", "run other was made from another registration"),
    ):
        assert main(["set", run, "--project", str(project)]) == 1
        err = capsys.readouterr().err
        assert err.startswith(f"not kept: {refusal}") and len(err.splitlines()) == 1
    assert not (project / "kept.json").exists()
    assert main(["set", "registered", "--project", str(tmp_path / "nowhere")]) != 0


# --- tensward reject and report ---


def _queueing_project(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], quantized: str | None = None
) -> Path:
    """Eight clients against a cap of 1: kv-sized-concurrency (6) leads, raise-concurrency (4)
    is the other change. ``quantized`` registers Gemma 4's anatomy, with component sizes."""
    model, _, prompts = make_registration_inputs(tmp_path, quantized)
    if quantized:
        make_gemma4_checkpoint(model)
    workload = {**REGISTRATION_CONFIG["workload"], "request_count": 40,
                "arrival": {"kind": "closed_loop", "concurrency": 8}}  # fmt: skip
    registration = {**REGISTRATION_CONFIG, "workload": workload, "case": case_for(quantized)}
    config = write_config(tmp_path / "queueing.json", registration)
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    capsys.readouterr()
    return project


def _offered(run_dir: Path) -> dict[str, str]:
    report = json.loads((run_dir / "report.json").read_text())
    return {b["name"]: b["tier"] for s in report["sections"] for b in s["blocks"]
            if b.get("kind") == "suggestion"}  # fmt: skip


def test_a_kv_bound_report_offers_the_engine_s_figure_and_that_command_serves_at_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hint = 3_000_000_000
    monkeypatch.setenv("FAKE_KV_HINT_BYTES", str(hint))
    monkeypatch.setenv("FAKE_KV_TOKENS", "20000")
    monkeypatch.setenv("FAKE_KV_USAGE", "0.95")  # waiting for KV space, not only for the cap
    project = _queueing_project(tmp_path, capsys)
    base = _analyse(project, capsys)
    assert base.run_dir is not None

    (block,) = [
        b for section in json.loads((base.run_dir / "report.json").read_text())["sections"]
        for b in section["blocks"] if "explicit-kv-size" in b.get("key", "").split("+")
    ]  # fmt: skip
    args = command_args(block["command"])
    size = hint - (1 << 30)  # the engine's figure less the 1 GiB margin
    assert f"kv-cache-memory-bytes={size}" in args

    larger = _analyse(project, capsys, *[w for a in sorted(args) for w in ("--engine-arg", a)])
    assert larger.code == 0 and larger.run_dir is not None
    assert (
        f"GPU KV cache size: {size // 27_700:,} tokens"
        in (larger.run_dir / "server.log").read_text()
    )


def test_reject_stops_offering_a_change_until_cleared(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _queueing_project(tmp_path, capsys)
    base = _analyse(project, capsys)
    six = _analyse(project, capsys, "--engine-arg", "max-num-seqs=6")
    assert base.run_dir is not None and six.run_dir is not None
    run, at = six.run_dir.name, ["--project", str(project)]
    reject = f'Not worth it? `tensward reject last --project {project} --reason "<why>"`'
    assert reject in six.report
    assert main(["set", "last", *at]) == 0 and main(["reject", "last", *at]) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"not rejected: run {run} is the kept setup: go back first with `")
    assert len(err.splitlines()) == 1 and not (project / "rejected.json").exists()
    assert main(["set", "registered", *at]) == 0
    capsys.readouterr()
    assert main(["reject", "last", *at, "--reason", "TPOT too high"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == (
        f"rejected the change of run {run} (the latest run): max-num-seqs=6 (made from run "
        f"{base.run_dir.name})"
    )
    again = _analyse(project, capsys)
    assert again.run_dir is not None
    assert _offered(again.run_dir) == {"raise-concurrency": "try_first"}
    assert (
        f"`kv-sized-concurrency` (max-num-seqs=6) tried in run {run}: rejected by you (TPOT too "
        "high)"
    ) in again.report
    assert main(["reject", "--list", *at]) == 0
    assert capsys.readouterr().out.startswith(f"run {run}: max-num-seqs=6, rejected by you")
    assert main(["set", run, *at]) == 0
    assert f"you rejected this change (run {run}); `" in capsys.readouterr().out
    kept = _analyse(project, capsys)
    assert f"Try: `tensward set registered --project {project}`" in kept.report
    assert main(["set", "registered", *at]) == 0
    capsys.readouterr()
    assert main(["reject", "--clear", run, *at]) == 0
    assert capsys.readouterr().out.startswith(f"cleared: run {run}'s change (max-num-seqs=6)")
    assert main(["report", base.run_dir.name, *at]) == 0
    assert capsys.readouterr().out == base.out
    assert _offered(base.run_dir)["kv-sized-concurrency"] == "try_first"


def test_report_writes_a_run_s_report_again_with_the_rejections(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _queueing_project(tmp_path, capsys)
    goal = ("--goal", "ttft", "--slo-ttft-p95", "900")
    base = _analyse(project, capsys, *goal)
    assert base.run_dir is not None
    written = {name: (base.run_dir / name).read_bytes() for name in ("report.json", "report.md")}
    at = ["--project", str(project)]
    assert main(["report", "last", *at]) == 0
    assert capsys.readouterr().out == base.out
    assert {name: (base.run_dir / name).read_bytes() for name in written} == written
    six = _analyse(project, capsys, *goal, "--engine-arg", "max-num-seqs=6")
    both = ("--engine-arg", "max-num-seqs=6", "--engine-arg", "enable-prefix-caching")
    assert six.run_dir is not None and _analyse(project, capsys, *goal, *both).run_dir
    assert main(["set", "last", *at]) == 0 and main(["reject", six.run_dir.name, *at]) == 1
    assert "(max-num-seqs=6) that is part of the kept setup" in capsys.readouterr().err
    assert main(["set", "registered", *at]) == 0
    assert main(["reject", six.run_dir.name, *at]) == 0
    capsys.readouterr()
    assert main(["report", base.run_dir.name, *at]) == 0
    out = capsys.readouterr().out
    assert _offered(base.run_dir) == {"raise-concurrency": "try_first"}
    assert "--engine-arg max-num-seqs=4 --goal ttft --slo-ttft-p95 900 --runtime local" in out
    assert f"max-num-seqs=6) tried in run {six.run_dir.name}: rejected by you\n" in out
    assert main(["report", six.run_dir.name, *at]) == 0
    assert f"you rejected max-num-seqs=6 (run {six.run_dir.name})" in capsys.readouterr().out


def _step_rows(parent: Path, step: Path, tpot: float = 1.0) -> None:
    """``step``'s requests replaced by ``parent``'s with the time after each first token scaled
    by ``tpot``: a step whose timings are known exactly."""
    rows = []
    for line in _lines(parent / "requests.jsonl"):
        record = json.loads(line)
        first, end = record["first_content_ns"], record["terminal_ns"]
        index = record["request_id"].rsplit(":", 1)[1]
        rows.append({**record, "request_id": f"{step.name}:{index}",
                     "terminal_ns": first + round((end - first) * tpot)})  # fmt: skip
    (step / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def _keep(run_dir: Path) -> dict[str, str]:
    report = json.loads((run_dir / "report.json").read_text())
    return {b["key"]: b["text"] for s in report["sections"] if s["id"] == "keep"
            for b in s["blocks"]}  # fmt: skip


def test_a_step_within_noise_is_recommended_only_under_the_within_noise_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    model, _, prompts = make_registration_inputs(tmp_path)
    case = {**REGISTRATION_CONFIG["case"], "max_num_seqs": 8}
    workload = {**REGISTRATION_CONFIG["workload"], "request_count": 48,
                "arrival": {"kind": "closed_loop", "concurrency": 4}}  # fmt: skip
    config = write_config(tmp_path / "four.json",
                          {**REGISTRATION_CONFIG, "case": case, "workload": workload})  # fmt: skip
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    base = _analyse(project, capsys)
    monkeypatch.setenv("TENSWARD_ADOPT_WITHIN_NOISE", "1")
    step = _analyse(project, capsys, "--engine-arg", "max-num-seqs=6")
    monkeypatch.delenv("TENSWARD_ADOPT_WITHIN_NOISE")
    assert base.run_dir is not None and step.run_dir is not None and step.code == 0
    assert "## What changed" in step.report and " ±" in step.report.split("## Diagnosis")[0]
    analysis = json.loads((step.run_dir / "run.json").read_text())["analysis"]
    assert analysis["adopt_within_noise"] is True and "adopted_within_noise" in analysis
    assert "--adopt-within-noise" not in analysis["flags"]
    at = ["--project", str(project)]
    _step_rows(base.run_dir, step.run_dir)
    assert main(["report", step.run_dir.name, *at]) == 0
    keep = _keep(step.run_dir)
    assert re.fullmatch(
        r"Tensward does not recommend keeping this change: within noise: requests per second "
        r"\+0% ±3%\.",
        keep["not_kept"],
    ), keep
    assert keep["command"].startswith("If you still want it: `tensward set ")
    _step_rows(base.run_dir, step.run_dir, tpot=0.99)
    assert main(["report", step.run_dir.name, *at]) == 0
    capsys.readouterr()
    keep = _keep(step.run_dir)
    assert "not_kept" not in keep
    assert keep["kept_within_noise"].startswith("Within noise (requests per second +1% ±3%); ")
    assert keep["command"] == f"`tensward set last --project {project}`"
    adopted = json.loads((step.run_dir / "report.json").read_text())["goal"]["adopted_within_noise"]
    assert adopted["metric"] == "req_s" and adopted["policy"] == "environment"
    assert 0.005 <= adopted["change"] <= adopted["band"]


# --- two changes in one step ---

FIXED = Prediction(tpot=-0.2, req_s=0.25, text="predicted: TPOT -20%, req/s +25% at this cap")


def _offering(monkeypatch: pytest.MonkeyPatch, partner: str) -> None:
    """``applicable`` offers fp8-weights and ``partner`` in Try first for queueing, and only
    fp8-weights gets a prediction; everything after ``applicable`` runs as it is."""
    entries = {entry.name: entry for entry in VLLM.playbook()}

    def found(*args: Any, **kwargs: Any) -> tuple[list[Suggestion], list]:
        return [
            Suggestion(entry=entries[name], reason=f"{name} fires", tier="try_first",
                       addresses=frozenset({"queueing"}))
            for name in ("fp8-weights", partner)
        ], []  # fmt: skip

    monkeypatch.setattr("tensward.suggest.applicable", found)
    monkeypatch.setattr(
        "tensward.suggest.predict",
        lambda situation, proposed, *, entry, linear_bytes, cap=None: (
            FIXED if entry == "fp8-weights" else None
        ),
    )


def _try_first(run_dir: Path) -> list[dict]:
    report = json.loads((run_dir / "report.json").read_text())
    return [b for s in report["sections"] for b in s["blocks"]
            if b.get("kind") == "suggestion" and b["tier"] == "try_first"]  # fmt: skip


def test_a_decode_held_ttft_figure_leads_try_first_under_the_ttft_goal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE_TOOL_CALL_HOLD_S", "0.6")
    project = _chat_project(tmp_path, "held", tool_calling=True)
    _offering(monkeypatch, "raise-prefill-batch")
    held = Prediction(
        tpot=-0.2, req_s=0.25, ttft=-0.25,
        text="predicted: TPOT -20%, TTFT -25% (the first content waits 12 decode steps for the "
        "tool-call parser)",
    )  # fmt: skip
    monkeypatch.setattr(
        "tensward.suggest.predict",
        lambda situation, proposed, *, entry, **kw: held if entry == "fp8-weights" else None,
    )
    run = _analyse(project, capsys, "--goal", "ttft").run_dir
    assert run is not None
    first = _try_first(run)[0]
    assert first["name"] == "fp8-weights" and first["predicted"]["ttft"] == -0.25
    assert "TTFT -25% (the first content waits 12 decode steps" in first["reason"]


def test_a_tpot_limit_with_room_lets_a_step_budget_join_fp8_weights(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _queueing_project(tmp_path, capsys)
    _offering(monkeypatch, "raise-prefill-batch")
    base = _analyse(project, capsys)
    assert base.metrics is not None
    tpot = base.metrics["tpot_p95_ms"]
    at = ["--project", str(project)]
    for above, members in ((1.4, ["fp8-weights", "raise-prefill-batch"]), (1.05, None)):
        run = _analyse(project, capsys, "--slo-tpot-p95", f"{tpot * above:.3f}").run_dir
        assert run is not None
        first = _try_first(run)[0]
        assert first.get("members") == members and first["predicted"]["req_s"] == 0.25
        assert ("max-num-batched-tokens" in first["command"]) is bool(members)
        written = (run / "report.json").read_bytes()
        assert main(["report", run.name, *at]) == 0
        assert (run / "report.json").read_bytes() == written


def test_a_reused_or_rebuilt_report_carries_the_fresh_runs_fp8_weights_prediction(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    gpu = GpuInfo("NVIDIA L4", 24 * 2**30, 0, index="0", uuid="GPU-1", driver="580.95.05")
    monkeypatch.setattr("tensward.environment.detect_devices", lambda: (gpu,))
    monkeypatch.setattr("tensward.engines.vllm.engine._installed_version", lambda path: "0.30.0")
    project = _queueing_project(tmp_path, capsys, "compressed-tensors-int4")
    _offering(monkeypatch, "raise-prefill-batch")
    seen: list[int | None] = []

    def predicted(
        situation: Any, proposed: Any, *, entry: str, linear_bytes: int | None, cap: int | None
    ) -> Any:
        seen.append(linear_bytes)
        return FIXED if linear_bytes else None

    monkeypatch.setattr("tensward.suggest.predict", predicted)
    fresh = _analyse(project, capsys, "--goal", "tpot")
    assert fresh.run_dir is not None and seen and seen[0]
    reused = _analyse(project, capsys, "--goal", "tpot")
    assert reused.run_dir is not None and "reused_from" in _record(reused.run_dir)
    assert main(["report", fresh.run_dir.name, "--project", str(project)]) == 0
    for run in (fresh.run_dir, reused.run_dir):
        assert _try_first(run)[0]["predicted"]["text"] == FIXED.text
    assert set(seen) == {seen[0]}


def test_a_combined_step_runs_and_its_rejection_brings_both_changes_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _queueing_project(tmp_path, capsys)
    _offering(monkeypatch, "lower-prefill-batch")
    tpot = ("--goal", "tpot")
    base = _analyse(project, capsys, *tpot)
    assert base.run_dir is not None
    offered = _try_first(base.run_dir)
    assert [(b["name"], b.get("members")) for b in offered] == [
        ("fp8-weights", ["fp8-weights", "lower-prefill-batch"])
    ]
    assert "Together with: lower-prefill-batch fires" in offered[0]["reason"]
    together = _analyse(project, capsys, *shlex.split(offered[0]["command"])[4:])
    assert together.run_dir is not None and together.code == 0
    at = ["--project", str(project)]
    assert main(["reject", together.run_dir.name, *at]) == 0
    assert main(["report", base.run_dir.name, *at]) == 0
    capsys.readouterr()
    singles = [(b["name"], b.get("members")) for b in _try_first(base.run_dir)]
    assert singles == [("fp8-weights", None), ("lower-prefill-batch", None)]
    capped = _analyse(project, capsys, *tpot, "--engine-arg", "max-num-seqs=4096")
    assert capped.run_dir is not None
    assert not any("members" in b for b in _try_first(capped.run_dir))
    assert "is below the concurrency cap of 4096: vLLM refuses it" in capped.report


def test_lossless_changes_run_as_one_bundle_over_the_full_window_and_a_worse_one_is_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _queueing_project(tmp_path, capsys)
    entries = {entry.name: entry for entry in VLLM.playbook()}
    small = ("raise-prefill-batch", "kv-sharing-fast-prefill", "stream-interval")
    monkeypatch.setattr("tensward.suggest.applicable", lambda *a, **k: ([
        Suggestion(entry=entries[name], reason=f"{name} fires", tier="try_first",
                   addresses=frozenset({"queueing"}))
        for name in small
    ], []))  # fmt: skip
    base = _analyse(project, capsys, "--slo-ttft-p95", "10000")
    assert base.run_dir is not None
    (bundle,) = _try_first(base.run_dir)
    assert sorted(bundle["members"]) == sorted(small) and not bundle["quality_risk"]
    assert bundle["command"].endswith("--window full")

    bundled = _analyse(project, capsys, *shlex.split(bundle["command"])[4:])
    assert bundled.run_dir is not None and bundled.code == 0
    assert not json.loads((bundled.run_dir / "run.json").read_text())["analysis"][
        "adopt_within_noise"
    ]
    plan = json.loads((bundled.run_dir / "window.json").read_text())["plan"]
    assert plan["reason"] == "--window full"
    at = ["--project", str(project)]

    for scale in (1.0, 1.01):
        _step_rows(base.run_dir, bundled.run_dir, tpot=scale)
        assert main(["report", bundled.run_dir.name, *at]) == 0
        capsys.readouterr()
        keep = _keep(bundled.run_dir)
        assert "not_kept" not in keep, keep
        assert "lossless bundle" in keep["kept_within_noise"]
        assert keep["command"] == f"`tensward set last --project {project}`"
        goal = json.loads((bundled.run_dir / "report.json").read_text())["goal"]
        assert goal["limits"] == {"ttft_ms": 10000}
        adopted = goal["adopted_within_noise"]
        assert adopted["policy"] == "lossless_bundle"
        assert -adopted["band"] < adopted["change"] <= 0
        assert (adopted["change"] < 0) == (scale > 1)

    _step_rows(base.run_dir, bundled.run_dir, tpot=1.2)
    _, guarded = rebuild(project, bundled.run_dir.name, goal="ttft")
    assert "within noise: TTFT p95" in _keep(bundled.run_dir)["not_kept"]
    assert guarded.goal is not None and "adopted_within_noise" not in guarded.goal
    _step_rows(base.run_dir, bundled.run_dir, tpot=1.01)
    rebuild(project, bundled.run_dir.name, goal="throughput")

    def after(reason: str) -> list[list[str]]:
        assert bundled.run_dir is not None and base.run_dir is not None
        main(["reject", "--clear", bundled.run_dir.name, *at])
        assert main(["reject", bundled.run_dir.name, *at, "--reason", reason]) == 0
        assert main(["report", base.run_dir.name, *at]) == 0
        capsys.readouterr()
        return sorted(sorted(b.get("members") or [b["name"]]) for b in _try_first(base.run_dir))

    assert after("tensward: does not recommend keeping this run") == []
    assert after("goal: req/s 10 worse than the best adopted run's 11") == [
        [name] for name in sorted(small)
    ]
    assert after("server did not start: unrecognized arguments: --stream-interval") == [
        ["kv-sharing-fast-prefill", "raise-prefill-batch"],
        ["stream-interval"],
    ]


# --- answers before / after ---


def test_a_change_that_can_affect_answers_shows_them_before_and_after(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = registered_project(tmp_path)
    base = _analyse(project, capsys)
    risky = _analyse(project, capsys, "--engine-arg", "kv-cache-dtype=fp8_e4m3")
    assert base.run_dir is not None and risky.run_dir is not None
    assert "## Answers before / after" in risky.report
    assert f"This run changes the KV-cache precision from run {base.run_dir.name}." in (
        risky.report
    )
    assert "on 0 of 2 prompts; 2 are identical" in risky.report
    assert "Tensward does not judge which answers are better." in risky.report
    quality = json.loads((risky.run_dir / "quality.json").read_text())
    assert quality["before"] == base.run_dir.name and quality["changes"] == ["kv_cache_precision"]
    prompts = {
        json.loads(line)["id"] for line in (tmp_path / "workload.jsonl").read_text().splitlines()
    }
    assert {pair["prompt_id"] for pair in quality["pairs"]} == prompts
    again = _analyse(project, capsys, "--engine-arg", "kv-cache-dtype=fp8_e4m3")
    assert f"shown before / after in run {risky.run_dir.name}'s report" in again.report
    assert again.run_dir is not None and not (again.run_dir / "quality.json").exists()
    safe = _analyse(project, capsys, "--engine-arg", "max-num-seqs=4")
    assert "## Answers before / after" not in safe.report
    assert safe.run_dir is not None and not (safe.run_dir / "quality.json").exists()


def test_answers_that_break_into_loops_are_flagged_and_keeping_them_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = registered_project(tmp_path)
    base = _analyse(project, capsys)
    broken = _analyse(project, capsys, "--engine-arg", "kv-cache-dtype=fp8_inc")
    assert base.run_dir is not None and broken.run_dir is not None
    words = f"answers repeat themselves or are empty, against 0 of 10 in run {base.run_dir.name}"
    header = broken.report.split("## ", 1)[0]
    assert "Warning: answers look broken after this change: at least 10 of 10 " + words in header
    assert "answers that repeat themselves or are empty: 0 of 10 before, 10 of 10 after" in (
        broken.report
    )
    quality = json.loads((broken.run_dir / "quality.json").read_text())
    assert quality["summary"]["broken"] == {"before": [0, 10], "after": [10, 10], "rose": True}
    at = ["--project", str(project)]
    assert main(["set", "last", *at]) == 1
    assert "not kept: answers look broken in run" in capsys.readouterr().err
    assert main(["set", "last", *at, "--accept-answer-changes"]) == 0
    assert "warning: answers look broken in run" in capsys.readouterr().out


def test_answers_before_after_says_when_a_run_kept_no_answers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project = registered_project(tmp_path)
    base = _analyse(project, capsys, "--no-retain-responses")
    risky = _analyse(project, capsys, "--engine-arg", "kv-cache-dtype=fp8_e4m3")
    assert base.run_dir is not None and risky.run_dir is not None
    assert (
        f"Run {base.run_dir.name} kept no answers (--no-retain-responses), so they cannot be "
        "shown before and after."
    ) in risky.report
    assert not (risky.run_dir / "quality.json").exists()


def _reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> tuple[Path, Analysed, list[int]]:
    """A project with one measured run on a named GPU, and a count of the servers started."""
    gpu = GpuInfo("NVIDIA L4", 24 * 2**30, 0, index="0", uuid="GPU-1", driver="580.95.05")
    monkeypatch.setattr("tensward.environment.detect_devices", lambda: (gpu,))
    monkeypatch.setattr("tensward.engines.vllm.engine._installed_version", lambda path: "0.30.0")
    launches: list[int] = []
    start = LocalProcessRuntime.start

    def counted(self: LocalProcessRuntime, spec: ServeSpec) -> RunningServer:
        launches.append(1)
        return start(self, spec)

    monkeypatch.setattr(LocalProcessRuntime, "start", counted)
    project = registered_project(tmp_path)
    return project, _analyse(project, capsys), launches


def _record(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text())


@pytest.mark.parametrize("selection", ["traversal", "symlink", "runs-root"])
def test_explicit_reuse_stays_in_the_project_and_copies_the_run_it_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selection: str,
) -> None:
    project, first, launches = _reference(tmp_path, monkeypatch, capsys)
    assert first.code == 0 and first.run_dir is not None
    foreign = tmp_path / "B"
    shutil.copytree(project, foreign)
    snapshot = json.loads((project / "project.json").read_text())["snapshot_id"]
    assert json.loads((foreign / "project.json").read_text())["snapshot_id"] == snapshot
    original = first.run_dir.name
    target = original[:-4] + ("0000" if original[-4:] != "0000" else "1111")
    foreign_run = foreign / "runs" / target
    shutil.copytree(first.run_dir, foreign_run)
    local_run = project / "runs" / target
    if selection == "traversal":
        shutil.copytree(first.run_dir, local_run)
        record = _record(local_run)
        major, minor, *_ = record["tensward_version"].split(".")
        (local_run / "run.json").write_text(
            json.dumps({**record, "tensward_version": f"{major}.{int(minor) + 1}.0"})
        )
        selected = f"{original}/../../../B/runs/{target}"
    elif selection == "symlink":
        local_run.symlink_to(foreign_run, target_is_directory=True)
        selected = target
    else:
        (project / "runs").rename(project / "original-runs")
        (project / "runs").symlink_to(foreign / "runs", target_is_directory=True)
        selected = target
    foreign_record = (foreign_run / "run.json").read_bytes()
    local_record = (local_run / "run.json").read_bytes()
    started = len(launches)

    measured = _analyse(project, capsys, "--reuse-run", selected)

    assert measured.code == 0 and measured.run_dir is not None
    assert len(launches) > started
    assert "not reused:" in measured.err and "measuring the step" in measured.err
    assert "reused_from" not in _record(measured.run_dir)
    assert (foreign_run / "run.json").read_bytes() == foreign_record
    assert (local_run / "run.json").read_bytes() == local_record


def test_a_plain_analyse_reuses_the_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project, first, launches = _reference(tmp_path, monkeypatch, capsys)
    assert first.run_dir is not None and len(launches) > 0
    started = len(launches)
    stored = json.loads(
        (Path(__file__).parent / "replays/l4-steps/20261014T164058Z-8255/metrics.json").read_text()
    )["diagnosis"]
    metrics_file = first.run_dir / "metrics.json"
    metrics_file.write_text(
        json.dumps({**json.loads(metrics_file.read_text()), "diagnosis": stored})
    )
    again = _analyse(project, capsys, "--goal", "tpot")
    assert again.run_dir is not None and again.run_dir != first.run_dir and again.code == 0
    assert len(launches) == started, again.err
    assert _record(again.run_dir)["reused_from"]["run"] == first.run_dir.name
    assert f"Reused run {first.run_dir.name}, measured at " in again.out
    assert "on NVIDIA L4; not measured again (`--remeasure` measures it)" in again.out
    assert "goal: time per output token (TPOT p95)" in again.report
    kept = json.loads((again.run_dir / "metrics.json").read_text())["diagnosis"]
    assert kept == stored and kept["primary"] == "long_context"
    assert _from_json(Diagnosis, kept).primary == "long_context"
    assert json.loads((again.run_dir / "report.json").read_text())["goal"]["goal"] == "tpot"
    assert (again.run_dir / "requests.jsonl").read_bytes() == (
        first.run_dir / "requests.jsonl"
    ).read_bytes()
    measured = _analyse(project, capsys, "--remeasure")
    assert len(launches) > started and measured.run_dir is not None
    assert "reused_from" not in _record(measured.run_dir)


def test_a_reuse_counts_goodput_again_under_the_new_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project, first, _ = _reference(tmp_path, monkeypatch, capsys)
    loose = _analyse(project, capsys, "--slo-tpot-p95", "100000")
    tight = _analyse(project, capsys, "--slo-tpot-p95", "0.000001")
    assert loose.metrics is not None and tight.metrics is not None
    assert tight.run_dir is not None and "reused_from" in _record(tight.run_dir)
    assert loose.metrics["slo_attainment"] == 1.0 and tight.metrics["slo_attainment"] == 0.0
    assert tight.metrics["goodput"] == 0.0
    window = json.loads((tight.run_dir / "window.json").read_text())
    assert window["recomputed"] == ["goodput", "slo_attainment"]
    assert json.loads((tight.run_dir / "report.json").read_text())["goal"]["limits"] == {
        "tpot_ms": 1e-06
    }


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("engine_version", "ran vLLM 0.0.1"),
        ("planned", "was measured over a planned window"),
        ("failures", "of its requests fail (the limit is 2%)"),
        ("old", "h ago (the limit is 24 h)"),
        ("environment", "ran with another inherited engine environment"),
        ("no_answers", "kept no answers"),
        ("gpu", "ran on another GPU or driver"),
    ],
)
def test_a_run_that_is_not_trusted_is_measured_again_with_the_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    change: str,
    reason: str,
) -> None:
    project, first, launches = _reference(tmp_path, monkeypatch, capsys)
    assert first.run_dir is not None
    run_dir, started = first.run_dir, len(launches)
    record, metrics = _record(run_dir), json.loads((run_dir / "metrics.json").read_text())
    if change == "engine_version":
        record["engine_version"] = "0.0.1"
    elif change == "planned":
        (run_dir / "window.json").write_text(json.dumps({"plan": {"requests": 80, "seconds": 60}}))
    elif change == "failures":
        metrics["failed"] = metrics["succeeded"]
    elif change == "old":
        run_dir = run_dir.rename(run_dir.parent / f"20200101T000000Z-{run_dir.name[-4:]}")
    elif change == "environment":
        monkeypatch.setenv("VLLM_LOGGING_LEVEL", "DEBUG")
    elif change == "no_answers":
        record["retained_responses"] = False
    else:
        record["gpus_identity"][0]["driver"] = "570.1"
    (run_dir / "run.json").write_text(json.dumps(record))
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    again = _analyse(project, capsys)
    assert again.run_dir is not None and len(launches) > started
    assert "reused_from" not in _record(again.run_dir)
    assert f"not reused: run {run_dir.name} " in again.err + again.out
    assert reason in again.err + again.out


def test_a_reused_run_is_never_a_reference_and_an_engine_argument_always_measures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project, first, launches = _reference(tmp_path, monkeypatch, capsys)
    assert first.run_dir is not None
    reused = _analyse(project, capsys)
    assert reused.run_dir is not None and "reused_from" in _record(reused.run_dir)
    started = len(launches)
    shutil.rmtree(first.run_dir)
    only = _analyse(project, capsys)
    assert len(launches) > started and "is itself a reused run" in only.err + only.out
    started = len(launches)
    changed = _analyse(project, capsys, "--engine-arg", "max-num-seqs=4")
    assert len(launches) > started and "not reused:" not in changed.err + changed.out


def test_import_copies_a_run_of_the_same_registration_and_a_plain_analyse_reuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project, first, launches = _reference(tmp_path, monkeypatch, capsys)
    assert first.run_dir is not None
    model = tmp_path / "model"
    config = write_config(tmp_path / "serving.json", REGISTRATION_CONFIG)
    prompts = tmp_path / "workload.jsonl"
    same, other = tmp_path / "same", tmp_path / "other"
    assert main(["init", "--project", str(same), "--model", str(model), "--config", str(config),
                 "--prompts", str(prompts)]) == 0  # fmt: skip
    changed = write_prompts(tmp_path / "other.jsonl", [{"id": "x", "prompt": "another prompt"}])
    assert main(["init", "--project", str(other), "--model", str(model), "--config", str(config),
                 "--prompts", str(changed)]) == 0  # fmt: skip
    capsys.readouterr()
    assert main(["import", str(first.run_dir), "--project", str(other)]) == 1
    assert "not imported: run " in capsys.readouterr().err
    assert not (other / "runs").exists() or not list((other / "runs").iterdir())
    assert main(["import", str(first.run_dir), "--project", str(same)]) == 0
    assert first.run_dir.name in capsys.readouterr().out
    imported = same / "runs" / first.run_dir.name
    assert _record(imported)["imported_from"]["run"] == first.run_dir.name
    assert main(["import", str(first.run_dir), "--project", str(same)]) == 1
    capsys.readouterr()
    started = len(launches)
    again = _analyse(same, capsys)
    assert again.run_dir is not None and len(launches) == started
    assert _record(again.run_dir)["reused_from"]["run"] == first.run_dir.name


# --- the planned window ---


def _windowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict] | None = None,
    *,
    request_count: int = 100,
) -> Path:
    """A project from 2 clients whose steps plan a 1 s window, at least 20
    requests: the L4 cell's rule at the fake server's speed."""
    monkeypatch.setattr(window, "WINDOWS_S", (1.0,))
    monkeypatch.setattr(window, "MIN_REQUESTS", 20)
    monkeypatch.setattr(window, "JUDGE_REQUESTS", 20)
    monkeypatch.setenv("FAKE_TOKEN_DELAY_S", "0.002")
    model, _, _ = make_registration_inputs(tmp_path)
    workload = {**REGISTRATION_CONFIG["workload"], "request_count": request_count,
                "arrival": {"kind": "closed_loop", "concurrency": 2}}  # fmt: skip
    config = write_config(tmp_path / "windowed.json", {**REGISTRATION_CONFIG, "workload": workload})
    prompts = write_prompts(tmp_path / "windowed.jsonl", rows)
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    return project


def _window(run: Analysed) -> dict:
    assert run.run_dir is not None, run.err
    return json.loads((run.run_dir / "window.json").read_text())


def _sent(run: Analysed) -> list[str]:
    """The prompt ids the run sent, in the order they were offered."""
    assert run.run_dir is not None, run.err
    rows = [json.loads(row) for row in _lines(run.run_dir / "requests.jsonl")]
    rows.sort(key=lambda row: int(row["request_id"].rsplit(":", 1)[1]))
    return [row["prompt_id"] for row in rows]


def test_a_step_measures_the_planned_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _windowed(tmp_path, monkeypatch)
    first = _analyse(project, capsys)
    assert len(_sent(first)) == 100
    assert _window(first)["plan"] == {
        "requests": 100, "seconds": None, "reason": "no earlier run to plan from"
    }  # fmt: skip
    assert "window: configured count (no earlier run to plan from)" in first.err
    assert re.search(r"testing it: about \d+ min on this setup \(this run: \d+ s to load, a 1 s "
                     r"window planned\)", first.report)  # fmt: skip
    step = _analyse(project, capsys, "--engine-arg", "max-num-seqs=8")
    assert step.run_dir is not None and first.run_dir is not None
    planned = _record(step.run_dir)["analysis"]["window"]
    assert planned["seconds"] == 1.0 and 8 <= planned["requests"] < 100
    assert len(_sent(step)) == planned["requests"]
    assert _sent(step) == _sent(first)[: planned["requests"]]
    recorded = _window(step)
    assert recorded["plan"] == planned and recorded["parent"] == first.run_dir.name
    assert recorded["startup"]["ready_s"] is not None
    assert f"window: {planned['requests']} requests (about 1 s); from run " in step.err
    full = _analyse(project, capsys, "--engine-arg", "max-num-seqs=8", "--window", "full")
    assert len(_sent(full)) == 100 and _window(full)["plan"]["reason"] == "--window full"
    before = (step.run_dir / "window.json").read_bytes()
    assert main(["report", step.run_dir.name, "--project", str(project)]) == 0
    assert (step.run_dir / "window.json").read_bytes() == before


def test_a_short_step_compares_only_the_answers_both_runs_gave(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    texts = [f"question {i} about the parcels" for i in range(40)]
    rows = [{"id": f"q{i}", "prompt": t + "?" * (len(t) % 2)} for i, t in enumerate(texts)]
    project = _windowed(tmp_path, monkeypatch, rows)
    first = _analyse(project, capsys)
    step = _analyse(project, capsys, "--engine-arg", "kv-cache-dtype=fp8_e4m3")
    assert first.run_dir is not None and step.run_dir is not None and step.code == 0
    shared = set(_sent(step))
    assert len(shared) < 40 and _window(step)["plan"]["seconds"] == 1.0
    quality = json.loads((step.run_dir / "quality.json").read_text())
    assert {pair["prompt_id"] for pair in quality["pairs"]} == shared
    summary = quality["summary"]
    assert (summary["prompts"], summary["identical"], summary["differ"]) == (
        len(shared), len(shared), 0
    )  # fmt: skip
    assert summary["unanswered"] == 0
    both = f"the {len(shared)} prompts both runs sent"
    assert f"on 0 of {both}; {len(shared)} are identical" in step.report
    assert f"All answers identical to {first.run_dir.name}" in step.report
    assert "Answers changed" not in step.report and "got no answer" not in step.report


def test_compare_reads_a_short_step_on_the_prompts_both_runs_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [{"id": f"q{i}", "prompt": f"question {i} about the parcels"} for i in range(40)]
    project = _windowed(tmp_path, monkeypatch, rows)
    first = _analyse(project, capsys)
    step = _analyse(project, capsys, "--engine-arg", "kv-cache-dtype=fp8_e4m3")
    assert first.run_dir is not None and step.run_dir is not None
    shared = len(set(_sent(step)))
    assert shared < 40
    at = ["--project", str(project)]
    assert main(["compare", first.run_dir.name, step.run_dir.name, *at]) == 0
    out = capsys.readouterr().out
    assert "got no answer" not in out
    assert re.search(rf"\*\*Answers changed: \d+ of {shared} prompts\*\*", out), out
    assert re.search(rf"\d+ of the {shared} prompts both runs sent differ", out), out


def test_a_step_whose_parent_kept_no_requests_measures_the_configured_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _windowed(tmp_path, monkeypatch)
    first = _analyse(project, capsys)
    assert first.run_dir is not None
    (first.run_dir / "requests.jsonl").unlink()
    step = _analyse(project, capsys, "--engine-arg", "max-num-seqs=8")
    assert len(_sent(step)) == 100
    plan = _window(step)["plan"]
    assert plan["seconds"] is None
    assert plan["reason"] == f"run {first.run_dir.name} kept no requests.jsonl"
    assert f"window: configured count (run {first.run_dir.name} kept no requests.jsonl)" in (
        step.err
    )


def test_a_given_window_plans_from_the_same_setup_and_never_reuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project, first, launches = _reference(tmp_path, monkeypatch, capsys)
    (tmp_path / "fresh").mkdir()
    fresh = _analyse(registered_project(tmp_path / "fresh"), capsys, "--window", "60")
    assert "window: configured count (no earlier run to plan from)" in fresh.err
    assert first.run_dir is not None
    requests = first.run_dir / "requests.jsonl"
    rows = [json.loads(line) for line in _lines(requests)]
    start = rows[0]["dispatch_ns"]
    stretched = [{**row, "dispatch_ns": start + (row["dispatch_ns"] - start) * 10_000}
                 for row in rows]  # fmt: skip
    requests.write_text("".join(json.dumps(row) + "\n" for row in stretched))
    monkeypatch.setattr(window, "MIN_REQUESTS", 4)
    monkeypatch.setattr(window, "MIN_PER_CLIENT", 2)
    started = len(launches)
    again = _analyse(project, capsys, "--window", "60")
    assert again.run_dir is not None and len(launches) > started
    assert "reused_from" not in _record(again.run_dir) and "not reused" not in again.err
    plan = _window(again)["plan"]
    assert plan["seconds"] == 60.0 and plan["requests"] < len(_sent(first))
    assert len(_sent(again)) == plan["requests"]


def test_a_second_load_of_an_unchanged_project_reads_nothing_and_still_plans_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = registered_project(tmp_path)
    reads: list[Path] = []
    real = inputs.PromptFile.__init__

    def counting(self: inputs.PromptFile, path: Path) -> None:
        reads.append(path)
        real(self, path)

    monkeypatch.setattr(inputs.PromptFile, "__init__", counting)
    monkeypatch.setattr(project_module, "_DERIVED", [])
    first = load_project(project)
    assert len(reads) == 1
    with active(AnalysisContext(window_plan={"requests": 4})):
        planned = load_project(project)
    assert len(reads) == 1
    assert planned.config.workload.request_count == 4
    assert planned.prompts == first.prompts
    assert first.config.workload.request_count > 4
    prompts = Path(first.record.prompts_path)
    prompts.write_text(prompts.read_text() + json.dumps({"id": "third", "prompt": "more"}) + "\n")
    with pytest.raises(PreflightError):
        load_project(project)
    assert len(reads) == 2


def test_a_large_workload_sends_the_same_sample_over_a_planned_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [{"id": f"q{i}", "prompt": f"question {i} about parcels"} for i in range(10_001)]
    project = _windowed(tmp_path, monkeypatch, rows)
    full = _analyse(project, capsys)
    step = _analyse(project, capsys, "--engine-arg", "max-num-seqs=8")
    assert full.run_dir is not None and step.run_dir is not None
    assert _window(step)["plan"]["seconds"] == 1.0
    sample = _record(full.run_dir)["workload_sample"]
    assert _record(step.run_dir)["workload_sample"] == sample and sample["records"] == 103
    assert _sent(step) == _sent(full)[: len(_sent(step))] and len(_sent(step)) < 100


def _revert(run: Analysed) -> list[str]:
    assert run.run_dir is not None
    (block,) = [b for b in _try_first(run.run_dir) if b["name"] == "revert-last-change"]
    return shlex.split(block["command"])[4:]


def test_a_revert_plans_from_the_run_it_goes_back_to_not_the_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _windowed(tmp_path, monkeypatch)
    tpot = ("--goal", "tpot")
    thrash = ("--engine-arg", "gpu-memory-utilization=0.05")
    _analyse(project, capsys, *tpot)
    assert "--engine-arg" not in _revert(_analyse(project, capsys, *tpot, *thrash))
    measured = _analyse(project, capsys, *tpot, "--remeasure")
    assert _window(measured)["plan"]["seconds"] is None and _window(measured)["parent"] is None
    prefix = ("--engine-arg", "enable-prefix-caching")
    _analyse(project, capsys, *tpot, *prefix)
    worse = _analyse(project, capsys, *tpot, *prefix, *thrash)
    reverted = _analyse(project, capsys, *_revert(worse))
    assert measured.run_dir is not None
    assert _window(reverted)["parent"] == measured.run_dir.name


# --- a session of steps ---


def test_a_session_of_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gpu = GpuInfo("NVIDIA L4", 24 * 2**30, 0, index="0", uuid="GPU-1", driver="580.95.05")
    monkeypatch.setattr("tensward.environment.detect_devices", lambda: (gpu,))
    monkeypatch.setattr("tensward.engines.vllm.engine._installed_version", lambda path: "0.30.0")
    project = _windowed(tmp_path, monkeypatch)
    _offering(monkeypatch, "lower-prefill-batch")
    monkeypatch.setattr(
        "tensward.suggest.predict",
        lambda situation, proposed, *, entry, linear_bytes, cap=None: (
            Prediction(tpot=-0.9, req_s=0.25, text="predicted: TPOT -90%")
            if entry == "fp8-weights" else None
        ),
    )  # fmt: skip
    ids = (time.strftime("%Y%m%dT%H%M%SZ-0000", time.gmtime(time.time() + i)) for i in count())
    for module in ("analyse", "reuse"):  # runs made within one second keep their order
        monkeypatch.setattr(f"tensward.{module}.new_run_id", lambda: next(ids))
    table = window._table()  # the fake server's TPOT moves several percent between launches:
    floor = {**table["floor"], "tpot_p95_ms": 0.3}  # never worse, always short of -90%
    monkeypatch.setattr(window, "_table", lambda: {**table, "floor": floor})
    at = ["--project", str(project)]
    tpot = ("--goal", "tpot")
    reference = _analyse(project, capsys, *tpot)
    assert reference.run_dir is not None and reference.code == 0
    assert len(_sent(reference)) == _window(reference)["plan"]["requests"] == 100
    assert _window(reference)["plan"]["reason"] == "no earlier run to plan from"
    assert "window: configured count (no earlier run to plan from)" in reference.err
    assert set(json.loads((reference.run_dir / "report.json").read_text())) == {
        "goal", "kb", "ran", "run_id", "schema_version", "sections"
    }  # fmt: skip
    again = _analyse(project, capsys, *tpot)
    assert again.run_dir is not None and again.code == 0
    assert _record(again.run_dir)["reused_from"]["run"] == reference.run_dir.name
    offered = _try_first(again.run_dir)
    assert [(b["name"], b.get("members")) for b in offered] == [
        ("fp8-weights", ["fp8-weights", "lower-prefill-batch"])
    ]
    together = _analyse(project, capsys, *shlex.split(offered[0]["command"])[4:])
    assert together.run_dir is not None and together.code == 0
    recorded = _window(together)
    assert recorded["parent"] == again.run_dir.name and recorded["plan"]["seconds"] == 1.0
    assert len(_sent(together)) == recorded["plan"]["requests"] < 100
    assert _sent(together) == _sent(reference)[: len(_sent(together))]
    assert set(recorded["memory"]) == {"peak_used_mib", "total_mib"}
    assert recorded["plan"].get("steady") is None or isinstance(recorded["plan"]["steady"], dict)
    for cache in (recorded.get("caches") or {}).values():
        assert set(cache) == {"before", "after", "warm"}
    assert {d["metric"] for d in recorded["deltas"]} >= {"tpot_p95_ms", "req_s"}
    anchor = _try_first(together.run_dir)[0]
    assert anchor["name"] == "revert-last-change"
    assert "the gain fell short of the prediction for fp8-weights alone" in anchor["reason"]
    assert "quantization=fp8" in anchor["command"]
    assert "max-num-batched-tokens" not in anchor["command"]
    alone = _analyse(project, capsys, *shlex.split(anchor["command"])[4:])
    assert alone.run_dir is not None and alone.code == 0
    assert _window(alone)["plan"]["seconds"] == 1.0
    assert _window(alone)["parent"] == again.run_dir.name
    assert len(_sent(alone)) == _window(alone)["plan"]["requests"] < 100
    assert f"## What changed vs your current setup (run {again.run_dir.name})" in alone.report
    assert main(["set", alone.run_dir.name, *at, "--accept-answer-changes"]) == 0
    kept = json.loads((project / "kept.json").read_text())
    assert kept["run"] == alone.run_dir.name


def test_a_text_only_session_runs_a_bundle_then_reuses_a_step_after_a_judge_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gpu = GpuInfo("NVIDIA L4", 24 * 2**30, 0, index="0", uuid="GPU-1", driver="580.95.05")
    monkeypatch.setattr("tensward.environment.detect_devices", lambda: (gpu,))
    monkeypatch.setattr("tensward.engines.vllm.engine._installed_version", lambda path: "0.30.0")
    monkeypatch.setenv("FAKE_MEDIA_PROFILE", "1")
    monkeypatch.setenv("FAKE_KV_USAGE", "0.25")
    monkeypatch.setenv("FAKE_TOKEN_DELAY_S", "0.005")
    project = _queueing_project(tmp_path, capsys, "compressed-tensors-int4")
    ids = (time.strftime("%Y%m%dT%H%M%SZ-0000", time.gmtime(time.time() + i)) for i in count())
    for module in ("analyse", "reuse"):
        monkeypatch.setattr(f"tensward.{module}.new_run_id", lambda: next(ids))
    entries = {entry.name: entry for entry in VLLM.playbook()}
    small = ("raise-prefill-batch", "kv-sharing-fast-prefill", "stream-interval")
    names = small
    monkeypatch.setattr("tensward.suggest.applicable", lambda *a, **k: ([
        Suggestion(entry=entries[name], reason=f"{name} fires", tier="try_first",
                   addresses=frozenset({"queueing"}))
        for name in names
    ], []))  # fmt: skip
    at = ["--project", str(project)]
    base = _analyse(project, capsys, "--slo-ttft-p95", "10000")
    assert base.code == 0 and base.run_dir is not None and base.metrics is not None
    argv = json.loads((base.run_dir / "serve.json").read_text())["identity"]["argv"]
    assert "--language-model-only" not in argv
    (bundle,) = _try_first(base.run_dir)
    assert sorted(bundle["members"]) == sorted(small) and not bundle["quality_risk"]
    assert bundle["command"].endswith("--window full")
    args = command_args(bundle["command"])
    assert "language-model-only" in args
    (size,) = [int(a.split("=", 1)[1]) for a in args if a.startswith("kv-cache-memory-bytes=")]
    assert base.metrics["peak_kv_usage"] > 0
    assert size / 2**30 >= base.metrics["kv_available_gib"] * base.metrics["peak_kv_usage"]

    bundled = _analyse(project, capsys, *shlex.split(bundle["command"])[4:])
    assert bundled.code == 0 and bundled.run_dir is not None
    assert _window(bundled)["parent"] == base.run_dir.name
    assert _window(bundled)["plan"]["reason"] == "--window full"
    argv = json.loads((bundled.run_dir / "serve.json").read_text())["identity"]["argv"]
    assert "--language-model-only" in argv and str(size) in argv
    assert main(["set", bundled.run_dir.name, *at, "--accept-answer-changes"]) == 0

    names = ("compact-json",)
    assert main(["report", bundled.run_dir.name, *at]) == 0
    (offer,) = [b for b in _try_first(bundled.run_dir) if b["name"] == names[0]]
    assert offer["changes_answers"] and offer["quality_risk"]
    measured = _analyse(project, capsys, *shlex.split(offer["command"])[4:])
    assert measured.code == 0 and measured.run_dir is not None
    assert _window(measured)["parent"] == bundled.run_dir.name
    assert main(["reject", measured.run_dir.name, *at, "--reason", "judge: timeout"]) == 0
    assert main(["report", bundled.run_dir.name, *at]) == 0
    (again,) = [b for b in _try_first(bundled.run_dir) if b["name"] == names[0]]
    assert _without(again["command"], "--reuse-run", measured.run_dir.name) == shlex.split(
        offer["command"]
    )
    reused = _analyse(project, capsys, *shlex.split(again["command"])[4:])
    assert reused.code == 0 and reused.run_dir is not None
    assert _record(reused.run_dir)["reused_from"]["run"] == measured.run_dir.name
    assert (reused.run_dir / "responses.jsonl").read_bytes() == (
        measured.run_dir / "responses.jsonl"
    ).read_bytes()


def test_analyse_refuses_an_act_order_gptq_checkpoint_before_starting_a_server(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tensward.engines.vllm.engine._installed_version", lambda command: "0.30.0")
    model, config, prompts = make_registration_inputs(tmp_path, "gptq")
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    capsys.readouterr()
    command = shlex.join([sys.executable, str(FAKE_SERVER)])

    code = main(["analyse", "--project", str(project), "--runtime", "local",
                 "--local-command", command])  # fmt: skip

    assert code == 3
    assert "act-order (desc_act) are not supported by vLLM 0.30" in capsys.readouterr().err
    assert not (project / "runs").exists() or not list((project / "runs").iterdir())


def _guarded_project(tmp_path: Path) -> Path:
    model, _, prompts = make_registration_inputs(tmp_path)
    workload = {**REGISTRATION_CONFIG["workload"], "request_count": 48,
                "arrival": {"kind": "closed_loop", "concurrency": 16}}  # fmt: skip
    config = write_config(tmp_path / "serving.json", {
        **REGISTRATION_CONFIG, "case": {**REGISTRATION_CONFIG["case"], "max_num_seqs": 16},
        "workload": workload,
    })  # fmt: skip
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    return project


def test_a_step_whose_kv_pool_is_under_half_its_parents_stops_at_start_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tensward.history import stopped_runs

    project = _guarded_project(tmp_path)
    fake = shlex.join([sys.executable, str(FAKE_SERVER)])
    hangs = tmp_path / "hangs"
    hangs.write_text(
        "#!/bin/sh\necho 'INFO Available KV cache memory: 0.25 GiB'\n"
        "echo 'INFO GPU KV cache size: 9,668 tokens'\nexec sleep 60\n"
    )
    hangs.chmod(0o755)

    def analyse(command: str, *args: str) -> tuple[int, float]:
        began = time.monotonic()
        code = main(["analyse", "--project", str(project), "--runtime", "local",
                     "--local-command", command, "--ready-timeout", "60", *args])  # fmt: skip
        return code, time.monotonic() - began

    monkeypatch.setenv("FAKE_KV_TOKENS", "40960")
    monkeypatch.setenv("FAKE_KV_USAGE", "0.9")
    assert analyse(fake)[0] == 0
    monkeypatch.delenv("FAKE_KV_USAGE")
    (parent,) = (project / "runs").iterdir()
    measured = json.loads((parent / "metrics.json").read_text())
    assert measured["peak_kv_usage"] == 0.9
    assert measured["peak_kv_usage"] * measured["kv_available_gib"] > 2 * 0.25
    monkeypatch.setenv("FAKE_MEDIA_PROFILE", "1")
    capsys.readouterr()
    for command in (fake, str(hangs)):
        code, seconds = analyse(command, "--engine-arg", "language-model-only")
        assert code != 0 and seconds < 15
        assert "stopped at start-up: KV pool 0.25 GiB" in capsys.readouterr().err

    stopped = stopped_runs(project)
    assert [(args, reason.split(" against")[0]) for _, args, reason in stopped] == [
        (frozenset({"language-model-only"}), "stopped at start-up: KV pool 0.25 GiB"),
    ] * 2
    for run_id, _, _ in stopped:
        run_dir = project / "runs" / run_id
        assert not (run_dir / "run.json").exists()
        record = json.loads((run_dir / "window.json").read_text())["startup_guard"]
        assert record["pool_tokens"] == 9668 and record["used_tokens"] > 2 * 9668
        assert record["pool_gib"] == 0.25 and record["used_gib"] > 2 * 0.25
    sized = "kv-cache-memory-bytes=2770000000"
    assert analyse(fake, "--engine-arg", "language-model-only", "--engine-arg", sized)[0] == 0
    assert len(stopped_runs(project)) == 2


def test_a_server_with_no_memory_left_for_its_kv_cache_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tensward.history import stopped_runs

    project = _guarded_project(tmp_path)
    serve = tmp_path / "serve"
    serve.write_text(
        "#!/bin/sh\necho '(EngineCore pid=7) ValueError: No available memory for the cache "
        "blocks. Try increasing `gpu_memory_utilization` when initializing the engine'\nexit 1\n"
    )
    serve.chmod(0o755)
    capsys.readouterr()

    assert main(["analyse", "--project", str(project), "--runtime", "local",
                 "--local-command", str(serve)]) == 1  # fmt: skip

    assert (
        "the server did not start: no memory left for the KV cache (vLLM: No available memory for "
        "the cache blocks)"
    ) in capsys.readouterr().err
    assert [args for _, args, _ in stopped_runs(project)] == [frozenset()]


def _shared_layer_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("FAKE_TOKEN_DELAY_S", "0.005")
    model = make_checkpoint(tmp_path / "model")
    config = json.loads((model / "config.json").read_text())
    (model / "config.json").write_text(
        json.dumps({**config, "num_hidden_layers": 4, "num_kv_shared_layers": 2})
    )
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(write_config(tmp_path / "serving.json")),
                 "--prompts", str(write_prompts(tmp_path / "workload.jsonl"))]) == 0  # fmt: skip
    return project


def test_fast_prefill_is_compared_by_its_answers_though_it_is_lossless_for_the_driver(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _shared_layer_project(tmp_path, monkeypatch)
    base = _analyse(project, capsys, "--goal", "ttft")
    assert base.run_dir is not None
    offered = [
        block
        for section in json.loads((base.run_dir / "report.json").read_text())["sections"]
        for block in section["blocks"]
        if block.get("name") == "kv-sharing-fast-prefill"
    ]
    assert len(offered) == 1 and offered[0]["checks_answers"] is True
    assert (
        offered[0]["quality_risk"] is False
        and "2 of 4 layers reuse earlier KV" in (offered[0]["reason"])
    )

    step = _analyse(project, capsys, *shlex.split(offered[0]["command"])[4:])
    assert step.run_dir is not None and "## Answers before / after" in step.report
    quality = json.loads((step.run_dir / "quality.json").read_text())
    assert quality["before"] == base.run_dir.name
    assert "kv-sharing-fast-prefill" in quality["changes"]
    assert quality["pairs"]


def test_fast_prefill_run_by_hand_is_compared_though_the_report_did_not_offer_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _shared_layer_project(tmp_path, monkeypatch)
    base = _analyse(project, capsys, "--goal", "tpot")
    assert base.run_dir is not None
    assert "kv-sharing-fast-prefill" not in (base.run_dir / "report.json").read_text()

    step = _analyse(project, capsys, "--goal", "tpot", "--engine-arg", "kv-sharing-fast-prefill")

    assert step.run_dir is not None and "## Answers before / after" in step.report
    quality = json.loads((step.run_dir / "quality.json").read_text())
    assert quality["before"] == base.run_dir.name
    assert quality["changes"] == ["kv-sharing-fast-prefill"]


def test_a_kv_bound_setting_measured_again_plans_a_steady_window_from_its_full_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(window, "STEADY_MIN_S", 8.0)
    monkeypatch.setenv("FAKE_TOKEN_DELAY_S", "0.001")
    model, _, _ = make_registration_inputs(tmp_path)
    workload = {**REGISTRATION_CONFIG["workload"], "request_count": 240,
                "arrival": {"kind": "closed_loop", "concurrency": 8}}  # fmt: skip
    config = write_config(tmp_path / "loop.json", {**REGISTRATION_CONFIG, "workload": workload})
    rows = [{"id": f"q{i}", "prompt": f"question {i} about parcels"} for i in range(240)]
    prompts = write_prompts(tmp_path / "loop.jsonl", rows)
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    words = ("--goal", "throughput", "--engine-arg", "gpu-memory-utilization=0.05")
    full = _analyse(project, capsys, *words)
    assert full.run_dir is not None and _window(full)["plan"]["seconds"] is None
    for _ in range(2):
        again = _analyse(project, capsys, *words, "--window", "auto", "--remeasure")
        plan = _window(again)["plan"]
        assert plan.get("steady", {}).get("source") == full.run_dir.name, plan
        assert set(plan["steady"]["bias"]) == {"req_s", "tpot_p95_ms"}
        assert len(_sent(again)) == plan["requests"] < len(_sent(full))


# --- stacks of changes ---

STACKED = ("fp8-weights", "raise-prefill-batch", "kv-sharing-fast-prefill")


def _stacking(monkeypatch: pytest.MonkeyPatch, predicted: Prediction) -> None:
    """``applicable`` offers the three changes of ``STACKED`` in Try first for queueing, and
    only fp8 weights get a prediction."""
    entries = {entry.name: entry for entry in VLLM.playbook()}

    def found(*args: Any, **kwargs: Any) -> tuple[list[Suggestion], list]:
        return [
            Suggestion(entry=entries[name], reason=f"{name} fires", tier="try_first",
                       addresses=frozenset({"queueing"}))
            for name in STACKED
        ], []  # fmt: skip

    monkeypatch.setattr("tensward.suggest.applicable", found)
    monkeypatch.setattr(
        "tensward.suggest.predict",
        lambda situation, proposed, *, entry, **kw: predicted if entry == "fp8-weights" else None,
    )


def test_a_stack_of_three_falls_short_and_its_fp8_weights_are_measured_alone_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _windowed(tmp_path, monkeypatch)
    _stacking(monkeypatch, Prediction(tpot=-0.2, req_s=0.9, text="predicted: req/s +90%"))
    table = window._table()  # never worse, always short of +90%
    monkeypatch.setattr(
        window, "_table", lambda: {**table, "floor": {**table["floor"], "req_s": 0.3}}
    )
    base = _analyse(project, capsys)
    assert base.run_dir is not None and base.code == 0
    stack = _try_first(base.run_dir)[0]
    assert stack["members"][0] == "fp8-weights" and set(stack["members"]) == set(STACKED)
    assert stack["checks_answers"]
    sized = [a for a in command_args(stack["command"]) if a.startswith("kv-cache-memory-bytes=")]
    assert len(sized) == 1 and "The KV cache is set to" in stack["reason"]

    together = _analyse(project, capsys, *shlex.split(stack["command"])[4:])

    assert together.run_dir is not None and together.code == 0
    split = _try_first(together.run_dir)[0]
    assert split["name"] == "revert-last-change"
    assert "the gain fell short of the prediction for fp8-weights alone" in split["reason"]
    assert command_args(split["command"]) == {"quantization=fp8"}
    alone = _analyse(project, capsys, *shlex.split(split["command"])[4:])
    assert alone.run_dir is not None and alone.code == 0
    rest = _try_first(alone.run_dir)[0]
    assert set(rest["members"]) == set(STACKED[1:])


def _without(command: str, *words: str) -> list[str]:
    split = shlex.split(command)
    at = split.index(words[0])
    assert split[at : at + len(words)] == list(words)
    return split[:at] + split[at + len(words) :]


def test_a_stack_whose_answer_check_failed_twice_is_followed_by_its_lossless_remainder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _queueing_project(tmp_path, capsys)
    _stacking(monkeypatch, Prediction(tpot=-0.2, req_s=0.25, text="predicted: TPOT -20%"))
    base = _analyse(project, capsys)
    assert base.run_dir is not None
    stack = _try_first(base.run_dir)[0]
    assert set(stack["members"]) == set(STACKED) and stack["quality_risk"]
    sibling = _analyse(project, capsys, "--engine-arg", "max-num-batched-tokens=2048")
    assert sibling.run_dir is not None and sibling.code == 0
    at = ["--project", str(project)]
    assert (
        main(["reject", sibling.run_dir.name, *at, "--reason", "judge: excess worse 0.05 > 0.03"])
        == 0
    )
    capsys.readouterr()
    together = _analyse(project, capsys, *shlex.split(stack["command"])[4:])
    assert together.run_dir is not None and together.code == 0
    step = together.run_dir.name
    assert _window(together)["parent"] == base.run_dir.name

    assert main(["reject", step, *at, "--reason", "judge: timeout"]) == 0
    assert "tensward report offers it once more" in capsys.readouterr().out
    assert main(["report", base.run_dir.name, *at]) == 0
    assert _without(_try_first(base.run_dir)[0]["command"], "--reuse-run", step) == shlex.split(
        stack["command"]
    )
    from tensward.reuse import reusable_run

    assert reusable_run(project, step, sibling.run_dir.name) is not None
    measurement = together.run_dir / "metrics.json"
    stamp = measurement.stat()
    yesterday = time.time() - 24 * 3600 - 60
    os.utime(measurement, (yesterday, yesterday))
    try:
        assert main(["report", base.run_dir.name, *at]) == 0
        assert _without(_try_first(base.run_dir)[0]["command"], "--remeasure") == shlex.split(
            stack["command"]
        )
    finally:
        os.utime(measurement, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert main(["report", base.run_dir.name, *at]) == 0
    assert (
        f"tried in run {step}: its answer check did not run to a result (judge: timeout), so it "
        "is offered once more"
    ) in capsys.readouterr().out
    again = _try_first(base.run_dir)[0]
    assert _without(again["command"], "--reuse-run", step) == shlex.split(stack["command"])
    started = time.monotonic()
    reused = _analyse(project, capsys, *shlex.split(again["command"])[4:])
    assert time.monotonic() - started < 30 and reused.code == 0 and reused.run_dir is not None
    assert _record(reused.run_dir)["reused_from"]["run"] == step
    answers = "responses.jsonl"
    assert (reused.run_dir / answers).read_bytes() == (together.run_dir / answers).read_bytes()

    assert main(["reject", reused.run_dir.name, *at, "--reason", "judge: judge_failed"]) == 0
    assert "no longer offer it" in capsys.readouterr().out
    assert main(["report", base.run_dir.name, *at]) == 0

    capsys.readouterr()
    remainder = next(b for b in _try_first(base.run_dir) if b.get("members"))
    assert set(remainder["members"]) == set(STACKED[1:])
    assert not remainder["quality_risk"] and remainder["checks_answers"]
    assert "quantization" not in remainder["command"]
    assert "kv-cache-memory-bytes" not in remainder["command"]


def _load_watch_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    monkeypatch.setattr("tensward.runtime.LOAD_WATCH_EVERY_S", 0.2)
    monkeypatch.setenv("FAKE_STATS_EVERY_S", "0.5")
    monkeypatch.setenv("FAKE_TOKEN_DELAY_S", "0.005")
    project = _guarded_project(tmp_path)
    return project, shlex.join([sys.executable, str(FAKE_SERVER)])


def _analyse_with(project: Path, command: str, *args: str) -> int:
    return main(["analyse", "--project", str(project), "--runtime", "local",
                 "--local-command", command, "--ready-timeout", "60", *args])  # fmt: skip


def test_a_step_that_queues_where_its_parent_did_not_is_stopped_in_load_and_left_to_reject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tensward.history import stopped_runs

    project, fake = _load_watch_project(tmp_path, monkeypatch)
    assert _analyse_with(project, fake) == 0
    parent = next((project / "runs").iterdir())
    started = json.loads((parent / "window.json").read_text())["startup"]
    assert started["start_class"] == "cache_hit" and "ready" in started["phases"]
    capsys.readouterr()
    monkeypatch.setenv("FAKE_LOAD_QUEUE", "1")

    code = _analyse_with(project, fake, "--engine-arg", "enable-prefix-caching")

    assert code != 0
    assert "stopped in load: 40 requests waiting at 96% KV usage" in capsys.readouterr().err
    ((run_id, args, reason),) = stopped_runs(project)
    run_dir = project / "runs" / run_id
    assert args == frozenset({"enable-prefix-caching"}) and reason.startswith("stopped in load")
    assert (run_dir / "report.json").is_file() and not (run_dir / "metrics.json").exists()
    watch = json.loads((run_dir / "window.json").read_text())["load_watch"]
    assert watch["state"] == "stopped" and watch["parent"] == parent.name
    assert main(["reject", run_id, "--project", str(project)]) == 0
    assert "enable-prefix-caching" in capsys.readouterr().out
    assert main(["set", "last", "--project", str(project)]) == 0
    assert f"run {parent.name} (the latest run)" in capsys.readouterr().out


def test_a_step_is_not_stopped_in_load_when_its_parent_queued_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensward.history import stopped_runs

    project, fake = _load_watch_project(tmp_path, monkeypatch)
    at = ["--project", str(project)]
    monkeypatch.setenv("FAKE_LOAD_QUEUE", "1")
    cache = ["--engine-arg", "enable-prefix-caching"]
    assert _analyse_with(project, fake, *cache) == 0
    base = next((project / "runs").iterdir()).name
    assert main(["set", base, *at]) == 0
    monkeypatch.delenv("FAKE_LOAD_QUEUE")

    assert _analyse_with(project, fake, *cache, "--engine-arg", "max-num-batched-tokens=1024") == 0

    assert not stopped_runs(project)
    assert main(["set", base, *at]) == 0
    monkeypatch.setenv("FAKE_LOAD_QUEUE", "1")

    assert _analyse_with(project, fake, *cache, "--engine-arg", "max-num-batched-tokens=2048") == 0

    assert not stopped_runs(project)
    last = sorted((project / "runs").iterdir())[-1]
    assert json.loads((last / "window.json").read_text())["parent"] == base

    monkeypatch.delenv("FAKE_LOAD_QUEUE")
    assert main(["set", "registered", *at]) == 0
    assert _analyse_with(project, fake) == 0
    registered = sorted((project / "runs").iterdir())[-1].name
    assert _analyse_with(project, fake, "--engine-arg", "max-num-batched-tokens=1024") == 0
    assert main(["set", registered, *at]) == 0

    assert _analyse_with(project, fake, "--engine-arg", "max-num-batched-tokens=4096") == 0

    last = sorted((project / "runs").iterdir())[-1]
    assert json.loads((last / "window.json").read_text())["parent"] == registered


def test_a_step_that_declares_a_smaller_kv_pool_is_not_stopped_in_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensward.history import stopped_runs

    project, fake = _load_watch_project(tmp_path, monkeypatch)
    assert _analyse_with(project, fake) == 0
    monkeypatch.setenv("FAKE_LOAD_QUEUE", "1")

    assert _analyse_with(project, fake, "--engine-arg", "gpu-memory-utilization=0.7") == 0

    assert not stopped_runs(project)


def test_a_load_watcher_the_run_has_ended_records_and_stops_nothing() -> None:
    from tensward.context import AnalysisContext
    from tensward.runtime import _watch_load, end_load_watch

    class Server:
        stopped = False

        def is_running(self) -> bool:
            return True

        def log_text(self) -> str:
            return (
                "10-12 18:00:00 [loggers.py] Running: 3 reqs, Waiting: 40 reqs, "
                "GPU KV cache usage: 96.0%"
            )

        def stop(self) -> None:
            self.stopped = True

    context = AnalysisContext()
    context.window_extra["load_watch"] = {"parent": "p", "state": "watching"}
    end_load_watch(context)
    server = Server()

    _watch_load(server, context.window_extra["load_watch"], 0.01)  # type: ignore[arg-type]

    assert context.window_extra["load_watch"] == {"parent": "p", "state": "ended"}
    assert not server.stopped


def test_an_mxfp4_checkpoint_is_analysed_and_offered_only_lossless_changes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for module in ("project", "measure"):
        monkeypatch.setattr(
            f"tensward.{module}.detect_devices", lambda: (GpuInfo("NVIDIA L4", 23034 * 2**20, 0),)
        )
    model, _, prompts = make_registration_inputs(tmp_path)
    make_mxfp4_checkpoint(model)
    workload = {**REGISTRATION_CONFIG["workload"], "request_count": 40,
                "arrival": {"kind": "closed_loop", "concurrency": 8}}  # fmt: skip
    case = {**REGISTRATION_CONFIG["case"], "weight_precision": "mxfp4",
            "activation_dtype": "bfloat16"}  # fmt: skip
    config = write_config(
        tmp_path / "mxfp4.json", {**REGISTRATION_CONFIG, "workload": workload, "case": case}
    )
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    capsys.readouterr()

    done = _analyse(project, capsys)

    assert done.code == 0 and done.run_dir is not None
    report = json.loads((done.run_dir / "report.json").read_text())
    blocks = [b for s in report["sections"] for b in s["blocks"] if b.get("kind") == "suggestion"]
    assert {b["name"] for b in blocks} == {"kv-sized-concurrency", "raise-concurrency"}
    assert not any(b.get("quality_risk") for b in blocks)


def test_with_tools_a_run_short_of_kv_space_tries_the_fp8_pair_in_one_step_then_fp8_kv_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    loop = {"kind": "closed_loop", "concurrency": 4}
    project = _chat_project(tmp_path, "pair", tool_calling=True, request_count=24, arrival=loop)
    entries = {entry.name: entry for entry in VLLM.playbook()}
    drifting = dataclasses.replace(  # the fake server refuses fp8 and answers fp8_e4m3 drifted
        entries["fp8-kv-cache"],
        apply=lambda situation: dataclasses.replace(situation.settings, kv_cache_dtype="fp8_e4m3"),
        gates=(),  # its Qwen2 gate
    )
    found = [
        Suggestion(entry=entry, reason=f"{entry.name} fires", tier="try_first",
                   addresses=frozenset({"kv_capacity"}))
        for entry in (entries["fp8-weights"], drifting)
    ]  # fmt: skip
    monkeypatch.setattr("tensward.suggest.applicable", lambda *a, **k: (found, []))
    monkeypatch.setattr(
        "tensward.suggest.predict",
        lambda situation, proposed, *, entry, **kw: FIXED if entry == "fp8-weights" else None,
    )
    pair = Prediction(tpot=-0.1, req_s=0.6, text="predicted: req/s +60% with about 4 running")
    monkeypatch.setattr("tensward.suggest.predict_pair", lambda situation, **kw: pair)
    short = ("--engine-arg", "gpu-memory-utilization=0.05")
    base = _analyse(project, capsys, "--goal", "throughput", *short)
    assert base.run_dir is not None and base.metrics is not None
    assert base.metrics["preemptions"] > 0
    stack = _try_first(base.run_dir)[0]
    assert stack["members"] == ["fp8-weights", "fp8-kv-cache"]
    assert {"quantization=fp8", "kv-cache-dtype=fp8_e4m3"} <= command_args(stack["command"])
    together = _analyse(project, capsys, *shlex.split(stack["command"])[4:])
    assert together.run_dir is not None
    at = ["--project", str(project)]

    assert main(["reject", together.run_dir.name, *at, "--reason", "judge: worse"]) == 0
    assert main(["report", base.run_dir.name, *at]) == 0

    capsys.readouterr()
    alone = _try_first(base.run_dir)[0]
    assert alone["name"] == "fp8-kv-cache" and "members" not in alone
    assert command_args(alone["command"]) - {"gpu-memory-utilization=0.05"} == {
        "kv-cache-dtype=fp8_e4m3"
    }


def test_a_tpot_session_measures_full_step_0_and_a_judged_child_at_least_150(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [{"id": f"q{i}", "prompt": f"question {i} about parcels"} for i in range(200)]
    project = _windowed(tmp_path, monkeypatch, rows, request_count=200)
    monkeypatch.setattr(window, "WINDOWS_S", (3.0,))
    monkeypatch.setattr(window, "JUDGE_REQUESTS", 150)
    _stacking(monkeypatch, Prediction(tpot=-0.2, req_s=0.25, text="predicted: TPOT -20%"))
    first = _analyse(project, capsys, "--goal", "tpot")
    assert first.code == 0 and first.run_dir is not None
    assert len(_sent(first)) == _window(first)["plan"]["requests"] == 200
    assert _window(first)["plan"]["reason"] == "no earlier run to plan from"
    assert "window: configured count (no earlier run to plan from)" in first.err
    stack = _try_first(first.run_dir or tmp_path)[0]
    assert stack["quality_risk"]

    monkeypatch.setattr(window, "MIN_PER_CLIENT", 25)  # its per-client floor is only 50
    judged = _analyse(project, capsys, *shlex.split(stack["command"])[4:])

    assert judged.code == 0 and judged.run_dir is not None
    assert len(_sent(judged)) == _window(judged)["plan"]["requests"] == 150
    assert _window(judged)["parent"] == first.run_dir.name
    assert _window(judged)["plan"]["seconds"] == 3.0
    assert _sent(judged) == _sent(first)[:150]
    assert judged.responses is not None and len(judged.responses) == 150
    quality = json.loads((judged.run_dir / "quality.json").read_text())
    assert {pair["prompt_id"] for pair in quality["pairs"]} == set(_sent(judged))
    assert quality["summary"]["prompts"] == 150
    assert quality["summary"]["unanswered"] == 0
    assert "on the 150 prompts both runs sent" in judged.report
    monkeypatch.setattr(window, "MIN_PER_CLIENT", 4)
    for goal in ("tpot", "ttft"):
        again = _analyse(project, capsys, "--goal", goal, "--remeasure")
        assert len(_sent(again)) == _window(again)["plan"]["requests"] == 200, goal
        assert _window(again)["plan"]["reason"] == "no earlier run to plan from", goal


def test_engine_env_reaches_the_server_and_the_next_command_but_not_the_callers_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tensward.engines.vllm.engine import VllmEngine
    from tensward.history import past_runs

    seen: list[str | None] = []
    start = LocalProcessRuntime.start

    def started(self: LocalProcessRuntime, spec: ServeSpec) -> RunningServer:
        seen.append(os.environ.get("VLLM_USE_V2_MODEL_RUNNER"))
        return start(self, spec)

    monkeypatch.setattr(LocalProcessRuntime, "start", started)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")  # restored after the test
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "DEBUG")
    project = _structured_project(tmp_path, "loop")  # its compact-json step prints a command
    run = _analyse(project, capsys, "--engine-env", "VLLM_USE_V2_MODEL_RUNNER=0")
    assert run.code == 0 and run.run_dir is not None and seen and set(seen) == {"0"}
    assert _record(run.run_dir)["inherited_env"] == {
        "VLLM_LOGGING_LEVEL": "DEBUG", "VLLM_USE_V2_MODEL_RUNNER": "0",
    }  # fmt: skip
    report = json.loads((run.run_dir / "report.json").read_text())
    steps = next(s for s in report["sections"] if s["id"] == "next_steps")["blocks"]
    commands = [b["command"] for b in steps if b["kind"] == "suggestion" and b["command"]]
    assert commands and all("--engine-env VLLM_USE_V2_MODEL_RUNNER=0" in c for c in commands)
    assert not any("VLLM_LOGGING_LEVEL" in command for command in commands)
    (past,) = past_runs(project, VllmEngine())
    assert past.engine_args == {"VLLM_USE_V2_MODEL_RUNNER=0"}

    with pytest.raises(SystemExit):
        main(["analyse", "--project", str(project), "--engine-env", "VLLM_ATTENTION_BACKEND=x"])
    assert "VLLM_ATTENTION_BACKEND is not an engine variable" in capsys.readouterr().err
    assert main(["analyse", "--project", str(project), "--runtime", "docker",
                 "--engine-env", "VLLM_USE_V2_MODEL_RUNNER=0"]) != 0  # fmt: skip
    assert "--engine-env needs --runtime local" in capsys.readouterr().err


def test_engine_env_refuses_a_variable_the_registered_setup_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def never(self: LocalProcessRuntime, spec: ServeSpec) -> NoReturn:
        raise AssertionError("a server was started")

    monkeypatch.setattr(LocalProcessRuntime, "start", never)
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts),
                 "--current", f"VLLM_USE_V2_MODEL_RUNNER=1 vllm serve {model}"]) == 0  # fmt: skip
    capsys.readouterr()
    run = _analyse(project, capsys, "--engine-env", "VLLM_USE_V2_MODEL_RUNNER=0")
    assert run.code != 0 and run.run_dir is None
    assert (
        "--engine-env VLLM_USE_V2_MODEL_RUNNER: the registered setup already sets "
        "VLLM_USE_V2_MODEL_RUNNER=1; Tensward cannot override it. Change the registered command, "
        "or drop --engine-env VLLM_USE_V2_MODEL_RUNNER."
    ) in run.err


def test_without_a_knowledge_pack_each_run_keeps_its_facts_and_a_step_adds_a_local_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = Path(os.environ["TENSWARD_KB_HOME"]) / "local" / "rows.jsonl"
    project = _queueing_project(tmp_path, capsys)
    base = _analyse(project, capsys)
    assert base.run_dir is not None
    report = json.loads((base.run_dir / "report.json").read_text())
    assert report["kb"] == {"pack": None}
    assert "Why:" not in base.report and "Worth measuring" not in base.report
    stored = json.loads((base.run_dir / "facts.json").read_text())
    assert stored["facts_version"] == 2 and stored["digest"].startswith("sha256:")
    assert stored["facts"]["goal"] == "throughput" and not rows.exists()

    step = _analyse(project, capsys, "--engine-arg", "max-num-seqs=6")
    assert step.run_dir is not None
    (line,) = rows.read_text().splitlines()
    row = json.loads(line)
    assert row["row"]["source"] == "local" and row["row"]["row_id"] == step.run_dir.name
    assert row["provenance"]["runs"] == {"parent": base.run_dir.name, "child": step.run_dir.name}
    assert row["features"]["facts"] == stored["facts"]
    assert row["action"]["lever_keys"] == ["max_concurrent_requests=up"]
    assert row["action"]["tier"] == "try_first" and str(tmp_path) not in line

    assert main(["report", "last", "--project", str(project), "--kb"]) == 0
    listing = capsys.readouterr().out.splitlines()
    assert listing[0] == "no knowledge pack installed"
    assert (
        any(line.endswith("(try_first)") for line in listing) and "  from the playbook" in listing
    )


def test_an_environment_only_run_is_kept_with_its_environment_and_can_be_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    project = _queueing_project(tmp_path, capsys)
    at = ["--project", str(project)]
    base = _analyse(project, capsys)
    env = _analyse(project, capsys, "--engine-env", "VLLM_USE_V2_MODEL_RUNNER=0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")  # the run left its own value in os.environ
    assert base.run_dir is not None and env.run_dir is not None
    run = env.run_dir.name

    assert main(["set", "last", *at]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"kept the settings of run {run} (the latest run): tensward analyse starts from them",
        "before: the registered setup (no engine arguments)",
        "after:  --engine-env VLLM_USE_V2_MODEL_RUNNER=0",
        f"undo:   tensward set registered --project {project}",
        f"serve:  VLLM_USE_V2_MODEL_RUNNER=0 tensward serve start --project {project} "
        "--from current",
    ]
    assert json.loads((project / "kept.json").read_text())["env"] == ["VLLM_USE_V2_MODEL_RUNNER=0"]
    assert main(["reject", "last", *at]) == 1
    assert f"run {run} is the kept setup" in capsys.readouterr().err

    again = _analyse(project, capsys)
    assert again.run_dir is not None
    assert _record(again.run_dir)["inherited_env"]["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert "starting from the kept setup (run " + run + "): VLLM_USE_V2_MODEL_RUNNER=0" in again.err
    assert f"This run measured the kept setup (run {run})" in again.report
    assert main(["report", "last", *at]) == 0
    commands = [
        b["command"] for s in json.loads((again.run_dir / "report.json").read_text())["sections"]
        for b in s["blocks"] if b.get("kind") == "suggestion" and b["command"]
    ]  # fmt: skip
    assert commands and all("--engine-env VLLM_USE_V2_MODEL_RUNNER=0" in c for c in commands)

    assert main(["set", "registered", *at]) == 0
    capsys.readouterr()
    assert main(["reject", run, *at, "--reason", "no gain"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == (
        f"rejected the change of run {run}: VLLM_USE_V2_MODEL_RUNNER=0 (made from run "
        f"{base.run_dir.name})"
    )


def _aged(run_dir: Path, days: int, cap_w: float | None) -> Path:
    facts_file = run_dir / "facts.json"
    facts = json.loads(facts_file.read_text())
    facts["facts"]["gpu.power_cap_w"] = cap_w
    facts_file.write_text(json.dumps(facts))
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time() - days * 86400))
    return run_dir.rename(run_dir.parent / f"{stamp}-{run_dir.name[-4:]}")


@pytest.mark.parametrize(
    ("days", "cap_w", "refused"),
    [(5, 72.0, None), (15, 72.0, "was measured 15 days ago (the limit is 14 days)"),
     (5, 60.0, "ran under a 60 W power cap, this machine enforces 72 W")],
    ids=["accepted", "fifteen-days", "other-power-cap"],
)  # fmt: skip
def test_borrow_reference_reuses_an_older_reference_and_widens_a_step_from_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    days: int,
    cap_w: float,
    refused: str | None,
) -> None:
    project, first, launches = _reference(tmp_path, monkeypatch, capsys)
    assert first.run_dir is not None
    monkeypatch.setattr("tensward.reuse._current_power_cap", lambda runtime: 72.0)
    old = _aged(first.run_dir, days, cap_w)
    started = len(launches)
    borrowed = _analyse(project, capsys, "--borrow-reference")
    assert borrowed.run_dir is not None
    if refused is not None:
        assert len(launches) > started
        assert f"not reused: run {old.name} {refused}" in borrowed.err + borrowed.out
        return
    assert len(launches) == started
    reused = _record(borrowed.run_dir)["reused_from"]
    assert reused["run"] == old.name and reused["borrowed"] is True
    assert "borrowed as the reference" in borrowed.out
    step = _analyse(project, capsys, "--engine-arg", "max-num-seqs=4")
    assert step.run_dir is not None
    window = json.loads((step.run_dir / "window.json").read_text())
    assert window["parent"] == borrowed.run_dir.name
    ttft = next(d for d in window["deltas"] if d["metric"] == "ttft_p95_ms")
    assert ttft["band"] > 0.12


def test_under_tpot_a_run_held_back_by_kv_space_is_offered_the_held_batch_and_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    loop = {"kind": "closed_loop", "concurrency": 4}
    project = _chat_project(
        tmp_path, "held", tool_calling=True, request_count=24, arrival=loop,
        current="vllm serve {model} --enable-auto-tool-choice --tool-call-parser hermes "
        "--max-num-seqs 16",
    )  # fmt: skip
    entries = {entry.name: entry for entry in VLLM.playbook()}
    drifting = dataclasses.replace(
        entries["fp8-kv-cache"],
        apply=lambda situation: dataclasses.replace(situation.settings, kv_cache_dtype="fp8_e4m3"),
        gates=(),
    )
    found = [
        Suggestion(entry=entry, reason=f"{entry.name} fires", tier="try_first",
                   addresses=frozenset({"kv_capacity"}))
        for entry in (entries["fp8-weights"], drifting)
    ]  # fmt: skip
    monkeypatch.setattr("tensward.suggest.applicable", lambda *a, **k: (found, []))
    monkeypatch.setattr(
        "tensward.suggest.predict",
        lambda situation, proposed, *, entry, **kw: FIXED if entry == "fp8-weights" else None,
    )
    held = Prediction(tpot=-0.3, req_s=0.5, text="predicted: TPOT -30%, req/s +50% at this cap")
    monkeypatch.setattr("tensward.suggest.predict_stack", lambda situation, members, **kw: held)
    monkeypatch.setattr("tensward.suggest.demand", lambda m: 8)
    monkeypatch.setattr("tensward.suggest.LATENCY_BOUND", 100.0)  # the fake server's TTFT is ms
    short = ("--engine-arg", "gpu-memory-utilization=0.05")
    base = _analyse(project, capsys, "--goal", "tpot", *short)
    assert base.run_dir is not None and base.metrics is not None
    assert base.metrics["preemptions"] > 0
    stack = _try_first(base.run_dir)[0]
    assert stack["members"] == ["fp8-weights", "lower-concurrency"]
    assert stack["reason"].startswith("Held batch: cap 4, predicted TPOT -30%")
    assert {"max-num-seqs=4", "quantization=fp8"} <= command_args(stack["command"])

    together = _analyse(project, capsys, *shlex.split(stack["command"])[4:])

    assert together.run_dir is not None and together.code == 0
    assert len(_sent(together)) == _window(together)["plan"]["requests"]
    assert {"max-num-seqs=4", "quantization=fp8"} <= set(_record(together.run_dir)["engine_args"])


@pytest.mark.parametrize(
    ("change", "refused"),
    [({"gpus_identity": [{"uuid": "GPU-elsewhere"}]}, "ran on another machine's GPU"),
     ({"imported_from": {"run": "x", "project": "/p"}}, "was imported from another project")],
    ids=["other-uuid", "imported"],
)  # fmt: skip
def test_borrow_reference_refuses_a_run_from_another_machine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    change: dict[str, Any],
    refused: str,
) -> None:
    project, first, launches = _reference(tmp_path, monkeypatch, capsys)
    assert first.run_dir is not None
    monkeypatch.setattr("tensward.reuse._current_power_cap", lambda runtime: 72.0)
    old = _aged(first.run_dir, 5, 72.0)
    record = _record(old)
    if "gpus_identity" in change:
        change = {"gpus_identity": [{**record["gpus_identity"][0], **change["gpus_identity"][0]}]}
    (old / "run.json").write_text(json.dumps({**record, **change}))
    started = len(launches)
    borrowed = _analyse(project, capsys, "--borrow-reference")
    assert len(launches) > started
    assert f"not reused: run {old.name} {refused}" in borrowed.err + borrowed.out


def test_a_local_training_row_holds_no_path_from_an_engine_argument(tmp_path: Path) -> None:
    from tensward.kb_run import _delta

    base = Settings()
    path = str(tmp_path / "t.jinja")
    after = dataclasses.replace(
        base, extra_args={"chat-template": path, "speculative-config": f'{{"model": "{path}"}}',
                          "max-model-len": "8192"},
    )  # fmt: skip
    _, flags = _delta(VLLM, base, after)
    assert flags["chat-template"] == [None, "<set>"] and flags["max-model-len"] == [None, "8192"]
    assert str(tmp_path) not in json.dumps(flags)


def test_the_kept_setup_s_env_meets_the_same_refusals_as_engine_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def never(self: LocalProcessRuntime, spec: ServeSpec) -> NoReturn:
        raise AssertionError("a server was started")

    monkeypatch.setattr(LocalProcessRuntime, "start", never)
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts),
                 "--current", f"VLLM_USE_V2_MODEL_RUNNER=1 vllm serve {model}"]) == 0  # fmt: skip
    capsys.readouterr()
    snapshot = json.loads((project / "project.json").read_text())["snapshot_id"]
    (project / "kept.json").write_text(json.dumps({
        "run": "20260101T000000Z-0000", "engine_args": [], "answer_changes": [],
        "previous": "registered", "snapshot_id": snapshot,
        "env": ["VLLM_USE_V2_MODEL_RUNNER=0"],
    }))  # fmt: skip

    local = _analyse(project, capsys)
    assert local.code != 0 and local.run_dir is None
    assert "the registered setup already sets VLLM_USE_V2_MODEL_RUNNER=1" in local.err

    assert main(["analyse", "--project", str(project), "--runtime", "docker"]) != 0
    assert "--engine-env needs --runtime local" in capsys.readouterr().err

    name = "TENSWARD_ENGINE_LAUNCHER"
    monkeypatch.setenv(name, "")
    monkeypatch.delenv(name)
    kept_file = project / "kept.json"
    kept = json.loads(kept_file.read_text())
    kept_file.write_text(json.dumps({**kept, "env": [f"{name}=false"]}))
    injected = _analyse(project, capsys)
    assert injected.code != 0 and injected.run_dir is None
    assert name in injected.err and "not an engine variable" in injected.err
    assert name not in os.environ
