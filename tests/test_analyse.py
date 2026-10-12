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
from pydantic import ValidationError
from registration_fixtures import (
    QUANTIZED_CHECKPOINTS,
    REGISTRATION_CONFIG,
    SAFETENSORS_NAME,
    make_gemma4_checkpoint,
    make_image_workload,
    make_registration_inputs,
    probe_registration,
    registered_project,
    safetensors_from_tensors,
    write_config,
    write_prompts,
)
from test_playbook import situation

from tensward import capacity, progress
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
from tensward.inputs import PromptEntry
from tensward.measurement import Measurement
from tensward.platforms import Device as GpuInfo
from tensward.playbook import WorkloadFacts, ranked
from tensward.profiling.trace import read_trace
from tensward.progress import ProgressEvent
from tensward.project import apply_engine_args, sample_of
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
        "PhaseStarted", "Note", "ServerLoading", "ServerReady", "WindowOpened", "RequestsDone",
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


def _queueing_project(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    """Eight clients against a cap of 1: kv-sized-concurrency (6) leads, raise-concurrency (4)
    is the other change."""
    model, _, prompts = make_registration_inputs(tmp_path)
    workload = {**REGISTRATION_CONFIG["workload"], "request_count": 40,
                "arrival": {"kind": "closed_loop", "concurrency": 8}}  # fmt: skip
    config = write_config(tmp_path / "queueing.json", {**REGISTRATION_CONFIG, "workload": workload})
    project = tmp_path / "project"
    assert main(["init", "--project", str(project), "--model", str(model),
                 "--config", str(config), "--prompts", str(prompts)]) == 0  # fmt: skip
    capsys.readouterr()
    return project


def _offered(run_dir: Path) -> dict[str, str]:
    report = json.loads((run_dir / "report.json").read_text())
    return {b["name"]: b["tier"] for s in report["sections"] for b in s["blocks"]
            if b.get("kind") == "suggestion"}  # fmt: skip


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
