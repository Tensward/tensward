"""End-to-end ``tensward analyse`` against a fake vLLM server, plus the docker argv contract."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import shlex
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

from tensward.analyse import AnalyseFailure, apply_engine_args, suggestion_command
from tensward.ceilings import gpu_spec
from tensward.classify import classify
from tensward.cli import main
from tensward.client import RequestPlan, run_workload
from tensward.engines import ENGINES
from tensward.engines.protocol import Settings
from tensward.engines.vllm import SPECULATIVE_CONFIG_FLAG
from tensward.engines.vllm_playbook import NGRAM_SPECULATION, PLAYBOOK
from tensward.images import ImageSource
from tensward.measurement import Measurement
from tensward.platforms import Device as GpuInfo
from tensward.playbook import (
    Entry,
    LowBar,
    WorkloadFacts,
    applicable,
    fitting_max_context_len,
    ranked,
    rungs,
)
from tensward.report import render_next_steps
from tensward.runtime import DockerRuntime, ServeSpec, process_start_time
from tensward.workload import (
    ArrivalSpec,
    ChatMessage,
    ChatRequest,
    ImagePart,
    ImageUrl,
    WorkloadSpec,
)

SHORT_WINDOW = (
    "the measurement window was only {} s; throughput is noisy; raise request_count to about {}"
)


def checks_of(report: str) -> list[str]:
    """The lines of the report's Checks section, with the short-window line's length masked."""
    if "## Checks" not in report:
        return []
    section = report.split("## Checks", 1)[1].split("\n## ", 1)[0]
    return [
        re.sub(r"about \d+", "about {}", re.sub(r"only [\d.]+ s", "only {} s", line[2:]))
        for line in section.splitlines()
        if line.startswith("- ")
    ]


def suggest(
    signals: Measurement, facts: WorkloadFacts, settings: Settings, *, allow_quality_changes: bool
) -> list[tuple[Entry, str]]:
    found, _ = applicable(
        PLAYBOOK, signals, facts, settings, allow_quality_changes=allow_quality_changes
    )
    return found


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
    monkeypatch.setattr("tensward.analyse.shutil.which", lambda name: None)  # no Nsight Compute
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
    ran = summary.splitlines()[2]
    assert ran.startswith("Ran: vLLM version unknown (local command python")
    assert ran.endswith("; checkpoint: safetensors, awq int4 group 128")
    assert json.loads((run_dir / "run.json").read_text())["format"] == "hf-safetensors"
    assert (
        "\n## Your current setup + overrides (no-async-scheduling, compilation-config=" in summary
    )
    assert "\n- source: declared in config\n" in summary
    assert "- settings: dtype=float16, max_concurrent_requests=1, max_context_len=2048" in summary
    assert "3 of 10 requests failed" in summary and "hardware ceiling reached " in summary
    assert "your current setup fails requests" in summary
    assert "10 requests from 3 distinct prompts: prefix-cache hit rate" in summary
    assert summary.index("## Diagnosis") < summary.index("## What to try next")
    assert summary.index("## What to try next") < summary.index("## Your current setup")
    assert "only 7 requests succeeded" in summary  # 10 requests, 3 too long
    assert "prompts that do not fit max_context_len 2048: 1" in summary
    assert "  - GPU compute, tensor-bound kernels — kernel counters: " in summary  # no ncu here
    assert "For host / CPU overhead:" in summary  # the synthetic trace is 22% idle
    assert "  evidence: " in summary and "; may cost: " in summary
    assert not re.search(r"\b(?:[A-Z]{2,4}-\d+|A\d+\.\d+)\b", summary)  # no private catalog IDs
    diagnosis = json.loads((run_dir / "metrics.json").read_text())["diagnosis"]
    assert [f["bottleneck"] for f in diagnosis["findings"]] == [
        "queueing", "kv_capacity", "prefill", "prefill_stalls_decode", "decode_bandwidth",
        "gpu_compute", "host_overhead", "long_context", "speculation",
    ]  # fmt: skip
    assert diagnosis["gates"][-1]["state"] == "critical"  # three requests did not fit
    assert "queue_share" in diagnosis["thresholds"] and diagnosis["not_applicable"]
    assert "  Try: `tensward analyse --project " in summary and "--engine-arg " in summary
    assert "Not applicable here:\n" in summary
    assert "- `ngram-speculation`: the pinned CUDA graph sizes (max 3)" in summary
    assert "- 3 x HTTP 400: This model's maximum context length is 2048" in summary
    assert "(prompts: too-long)" in summary
    assert "- checkpoint: awq int4 group 128" in summary
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["kv_capacity_tokens"] == 40416 and metrics["kv_capacity_estimate_tokens"]
    assert "KV cache capacity (measured by the engine): 40416 tokens, 1.2 full-length" in summary
    assert "- kernel: MarlinLinearKernel (AutoAWQMarlinLinearMethod)\n" in summary
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["quant_kernels"] == [
        {"name": "MarlinLinearKernel", "layer": "AutoAWQMarlinLinearMethod", "slow": False}
    ]
    assert "## Prompts that do not fit the context window" in summary
    assert "the longest needs 3032 tokens" in summary
    assert "`fit-context`" in summary  # 3032 tokens fit the model's 4096, in steps of 256
    assert (
        "`prefix-caching`: 2 of 3 prompts (67%) share a prompt prefix of up to 300 words" in summary
    )
    # Total counts prompt tokens too; goodput counts only requests inside the SLO (5 s TTFT here).
    assert metrics["total_throughput"] > metrics["output_throughput"] > 0
    assert 0 < metrics["goodput"] <= metrics["request_throughput"]
    assert metrics["slo_attainment"] == pytest.approx(0.7)  # the 3 refused requests miss it
    assert "- goodput (TTFT <= 5000 ms and TPOT <= 100 ms per request): " in summary
    assert "- total throughput (prompt + output): " in summary
    assert "- start-up wave (first 1 request, " in summary
    assert metrics["startup_wave"]["requests"] == 1
    ceilings = metrics["ceilings"]
    assert "## Hardware ceilings (theoretical upper bounds, not targets)" in summary
    assert "- GPU: L4\n- memory bandwidth: 300.0 GB/s (bandwidth: datasheet" in summary
    assert "- dense tensor rate: 121 TFLOPS (datasheet)" in summary
    assert ceilings["gpu"] == "L4" and ceilings["unavailable"] is None
    assert ceilings["decode_ceiling_batch1_tok_s"] > 0 and ceilings["prefill_ceiling_tok_s"] > 0
    assert ceilings["measured_decode_tok_s"] > 0 and ceilings["measured_prefill_tok_s"] > 0
    assert gpu_spec((("NVIDIA Foo", 1),), None) == (0, "NVIDIA Foo", None)
    trace = metrics["trace"]  # a separate launch; the fake server serves the synthetic trace
    assert trace["status"] == "ok" and trace["kernels"] == 80
    assert "## GPU timeline (profiled - diagnostic only)" in summary
    assert "GPU busy 67.6%, 80 kernels" in summary
    assert "- 3 idle gaps of at least 50 us with no GPU activity: 21.7% of the window" in summary
    assert "NOT TRUSTED" not in summary
    assert "Cause analysis of GPU idle time and the fixes for it are available with " in summary
    assert "Tensward Optimize." in summary
    assert "Where the GPU goes idle" not in summary and "`async-scheduling`" not in summary
    assert trace["gap_count"] == 3 and trace["analysis"] is None
    # --counters implies --trace; without ncu the report says so and the analysis still succeeds.
    assert "## Kernel counters (Nsight Compute - diagnostic only)" in summary
    assert "- counters unavailable (ncu not found)" in summary
    assert metrics["counters"]["status"] == "unavailable" and metrics["counters"]["kernels"] == []
    assert not {"evidence.json", "report.json"} & names
    assert checks_of(summary) == [SHORT_WINDOW]  # every vLLM 0.30 metric is exposed and read

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


def _chat_project(tmp_path: Path, name: str, *, tool_calling: bool) -> Path:
    """Register a chat workload (one plain chat, one offering a tool) on a Qwen2-family model."""
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

    assert main(["analyse", "--project", str(project), "--runtime", "local",
                 "--engine-arg", "max-num-seqs=3",
                 "--local-command", command, "--ready-timeout", "30"]) == 0  # fmt: skip

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


def test_a_batch_below_one_media_item_is_not_suggested_while_media_inputs_are_on() -> None:
    stalled = Measurement(1, 0, tpot_p50_ms=10.0, tpot_p95_ms=50.0)
    settings = Settings(prefill_batch_tokens=4096)
    names = lambda f, s: [r.name for r, _ in suggest(stalled, f, s, allow_quality_changes=False)]  # noqa: E731
    assert "lower-prefill-batch" in names(WorkloadFacts(("p",), 128, 4096), settings)
    media = WorkloadFacts(("p",), 128, 4096, media_encoders=True)
    assert "lower-prefill-batch" not in names(media, settings)
    _, gated = applicable(PLAYBOOK, stalled, media, settings, allow_quality_changes=False)
    assert [(entry.name, "media item" in why) for entry, why in gated] == [
        ("lower-prefill-batch", True)
    ]
    text_only = dataclasses.replace(settings, media_inputs=False)
    assert "lower-prefill-batch" in names(media, text_only)


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

    on = _analyse(_chat_project(tmp_path, "on", tool_calling=True), capsys)
    assert on.code == 0
    report, responses = on.report, on.responses
    assert checks_of(report) == [SHORT_WINDOW] and "enable-tool-calling" not in report
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
    monkeypatch.setattr("tensward.analyse.detect_devices", lambda: (a10,))
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


def test_vllm_quant_kernels_come_from_the_log_and_slow_ones_suggest_dropping_the_override() -> None:
    engine = ENGINES["vllm"]
    log = (
        "INFO Using ExllamaLinearKernel for AutoGPTQLinearMethod\n"
        "INFO Selected CutlassFP8ScaledMMLinearKernel for Fp8LinearMethod\n"
        "WARNING Layer 'x' is not supported by AutoAWQMarlin. "
        "Falling back to unoptimized AWQ kernels.\n"
    )
    kernels = engine.parse_quant_kernels(log)
    assert [(k.name, k.slow) for k in kernels] == [
        ("ExllamaLinearKernel", True),
        ("CutlassFP8ScaledMMLinearKernel", False),
        ("unoptimized AWQ kernels", True),
    ]
    assert engine.parse_quant_kernels("INFO Using FLASH_ATTN attention backend") == ()

    measurement = Measurement(1, 0, quant_kernels=kernels)
    facts = WorkloadFacts(("p",), 128, 4096)
    forced = Settings("float16", 8, 2048, 0.8, 2048, "auto", True, quantization="gptq")
    ((recipe, reason),) = suggest(measurement, facts, forced, allow_quality_changes=False)
    assert recipe.name == "fast-quant-kernel" and "ExllamaLinearKernel" in reason
    assert recipe.apply(measurement, facts, forced).quantization is None
    auto = dataclasses.replace(forced, quantization=None)
    assert suggest(measurement, facts, auto, allow_quality_changes=False) == []


def test_ngram_speculation_is_judged_on_the_part_of_the_prompt_not_shared() -> None:
    measurement = Measurement(1, 0, tpot_p50_ms=10.0)
    settings = Settings()

    def names(facts: WorkloadFacts) -> list[str]:
        found = suggest(measurement, facts, settings, allow_quality_changes=False)
        return [recipe.name for recipe, _ in found]

    unique = tuple(" ".join(f"w{i}x{j}" for j in range(300)) for i in range(8))
    agent = tuple(" ".join(["system"] * 1900) + f" turn {i} " + "ask " * 12 for i in range(8))
    assert "ngram-speculation" in names(WorkloadFacts(unique, 128, 4096))
    assert "ngram-speculation" not in names(WorkloadFacts(agent, 128, 4096))


def test_any_fp8_kv_cache_counts_as_already_applied() -> None:
    facts = WorkloadFacts(("p",), 128, 4096)
    pressured = Measurement(1, 0, peak_kv_usage=0.97)
    settings = Settings("float16", 8, 2048, 0.8, 2048, "auto", True)
    names = lambda s: [r.name for r, _ in suggest(pressured, facts, s, allow_quality_changes=True)]  # noqa: E731
    assert "fp8-kv-cache" in names(settings)
    for dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
        assert "fp8-kv-cache" not in names(dataclasses.replace(settings, kv_cache_dtype=dtype))


def test_every_recipe_change_round_trips_through_engine_args() -> None:
    engine = ENGINES["vllm"]
    before = Settings("float16", 8, 2048, 0.8, 2048, "auto", True, quantization="gptq")
    after = dataclasses.replace(
        before,
        quantization=None,
        kv_memory_fraction=0.95,
        kv_cache_dtype="fp8",
        prefix_caching=True,
        cuda_graphs=False,
        extra_args={"--speculative-config": '{"method": "ngram"}'},
    )
    texts = engine.engine_args_between(before, after)
    assert "quantization=none" in texts and "gpu-memory-utilization=0.95" in texts
    rebuilt = before
    for text in texts:
        rebuilt = engine.with_engine_arg(rebuilt, text)
    assert rebuilt == after
    assert engine.engine_args_between(after, after) == []
    overridden = dataclasses.replace(before, max_concurrent_requests=4)
    command = suggestion_command(
        engine, Path("p"), before, dataclasses.replace(overridden, max_concurrent_requests=32)
    )
    assert command.count("max-num-seqs") == 1 and "max-num-seqs=32" in command


def test_suggested_settings_stay_startable_and_keep_their_graphs() -> None:
    engine = ENGINES["vllm"]

    def parse(flags: str) -> Settings:
        return engine.parse_setup(f"vllm serve /m {flags}").settings

    def speculating(settings: Settings, config: str = NGRAM_SPECULATION) -> Settings:
        return dataclasses.replace(
            settings, extra_args={**settings.extra_args, SPECULATIVE_CONFIG_FLAG: config}
        )

    pinned = parse(
        """--max-num-seqs 3 --async-scheduling -cc '{"cudagraph_capture_sizes": [1, 2, 3]}'"""
    )
    wider, note = engine.consistent(pinned, dataclasses.replace(pinned, max_concurrent_requests=8))
    sizes = json.loads(wider.extra_args["--compilation-config"])["cudagraph_capture_sizes"]
    assert max(sizes) >= 8 and note and "graph" in note
    assert engine.max_graph_batch(pinned) == 3 and engine.max_graph_batch(wider) >= 8
    cached = dataclasses.replace(pinned, prefix_caching=True)
    assert engine.consistent(pinned, cached) == (cached, None)  # a low pin is left alone

    started, note = engine.consistent(pinned, speculating(pinned))
    assert started.async_scheduling is False and note and "async" in note
    reason = engine.unstartable(pinned, started)  # sizes [1, 2, 3] fit no 5-token n-gram step
    assert reason and "n-gram" in reason and "max 3" in reason
    assert engine.unstartable(Settings(), speculating(Settings())) is None
    tuned = parse(
        """--max-num-seqs 8 -cc '{"cudagraph_capture_sizes": [1, 2, 3, 4, 5, 6, 7, 8]}'"""
    )
    reason = engine.unstartable(tuned, speculating(tuned))  # n-gram would rebuild them as [5]
    assert reason and "only 1 sequences instead of 8" in reason
    assert engine.unstartable(tuned, tuned) is None

    sixteen = parse("""-cc '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16]}'""")
    assert engine.max_graph_batch(speculating(sixteen)) == 2  # v0.30 rounds them to [5, 10], k = 4
    eagle = '{"method": "eagle", "num_speculative_tokens": 4}'
    assert engine.max_graph_batch(speculating(sixteen, eagle)) == 3  # the V2 runner rounds nothing
    capped = parse("--max-cudagraph-capture-size 4 --max-num-seqs 4")
    wider, _ = engine.consistent(capped, dataclasses.replace(capped, max_concurrent_requests=8))
    assert (
        engine.max_graph_batch(wider) or 0
    ) >= 8 and "--compilation-config" not in wider.extra_args

    assert (
        "--compilation-config.cudagraph_mode"
        in parse("--compilation-config.cudagraph_mode=NONE").extra_args
    )
    for refused in ("--cudagraph-capture-sizes 1 2 3", "-cc.cudagraph_capture_sizes=[1,2]"):
        with pytest.raises(ValueError, match="compilation-config"):
            parse(refused)


def test_docker_run_argv_keeps_the_api_key_out_and_never_pulls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[list[str], dict | None]] = []

    class Done:
        returncode = 0
        stdout = "sha256:abc\n"
        stderr = ""

    def fake_run(argv, **kwargs):
        calls.append((list(argv), kwargs.get("env")))
        return Done()

    monkeypatch.setattr("tensward.runtime.subprocess.run", fake_run)
    secret = "s3cret-api-key-value"
    spec = ServeSpec(
        engine=ENGINES["vllm"],
        model_dir=tmp_path,
        served_model_name="m",
        settings=Settings("bfloat16", 8, 2048, 0.8, 2048, "auto", True, extra_args={"--x": "1"}),
        port=18000,
        api_key=secret,
    )
    server = DockerRuntime("vllm/vllm-openai:test").start(spec)

    inspect_argv, _ = calls[0]
    assert inspect_argv[:3] == ["docker", "image", "inspect"]
    run_argv, run_env = calls[1]
    assert not any(secret in token for token in run_argv)
    assert run_env is not None and run_env["VLLM_API_KEY"] == secret
    assert "--pull" in run_argv and run_argv[run_argv.index("--pull") + 1] == "never"
    assert run_argv[:3] == ["docker", "run", "-d"]
    assert f"{tmp_path}:/model:ro" in run_argv
    assert "127.0.0.1:18000:8000" in run_argv
    image_at = run_argv.index("vllm/vllm-openai:test")
    assert run_argv[image_at + 1 :][:4] == ["--model", "/model", "--served-model-name", "m"]
    assert run_argv[-2:] == ["--x", "1"]  # engine-specific extras come last
    assert "--enable-prefix-caching" in run_argv[image_at:]
    assert server.identity()["image_id"] == "sha256:abc"
    server.stop()
    assert calls[-1][0][:3] == ["docker", "rm", "-f"]


def test_fit_context_rounds_up_and_never_exceeds_the_model_limit() -> None:
    def fit(longest_prompt: int, model_limit: int) -> int | None:
        signals = Measurement(1, 1, max_prompt_tokens=longest_prompt, too_long=("p",))
        return fitting_max_context_len(signals, WorkloadFacts(("p",), 128, model_limit))

    assert fit(2100, 32768) == 2304  # 2228 tokens rounded up to a multiple of 256
    assert fit(2100, 2240) == 2240  # capped by the model, which still holds 2228
    assert fit(2100, 2200) is None  # 2228 tokens cannot fit the model at all


def test_raise_concurrency_jumps_to_observed_demand_within_kv_headroom() -> None:
    from tensward.engines.vllm_playbook import _raised_concurrency

    # Gauges are floats, as vLLM exports them.
    seen = Measurement(10, 0, peak_running=8.0, peak_waiting=24.0, peak_kv_usage=0.06)
    assert _raised_concurrency(seen, 8) == 32  # running + waiting, not just double
    tight = Measurement(10, 0, peak_running=8.0, peak_waiting=24.0, peak_kv_usage=0.5)
    assert _raised_concurrency(tight, 8) == 8  # 8 * 0.85 / 0.5 leaves no room to grow

    # The reason names the declared load; it never presents it as measured traffic.
    recipe = next(r for r in PLAYBOOK if r.name == "raise-concurrency")
    facts = WorkloadFacts(("p",), 128, 4096, load="32 concurrent clients")
    reason = recipe.applies(seen, facts, Settings(max_concurrent_requests=8))
    assert reason is not None and "at your declared load of 32 concurrent clients" in reason
    assert (
        "highest sampled running plus waiting was 32" in reason and "not measured traffic" in reason
    )
    assert "cuts queueing (TTFT) but slows each token (TPOT)" in reason


@pytest.mark.parametrize(
    "signals, settings, offered",
    [
        # TPOT tail 1.6x against the 2.0 gate: near, offered with its value and the gate.
        (dict(tpot_p50_ms=10.0, tpot_p95_ms=16.0), {}, "TPOT p95 is 1.6x p50; the gate is 2.0x"),
        (dict(tpot_p50_ms=10.0, tpot_p95_ms=13.0), {}, None),  # 1.3x is under 0.7 of the gate
        (dict(tpot_p50_ms=10.0, tpot_p95_ms=25.0), {}, None),  # past the gate: exact, not near
        (
            dict(ttft_p50_ms=300.0, tpot_p50_ms=20.0, peak_waiting=3.0),  # 15x against 20
            {},
            "TTFT p50 is 15x TPOT p50 with requests waiting; the gate is 20x",
        ),
        # Queued at a cap of 32 with KV at 60%: raising it cannot fit, so more memory is offered.
        (
            dict(peak_running=32.0, peak_waiting=8.0, peak_in_flight=40, peak_kv_usage=0.6),
            dict(max_concurrent_requests=32),
            "raising max_concurrent_requests above 32 is blocked",
        ),
        (  # already at 0.95: nothing to offer
            dict(peak_running=32.0, peak_waiting=8.0, peak_in_flight=40, peak_kv_usage=0.6),
            dict(max_concurrent_requests=32, kv_memory_fraction=0.95),
            None,
        ),
        (  # exactly at the gate: neither exact nor near, nothing is offered
            dict(tpot_p50_ms=10.0, tpot_p95_ms=20.0),
            {},
            None,
        ),
        (  # at 20% the cap can be raised, so nothing is blocked
            dict(peak_running=32.0, peak_waiting=8.0, peak_in_flight=40, peak_kv_usage=0.2),
            dict(max_concurrent_requests=32),
            None,
        ),
    ],
)
def test_low_bar_entries_are_offered_only_on_request(
    signals: dict[str, Any], settings: dict[str, Any], offered: str | None
) -> None:
    measurement = Measurement(10, 0, **signals)
    facts = WorkloadFacts(("p",), 128, 4096)
    current = Settings(**settings)
    plain, _ = applicable(PLAYBOOK, measurement, facts, current, allow_quality_changes=True)
    found, _ = applicable(
        PLAYBOOK, measurement, facts, current, allow_quality_changes=True, near=True
    )
    low = [reason for _, reason in found if isinstance(reason, LowBar)]
    assert not any(isinstance(reason, LowBar) for _, reason in plain)
    assert [offered in reason for reason in low] == ([True] if offered else [])


@pytest.mark.parametrize(
    "kv, fraction, see",
    [
        (0.6, None, "; see `more-kv-memory` under Could help"),
        (0.89, None, "; see `more-kv-memory` under Could help"),
        (0.92, None, None),  # KV-bound: the diagnosis is KV capacity, more-kv-memory is Try first
        (0.6, 0.95, ""),  # nothing to offer, the line still shows
    ],
)
def test_next_steps_say_why_the_cap_cannot_rise(
    kv: float, fraction: float | None, see: str | None
) -> None:
    measurement = Measurement(
        10, 0, queue_share=0.9, peak_running=32.0, peak_waiting=8.0, peak_in_flight=40,
        peak_kv_usage=kv, tpot_p50_ms=10.0, tpot_p95_ms=16.0,
    )  # fmt: skip
    settings = Settings(max_concurrent_requests=32, kv_memory_fraction=fraction)
    facts = WorkloadFacts(("p",), 128, 4096)
    found, _ = applicable(
        PLAYBOOK, measurement, facts, settings, allow_quality_changes=True, near=True
    )
    suggestions = ranked(found)
    assert suggestions[-1][0].name == "lower-prefill-batch"  # near items come last
    text = render_next_steps(
        classify(measurement, settings, None),
        suggestions,
        lambda e: None,
        blocked={"queueing": "raising the cap is blocked"},
        retained=True,
    )
    if see is None:
        assert (
            "blocked" not in text
            and "Try first:\nFor KV-cache capacity:\n- `more-kv-memory`" in text
        )
    else:
        assert (
            f"Try first:\nFor queueing before scheduling:\n- raising the cap is blocked{see}\n"
            in text
        )


def test_lower_concurrency_needs_a_cap_that_binds() -> None:
    recipe = next(r for r in PLAYBOOK if r.name == "lower-concurrency")
    facts = WorkloadFacts(("p",), 128, 4096)
    settings = Settings(max_concurrent_requests=64)
    idle = Measurement(10, 0, peak_running=25.0, peak_waiting=0.0, preemptions=0)
    assert recipe.applies(idle, facts, settings) is None  # 32 would never bind at a peak of 25
    thrashing = Measurement(10, 0, peak_running=25.0, peak_waiting=0.0, preemptions=5)
    assert recipe.applies(thrashing, facts, settings) is not None


def test_numeric_recipes_walk_their_setting_in_steps_of_at_most_four() -> None:
    from tensward.playbook import _ladder

    assert _ladder(8, 64) == [16, 32, 64]
    assert _ladder(8, 512) == [16, 64, 128, 512]  # six doublings, four spread evenly
    assert _ladder(2048, 8192) == [4096, 8192]
    assert _ladder(8192, 3072) == [4096, 3072]  # down, never past the target
    assert _ladder(0.85, 0.95) == [0.9, 0.95]

    seen = Measurement(10, 0, peak_running=8.0, peak_waiting=24.0, peak_kv_usage=0.06)
    current = Settings("float16", 8, 2048, 0.8, 2048, "auto", True)
    facts = WorkloadFacts(("p",), 128, 4096)
    raise_concurrency = next(r for r in PLAYBOOK if r.name == "raise-concurrency")
    steps = rungs(raise_concurrency, seen, facts, current)
    assert [s.max_concurrent_requests for s in steps] == [16, 32]  # the evidence says 32
    assert all(s.prefill_batch_tokens == 2048 for s in steps)
    assert rungs(raise_concurrency, seen, facts, current, prepare=lambda s: None) == []
    assert rungs(raise_concurrency, seen, facts, current, prepare=lambda s: current) == []


def test_analyse_without_a_current_setup_measures_the_engine_defaults(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The config declares no serving fields, so nothing is passed to the engine (the fake
    defaults to 256 sequences, room for 9 in its KV cache) and the recipes fall back to what
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
    assert "max_context_len is 4096" in report  # what the engine chose, as it reported it
    argv = json.loads((result.run_dir / "serve.json").read_text())["identity"]["argv"]
    assert not {"--max-num-seqs", "--max-model-len", "--gpu-memory-utilization"} & set(argv)


def test_trust_remote_code_can_only_come_from_the_registered_setup() -> None:
    engine = ENGINES["vllm"]
    with pytest.raises(AnalyseFailure, match="register the project again"):
        apply_engine_args(engine, Settings(), ["trust-remote-code"])
    trusted = engine.parse_setup("vllm serve /m --trust-remote-code").settings
    assert apply_engine_args(engine, trusted, ["max-num-seqs=4"]).max_concurrent_requests == 4
