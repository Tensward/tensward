"""The vLLM adapter's reading of a customer's command line."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from pathlib import Path

import httpx
import pytest
from test_playbook import situation, suggest

from tensward.checks import checks
from tensward.classify import classify
from tensward.engines import ENGINES
from tensward.engines.vllm import VLLM
from tensward.engines.vllm.capabilities import speculative_config
from tensward.engines.vllm.command import SPECULATIVE_CONFIG_FLAG, _installed_version
from tensward.engines.vllm.predicates import NGRAM
from tensward.measure import queue_share, signal_increase
from tensward.measurement import Measurement
from tensward.playbook import WorkloadFacts
from tensward.prometheus import read_signals
from tensward.runtime import DockerRuntime, ServeSpec
from tensward.settings import QuantKernel, Settings
from tensward.signal_source import Scrape
from tensward.suggest import suggestion_command

NGRAM_CONFIG = speculative_config(NGRAM)


def test_flag_spellings_all_reach_the_same_settings() -> None:
    forms = [
        "vllm serve m --max-num-seqs 64 -q awq --no-enable-prefix-caching --max-model-len 8k",
        "vllm serve m --max_num_seqs=64 --quantization=awq --enable-prefix-caching=false "
        "--max_model_len 8k",
        "python3 -m vllm.entrypoints.openai.api_server --model m \\\n --max-num-seqs 64 "
        "--quantization awq --no-enable-prefix-caching --max-model-len=8000",
    ]
    parsed = [VLLM.parse_setup(form).settings for form in forms]

    assert parsed[0] == parsed[1] == parsed[2]
    assert (parsed[0].max_concurrent_requests, parsed[0].max_context_len) == (64, 8000)
    assert (parsed[0].quantization, parsed[0].prefix_caching) == ("awq", False)


def test_unknown_flags_are_kept_and_secrets_are_neither_kept_nor_shown() -> None:
    parsed = VLLM.parse_setup(
        "vllm serve m --api-key sk-1 --tensor-parallel-size 2 --enforce_eager --foo=bar"
    )

    assert dict(parsed.settings.extra_args) == {"--tensor-parallel-size": "2", "--foo": "bar"}
    assert parsed.settings.cuda_graphs is False
    assert "sk-1" not in parsed.text and "--api-key" in parsed.notes[0]


@pytest.mark.parametrize("text", ["vllm serve a b", "vllm serve m --config x.yaml", "ls"])
def test_commands_it_cannot_reproduce_are_refused(text: str) -> None:
    with pytest.raises(ValueError):
        VLLM.parse_setup(text)


def test_signals_are_read_and_a_renamed_metric_is_reported_missing() -> None:
    text = (
        "# HELP vllm:num_requests_running Running.\n"
        "# TYPE vllm:num_requests_running gauge\n"
        'vllm:num_requests_running{engine="0",model_name="m"} 3.0\n'
        'vllm:kv_cache_usage_perc{engine="0"} 0.25\n'
        'vllm:num_requests_waiting{engine="0",model_name="a"} 1.0\n'
        'vllm:num_requests_waiting{engine="0",model_name="b"} 2.0\n'
        "vllm:prompt_tokens_total 1.5e3\n"
    )
    signals = read_signals(text, VLLM.signals)

    assert (signals.running, signals.kv_usage, signals.prompt_tokens) == (3.0, 0.25, 1500.0)
    assert signals.waiting is None  # two series: ambiguous, so not read
    assert signals.preemptions is None  # absent, e.g. renamed by a newer release

    (check,) = checks(
        VLLM,
        signals,
        WorkloadFacts(prompts=(), output_tokens=1, context_limit=1),
        Settings(),
        None,
        "",
        "",
        "config",
    )
    assert (
        "waiting, preemptions, prefix_cache_hits, prefix_cache_queries, generation_tokens" in check
    )

    fixtures = Path(__file__).parent
    ngram = read_signals((fixtures / "vllm030_ngram_metrics.txt").read_text("utf-8"), VLLM.signals)
    assert (ngram.spec_drafts, ngram.spec_accepted_tokens) == (492.0, 950.0)
    assert ngram.queue_seconds == pytest.approx(105.25, abs=0.01)
    assert ngram.prefill_seconds == pytest.approx(39.27, abs=0.01)
    scrape = (fixtures / "vllm030_gemma4_metrics.txt").read_text("utf-8")
    gemma4 = read_signals(scrape, VLLM.signals)
    assert gemma4.spec_drafts is None  # recorded with speculation off
    found = checks(
        VLLM,
        gemma4,
        WorkloadFacts(prompts=(), output_tokens=1, context_limit=1),
        Settings(),
        None,
        "",
        "",
        "config",
    )
    assert not any("spec_" in line for line in found)


SECRET_COMMANDS = [
    "vllm serve /m --hf-token hf_SECRET --max-num-seqs 8",
    "vllm serve /m --hf_token=hf_SECRET --max-num-seqs 8",
    "docker run --env=HF_TOKEN=hf_SECRET img --model /m --max-num-seqs 8",
    "docker run -e HF_TOKEN=hf_SECRET img --model /m --max-num-seqs 8",
    "docker run --env HF_TOKEN=hf_SECRET img --model /m --max-num-seqs 8",
    "docker run -eHF_TOKEN=hf_SECRET img --model /m --max-num-seqs 8",
    "HF_TOKEN=hf_SECRET vllm serve /m --max-num-seqs 8",
    "VLLM_MY_TOKEN=hf_SECRET vllm serve /m --max-num-seqs 8",
    "vllm serve /m --api-key hf_SECRET other --max-num-seqs 8",
    "vllm serve /m --api-key=hf_SECRET --max-num-seqs 8",
]


@pytest.mark.parametrize("command", SECRET_COMMANDS)
def test_a_secret_is_in_no_stored_or_printed_form(command: str) -> None:
    parsed = VLLM.parse_setup(command)
    settings = parsed.settings
    stored = repr((parsed, settings.extra_args, settings.extra_env))
    argv = VLLM.launch_argv(settings, model="/m", served_model_name="n", host="h", port=1)

    assert "hf_SECRET" not in stored  # stored text, notes, settings
    assert "hf_SECRET" not in " ".join(argv)  # engine argv (and so serve.json identity)
    assert settings.max_concurrent_requests == 8  # the rest of the command is still understood
    assert "***" in parsed.text or "-e" in parsed.text


def test_secret_flags_are_dropped_with_a_note_and_not_taken_as_engine_args() -> None:
    parsed = VLLM.parse_setup("vllm serve /m --hf-token hf_SECRET")

    assert dict(parsed.settings.extra_args) == {}
    assert "HF_TOKEN" in parsed.notes[0] and "--hf-token" in parsed.notes[0]
    with pytest.raises(ValueError, match="secret"):
        VLLM.with_engine_arg(Settings(), "hf-token=hf_SECRET")


def test_names_that_only_contain_token_or_key_are_not_secrets() -> None:
    parsed = VLLM.parse_setup(
        "vllm serve /m --max-num-batched-tokens 4096 --tokenizer-mode auto --kv-cache-dtype fp8"
    )

    assert parsed.text.count("***") == 0
    assert parsed.settings.prefill_batch_tokens == 4096


@pytest.mark.parametrize(
    ("command", "gpus"),
    [
        ("docker run --gpus all img --model /m", ()),
        ("docker run --gpus 2 img --model /m", ()),
        ("docker run --gpus device=1 img --model /m", ("1",)),
        ("docker run --gpus '\"device=0,1\"' img --model /m", ("0", "1")),
        ("docker run -e CUDA_VISIBLE_DEVICES=2 img --model /m", ("2",)),
        ("CUDA_VISIBLE_DEVICES=3 vllm serve /m", ("3",)),
    ],
)
def test_the_gpus_a_current_command_selects_are_recorded_not_ignored(
    command: str, gpus: tuple[str, ...]
) -> None:
    parsed = VLLM.parse_setup(command)
    assert parsed.gpus == gpus
    assert not any("--gpus" in note or "CUDA_VISIBLE" in note for note in parsed.notes)


@pytest.mark.parametrize(
    "switch", ["-P", "--publish-all", "-q", "--no-healthcheck", "--oom-kill-disable"]
)
def test_docker_switches_without_a_value_do_not_swallow_the_image(switch: str) -> None:
    parsed = VLLM.parse_setup(f"docker run {switch} vllm/vllm-openai:v0.30.0 --model /m")
    assert parsed.image == "vllm/vllm-openai:v0.30.0"


@pytest.mark.parametrize(
    "given", ["--max-model-len auto", "--max-model-len -1", "--max-model-len=-1"]
)
def test_an_automatic_max_model_len_means_the_engine_default(given: str) -> None:
    parsed = VLLM.parse_setup(f"vllm serve /m {given}")
    assert parsed.settings.max_context_len is None
    assert VLLM.parse_setup("vllm serve /m --max-model-len 4k").settings.max_context_len == 4000


def test_a_bad_engine_arg_value_is_refused_without_naming_internals() -> None:
    with pytest.raises(ValueError, match=r"needs a context_length value") as error:
        VLLM.parse_setup("vllm serve /m --max-model-len lots")
    assert "_count" not in str(error.value)


def test_a_customer_media_switch_is_kept_and_round_trips() -> None:
    parsed = VLLM.parse_setup("vllm serve /m --language-model-only")
    assert parsed.settings.media_inputs is False
    argv = VLLM.launch_argv(parsed.settings, model="/m", served_model_name="m", host="h", port=1)
    assert "--language-model-only" in argv
    assert VLLM.engine_args_between(Settings(), parsed.settings) == ["language-model-only"]


def test_media_limits_round_trip() -> None:
    parsed = VLLM.parse_setup("""vllm serve /m --limit-mm-per-prompt '{"image": 2, "video": 0}'""")
    assert parsed.settings.media_limits == {"image": 2, "video": 0}
    argv = VLLM.launch_argv(parsed.settings, model="/m", served_model_name="m", host="h", port=1)
    assert json.loads(argv[argv.index("--limit-mm-per-prompt") + 1]) == {"image": 2, "video": 0}
    assert VLLM.with_engine_arg(Settings(), 'limit-mm-per-prompt={"image": 1}').media_limits == {
        "image": 1
    }


def test_gemma4_gets_its_tool_parser(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model_type": "gemma4"}')
    assert VLLM.default_tool_parser(tmp_path) == "gemma4"


def test_in_flight_tokens_follow_vllms_two_batches() -> None:
    settings = Settings(prefill_batch_tokens=2496)
    assert VLLM.kv_in_flight_tokens(settings, takes_images=False) == 2 * 2496


FIXTURE = Path(__file__).parent / "vllm030_gemma4_metrics.txt"


def test_cache_config_info_labels_give_the_measured_capacity() -> None:
    signals = read_signals(FIXTURE.read_text(), VLLM.signals)
    assert signals.kv_capacity_tokens == 40416
    assert signals.kv_max_concurrency == pytest.approx(1.2334135441732554)
    assert signals.generation_tokens is not None  # unlabelled samples still read


def test_labels_with_braces_and_quotes_parse() -> None:
    text = (
        'vllm:cache_config_info{note="a}b",quote="say \\"hi\\"",kv_cache_size_tokens="123"} 1.0\n'
        'vllm:num_requests_running{model_name="m"} 3.0\n'
    )
    signals = read_signals(text, VLLM.signals)
    assert signals.kv_capacity_tokens == 123 and signals.running == 3


def test_short_parallel_flags_count_as_parallelism() -> None:
    assert VLLM.parallel_degree(VLLM.parse_setup("vllm serve /m -tp 2 -pp 2").settings) == 4


@pytest.mark.parametrize(
    ("name", "expected"),
    [("python3", "0.30.0"), ("python3.12", "0.30.0"), ("python", "0.30.0"), ("pythonw", None)],
)
def test_a_python_launcher_is_probed_for_the_vllm_it_runs(
    name: str, expected: str | None, tmp_path: Path
) -> None:
    interpreter = tmp_path / name
    interpreter.write_text("#!/bin/sh\necho 0.30.0\n")
    interpreter.chmod(0o755)
    assert _installed_version(str(interpreter)) == expected


def test_the_server_log_names_the_version_that_ran() -> None:
    line = "INFO 10-05 [core.py:78] Initializing a V1 LLM engine (v0.30.0) with config: model='m'"
    assert re.search(VLLM.version_log, line)[1] == "0.30.0"


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
    facts = WorkloadFacts(prompts=("p",), output_tokens=128, context_limit=4096)
    forced = Settings(
        dtype="float16",
        max_concurrent_requests=8,
        max_context_len=2048,
        kv_memory_fraction=0.8,
        prefill_batch_tokens=2048,
        kv_cache_dtype="auto",
        prefix_caching=True,
        quantization="gptq",
    )
    ((entry, reason),) = suggest(measurement, facts, forced, allow_quality_changes=False)
    assert entry.name == "fast-quant-kernel" and "ExllamaLinearKernel" in reason
    assert entry.apply(situation(measurement, facts, forced)).quantization is None
    auto = dataclasses.replace(forced, quantization=None)
    assert suggest(measurement, facts, auto, allow_quality_changes=False) == []


def test_every_entry_change_round_trips_through_engine_args() -> None:
    engine = ENGINES["vllm"]
    before = Settings(
        dtype="float16",
        max_concurrent_requests=8,
        max_context_len=2048,
        kv_memory_fraction=0.8,
        prefill_batch_tokens=2048,
        kv_cache_dtype="auto",
        prefix_caching=True,
        quantization="gptq",
    )
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

    def speculating(settings: Settings, config: str = NGRAM_CONFIG) -> Settings:
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
        settings=Settings(
            dtype="bfloat16",
            max_concurrent_requests=8,
            max_context_len=2048,
            kv_memory_fraction=0.8,
            prefill_batch_tokens=2048,
            kv_cache_dtype="auto",
            prefix_caching=True,
            extra_args={"--x": "1"},
        ),
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


def two_engines(scale: float) -> str:
    """vLLM metrics from two data-parallel engines carrying unequal load, with the API server's
    CPU counter (no engine label) once and the cache info gauge on each engine."""
    lines = [f"process_cpu_seconds_total {10.0 * scale}"]
    for engine, share in (("0", 0.3), ("1", 0.7)):
        label = f'engine="{engine}",model_name="m"'
        counters = {
            "vllm:num_preemptions_total": 2, "vllm:prefix_cache_hits_total": 100,
            "vllm:prefix_cache_queries_total": 400, "vllm:prompt_tokens_total": 4000,
            "vllm:generation_tokens_total": 1000, "vllm:iteration_tokens_total_count": 500,
            "vllm:request_queue_time_seconds_sum": 90, "vllm:request_prefill_time_seconds_sum": 10,
        }  # fmt: skip
        lines += [f"{name}{{{label}}} {value * share * scale}" for name, value in counters.items()]
        by_source = "vllm:prompt_tokens_by_source_total"
        lines += [
            f'{by_source}{{{label},source="local_compute"}} {3000 * share * scale}',
            f'{by_source}{{{label},source="local_cache_hit"}} {1000 * share * scale}',
            f"vllm:num_requests_running{{{label}}} {16 * share}",
            f"vllm:num_requests_waiting{{{label}}} {8 * share}",
            f"vllm:kv_cache_usage_perc{{{label}}} {share}",
            f'vllm:cache_config_info{{{label},num_gpu_blocks="1000",block_size="16"}} 1.0',
        ]
    return "\n".join(lines) + "\n"


def test_data_parallel_engines_are_summed_only_where_a_sum_means_something() -> None:
    async def scrape(text: str) -> Scrape | None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text=text))
        async with httpx.AsyncClient(transport=transport) as client:
            return await VLLM.signal_source.read(client, "http://engine")

    before, after = asyncio.run(scrape(two_engines(1.0))), asyncio.run(scrape(two_engines(3.0)))
    assert before is not None and after is not None and after.replicas == 2
    signals = after.signals
    assert (signals.running, signals.waiting, signals.kv_usage, signals.kv_blocks) == (None,) * 4
    assert signals.generation_tokens == pytest.approx(3000.0)
    assert signals.prompt_tokens_computed == pytest.approx(9000.0)
    assert signals.frontend_cpu_seconds == pytest.approx(30.0)  # unlabelled: read once
    facts = WorkloadFacts(prompts=(), output_tokens=1, context_limit=1)
    assert checks(VLLM, signals, facts, Settings(), None, "", "", "config", replicas=2) == []

    shared = queue_share(signal_increase(before.signals, after.signals))
    assert shared == pytest.approx(0.9)
    measured = Measurement(
        100, 0, ttft_p50_ms=900.0, tpot_p50_ms=20.0, tpot_p95_ms=24.0, peak_in_flight=32,
        queue_share=shared, replicas=after.replicas,
    )  # fmt: skip
    diagnosis = classify(measured, Settings(max_concurrent_requests=8), None)
    found = {f.bottleneck: f for f in diagnosis.findings}
    for name in (
        "queueing", "kv_capacity", "prefill_stalls_decode", "decode_bandwidth", "long_context",
        "frontend_cpu",
    ):  # fmt: skip
        assert (found[name].state, found[name].evidence) == (
            "cant_tell", "2 data-parallel engines; per-engine limits are not modelled"
        ), name  # fmt: skip
    assert (found["prefill"].state, found["prefill"].evidence) == (
        "clear", "TTFT is mostly time spent queued"
    )  # fmt: skip

    hybrid = dataclasses.replace(
        measured, hybrid_cache=True, kv_blocks=10.0, peak_kv_usage=8 / 9, peak_running=2.0,
        peak_waiting=30.0,
    )  # fmt: skip
    ruled = {f.bottleneck: f for f in classify(hybrid, Settings(), None).findings}
    assert ruled["kv_capacity"].state == "cant_tell"  # no engine rule's crossing for a class absent

    assert VLLM.frontend_processes(Settings(extra_args={"-dp": "2"})) == 2
    assert VLLM.frontend_processes(Settings(extra_args={"--data-parallel-size": "4"})) == 4
    assert VLLM.frontend_processes(Settings(api_server_count=1, extra_args={"-dp": "2"})) == 1


# vLLM f0c44cc fused_moe/oracle/*.py and quantization/utils/humming/moe.py (v0.30.0:
# quantization/utils/humming_utils.py).
MOE_LINES = [
    (
        "Using TRITON Fp8 MoE backend out of potential backends: ['TRITON', 'MARLIN'].",
        "TRITON",
        "Fp8 MoE",
    ),
    (
        "Using FlashInfer TRTLLM Unquantized MoE backend out of potential backends: "
        "['FlashInfer TRTLLM'].",
        "FlashInfer TRTLLM",
        "Unquantized MoE",
    ),
    (
        "Using FlashInfer CUTLASS Unquantized MoE backend out of potential backends: "
        "['FlashInfer CUTLASS'].",
        "FlashInfer CUTLASS",
        "Unquantized MoE",
    ),
    (
        "Using ROCm AITER Unquantized MoE backend out of potential backends: ['ROCm AITER'].",
        "ROCm AITER",
        "Unquantized MoE",
    ),
    (
        "Using TRITON Unquantized MoE backend out of potential backends: ['TRITON'].",
        "TRITON",
        "Unquantized MoE",
    ),
    (
        "Using TritonExperts MoE backend",
        "TritonExperts",
        "MoE",
    ),
    (
        "Using CUTLASS W4A8 MoE backend.",
        "CUTLASS",
        "W4A8 MoE",
    ),
    (
        "Using CUTLASS W4A8 Int8 MoE backend out of potential backends: ['CUTLASS'].",
        "CUTLASS",
        "W4A8 Int8 MoE",
    ),
    (
        "Using TRITON Int8 MoE backend out of potential backends: ['TRITON'].",
        "TRITON",
        "Int8 MoE",
    ),
    (
        "Using 'MARLIN' WNA16 MoE backend.",
        "MARLIN",
        "WNA16 MoE",
    ),
    (
        "Using 'TRITON' Mxfp4 MoE backend.",
        "TRITON",
        "Mxfp4 MoE",
    ),
    (
        "Using 'FLASHINFER_TRTLLM' MxFp8 MoE backend (user-requested).",
        "FLASHINFER_TRTLLM",
        "MxFp8 MoE",
    ),
    (
        "Using 'FLASHINFER_TRTLLM' MxFp8 MoE backend.",
        "FLASHINFER_TRTLLM",
        "MxFp8 MoE",
    ),
    (
        "Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend out of potential backends: "
        "['FLASHINFER_CUTLASS'].",
        "FLASHINFER_CUTLASS",
        "NvFp4 MoE",
    ),
    (
        "Using HummingExperts Humming MoE backend.",
        "HummingExperts",
        "Humming MoE",
    ),
    (
        "Using FlashInfer for top-p & top-k sampling. MoE backend chosen later",
        None,
        "",
    ),
]


@pytest.mark.parametrize(("line", "backend", "layer"), MOE_LINES)
def test_every_moe_backend_line_vllm_writes_is_read(
    line: str, backend: str | None, layer: str
) -> None:
    expected = () if backend is None else (QuantKernel(backend, layer, slow=False),)
    assert ENGINES["vllm"].parse_quant_kernels(f"(EngineCore pid=1) INFO {line}\n") == expected


def test_a_moe_backend_is_never_read_across_log_lines() -> None:
    log = (
        "(EngineCore pid=1) INFO 10-07 [cuda.py:300] Using FLASH_ATTN attention backend.\n"
        "(EngineCore pid=1) INFO 10-07 [fp8.py:415] Using FlashInfer CUTLASS Fp8 MoE backend out "
        "of potential backends: ['FlashInfer CUTLASS', 'TRITON'].\n"
    )
    expected = QuantKernel("FlashInfer CUTLASS", "Fp8 MoE", slow=False)
    assert ENGINES["vllm"].parse_quant_kernels(log) == (expected,)
