"""The vLLM adapter's reading of a customer's command line."""

from __future__ import annotations

from pathlib import Path

import pytest

from tensward.engines.protocol import Settings
from tensward.engines.vllm import VLLM
from tensward.recommendations import WorkloadFacts
from tensward.report import checks


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
    signals = VLLM.parse_signals(text)

    assert (signals.running, signals.kv_usage, signals.prompt_tokens) == (3.0, 0.25, 1500.0)
    assert signals.waiting is None  # two series: ambiguous, so not read
    assert signals.preemptions is None  # absent, e.g. renamed by a newer release

    (check,) = checks(VLLM, text, WorkloadFacts((), 1, 1), Settings())
    assert (
        "waiting, preemptions, prefix_cache_hits, prefix_cache_queries, generation_tokens" in check
    )


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


def test_gemma4_gets_its_tool_parser(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model_type": "gemma4"}')
    assert VLLM.default_tool_parser(tmp_path) == "gemma4"


def test_in_flight_tokens_follow_vllms_two_batches() -> None:
    assert VLLM.kv_in_flight_tokens(Settings(prefill_batch_tokens=2496)) == 2 * 2496


FIXTURE = Path(__file__).parent / "vllm030_gemma4_metrics.txt"


def test_cache_config_info_labels_give_the_measured_capacity() -> None:
    signals = VLLM.parse_signals(FIXTURE.read_text())
    assert signals.kv_capacity_tokens == 40416
    assert signals.kv_max_concurrency == pytest.approx(1.2334135441732554)
    assert signals.generation_tokens is not None  # unlabelled samples still read


def test_labels_with_braces_and_quotes_parse() -> None:
    text = (
        'vllm:cache_config_info{note="a}b",quote="say \\"hi\\"",kv_cache_size_tokens="123"} 1.0\n'
        'vllm:num_requests_running{model_name="m"} 3.0\n'
    )
    signals = VLLM.parse_signals(text)
    assert signals.kv_capacity_tokens == 123 and signals.running == 3


def test_short_parallel_flags_count_as_parallelism() -> None:
    assert VLLM.parallel_degree(VLLM.parse_setup("vllm serve /m -tp 2 -pp 2").settings) == 4
