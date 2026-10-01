"""``tensward init`` and ``tensward inspect`` through the real entry point.

Inputs are tiny synthetic files from ``registration_fixtures``; nothing loads a model.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
from pathlib import Path

import pytest
from registration_fixtures import (
    QUANTIZED_CHECKPOINTS,
    REGISTRATION_CONFIG,
    make_registration_inputs,
    safetensors_bytes,
    write_config,
)

from tensward.cli import main

MARKER = "synthetic-malicious-marker-value"


def run(capsys: pytest.CaptureFixture[str], *arguments: str) -> tuple[int, str, str]:
    code = main(list(arguments))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def init(capsys, project: Path, model: Path, config: Path, prompts: Path, *extra: str):
    return run(
        capsys, "init", "--project", str(project), "--model", str(model),
        "--config", str(config), "--prompts", str(prompts), *extra,
    )  # fmt: skip


def refusal(err: str) -> str:
    """Return the fixed code of the single JSON line a refusal prints."""
    payload = json.loads(err)
    assert set(payload) == {"code", "message"}
    return payload["code"]


def test_init_then_inspect_report_one_sanitized_private_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"

    code, out, _ = init(capsys, project, model, config, prompts)
    assert code == 0
    summary = json.loads(out)
    assert summary["registration_state"] == "registered"
    assert str(tmp_path) not in out and "synthetic private prompt" not in out
    assert stat.S_IMODE(project.stat().st_mode) == 0o700
    document = project / "project.json"
    assert stat.S_IMODE(document.stat().st_mode) == 0o600

    before = document.read_bytes()
    assert run(capsys, "inspect", "--project", str(project))[1] == out
    assert init(capsys, project, model, config, prompts)[1] == out  # identical retry
    assert document.read_bytes() == before
    assert not {"retain_responses", "input_retention", "runtime_validation"} & set(summary)


def test_a_record_with_the_retired_policy_fields_still_loads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    out = init(capsys, project, model, config, prompts)[1]
    document = project / "project.json"
    record = json.loads(document.read_text())
    record |= {
        "input_retention": "reference_only",
        "retain_responses": False,
        "resource_policy_state": "unconfigured",
        "execution_authorization": "none",
    }
    document.write_text(json.dumps(record))
    assert run(capsys, "inspect", "--project", str(project))[1] == out


@pytest.mark.parametrize(
    ("current", "image", "settings"),
    [
        (  # an unknown flag is kept verbatim; the flags Tensward owns are dropped
            "vllm serve {model} --max_num_seqs=4 --swap-space 8 --host 0.0.0.0 --port 9",
            None,
            {"max_concurrent_requests": 4, "extra_args": {"--swap-space": "8"}},
        ),
        (  # docker flags are ignored, the image is recorded, engine env is kept
            "docker run --gpus all -p 8000:8000 -v {model}:/model -e VLLM_ATTENTION_BACKEND=X "
            "-e HF_TOKEN=secret vllm/vllm-openai:v0.30.0 --model /model --enable-prefix-caching",
            "vllm/vllm-openai:v0.30.0",
            {"prefix_caching": True, "extra_env": {"VLLM_ATTENTION_BACKEND": "X"}},
        ),
    ],
)
def test_init_imports_the_current_setup_as_the_baseline(
    current: str, image: str | None, settings: dict, tmp_path: Path, capsys
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"

    code, out, _ = init(capsys, project, model, config, prompts, "--current",
                        current.format(model=model))  # fmt: skip

    assert code == 0
    setup = json.loads(out)["current_setup"]
    assert setup["source"] == "imported from --current" and setup["image"] == image
    assert settings.items() <= setup["settings"].items()
    assert "secret" not in out and setup["dropped_or_ignored"]
    assert run(capsys, "inspect", "--project", str(project))[1] == out
    changed = init(capsys, project, model, config, prompts, "--current", f"vllm serve {model}")
    assert refusal(changed[2]) == "project_inputs_changed"  # the current setup is identity


def test_init_registers_an_unquantized_fp16_checkpoint_and_a_minimal_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, _, prompts = make_registration_inputs(tmp_path)
    edit_json(model / "config.json", torch_dtype="float16")
    (model / "model.safetensors").write_bytes(safetensors_bytes(dtype="F16"))
    workload = {k: v for k, v in REGISTRATION_CONFIG["workload"].items() if k != "mode"}
    minimal = {
        "schema_version": "1",
        "case": {"weight_precision": "fp16", "activation_dtype": "float16"},
        "workload": workload,
        "warmup_requests": 0,
    }

    config = write_config(tmp_path / "min.json", minimal)

    code, out, _ = init(capsys, tmp_path / "project", model, config, prompts)

    assert code == 0
    assert json.loads(out)["weights"]["weight_precision"] == "fp16"


def test_init_accepts_any_model_path_in_the_current_command_and_notes_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)

    code, out, _ = init(capsys, tmp_path / "project", model, config, prompts,
                        "--current", "vllm serve /mnt/other/Host-Path")  # fmt: skip

    assert code == 0
    notes = json.loads(out)["current_setup"]["dropped_or_ignored"]
    assert any("Tensward measures the registered checkpoint" in note for note in notes)


def test_init_without_a_current_setup_uses_the_config_serving_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)

    _, out, _ = init(capsys, tmp_path / "project", model, config, prompts)

    setup = json.loads(out)["current_setup"]
    assert setup["source"] == "declared in config" and setup["command"] is None
    assert setup["settings"]["max_concurrent_requests"] == 1


def test_init_with_only_checkpoint_and_tool_fields_declared_runs_the_engine_defaults(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """weight_precision, activation_dtype and tool_calling are not serving behaviour."""
    model, _, prompts = make_registration_inputs(tmp_path)
    case = {"weight_precision": "bf16", "activation_dtype": "bfloat16", "tool_calling": True}
    config = write_config(tmp_path / "bare.json", {**REGISTRATION_CONFIG, "case": case})

    _, out, _ = init(capsys, tmp_path / "project", model, config, prompts)

    setup = json.loads(out)["current_setup"]
    assert setup["source"] == "engine defaults (no current setup provided)"
    assert setup["settings"]["tool_calling"] is True


@pytest.mark.parametrize(
    "current",
    [
        "vllm serve {model} -q awq",
        "services:\n  vllm:\n    image: x",
        "make serve",
    ],
)
def test_init_refuses_a_current_setup_it_cannot_reproduce(
    current: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    file = tmp_path / "current.txt"
    file.write_text(current.format(model=model), encoding="utf-8")

    code, out, err = init(capsys, tmp_path / "project", model, config, prompts,
                          "--current-file", str(file))  # fmt: skip

    assert code != 0 and out == ""
    assert refusal(err) in ("project_config_unsupported", "project_inputs_invalid")
    assert not (tmp_path / "project" / "project.json").exists()


@pytest.mark.parametrize("fmt", QUANTIZED_CHECKPOINTS)
def test_init_registers_quantized_checkpoints_and_inspect_reports_them(
    fmt: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path, fmt)
    project = tmp_path / "project"

    code, out, _ = init(capsys, project, model, config, prompts)

    assert code == 0
    assert json.loads(out)["weights"] == QUANTIZED_CHECKPOINTS[fmt][3]
    assert run(capsys, "inspect", "--project", str(project))[1] == out


def test_a_quantized_checkpoint_needs_a_matching_serving_precision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, _, prompts = make_registration_inputs(tmp_path, "awq")
    bf16_config = write_config(tmp_path / "bf16.json")

    code, _, err = init(capsys, tmp_path / "project", model, bf16_config, prompts)

    assert code != 0 and refusal(err) == "project_config_unsupported"


@pytest.mark.parametrize(
    "quantization_config, expected",
    [
        ({"quant_method": "bitsandbytes", "load_in_4bit": True}, "'bitsandbytes'"),
        ({"quant_method": "awq", "bits": 8, "group_size": 128}, "8 bits"),
    ],
)
def test_init_refuses_quantization_it_cannot_serve_and_names_it(
    quantization_config: dict, expected: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path, "awq")
    document = json.loads((model / "config.json").read_text())
    document["quantization_config"] = quantization_config
    (model / "config.json").write_text(json.dumps(document))

    code, _, err = init(capsys, tmp_path / "project", model, config, prompts)

    assert code != 0 and expected in json.loads(err)["message"]


@pytest.mark.parametrize(
    "mistake, expected",
    [
        ("missing_tokenizer", "checkpoint_layout_invalid"),
        ("missing_weights", "checkpoint_layout_invalid"),
        ("missing_model_config", "checkpoint_layout_invalid"),
        ("missing_serving_config", "project_inputs_invalid"),
        ("prompts_not_json", "project_inputs_invalid"),
        ("prompts_duplicate_ids", "project_inputs_invalid"),
        ("prompts_empty", "project_inputs_invalid"),
    ],
)
def test_init_refuses_the_common_registration_mistakes(
    mistake: str, expected: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    if mistake == "missing_tokenizer":
        (model / "tokenizer.json").unlink()
    elif mistake == "missing_weights":
        (model / "model.safetensors").unlink()
    elif mistake == "missing_model_config":
        (model / "config.json").unlink()
    elif mistake == "missing_serving_config":
        config = tmp_path / "absent.json"
    elif mistake == "prompts_not_json":
        prompts.write_text("this is not json\n", encoding="utf-8")
    elif mistake == "prompts_duplicate_ids":
        row = json.dumps({"id": "same", "prompt": "hello"})
        prompts.write_text(f"{row}\n{row}\n", encoding="utf-8")
    else:
        prompts.write_text("", encoding="utf-8")

    code, out, err = init(capsys, project, model, config, prompts)

    assert code != 0 and out == ""
    assert refusal(err) == expected
    assert not (project / "project.json").exists()
    if mistake == "missing_serving_config":
        message = json.loads(err)["message"]
        assert f"not found at {config}" in message and "mount it at the same path" in message


def test_init_refuses_a_project_inside_the_model_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)

    code, _, err = init(capsys, model / "state", model, config, prompts)

    assert code != 0 and refusal(err) == "project_layout_invalid"


def test_a_refusal_never_echoes_declared_values_or_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    write_config(config, {**REGISTRATION_CONFIG, "injected_key": MARKER})
    prompts.write_text(json.dumps({"id": "first", "prompt": MARKER}) + "\n", encoding="utf-8")

    code, out, err = init(capsys, tmp_path / "project", model, config, prompts)

    assert code != 0
    assert MARKER not in out + err and str(tmp_path) not in out + err


def test_inspect_refuses_an_unknown_project(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, err = run(capsys, "inspect", "--project", str(tmp_path / "absent"))

    assert code != 0 and refusal(err) == "project_not_registered"


def test_inspect_notices_changed_inputs_and_a_tampered_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    assert init(capsys, project, model, config, prompts)[0] == 0
    document = project / "project.json"
    original = document.read_bytes()

    (model / "extra-input.json").write_text("{}", encoding="utf-8")
    _, _, err = run(capsys, "inspect", "--project", str(project))
    assert refusal(err) == "checkpoint_inventory_unexpected"
    (model / "extra-input.json").unlink()

    payload = json.loads(original)
    payload["snapshot_id"] = "0" * 64
    document.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    _, _, err = run(capsys, "inspect", "--project", str(project))
    assert refusal(err) == "project_record_invalid"


def test_a_missing_subcommand_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main([])
    assert raised.value.code == 2
    assert "usage" in capsys.readouterr().err


def test_optimize_without_the_optimizer_package_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("tensward.cli.entry_points", lambda group: [])
    code, _, err = run(capsys, "optimize", "--project", "p", "--objective", "latency")
    assert code == 1 and "tensward-optimize package" in err


def test_identities_are_stable_across_releases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pinned values: a change here would make every project registered before it read as
    changed. Only change them on purpose, with a new schema version."""
    model, config, prompts = make_registration_inputs(tmp_path)

    summary = json.loads(init(capsys, tmp_path / "project", model, config, prompts)[1])

    assert summary["artifact_fingerprint"] == (
        "904fee176b668aa295bf676f605de2bc64877526508b65d3c07ff3f10136f168"
    )
    assert summary["config_digest"] == (
        "6dbc9a15d52069f5a9bce55d13ff8d9ac6c7bec63f3227044e8a9410f43fc842"
    )
    assert summary["workload_digest"] == (
        "07cb75fa229bc2e31d75108308937a1b3f706cfb74c3b36cbba7985309fc4dfe"
    )
    assert summary["snapshot_id"] == (
        "e70a9e28fa4a80bab612b7785fca4bba4fa05431105a0c187d95cb26514162e2"
    )


def edit_json(path: Path, **changes: object) -> None:
    document = json.loads(path.read_text())
    path.write_text(json.dumps({**document, **changes}))


def write_shards(model: Path, index: dict[str, str]) -> None:
    (model / "model.safetensors").unlink()
    for shard in set(index.values()):
        (model / shard).write_bytes(safetensors_bytes())
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))


SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("auto_map", "checkpoint_unsupported"),
        ("trust_remote_code", "checkpoint_unsupported"),
        ("vision_config", "checkpoint_unsupported"),
        ("tokenizer_file_outside", "checkpoint_unsupported"),
        ("fp16_tensor", "checkpoint_precision_unsupported"),
        ("truncated_weights", "checkpoint_layout_invalid"),
        ("unindexed_shard", "checkpoint_layout_invalid"),
        ("symlinked_input", "checkpoint_inventory_unsafe"),
    ],
)
def test_init_refuses_checkpoints_it_cannot_identify_safely(
    mutation: str, expected: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    if mutation == "auto_map":
        edit_json(model / "config.json", auto_map={"AutoModel": "remote.Model"})
    elif mutation == "trust_remote_code":
        edit_json(model / "config.json", nested={"trust_remote_code": True})
    elif mutation == "vision_config":
        edit_json(model / "config.json", vision_config={})
    elif mutation == "tokenizer_file_outside":
        edit_json(model / "tokenizer_config.json", vocab_file="/etc/passwd")
    elif mutation == "fp16_tensor":
        (model / "model.safetensors").write_bytes(safetensors_bytes(dtype="F16"))
    elif mutation == "truncated_weights":
        (model / "model.safetensors").write_bytes(safetensors_bytes()[:-1])
    elif mutation == "unindexed_shard":
        write_shards(model, {"a": SHARDS[0], "b": SHARDS[1]})
        (model / "model-00003-of-00002.safetensors").write_bytes(safetensors_bytes())
    else:
        outside = tmp_path / "outside.json"
        outside.write_text("{}")
        (model / "vocab.json").symlink_to(outside)

    code, out, err = init(capsys, tmp_path / "project", model, config, prompts)

    assert code == 3 and out == "" and refusal(err) == expected
    assert not (tmp_path / "project" / "project.json").exists()


def test_init_registers_a_sharded_checkpoint_and_tokens_that_look_like_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    write_shards(model, {"a": SHARDS[0], "b": SHARDS[1]})
    tokenizer = json.loads((model / "tokenizer.json").read_text())
    tokenizer["model"]["vocab"] = {"_file": 0, "auto_map": 1, "tokenizer_file": 2}
    (model / "tokenizer.json").write_text(json.dumps(tokenizer))
    project = tmp_path / "project"

    code, out, _ = init(capsys, project, model, config, prompts)

    assert code == 0 and run(capsys, "inspect", "--project", str(project))[1] == out


def test_a_checkpoint_edited_after_registration_is_noticed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    assert init(capsys, project, model, config, prompts)[0] == 0

    (model / "tokenizer_config.json").write_text(json.dumps({"chat_template": "changed"}))

    assert refusal(run(capsys, "inspect", "--project", str(project))[2]) == "project_inputs_changed"


def test_init_refuses_a_project_directory_others_can_read_and_leaves_it_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    project.chmod(0o755)

    code, _, err = init(capsys, project, model, config, prompts)

    assert code == 2 and refusal(err) == "project_state_unsafe"
    assert stat.S_IMODE(project.stat().st_mode) == 0o755 and not list(project.iterdir())


def test_root_may_use_a_project_another_user_owns_but_other_users_may_not(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    assert init(capsys, project, model, config, prompts)[0] == 0

    monkeypatch.setattr("os.geteuid", lambda: os.stat(project).st_uid + 1)
    assert refusal(run(capsys, "inspect", "--project", str(project))[2]) == "project_state_unsafe"
    monkeypatch.setattr("os.geteuid", lambda: 0)
    assert run(capsys, "inspect", "--project", str(project))[0] == 0


def test_init_refuses_while_another_registration_holds_the_project(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    project.mkdir(mode=0o700)
    with (project / ".registration.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)

        code, _, err = init(capsys, project, model, config, prompts)

    assert code == 2 and refusal(err) == "project_busy"
    assert not (project / "project.json").exists()


def test_the_shipped_examples_register(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """examples/config.json and examples/prompts.jsonl are what the README tells people to use."""
    examples = next(
        p / "examples" for p in Path(__file__).resolve().parents if (p / "examples").is_dir()
    )
    model, _, _ = make_registration_inputs(tmp_path, quantized="awq")
    document = json.loads((model / "config.json").read_text())
    (model / "config.json").write_text(json.dumps(document | {"max_position_embeddings": 32768}))
    command = "vllm serve /models/model --dtype float16 --max-model-len 8192 --max-num-seqs 8"
    code, out, err = init(
        capsys, tmp_path / "project", model, examples / "config.json",
        examples / "prompts.jsonl", "--current", command,
    )  # fmt: skip
    assert code == 0, err
    assert json.loads(out)["registration_state"] == "registered"


def test_docker_runtime_defaults_to_the_image_the_current_command_ran(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import argparse

    from tensward.cli import runtime_for

    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"

    def image(chosen: str | None = None) -> str:
        arguments = argparse.Namespace(
            engine="vllm", runtime="docker", image=chosen, project=project, gpus=None
        )
        return str(runtime_for(arguments).image)  # type: ignore[attr-defined]

    default = "vllm/vllm-openai:v0.30.0"
    assert image() == default  # nothing registered yet
    current = f"docker run -v {model}:/m vllm/vllm-openai:v0.29.1 --model /m"
    init(capsys, project, model, config, prompts, "--current", current)
    assert image() == "vllm/vllm-openai:v0.29.1"
    assert image("mine:1") == "mine:1"  # --image still overrides


@pytest.fixture
def hashing(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """One entry per full hash of the checkpoint's files."""
    import hashlib
    import types

    from tensward import artifacts

    calls: list[int] = []

    def counting_sha256() -> object:
        calls.append(1)
        return hashlib.sha256()

    monkeypatch.setattr(artifacts, "hashlib", types.SimpleNamespace(sha256=counting_sha256))
    return calls


def test_weights_are_hashed_once_then_only_when_a_file_changes_or_is_forced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], hashing: list[int]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    inspect = ("inspect", "--project", str(project))
    assert init(capsys, project, model, config, prompts)[0] == 0
    assert len(hashing) == 1

    assert run(capsys, *inspect)[0] == 0 and run(capsys, *inspect)[0] == 0
    assert len(hashing) == 1  # unchanged files: not hashed again

    weights = model / "model.safetensors"
    original, stamp = weights.read_bytes(), weights.stat()
    weights.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))  # same size, new mtime
    assert refusal(run(capsys, *inspect)[2]) == "project_inputs_changed"
    assert len(hashing) == 2  # a changed mtime is noticed by hashing

    weights.write_bytes(original)
    os.utime(weights, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert run(capsys, *inspect)[0] == 0  # restored: hashed again, now matches
    tampered = original[:-1] + bytes([original[-1] ^ 1])
    weights.write_bytes(tampered)
    os.utime(weights, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    count = len(hashing)
    assert run(capsys, *inspect)[0] == 0 and len(hashing) == count  # the cache's documented limit

    assert refusal(run(capsys, *inspect, "--verify-weights")[2]) == "project_inputs_changed"
    assert len(hashing) == count + 1


def test_slow_hashing_says_so_in_every_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from tensward import artifacts

    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    monkeypatch.setattr(artifacts, "SLOW_HASH_S", 0.05)
    real = artifacts.read_chunks
    monkeypatch.setattr(artifacts, "read_chunks", lambda *args: (time.sleep(0.2), real(*args))[1])

    for code, _, err in (
        init(capsys, project, model, config, prompts),
        run(capsys, "inspect", "--project", str(project), "--verify-weights"),
    ):
        assert code == 0 and "hashing the model weights" in err


def test_an_io_error_during_registration_names_the_cause_and_the_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno

    def fail(*args, **kwargs):
        raise OSError(errno.EACCES, os.strerror(errno.EACCES), "/srv/project/project.json")

    monkeypatch.setattr("tensward.cli.load_project", fail)

    code, _, err = run(capsys, "inspect", "--project", str(tmp_path))

    payload = json.loads(err)
    assert code == 2 and payload["code"] == "runner_failure"
    assert payload["message"] == "I/O error: Permission denied: /srv/project/project.json"


def test_handing_back_to_the_owner_never_follows_a_link_out_of_the_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensward.project import hand_back_to_owner

    project, outside = tmp_path / "project", tmp_path / "outside"
    (project / "runs").mkdir(parents=True)
    outside.mkdir()
    (project / "runs" / "report.md").write_text("x")
    (project / "link").symlink_to(outside)  # another user swapped a directory for a link
    changed = []
    real_stat = os.stat

    class RootOwned:
        """Everything inside the project looks root-owned, as after a `sudo` run."""

        def __init__(self, info: os.stat_result) -> None:
            self.st_uid = 0
            self.__dict__.update(
                {k: getattr(info, k) for k in dir(info) if k.startswith("st_")} | {"st_uid": 0}
            )

    def fake_stat(path, **kw):
        info = real_stat(path, **kw)
        return RootOwned(info) if kw.get("dir_fd") is not None else info

    monkeypatch.setattr("os.stat", fake_stat)
    monkeypatch.setattr("os.geteuid", lambda: 0)
    monkeypatch.setattr("os.chown", lambda path, uid, gid, **kw: changed.append((path, kw)))

    hand_back_to_owner(project)

    names = {name for name, _ in changed}
    assert names == {"runs", "link", "report.md"}  # the link itself, never what it points to
    assert all(kw["dir_fd"] is not None and kw["follow_symlinks"] is False for _, kw in changed)
    assert not list(outside.iterdir())
