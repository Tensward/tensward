"""``tensward init`` and ``tensward inspect`` through the real entry point.

Inputs are tiny synthetic files from ``registration_fixtures``; nothing loads a model.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
import sys
from pathlib import Path

import pytest
from image_fixtures import webp
from registration_fixtures import (
    QUANTIZED_CHECKPOINTS,
    REGISTRATION_CONFIG,
    SHARDS,
    make_gemma4_checkpoint,
    make_image_workload,
    make_registration_inputs,
    safetensors_bytes,
    write_config,
    write_shards,
)

from tensward.cli import main
from tensward.engines import ENGINES
from tensward.engines.vllm import VllmEngine
from tensward.files import write_json
from tensward.inputs import PromptEntry, workload_digest
from tensward.platforms import Device as GpuInfo


class OtherEngine(VllmEngine):
    """A second engine for resolution: launched by ``other-server``."""

    name = "other"
    label = "Other"
    launchers = "`other-server ...`"

    def recognizes(self, command: str) -> bool:
        return command.startswith("other-server")


MARKER = "synthetic-malicious-marker-value"
SPLIT = {SHARDS[0]: {"a": ("BF16", (1,))}, SHARDS[1]: {"b": ("BF16", (1,))}}


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


def test_a_gptqmodel_checkpoint_and_its_production_command_register_as_given(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path, "gptq")
    (model / "quant_log.csv").write_text("layer,loss\n0,0.01\n")
    edit_json(model / "config.json", quantization_config={
        **json.loads((model / "config.json").read_text())["quantization_config"],
        "meta": {"quantizer": ["gptqmodel:4.0"], "offload_to_disk_path": "/tmp/gptq_offload"},
    })  # fmt: skip
    command = (
        "vllm serve /m --quantization auto_gptq --long-prefill-token-threshold 0 "
        "--hf-token hf_SECRET"
    )

    code, out, err = init(capsys, tmp_path / "project", model, config, prompts,
                          "--current", command)  # fmt: skip

    assert code == 0, err
    setup = json.loads(out)["current_setup"]["settings"]
    assert setup["extra_args"]["--long-prefill-token-threshold"] == "0"
    assert "hf_SECRET" not in out
    wrong = init(capsys, tmp_path / "other", model, config, prompts,
                 "--current", "vllm serve /m --quantization awq")  # fmt: skip
    assert refusal(wrong[2]) == "project_config_unsupported"
    assert "awq" in wrong[2] and "gptq" in wrong[2]


@pytest.mark.parametrize(
    "arrival",
    [
        {"kind": "closed_loop", "concurrency": 1},
        {"kind": "open_loop", "rate_rps": 2.0},
        {"kind": "capped", "rate_rps": 2.0, "max_inflight": 4},
    ],
)
def test_init_registers_an_unquantized_fp16_checkpoint_and_a_minimal_config(
    arrival: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, _, prompts = make_registration_inputs(tmp_path)
    edit_json(model / "config.json", torch_dtype="float16")
    (model / "model.safetensors").write_bytes(safetensors_bytes(dtype="F16"))
    workload = {k: v for k, v in REGISTRATION_CONFIG["workload"].items() if k != "mode"}
    workload["arrival"] = arrival
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
    assert "the checkpoint provides int4 with float16" in json.loads(err)["message"]


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


def test_init_registers_an_image_text_moe_checkpoint_and_reports_its_anatomy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path, "compressed-tensors-int4")
    make_gemma4_checkpoint(model)
    project = tmp_path / "project"

    code, out, err = init(capsys, project, model, config, prompts)

    assert code == 0, err
    anatomy = json.loads(out)["anatomy"]
    assert anatomy["model_type"] == "gemma4" and anatomy["modalities"] == ["image", "text", "video"]
    assert anatomy["moe"]["experts"] == 2 and anatomy["moe"]["experts_per_token"] == 1
    assert anatomy["max_tokens_per_image"] == 280
    k_eq_v = {g["kind"]: g["k_eq_v"] for g in anatomy["attention"]}
    assert k_eq_v == {"sliding": False, "full": True}
    assert run(capsys, "inspect", "--project", str(project))[1] == out


def test_init_reports_memory_other_processes_use_without_failing_the_fit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    busy = (GpuInfo("NVIDIA L4", 23034 * 2**20, 20 * 2**30),)
    monkeypatch.setattr("tensward.project.detect_devices", lambda: busy)
    model, config, prompts = make_registration_inputs(tmp_path, "compressed-tensors-int4")
    make_gemma4_checkpoint(model)

    code, out, _ = init(capsys, tmp_path / "project", model, config, prompts)

    fit = json.loads(out)["fit"]
    assert code == 0 and fit["verdict"] == "fits"  # judged on total memory: the model is tiny
    assert fit["in_use_gib"] == 20.0 and "needs the GPU free" in fit["reason"]


def test_an_unrecognised_layout_registers_with_the_anatomy_unavailable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)  # one tensor named "weight"

    code, out, _ = init(capsys, tmp_path / "project", model, config, prompts)

    assert code == 0
    anatomy = json.loads(out)["anatomy"]
    assert anatomy["components_bytes"] is None
    assert any("unclassified" in reason for reason in anatomy["unavailable"])


def test_a_processor_config_that_runs_code_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path, "compressed-tensors-int4")
    make_gemma4_checkpoint(model)
    edit_json(model / "processor_config.json", auto_map={"AutoProcessor": "remote.P"})

    code, _, err = init(capsys, tmp_path / "project", model, config, prompts)

    assert code == 3 and refusal(err) == "checkpoint_unsupported"


def test_a_tensor_in_two_shards_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    write_shards(model, {SHARDS[0]: {"w": ("BF16", (1,))}, SHARDS[1]: {"w": ("BF16", (1,))}})

    code, _, err = init(capsys, tmp_path / "project", model, config, prompts)

    assert code == 3 and refusal(err) == "checkpoint_layout_invalid"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("auto_map", "checkpoint_unsupported"),
        ("trust_remote_code", "checkpoint_unsupported"),
        ("tokenizer_file_outside", "checkpoint_unsupported"),
        ("fp16_tensor", "checkpoint_precision_unsupported"),
        ("truncated_weights", "checkpoint_layout_invalid"),
        ("unindexed_shard", "checkpoint_layout_invalid"),
        ("symlinked_input", "checkpoint_inventory_unsafe"),
        ("consolidated_copy", "checkpoint_inventory_unexpected"),
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
    elif mutation == "tokenizer_file_outside":
        edit_json(model / "tokenizer_config.json", vocab_file="/etc/passwd")
    elif mutation == "fp16_tensor":
        (model / "model.safetensors").write_bytes(safetensors_bytes(dtype="F16"))
    elif mutation == "truncated_weights":
        (model / "model.safetensors").write_bytes(safetensors_bytes()[:-1])
    elif mutation == "unindexed_shard":
        write_shards(model, SPLIT)
        (model / "model-00003-of-00002.safetensors").write_bytes(safetensors_bytes())
    elif mutation == "consolidated_copy":
        (model / "consolidated.safetensors").write_bytes(safetensors_bytes())
    else:
        outside = tmp_path / "outside.json"
        outside.write_text("{}")
        (model / "vocab.json").symlink_to(outside)

    code, out, err = init(capsys, tmp_path / "project", model, config, prompts)

    assert code == 3 and out == "" and refusal(err) == expected
    assert ("hf download" in err) == (mutation == "consolidated_copy")
    assert not (tmp_path / "project" / "project.json").exists()


def test_init_registers_a_sharded_checkpoint_and_tokens_that_look_like_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    write_shards(model, SPLIT)
    tokenizer = json.loads((model / "tokenizer.json").read_text())
    tokenizer["model"]["vocab"] = {"_file": 0, "auto_map": 1, "tokenizer_file": 2}
    (model / "tokenizer.json").write_text(json.dumps(tokenizer))
    (model / "params.json").write_text("{}")
    (model / "tokenizer.model.v3").write_bytes(b"spm")
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


def test_the_shipped_image_examples_register(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    examples = next(
        p / "examples" for p in Path(__file__).resolve().parents if (p / "examples").is_dir()
    )
    model, _, _ = make_registration_inputs(tmp_path, "compressed-tensors-int4", api="chat")
    make_gemma4_checkpoint(model)
    code, out, err = init(
        capsys, tmp_path / "project", model, examples / "config-images.json",
        examples / "prompts-images.jsonl",
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
        assert code == 0 and "of model weights" in err


def test_a_declared_file_that_vanishes_before_hashing_is_a_changed_checkpoint(
    tmp_path: Path,
) -> None:
    from tensward.artifacts import fingerprint_files
    from tensward.errors import CHECKPOINT_CHANGED, PreflightError

    with pytest.raises(PreflightError) as caught:
        fingerprint_files(tmp_path, [("weights", "gone.safetensors")])

    assert caught.value.code == CHECKPOINT_CHANGED


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


def test_init_registers_an_image_workload_and_inspect_reports_its_images(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, _ = make_registration_inputs(tmp_path, "compressed-tensors-int4", api="chat")
    make_gemma4_checkpoint(model)
    prompts = make_image_workload(tmp_path)
    project = tmp_path / "project"

    code, out, err = init(capsys, project, model, config, prompts)

    assert code == 0, err
    record = json.loads((project / "project.json").read_text())
    assert [i["url"] for i in record["images"]] == ["images/invoice.png", "images/chart.webp"]
    assert str(tmp_path) not in out
    assert run(capsys, "inspect", "--project", str(project))[1] == out


@pytest.mark.parametrize(
    "url", ["https://example.com/a.png", "data:image/png;base64,AAAA", "/etc/a.png", "../a.png"]
)
def test_remote_and_inline_image_urls_are_refused_with_the_fix(
    url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, _ = make_registration_inputs(tmp_path, "compressed-tensors-int4", api="chat")
    make_gemma4_checkpoint(model)
    prompts = make_image_workload(tmp_path, url=url)

    code, _, err = init(capsys, tmp_path / "project", model, config, prompts)

    message = json.loads(err)["message"]
    assert code == 3 and "relative path" in message and "next to the prompts file" in message
    assert "Input should be" not in message and url not in message


def test_images_need_a_checkpoint_that_takes_images(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, _ = make_registration_inputs(tmp_path, api="chat")  # text-only checkpoint
    code, _, err = init(capsys, tmp_path / "project", model, config, make_image_workload(tmp_path))
    assert code == 3 and "takes no images" in json.loads(err)["message"]


def test_an_edited_image_is_named(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model, config, _ = make_registration_inputs(tmp_path, "compressed-tensors-int4", api="chat")
    make_gemma4_checkpoint(model)
    prompts = make_image_workload(tmp_path)
    project = tmp_path / "project"
    assert init(capsys, project, model, config, prompts)[0] == 0
    (prompts.parent / "images" / "chart.webp").write_bytes(webp(64, 64))

    code, _, err = run(capsys, "inspect", "--project", str(project))

    assert code != 0 and "images/chart.webp" in json.loads(err)["message"]


def test_string_content_digests_are_unchanged() -> None:
    entry = PromptEntry.model_validate_json(
        '{"id": "a", "messages": [{"role": "user", "content": "hi"}]}'
    )
    # The 0.1.3 digest of this one-record workload: pins that string content dumps as before.
    assert (
        workload_digest([entry])
        == "95aac3687b2af8e06857446b3b457c195bcb63fb706359c68e2615cd53db0587"
    )


def test_compare_refuses_runs_of_other_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    project = tmp_path / "project"
    assert init(capsys, project, model, config, prompts)[0] == 0
    for name in ("a", "b"):
        run_dir = project / "runs" / name
        run_dir.mkdir(parents=True)
        write_json(run_dir / "run.json", {
            "schema_version": "1", "snapshot_id": "0" * 64, "engine_args": [],
            "settings": {}, "retained_responses": True, "temperature": 0.0,
            "runtime": "local process", "gpus": None, "image": None, "inherited_env": {},
        })  # fmt: skip

    code, _, err = run(capsys, "compare", "--project", str(project), "a", "b")

    assert code != 0 and refusal(err) == "project_inputs_changed"


def test_init_resolves_the_engine_and_refuses_what_it_cannot_serve(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    model, config, prompts = make_registration_inputs(tmp_path)
    gguf = tmp_path / "gguf"
    gguf.mkdir()
    (gguf / "config.json").write_text("{}")
    (gguf / "model-Q4_K_M.gguf").write_bytes(b"GGUF")

    code, out, err = init(capsys, tmp_path / "g", gguf, config, prompts)

    assert code == 3 and out == "" and refusal(err) == "checkpoint_inventory_unexpected"
    assert "supported with the llama.cpp engine, coming in a later" in json.loads(err)["message"]
    assert not (tmp_path / "g" / "project.json").exists()

    monkeypatch.setattr("tensward.project.detect_platform", lambda: None)
    code, out, err = init(capsys, tmp_path / "defaults", model, config, prompts)
    assert code == 0 and json.loads(out)["environment"]["engine_choice"] == (
        "chosen for a safetensors checkpoint; no supported accelerator detected"
    )

    project = tmp_path / "project"
    code, out, err = init(capsys, project, model, config, prompts,
                          "--current", f"vllm serve {model}")  # fmt: skip
    assert code == 0, err
    assert json.loads(out)["environment"] == {
        "platform": None, "engine": "vllm", "engine_choice": "from your --current command",
        "format": "hf-safetensors",
    }  # fmt: skip
    assert "engine: vllm (from your --current command)" in err

    record = json.loads((project / "project.json").read_text())
    assert "engine" not in record["current_setup"]  # as written before 0.2.0
    assert run(capsys, "inspect", "--project", str(project))[1] == out

    monkeypatch.setitem(ENGINES, "other", OtherEngine())
    code, _, err = init(capsys, tmp_path / "p2", model, config, prompts,
                        "--engine", "vllm", "--current", "other-server -m /m")  # fmt: skip
    assert code == 3 and refusal(err) == "project_inputs_invalid"
    assert "--engine vllm conflicts with --current, which runs other" in err
    code, _, err = run(capsys, "analyse", "--project", str(project), "--engine", "other")
    assert refusal(err) == "project_config_unsupported" and not (project / "runs").exists()

    newer = tmp_path / "newer"
    code, out, _ = init(capsys, newer, model, config, prompts, "--engine", "other")
    assert code == 0 and json.loads(out)["environment"]["engine_choice"] == "from --engine"
    monkeypatch.delitem(ENGINES, "other")
    code, _, err = run(capsys, "inspect", "--project", str(newer))
    assert refusal(err) == "project_record_invalid" and "Traceback" not in err

    missing = tmp_path / "missing" / "vllm"
    code, _, err = run(capsys, "analyse", "--project", str(project), "--runtime", "local",
                       "--local-command", str(missing))  # fmt: skip
    assert code == 2 and refusal(err.splitlines()[-1]) == "engine_unavailable"
    assert "install vLLM" in err and not (project / "runs").exists()


def test_env_reports_the_machine_the_engines_and_what_works_here(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    def fake(name: str, body: str) -> None:
        (bin_dir / name).write_text(f"#!/bin/sh\n{body}\n")
        (bin_dir / name).chmod(0o755)

    fake("nvidia-smi", "echo 'NVIDIA L4, 23034, 0, 0, GPU-1a2b, 00000000:00:03.0, 595.91.07, "
                       "0x27B810DE, 95.04.29'")  # fmt: skip
    fake("docker", '[ "$1 $2" = "image inspect" ] && echo "{}" && exit 0\nexit 1')
    fake("python", "echo 0.30.0")
    (bin_dir / "vllm").write_text(f"#!{bin_dir / 'python'}\n")  # the version is the interpreter's
    (bin_dir / "vllm").chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join([str(bin_dir), "/usr/bin", "/bin"]))
    monkeypatch.setitem(sys.modules, "pynvml", None)  # no NVML fallback

    code, out, _ = run(capsys, "env", "--json")
    report = json.loads(out)
    assert code == 0 and report["platform"]["name"] == "nvidia"
    assert [device["name"] for device in report["platform"]["devices"]] == ["NVIDIA L4"]
    (vllm,) = report["engines"]
    assert vllm["docker"]["target"] == "vllm/vllm-openai:v0.30.0"
    assert vllm["docker"]["available"] and vllm["docker"]["version"] == "0.30.0"
    assert vllm["local"]["available"] and vllm["local"]["version"] == "0.30.0"
    assert report["combinations"] == [
        {"engine": "vllm", "format": "hf-safetensors", "platform": "nvidia",
         "runtimes": ["docker", "local"]}
    ]  # fmt: skip
    code, out, _ = run(capsys, "env")
    assert code == 0 and "NVIDIA L4 (driver 595.91.07" in out
    assert "works here: vllm + safetensors on NVIDIA (docker, local)" in out

    fake("nvidia-smi", "exit 1")
    code, out, _ = run(capsys, "env", "--json")
    assert code == 0 and json.loads(out)["platform"] is None
    assert json.loads(out)["combinations"] == []
    assert "platform: no supported accelerator detected" in run(capsys, "env")[1]

    fake("docker", "echo 'permission denied while trying to connect' >&2\nexit 1")
    code, out, _ = run(capsys, "env", "--json", "--runtime", "docker")
    docker = json.loads(out)["engines"][0]["docker"]
    assert docker["reason"] == "docker_unreachable" and "permission denied" in docker["how_to_get"]

    fake("docker", "exec sleep 30")
    monkeypatch.setattr("tensward.probes.PROBE_TIMEOUT_S", 0.5)
    code, out, _ = run(capsys, "env", "--json", "--runtime", "docker")
    assert code == 0 and json.loads(out)["engines"][0]["docker"]["available"] is False
