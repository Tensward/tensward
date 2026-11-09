import json
import re
import shlex
from dataclasses import replace
from pathlib import Path

import pytest
from alternative_records import LFM2, RECORDS, TESTER
from registration_fixtures import write_gguf

from tensward import alternatives
from tensward.alternatives import (
    YOUR_FLAGS,
    Alternative,
    checkpoint_dir,
    init_command,
    matching,
    model_identity,
    parse,
    serve_command,
    steps,
    wrapped,
)
from tensward.engines import ENGINES
from tensward.engines.vllm.command import VLLM_VERSION

HEADERS = Path(__file__).parent / "gguf_headers"
REVISION = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def records(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alternatives, "table", lambda: parse(RECORDS))
    monkeypatch.setattr(alternatives, "on_path", lambda program: True)


def _names(facts: dict, goal: str = "tpot") -> list[str]:
    return [a.name for a in matching(facts, goal)]


def _record(name: str) -> Alternative:
    return next(a for a in parse(RECORDS) if a.name == name)


def _gguf(path: Path, **metadata: object) -> Path:
    description = json.loads((HEADERS / "qwen3.8-27b-ud-q4_k_m.json").read_text())
    description["metadata"] = {**description["metadata"], **metadata}
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_gguf(path, description)[0]


def test_the_tester_case_gets_the_engine_and_the_file_largest_gain_first() -> None:
    assert _names(TESTER) == ["qwen38-27b-l4-agent-vllm-int4", "qwen38-27b-l4-agent-iq4-xs"]


def test_an_lfm2_case_matches_across_family_spellings() -> None:
    assert _names(LFM2) == ["lfm2-8b-a1b-l4-chat-vllm"]


def test_a_goal_only_the_file_helps_gets_the_file_alone() -> None:
    assert _names(TESTER, "throughput") == ["qwen38-27b-l4-agent-iq4-xs"]


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ({"model.identity": None}, "another or an unknown model"),
        ({"model.identity": "qwen3827blegal"}, "another or an unknown model"),
        ({"model.identity": "qwen3527b"}, "another or an unknown model"),
        ({"model.family": "llama"}, "another model family"),
        ({"model.params_b": 8.0}, "another size"),
        ({"gpu.model": "L40S"}, "another GPU"),
        ({"gpu.count": 2}, "another GPU"),
        ({"workload.clients": 8}, "concurrency"),
        ({"workload.tools": False}, "another workload shape"),
        ({"engine.name": "vllm"}, "another engine runs now"),
        ({"model.ftype": "IQ4_XS"}, "another file"),
        ({"workload.prompt_tokens_max": 120_000}, "a longer context than measured"),
        ({"platform.name": "apple"}, "unsupported here"),
    ],
)
def test_nothing_is_shown_when_any_check_fails(change: dict, why: str) -> None:
    facts = TESTER | change
    refusals = {a.name: a.refusal(facts, "tpot") for a in parse(RECORDS)}
    assert why in refusals.values()
    if why == "unsupported here":
        assert _names(facts) == ["qwen38-27b-l4-agent-iq4-xs"]
    elif why == "a longer context than measured":
        assert refusals["qwen38-27b-l4-agent-iq4-xs"] == "does not fit"
        assert _names(facts) == []
    else:
        assert _names(facts) == []


def test_what_already_runs_is_not_offered() -> None:
    assert _record("qwen38-27b-l4-agent-iq4-xs").already_run({"model.ftype": "IQ4_XS"})
    assert _record("qwen38-27b-l4-agent-vllm-int4").already_run({"engine.name": "vllm"})


@pytest.mark.parametrize(
    "name",
    ["under-three-percent", "within-noise", "one-run", "mixed-sign", "more-broken",
     "wrong-direction"],
)  # fmt: skip
def test_a_record_that_fails_the_pre_registered_rule_never_shows(name: str) -> None:
    assert not _record(name).passes("tpot") and name not in _names(TESTER)


def test_a_placeholder_never_shows() -> None:
    assert "placeholder" not in {a.name for a in parse(RECORDS)}


def test_answers_are_within_the_usual_variation_only_down_to_two_standard_errors() -> None:
    file = _record("qwen38-27b-l4-agent-iq4-xs")
    assert replace(file, identical=57).answers_within()  # 57 against 58 of 64
    assert not file.answers_within()  # 40 against 58 of 64
    assert not _record("qwen38-27b-l4-agent-vllm-int4").answers_within()  # 12 against 144
    exact = replace(_record("qwen38-27b-l4-agent-iq4-xs"), identical=63, floor=64)
    assert not exact.answers_within()


def test_identity_comes_from_the_checkpoint_and_refuses_anything_unclear(tmp_path: Path) -> None:
    assert model_identity(_gguf(tmp_path / "a" / "x.gguf")) == "qwen3827b"
    tuned = _gguf(tmp_path / "b" / "x.gguf", **{"general.finetune": "Legal"})
    assert model_identity(tuned) is None
    named = {"general.finetune": "Legal", "general.name": "Qwen3.8-27B Legal"}
    assert model_identity(_gguf(tmp_path / "e" / "x.gguf", **named)) == "qwen3827blegal"
    renamed = _gguf(tmp_path / "c" / "x.gguf", **{"general.name": "Qwen3.8-27B Abliterated"})
    assert model_identity(renamed) is None
    other = {"general.name": "Qwen3.5-27B", "general.basename": "Qwen3.5-27B"}
    assert model_identity(_gguf(tmp_path / "d" / "x.gguf", **other)) == "qwen3527b"
    folder = tmp_path / "Qwen3.8-27B-int4"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"_name_or_path": "Qwen/Qwen3.8-27B"}))
    assert model_identity(folder) == "qwen3827b"
    mine = tmp_path / "my-model"
    mine.mkdir()
    (mine / "config.json").write_text(json.dumps({"_name_or_path": "Qwen/Qwen3.8-27B"}))
    assert model_identity(mine) is None


@pytest.mark.parametrize(
    "spelling",
    ["-m {path}", "--model {path}", "--model={path}", "-m models/Qwen3.8-27B-Q4_K_M.gguf"],
)
def test_a_file_alternative_replaces_only_the_model_path(spelling: str, tmp_path: Path) -> None:
    model = tmp_path / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
    current = f"llama-server {spelling.format(path=model)} --jinja -c 16384 -np 3"
    record = _record("qwen38-27b-l4-agent-iq4-xs")
    command, runs = serve_command(record, TESTER, current, model)
    llamacpp = ENGINES["llamacpp"]
    parsed = llamacpp.parse_setup(command)
    assert runs and parsed.model == str(model.parent / "Qwen3.8-27B-UD-IQ4_XS.gguf")
    assert parsed.settings == llamacpp.parse_setup(current).settings


def test_a_command_without_a_clear_model_path_says_to_add_your_flags(tmp_path: Path) -> None:
    model = tmp_path / "Qwen3.8-27B-Q4_K_M.gguf"
    record = _record("qwen38-27b-l4-agent-iq4-xs")
    for current in ("llama-server -hf unsloth/Qwen3.8-27B-GGUF --jinja", None):
        command, runs = serve_command(record, TESTER, current, model)
        assert not runs and command.endswith(f" {YOUR_FLAGS}")


def test_a_file_alternative_prints_every_step_in_a_new_folder_like_the_project(
    tmp_path: Path,
) -> None:
    model = tmp_path / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
    project = tmp_path / "models" / ".tensward"
    record = _record("qwen38-27b-l4-agent-iq4-xs")
    choices, screen = steps(record, TESTER, f"llama-server -m {model} --jinja", model, project)
    download, folder, init = (command for command, _ in choices)
    assert screen == [f"get the file first: {download}", folder, init]
    assert shlex.split(download) == [
        "hf", "download", "unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-IQ4_XS.gguf",
        "--revision", REVISION, "--local-dir", str(model.parent),
    ]  # fmt: skip
    assert folder == "mkdir -p ~/tensward-iq4xs && cd ~/tensward-iq4xs"
    words = shlex.split(init)
    assert words[:3] == ["tensward", "init", "--current"] and words[4:] == ["--like", str(project)]
    assert "--jinja" in ENGINES["llamacpp"].parse_setup(words[3]).text


def test_an_init_step_with_a_blanked_secret_says_to_fill_it_in(tmp_path: Path) -> None:
    model = tmp_path / "models" / "Qwen3.8-27B-Q4_K_M.gguf"
    record = _record("qwen38-27b-l4-agent-iq4-xs")
    current = f"llama-server -m {model} --api-key ***"
    choices, screen = steps(record, TESTER, current, model, tmp_path)
    assert "***" in screen[-1] and choices[-1][1].endswith("with the values shown as *** filled in")


def test_missing_uv_and_hf_are_installed_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(alternatives, "on_path", lambda program: False)
    model = tmp_path / "Qwen3.8-27B-Q4_K_M.gguf"
    record = _record("qwen38-27b-l4-agent-iq4-xs")
    _, screen = steps(record, TESTER, f"llama-server -m {model}", model, tmp_path)
    assert screen[:2] == [
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "uv tool install huggingface_hub",
    ]


def test_an_engine_alternative_is_the_measured_command_with_tools_and_passes_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(alternatives, "installed", lambda engine: False)
    model = tmp_path / "models" / "qwen3.8-27b-q4" / "Qwen3.8-27B-Q4_K_M.gguf"
    record = _record("qwen38-27b-l4-agent-vllm-int4")
    choices, screen = steps(record, TESTER, f"llama-server -m {model}", model, tmp_path)
    install, download, folder, init = (command for command, _ in choices)
    assert install == f"uv tool install vllm=={VLLM_VERSION}"
    assert screen[0] == install
    assert screen[1].startswith("sudo apt install -y cuda-toolkit-13-0")
    assert screen[2:] == [f"get the checkpoint first: {download}", folder, init]
    assert shlex.split(download)[3:5] == ["--revision", REVISION]
    vllm = ENGINES["vllm"]
    parsed = vllm.parse_setup(shlex.split(init)[3])
    assert parsed.model == str(tmp_path / "models" / "Qwen3.8-27B-int4")
    loose = tmp_path / "downloads" / "Qwen3.8-27B-Q4_K_M.gguf"
    assert checkpoint_dir(record, loose) == Path.home() / "models" / "Qwen3.8-27B-int4"
    assert parsed.settings.max_context_len == 4096
    assert parsed.settings.max_concurrent_requests == 3
    assert parsed.settings.tool_parser == "qwen3_xml"
    assert vllm.tool_calling_fix(parsed.settings, "command") is None
    once, _ = serve_command(record, TESTER, None, model)
    assert shlex.split(once).count("--enable-auto-tool-choice") == 1


def test_an_engine_alternative_needs_a_parser_for_tools() -> None:
    bare = replace(_record("qwen38-27b-l4-agent-vllm-int4"), tool_parser="")
    assert bare.refusal(TESTER, "tpot") == "no tool-call parser for a workload with tools"


def test_a_long_command_wraps_within_80_columns_and_reads_as_the_same_command() -> None:
    model = "/home/tester/models/qwen3.8-27b-q4/Qwen3.8-27B-INT4"
    serve = f"vllm serve --model {model} --max-num-seqs 3 --max-model-len 4096 " * 3
    for text in (f'tensward init --current "{serve.strip()}" --like /p', f"echo '{serve}'"):
        lines = wrapped(text, "    5. ")
        assert all(len(line) <= 80 for line in lines) or "'" in text
        joined = re.sub(r" \\\n +", " ", "\n".join(lines)).removeprefix("    5. ")
        assert shlex.split(joined) == shlex.split(text)


def test_a_wrapped_init_step_keeps_quoted_arguments_whole() -> None:
    serve = shlex.join([
        "llama-server", "-m", "/home/tester/models/Qwen3.8-27B-IQ4_XS.gguf", "--jinja",
        "--chat-template-kwargs", '{"enable_thinking": false}', "--alias", "tester's model",
        "--ctx-size", "16384", "--parallel", "3",
    ])  # fmt: skip
    text = init_command(serve, Path("/home/tester/tensward"))
    lines = wrapped(text, "    5. ")
    assert len(lines) > 1
    words = shlex.split("\n".join(lines).removeprefix("    5. "))
    assert [word for word in words if word != "\n"] == [
        "tensward", "init", "--current", serve, "--like", "/home/tester/tensward",
    ]  # fmt: skip
