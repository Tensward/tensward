"""Ollama in ``tensward init``: what is read and said, the files linked or copied into
``~/models`` and the refusals on hostile input."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from registration_fixtures import write_gguf
from test_llamacpp import (
    _cheap_hash,
    _describe,
    _l4,
    _llama_on_path,
    _serve_fake,
)

from tensward import ollama
from tensward import ollama_import as store
from tensward.cli import main
from tensward.wizard import Asker

DIGEST = "a" * 64
L4_LINE = "Hardware: 1× NVIDIA L4 22.5 GiB"


class World:
    """A fake Ollama on a port, its store under ``tmp_path`` and a home to import into."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp_path = tmp_path
        self.home = tmp_path / "home"
        self.store = tmp_path / "store"
        (self.store / "blobs").mkdir(parents=True)
        self.home.mkdir()
        monkeypatch.setenv("HOME", str(self.home))
        monkeypatch.setattr("tensward.ollama.store_root", lambda state: (self.store, None, None))
        # init offers a llama-server running on this machine as the current setup
        monkeypatch.setattr("tensward.detect.running_servers", lambda: ())
        monkeypatch.setattr("tensward.ollama.running_servers", lambda: ())
        _llama_on_path(tmp_path, monkeypatch, ["  CUDA0: NVIDIA L4 (22563 MiB, 22369 MiB free)"])
        (self.gguf,) = write_gguf(
            self.store / "blobs" / f"sha256-{DIGEST}", _describe("qwen3-0.6b-q8_0")
        )
        self.seen = tmp_path / "seen.txt"

    @contextmanager
    def serving(self, **env: str) -> Iterator[str]:
        blob = {"FAKE_OLLAMA_BLOB": str(self.gguf)}
        server, url = _serve_fake(self.gguf, "ollama", self.seen, **blob, **env)
        try:
            yield url
        finally:
            server.terminate()
            server.wait()


def _terminal(
    monkeypatch: pytest.MonkeyPatch, answers: dict[str, list[str]] | None = None
) -> list[str]:
    """Pretend to be a terminal. A prompt that starts with a key of ``answers`` gets the next
    answer listed for it, every other prompt is Enter. Returns what was asked and written."""
    shown: list[str] = []
    queue = {**{"Run the analysis now": ["n"]}, **(answers or {})}

    def read(prompt: str) -> str:
        shown.append(prompt)
        for key, replies in queue.items():
            if prompt.startswith(key) and replies:
                return replies.pop(0)
        return ""

    ask = Asker(read=read, write=shown.append)
    monkeypatch.setattr("tensward.wizard.Asker", lambda: ask)
    monkeypatch.delenv("TENSWARD_JSON", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    return shown


def _init(project: Path, *args: str) -> int:
    with _l4(), _cheap_hash():
        return main(["init", "--project", str(project), *args])


def _prompts(path: Path, *sizes: int) -> Path:
    rows = [
        {"id": f"p{n}", "messages": [{"role": "user", "content": "word " * (size // 5)}]}
        for n, size in enumerate(sizes)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _engine_of(project: Path) -> str:
    record = json.loads((project / "project.json").read_text())
    return str(record["current_setup"]["engine"])


def test_import_continues_into_a_llamacpp_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(tmp_path, monkeypatch)
    target = world.home / "models" / "fake-1b.gguf"
    with world.serving() as url:
        _terminal(monkeypatch)
        assert _init(tmp_path / "one", "--endpoint", url) == 0
        err = capsys.readouterr().err
        assert os.path.samestat(os.stat(target), os.stat(world.gguf))
        assert "Linked ~/models/fake-1b.gguf to your Ollama file (no extra disk)" in err
        assert _engine_of(tmp_path / "one") == "llamacpp"
        assert "built on llama.cpp but not llama-server" not in err
        shown = _terminal(monkeypatch)
        assert _init(tmp_path / "two", "--endpoint", url) == 0
    assert any(
        "(already in ~/models)" in line for line in shown + capsys.readouterr().err.split("\n")
    )
    assert _engine_of(tmp_path / "two") == "llamacpp"


def _screen(
    capsys: pytest.CaptureFixture[str], world: World, tmp_path: Path, *args: str, **env: str
) -> tuple[int, str]:
    with world.serving(**env) as url:
        code = _init(tmp_path / "project", "--endpoint", url, *args)
    return code, capsys.readouterr().err


def test_init_shows_the_ollama_check_and_its_fixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(tmp_path, monkeypatch)
    monkeypatch.delenv("TENSWARD_JSON")
    long = _prompts(tmp_path / "long.jsonl", 24000, 24000, 400)
    flags = ("--prompts", str(long), "--concurrency", "3")

    code, err = _screen(capsys, world, tmp_path, *flags)
    assert code == 3 and "Ollama 0.5.1 is running at http://127.0.0.1:" in err
    assert (
        "Too long: 2 of 3 prompts are over 4096 tokens (estimate: 4 characters per token)." in err
    )
    assert "its last 2,000 tokens (half the context)" in err
    assert "OLLAMA_CONTEXT_LENGTH=7168 OLLAMA_NUM_PARALLEL=3 ollama serve" in err
    assert "Next:" in err and f"tensward init --model {world.home}/models/fake-1b.gguf" in err

    code, err = _screen(capsys, world, tmp_path, *flags, FAKE_OLLAMA_PARAMS="num_ctx 4096")
    assert "PARAMETER num_ctx 7168" in err and "OLLAMA_CONTEXT_LENGTH" not in err
    assert "OLLAMA_NUM_PARALLEL=3" in err

    code, err = _screen(capsys, world, tmp_path, *flags, FAKE_OLLAMA_TEMPLATE="native")
    assert "refused with an error (HTTP 400), not cut" in err

    short = _prompts(tmp_path / "short.jsonl", 4000, 400)
    code, err = _screen(capsys, world, tmp_path, "--prompts", str(short), FAKE_OLLAMA_CTX="32768")
    assert "Your prompts fit: the longest is about 1,000 of 32768 tokens (estimate)." in err
    assert "OLLAMA_CONTEXT_LENGTH" not in err
    assert "Set OLLAMA_NUM_PARALLEL to your peak concurrent requests." in err

    code, err = _screen(capsys, world, tmp_path, FAKE_OLLAMA_LOADED="0", FAKE_OLLAMA_TWO="1")
    assert code == 3 and "Model:    pick one with --model NAME" in err
    code, err = _screen(capsys, world, tmp_path, "--model", "fake:1b", FAKE_OLLAMA_LOADED="0")
    assert "not loaded" in err and "(Ollama's default for this GPU, an estimate)" in err
    assert "Your prompts are not checked." in err

    monkeypatch.setattr("tensward.ollama.ollama_processes", lambda: ((7, "ollama serve"),))
    monkeypatch.setattr("tensward.ollama._process_cgroup", lambda pid: "0::/docker/abc/ollama")
    code, err = _screen(capsys, world, tmp_path, *flags)
    assert "-e OLLAMA_CONTEXT_LENGTH=7168 -e OLLAMA_NUM_PARALLEL=3" in err
    assert "your user can't ask docker" in err and "systemctl" not in err
    monkeypatch.setattr(
        "tensward.ollama._process_cgroup", lambda pid: "0::/system.slice/ollama.service"
    )
    monkeypatch.setattr("tensward.ollama._systemd", lambda: ({}, "ollama", True, True))
    code, err = _screen(capsys, world, tmp_path, *flags)
    assert 'Environment="OLLAMA_CONTEXT_LENGTH=7168"' in err
    monkeypatch.setattr("tensward.ollama.ollama_processes", lambda: ())

    near = ollama.check_prompts([{"prompt": "x" * 13000}, {"prompt": "x" * 100}], 4096)
    assert near is not None and (near.over, near.near, near.longest) == (0, 1, 3250)

    code, err = _screen(capsys, world, tmp_path, *flags, "--json")
    document = json.loads(err.strip().splitlines()[-1])
    assert code == 3 and document["code"] == "project_config_unsupported"
    assert document["ollama"]["prompts"]["over"] == 2
    assert document["ollama"]["fixes"] == ["OLLAMA_CONTEXT_LENGTH=7168", "OLLAMA_NUM_PARALLEL=3"]
    assert (tmp_path / "home" / "models").exists() is False


def test_init_goes_on_without_ollama_when_the_user_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(tmp_path, monkeypatch)
    own = world.home / "models" / "own.gguf"
    own.parent.mkdir()
    own.write_bytes(world.gguf.read_bytes())

    def model_line() -> dict[str, list[str]]:
        return {"Path to your model": [str(own)]}

    with world.serving() as url:
        monkeypatch.setattr("tensward.ollama.DEFAULT_URL", url)
        _terminal(monkeypatch, {"[Enter / n / q]": ["n"], **model_line()})
        assert _init(tmp_path / "n", "--model", "fake:1b") == 0
        assert _engine_of(tmp_path / "n") == "llamacpp"
        assert not (world.home / "models" / "fake-1b.gguf").exists()

        shown = _terminal(monkeypatch)
        assert _init(tmp_path / "cur", "--current", f"llama-server -m {own} -ngl 99") == 0
        assert not any("Which Ollama model" in line for line in shown)

    with world.serving(FAKE_OLLAMA_LOADED="0", FAKE_OLLAMA_TWO="1") as url:
        monkeypatch.setattr("tensward.ollama.DEFAULT_URL", url)
        _terminal(monkeypatch, {"[1] ": ["3"], **model_line()})
        assert _init(tmp_path / "list") == 0
        assert _engine_of(tmp_path / "list") == "llamacpp"

    with world.serving(FAKE_OLLAMA_EMPTY="1") as url:
        monkeypatch.setattr("tensward.ollama.DEFAULT_URL", url)
        _terminal(monkeypatch, model_line())
        assert _init(tmp_path / "empty") == 0
        assert "Ollama is running but has no models yet." in capsys.readouterr().err

    with world.serving() as url:
        monkeypatch.setattr("tensward.ollama.DEFAULT_URL", url)
        assert _init(tmp_path / "scripted", "--yes") == 3
        assert ollama.SCRIPTED_HINT in capsys.readouterr().err

    monkeypatch.setattr("tensward.ollama.shutil", SimpleNamespace(which=lambda name: "/bin/ollama"))
    assert _init(tmp_path / "stopped", "--yes") == 3
    assert ollama.STOPPED_NOTE in capsys.readouterr().err


def _state(root: Path | None, name: str = "m:1b", **where: object) -> ollama.State:
    env = {"OLLAMA_MODELS": str(root)} if root else {}
    tag = {"name": name, "size": 1000, "details": {"format": "gguf"}}
    return ollama.State(
        "http://127.0.0.1:11434", ollama.Where(env=env, **where), version="0.40", tags=[tag]
    )


def _show(directory: Path | str, digest: str = DIGEST) -> dict[str, str]:
    return {"modelfile": f"# Modelfile\nFROM {directory}/blobs/sha256-{digest}\n"}


def _manifest(root: Path, name: str, layers: list[tuple[str, str]]) -> None:
    body = {"layers": [{"mediaType": k, "digest": f"sha256:{d}", "size": 1} for k, d in layers]}
    blob = root / "blobs" / f"sha256-{'b' * 64}"
    blob.write_text(json.dumps(body))
    link = root / "manifests-v2" / "ollama.com" / "library" / name.split(":")[0] / "1b"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.unlink(missing_ok=True)
    link.symlink_to(blob)


def test_import_chooses_the_command_for_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = tmp_path / "store"
    (root / "blobs").mkdir(parents=True)
    blob = root / "blobs" / f"sha256-{DIGEST}"
    blob.write_bytes(b"GGUF")
    model = "m:1b"
    monkeypatch.setattr("tensward.ollama_import.gguf_problem", lambda path, label: None)
    monkeypatch.setattr("tensward.ollama.store.gguf_problem", lambda path, label: None)

    plan = ollama.plan_import(_state(root), model, _show(root))
    assert plan.kind == "link" and "link, no extra disk" in plan.row
    assert ollama.import_commands(plan)[0].startswith("mkdir -p ")

    with monkeypatch.context() as other:
        mine = os.getuid()
        other.setattr("tensward.ollama_import.os.getuid", lambda: mine + 1)
        plan = ollama.plan_import(_state(root), model, _show(root))
    assert plan.kind == "copy"
    assert (
        "cp " in ollama.import_commands(plan)[0] and "sudo" not in ollama.import_commands(plan)[0]
    )

    gone = _show(root, "c" * 64)
    plan = ollama.plan_import(_state(root), model, gone)
    (command,) = ollama.import_commands(plan)
    assert plan.kind == "sudo" and "sudo cp" in command and 'chown -h "$USER:"' in command
    assert command.index(".part") < command.index("mv -n")

    docker = _state(None, container="ollama", container_root=store.CONTAINER_ROOT, kind="docker")
    plan = ollama.plan_import(docker, model, _show(store.CONTAINER_ROOT))
    (command,) = ollama.import_commands(plan)
    assert (
        plan.kind == "docker"
        and f"docker cp ollama:/root/.ollama/models/blobs/sha256-{DIGEST}" in command
    )
    assert "sudo" not in command

    mounted = _state(
        None,
        container="ollama",
        container_root=store.CONTAINER_ROOT,
        kind="docker",
        mounts=((str(Path(store.CONTAINER_ROOT).parent), str(root.parent)),),
    )
    (root.parent / "models").symlink_to(root)
    plan = ollama.plan_import(mounted, model, _show(store.CONTAINER_ROOT))
    assert plan.kind == "link"

    held = _state(root)
    escape = "x\x1b[2Jy"
    assert "\x1b" not in ollama.stop_command(held, escape)
    held.tags = [{"name": model, "details": {"format": f"evil{escape}"}}]
    assert "\x1b" not in str(ollama.plan_import(held, model, _show(root)).refusal)

    shards = [DIGEST, "d" * 64, "e" * 64]
    for digest in shards[1:]:
        (root / "blobs" / f"sha256-{digest}").write_bytes(b"GGUF")
    _manifest(root, "m:1b", [(store.MODEL_LAYER, d) for d in shards])
    plan = ollama.plan_import(_state(root), model, _show(root))
    assert plan.target is not None and plan.target.first.name == "m-1b-00001-of-00003.gguf"
    assert len(ollama.import_commands(plan)) == 3

    _manifest(
        root,
        "m:1b",
        [(store.MODEL_LAYER, DIGEST), ("application/vnd.ollama.image.adapter", "d" * 64)],
    )
    assert "fine-tune adapter" in ollama.plan_import(_state(root), model, _show(root)).refusal

    state = _state(root)
    state.tags = [{"name": model, "details": {"format": "safetensors"}}]
    assert "stored as safetensors" in ollama.plan_import(state, model, _show(root)).refusal
    state.tags = [{"name": model, "remote_host": "https://ollama.com"}]
    assert "is a cloud model" in ollama.plan_import(state, model, _show(root)).refusal

    (root / "manifests-v2" / "ollama.com" / "library" / "m" / "1b").unlink()
    outside = tmp_path / "outside"
    outside.write_text('{"layers": []}')
    (root / "manifests-v2" / "ollama.com" / "library" / "m" / "1b").symlink_to(outside)
    assert store.read_manifest(root, "m:1b") is None

    for name in ("../x:1b", "a b:1b", "m/../x:1b", "x" * 81):
        assert store.parse_name(name) is None
        assert ollama.plan_import(_state(root, name), name, _show(root)).kind == "none"
    bad = ollama.plan_import(_state(root), model, _show(root, "zz"))
    assert bad.kind == "none"
    assert ollama.plan_import(_state(root), model, _show(tmp_path / "elsewhere")).refusal == (
        store.NOT_VISIBLE
    )

    (tmp_path / "home" / "models").mkdir(parents=True)
    (tmp_path / "home" / "models" / "m-1b.gguf").write_bytes(b"other bytes")
    plan = ollama.plan_import(_state(root), model, _show(root))
    assert plan.different == "m-1b.gguf"

    if os.getuid() != 0:
        (root / "blobs").chmod(0)
        try:
            plan = ollama.plan_import(_state(root), model, _show(root))
        finally:
            (root / "blobs").chmod(0o755)
        assert plan.kind == "sudo" and "sudo cp" in ollama.import_commands(plan)[0]

    data = tmp_path / "data.bin"
    data.write_bytes(b"abc")
    digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert store.matches_digest(data, f"sha256-{digest}")
    assert not store.matches_digest(data, f"sha256-{'0' * 64}")


def test_context_source_order() -> None:
    gib = 2**30
    state = _state(None)
    state.loaded = [{"name": "m:1b", "context_length": 8192}]
    show = {
        "parameters": "num_ctx 6000",
        "model_info": {"general.architecture": "q", "q.context_length": 7000},
    }
    state.where.env["OLLAMA_CONTEXT_LENGTH"] = "5000"
    pick = ollama.context_value
    assert pick(state, "m:1b", show, 80 * gib) == ollama.Value(7000, "loaded")
    state.loaded = []
    assert pick(state, "m:1b", show, None) == ollama.Value(6000, "modelfile")
    assert pick(state, "m:1b", {}, None) == ollama.Value(5000, "environment")
    del state.where.env["OLLAMA_CONTEXT_LENGTH"]
    for size, wanted in ((22.5, 4096), (23.99, 32768), (47, 262144)):
        assert pick(state, "m:1b", {}, int(size * gib)) == ollama.Value(wanted, "default", True)
    state.where.env["OLLAMA_GPU_OVERHEAD"] = str(2 * gib)
    assert pick(state, "m:1b", {}, 24 * gib).value == 4096
    assert ollama.keep_alive_seconds("5m") == 300 and ollama.keep_alive_seconds("-1") == -1
    assert ollama.keep_alive_seconds("1h30m") == 5400


def test_a_bare_model_name_is_the_latest_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(None, "q27f:latest")
    arguments = SimpleNamespace(model="q27f")
    assert ollama._pick_model(arguments, state, None) == "q27f:latest"
    assert ollama._full_name(state, "q27f:1b") == "q27f:1b"
    assert ollama._full_name(_state(None, "q27f:1b"), "q27f") == "q27f"


def test_init_takes_a_bare_name_to_the_latest_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(None, "q27f:latest")
    monkeypatch.setattr(ollama, "read_state", lambda *args: state)
    monkeypatch.setattr(ollama, "detect_hardware", lambda: SimpleNamespace(kind="nvidia"))
    monkeypatch.setattr(ollama, "running_servers", lambda: [])
    imported = ollama.Step(reset=True)
    monkeypatch.setattr(ollama, "_flow", lambda *args, **kwargs: imported)
    arguments = SimpleNamespace(model="q27f", config=None, json=False)
    assert ollama.init_step(arguments, None, None) is imported
    assert arguments.model == "q27f:latest"


RUNNER = (
    "/opt/ollama-v0.40.16/lib/ollama/llama-server --model /store/blobs/sha256-{digest} "
    "--port 38711 --host 127.0.0.1 --no-webui --offline -c 32768 -np 8 --log-verbosity 4 "
    "--no-log-prefix --no-log-timestamps --flash-attn auto -b 512 -ub 512 --context-shift "
    "--keep 4"
)


def test_import_starts_from_the_loaded_runner_and_names_the_kept_step_in_ollama_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(tmp_path, monkeypatch)
    monkeypatch.setattr("tensward.ollama.ollama_processes", lambda: ((7, "ollama serve"),))
    monkeypatch.setattr("tensward.ollama._process_cgroup", lambda pid: "0::/user.slice")
    runner = RUNNER.format(digest=DIGEST)
    monkeypatch.setattr("tensward.detect._processes", lambda: [(8, 7, runner)])
    with world.serving() as url:
        _terminal(monkeypatch)
        assert _init(tmp_path / "one", "--endpoint", url) == 0
    err = capsys.readouterr().err
    assert "started with the settings your Ollama runs it with" in err
    command = json.loads((tmp_path / "one" / "project.json").read_text())["current_setup"]
    settings = command["settings"]
    assert settings["prefill_batch_tokens"] == 512 and settings["max_concurrent_requests"] == 8
    assert command["source"] != "defaults"
    assert any("--port" in line for line in err.splitlines())
    choices = json.loads((tmp_path / "one" / "choices.json").read_text())
    assert choices["ollama"] and choices["baseline"] == "command"

    from tensward import ollama_settings

    assert ollama_settings.lines("0.40.16", ["--checkpoint-min-step=0"]) == [
        "In Ollama 0.40.16:",
        '  Environment="LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT=0"   for ollama serve '
        "(sudo systemctl edit ollama); the runner's log then says min spacing = 0",
        "Then: sudo systemctl restart ollama",
    ]
    assert ollama_settings.lines("0.40.16", []) == []
    # G4: llamacpp-slots kept on an Ollama project, in Ollama 0.40.16's own settings
    kept = ["parallel=4", "ctx-size=16384", "n-gpu-layers=43", "ctx-checkpoints=1"]
    assert ollama_settings.lines("0.40.16", kept) == [
        "In Ollama 0.40.16:",
        '  Environment="OLLAMA_NUM_PARALLEL=4"   for ollama serve (sudo systemctl edit ollama)',
        "  PARAMETER num_ctx 4096   in the model's Modelfile",
        "  PARAMETER num_gpu 43   in the model's Modelfile",
        "  (num_ctx is per request: Ollama starts its runner with num_ctx x 4 slots, "
        "--ctx-size 16384)",
        "  --ctx-checkpoints 1: this change needs llama.cpp; Ollama has no setting for it",
        "Then: ollama create <model> -f Modelfile   (it starts FROM <model>)",
        "Then: sudo systemctl restart ollama",
    ]
    other = ["ctx-size=8192", "batch-size=1024", "ubatch-size=1024", "threads=8",
             "flash-attn=on", "cache-type-k=q8_0", "cache-type-v=q8_0"]  # fmt: skip
    shown = ollama_settings.lines("0.40.16", other, parallel=2)
    assert "  PARAMETER num_ctx 4096   in the model's Modelfile" in shown
    assert "  PARAMETER num_batch 1024   in the model's Modelfile" in shown
    assert "  PARAMETER num_thread 8   in the model's Modelfile" in shown
    assert any('"OLLAMA_FLASH_ATTENTION=1"' in line for line in shown)
    assert any('"OLLAMA_KV_CACHE_TYPE=q8_0"' in line for line in shown)
    # O35: pinned loading is use_mmap false (llm/llama_server.go appendLoadModeArgs)
    pinned = ollama_settings.lines("0.40.16", ["n-cpu-moe=17", "load-mode=none"])
    assert "  PARAMETER use_mmap false   in the model's Modelfile" in pinned
    # O35's kept setup: num_gpu 41 without the expert offload ran out of memory in Ollama
    kept = ["n-gpu-layers=41", "n-cpu-moe=18", "load-mode=none", "threads=16"]
    shown = ollama_settings.lines("0.40.16", kept)
    assert not any("num_gpu" in line and "PARAMETER" in line for line in shown)
    assert "  PARAMETER num_thread 16   in the model's Modelfile" in shown
    assert any(
        line.startswith("  --n-gpu-layers 41 with --n-cpu-moe 18 (one placement;")
        and "keep your current num_gpu" in line
        and line.endswith(ollama_settings.NO_SETTING)
        for line in shown
    )
    split = ollama_settings.lines("0.40.16", ["batch-size=2048", "ubatch-size=512"])
    assert split[1] == (
        "  --batch-size 2048 --ubatch-size 512 (num_batch sets -b and -ub to one value): "
        "this change needs llama.cpp; Ollama has no setting for it"
    )


def _no_offers(project: Path) -> list[object]:
    from tensward.cli_try import NothingToTry

    raise NothingToTry("none")


@pytest.mark.parametrize("offers", [_no_offers, lambda project: [object()]])
def test_try_keep_names_the_kept_step_in_ollama_words(
    offers: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tensward import cli_try
    from tensward.kept import Kept

    kept = Kept("0002", ("--checkpoint-min-step=0", "--n-gpu-layers=66"), (), "registered", "s")
    monkeypatch.setattr(
        cli_try, "keep_reworded", lambda p, step: ("--checkpoint-min-step 0", "why")
    )
    monkeypatch.setattr(cli_try, "offers", offers)
    monkeypatch.setattr(cli_try, "read_kept", lambda p: kept)
    monkeypatch.setattr(cli_try, "read_choices", lambda p: SimpleNamespace(ollama="0.40.16"))
    monkeypatch.setattr(cli_try, "_held", lambda p, k: (frozenset(), frozenset()))
    cli_try._keep_reworded(tmp_path, as_json=False)
    out = capsys.readouterr().out
    assert out.startswith("Kept --checkpoint-min-step 0 (why).\nIn Ollama 0.40.16:\n"), out
    assert "LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT=0" in out


def test_import_without_a_loaded_runner_measures_the_defaults_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(tmp_path, monkeypatch)
    with world.serving() as url:
        _terminal(monkeypatch)
        assert _init(tmp_path / "one", "--endpoint", url) == 0
    err = capsys.readouterr().err
    assert "Ollama has not loaded this model, so Tensward measures llama.cpp's defaults" in err
    assert 'ollama run fake:1b "" and run init again.' in err
    choices = json.loads((tmp_path / "one" / "choices.json").read_text())
    assert choices["baseline"] == "defaults"


def test_the_screen_reads_the_runner_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(tmp_path, monkeypatch)
    monkeypatch.delenv("TENSWARD_JSON")
    monkeypatch.setattr("tensward.ollama.ollama_processes", lambda: ((7, "ollama serve"),))
    monkeypatch.setattr("tensward.ollama._process_cgroup", lambda pid: "0::/user.slice")
    monkeypatch.setattr(
        "tensward.detect._processes", lambda: [(8, 7, RUNNER.format(digest=DIGEST))]
    )
    env: dict[int, dict[str, str]] = {}
    monkeypatch.setattr(
        "tensward.ollama._process_environment", lambda pid, prefix="OLLAMA_": env.get(pid)
    )

    code, err = _screen(capsys, world, tmp_path)
    assert "Runner:    settings unknown" in err and "MoE cache" not in err

    env[8] = {"LLAMA_ARG_MOE_CACHE_MIB": "4096", "LLAMA_ARG_KV_UNIFIED": "1"}
    code, err = _screen(capsys, world, tmp_path)
    assert "LLAMA_ARG_MOE_CACHE_MIB=4096, LLAMA_ARG_KV_UNIFIED=1" in err
    assert 'HTTP 500 "MoE cache requires a MoE model"' in err and "Remove the variable" in err
    assert "Context:   32768 tokens per request (one pool of 32768 shared by 8 requests)" in err

    env[8]["LLAMA_ARG_MOE_CACHE_MIB"] = "0"
    code, err = _screen(capsys, world, tmp_path)
    assert "MOE_CACHE_MIB=0 (off)" in err and "MoE cache requires" not in err


def test_the_runner_command_uses_the_files_own_template():
    """Ollama starts llama-server with --no-jinja --chat-template chatml because it renders the
    prompt and reads tool calls itself; the llama.cpp command made from it must use the file's
    own template, as Ollama 0.40.16 ran gemma 4 (G4 runner line)."""
    blob = f"sha256-{DIGEST}"
    where = ollama.Where(
        kind="service",
        runners=(
            f"/usr/lib/ollama/llama-server --model /store/blobs/{blob} -c 4096 -np 1 --no-jinja "
            "--chat-template chatml --flash-attn auto -b 512",
        ),
    )
    located = store.Located(blobs=(blob,), root=Path("/store"), container=None, container_root=None)
    command = ollama.runner_command(where, located, Path("/m/gemma.gguf"))
    assert command == "llama-server --model /m/gemma.gguf -c 4096 -np 1 --flash-attn auto -b 512"
