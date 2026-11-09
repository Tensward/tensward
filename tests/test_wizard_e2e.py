"""``tensward init`` end to end: a pseudo-terminal session with the fake vLLM and nvidia-smi on
PATH, then the same setup from flags only, and the path an agent takes without a terminal."""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from registration_fixtures import (
    SAFETENSORS_NAME,
    make_registration_inputs,
    safetensors_from_tensors,
)

HERE = Path(__file__).resolve().parent
SATURATED_ROW = "99,69.5,70.00,1230,1590,Active,Not Active,Not Active"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Look up {name}.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    for name in ("find_invoice", "find_customer")
]


def shim(path: Path, target: Path) -> None:
    path.write_text(
        f"#!{sys.executable}\nimport runpy, sys\n"
        f"sys.argv[0] = {str(target)!r}\n"
        f"runpy.run_path({str(target)!r}, run_name='__main__')\n"
    )
    path.chmod(0o755)


def machine(tmp_path: Path) -> dict[str, str]:
    """A PATH holding a vLLM 0.30.0 and an L4, with HOME and the cache inside ``tmp_path``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim(bin_dir / "vllm", HERE / "fake_vllm_server.py")
    shim(bin_dir / "nvidia-smi", HERE / "fake_nvidia_smi.py")
    metadata = tmp_path / "site" / "vllm-0.30.0.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: vllm\nVersion: 0.30.0\n")
    home = tmp_path / "home"
    home.mkdir()
    return {
        **{
            k: v
            for k, v in os.environ.items()
            if k != "TENSWARD_JSON" and not k.startswith(("VLLM_", "LLAMA_"))
        },  # fmt: skip
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "PYTHONPATH": str(tmp_path / "site"),
        "PYTHONUNBUFFERED": "1",
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "FAKE_SMI_ROW": SATURATED_ROW,
        "FAKE_SMI_DEVICE": "NVIDIA L4",
        "FAKE_TOKEN_DELAY_S": "0.0003",
    }


def checkpoint(tmp_path: Path) -> Path:
    model, _, _ = make_registration_inputs(tmp_path / "inputs", api="chat")
    document = json.loads((model / "config.json").read_text())
    dims = {
        "model_type": "qwen3",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
    }
    dims |= {"num_key_value_heads": 2, "intermediate_size": 128, "vocab_size": 100}
    (model / "config.json").write_text(json.dumps({**document, **dims}))
    tensors = {"model.embed_tokens.weight": ("BF16", (100, 64))}
    for layer in range(2):
        prefix = f"model.layers.{layer}"
        shapes = {"self_attn.q_proj": (64, 64), "self_attn.k_proj": (32, 64)}
        shapes |= {"self_attn.v_proj": (32, 64), "self_attn.o_proj": (64, 64)}
        shapes |= {"mlp.gate_proj": (128, 64), "mlp.up_proj": (128, 64), "mlp.down_proj": (64, 128)}
        tensors |= {f"{prefix}.{name}.weight": ("BF16", shape) for name, shape in shapes.items()}
    (model / SAFETENSORS_NAME).write_bytes(safetensors_from_tensors(tensors))
    return model


class Terminal:
    """``tensward`` on a pseudo-terminal: wait for a prompt, answer it."""

    def __init__(self, arguments: list[str], cwd: Path, env: dict[str, str]) -> None:
        self.master, slave = os.openpty()
        self.process = subprocess.Popen(
            [sys.executable, "-m", "tensward.cli", *arguments],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            cwd=cwd,
            env=env,
            close_fds=True,
        )
        os.close(slave)
        self.text = ""
        self.cursor = 0

    def _read(self, wait: float) -> bool:
        if not select.select([self.master], [], [], wait)[0]:
            return True
        try:
            data = os.read(self.master, 65536)
        except OSError:
            return False
        self.text += data.decode(errors="replace")
        return bool(data)

    def expect(self, needle: str, seconds: float = 60) -> None:
        deadline = time.monotonic() + seconds
        while needle not in self.text[self.cursor :]:
            if time.monotonic() > deadline or not self._read(0.5):
                self.process.kill()
                pytest.fail(f"never saw {needle!r}; the session so far:\n{self.text}")
        self.cursor = self.text.index(needle, self.cursor) + len(needle)

    def send(self, line: str) -> None:
        os.write(self.master, f"{line}\n".encode())

    def finish(self, seconds: float = 180) -> int:
        deadline = time.monotonic() + seconds
        while self._read(0.5):
            if time.monotonic() > deadline:
                self.process.kill()
                pytest.fail(f"still running; the session so far:\n{self.text}")
        status = self.process.wait(timeout=30)
        os.close(self.master)
        return status


def tensward(
    arguments: list[str],
    cwd: Path,
    env: dict[str, str],
    stdin: int = subprocess.DEVNULL,
    timeout: float = 20,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tensward.cli", *arguments],
        cwd=cwd,
        env=env,
        stdin=stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_wizard_from_model_path_to_first_report(tmp_path: Path) -> None:
    env = machine(tmp_path)
    model = checkpoint(tmp_path)
    tools = tmp_path / "tools.json"
    tools.write_text(json.dumps(TOOLS))
    work = tmp_path / "work"
    work.mkdir()

    session = Terminal(["init", "--model", str(model)], work, env)
    session.expect("That's fine: Tensward starts its own copy to measure.")
    session.expect("[Enter] engine defaults")
    session.send("")
    session.expect("[e] edit engine/model/baseline")
    session.send("")
    session.expect("Workload [chat]:")
    session.send("9")
    session.expect("type a number from 1 to 6")
    session.send("3")
    session.expect("Tool calls [yes")
    session.send("3")
    session.expect("tools file")
    session.send("missing.json")
    session.expect("does not exist")
    session.send(str(tools))
    session.expect("Peak concurrent requests?")
    session.send("")
    session.expect("Goal [TTFT]:")
    session.send("")
    session.expect("Run the analysis now?")
    session.send("")
    status = session.finish()
    assert status == 0, session.text
    transcript = session.text
    for expected in (
        "Engine:   vLLM 0.30.0",
        "Hardware: 1× NVIDIA L4",
        "engine defaults",
        "Fingerprinting the model",
        "Project saved in ./.tensward (runs, reports).",
        "tensward init --model",
        "--tools",
        "--yes",
        "Ready after",
        "Ran: vLLM 0.30.0",
    ):
        assert expected in transcript
    project = work / ".tensward"
    assert (project / "runs" / "0001-baseline" / "report.json").is_file()
    choices = json.loads((project / "choices.json").read_text())
    assert (choices["goal"], choices["workload"]) == ("ttft", "agent")

    before = {name: (project / name).read_bytes() for name in ("kept.json", "rejected.json")
              if (project / name).exists()}  # fmt: skip
    interrupted = subprocess.Popen(
        [sys.executable, "-m", "tensward.cli", "try", "--yes"],
        cwd=work,
        env={**env, "FAKE_LOAD_SECONDS": "60"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert interrupted.stdout is not None
    seen = []
    for line in interrupted.stdout:
        seen.append(line)
        if "loading the model" in line:
            break
    else:
        pytest.fail(f"try ended before loading the model ({interrupted.wait()}):\n{''.join(seen)}")
    deadline = time.monotonic() + 30
    served: list[Path] = []
    while not served and time.monotonic() < deadline:
        served = [p for p in (project / "runs").glob("*/serve.json")
                  if p.parent.name != "0001-baseline"]  # fmt: skip
        time.sleep(0.05)
    assert served, "the step's server never recorded itself"
    interrupted.send_signal(signal.SIGINT)
    code = interrupted.wait(timeout=60)
    assert code == 130, code and interrupted.stdout.read()
    for path in served:
        pid = json.loads(path.read_text())["identity"]["pid"]
        assert subprocess.run(["pgrep", "-P", str(pid)], capture_output=True).returncode == 1
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    after = {name: (project / name).read_bytes() for name in ("kept.json", "rejected.json")
             if (project / name).exists()}  # fmt: skip
    assert after == before

    tried = tensward(["try", "--yes"], work, env, timeout=120)
    assert "Tried:" in tried.stdout, tried.stdout + tried.stderr
    kept = "Kept in this project" in tried.stdout
    assert kept or "Not kept" in tried.stdout
    assert tried.returncode == (0 if kept else 3)
    listed = tensward(["runs"], work, env).stdout
    assert "0001-baseline" in listed and "0003-" in listed
    if kept:
        exported = tensward(["export", "--format", "compose"], work, env).stdout
        assert "--" in exported and "vllm/vllm-openai" in exported
        assert "Undone:" in tensward(["undo"], work, env).stdout
    assert tensward(["undo"], work, env).returncode == 1

    other = tmp_path / "work2"
    other.mkdir()
    flags = ["--model", str(model), "--workload", "chat", "--tools", "none", "--concurrency", "4"]
    flags += ["--goal", "throughput", "--engine-defaults", "--yes", "--no-run", "--json"]
    done = tensward(["init", *flags], other, env)
    assert done.returncode == 0, done.stderr
    assert "snapshot_id" in json.loads(done.stdout)
    assert "Fingerprinting" not in done.stdout + done.stderr

    refused = tensward(
        ["init", "--model", str(model), "--workload", "code", "--yes", "--no-run"], work, env
    )
    assert refused.returncode == 3
    for expected in ("[project_inputs_changed]", "--force"):
        assert expected in refused.stderr
    assert tensward(["inspect", "--json"], work, env).returncode == 0

    forced = tensward(
        ["init", "--model", str(model), "--workload", "code", "--yes", "--no-run", "--force"],
        work,
        env,
    )
    assert forced.returncode == 0, forced.stderr
    archives = list(work.glob(".tensward.archive-*"))
    assert len(archives) == 1 and (archives[0] / "runs" / "0001-baseline").is_dir()
    assert not (work / ".tensward" / "runs").exists()
    assert json.loads((work / ".tensward" / "choices.json").read_text())["workload"] == "code"


def test_agent_path_never_waits(tmp_path: Path) -> None:
    env = machine(tmp_path)
    model = checkpoint(tmp_path)
    work = tmp_path / "work"
    work.mkdir()

    bare = tensward(["init"], work, env)
    assert bare.returncode == 3
    assert "--model" in bare.stderr and "--workload" in bare.stderr

    as_json = tensward(["init", "--json"], work, env)
    assert json.loads(as_json.stderr.strip().splitlines()[-1])["code"] == "project_inputs_invalid"

    done = tensward(["init", "--model", str(model), "--json", "--yes", "--no-run"], work, env)
    assert done.returncode == 0, done.stderr
    assert len(done.stdout.strip().splitlines()) == 1
    assert "snapshot_id" in json.loads(done.stdout)


def test_the_printed_like_command_reaches_the_first_analysis_on_enter(tmp_path: Path) -> None:
    env = machine(tmp_path)
    model = checkpoint(tmp_path)
    serve = f"vllm serve {model} --enable-auto-tool-choice --tool-call-parser hermes"
    base = tmp_path / "base"
    base.mkdir()
    flags = ["--workload", "agent", "--tools", "default", "--concurrency", "3", "--goal", "tpot"]
    done = tensward(
        ["init", "--model", str(model), *flags, "--current", serve, "--yes", "--no-run"],
        base,
        env,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    new = tmp_path / "new"
    new.mkdir()
    session = Terminal(["init", "--current", serve, "--like", str(base / ".tensward")], new, env)
    session.expect("[Enter] continue")
    session.send("")
    session.expect("Run the analysis now?")
    session.send("")
    assert session.finish() == 0, session.text
    assert "Done: " in session.text and "Next:" in session.text
    assert (new / ".tensward" / "choices.json").read_text() == (
        base / ".tensward" / "choices.json"
    ).read_text()
