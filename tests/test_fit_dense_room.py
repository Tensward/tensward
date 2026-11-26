from dataclasses import replace
from pathlib import Path

import pytest

from tensward.anatomy import AttentionGroup, ModelAnatomy
from tensward.engines.llamacpp.engine import LLAMACPP, resolution
from tensward.engines.llamacpp.flags import HelpManifest
from tensward.engines.llamacpp.prelaunch import fit_advice
from tensward.engines.llamacpp.startup import read_startup
from tensward.platforms.cpu import host_facts
from tensward.settings import Settings

GIB = 1 << 30


def test_a_dense_model_whose_kv_cache_cannot_fit_is_refused_with_an_offer() -> None:
    """Qwen3-32B Q4_K_M on a 22.5 GiB L4 with `-np 4 -c 32768 --fit off`: the weights take 18.0
    GiB and the f16 KV cache 8.0 GiB; the launch died of CUDA out-of-memory allocating the KV."""
    layer = int(17.5 * GIB / 64)
    anatomy = ModelAnatomy(
        "qwen3", True, 5120, 64, 151936, False, None, None, {},
        (AttentionGroup("full", 64, 8, 128, None, False),), None, None, (),
        weight_bytes_by_role={"layers": 64 * layer, "output": int(0.5 * GIB)},
        layer_bytes=(layer,) * 64, expert_layer_bytes=(0,) * 64,
    )  # fmt: skip
    settings = replace(
        Settings(), max_concurrent_requests=4, max_context_len=32768, extra_args={"--fit": "off"}
    )
    host = replace(host_facts(), mem_available_gib=48.0)
    advice = fit_advice(
        anatomy, settings, "NVIDIA L4", int(22.5 * GIB), host, None, 40960,
        lambda proposed: f"llama-server --n-gpu-layers {proposed.extra_args['--n-gpu-layers']}",
        None,
    )  # fmt: skip
    assert advice is not None
    text, fails = advice
    assert fails and "the KV cache 8.0 GiB for 32768 tokens" in text
    assert "`--n-gpu-layers " in text and "Try: `llama-server --n-gpu-layers " in text


def test_a_forced_ngl_on_a_build_with_fit_is_offered_the_engines_own_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qwen3-32B Q4_K_M on an L4 with `-ngl 99`: a layer count worked out here (45 of 65) placed
    fewer layers than llama.cpp's --fit does (51 of 65, +31% req/s), so the offer is to drop the
    forced count."""
    layer = int(17.5 * GIB / 64)
    anatomy = ModelAnatomy(
        "qwen3", True, 5120, 64, 151936, False, None, None, {},
        (AttentionGroup("full", 64, 8, 128, None, False),), None, None, (),
        weight_bytes_by_role={"layers": 64 * layer, "output": int(0.5 * GIB)},
        layer_bytes=(layer,) * 64, expert_layer_bytes=(0,) * 64,
    )  # fmt: skip
    settings = replace(
        Settings(), max_concurrent_requests=4, max_context_len=32768,
        extra_args={"--n-gpu-layers": 99},
    )  # fmt: skip
    host = replace(host_facts(), mem_available_gib=48.0)
    offered: list[Settings] = []

    def command(proposed: Settings) -> str:
        offered.append(proposed)
        return "llama-server"

    def advise() -> str:
        found = fit_advice(
            anatomy, settings, "NVIDIA L4", int(22.5 * GIB), host, None, 40960, command, None
        )
        assert found is not None
        return found[0]

    monkeypatch.setattr(
        "tensward.engines.llamacpp.prelaunch.known_build",
        lambda: (None, HelpManifest(frozenset({"--fit"}))),
    )
    text = advise()
    assert "Drop `--n-gpu-layers 99` and let llama.cpp place the layers" in text
    assert offered[0].extra_args["--n-gpu-layers"] == "auto"
    assert "--n-gpu-layers 45" not in text
    monkeypatch.setattr("tensward.engines.llamacpp.prelaunch.known_build", lambda: None)
    assert "`--n-gpu-layers 4" in advise()


@pytest.mark.parametrize(("cpu_moe", "fits"), [(18, True), (17, False)])
def test_an_explicit_expert_offload_is_refused_only_where_it_runs_out_of_memory(
    cpu_moe: int, fits: bool
) -> None:
    """Qwen3.5-35B-A3B Q8_0 on an L4 (b11669) with `-c 16384 -np 2 -ub 4096 --n-cpu-moe N`: N=18
    ran at 22,301 MiB of 23,034 and N=17 ran out of memory at its first long prompt. A compute
    buffer of 576 MiB scaled by -ub had refused N=18 at 24.4 GiB."""
    mib = 1 << 20
    expert, layer = 816 * mib, int(866.7 * mib)
    anatomy = ModelAnatomy(
        "qwen35moe", True, 2048, 40, 248320, False, None, None, {},
        (AttentionGroup("full", 10, 2, 256, None, False),), None, None, (),
        state_bytes_per_seq=int(125.6 * mib),
        weight_bytes_by_role={"layers": 40 * layer},
        layer_bytes=(layer,) * 40, expert_layer_bytes=(expert,) * 40,
    )  # fmt: skip
    settings = replace(
        Settings(), max_concurrent_requests=2, max_context_len=16384,
        extra_args={
            "--fit": "off", "--n-gpu-layers": "all", "--n-cpu-moe": str(cpu_moe),
            "--ubatch-size": "4096", "--batch-size": "4096",
        },
    )  # fmt: skip
    host = replace(host_facts(), mem_available_gib=60.0)
    advice = fit_advice(anatomy, settings, "NVIDIA L4", 23034 * mib, host, None, 262144, None, None)
    assert (advice is None) == fits


def test_glm_fit_placement_is_held_with_the_dense_blocks_counted() -> None:
    """GLM-5.3-Flash UD-Q2_K_XL (3 dense blocks, 42 expert blocks, 1 next-token block) on an
    80 GB H100: --fit logged "46 layers (12 overflowing)", and `--n-cpu-moe 11` ran out of
    device memory (82,888 MiB); `-ngl all --n-cpu-moe 14` starts."""
    log = (Path(__file__).parent / "fixtures" / "glm-fit-h100.server.log").read_text()
    started = read_startup(log)
    expert = 2214 * 2**20
    anatomy = ModelAnatomy(
        "glm5-next", True, 4096, 46, 154880, False, None, None, {}, (), None, None, (),
        layer_bytes=(GIB,) * 3 + (expert + GIB,) * 43,
        expert_layer_bytes=(0,) * 3 + (expert,) * 43, mtp_layers=1,
    )  # fmt: skip
    settings = replace(Settings(), extra_args={"--fit": "on"})
    resolved = resolution(started, {}, settings, anatomy)
    assert resolved is not None and resolved.levers["host_expert_layers"] == 14
    held = LLAMACPP.repinned(settings, resolved)
    assert held is not None and held.extra_args["--fit"] == "off"
    assert held.extra_args["--n-cpu-moe"] == "14"


def test_a_partly_moved_block_is_not_held_as_a_whole_one() -> None:
    """Qwen3.8-Flash-Next UD-Q4_K_XL on an 80 GB H100, blocks of uneven expert size: --fit
    logged "49 layers (4 overflowing)", two whole blocks and part of a third; `--n-cpu-moe 3`
    starts."""
    log = (Path(__file__).parent / "fixtures" / "flash-next-fit-h100.server.log").read_text()
    mib = [1500] * 48
    mib[2], mib[4], mib[30], mib[46], mib[47] = 1950, 1750, 1750, 1750, 1750
    anatomy = ModelAnatomy(
        "qwen3next", True, 2048, 48, 151936, False, None, None, {}, (), None, None, (),
        layer_bytes=tuple((m + 200) * 2**20 for m in mib),
        expert_layer_bytes=tuple(m * 2**20 for m in mib),
    )  # fmt: skip
    settings = replace(Settings(), extra_args={"--fit": "on"})
    resolved = resolution(read_startup(log), {}, settings, anatomy)
    assert resolved is not None and resolved.levers["host_expert_layers"] == 3
