from dataclasses import replace

import pytest

from tensward.anatomy import AttentionGroup, ModelAnatomy
from tensward.engines.llamacpp.flags import HelpManifest
from tensward.engines.llamacpp.prelaunch import fit_advice
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
