"""Fit verdicts on the real Gemma 4 26B-A4B AWQ anatomy (17.2 GB of weights)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from test_anatomy import checkpoint_anatomy, gemma

from tensward.anatomy import Components
from tensward.engines.vllm import VLLM
from tensward.fit import Fit, estimate_fit
from tensward.platforms import Device as GpuInfo
from tensward.settings import Settings

L4 = (GpuInfo("NVIDIA L4", 23034 * 2**20, 300 * 2**20),)
T4 = (GpuInfo("Tesla T4", 15360 * 2**20, 0),)
OVERHEAD = VLLM.memory_overhead_bytes


def fit(gpus: tuple[GpuInfo, ...], **settings: int) -> Fit:
    batch = settings.get("prefill_batch_tokens") or 2048
    return estimate_fit(gemma(), Settings(**settings), concurrency=4, avg_tokens=3000,
                        gpus=gpus, selected=(), default_fraction=0.92, overhead_bytes=OVERHEAD,
                        in_flight_tokens=2 * batch, loads_encoders=True)  # fmt: skip


def test_the_26b_model_fits_an_l4_but_not_a_t4() -> None:
    l4 = fit(L4, max_context_len=8192)
    assert l4.verdict in ("fits", "tight")
    assert l4.max_context_tokens is not None and l4.max_context_tokens >= 8192
    t4 = fit(T4, max_context_len=8192)
    assert t4.verdict == "likely does not fit" and "smaller" in t4.reason


def test_the_l4_estimate_reproduces_vllms_capacity() -> None:
    l4 = fit(L4, max_context_len=32768, prefill_batch_tokens=2496)
    assert l4.capacity_tokens == pytest.approx(40416, rel=0.05)  # vLLM 0.30 on an L4


def test_the_a10g_estimate_with_media_inputs_reproduces_vllms_capacity() -> None:
    settings = Settings(max_context_len=8192)
    in_flight = VLLM.kv_in_flight_tokens(settings, takes_images=True)
    a10g = estimate_fit(gemma(), settings, concurrency=4, avg_tokens=3000,
                        gpus=(GpuInfo("NVIDIA A10G", 23028 * 2**20, 0),), selected=(),
                        default_fraction=0.92, overhead_bytes=OVERHEAD,
                        in_flight_tokens=in_flight, loads_encoders=True)  # fmt: skip
    assert a10g.capacity_tokens == pytest.approx(13839, rel=0.05)  # vLLM 0.30 on an A10G


def test_fit_without_a_declared_concurrency_checks_one_sequence() -> None:
    one = estimate_fit(gemma(), Settings(max_context_len=8192), concurrency=None,
                       avg_tokens=3000, gpus=L4, selected=(), default_fraction=0.92,
                       overhead_bytes=OVERHEAD, loads_encoders=True)  # fmt: skip
    assert one.verdict != "not checked" and "one sequence" in one.reason


def test_no_gpu_or_several_gpus_is_not_checked() -> None:
    assert fit(()).verdict == "not checked"
    two = estimate_fit(gemma(), Settings(), concurrency=1, avg_tokens=10, gpus=L4 * 2,
                       selected=("0", "1"), default_fraction=0.92,
                       overhead_bytes=OVERHEAD, loads_encoders=True)  # fmt: skip
    assert two.verdict == "not checked"


def test_the_context_that_exactly_fills_the_pool_fails_for_the_null_block() -> None:
    anatomy = checkpoint_anatomy("gpt_oss_20b", weight_bits=4)
    assert anatomy.components is not None
    weights = sum(getattr(anatomy.components, name) for name in Components.__slots__)
    context = 4096
    usable = L4[0].total_bytes
    overhead = (
        usable - weights - (anatomy.kv_bytes(context, "auto") or 0)
    )  # the pool is the context

    def at(context_limit: int) -> Fit:
        return estimate_fit(anatomy, Settings(max_context_len=context_limit), concurrency=1,
                            avg_tokens=100, gpus=L4, selected=(), default_fraction=1.0,
                            overhead_bytes=overhead, loads_encoders=True)  # fmt: skip

    edge = at(context)
    assert edge.max_context_tokens is not None and edge.max_context_tokens < context
    assert edge.verdict == "tight" and "refuses to start" in edge.reason
    assert at(context - 64).max_context_tokens == context - 64  # a few blocks less is enough


def test_a_linear_attention_int4_model_that_leaves_little_beyond_the_weights_is_tight() -> None:
    parts = Components(  # Qwen3.8 27B W4A16 (A-run2 init)
        text_dense=12596073792, embedding=2542796800, lm_head=2542796800, routed_experts=0,
        shared_experts=0, vision=921460192, audio=0, unclassified=0, mtp=849398784,
    )  # fmt: skip
    anatomy = replace(
        gemma(), components=parts, attention=(), vision=None, moe=None,
        unavailable=("layers with a state for each sequence are not modelled",),
    )  # fmt: skip
    l4 = estimate_fit(anatomy, Settings(), concurrency=4, avg_tokens=3000, gpus=L4, selected=(),
                      default_fraction=0.92, overhead_bytes=OVERHEAD, in_flight_tokens=4096,
                      loads_encoders=True)  # fmt: skip
    assert l4.verdict == "tight" and "--enforce-eager" in l4.reason
    assert l4.weights_bytes == sum(parts.__getattribute__(n) for n in Components.__slots__)


def test_vllm_offers_a_context_that_starts_when_the_full_one_cannot() -> None:
    """Qwen3.8-27B INT4 on an L4 (vLLM 0.30): 17.71 GiB of weights, 0.1 GiB left of the 20.7 GiB
    usable, default context 262144; the CUDA graph profile died allocating 12.25 GiB of KV."""
    from types import SimpleNamespace

    from tensward.project import _context_advice, _startup_context

    gib = 2**30
    fit = Fit(
        "tight", "", "NVIDIA L4", int(22.5 * gib), 0, int(20.7 * gib), int(18.1 * gib),
        int(2.5 * gib), int(0.1 * gib),
    )  # fmt: skip
    resolved = SimpleNamespace(
        record=SimpleNamespace(
            artifact=SimpleNamespace(metadata=SimpleNamespace(context_limit=262144))
        ),
        prompts=(SimpleNamespace(text="x" * 12000),),
        config=SimpleNamespace(workload=SimpleNamespace(output_tokens=512)),
    )
    context = _startup_context(resolved, Settings(), fit)
    assert context == 3 * 1024 * 2  # (3000 + 512) * 1.5 rounded up to 1024
    text, certain = _context_advice(VLLM, Settings(), fit, context, None)
    assert certain and "--max-model-len 6144" in text and "--enforce-eager" in text
    assert _startup_context(resolved, Settings(max_context_len=4096), fit) is None


GIB = 2**30
QWEN_FIT = Fit(
    "tight", "", "NVIDIA L4", int(22.5 * GIB), 0, int(20.7 * GIB), int(18.1 * GIB),
    int(2.5 * GIB), int(0.1 * GIB),
)  # fmt: skip


def qwen_project():
    from types import SimpleNamespace

    return SimpleNamespace(
        anatomy=replace(
            gemma(),
            layers=64,
            state_layers=48,
            state_bytes_per_seq=153944064,
            state_kv_token_bytes=2048,
        ),
        record=SimpleNamespace(
            artifact=SimpleNamespace(metadata=SimpleNamespace(context_limit=262144))
        ),
        prompts=(SimpleNamespace(text="x" * 12000),),
        config=SimpleNamespace(
            workload=SimpleNamespace(
                arrival=SimpleNamespace(kind="closed_loop", concurrency=8), output_tokens=512
            )
        ),
    )


def test_vllm_offers_the_workloads_concurrency_when_the_graph_profile_cache_cannot_fit() -> None:
    """Qwen3.8-27B INT4 on an L4: vLLM 0.30 allocated 256 blocks x 16 layers x 3.06 MiB =
    12.25 GiB to capture graphs, at --max-model-len 4096 as well."""
    from tensward.project import _profile_pool_advice, _profile_pool_bytes

    resolved = qwen_project()
    assert abs(_profile_pool_bytes(resolved.anatomy, 256) / GIB - 12.25) < 0.05
    text, certain = _profile_pool_advice(resolved, VLLM, Settings(), QWEN_FIT, None)
    assert certain and "`--max-num-seqs 8`" in text and "12.2 GiB" in text
    kept = Settings(max_concurrent_requests=8, max_context_len=4096, kv_memory_fraction=0.95)
    assert _profile_pool_advice(resolved, VLLM, kept, QWEN_FIT, None) is None


def test_a_rejected_fit_change_moves_the_offer_to_the_next_one() -> None:
    from tensward.project import _profile_pool_advice

    resolved = qwen_project()
    args = (resolved, VLLM, Settings(), QWEN_FIT, None)
    text, _ = _profile_pool_advice(*args, {"max-num-seqs"})
    assert "`--enforce-eager`" in text
    assert _profile_pool_advice(*args, {"max-num-seqs", "enforce-eager"}) is None


def test_the_hybrid_int4_offer_names_both_changes_and_passes_the_pre_flight() -> None:
    """Qwen3.8-27B INT4 on an L4: --max-num-seqs 8 alone left the full-context KV pool check
    to fail, and tensward's own pre-flight refused it."""
    from tensward.project import _profile_pool_advice

    resolved = qwen_project()
    commands: list[Settings] = []
    text, _ = _profile_pool_advice(
        resolved, VLLM, Settings(), QWEN_FIT, lambda s: commands.append(s) or ""
    )
    assert "--max-model-len 4096" in text
    assert (
        commands[0].max_concurrent_requests == 8
        and commands[0].max_context_len == 4096
        and commands[0].kv_memory_fraction == 0.95
    )
    assert _profile_pool_advice(resolved, VLLM, commands[0], QWEN_FIT, None) is None
    from tensward.project import OFFER_WORDS, fit_offer_name

    name = fit_offer_name(("max-num-seqs=8", "max-model-len=4096"))
    assert OFFER_WORDS[name] == "fewer sequences"


def test_hybrid_request_bytes_match_the_vllm_log_and_a_model_that_cannot_hold_one_is_said_so() -> (
    None
):
    """Qwen3.8-27B INT4, run 0001: a 4096-token request needs 0.43 GiB (0.48 with the null
    block); vLLM had 0.37 GiB for the pool where the model predicted 0.1."""
    from tensward.project import _hybrid_request_bytes, _profile_pool_advice

    resolved = qwen_project()
    assert abs(_hybrid_request_bytes(resolved.anatomy, 4096) / GIB - 0.478) < 0.01
    small = replace(QWEN_FIT, kv_available_bytes=-int(0.5 * GIB))
    text, certain = _profile_pool_advice(resolved, VLLM, Settings(), small, lambda s: "")
    assert certain and text.startswith("This model does not fit this GPU usefully under vLLM")
    assert "Try:" not in text and "llama.cpp with CPU offload" in text
