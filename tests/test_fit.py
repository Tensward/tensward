"""Fit verdicts on the real Gemma 4 26B-A4B AWQ anatomy (17.2 GB of weights)."""

from __future__ import annotations

import pytest
from test_anatomy import gemma

from tensward.engines.protocol import Settings
from tensward.engines.vllm import VLLM
from tensward.fit import Fit, GpuInfo, estimate_fit

L4 = (GpuInfo("NVIDIA L4", 23034 * 2**20, 300 * 2**20),)
T4 = (GpuInfo("Tesla T4", 15360 * 2**20, 0),)
OVERHEAD = VLLM.memory_overhead_bytes


def fit(gpus: tuple[GpuInfo, ...], **settings: int) -> Fit:
    batch = settings.get("prefill_batch_tokens") or 2048
    return estimate_fit(gemma(), Settings(**settings), concurrency=4, avg_tokens=3000,
                        gpus=gpus, selected=(), default_fraction=0.92, overhead_bytes=OVERHEAD,
                        in_flight_tokens=2 * batch)  # fmt: skip


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
                        in_flight_tokens=in_flight)  # fmt: skip
    assert a10g.capacity_tokens == pytest.approx(13839, rel=0.05)  # vLLM 0.30 on an A10G


def test_fit_without_a_declared_concurrency_checks_one_sequence() -> None:
    one = estimate_fit(gemma(), Settings(max_context_len=8192), concurrency=None,
                       avg_tokens=3000, gpus=L4, selected=(), default_fraction=0.92,
                       overhead_bytes=OVERHEAD)  # fmt: skip
    assert one.verdict != "not checked" and "one sequence" in one.reason


def test_no_gpu_or_several_gpus_is_not_checked() -> None:
    assert fit(()).verdict == "not checked"
    two = estimate_fit(gemma(), Settings(), concurrency=1, avg_tokens=10, gpus=L4 * 2,
                       selected=("0", "1"), default_fraction=0.92,
                       overhead_bytes=OVERHEAD)  # fmt: skip
    assert two.verdict == "not checked"
