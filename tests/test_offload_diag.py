"""Offloaded experts: the diagnosis names the host-memory limit and the offers move it."""

from __future__ import annotations

from tensward.anatomy import AttentionGroup, ModelAnatomy
from tensward.classify import EngineRule, Finding, classify
from tensward.engines.llamacpp.placement import (
    DecodeModel,
    Fit2,
    Placement,
    Regime,
    next_steps,
)
from tensward.engines.vllm import VllmEngine
from tensward.measurement import Measurement
from tensward.settings import Settings
from tensward.stands import stands_rows, where_it_stands

GIB = 1 << 30
LAYER = int(0.85 * GIB)  # one layer's experts, Qwen3.5-35B-A3B Q8 on an L4


def anatomy() -> ModelAnatomy:
    return ModelAnatomy(
        "qwen3moe", True, 2048, 40, 248000, False, 256, 8, {},
        (AttentionGroup("full", 40, 2, 256, None, False),), None, None, (),
        weight_bytes_by_role={"layers": 40 * LAYER},
        layer_bytes=(LAYER + 100_000_000,) * 40, expert_layer_bytes=(LAYER,) * 40,
    )  # fmt: skip


def held(room: int, kv: int) -> list:
    placed = Placement(
        device_weight_bytes=21 * GIB, host_weight_bytes=18 * LAYER, host_expert_bytes=18 * LAYER,
        host_layers=0, host_expert_layers=18, kv_on_host=False,
        device_bytes_per_token=2_700_000_000, host_bytes_per_token=480_000_000,
    )  # fmt: skip
    fit = Fit2(
        device_need=22 * GIB, device_free=22 * GIB + room, host_need=15 * GIB, host_free=60 * GIB,
        fits=True, reason="", kv_bytes=kv, compute_buffer_bytes=600_000_000, pool_tokens=16384,
        ubatch=512, slots=4, need_tokens=2050, peak_running=4,
    )  # fmt: skip
    model = DecodeModel(300, 35, 30.0, 4, False)
    regime = Regime("expert_offload", "mixed", False, False)
    steps = next_steps(
        anatomy(), placed, model, fit, regime, 97.7, "", False, topology="discrete"
    )  # fmt: skip
    return list(steps)


def test_a_held_placement_is_offered_only_the_room_that_admits_experts() -> None:
    """The README run left 0.5 GB free beside 0.3 GiB of KV: no single move admits an expert
    layer, so nothing is offered. With 2 GiB of KV, a q8_0 cache admits one."""
    assert held(500_000_000, 335_544_320) == []
    steps = [s for s in held(500_000_000, 2 * GIB) if s.lever == "kv_cache_precision"]
    assert steps and steps[0].cpu_moe == 17 and "--n-cpu-moe 17" in steps[0].why


def test_offloaded_experts_name_the_limit_before_the_load_does() -> None:
    reading = Finding(
        "offload", "warning", "limited by reading the CPU-offloaded experts each step"
    )

    class Modelled(VllmEngine):
        engine_rules = (EngineRule(bottleneck="offload", rule=lambda _: reading),)

    engine = Modelled()
    m = Measurement(
        160, 0, queue_share=0.0, peak_kv_usage=0.1, preemptions=0.0, peak_running=4.0,
        peak_waiting=0.0, mean_in_flight=4.0, peak_in_flight=4, ttft_p50_ms=2800.0,
        tpot_p50_ms=97.7, tpot_p95_ms=155.5,
    )  # fmt: skip
    diagnosis = classify(m, Settings(max_concurrent_requests=4), None, engine=engine)
    assert diagnosis.primary == "offload"


def test_where_it_stands_reads_the_measured_signals_and_names_the_rest() -> None:
    run = {
        "diagnosis": {
            "primary": "offload", "requests": 160,
            "gpu_state": {"utilization": 39.4, "memory_peak_mib": 21402.0,
                          "memory_total_mib": 23034.0},
        },
        "ceilings": {"decode_pct_of_ceiling": 20.5},
        "placement": {
            "regime": {"placement": "expert_offload"},
            "placement": {"host_expert_layers": 18, "host_layers": 0,
                          "host_weight_bytes": 15_401_481_666},
            "environment": {"mem_gib": 62.8, "mem_available_gib": 60.2},
        },
    }  # fmt: skip
    line = where_it_stands(run)
    assert "GPU active 39% of the time" in line and "decode at 20% of its ceiling" in line
    assert "VRAM 20.9 of 22.5 GiB used (1.6 free)" in line
    assert "host RAM 2.6 of 62.8 GiB used" in line and "experts of 18 layers on the CPU" in line
    assert line.endswith("limited by reading the CPU-offloaded experts each step")
    bare = where_it_stands({"diagnosis": {"primary": None, "requests": 5}})
    assert "VRAM not measured" in bare and "host RAM not measured" in bare
    after = {**run, "ceilings": {"decode_pct_of_ceiling": 31.0}}
    assert ("Decode vs ceiling", "20%", "31%", "+10 pts") in stands_rows(run, after)
