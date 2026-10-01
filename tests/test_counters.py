"""Metric groups: each is one ncu run, and a group that failed stays unknown in the merge."""

from pathlib import Path

import pytest
from real_ncu import MARLIN, real_a10g_marlin

from tensward.counters import (
    GROUPS,
    SelectedKernel,
    function_name,
    group_command,
    kernel_counters,
    match_ncu_name,
    merge_groups,
    parse_raw_csv,
    probe_failure_reason,
)

REAL_L4 = Path(__file__).with_name("ncu_raw_probe_l4.csv")


def test_groups_partition_the_metrics_and_each_run_asks_for_its_own_group_and_passes() -> None:
    keys = [key for group in GROUPS.values() for key in group]
    assert len(keys) == len(set(keys))
    kernel = SelectedKernel("marlin::Marlin<256, 4>", 0.6, 10)
    memory = group_command("ncu", [kernel], Path("log"), "memory")
    assert memory[memory.index("--kernel-name") + 1] == "regex:^(Marlin)$"
    metrics = memory[memory.index("--metrics") + 1]
    assert ["--profile-from-start", "off"] == memory[memory.index("--profile-from-start") :][:2]
    assert metrics == "dram__throughput.avg.pct_of_peak_sustained_elapsed,profiler__replayer_passes"


def test_merged_groups_keep_their_values_and_a_failed_group_stays_unknown() -> None:
    (real,) = parse_raw_csv(REAL_L4.read_text())
    assert real.values["passes"] == 8  # the full collection this split replaces
    selected = SelectedKernel(real.name, 0.6, 10)
    occupancy = kernel_counters(selected, "x", [real])  # a fixture row has every column
    memory = kernel_counters(selected, "x", [], "ncu exited 1")  # ran, wrote nothing
    merged = merge_groups(
        selected, "x", {"occupancy": occupancy, "memory": memory}, {"tensor": "needed 2 passes"}
    )
    assert merged.value("achieved_occupancy_pct") == pytest.approx(15.75)
    assert merged.value("dram_pct") is None and merged.value("tensor_pct") is None
    assert merged.unavailable == {
        "memory": "ncu exited 1",
        "tensor": "needed 2 passes",
    }
    assert merge_groups(selected, "x", {"memory": memory}, {}).values == {}


FLASH = (
    "void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, "
    "cutlass::half_t, Flash_kernel_traits<128, 64, 128, 4, cutlass::half_t> >, true, false, "
    "false, false, true, false, false, false>(flash::Flash_fwd_params)"
)  # real
FLASH_OTHER = FLASH.replace("true, false, false, false, true", "false, false, false, false, true")
# Real ncu spellings from an A10G: no "0/1" vs "false", no "l" suffixes, "const T *" parameters,
# and the namespace present in a gated run but absent in a startup run.
MARLIN_NCU = (
    "void marlin::Marlin<1125899906910725, 1125899906843648, 1125899906910725, 1125899906910725, "
    "256, 4, 16, 4, 0, 4, 8, 0>(const int4 *, const int4 *, int4 *, bool, int)"
)
MARLIN_NCU_STARTUP = MARLIN_NCU.replace("marlin::Marlin", "Marlin")
FLASH_NCU = (
    "void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, 0, 0, "
    "cutlass::half_t, Flash_kernel_traits<128, 64, 128, 4, cutlass::half_t>>, 1, 0, 0, 0, 1, 0, "
    "0, 0>(flash::Flash_fwd_params)"
)


@pytest.mark.parametrize(
    ("trace", "ncu_names", "matched", "note"),
    [
        (MARLIN, [MARLIN_NCU, "other_kernel"], MARLIN_NCU, None),
        (MARLIN, [MARLIN_NCU_STARTUP], MARLIN_NCU_STARTUP, None),
        (MARLIN, [MARLIN_NCU.replace("0, 4, 8, 0", "0, 4, 16, 0"), MARLIN_NCU], MARLIN_NCU, None),
        (FLASH, [FLASH_NCU, FLASH_NCU.replace(", 1, 0", ", 0, 0", 1)], FLASH_NCU, None),
        (FLASH_OTHER, [FLASH_NCU], None, "not seen by ncu; ncu printed for this function: flash"),
        (MARLIN, [], None, "ncu printed for this run (other functions): nothing"),
        (MARLIN, [FLASH_NCU], None, "other functions): flash::flash_fwd_splitkv_kernel<"),
        (MARLIN, [MARLIN_NCU, MARLIN_NCU_STARTUP], None, "ambiguous"),
    ],
)  # fmt: skip
def test_trace_kernels_match_the_names_ncu_prints_uniquely_or_not_at_all(
    trace: str, ncu_names: list[str], matched: str | None, note: str | None
) -> None:
    assert function_name(trace) in {"Marlin", "flash_fwd_splitkv_kernel"}
    found, why = match_ncu_name(trace, ncu_names)
    assert found == matched
    assert (why is None) if note is None else (note in why)


def test_real_a10g_groups_merge_into_one_prefill_marlin_kernel() -> None:
    kernel = real_a10g_marlin()
    assert kernel.value("grid") == 80 and kernel.value("sm_count") == 80
    assert kernel.value("achieved_occupancy_pct") == pytest.approx(16.66)
    assert kernel.value("eligible_warps") == pytest.approx(0.15)
    assert kernel.value("issue_active_pct") == pytest.approx(11.95)
    assert kernel.value("tensor_pct") == pytest.approx(45.44)
    assert kernel.value("dram_pct") == pytest.approx(21.37)
    assert kernel.unavailable == {}


def test_a_failed_probe_names_its_cause() -> None:
    log = "==ERROR== ERR_NVGPUCTRPERM - The user does not have permission\n"
    assert "--cap-add SYS_ADMIN" in probe_failure_reason("", log)
    torch = "ModuleNotFoundError: No module named 'torch'"
    assert "same Python environment" in probe_failure_reason(torch, "")
    assert probe_failure_reason("a\nb", "==WARNING== slow") == "==WARNING== slow | a | b"
