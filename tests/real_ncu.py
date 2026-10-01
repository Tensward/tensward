"""The real A10G counter runs (ncu_a10g_*.csv), shared by the tests."""

from pathlib import Path

from tensward.counters import (
    GROUPS,
    KernelCounters,
    SelectedKernel,
    kernel_counters,
    match_ncu_name,
    merge_groups,
    parse_raw_csv,
)

MARLIN = (
    "void marlin::Marlin<1125899906910725l, 1125899906843648l, 1125899906910725l, "
    "1125899906910725l, 256, 4, 16, 4, false, 4, 8, false>(int4 const*, int4 const*, int4*, "
    "int4*, int4 const*, float const*, int4 const*, float const*, int4 const*, int, int, int, "
    "int, int*, bool, bool, bool, int)"
)  # real, from an L4 PyTorch trace


def real_a10g_marlin() -> KernelCounters:
    """The prefill Marlin kernel of the real gated workload on an A10G (80 SMs): its first
    launch in each of the three group runs, matched to the trace's spelling and merged."""
    selected = SelectedKernel(MARLIN, 0.63, 10)
    parts = {}
    for group in GROUPS:
        launches = parse_raw_csv((Path(__file__).parent / f"ncu_a10g_{group}.csv").read_text())
        ncu_name, why = match_ncu_name(MARLIN, [launch.name for launch in launches])
        assert why is None
        first = next(launch for launch in launches if launch.name == ncu_name)
        parts[group] = kernel_counters(selected, ncu_name, [first])
    return merge_groups(selected, parts["occupancy"].ncu_name, parts, {})
