"""The profiling sections of a run report: the hardware ceilings, the GPU timeline of a
profiled launch and its kernel counters."""

from __future__ import annotations

from ..ceilings import Ceilings
from ..profiling.counters import GROUP_OF, CountersSummary, KernelCounters, short_name
from ..profiling.trace import GAP_MIN_US, IDLE_HINT_SHARE, TraceSummary
from .model import Block, Figure, Section, Table, item

# Each "kernel" cell is a short name; the legend gives every column its unit and denominator.
COLUMNS = (
    ("duration_us", "duration us", "{:,.0f}"),
    ("achieved_occupancy_pct", "achieved occ %", "{:.1f}"),
    ("eligible_warps", "eligible warps", "{:.2f}"),
    ("issue_active_pct", "issue %", "{:.1f}"),
    ("dram_pct", "DRAM %", "{:.1f}"),
    ("tensor_pct", "tensor %", "{:.1f}"),
)
LEGEND = (
    "achieved occ: active warps as % of the SM's maximum, averaged over cycles the SM was active",
    "eligible warps: warps ready to issue, per scheduler, per cycle the scheduler was active",
    "issue: issue slots used, % of peak, over cycles the scheduler was active",
    "DRAM: memory throughput as % of peak, over the kernel's elapsed cycles",
    "tensor: tensor-pipe (HMMA) busy, % of peak, over cycles the SM was active",
    "theoretical occ: warps per SM the launch configuration allows, limited by the named resource",
)


def _moe(ceilings: Ceilings) -> list[Block]:
    if ceilings.moe_experts is None:
        return []
    experts, per_token = ceilings.moe_experts
    blocks: list[Block] = [
        item("moe", f"mixture of experts: {per_token} of {experts} experts per token")
    ]
    if ceilings.experts_per_step is not None and ceilings.expert_bytes_per_step is not None:
        blocks.append(
            item(
                "moe_experts_per_step",
                f"at the measured batch a decode step reads about "
                f"{ceilings.experts_per_step:.0f} of {experts} experts per layer "
                f"({ceilings.expert_bytes_per_step / 1e9:.2f} GB), with uniform routing; real "
                "routing overlaps and reads fewer, so measured decode can sit between this and "
                "the ceiling, which assumes each layer reads only "
                f"{per_token} experts. Steps that mix prefill read up to all experts, so the "
                "decode ceiling holds for decode-only steps",
            )
        )
    return blocks


def ceilings_section(ceilings: Ceilings | None) -> Section:
    """The hardware ceilings, always labelled as theoretical upper bounds."""
    title = "Hardware ceilings (theoretical upper bounds, not targets)"
    if ceilings is None:
        return Section(id="ceilings", title=title, blocks=())
    if ceilings.unavailable:
        unavailable = item("unavailable", f"ceilings: {ceilings.unavailable}")
        return Section(id="ceilings", title=title, blocks=(unavailable,))

    def show(key: str, label: str, value: float | None, unit: str, digits: int = 0) -> Figure:
        return Figure(key=key, label=label, value=value, unit=unit, digits=digits, grouped=True)

    if ceilings.tensor_tflops is None:
        tensor = (
            f"dense tensor rate: unavailable (no published dense tensor rate for {ceilings.gpu})"
        )
    else:
        tensor = f"dense tensor rate: {ceilings.tensor_tflops:g} TFLOPS (datasheet)"
    # Set whenever ceilings are available (the unavailable case returned above).
    assert ceilings.weight_bytes_per_step is not None
    cross = ""
    if ceilings.derived_bandwidth_gbs:
        cross = f"; device-derived cross-check {ceilings.derived_bandwidth_gbs:,.1f} GB/s"
    kv = ceilings.kv_bytes_per_sequence
    blocks = (
        item("gpu", f"GPU: {ceilings.gpu}"),
        item(
            "bandwidth",
            f"memory bandwidth: {ceilings.bandwidth_gbs:,.1f} GB/s "
            f"(bandwidth: {ceilings.bandwidth_source}{cross})",
        ),
        item("tensor_rate", tensor),
        item(
            "weights_per_step",
            f"per decode step (one sequence): {ceilings.weight_bytes_per_step / 1e9:.2f} GB of "
            "weights (the vision encoder and the embedding gather are not read)",
        ),
        show(
            "kv_per_sequence",
            "KV cache read per sequence at the average context",
            None if kv is None else kv / 2**20,
            " MiB",
            1,
        ),
        *_moe(ceilings),
        show("avg_context", "average context per sequence", ceilings.avg_context_tokens, " tokens"),
        show("avg_batch", "average running batch", ceilings.avg_running_batch, " sequences", 1),
        show(
            "decode_ceiling_batch1",
            "decode ceiling, one sequence",
            ceilings.decode_ceiling_batch1_tok_s,
            " tok/s",
        ),
        show(
            "decode_ceiling",
            "decode ceiling at the measured batch",
            ceilings.decode_ceiling_tok_s,
            " tok/s",
        ),
        show("decode_measured", "decode measured", ceilings.measured_decode_tok_s, " tok/s"),
        show("prefill_ceiling", "prefill ceiling", ceilings.prefill_ceiling_tok_s, " tok/s"),
        show("prefill_measured", "prefill measured", ceilings.measured_prefill_tok_s, " tok/s"),
        show(
            "decode_share",
            "decode, share of its ceiling",
            ceilings.decode_pct_of_ceiling,
            "%",
            1,
        ),
        show(
            "prefill_share",
            "prefill, share of its ceiling",
            ceilings.prefill_pct_of_ceiling,
            "%",
            1,
        ),
        show(
            "window_share",
            "window time the tokens need at both ceilings",
            ceilings.ceiling_time_pct_of_window,
            "%",
            1,
        ),
        *(
            [
                item(
                    "exceeds_bound",
                    f"{', '.join(ceilings.exceeds_bound)} share exceeds the theoretical bound: "
                    "the measurement or the ceiling assumption is wrong, so it is not shown",
                )
            ]
            if ceilings.exceeds_bound
            else []
        ),
        item(
            "shared_gpu",
            "prefill and decode share the GPU, so each share alone understates how busy it "
            "was; the last line combines them. Attention FLOPs and activation traffic are "
            "ignored.",
        ),
    )
    return Section(id="ceilings", title=title, blocks=blocks)


def trace_section(summary: TraceSummary, *, analysis_available: bool) -> Section:
    """The headline facts of the profiled launch."""
    title = "GPU timeline (profiled - diagnostic only)"
    if summary.status == "unavailable":
        blocks = (item("unavailable", f"trace unavailable: {summary.note}"),)
        return Section(id="trace", title=title, blocks=blocks)
    found = [
        item(
            "caveat",
            "profiled timing is not a speed claim: the profiler slows the engine, and this ran "
            "on a separate short launch after the clean measurement",
        ),
        item(
            "window",
            f"window: {summary.window_ms:,.1f} ms, GPU busy {summary.gpu_busy_share:.1%}, "
            f"{summary.kernels:,} kernels",
        ),
        item(
            "gaps",
            f"{summary.gap_count:,} idle gaps of at least {GAP_MIN_US:g} us with no GPU "
            f"activity: {summary.idle_share:.1%} of the window",
        ),
    ]
    if summary.status == "untrusted":
        found.append(
            item(
                "untrusted",
                f"NOT TRUSTED, no recommendation is drawn from it: {summary.note}",
                "warning",
            )
        )
    if not analysis_available and (summary.idle_share or 0) >= IDLE_HINT_SHARE:
        found.append(
            item(
                "analysis_hint",
                "Cause analysis of GPU idle time and the fixes for it are available with the "
                "optimize command.",
            )
        )
    return Section(id="trace", title=title, blocks=tuple(found))


def _cell(kernel: KernelCounters, key: str, pattern: str) -> str:
    value = kernel.value(key)
    if value is not None:
        return pattern.format(value)
    group = GROUP_OF[key]
    if group in kernel.unavailable:
        return f"unavailable (group {group}: {kernel.unavailable[group]})"
    return "n/a"


def _geometry(kernel: KernelCounters) -> str:
    grid, block = kernel.value("grid"), kernel.value("block")
    if grid is None or block is None:
        return "n/a"
    sms = kernel.value("sm_count")
    return f"{grid:,.0f} x {block:,.0f}" + ("" if sms is None else f" ({sms:.0f} SMs)")


def _resources(kernel: KernelCounters) -> str:
    registers = kernel.value("registers")
    static, dynamic = kernel.value("smem_static_b"), kernel.value("smem_dynamic_b")
    smem = "n/a" if static is None or dynamic is None else f"{(static + dynamic) / 1024:.0f} KiB"
    return f"{'n/a' if registers is None else f'{registers:.0f}'} regs, {smem} smem"


def _occupancy(kernel: KernelCounters) -> str:
    theoretical = kernel.value("theoretical_occupancy_pct")
    if theoretical is None:
        return "n/a"
    return f"{theoretical:.1f}% ({kernel.occupancy_limiter or 'limiter n/a'})"


def counters_section(summary: CountersSummary, *, analysis_available: bool) -> Section:
    """The facts: one row per profiled kernel, every counter with its unit and denominator."""
    title = "Kernel counters (Nsight Compute - diagnostic only)"
    blocks: list[Block] = []
    if summary.qualification is not None:
        q = summary.qualification
        groups = [
            f"\n  - group {g.name}: "
            + (f"unavailable ({g.reason})" if g.reason else "qualified")
            + f"; replay passes {'n/a' if g.passes is None else f'{g.passes:g}'}; probe "
            f"launches eager {g.eager_launches}, graph {g.graph_launches}; GPU memory after "
            f"the probe {g.memory_after_mib} MiB"
            for g in q.groups
        ]
        blocks.append(
            item(
                "qualification",
                f"profiler qualification: {'passed' if q.passed else 'FAILED'}"
                + (f" ({q.reason})" if q.reason else "")
                + f"; ncu {q.ncu_version or 'version unknown'}; GPU memory cold "
                f"{q.memory_cold_mib} MiB, after profiling {summary.memory_after_mib} MiB"
                + "".join(groups),
            )
        )
    if summary.status != "ok":
        blocks.append(item("unavailable", f"counters unavailable ({summary.note})"))
        return Section(id="counters", title=title, blocks=tuple(blocks))
    if summary.note:
        blocks.append(item("note", f"NOTE: {summary.note}", "warning"))
    blocks += [
        item(
            "replay_caveat",
            "ncu replays each profiled kernel with flushed caches at unlocked (boost) clocks: "
            "durations are diagnostic, never a speed claim; values are medians over the "
            "profiled launches",
        ),
        Table(
            key="kernels",
            header=(
                "kernel",
                "GPU time",
                "launches",
                "grid x block",
                "resources",
                "theoretical occ",
                *(label for _, label, _ in COLUMNS),
            ),
            rows=tuple(
                (
                    f"`{short_name(kernel.name)}`",
                    f"{kernel.share:.1%}",
                    str(kernel.launches),
                    _geometry(kernel),
                    _resources(kernel),
                    _occupancy(kernel),
                    *(_cell(kernel, key, pattern) for key, _, pattern in COLUMNS),
                )
                for kernel in summary.kernels
            ),
        ),
        *(item("legend", entry) for entry in LEGEND),
    ]
    blocks += [
        item("kernel_note", f"{kernel.note}: `{short_name(kernel.name)}`")
        for kernel in summary.kernels
        if kernel.note
    ]
    if not analysis_available:
        blocks.append(
            item(
                "analysis_hint",
                "Which regime each kernel is in, and whether its low occupancy is by design, is "
                "interpreted by the optimize command.",
            )
        )
    return Section(id="counters", title=title, blocks=tuple(blocks))
