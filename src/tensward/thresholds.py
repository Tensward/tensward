"""The thresholds the bottleneck classifier reads, with what each means and where it comes from.

The literature gives few cut-offs, so none of these values is calibrated yet: they are starting
points until runs that induce each bottleneck on real GPUs set them per hardware tier. An
uncalibrated threshold never yields high confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_CONFIDENT_REQUESTS = 30  # fewer successful requests cap a diagnosis at "possible"
MIN_PREEMPTIONS = 2  # a single preemption in a short run is not KV pressure


@dataclass(frozen=True, slots=True)
class Threshold:
    name: str
    meaning: str
    warning: float
    critical: float
    source: str
    below: bool = False  # a value at or under the levels crosses them
    calibrated: bool = False
    calibrated_on: str | None = None  # hardware tier and date, once calibrated

    def crossed(self, value: float) -> str:
        """One of "critical", "warning" or "clear"."""

        def past(level: float) -> bool:
            return value <= level if self.below else value >= level

        if past(self.critical):
            return "critical"
        return "warning" if past(self.warning) else "clear"

    def margin(self, value: float) -> float:
        """How far ``value`` is toward or past the warning level: 1.0 at the level."""
        return self.warning / value if self.below else value / self.warning


QUEUE_SHARE = Threshold(
    "queue_share",
    "share of server-side TTFT spent queued before scheduling",
    0.2,
    0.5,
    "no published cut-off; a starting point until calibrated on GPUs (Tensward's own runs saw "
    "queueing make up most of TTFT)",
)
PREEMPTION_RATE = Threshold(
    "preemption_rate",
    "preemptions per request",
    0.01,
    0.05,
    "vLLM docs give no rate; Llumnix (OSDI'24) saw 8% of requests preempted",
)
KV_PEAK = Threshold(
    "kv_peak",
    "highest sampled share of the engine's KV-cache pool in use",
    0.9,
    0.98,
    "llm-d stops routing at 0.8; SGLang calls 0.9 healthy for offline batches",
)
TTFT_OVER_TPOT = Threshold(
    "ttft_over_tpot",
    "TTFT p50 over TPOT p50",
    20.0,
    50.0,
    "no published cut-off; a starting point until calibrated on GPUs",
)
TPOT_TAIL = Threshold(
    "tpot_tail",
    "TPOT p95 over TPOT p50",
    2.0,
    3.0,
    "Sarathi-Serve (OSDI'24) shows the stalls; no published ratio",
)
DECODE_OF_CEILING = Threshold(
    "decode_of_ceiling",
    "decode tokens/s as a percent of the memory-bandwidth ceiling",
    50.0,
    70.0,
    "Databricks measured 55-60% bandwidth use at batch 1",
)
TENSOR_BUSY = Threshold(
    "tensor_busy",
    "tensor-pipe busy percent of the kernel with the most GPU time",
    50.0,
    70.0,
    "no published cut-off; measured from kernel counters, never inferred from batch size",
)
GPU_IDLE = Threshold(
    "gpu_idle",
    "share of the traced window the GPU sat idle, in gaps of at least 50 us",
    0.10,
    0.25,
    "no published cut-off; a busy L4 trace showed 0.1% idle",
)
KV_OVER_WEIGHTS = Threshold(
    "kv_over_weights",
    "KV-cache bytes a decode step reads over the weight bytes",
    0.5,
    1.0,
    "no published cut-off; MagicDec shows KV reads limit decode at large batch and long context",
)
SPEC_ACCEPTANCE = Threshold(
    "spec_acceptance",
    "tokens per speculative draft (1 + accepted tokens per draft)",
    1.5,
    1.1,
    "no published cut-off; speculation pays only when acceptance beats the draft's cost",
    below=True,
)
THRESHOLDS = (
    QUEUE_SHARE,
    PREEMPTION_RATE,
    KV_PEAK,
    TTFT_OVER_TPOT,
    TPOT_TAIL,
    DECODE_OF_CEILING,
    TENSOR_BUSY,
    GPU_IDLE,
    KV_OVER_WEIGHTS,
    SPEC_ACCEPTANCE,
)
