"""The thresholds the bottleneck classifier reads, with what each means and where it comes from.

The literature gives few cut-offs. queue_share and decode_of_ceiling are calibrated: Tensward's
runs on an L4 and an A10G induced and avoided each bottleneck, and the values held on both GPUs.
The rest are starting points until runs on real GPUs set them. An uncalibrated threshold never
yields high confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CALIBRATED_GPUS = ("L4", "A10G")
CALIBRATED_ON = "NVIDIA L4 and A10G, vLLM 0.30, 2026-10-04"
MIN_CONFIDENT_REQUESTS = 30  # fewer successful requests cap a diagnosis at "possible"
MIN_PREEMPTIONS = 2  # a single preemption in a short run is not KV pressure
MIN_MARGIN_VALUE = 1e-6  # a value of 0 ranks as far past a "below" level instead of dividing
NEAR_THRESHOLD = 0.7  # a signal this share of its gate or more is near it


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

    def crossed(self, value: float) -> Literal["critical", "warning", "clear"]:
        """One of "critical", "warning" or "clear"."""

        def past(level: float) -> bool:
            return value <= level if self.below else value >= level

        if past(self.critical):
            return "critical"
        return "warning" if past(self.warning) else "clear"

    def margin(self, value: float) -> float:
        """How far ``value`` is toward or past the warning level: 1.0 at the level."""
        return self.warning / max(value, MIN_MARGIN_VALUE) if self.below else value / self.warning

    def near(self, value: float) -> bool:
        """Whether ``value`` has not crossed the warning level but is within
        NEAR_THRESHOLD of it."""
        if self.below:
            return self.warning < value <= self.warning / NEAR_THRESHOLD
        return NEAR_THRESHOLD * self.warning <= value < self.warning


QUEUE_SHARE = Threshold(
    "queue_share",
    "share of server-side TTFT spent queued before scheduling",
    0.2,
    0.5,
    "set from Tensward's runs on L4 and A10G that induced and avoided this bottleneck (no "
    "published cut-off)",
    calibrated=True,
    calibrated_on=CALIBRATED_ON,
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
    "set from Tensward's runs on L4 and A10G that induced and avoided this bottleneck "
    "(Databricks measured 55-60% bandwidth use at batch 1)",
    calibrated=True,
    calibrated_on=CALIBRATED_ON,
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
    "no published cut-off; Tensward's runs showed 70-81% idle with CUDA graphs off on a 0.5B "
    "model; not yet checked on runs without host overhead",
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
SPEC_COVERAGE = Threshold(
    "spec_coverage",
    "share of generated tokens that came from accepted drafts",
    0.2,
    0.1,
    "no published cut-off; Tensward's runs lost throughput at about 5% coverage with 32 clients "
    "and gained at about 43% with 1 client; low coverage at low concurrency not yet measured",
    below=True,
)
FRONTEND_CPU = Threshold(
    "frontend_cpu",
    "CPU cores the engine's API-server process used over the window",
    0.7,
    0.9,
    "no published cut-off; the API server is one process, so about one core is its ceiling; "
    "not yet calibrated",
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
    SPEC_COVERAGE,
    FRONTEND_CPU,
)
