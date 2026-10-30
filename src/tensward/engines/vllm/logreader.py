"""vLLM's reading of its own server log and scrapes."""

from __future__ import annotations

import tomllib
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from ...settings import Settings
from . import histograms, logs, start_phases, startup
from .flags import canonical_flags

if TYPE_CHECKING:
    from ...context import AnalysisContext
    from ...playbook import Situation
    from ...runtime import Runtime

STARTUP_MEMORY = (
    "peak_activation_gib",
    "graphs_gib",
    "fit_kv_bytes",
)  # from the start's memory line


HASH_FILE = "data/compile_hash.toml"


@cache
def _hash_neutral() -> frozenset[str]:
    table = tomllib.loads(files("tensward").joinpath(HASH_FILE).read_text("utf-8"))
    return frozenset(entry["flag"] for entry in table["neutral"])


def _text(path: Path) -> str | None:
    try:
        return path.read_text("utf-8", errors="replace")
    except OSError:
        return None


class VllmLog:
    start_classes: tuple[str, ...] = start_phases.CLASSES

    def start_class(self, log: str, caches: Any, settings: Settings) -> str | None:
        return start_phases.start_class(
            start_phases.phases(log),
            start_phases.cache_state(log, caches),
            settings.quantization,
        )

    def start_class_for(self, flags: frozenset[str], quantization: str | None) -> str:
        caches = {"aot_hit": self.compile_neutral(flags), "empty_before": False}
        return start_phases.start_class(None, caches, quantization)

    def compile_neutral(self, flags: frozenset[str]) -> bool:
        return flags <= _hash_neutral()

    def start_cost(self, flags: frozenset[str], quantization: str | None) -> int:
        return self.start_classes.index(self.start_class_for(flags, quantization))

    def window_facts(
        self,
        log: str,
        compile_cache: str | None,
        caches: Any = None,
        quantization: str | None = None,
    ) -> dict[str, Any]:
        """The start-up figures of window.json: the start times, the pool and memory lines, and
        how the start went (``caches``: the run's recorded compile caches)."""
        timing = start_phases.phases(log)
        usage = startup.memory_usage(log)
        return {
            **startup.startup(log, compile_cache),
            "kv_pool_tokens": startup.kv_pool_tokens(log),
            "memory_usage": usage,
            "forced_backend": startup.forced_backend(log),
            "phases": timing,
            "start_class": start_phases.start_class(
                timing, start_phases.cache_state(log, caches), quantization
            ),
            **{name: (usage or {}).get(name) for name in STARTUP_MEMORY},
        }

    def kv_pool(self, log: str) -> tuple[int | None, float | None]:
        return startup.kv_pool_tokens(log), startup.kv_available_gib(log)

    def slot_tokens(self, log: str) -> int | None:
        return None

    def load_lines(self, log: str) -> int | None:
        return start_phases.load_lines(log)

    def load_queue(self, log: str, skip: int = 0) -> start_phases.LoadQueue | None:
        return start_phases.load_queue(log, skip)

    def no_kv_memory(self, log: str) -> bool:
        return startup.no_kv_memory(log)

    def step_budget(self, log: str) -> int | None:
        return logs.step_budget(log)

    def memory_usage(self, log: str) -> dict[str, float] | None:
        return startup.memory_usage(log)

    def run_facts(
        self,
        run_dir: Path | None,
        context: AnalysisContext,
        situation: Situation,
        span_s: float | None,
    ) -> dict[str, object]:
        found: dict[str, object] = _offload_facts(situation.settings)
        if run_dir is None:
            return found
        log = _text(run_dir / "server.log")
        if log is not None:
            if span_s is not None:
                counts = histograms.running_counts(log, span_s)
                found["run.running_p50"] = histograms.running_percentile(counts, 0.5)
                found["run.running_p90"] = histograms.running_percentile(counts, 0.9)
            found["run.start_class"] = self.start_class(
                log, context.window_extra.get("caches"), situation.settings
            )
        before = _text(run_dir / "metrics_before.prom")
        after = _text(run_dir / "metrics_after.prom")
        if before is not None and after is not None:
            latency = histograms.growth(before, after, histograms.REQUEST_TPOT)
            median = latency.quantile(0.5) if latency else None
            found["run.server_tpot_p50_ms"] = median * 1000 if median is not None else None
            steps = histograms.growth(before, after, histograms.ITERATION_TOKENS)
            peak = situation.measurement.peak_running
            found["run.mixed_step_share"] = steps.share_above(peak) if steps and peak else None
        return found

    def cache_observer(
        self, context: AnalysisContext, runtime: Runtime, env: Mapping[str, str]
    ) -> Callable[[Any], None] | None:
        from . import caches

        return caches.observer(context, runtime, env)


def _offload_facts(settings: Settings) -> dict[str, object]:
    """``--cpu-offload-gb`` keeps that much of the weights in host memory."""
    given = canonical_flags(settings.extra_args).get("--cpu-offload-gb")
    try:
        offloaded = float(given or 0) > 0
    except (TypeError, ValueError):
        offloaded = False
    return {
        "run.compute": "mixed" if offloaded else "gpu",
        "run.placement": "weight_offload" if offloaded else "full_gpu",
    }


VLLM_LOG = VllmLog()
