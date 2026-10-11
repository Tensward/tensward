"""The GPU's own state while a run is measured: utilisation, power against the enforced limit,
SM clock against its maximum and the throttle reasons. The platform's tool samples them about
once a second for the whole command; a window of the samples is summarised on request."""

from __future__ import annotations

import select
import subprocess
import threading
import time
from dataclasses import dataclass
from statistics import fmean
from typing import IO, Callable, Iterable, Sequence

from .platforms import GpuSample, Platform

FIRST_SAMPLE_S = 5.0  # no sample from any command by then: the device cannot be sampled
MIN_SAMPLES = 5  # fewer in the window: too few to judge
NO_WINDOW = "no measured window, so the samples cannot be told from load or idle time"


@dataclass(frozen=True, slots=True)
class GpuState:
    """The serving GPU (the busiest sampled one) over a window; ``unavailable`` says why there
    is no summary."""

    unavailable: str | None = None
    index: int | None = None
    samples: int = 0
    utilization: float | None = None  # mean percent
    power_w: float | None = None
    power_limit_w: float | None = None
    sm_mhz: float | None = None
    sm_max_mhz: float | None = None
    power_capped_share: float | None = None  # of the samples that reported the reason
    thermal_share: float | None = None


def _mean(values: Iterable[float | None]) -> float | None:
    known = [value for value in values if value is not None]
    return fmean(known) if known else None


def _share(values: Iterable[bool | None]) -> float | None:
    known = [value for value in values if value is not None]
    return sum(known) / len(known) if known else None


def summarize(samples: Sequence[GpuSample], start_ns: int, end_ns: int) -> GpuState:
    inside = [s for s in samples if start_ns <= s.at_ns <= end_ns]
    if len(inside) < MIN_SAMPLES:
        found = f"{len(inside)} GPU samples in the measured window"
        return GpuState(unavailable=f"{found}, fewer than {MIN_SAMPLES}")
    index = max(
        {s.index for s in inside},
        key=lambda i: _mean(s.utilization for s in inside if s.index == i) or 0.0,
    )
    own = [s for s in inside if s.index == index]
    return GpuState(
        index=index,
        samples=len(own),
        utilization=_mean(s.utilization for s in own),
        power_w=_mean(s.power_w for s in own),
        power_limit_w=_mean(s.power_limit_w for s in own),
        sm_mhz=_mean(s.sm_mhz for s in own),
        sm_max_mhz=_mean(s.sm_max_mhz for s in own),
        power_capped_share=_share(s.power_capped for s in own),
        thermal_share=_share(s.thermal for s in own),
    )


def _first_sample(
    stream: IO[str], parse: Callable[[str, int], GpuSample | None], deadline: float
) -> GpuSample | None:
    """The first line of ``stream`` that is a sample, skipping others; None at the end of the
    stream or at ``deadline`` (time.monotonic())."""
    while (left := deadline - time.monotonic()) > 0:
        ready, _, _ = select.select([stream], [], [], left)
        if not ready or not (line := stream.readline()):
            return None
        if (sample := parse(line, time.monotonic_ns())) is not None:
            return sample
    return None


class GpuSampler:
    """One sampling process for the life of a command; ``state`` summarises any window of it."""

    def __init__(self, platform: Platform, gpus: tuple[str, ...] | None) -> None:
        self._platform, self._gpus = platform, gpus
        self._samples: list[GpuSample] = []
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self.unavailable: str | None = None

    def start(self) -> None:
        commands = self._platform.sampler_commands(self._gpus)
        self.unavailable = f"{self._platform.label} devices cannot be sampled"
        deadline = time.monotonic() + FIRST_SAMPLE_S
        for command in commands:
            try:
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
                )
            except OSError as error:
                self.unavailable = (
                    f"the {self._platform.label} sampler could not start: {error.strerror}"
                )
                return
            assert process.stdout is not None
            try:
                first = _first_sample(process.stdout, self._platform.parse_sample, deadline)
            except BaseException:
                process.kill()
                process.wait()
                raise
            if first is not None:
                self._samples.append(first)
                self._process, self.unavailable = process, None
                self._thread = threading.Thread(
                    target=self._read, args=(process.stdout,), daemon=True
                )
                self._thread.start()
                return
            process.kill()
            process.wait()
            self.unavailable = f"the {self._platform.label} sampler gave no sample"

    def _read(self, stream: IO[str]) -> None:
        for line in stream:
            if (sample := self._platform.parse_sample(line, time.monotonic_ns())) is not None:
                self._samples.append(sample)

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def state(self, start_ns: int | None, end_ns: int | None) -> GpuState:
        if self.unavailable is not None:
            return GpuState(unavailable=self.unavailable)
        if start_ns is None or end_ns is None:
            return GpuState(unavailable=NO_WINDOW)
        return summarize(list(self._samples), start_ns, end_ns)
