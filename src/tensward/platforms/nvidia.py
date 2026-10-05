"""NVIDIA GPUs: detection (nvidia-smi, or NVML when nvidia-smi gives no answer), the datasheet
table, the CUDA driver's device facts, and how a runtime pins devices.

Every reference to NVIDIA, CUDA, nvidia-smi or libcuda in Tensward lives here.
"""

from __future__ import annotations

import ctypes
import json
import re
import sys
from dataclasses import asdict
from functools import cache
from typing import Any, Mapping, Sequence

from ..probes import run_probe
from .protocol import MIB, Device, DeviceIdentity, HardwareSpec

NVIDIA_SMI_QUERY = (
    "nvidia-smi",
    "--query-gpu=name,memory.total,memory.used,index,uuid,pci.bus_id,driver_version,"
    "pci.device_id,vbios_version",
    "--format=csv,noheader,nounits",
)
NOT_REPORTED = {"[N/A]", "N/A"}
VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
DEVICE_ORDER = "CUDA_DEVICE_ORDER"

# Conventions. Every figure is DENSE: NVIDIA datasheets print "dense | sparse*" pairs or a single
# number footnoted "with sparsity"; a figure is halved only where the source marks it as sparse.
# Tensor FP16/BF16 rates are for FP32 accumulation, which is what vLLM's fp16/bf16 GEMMs use.
# Data-center parts (L4, A10, A100, H100, L40S) and Turing's T4 (mixed FP16/FP32) publish one
# rate; GeForce parts publish separate FP16- and FP32-accumulate rates and take the latter.
# Rows whose figures could not be checked against an official NVIDIA page or PDF are omitted
# (for example A10G, where AWS publishes only "320 tensor cores, up to 250 TOPS, 24 GB" and no
# bandwidth or FP16 rate; and H100 PCIe, whose dense rates appear only in a rounded
# pre-release table). An unlisted GPU reports "ceilings unavailable". Sources, fetched 2026-09:
#   L4: nvidia.com/en-us/data-center/l4 (242 FP16/BF16, 485 FP8/INT8 with sparsity, 300 GB/s,
#     24 GB); Ada whitepaper Appendix D (images.nvidia.com/aem-dam/Solutions/Data-Center/l4/
#     nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf): 58 SMs, BF16 121 | 242
#   A10: nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/a10-datasheet.pdf
#     (FP16 125 | 250*, INT8 250 | 500*, 600 GB/s, 24 GB, 72 RT cores = 72 SMs)
#   T4: nvidia.com/en-us/data-center/tesla-t4 (65 FP16, 130 INT8, 16 GB, "320+" GB/s);
#     320 GB/s and 40 SMs from the Ada whitepaper Table 5
#   A100: nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-
#     nvidia-1758950-r4-web.pdf (FP16 312 | 624*, INT8 624 | 1248*; 1555 GB/s 40GB, 1935 80GB
#     PCIe, 2039 80GB SXM); 108 SMs from the Ampere GA100 whitepaper
#     (images.nvidia.com/aem-dam/en-zz/Solutions/data-center/
#     nvidia-ampere-architecture-whitepaper.pdf)
#   H100 SXM: nvidia.com/en-us/data-center/h100 (FP16 1,979*, FP8 3,958*, INT8 3,958* with
#     sparsity, 3.35 TB/s); 132 SMs from
#     developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth
#   L40S: nvidia.com/en-us/data-center/l40s (FP16/BF16 362.05 | 733*, FP8 and INT8 733 | 1,466*,
#     864 GB/s, 48 GB, 18,176 CUDA cores = 142 SMs)
#   RTX 4090: Ada whitepaper Table 2 (images.nvidia.com/aem-dam/Solutions/Data-Center/l4/
#     nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf): 128 SMs, FP16 with FP32 accumulate
#     165.2 | 330.4*, INT8 660.6 | 1321.2*, FP8 with FP32 accumulate 330.3 | 660.6*, 1008 GB/s
#   RTX 3090: Ampere GA102 whitepaper Table 9 (nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-
#     architecture-whitepaper-v2.pdf): 82 SMs, FP16 with FP32 accumulate 71 | 142*, INT8
#     284 | 568*, 936 GB/s; no FP8 on Ampere
GPUS: tuple[HardwareSpec, ...] = (
    HardwareSpec("L4", r"\bL4\b", 300, 121, 242.5, 242.5, 58),
    HardwareSpec("A10", r"\bA10\b", 600, 125, 250, None, 72),
    HardwareSpec("T4", r"\bT4\b", 320, 65, 130, None, 40),
    HardwareSpec("A100-40GB", r"A100.*40GB", 1555, 312, 624, None, 108),
    HardwareSpec("A100-80GB SXM", r"A100-SXM.*80GB", 2039, 312, 624, None, 108),
    HardwareSpec("A100-80GB PCIe", r"A100 80GB PCIe|A100-PCIE-80GB", 1935, 312, 624, None, 108),
    HardwareSpec("H100 SXM", r"H100 80GB HBM3|H100.SXM", 3350, 989.5, 1979, 1979, 132),
    HardwareSpec("L40S", r"\bL40S\b", 864, 362.05, 733, 733, 142),
    HardwareSpec("RTX 4090", r"RTX 4090", 1008, 165.2, 660.6, 330.3, 128),
    HardwareSpec("RTX 3090", r"RTX 3090", 936, 71, 284, None, 82),
)


# CUDA driver attribute ids (cuda.h): CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE (kHz),
# CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH (bits) and CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
_CU_MEMORY_CLOCK_RATE, _CU_MEMORY_BUS_WIDTH, _CU_MULTIPROCESSOR_COUNT = 36, 37, 16


def derived_bandwidth_gbs(memory_clock_khz: float, bus_width_bits: float) -> float:
    """Peak DRAM bandwidth in GB/s: clock x bus width x 2 (double data rate).

    This is the formula of NVIDIA's CUDA samples (deviceQuery, bandwidthTest): the CUDA
    ``memoryClockRate`` is the clock whose double is the per-pin data rate. Checked on the L4
    (Nsight Compute: 6,251,000 kHz, 192 bits): 6.251e9 x 2 x 24 B = 300.05 GB/s, the datasheet's
    300 GB/s. CUDA is used rather than NVML because NVML's memory clock for GDDR parts is the
    raw GDDR clock, whose relation to the data rate differs by memory generation, and
    nvidia-smi --query-gpu has no bus width field.
    """
    return memory_clock_khz * 1e3 * 2 * (bus_width_bits / 8) / 1e9


@cache
def _libcuda() -> ctypes.CDLL | None:
    """The CUDA driver library, initialised once; None when it is missing or refuses to start."""
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        return None if cuda.cuInit(0) else cuda
    except OSError:
        return None


def _device_attribute(device: int, attribute: int) -> int | None:
    """One attribute of a CUDA device handle, or None when the driver does not give it."""
    cuda = _libcuda()
    value = ctypes.c_int()
    try:
        if cuda is None or cuda.cuDeviceGetAttribute(ctypes.byref(value), attribute, device):
            return None
    except AttributeError:
        return None
    return value.value


@cache
def device_bandwidth_gbs(index: int = 0) -> float | None:
    """Peak DRAM bandwidth of CUDA device ``index`` via libcuda, or None (no driver, no device)."""
    cuda = _libcuda()
    device = ctypes.c_int()
    try:
        if cuda is None or cuda.cuDeviceGet(ctypes.byref(device), index):
            return None
    except AttributeError:
        return None
    clock = _device_attribute(device.value, _CU_MEMORY_CLOCK_RATE)
    width = _device_attribute(device.value, _CU_MEMORY_BUS_WIDTH)
    if clock is None or width is None or clock <= 0 or width <= 0:
        return None
    return derived_bandwidth_gbs(clock, width)


def driver_cuda_version() -> str | None:
    """The newest CUDA the installed driver supports (not the CUDA an engine was built with),
    as ``major.minor``; None when there is no driver."""
    cuda = _libcuda()
    version = ctypes.c_int()
    try:
        if cuda is None or cuda.cuDriverGetVersion(ctypes.byref(version)):
            return None
    except AttributeError:
        return None
    return f"{version.value // 1000}.{version.value % 1000 // 10}"


def device_sm_count(bus_id: str) -> int | None:
    """Streaming multiprocessors of the GPU at PCI ``bus_id`` (nvidia-smi's spelling), or None
    when the driver is missing or the process cannot see that GPU. Looking the device up by bus
    id keeps CUDA's device order and CUDA_VISIBLE_DEVICES out of it."""
    cuda = _libcuda()
    if cuda is None:
        return None
    domain, _, rest = bus_id.partition(":")
    # nvidia-smi prints an 8-digit upper-case domain; CUDA documents a 4-digit lower-case one.
    for spelling in dict.fromkeys([f"{domain[-4:]}:{rest}".lower(), bus_id]):
        device = ctypes.c_int()
        try:
            if cuda.cuDeviceGetByPCIBusId(ctypes.byref(device), spelling.encode()) == 0:
                return _device_attribute(device.value, _CU_MULTIPROCESSOR_COUNT)
        except AttributeError:
            return None
    return None


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def is_mig(token: str) -> bool:
    return token.startswith("MIG-") or ":" in token


_DRIVER_FACTS = (
    "import json, sys; "
    "from tensward.platforms.nvidia import device_sm_count, driver_cuda_version; "
    "print(json.dumps([driver_cuda_version(), device_sm_count(sys.argv[1])]))"
)


def _driver_facts(bus_id: str) -> tuple[str | None, int | None]:
    """(CUDA version the driver supports, SM count of the GPU at ``bus_id``), None for a fact
    that cannot be read."""
    out = run_probe([sys.executable, "-c", _DRIVER_FACTS, bus_id])
    try:
        cuda, sm_count = json.loads(out or "")
    except ValueError:
        return None, None
    return cuda, sm_count


def _smi_devices() -> tuple[Device, ...]:
    """Memory, identity and driver of each GPU from nvidia-smi; empty when it gives no
    answer."""
    out = run_probe(NVIDIA_SMI_QUERY)
    if out is None:
        return ()
    devices = []
    try:
        for line in out.strip().splitlines():
            name, total, used, index, uuid, bus_id, *rest = (
                field.strip() for field in line.rsplit(",", 8)
            )
            driver, device_id, vbios = (
                None if value in NOT_REPORTED or not value.strip("0.") else value for value in rest
            )
            devices.append(
                Device(name, int(total) * MIB, int(used) * MIB, index, uuid, bus_id,
                       driver, device_id, vbios)
            )  # fmt: skip
    except ValueError:
        return ()
    return tuple(devices)


def _nvml_devices() -> tuple[Device, ...]:
    """Name and memory of each GPU from NVML (the optional ``gpu`` extra); empty without it."""
    try:
        import pynvml  # noqa: PLC0415 - optional GPU extra

        pynvml.nvmlInit()
        devices = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = _text(pynvml.nvmlDeviceGetName(handle))
            devices.append(Device(name, memory.total, memory.used, str(index)))
        return tuple(devices)
    except Exception:  # noqa: BLE001 - NVML missing or refusing
        return ()


class NvidiaPlatform:
    name = "nvidia"
    label = "NVIDIA"

    def detect(self) -> tuple[Device, ...]:
        return _smi_devices() or _nvml_devices()

    def spec(self, device_name: str) -> HardwareSpec | None:
        return next((s for s in GPUS if re.search(s.name_pattern, device_name)), None)

    def derived_bandwidth(self, index: int) -> float | None:
        return device_bandwidth_gbs(index)

    def identity(self, device: Device) -> dict[str, Any]:
        """The device's identity fields. The CUDA driver's facts are read in a child process
        under the probe timeout, so a slow or wedged driver cannot stall this one."""
        cuda, sm_count = _driver_facts(device.bus_id)
        return asdict(
            DeviceIdentity(
                device.index,
                device.uuid,
                device.name,
                device.driver,
                cuda,
                device.pci_device_id,
                device.vbios,
                sm_count,
            )  # fmt: skip
        )

    def describe(self, device: Device, identity: Mapping[str, Any]) -> str:
        driver = identity["driver"] or "unknown"
        return f"{device.name} (driver {driver}, CUDA {identity['driver_cuda'] or 'unknown'})"

    def select(
        self, devices: tuple[Device, ...], selection: tuple[str, ...] | None
    ) -> tuple[tuple[Device, ...], tuple[str, ...]]:
        """Indices are nvidia-smi's; a UUID starting ``GPU-`` may be abbreviated, as CUDA
        accepts. A MIG device never matches."""
        if selection is None:
            return devices, ()

        def matches(token: str, device: Device) -> bool:
            return not is_mig(token) and (
                token == device.index
                or (token.startswith("GPU-") and device.uuid.startswith(token))
            )

        matched = tuple(d for d in devices if any(matches(t, d) for t in selection))
        return matched, tuple(t for t in selection if not any(matches(t, d) for d in devices))

    def selection_refusal(self, selection: Sequence[str]) -> str | None:
        return "MIG selections are not supported" if any(map(is_mig, selection)) else None

    def select_env(
        self, indices: tuple[str, ...] | None, environ: Mapping[str, str]
    ) -> dict[str, str]:
        if indices is None:
            return {}
        # PCI order numbers the GPUs as nvidia-smi does, unless the user chose an order
        return {
            VISIBLE_DEVICES: ",".join(indices),
            DEVICE_ORDER: environ.get(DEVICE_ORDER, "PCI_BUS_ID"),
        }

    def docker_device_args(self, indices: tuple[str, ...] | None) -> list[str]:
        # docker reads the value as CSV, so a list of devices needs the literal quotes
        return ["--gpus", "all" if indices is None else f'"device={",".join(indices)}"']


NVIDIA = NvidiaPlatform()
