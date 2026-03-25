import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class GpuSample:
    ts_ns: int
    gpu_util: Optional[float] = None
    sm_util: Optional[float] = None
    mem_util: Optional[float] = None
    mem_allocated_mb: Optional[float] = None
    mem_reserved_mb: Optional[float] = None
    mem_total_mb: Optional[float] = None
    power_w: Optional[float] = None
    temperature_c: Optional[float] = None
    bandwidth_gbps: Optional[float] = None
    bandwidth_tx_gbps: Optional[float] = None
    bandwidth_rx_gbps: Optional[float] = None
    backend: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsBackend:
    name = "base"

    def available(self) -> bool:
        return False

    def capabilities(self) -> Dict[str, Any]:
        return {"backend": self.name, "available": self.available()}

    def sample(self) -> GpuSample:
        raise NotImplementedError


class NvmlBackend(MetricsBackend):
    name = "nvml"

    def __init__(self, gpu_index: int = 0) -> None:
        self._gpu_index = gpu_index
        self._nvml = None
        self._handle = None
        self._initialized = False
        self._peak_mem_bandwidth_gbps: Optional[float] = None
        self._load_nvml()

    def _load_nvml(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            self._initialized = True
            self._peak_mem_bandwidth_gbps = self._estimate_peak_mem_bandwidth_gbps()
        except Exception:
            self._initialized = False
            self._peak_mem_bandwidth_gbps = None

    def _estimate_peak_mem_bandwidth_gbps(self) -> Optional[float]:
        if not self._initialized or self._handle is None or self._nvml is None:
            return None
        try:
            mem_clock_mhz = float(
                self._nvml.nvmlDeviceGetMaxClockInfo(self._handle, self._nvml.NVML_CLOCK_MEM)
            )
            bus_width_bits = float(self._nvml.nvmlDeviceGetMemoryBusWidth(self._handle))
            if mem_clock_mhz <= 0 or bus_width_bits <= 0:
                return None
            # DDR/GDDR/HBM effective bandwidth proxy:
            # bandwidth(bits/s) = 2 * clock(Hz) * bus_width(bits)
            bandwidth_gbps = 2.0 * (mem_clock_mhz * 1e6) * bus_width_bits / 1e9
            return bandwidth_gbps
        except Exception:
            return None

    def available(self) -> bool:
        return self._initialized and self._handle is not None

    def capabilities(self) -> Dict[str, Any]:
        caps = super().capabilities()
        bandwidth_note = (
            "bandwidth_gbps prefers NVML PCIe throughput as (TX + RX); "
            "if unavailable, fallback estimates HBM/GDDR as mem_util * peak theoretical bandwidth"
        )
        caps.update(
            {
                "gpu_util": True,
                "sm_util": True,
                "mem_util": True,
                "mem_allocated_mb": False,
                "mem_reserved_mb": False,
                "mem_total_mb": True,
                "power_w": True,
                "temperature_c": True,
                "bandwidth_gbps": True,
                "bandwidth_tx_gbps": True,
                "bandwidth_rx_gbps": True,
                "bandwidth_peak_theoretical_gbps": self._peak_mem_bandwidth_gbps,
                "bandwidth_note": bandwidth_note,
            }
        )
        return caps

    def sample(self) -> GpuSample:
        if not self.available():
            raise RuntimeError("NVML backend is unavailable")
        now_ns = time.perf_counter_ns()
        util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
        mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
        power_w = None
        temperature_c = None
        try:
            power_w = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        except Exception:
            power_w = None
        try:
            temperature_c = self._nvml.nvmlDeviceGetTemperature(
                self._handle, self._nvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            temperature_c = None
        mem_total_mb = mem.total / (1024.0 * 1024.0)
        bandwidth_gbps = None
        bandwidth_tx_gbps = None
        bandwidth_rx_gbps = None
        try:
            tx_kb_s = float(
                self._nvml.nvmlDeviceGetPcieThroughput(self._handle, self._nvml.NVML_PCIE_UTIL_TX_BYTES)
            )
            rx_kb_s = float(
                self._nvml.nvmlDeviceGetPcieThroughput(self._handle, self._nvml.NVML_PCIE_UTIL_RX_BYTES)
            )
            bandwidth_tx_gbps = tx_kb_s * 1024.0 * 8.0 / 1e9
            bandwidth_rx_gbps = rx_kb_s * 1024.0 * 8.0 / 1e9
            bandwidth_gbps = bandwidth_tx_gbps + bandwidth_rx_gbps
        except Exception:
            if self._peak_mem_bandwidth_gbps is not None:
                bandwidth_gbps = (float(util.memory) / 100.0) * self._peak_mem_bandwidth_gbps
        return GpuSample(
            ts_ns=now_ns,
            gpu_util=float(util.gpu),
            sm_util=float(util.gpu),
            mem_util=float(util.memory),
            mem_total_mb=mem_total_mb,
            power_w=power_w,
            temperature_c=temperature_c,
            bandwidth_gbps=bandwidth_gbps,
            bandwidth_tx_gbps=bandwidth_tx_gbps,
            bandwidth_rx_gbps=bandwidth_rx_gbps,
            backend=self.name,
        )


class TorchCudaBackend(MetricsBackend):
    name = "torch_cuda"

    def __init__(self, gpu_index: int = 0) -> None:
        self._gpu_index = gpu_index
        self._device = torch.device(f"cuda:{gpu_index}")

    def available(self) -> bool:
        return torch.cuda.is_available() and torch.cuda.device_count() > self._gpu_index

    def capabilities(self) -> Dict[str, Any]:
        caps = super().capabilities()
        caps.update(
            {
                "gpu_util": False,
                "sm_util": False,
                "mem_util": False,
                "mem_allocated_mb": True,
                "mem_reserved_mb": True,
                "mem_total_mb": True,
                "power_w": False,
                "temperature_c": False,
                "bandwidth_gbps": False,
                "bandwidth_tx_gbps": False,
                "bandwidth_rx_gbps": False,
                "bandwidth_peak_theoretical_gbps": None,
            }
        )
        return caps

    def sample(self) -> GpuSample:
        if not self.available():
            raise RuntimeError("CUDA is unavailable")
        now_ns = time.perf_counter_ns()
        with torch.cuda.device(self._device):
            mem_allocated = torch.cuda.memory_allocated(self._device)
            mem_reserved = torch.cuda.memory_reserved(self._device)
            total_mem = torch.cuda.get_device_properties(self._device).total_memory
        allocated_mb = mem_allocated / (1024.0 * 1024.0)
        reserved_mb = mem_reserved / (1024.0 * 1024.0)
        total_mb = total_mem / (1024.0 * 1024.0)
        mem_util = None
        if total_mb > 0:
            mem_util = (allocated_mb / total_mb) * 100.0
        return GpuSample(
            ts_ns=now_ns,
            mem_util=mem_util,
            mem_allocated_mb=allocated_mb,
            mem_reserved_mb=reserved_mb,
            mem_total_mb=total_mb,
            backend=self.name,
        )


def build_backend(gpu_index: int = 0, prefer: Optional[List[str]] = None) -> MetricsBackend:
    order = prefer or ["nvml", "torch_cuda"]
    by_name = {
        "nvml": NvmlBackend(gpu_index=gpu_index),
        "torch_cuda": TorchCudaBackend(gpu_index=gpu_index),
    }
    for name in order:
        backend = by_name.get(name)
        if backend is not None and backend.available():
            return backend
    return by_name["torch_cuda"]
