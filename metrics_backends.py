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
    mem_used_mb: Optional[float] = None
    mem_total_mb: Optional[float] = None
    power_w: Optional[float] = None
    temperature_c: Optional[float] = None
    bandwidth_gbps: Optional[float] = None
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
        self._load_nvml()

    def _load_nvml(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            self._initialized = True
        except Exception:
            self._initialized = False

    def available(self) -> bool:
        return self._initialized and self._handle is not None

    def capabilities(self) -> Dict[str, Any]:
        caps = super().capabilities()
        caps.update(
            {
                "gpu_util": True,
                "sm_util": True,
                "mem_util": True,
                "mem_used_mb": True,
                "mem_total_mb": True,
                "power_w": True,
                "temperature_c": True,
                "bandwidth_gbps": False,
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
        mem_used_mb = mem.used / (1024.0 * 1024.0)
        mem_total_mb = mem.total / (1024.0 * 1024.0)
        return GpuSample(
            ts_ns=now_ns,
            gpu_util=float(util.gpu),
            sm_util=float(util.gpu),
            mem_util=float(util.memory),
            mem_used_mb=mem_used_mb,
            mem_total_mb=mem_total_mb,
            power_w=power_w,
            temperature_c=temperature_c,
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
                "mem_used_mb": True,
                "mem_total_mb": True,
                "power_w": False,
                "temperature_c": False,
                "bandwidth_gbps": False,
            }
        )
        return caps

    def sample(self) -> GpuSample:
        if not self.available():
            raise RuntimeError("CUDA is unavailable")
        now_ns = time.perf_counter_ns()
        with torch.cuda.device(self._device):
            mem_used = torch.cuda.memory_allocated(self._device)
            mem_reserved = torch.cuda.memory_reserved(self._device)
            total_mem = torch.cuda.get_device_properties(self._device).total_memory
        used_mb = mem_used / (1024.0 * 1024.0)
        total_mb = total_mem / (1024.0 * 1024.0)
        mem_util = None
        if total_mb > 0:
            mem_util = (used_mb / total_mb) * 100.0
        # reserved is surfaced in mem_used when util backends are absent
        return GpuSample(
            ts_ns=now_ns,
            mem_util=mem_util,
            mem_used_mb=max(used_mb, mem_reserved / (1024.0 * 1024.0)),
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
