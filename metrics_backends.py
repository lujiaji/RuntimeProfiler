"""
GPU metrics backends: NVML GPM (preferred), NVML legacy, Torch CUDA fallback.

Resource domains:
- compute: SM / Tensor / FP / INT
- device memory: HBM / GDDR (DRAM bandwidth util vs capacity)
- host-device IO: PCIe (NVLink reserved)
"""

from __future__ import annotations

import time
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class GpuSample:
    """Single online sample; timestamps use perf_counter_ns unless noted."""

    ts_ns: int

    # --- [1] Time / window (filled by profiler export when run window known) ---
    rel_time_ms: Optional[float] = None
    in_run_window: Optional[bool] = None
    active_event_name: Optional[str] = None
    active_event_depth: Optional[int] = None

    # --- [2] Compute domain ---
    gpu_util_pct: Optional[float] = None  # legacy GPU util; weak signal
    sm_util_pct: Optional[float] = None
    sm_occupancy_pct: Optional[float] = None
    tensor_util_pct: Optional[float] = None
    fp16_util_pct: Optional[float] = None
    fp32_util_pct: Optional[float] = None
    int_util_pct: Optional[float] = None
    sm_clock_mhz: Optional[float] = None
    graphics_clock_mhz: Optional[float] = None
    power_w: Optional[float] = None
    temperature_c: Optional[float] = None

    # --- [3] Device memory domain ---
    dram_bw_util_pct: Optional[float] = None
    dram_bw_proxy_gbps: Optional[float] = None
    mem_util_pct: Optional[float] = None  # coarse busy-time / capacity-style from NVML
    alloc_mem_mb: Optional[float] = None
    reserved_mem_mb: Optional[float] = None
    active_mem_mb: Optional[float] = None
    total_mem_mb: Optional[float] = None
    mem_clock_mhz: Optional[float] = None

    # --- [4] PCIe / IO ---
    pcie_tx_mib_s: Optional[float] = None
    pcie_rx_mib_s: Optional[float] = None
    pcie_total_mib_s: Optional[float] = None
    nvlink_tx_mib_s: Optional[float] = None
    nvlink_rx_mib_s: Optional[float] = None

    # --- [5] Provenance ---
    source_compute: Optional[str] = None
    source_device_memory: Optional[str] = None
    source_io: Optional[str] = None
    source_alloc_mem: Optional[str] = None

    backend: str = "unknown"

    # --- Legacy aliases (older plots / scripts) ---
    gpu_util: Optional[float] = None
    sm_util: Optional[float] = None
    mem_util: Optional[float] = None
    mem_allocated_mb: Optional[float] = None
    mem_reserved_mb: Optional[float] = None
    mem_total_mb: Optional[float] = None
    bandwidth_gbps: Optional[float] = None
    bandwidth_tx_gbps: Optional[float] = None
    bandwidth_rx_gbps: Optional[float] = None

    def sync_legacy_aliases(self) -> None:
        """Keep deprecated field names in sync for downstream code."""
        if self.gpu_util_pct is not None:
            self.gpu_util = self.gpu_util_pct
        if self.sm_util_pct is not None:
            self.sm_util = self.sm_util_pct
        if self.mem_util_pct is not None:
            self.mem_util = self.mem_util_pct
        if self.alloc_mem_mb is not None:
            self.mem_allocated_mb = self.alloc_mem_mb
        if self.reserved_mem_mb is not None:
            self.mem_reserved_mb = self.reserved_mem_mb
        if self.total_mem_mb is not None:
            self.mem_total_mb = self.total_mem_mb
        if self.pcie_tx_mib_s is not None and self.pcie_rx_mib_s is not None:
            tx_gbps = self.pcie_tx_mib_s * (1024.0 * 1024.0) * 8.0 / 1e9
            rx_gbps = self.pcie_rx_mib_s * (1024.0 * 1024.0) * 8.0 / 1e9
            self.bandwidth_tx_gbps = tx_gbps
            self.bandwidth_rx_gbps = rx_gbps
            self.bandwidth_gbps = tx_gbps + rx_gbps
        elif self.dram_bw_proxy_gbps is not None:
            self.bandwidth_gbps = self.dram_bw_proxy_gbps

    def to_dict(self) -> Dict[str, Any]:
        self.sync_legacy_aliases()
        d = asdict(self)
        d["source"] = {
            "compute": self.source_compute,
            "device_memory": self.source_device_memory,
            "io": self.source_io,
            "alloc_mem": self.source_alloc_mem,
        }
        return d


class MetricsBackend:
    name = "base"

    def available(self) -> bool:
        return False

    def capabilities(self) -> Dict[str, Any]:
        return {"backend": self.name, "available": self.available()}

    def sample(self) -> GpuSample:
        raise NotImplementedError


def resolve_torch_device_index(requested_index: int) -> Optional[int]:
    """
    Resolve profiler GPU index to torch logical cuda index.

    RuntimeProfiler often uses physical indices for NVML (`--gpu-index`), while torch
    APIs use process-local logical indices after `CUDA_VISIBLE_DEVICES` remapping.
    """
    if not torch.cuda.is_available():
        return None
    count = int(torch.cuda.device_count())
    if count <= 0:
        return None

    # Direct logical index path.
    if 0 <= int(requested_index) < count:
        return int(requested_index)

    # Physical index to logical index remap via CUDA_VISIBLE_DEVICES.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        tokens = [t.strip() for t in cvd.split(",") if t.strip()]
        req = str(int(requested_index))
        for logical_idx, token in enumerate(tokens):
            if token == req and logical_idx < count:
                return logical_idx

    # Common single-GPU-visible case: map any requested index to logical 0.
    if count == 1:
        return 0
    return None


def _kb_s_to_mib_s(kb_s: float) -> float:
    return float(kb_s) / 1024.0


class NvmlLegacyBackend(MetricsBackend):
    """NVML without GPM: util, memory util, PCIe, power, clocks, temperature."""

    name = "nvml_legacy"

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
            return 2.0 * (mem_clock_mhz * 1e6) * bus_width_bits / 1e9
        except Exception:
            return None

    def available(self) -> bool:
        return self._initialized and self._handle is not None

    def capabilities(self) -> Dict[str, Any]:
        caps = super().capabilities()
        caps.update(
            {
                "gpm": False,
                "gpu_util": True,
                "sm_util_proxy": True,
                "mem_util": True,
                "pcie_mib_s": True,
                "power_w": True,
                "temperature_c": True,
                "clocks": True,
                "dram_bw_measured": False,
                "dram_bw_proxy": self._peak_mem_bandwidth_gbps is not None,
                "peak_dram_theoretical_gbps": self._peak_mem_bandwidth_gbps,
            }
        )
        return caps

    def sample(self) -> GpuSample:
        if not self.available():
            raise RuntimeError("NVML legacy backend is unavailable")
        nvml = self._nvml
        handle = self._handle
        assert nvml is not None and handle is not None

        now_ns = time.perf_counter_ns()
        util = nvml.nvmlDeviceGetUtilizationRates(handle)
        mem = nvml.nvmlDeviceGetMemoryInfo(handle)
        mem_total_mb = mem.total / (1024.0 * 1024.0)
        mem_util_pct = float(mem.used) / float(mem.total) * 100.0 if mem.total else None

        power_w = None
        temperature_c = None
        sm_clock_mhz = None
        mem_clock_mhz = None
        graphics_clock_mhz = None
        try:
            power_w = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            pass
        try:
            temperature_c = float(
                nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
            )
        except Exception:
            pass
        try:
            sm_clock_mhz = float(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM))
        except Exception:
            try:
                sm_clock_mhz = float(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_GRAPHICS))
            except Exception:
                pass
        try:
            mem_clock_mhz = float(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM))
        except Exception:
            pass
        try:
            graphics_clock_mhz = float(
                nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_GRAPHICS)
            )
        except Exception:
            pass

        pcie_tx_mib_s = None
        pcie_rx_mib_s = None
        try:
            tx_kb_s = float(
                nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_TX_BYTES)
            )
            rx_kb_s = float(
                nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_RX_BYTES)
            )
            pcie_tx_mib_s = _kb_s_to_mib_s(tx_kb_s)
            pcie_rx_mib_s = _kb_s_to_mib_s(rx_kb_s)
        except Exception:
            pass

        dram_bw_proxy = None
        if self._peak_mem_bandwidth_gbps is not None:
            dram_bw_proxy = (float(util.memory) / 100.0) * self._peak_mem_bandwidth_gbps

        pcie_total = None
        if pcie_tx_mib_s is not None and pcie_rx_mib_s is not None:
            pcie_total = pcie_tx_mib_s + pcie_rx_mib_s

        sample = GpuSample(
            ts_ns=now_ns,
            gpu_util_pct=float(util.gpu),
            sm_util_pct=float(util.gpu),
            mem_util_pct=float(util.memory),
            total_mem_mb=mem_total_mb,
            dram_bw_proxy_gbps=dram_bw_proxy,
            sm_clock_mhz=sm_clock_mhz,
            graphics_clock_mhz=graphics_clock_mhz,
            mem_clock_mhz=mem_clock_mhz,
            power_w=power_w,
            temperature_c=temperature_c,
            pcie_tx_mib_s=pcie_tx_mib_s,
            pcie_rx_mib_s=pcie_rx_mib_s,
            pcie_total_mib_s=pcie_total,
            source_compute="nvml_legacy",
            source_device_memory="nvml_legacy",
            source_io="nvml_legacy",
            source_alloc_mem=None,
            backend=self.name,
        )
        sample.sync_legacy_aliases()
        return sample


class _GpmHelper:
    """Best-effort NVML GPM metrics; API differs across driver / nvidia-ml-py versions."""

    def __init__(self, nvml: Any, handle: Any) -> None:
        self._nvml = nvml
        self._handle = handle
        self.ok = False
        self._sample_a = None
        self._sample_b = None
        self._toggle = False
        self._metric_ids: List[int] = []
        self._id_to_key: Dict[int, str] = {}
        self._init()

    def _metric(self, name: str) -> Optional[int]:
        v = getattr(self._nvml, name, None)
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            return None

    def _init(self) -> None:
        nvml = self._nvml
        mapping: List[Tuple[str, str]] = [
            ("NVML_GPM_METRIC_SM_UTIL", "sm_util_pct"),
            ("NVML_GPM_METRIC_SM_OCCUPANCY", "sm_occupancy_pct"),
            ("NVML_GPM_METRIC_ANY_TENSOR_UTIL", "tensor_util_pct"),
            ("NVML_GPM_METRIC_FP16_UTIL", "fp16_util_pct"),
            ("NVML_GPM_METRIC_FP32_UTIL", "fp32_util_pct"),
            ("NVML_GPM_METRIC_INTEGER_UTIL", "int_util_pct"),
            ("NVML_GPM_METRIC_DRAM_BW_UTIL", "dram_bw_util_pct"),
            ("NVML_GPM_METRIC_PCIE_TX_PER_SEC", "pcie_tx_mib_s"),
            ("NVML_GPM_METRIC_PCIE_RX_PER_SEC", "pcie_rx_mib_s"),
        ]
        for const_name, key in mapping:
            mid = self._metric(const_name)
            if mid is not None:
                self._metric_ids.append(mid)
                self._id_to_key[mid] = key

        if not self._metric_ids:
            return

        # Try allocate sample buffers (several possible API shapes)
        try:
            if hasattr(nvml, "nvmlGpmSampleAlloc"):
                self._sample_a = nvml.nvmlGpmSampleAlloc()
                self._sample_b = nvml.nvmlGpmSampleAlloc()
                self.ok = True
                return
        except Exception:
            pass
        try:
            if hasattr(nvml, "nvmlDeviceGpmSampleAlloc"):
                self._sample_a = nvml.nvmlDeviceGpmSampleAlloc(self._handle)
                self._sample_b = nvml.nvmlDeviceGpmSampleAlloc(self._handle)
                self.ok = True
                return
        except Exception:
            pass
        self.ok = False

    def close(self) -> None:
        nvml = self._nvml
        for s in (self._sample_a, self._sample_b):
            if s is None:
                continue
            try:
                if hasattr(nvml, "nvmlGpmSampleFree"):
                    nvml.nvmlGpmSampleFree(s)
            except Exception:
                pass

    def read_delta(self) -> Dict[str, float]:
        if not self.ok or not self._metric_ids:
            return {}
        nvml = self._nvml
        handle = self._handle
        cur, nxt = (self._sample_a, self._sample_b) if self._toggle else (self._sample_b, self._sample_a)
        self._toggle = not self._toggle
        try:
            if hasattr(nvml, "nvmlGpmSampleGet"):
                nvml.nvmlGpmSampleGet(handle, cur)
                time.sleep(0.002)
                nvml.nvmlGpmSampleGet(handle, nxt)
            elif hasattr(nvml, "nvmlDeviceGetGpmSample"):
                nvml.nvmlDeviceGetGpmSample(handle, cur)
                time.sleep(0.002)
                nvml.nvmlDeviceGetGpmSample(handle, nxt)
            else:
                return {}
        except Exception:
            return {}

        out: Dict[str, float] = {}
        try:
            if hasattr(nvml, "nvmlGpmMetricsGet"):
                # Struct-based API (nvidia-ml-py): build ctypes struct if available
                getter = nvml.nvmlGpmMetricsGet
                import ctypes

                class GpmMetricsGet(ctypes.Structure):
                    _fields_ = [
                        ("version", ctypes.c_uint),
                        ("device", ctypes.c_void_p),
                        ("sample1", ctypes.c_void_p),
                        ("sample2", ctypes.c_void_p),
                        ("metricCount", ctypes.c_uint),
                        ("metricIds", ctypes.POINTER(ctypes.c_uint)),
                        ("values", ctypes.POINTER(ctypes.c_double)),
                    ]

                ids_arr = (ctypes.c_uint * len(self._metric_ids))(*self._metric_ids)
                vals_arr = (ctypes.c_double * len(self._metric_ids))()
                st = GpmMetricsGet()
                st.version = getattr(nvml, "nvmlGpmMetricsGet_v1", ctypes.sizeof(GpmMetricsGet))
                st.device = ctypes.cast(handle, ctypes.c_void_p)
                st.sample1 = ctypes.cast(cur, ctypes.c_void_p)
                st.sample2 = ctypes.cast(nxt, ctypes.c_void_p)
                st.metricCount = len(self._metric_ids)
                st.metricIds = ctypes.cast(ids_arr, ctypes.POINTER(ctypes.c_uint))
                st.values = ctypes.cast(vals_arr, ctypes.POINTER(ctypes.c_double))
                ret = getter(ctypes.byref(st))
                if ret != 0:
                    return {}
                for i, mid in enumerate(self._metric_ids):
                    key = self._id_to_key.get(mid)
                    if key:
                        out[key] = float(vals_arr[i])
                return out
        except Exception:
            pass

        try:
            if hasattr(nvml, "nvmlDeviceGetGpmMetrics"):
                mids = self._metric_ids
                vals = nvml.nvmlDeviceGetGpmMetrics(handle, cur, nxt, mids)
                if vals is not None and len(vals) == len(mids):
                    for mid, v in zip(mids, vals):
                        key = self._id_to_key.get(int(mid))
                        if key:
                            out[key] = float(v)
                    return out
        except Exception:
            pass

        return out


class NvmlGpmOnlyBackend(MetricsBackend):
    """NVML GPM-only path (no legacy NVML fill). Requires full GPM API + two-sample delta."""

    name = "nvml_gpm_only"

    def __init__(self, gpu_index: int = 0) -> None:
        self._gpu_index = gpu_index
        self._nvml = None
        self._handle = None
        self._initialized = False
        self._gpm: Optional[_GpmHelper] = None
        self._load()

    def _load(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            self._initialized = True
            self._gpm = _GpmHelper(pynvml, self._handle)
            if not self._gpm.ok:
                self._gpm = None
        except Exception:
            self._initialized = False
            self._gpm = None

    def available(self) -> bool:
        return bool(self._initialized and self._handle is not None and self._gpm and self._gpm.ok)

    def capabilities(self) -> Dict[str, Any]:
        caps = super().capabilities()
        caps.update(
            {
                "gpm": self.available(),
                "gpu_util": False,
                "dram_bw_measured": self.available(),
            }
        )
        return caps

    def sample(self) -> GpuSample:
        if not self.available() or self._gpm is None:
            raise RuntimeError("NVML GPM-only backend is unavailable")
        now_ns = time.perf_counter_ns()
        gpm = self._gpm.read_delta()
        sample = GpuSample(ts_ns=now_ns, backend=self.name)
        for k, v in gpm.items():
            setattr(sample, k, v)
        sample.source_compute = "nvml_gpm" if gpm.get("sm_util_pct") is not None else None
        if gpm.get("dram_bw_util_pct") is not None:
            sample.source_device_memory = "nvml_gpm"
        if gpm.get("pcie_tx_mib_s") is not None or gpm.get("pcie_rx_mib_s") is not None:
            sample.source_io = "nvml_gpm"
        if sample.pcie_tx_mib_s is not None and sample.pcie_rx_mib_s is not None:
            sample.pcie_total_mib_s = sample.pcie_tx_mib_s + sample.pcie_rx_mib_s
        sample.sync_legacy_aliases()
        return sample


class NvmlGpmBackend(MetricsBackend):
    """
    Prefer NVML GPM for SM/tensor/DRAM BW/PCIe where available;
    fill power, temperature, clocks, and coarse signals from legacy NVML.
    """

    name = "nvml_gpm"

    def __init__(self, gpu_index: int = 0) -> None:
        self._legacy = NvmlLegacyBackend(gpu_index=gpu_index)
        self._gpm: Optional[_GpmHelper] = None
        self._gpm_active = False
        if self._legacy.available() and self._legacy._nvml is not None and self._legacy._handle is not None:
            self._gpm = _GpmHelper(self._legacy._nvml, self._legacy._handle)
            self._gpm_active = bool(self._gpm and self._gpm.ok)

    def available(self) -> bool:
        return self._legacy.available()

    def capabilities(self) -> Dict[str, Any]:
        caps = self._legacy.capabilities()
        caps["backend"] = self.name
        caps["gpm"] = self._gpm_active
        caps["dram_bw_measured"] = self._gpm_active
        return caps

    def sample(self) -> GpuSample:
        base = self._legacy.sample()
        base.backend = self.name
        if self._gpm_active and self._gpm is not None:
            gpm = self._gpm.read_delta()
            if gpm.get("sm_util_pct") is not None:
                base.sm_util_pct = gpm["sm_util_pct"]
                base.source_compute = "nvml_gpm"
            if gpm.get("sm_occupancy_pct") is not None:
                base.sm_occupancy_pct = gpm["sm_occupancy_pct"]
            if gpm.get("tensor_util_pct") is not None:
                base.tensor_util_pct = gpm["tensor_util_pct"]
            if gpm.get("fp16_util_pct") is not None:
                base.fp16_util_pct = gpm["fp16_util_pct"]
            if gpm.get("fp32_util_pct") is not None:
                base.fp32_util_pct = gpm["fp32_util_pct"]
            if gpm.get("int_util_pct") is not None:
                base.int_util_pct = gpm["int_util_pct"]
            if gpm.get("dram_bw_util_pct") is not None:
                base.dram_bw_util_pct = gpm["dram_bw_util_pct"]
                base.source_device_memory = "nvml_gpm"
                base.dram_bw_proxy_gbps = None
            if gpm.get("pcie_tx_mib_s") is not None:
                base.pcie_tx_mib_s = gpm["pcie_tx_mib_s"]
                base.source_io = "nvml_gpm"
            if gpm.get("pcie_rx_mib_s") is not None:
                base.pcie_rx_mib_s = gpm["pcie_rx_mib_s"]
                base.source_io = "nvml_gpm"
            if base.pcie_tx_mib_s is not None and base.pcie_rx_mib_s is not None:
                base.pcie_total_mib_s = base.pcie_tx_mib_s + base.pcie_rx_mib_s
        base.sync_legacy_aliases()
        return base


class TorchCudaBackend(MetricsBackend):
    name = "torch_cuda"

    def __init__(self, gpu_index: int = 0) -> None:
        self._gpu_index = gpu_index
        self._device = torch.device("cuda:0")

    def available(self) -> bool:
        return resolve_torch_device_index(self._gpu_index) is not None

    def capabilities(self) -> Dict[str, Any]:
        caps = super().capabilities()
        caps.update(
            {
                "gpm": False,
                "alloc_only": True,
                "high_confidence_classification": False,
            }
        )
        return caps

    def sample(self) -> GpuSample:
        if not self.available():
            raise RuntimeError("CUDA is unavailable")
        logical_idx = resolve_torch_device_index(self._gpu_index)
        if logical_idx is None:
            raise RuntimeError("Unable to map requested GPU index to torch logical device")
        device = torch.device(f"cuda:{logical_idx}")
        now_ns = time.perf_counter_ns()
        with torch.cuda.device(device):
            mem_allocated = torch.cuda.memory_allocated(device)
            mem_reserved = torch.cuda.memory_reserved(device)
            total_mem = torch.cuda.get_device_properties(device).total_memory
        allocated_mb = mem_allocated / (1024.0 * 1024.0)
        reserved_mb = mem_reserved / (1024.0 * 1024.0)
        total_mb = total_mem / (1024.0 * 1024.0)
        mem_util_pct = (allocated_mb / total_mb) * 100.0 if total_mb > 0 else None
        sample = GpuSample(
            ts_ns=now_ns,
            mem_util_pct=mem_util_pct,
            alloc_mem_mb=allocated_mb,
            reserved_mem_mb=reserved_mb,
            active_mem_mb=allocated_mb,
            total_mem_mb=total_mb,
            source_alloc_mem="torch",
            backend=self.name,
        )
        sample.sync_legacy_aliases()
        return sample


# Backward compatibility alias
NvmlBackend = NvmlLegacyBackend


def build_backend(gpu_index: int = 0, prefer: Optional[List[str]] = None) -> MetricsBackend:
    order = prefer or ["nvml_gpm", "nvml_legacy", "torch_cuda"]
    normalized: List[str] = []
    for name in order:
        n = name.strip().lower().replace("-", "_")
        if n == "nvml":
            n = "nvml_legacy"
        normalized.append(n)

    factories = {
        "nvml_gpm": lambda: NvmlGpmBackend(gpu_index=gpu_index),
        "nvml_gpm_only": lambda: NvmlGpmOnlyBackend(gpu_index=gpu_index),
        "nvml_legacy": lambda: NvmlLegacyBackend(gpu_index=gpu_index),
        "torch_cuda": lambda: TorchCudaBackend(gpu_index=gpu_index),
    }

    for name in normalized:
        factory = factories.get(name)
        if factory is None:
            continue
        backend = factory()
        if backend.available():
            return backend
    return TorchCudaBackend(gpu_index=gpu_index)
