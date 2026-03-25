import csv
import json
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from .classifiers import BoundClassification, classify_bound
from .event_tracer import EventTracer, TraceEvent
from .metrics_backends import GpuSample, MetricsBackend, build_backend
from .plotter import plot_event_timeline, plot_timeline


@dataclass
class RuntimeProfilerConfig:
    enabled: bool = True
    output_dir: str = "runtime_profile"
    sample_interval_ms: int = 20
    gpu_index: int = 0
    backend_preference: List[str] = field(default_factory=lambda: ["nvml", "torch_cuda"])
    topk_peaks: int = 3
    auto_plot: bool = True
    enable_torch_profiler: bool = False
    torch_profiler_record_shapes: bool = False
    torch_profiler_profile_memory: bool = True
    torch_profiler_with_stack: bool = False
    torch_profiler_with_flops: bool = False
    torch_profiler_export_chrome_trace: bool = True
    torch_profiler_topk_ops: int = 200


@dataclass
class ProfileResult:
    result: Any
    samples: List[GpuSample]
    events: List[TraceEvent]
    summary: Dict[str, Any]
    output_dir: str


class RuntimeProfiler:
    def __init__(
        self, config: Optional[RuntimeProfilerConfig] = None, backend: Optional[MetricsBackend] = None
    ) -> None:
        self.config = config or RuntimeProfilerConfig()
        self.backend = backend or build_backend(
            gpu_index=self.config.gpu_index, prefer=self.config.backend_preference
        )
        self.tracer = EventTracer()
        self._samples: List[GpuSample] = []
        self._samples_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _maybe_fill_torch_allocator_memory(self, sample: GpuSample) -> GpuSample:
        if sample.mem_allocated_mb is not None and sample.mem_reserved_mb is not None:
            return sample
        try:
            import torch

            if not torch.cuda.is_available() or torch.cuda.device_count() <= self.config.gpu_index:
                return sample
            device = torch.device(f"cuda:{self.config.gpu_index}")
            with torch.cuda.device(device):
                allocated = torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
                reserved = torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)
            if sample.mem_allocated_mb is None:
                sample.mem_allocated_mb = allocated
            if sample.mem_reserved_mb is None:
                sample.mem_reserved_mb = reserved
        except Exception:
            return sample
        return sample

    def _sample_loop(self) -> None:
        interval_s = max(0.001, float(self.config.sample_interval_ms) / 1000.0)
        while not self._stop.is_set():
            try:
                sample = self.backend.sample()
                sample = self._maybe_fill_torch_allocator_memory(sample)
                with self._samples_lock:
                    self._samples.append(sample)
            except Exception:
                # Sampling failures should not break model execution.
                pass
            time.sleep(interval_s)

    def start(self) -> None:
        if not self.config.enabled:
            return
        self._samples.clear()
        self.tracer.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True, name="runtime-profiler-sampler")
        self._thread.start()

    def stop(self) -> None:
        if not self.config.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    @contextmanager
    def trace(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Generator[None, None, None]:
        with self.tracer.trace(name=name, metadata=metadata):
            yield

    def trace_fn(self, name: Optional[str] = None):
        return self.tracer.trace_fn(name=name)

    def _max_metric(self, samples: List[GpuSample], field_name: str) -> Optional[Tuple[float, int]]:
        best_val = None
        best_ts = None
        for s in samples:
            value = getattr(s, field_name, None)
            if value is None:
                continue
            if best_val is None or value > best_val:
                best_val = float(value)
                best_ts = s.ts_ns
        if best_val is None or best_ts is None:
            return None
        return best_val, best_ts

    def _event_for_ts(self, events: List[TraceEvent], ts_ns: int) -> Optional[TraceEvent]:
        candidates = [e for e in events if e.start_ns <= ts_ns <= e.end_ns]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.depth)

    def _topk_metric_peaks(
        self, samples: List[GpuSample], metric: str, k: int
    ) -> List[Dict[str, Any]]:
        pairs = []
        for s in samples:
            value = getattr(s, metric, None)
            if value is None:
                continue
            pairs.append((float(value), s.ts_ns))
        pairs.sort(key=lambda x: x[0], reverse=True)
        return [{"value": v, "ts_ns": ts} for v, ts in pairs[:k]]

    def _build_summary(self, samples: List[GpuSample], events: List[TraceEvent]) -> Dict[str, Any]:
        bound: BoundClassification = classify_bound(samples)

        peaks = {
            "max_throughput": self._max_metric(samples, "sm_util"),
            "max_mem_util_percent": self._max_metric(samples, "mem_util"),
            "max_bandwidth_gbps": self._max_metric(samples, "bandwidth_gbps"),
            "max_bandwidth_tx_gbps": self._max_metric(samples, "bandwidth_tx_gbps"),
            "max_bandwidth_rx_gbps": self._max_metric(samples, "bandwidth_rx_gbps"),
            "max_memory_allocated_mb": self._max_metric(samples, "mem_allocated_mb"),
            "max_memory_reserved_mb": self._max_metric(samples, "mem_reserved_mb"),
        }
        peak_locations: Dict[str, Any] = {}
        for metric_name, peak in peaks.items():
            if peak is None:
                peak_locations[metric_name] = None
                continue
            _, ts_ns = peak
            event = self._event_for_ts(events, ts_ns)
            peak_locations[metric_name] = {
                "ts_ns": ts_ns,
                "function_name": event.name if event else None,
                "event_depth": event.depth if event else None,
            }

        def _peak_value(metric_key: str):
            peak = peaks[metric_key]
            return peak[0] if peak is not None else None

        summary = {
            "backend": self.backend.name,
            "capabilities": self.backend.capabilities(),
            "sample_count": len(samples),
            "event_count": len(events),
            "max_throughput": _peak_value("max_throughput"),
            "max_bandwidth": _peak_value("max_bandwidth_gbps"),
            "max_bandwidth_gbps": _peak_value("max_bandwidth_gbps"),
            "max_bandwidth_tx_gbps": _peak_value("max_bandwidth_tx_gbps"),
            "max_bandwidth_rx_gbps": _peak_value("max_bandwidth_rx_gbps"),
            "max_mem_util_percent": _peak_value("max_mem_util_percent"),
            "max_memory_allocated_mb": _peak_value("max_memory_allocated_mb"),
            "max_memory_reserved_mb": _peak_value("max_memory_reserved_mb"),
            "peak_locations": peak_locations,
            "topk_peaks": {
                "sm_util": self._topk_metric_peaks(samples, "sm_util", self.config.topk_peaks),
                "mem_util": self._topk_metric_peaks(samples, "mem_util", self.config.topk_peaks),
                "bandwidth_gbps": self._topk_metric_peaks(samples, "bandwidth_gbps", self.config.topk_peaks),
                "bandwidth_tx_gbps": self._topk_metric_peaks(
                    samples, "bandwidth_tx_gbps", self.config.topk_peaks
                ),
                "bandwidth_rx_gbps": self._topk_metric_peaks(
                    samples, "bandwidth_rx_gbps", self.config.topk_peaks
                ),
                "mem_allocated_mb": self._topk_metric_peaks(
                    samples, "mem_allocated_mb", self.config.topk_peaks
                ),
                "mem_reserved_mb": self._topk_metric_peaks(
                    samples, "mem_reserved_mb", self.config.topk_peaks
                ),
            },
            "bound_classification": bound.to_dict(),
        }
        return summary

    def _maybe_build_torch_profiler(self):
        if not self.config.enable_torch_profiler:
            return None, "disabled"
        try:
            import torch
            from torch.profiler import ProfilerActivity
        except Exception as e:
            return None, f"unavailable: {type(e).__name__}"
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        if not activities:
            return None, "no activities"
        try:
            profiler = torch.profiler.profile(
                activities=activities,
                record_shapes=self.config.torch_profiler_record_shapes,
                profile_memory=self.config.torch_profiler_profile_memory,
                with_stack=self.config.torch_profiler_with_stack,
                with_flops=self.config.torch_profiler_with_flops,
            )
            return profiler, None
        except Exception as e:
            return None, f"init_failed: {type(e).__name__}"

    def _extract_torch_timeline_from_trace(
        self,
        trace_json_path: str,
        sample_count: int,
        run_duration_s: float,
    ) -> Optional[Dict[str, List[float]]]:
        try:
            with Path(trace_json_path).open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None
        events = payload.get("traceEvents", [])
        if not isinstance(events, list) or not events:
            return None

        ts_values = [float(e.get("ts")) for e in events if isinstance(e, dict) and e.get("ts") is not None]
        if not ts_values:
            return None
        t0_us = min(ts_values)

        duration_s = run_duration_s
        if duration_s <= 0:
            max_us = max(ts_values)
            duration_s = max(1e-6, (max_us - t0_us) / 1e6)

        bin_count = min(4096, max(64, sample_count if sample_count > 0 else 256))
        bin_width_s = max(1e-6, duration_s / float(bin_count))
        active_seconds = [0.0 for _ in range(bin_count)]
        mem_event_mb = [0.0 for _ in range(bin_count)]

        def _event_bytes(args: Dict[str, Any]) -> Optional[float]:
            for k in ("bytes", "Bytes", "size", "num_bytes", "nbytes"):
                v = args.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
            return None

        for e in events:
            if not isinstance(e, dict):
                continue
            if e.get("ph") != "X":
                continue
            ts_us = e.get("ts")
            dur_us = e.get("dur")
            if ts_us is None or dur_us is None:
                continue
            start_s = (float(ts_us) - t0_us) / 1e6
            dur_s = max(0.0, float(dur_us) / 1e6)
            if dur_s <= 0.0:
                continue
            end_s = start_s + dur_s

            name = str(e.get("name", "")).lower()
            cat = str(e.get("cat", "")).lower()
            args = e.get("args") if isinstance(e.get("args"), dict) else {}

            is_cuda = ("cuda" in cat) or ("gpu" in cat) or ("kernel" in name)
            is_mem_event = ("memcpy" in name) or ("memory" in cat) or ("alloc" in name) or ("free" in name)
            event_bytes = _event_bytes(args) if is_mem_event else None

            left_bin = max(0, int(start_s / bin_width_s))
            right_bin = min(bin_count - 1, int(end_s / bin_width_s))
            for bi in range(left_bin, right_bin + 1):
                bin_l = bi * bin_width_s
                bin_r = (bi + 1) * bin_width_s
                overlap = max(0.0, min(end_s, bin_r) - max(start_s, bin_l))
                if overlap <= 0:
                    continue
                if is_cuda:
                    active_seconds[bi] += overlap
                if event_bytes is not None:
                    frac = overlap / dur_s
                    mem_event_mb[bi] += abs(event_bytes) * frac / (1024.0 * 1024.0)

        centers = [((i + 0.5) * bin_width_s) for i in range(bin_count)]
        active_ratio = [x / bin_width_s for x in active_seconds]
        mem_mb_s = [x / bin_width_s for x in mem_event_mb]
        return {
            "bin_centers_s": centers,
            "cuda_active_ratio": active_ratio,
            "cuda_mem_event_mb_per_s": mem_mb_s,
        }

    def _export_torch_profiler(
        self,
        output_dir: str,
        profiler_obj: Any,
        sample_count: int,
        run_duration_s: float,
    ) -> Dict[str, Any]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        ops_csv = out / "torch_profiler_ops.csv"
        rows: List[Dict[str, Any]] = []
        try:
            key_avgs = profiler_obj.key_averages()
            for evt in key_avgs:
                row = {
                    "op_name": getattr(evt, "key", None),
                    "count": getattr(evt, "count", None),
                    "cpu_time_total_us": getattr(evt, "cpu_time_total", None),
                    "self_cpu_time_total_us": getattr(evt, "self_cpu_time_total", None),
                    "cuda_time_total_us": getattr(evt, "cuda_time_total", None),
                    "self_cuda_time_total_us": getattr(evt, "self_cuda_time_total", None),
                    "cpu_memory_usage_bytes": getattr(evt, "cpu_memory_usage", None),
                    "self_cpu_memory_usage_bytes": getattr(evt, "self_cpu_memory_usage", None),
                    "cuda_memory_usage_bytes": getattr(evt, "cuda_memory_usage", None),
                    "self_cuda_memory_usage_bytes": getattr(evt, "self_cuda_memory_usage", None),
                    "flops": getattr(evt, "flops", None),
                }
                rows.append(row)
            rows.sort(
                key=lambda x: (
                    float(x["cuda_time_total_us"] or 0.0),
                    float(x["self_cuda_time_total_us"] or 0.0),
                    float(x["cpu_time_total_us"] or 0.0),
                ),
                reverse=True,
            )
            topk = rows[: max(1, int(self.config.torch_profiler_topk_ops))]
            with ops_csv.open("w", newline="", encoding="utf-8") as f:
                fieldnames = list(topk[0].keys()) if topk else []
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if fieldnames:
                    writer.writeheader()
                    for row in topk:
                        writer.writerow(row)
        except Exception:
            pass

        trace_path = None
        if self.config.torch_profiler_export_chrome_trace:
            try:
                trace_path = str(out / "torch_profiler_trace.json")
                profiler_obj.export_chrome_trace(trace_path)
            except Exception:
                trace_path = None

        timeline = None
        if trace_path is not None:
            timeline = self._extract_torch_timeline_from_trace(
                trace_json_path=trace_path,
                sample_count=sample_count,
                run_duration_s=run_duration_s,
            )

        return {
            "enabled": True,
            "ops_csv": str(ops_csv) if ops_csv.exists() else None,
            "trace_json": trace_path,
            "op_rows_total": len(rows),
            "timeline": timeline,
            "timeline_note": (
                "cuda_active_ratio is kernel active-time density (proxy for compute-unit occupancy); "
                "cuda_mem_event_mb_per_s is memory event rate parsed from torch trace events."
            ),
        }

    def _export(
        self,
        output_dir: str,
        samples: List[GpuSample],
        events: List[TraceEvent],
        summary: Dict[str, Any],
    ) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        samples_csv = out / "profile_samples.csv"
        with samples_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "ts_ns",
                "gpu_util",
                "sm_util",
                "mem_util",
                "mem_allocated_mb",
                "mem_reserved_mb",
                "mem_total_mb",
                "power_w",
                "temperature_c",
                "bandwidth_gbps",
                "bandwidth_tx_gbps",
                "bandwidth_rx_gbps",
                "backend",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in samples:
                writer.writerow(s.to_dict())

        events_json = out / "profile_events.json"
        with events_json.open("w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in events], f, indent=2, ensure_ascii=False)

        summary_json = out / "profile_summary.json"
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        if self.config.auto_plot:
            torch_timeline = None
            torch_profiler_summary = summary.get("torch_profiler", {})
            if isinstance(torch_profiler_summary, dict):
                torch_timeline = torch_profiler_summary.get("timeline")
            plot_timeline(
                samples,
                str(out / "profile_timeline.png"),
                torch_timeline=torch_timeline,
            )
            plot_event_timeline(events, str(out / "profile_events_timeline.png"))

    def run(self, target_fn: Callable[..., Any], *args, **kwargs) -> ProfileResult:
        if not self.config.enabled:
            out = Path(self.config.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            result = target_fn(*args, **kwargs)
            summary = {
                "enabled": False,
                "message": "Runtime profiler is disabled",
                "backend": self.backend.name,
            }
            return ProfileResult(
                result=result,
                samples=[],
                events=[],
                summary=summary,
                output_dir=str(out),
            )

        self.start()
        run_result = None
        torch_profiler_ctx, torch_profiler_reason = self._maybe_build_torch_profiler()
        run_start_ns = time.perf_counter_ns()
        try:
            if torch_profiler_ctx is None:
                with self.trace("target_fn"):
                    run_result = target_fn(*args, **kwargs)
            else:
                with torch_profiler_ctx:
                    with self.trace("target_fn"):
                        run_result = target_fn(*args, **kwargs)
        finally:
            run_end_ns = time.perf_counter_ns()
            self.stop()

        with self._samples_lock:
            samples = list(self._samples)
        events = self.tracer.get_events()
        summary = self._build_summary(samples, events)
        run_duration_s = max(0.0, float(run_end_ns - run_start_ns) / 1e9)
        summary["run_window"] = {
            "start_perf_counter_ns": run_start_ns,
            "end_perf_counter_ns": run_end_ns,
            "duration_s": run_duration_s,
        }
        if torch_profiler_ctx is not None:
            summary["torch_profiler"] = self._export_torch_profiler(
                self.config.output_dir,
                torch_profiler_ctx,
                sample_count=len(samples),
                run_duration_s=run_duration_s,
            )
        else:
            summary["torch_profiler"] = {"enabled": False, "reason": torch_profiler_reason}
        self._export(self.config.output_dir, samples, events, summary)
        return ProfileResult(
            result=run_result,
            samples=samples,
            events=events,
            summary=summary,
            output_dir=self.config.output_dir,
        )
