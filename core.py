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

    def _sample_loop(self) -> None:
        interval_s = max(0.001, float(self.config.sample_interval_ms) / 1000.0)
        while not self._stop.is_set():
            try:
                sample = self.backend.sample()
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
            "max_bandwidth": self._max_metric(samples, "mem_util"),
            "max_memory_used_mb": self._max_metric(samples, "mem_used_mb"),
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
            "max_bandwidth": _peak_value("max_bandwidth"),
            "max_memory_used_mb": _peak_value("max_memory_used_mb"),
            "peak_locations": peak_locations,
            "topk_peaks": {
                "sm_util": self._topk_metric_peaks(samples, "sm_util", self.config.topk_peaks),
                "mem_util": self._topk_metric_peaks(samples, "mem_util", self.config.topk_peaks),
                "mem_used_mb": self._topk_metric_peaks(samples, "mem_used_mb", self.config.topk_peaks),
            },
            "bound_classification": bound.to_dict(),
        }
        return summary

    def _export(self, output_dir: str, samples: List[GpuSample], events: List[TraceEvent], summary: Dict[str, Any]) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        samples_csv = out / "profile_samples.csv"
        with samples_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "ts_ns",
                "gpu_util",
                "sm_util",
                "mem_util",
                "mem_used_mb",
                "mem_total_mb",
                "power_w",
                "temperature_c",
                "bandwidth_gbps",
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
            plot_timeline(samples, str(out / "profile_timeline.png"))
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
        try:
            with self.trace("target_fn"):
                run_result = target_fn(*args, **kwargs)
        finally:
            self.stop()

        with self._samples_lock:
            samples = list(self._samples)
        events = self.tracer.get_events()
        summary = self._build_summary(samples, events)
        self._export(self.config.output_dir, samples, events, summary)
        return ProfileResult(
            result=run_result,
            samples=samples,
            events=events,
            summary=summary,
            output_dir=self.config.output_dir,
        )
