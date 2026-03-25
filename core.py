import csv
import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from .classifiers import BoundClassification, build_classification, classify_event_online
from .event_aggregate import (
    aggregate_events,
    enrich_samples_with_window,
    sort_top_events,
)
from .event_tracer import EventTracer, TraceEvent
from .merge_diagnosis import merge_online_offline
from .metrics_backends import GpuSample, MetricsBackend, build_backend
from .offline.ncu_parser import parse_ncu_csv_to_kernels, write_kernel_artifacts
from .offline.ncu_runner import NCUConfig, run_ncu
from .plotter import (
    plot_bound_evidence_heatmap,
    plot_event_summary_bars,
    plot_event_timeline,
    plot_kernel_drilldown,
    plot_main_figure,
)


PROFILE_SAMPLE_FIELDNAMES = [
    "ts_ns",
    "rel_time_ms",
    "in_run_window",
    "active_event_name",
    "active_event_depth",
    "gpu_util_pct",
    "sm_util_pct",
    "sm_occupancy_pct",
    "tensor_util_pct",
    "fp16_util_pct",
    "fp32_util_pct",
    "int_util_pct",
    "sm_clock_mhz",
    "graphics_clock_mhz",
    "power_w",
    "temperature_c",
    "dram_bw_util_pct",
    "dram_bw_proxy_gbps",
    "mem_util_pct",
    "alloc_mem_mb",
    "reserved_mem_mb",
    "active_mem_mb",
    "total_mem_mb",
    "mem_clock_mhz",
    "pcie_tx_mib_s",
    "pcie_rx_mib_s",
    "pcie_total_mib_s",
    "nvlink_tx_mib_s",
    "nvlink_rx_mib_s",
    "source_compute",
    "source_device_memory",
    "source_io",
    "source_alloc_mem",
    "backend",
]


@dataclass
class RuntimeProfilerConfig:
    enabled: bool = True
    output_dir: str = "runtime_profile"
    sample_interval_ms: int = 20
    gpu_index: int = 0
    backend_preference: List[str] = field(default_factory=lambda: ["nvml_gpm", "nvml_legacy", "torch_cuda"])
    topk_peaks: int = 3
    auto_plot: bool = True
    detail_plot_mode: bool = False
    enable_torch_profiler: bool = False
    torch_profiler_record_shapes: bool = False
    torch_profiler_profile_memory: bool = True
    torch_profiler_with_stack: bool = False
    torch_profiler_with_flops: bool = False
    torch_profiler_export_chrome_trace: bool = True
    torch_profiler_topk_ops: int = 200
    # Offline NCU (optional second pass; long-running)
    ncu: Dict[str, Any] = field(default_factory=dict)


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
        self._run_start_perf_ns: Optional[int] = None

    def _maybe_fill_torch_allocator_memory(self, sample: GpuSample) -> GpuSample:
        if sample.alloc_mem_mb is not None and sample.reserved_mem_mb is not None:
            sample.source_alloc_mem = sample.source_alloc_mem or "torch"
            sample.sync_legacy_aliases()
            return sample
        try:
            import torch

            if not torch.cuda.is_available() or torch.cuda.device_count() <= self.config.gpu_index:
                return sample
            device = torch.device(f"cuda:{self.config.gpu_index}")
            with torch.cuda.device(device):
                allocated = torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)
                reserved = torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)
            if sample.alloc_mem_mb is None:
                sample.alloc_mem_mb = allocated
            if sample.reserved_mem_mb is None:
                sample.reserved_mem_mb = reserved
            if sample.active_mem_mb is None:
                sample.active_mem_mb = allocated
            sample.source_alloc_mem = sample.source_alloc_mem or "torch"
        except Exception:
            return sample
        sample.sync_legacy_aliases()
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
                pass
            time.sleep(interval_s)

    def start(self) -> None:
        if not self.config.enabled:
            return
        self._samples.clear()
        self.tracer.clear()
        self._stop.clear()
        self._run_start_perf_ns = time.perf_counter_ns()
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

    def _build_summary(
        self,
        samples: List[GpuSample],
        events: List[TraceEvent],
        run_duration_s: float,
        offline_kernel_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        offline_hints = None
        if offline_kernel_summary:
            offline_hints = {
                "dominant_label": offline_kernel_summary.get("dominant_label"),
                "kernel_coverage_fraction": offline_kernel_summary.get("kernel_coverage_fraction"),
            }

        classification = build_classification(
            samples,
            offline_hints=offline_hints,
            duration_s=run_duration_s,
        )
        bound = BoundClassification(
            label=classification["label"],
            confidence=classification["confidence"],
            evidence={"structured": classification},
        )

        peaks = {
            "max_sm_util_pct": self._max_metric(samples, "sm_util_pct"),
            "max_dram_bw_util_pct": self._max_metric(samples, "dram_bw_util_pct"),
            "max_pcie_total_mib_s": self._max_metric(samples, "pcie_total_mib_s"),
            "max_mem_util_pct": self._max_metric(samples, "mem_util_pct"),
            "max_memory_allocated_mb": self._max_metric(samples, "alloc_mem_mb"),
            "max_memory_reserved_mb": self._max_metric(samples, "reserved_mem_mb"),
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

        rollups = aggregate_events(samples, events)
        rollup_dicts = [r.to_dict() for r in rollups]

        event_records: List[Dict[str, Any]] = []
        event_phase_labels: Dict[str, Dict[str, Any]] = {}
        for ev, r in zip(events, rollups):
            ws = [s for s in samples if ev.start_ns <= s.ts_ns <= ev.end_ns]
            evc = classify_event_online(ws, r.duration_ms)
            d = ev.to_dict()
            d["metrics_rollup"] = r.to_dict()
            d["event_classification"] = evc
            event_records.append(d)
            event_phase_labels[ev.name] = evc

        summary = {
            "backend": self.backend.name,
            "capabilities": self.backend.capabilities(),
            "resource_domains": {
                "compute": "SM / Tensor / FP / INT",
                "device_memory": "HBM / GDDR / on-device DRAM traffic vs capacity",
                "host_device_io": "PCIe (NVLink extensible)",
            },
            "sample_count": len(samples),
            "event_count": len(events),
            "max_sm_util_pct": _peak_value("max_sm_util_pct"),
            "max_dram_bw_util_pct": _peak_value("max_dram_bw_util_pct"),
            "max_pcie_total_mib_s": _peak_value("max_pcie_total_mib_s"),
            "max_mem_util_percent": _peak_value("max_mem_util_pct"),
            "max_memory_allocated_mb": _peak_value("max_memory_allocated_mb"),
            "max_memory_reserved_mb": _peak_value("max_memory_reserved_mb"),
            "peak_locations": peak_locations,
            "topk_peaks": {
                "sm_util_pct": self._topk_metric_peaks(samples, "sm_util_pct", self.config.topk_peaks),
                "dram_bw_util_pct": self._topk_metric_peaks(
                    samples, "dram_bw_util_pct", self.config.topk_peaks
                ),
                "pcie_total_mib_s": self._topk_metric_peaks(
                    samples, "pcie_total_mib_s", self.config.topk_peaks
                ),
                "mem_util_pct": self._topk_metric_peaks(samples, "mem_util_pct", self.config.topk_peaks),
                "alloc_mem_mb": self._topk_metric_peaks(samples, "alloc_mem_mb", self.config.topk_peaks),
            },
            "classification": classification,
            "bound_classification": bound.to_dict(),
            "top_memory_pressure_events": sort_top_events(rollups, "memory_pressure", self.config.topk_peaks),
            "top_compute_pressure_events": sort_top_events(rollups, "compute_pressure", self.config.topk_peaks),
            "top_pcie_io_events": sort_top_events(rollups, "pcie_io", self.config.topk_peaks),
            "top_latency_bound_events": sort_top_events(rollups, "latency_bound", self.config.topk_peaks),
            "event_rollups_csv_ready": rollup_dicts,
            "profile_events_enriched": event_records,
        }

        merged = merge_online_offline(
            classification,
            event_phase_labels,
            offline_kernel_summary,
        )
        summary["merged_diagnosis_preview"] = merged

        # Legacy flat keys used by older scripts
        summary["max_throughput"] = summary["max_sm_util_pct"]
        summary["max_bandwidth_gbps"] = None
        summary["max_bandwidth_tx_gbps"] = None
        summary["max_bandwidth_rx_gbps"] = None

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
                "cuda_active_ratio is kernel active-time density from torch trace; "
                "not hardware SM occupancy."
            ),
        }

    def _write_sample_row(self, s: GpuSample) -> Dict[str, Any]:
        s.sync_legacy_aliases()
        d = {k: getattr(s, k, None) for k in PROFILE_SAMPLE_FIELDNAMES}
        return d

    def _maybe_run_ncu(self, output_dir: str) -> Dict[str, Any]:
        raw = self.config.ncu or {}
        if not raw.get("enabled"):
            return {"enabled": False}
        cfg = NCUConfig.from_dict(raw)
        cfg.output_dir = str(Path(output_dir) / "ncu")
        res = run_ncu(cfg)
        return {"enabled": True, "result": res}

    def _maybe_load_offline_kernels(self, output_dir: str) -> Optional[Dict[str, Any]]:
        p = Path(output_dir) / "kernel_bound_summary.json"
        if not p.exists():
            p2 = Path(output_dir) / "ncu" / "kernel_bound_summary.json"
            p = p2 if p2.exists() else p
        if not p.exists():
            return None
        try:
            with p.open(encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

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
            writer = csv.DictWriter(f, fieldnames=PROFILE_SAMPLE_FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            for s in samples:
                writer.writerow(self._write_sample_row(s))

        events_json = out / "profile_events.json"
        enriched = summary.get("profile_events_enriched", [e.to_dict() for e in events])
        with events_json.open("w", encoding="utf-8") as f:
            json.dump(enriched, f, indent=2, ensure_ascii=False)

        rollups = summary.get("event_rollups_csv_ready", [])
        ev_sum_csv = out / "profile_event_summary.csv"
        if rollups:
            with ev_sum_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rollups[0].keys()))
                w.writeheader()
                for row in rollups:
                    w.writerow(row)

        summary_json = out / "profile_summary.json"
        summary_copy = dict(summary)
        summary_copy.pop("event_rollups_csv_ready", None)
        summary_copy.pop("profile_events_enriched", None)
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary_copy, f, indent=2, ensure_ascii=False)

        rollup_objs = aggregate_events(samples, events)
        ks = summary.get("offline_kernel_summary") or {}
        kernel_rows = ks.get("kernels") if isinstance(ks, dict) else None
        if self.config.auto_plot:
            plot_main_figure(
                samples,
                events,
                str(out / "profile_main_figure.png"),
                detail_mode=self.config.detail_plot_mode,
            )
            plot_event_timeline(events, str(out / "profile_events_timeline.png"))
            if rollup_objs:
                plot_event_summary_bars(rollup_objs, str(out / "profile_event_bars.png"))
                plot_bound_evidence_heatmap(rollup_objs, str(out / "profile_event_heatmap.png"))
            if kernel_rows:
                plot_kernel_drilldown(kernel_rows, str(out / "kernel_drilldown.png"))

        merged_path = out / "merged_diagnosis_report.json"
        with merged_path.open("w", encoding="utf-8") as f:
            json.dump(summary.get("merged_diagnosis_preview", {}), f, indent=2, ensure_ascii=False)

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
        run_duration_s = max(0.0, float(run_end_ns - run_start_ns) / 1e9)
        enrich_samples_with_window(samples, events, run_start_ns, run_end_ns)

        offline_ncu_report: Dict[str, Any] = {"enabled": bool((self.config.ncu or {}).get("enabled"))}
        kernel_summary_prebuilt: Optional[Dict[str, Any]] = None
        if offline_ncu_report["enabled"]:
            offline_ncu_report["run"] = self._maybe_run_ncu(self.config.output_dir)
            ncu_dir = Path(self.config.output_dir) / "ncu"
            if ncu_dir.is_dir():
                csv_files = list(ncu_dir.glob("*.csv"))
                preferred = ncu_dir / "ncu_report.csv"
                ordered = [preferred] if preferred.exists() else []
                ordered.extend(sorted(f for f in csv_files if f != preferred))
                for f in ordered:
                    rows = parse_ncu_csv_to_kernels(str(f))
                    if rows:
                        arts = write_kernel_artifacts(rows, str(ncu_dir))
                        offline_ncu_report["artifacts"] = arts
                        kernel_summary_prebuilt = json.loads(
                            Path(arts["kernel_bound_summary_json"]).read_text(encoding="utf-8")
                        )
                        break

        if kernel_summary_prebuilt is None:
            kernel_summary_prebuilt = self._maybe_load_offline_kernels(self.config.output_dir)

        summary = self._build_summary(samples, events, run_duration_s, offline_kernel_summary=kernel_summary_prebuilt)
        summary["run_window"] = {
            "start_perf_counter_ns": run_start_ns,
            "end_perf_counter_ns": run_end_ns,
            "duration_s": run_duration_s,
        }
        summary["offline_ncu"] = offline_ncu_report
        if kernel_summary_prebuilt:
            summary["offline_kernel_summary"] = kernel_summary_prebuilt

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
