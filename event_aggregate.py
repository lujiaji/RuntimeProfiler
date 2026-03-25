"""Aggregate online samples onto trace events and build per-event statistics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .event_tracer import TraceEvent
from .metrics_backends import GpuSample


def _p95(values: List[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    idx = int(math.ceil(0.95 * len(s))) - 1
    idx = max(0, min(idx, len(s) - 1))
    return float(s[idx])


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _max(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(max(values))


def samples_in_event_window(samples: Sequence[GpuSample], ev: TraceEvent) -> List[GpuSample]:
    return [s for s in samples if ev.start_ns <= s.ts_ns <= ev.end_ns]


def active_event_at_ts(events: Sequence[TraceEvent], ts_ns: int) -> Optional[TraceEvent]:
    candidates = [e for e in events if e.start_ns <= ts_ns <= e.end_ns]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e.depth)


def enrich_samples_with_window(
    samples: List[GpuSample],
    events: List[TraceEvent],
    run_start_ns: int,
    run_end_ns: int,
) -> None:
    """Mutate samples in place: rel_time_ms, in_run_window, active event."""
    for s in samples:
        s.rel_time_ms = (s.ts_ns - run_start_ns) / 1e6
        s.in_run_window = bool(run_start_ns <= s.ts_ns <= run_end_ns)
        ae = active_event_at_ts(events, s.ts_ns)
        if ae is not None:
            s.active_event_name = ae.name
            s.active_event_depth = ae.depth
        else:
            s.active_event_name = None
            s.active_event_depth = None
        s.sync_legacy_aliases()


@dataclass
class EventMetricRollup:
    event_name: str
    depth: int
    duration_ms: float
    sample_count: int
    avg_sm_util_pct: Optional[float]
    p95_sm_util_pct: Optional[float]
    max_sm_util_pct: Optional[float]
    avg_sm_occupancy_pct: Optional[float]
    p95_sm_occupancy_pct: Optional[float]
    max_sm_occupancy_pct: Optional[float]
    avg_tensor_util_pct: Optional[float]
    p95_tensor_util_pct: Optional[float]
    max_tensor_util_pct: Optional[float]
    avg_dram_bw_util_pct: Optional[float]
    p95_dram_bw_util_pct: Optional[float]
    max_dram_bw_util_pct: Optional[float]
    avg_pcie_total_mib_s: Optional[float]
    avg_pcie_rx_mib_s: Optional[float]
    avg_pcie_tx_mib_s: Optional[float]
    p95_pcie_total_mib_s: Optional[float]
    max_pcie_total_mib_s: Optional[float]
    avg_alloc_mem_mb: Optional[float]
    max_alloc_mem_mb: Optional[float]
    avg_reserved_mem_mb: Optional[float]
    max_reserved_mem_mb: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def rollup_event(ev: TraceEvent, window_samples: List[GpuSample]) -> EventMetricRollup:
    dur_ms = (ev.end_ns - ev.start_ns) / 1e6
    def col(getter) -> List[float]:
        out: List[float] = []
        for s in window_samples:
            v = getter(s)
            if v is not None:
                out.append(float(v))
        return out

    sm = col(lambda s: s.sm_util_pct)
    occ = col(lambda s: s.sm_occupancy_pct)
    ten = col(lambda s: s.tensor_util_pct)
    dram = col(lambda s: s.dram_bw_util_pct)
    pcie = col(lambda s: s.pcie_total_mib_s)
    pcie_rx = col(lambda s: s.pcie_rx_mib_s)
    pcie_tx = col(lambda s: s.pcie_tx_mib_s)
    alloc = col(lambda s: s.alloc_mem_mb)
    rsv = col(lambda s: s.reserved_mem_mb)

    return EventMetricRollup(
        event_name=ev.name,
        depth=ev.depth,
        duration_ms=dur_ms,
        sample_count=len(window_samples),
        avg_sm_util_pct=_mean(sm),
        p95_sm_util_pct=_p95(sm),
        max_sm_util_pct=_max(sm),
        avg_sm_occupancy_pct=_mean(occ),
        p95_sm_occupancy_pct=_p95(occ),
        max_sm_occupancy_pct=_max(occ),
        avg_tensor_util_pct=_mean(ten),
        p95_tensor_util_pct=_p95(ten),
        max_tensor_util_pct=_max(ten),
        avg_dram_bw_util_pct=_mean(dram),
        p95_dram_bw_util_pct=_p95(dram),
        max_dram_bw_util_pct=_max(dram),
        avg_pcie_total_mib_s=_mean(pcie),
        avg_pcie_rx_mib_s=_mean(pcie_rx),
        avg_pcie_tx_mib_s=_mean(pcie_tx),
        p95_pcie_total_mib_s=_p95(pcie),
        max_pcie_total_mib_s=_max(pcie),
        avg_alloc_mem_mb=_mean(alloc),
        max_alloc_mem_mb=_max(alloc),
        avg_reserved_mem_mb=_mean(rsv),
        max_reserved_mem_mb=_max(rsv),
    )


def aggregate_events(samples: List[GpuSample], events: List[TraceEvent]) -> List[EventMetricRollup]:
    rollups: List[EventMetricRollup] = []
    for ev in events:
        ws = samples_in_event_window(samples, ev)
        rollups.append(rollup_event(ev, ws))
    return rollups


def sort_top_events(
    rollups: Sequence[EventMetricRollup],
    key: str,
    k: int,
) -> List[Dict[str, Any]]:
    """key: memory_pressure | compute_pressure | pcie_io | latency_bound"""
    scored: List[Tuple[float, EventMetricRollup]] = []
    for r in rollups:
        if key == "memory_pressure":
            score = r.avg_dram_bw_util_pct or r.max_dram_bw_util_pct or 0.0
        elif key == "compute_pressure":
            score = r.avg_sm_util_pct or r.max_sm_util_pct or 0.0
            score = max(score, r.avg_tensor_util_pct or 0.0)
        elif key == "pcie_io":
            score = r.avg_pcie_total_mib_s or r.max_pcie_total_mib_s or 0.0
        elif key == "latency_bound":
            sm = r.avg_sm_util_pct or 0.0
            dram = r.avg_dram_bw_util_pct or 0.0
            pcie = r.avg_pcie_total_mib_s or 0.0
            low = 35.0
            if sm < low and dram < low and pcie < (r.duration_ms / 1000.0 * 100.0 + 1.0):
                score = r.duration_ms
            else:
                score = 0.0
        else:
            score = 0.0
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict[str, Any]] = []
    for sc, r in scored[:k]:
        d = r.to_dict()
        d["_score"] = sc
        out.append(d)
    return out
