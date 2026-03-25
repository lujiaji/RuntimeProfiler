"""
Bottleneck classification: rule + evidence + confidence (not simple mean scoring).

Labels: compute_bound, memory_bound, pcie_io_bound, mixed_bound, latency_bound
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .metrics_backends import GpuSample

SM_HIGH = 62.0
DRAM_HIGH = 62.0
OCC_LOW = 25.0
PCIE_FRAC_OF_MAX = 0.45
UTIL_LOW = 38.0


@dataclass
class BoundClassification:
    """Backward-compatible minimal shape; prefer classification dict in summary."""

    label: str
    confidence: float
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "confidence": self.confidence, "evidence": self.evidence}


def _mean(vals: Sequence[float]) -> Optional[float]:
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _p95_simple(values: List[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    import math

    idx = int(math.ceil(0.95 * len(s))) - 1
    idx = max(0, min(idx, len(s) - 1))
    return float(s[idx])


def _collect_series(samples: Sequence[GpuSample]) -> Dict[str, List[float]]:
    keys = [
        "sm_util_pct",
        "sm_occupancy_pct",
        "tensor_util_pct",
        "fp16_util_pct",
        "fp32_util_pct",
        "dram_bw_util_pct",
        "dram_bw_proxy_gbps",
        "pcie_total_mib_s",
        "pcie_rx_mib_s",
        "pcie_tx_mib_s",
        "gpu_util_pct",
        "mem_util_pct",
    ]
    out: Dict[str, List[float]] = {k: [] for k in keys}
    for s in samples:
        for k in keys:
            v = getattr(s, k, None)
            if v is not None:
                out[k].append(float(v))
    return out


def build_metric_coverage(samples: Sequence[GpuSample]) -> Dict[str, Any]:
    """Which domains have real signals vs proxies / missing."""
    if not samples:
        return {
            "compute": "none",
            "device_memory": "none",
            "io": "none",
            "alloc_mem": "none",
        }
    sc = samples[0].source_compute
    sd = samples[0].source_device_memory
    si = samples[0].source_io
    sa = samples[0].source_alloc_mem
    for s in samples[1:]:
        if s.source_compute != sc:
            sc = "mixed"
        if s.source_device_memory != sd:
            sd = "mixed"
        if s.source_io != si:
            si = "mixed"
        if s.source_alloc_mem != sa:
            sa = "mixed"
    return {
        "compute": sc or "none",
        "device_memory": sd or "none",
        "io": si or "none",
        "alloc_mem": sa or "none",
    }


def build_limitations(
    coverage: Dict[str, Any],
    has_gpm_dram: bool,
    has_measured_pcie: bool,
) -> List[str]:
    lim: List[str] = []
    if coverage.get("compute") == "none":
        lim.append("No compute utilization signal; SM/tensor evidence unavailable.")
    if coverage.get("device_memory") in ("none", "nvml_legacy", "proxy"):
        if not has_gpm_dram:
            lim.append(
                "Device DRAM bandwidth is not measured (NVML GPM DRAM BW util missing); "
                "memory_bound vs compute_bound discrimination is weaker."
            )
    if coverage.get("alloc_mem") == "torch" and coverage.get("compute") == "none":
        lim.append(
            "Torch-only allocator metrics: cannot issue high-confidence compute_bound or memory_bound "
            "(throughput-bound vs capacity)."
        )
    if not has_measured_pcie and coverage.get("io") != "nvml_gpm":
        lim.append("PCIe counters may be legacy KB/s rollup; interpret host-device pressure carefully.")
    return lim


def _score_compute(series: Dict[str, List[float]], occ: List[float]) -> Tuple[float, List[str]]:
    evidence: List[str] = []
    score = 0.0
    sm_m = _mean(series.get("sm_util_pct", []))
    ten_m = _mean(series.get("tensor_util_pct", []))
    fp16_m = _mean(series.get("fp16_util_pct", []))
    fp32_m = _mean(series.get("fp32_util_pct", []))
    if sm_m is not None and sm_m >= SM_HIGH:
        score += 1.2
        evidence.append(f"avg_sm_util_pct={sm_m:.1f}% (high)")
    if ten_m is not None and ten_m >= SM_HIGH:
        score += 1.0
        evidence.append(f"avg_tensor_util_pct={ten_m:.1f}% (high)")
    if fp16_m is not None and fp16_m >= SM_HIGH:
        score += 0.6
        evidence.append(f"avg_fp16_util_pct={fp16_m:.1f}% (high)")
    if fp32_m is not None and fp32_m >= SM_HIGH:
        score += 0.6
        evidence.append(f"avg_fp32_util_pct={fp32_m:.1f}% (high)")
    occ_m = _mean(occ)
    if occ_m is not None and occ_m < OCC_LOW and sm_m is not None and sm_m >= SM_HIGH:
        score -= 0.4
        evidence.append(f"sm_occupancy_pct low ({occ_m:.1f}%) vs high SM util → possible launch/sync inefficiency")
    return score, evidence


def _score_memory(
    series: Dict[str, List[float]],
    has_gpm_dram: bool,
) -> Tuple[float, List[str]]:
    evidence: List[str] = []
    score = 0.0
    dram_m = _mean(series.get("dram_bw_util_pct", []))
    dram_p95 = _p95_simple(series.get("dram_bw_util_pct", []))
    if dram_m is not None and dram_m >= DRAM_HIGH:
        score += 1.5 if has_gpm_dram else 0.7
        tag = "measured_dram_bw_util" if has_gpm_dram else "dram_bw_util_proxy_weak"
        evidence.append(f"{tag}: avg={dram_m:.1f}% p95={dram_p95 or 0:.1f}%")
    proxy = _mean(series.get("dram_bw_proxy_gbps", []))
    if (dram_m is None or not has_gpm_dram) and proxy is not None:
        evidence.append(f"dram_bw_proxy_gbps avg={proxy:.2f} (not device DRAM measured throughput)")
    mem_u = _mean(series.get("mem_util_pct", []))
    if mem_u is not None:
        evidence.append(
            f"mem_util_pct avg={mem_u:.1f}% (coarse busy/capacity indicator, not memory_throughput_bound alone)"
        )
    return score, evidence


def _score_pcie(series: Dict[str, List[float]]) -> Tuple[float, List[str]]:
    evidence: List[str] = []
    pcie = series.get("pcie_total_mib_s", [])
    if not pcie:
        return 0.0, evidence
    mx = max(pcie)
    thr = max(50.0, mx * PCIE_FRAC_OF_MAX)
    high_frac = sum(1 for x in pcie if x >= thr) / max(1, len(pcie))
    avg = _mean(pcie)
    rx_m = _mean(series.get("pcie_rx_mib_s", []))
    tx_m = _mean(series.get("pcie_tx_mib_s", []))
    score = 0.0
    if avg is not None and mx > 0 and high_frac >= 0.25:
        score += 1.2
        evidence.append(
            f"pcie_total_mib_s sustained: avg={avg:.1f} max={mx:.1f} (host-device IO pressure)"
        )
        if rx_m is not None and tx_m is not None:
            if rx_m > tx_m * 1.3:
                evidence.append("PCIe RX dominates → likely H2D / feed weights or KV fetch patterns")
            elif tx_m > rx_m * 1.3:
                evidence.append("PCIe TX dominates → likely D2H / offload or result staging")
            else:
                evidence.append("PCIe RX/TX both elevated → bidirectional paging or mixed IO")
    return score, evidence


def _score_latency(
    series: Dict[str, List[float]],
    duration_s: float,
    sample_count: int,
) -> Tuple[float, List[str]]:
    evidence: List[str] = []
    sm_m = _mean(series.get("sm_util_pct", [])) or 0.0
    dram_m = _mean(series.get("dram_bw_util_pct", [])) or 0.0
    pcie_m = _mean(series.get("pcie_total_mib_s", [])) or 0.0
    occ_m = _mean(series.get("sm_occupancy_pct", []))
    score = 0.0
    if (
        sm_m < UTIL_LOW
        and dram_m < UTIL_LOW
        and pcie_m < max(80.0, UTIL_LOW * 2)
        and duration_s > 0.05
        and sample_count >= 3
    ):
        score += 1.0
        evidence.append(
            f"Low SM/DRAM/PCIe vs runtime (duration~{duration_s:.3f}s): underutilization / sync / small kernels suspected"
        )
    if occ_m is not None and occ_m < OCC_LOW:
        score += 0.5
        evidence.append(f"Low sm_occupancy_pct (~{occ_m:.1f}%) → warps not resident / latency style stalls")
    return score, evidence


def _confidence_from_sources(
    coverage: Dict[str, Any],
    has_gpm_compute: bool,
    has_gpm_dram: bool,
    has_offline: bool,
) -> float:
    if has_offline:
        return 0.88
    if has_gpm_compute and coverage.get("compute") == "nvml_gpm":
        base = 0.72
        if has_gpm_dram and coverage.get("device_memory") == "nvml_gpm":
            base += 0.08
        return min(0.9, base)
    if coverage.get("compute") == "nvml_legacy":
        return 0.48
    if coverage.get("alloc_mem") == "torch" and coverage.get("compute") == "none":
        return 0.22
    return 0.35


def classify_bound(samples: List[GpuSample]) -> BoundClassification:
    """Backward-compatible wrapper; evidence embeds full structured classification."""
    full = build_classification(samples, offline_hints=None, duration_s=None)
    return BoundClassification(
        label=str(full["label"]),
        confidence=float(full["confidence"]),
        evidence={"structured": full},
    )


def build_classification(
    samples: List[GpuSample],
    offline_hints: Optional[Dict[str, Any]],
    duration_s: Optional[float],
) -> Dict[str, Any]:
    """
    Returns dict matching summary classification schema:
    label, confidence, primary_evidence, secondary_evidence, metric_coverage, limitations
    """
    coverage = build_metric_coverage(samples)
    series = _collect_series(samples)
    has_gpm_dram = any(s.dram_bw_util_pct is not None and s.source_device_memory == "nvml_gpm" for s in samples)
    has_measured_pcie = any(s.source_io == "nvml_gpm" for s in samples)
    limitations = build_limitations(coverage, has_gpm_dram, has_measured_pcie)

    dur = duration_s
    if dur is None and samples:
        dur = (samples[-1].ts_ns - samples[0].ts_ns) / 1e9
    dur = dur or 0.0

    c_score, c_ev = _score_compute(series, series.get("sm_occupancy_pct", []))
    m_score, m_ev = _score_memory(series, has_gpm_dram)
    p_score, p_ev = _score_pcie(series)
    l_score, l_ev = _score_latency(series, dur, len(samples))

    if offline_hints:
        limitations.append(
            "Offline kernel profiler hints merged; see offline section for kernel-level evidence."
        )

    votes: List[Tuple[str, float, List[str]]] = [
        ("compute_bound", c_score, c_ev),
        ("memory_bound", m_score, m_ev),
        ("pcie_io_bound", p_score, p_ev),
        ("latency_bound", l_score, l_ev),
    ]
    votes.sort(key=lambda x: x[1], reverse=True)
    top_label, top_score, top_ev = votes[0]
    second_label, second_score, second_ev = votes[1] if len(votes) > 1 else ("unknown", 0.0, [])

    primary_evidence: List[str] = list(top_ev)
    secondary_evidence: List[str] = []
    label = top_label

    strong = [v for v in votes if v[1] >= 1.0]
    if len(strong) >= 2:
        label = "mixed_bound"
        primary_evidence = [f"Multiple resource domains show strong signals: {', '.join(v[0] for v in strong)}"]
        for v in strong:
            primary_evidence.extend(v[2][:2])
        secondary_evidence = [
            f"runner-up {second_label} score={second_score:.2f}",
            *second_ev[:1],
        ]
    elif top_score < 0.45:
        label = "latency_bound" if l_score >= 0.5 else "mixed_bound"
        primary_evidence = top_ev or ["No dominant online bottleneck signature; treat as mixed or latency."]
        secondary_evidence = [f"scores compute={c_score:.2f} mem={m_score:.2f} pcie={p_score:.2f} lat={l_score:.2f}"]
    else:
        secondary_evidence = [f"{second_label} secondary score={second_score:.2f}", *second_ev[:2]]

    if offline_hints:
        ok = offline_hints.get("dominant_label")
        if ok:
            label = str(ok)
            primary_evidence.insert(0, "Offline profiler (ncu/CUPTI) overrides online global label where hotspots covered.")
            cov = offline_hints.get("kernel_coverage_fraction")
            if cov is not None:
                secondary_evidence.append(f"Offline kernel time coverage ~{float(cov)*100:.1f}%")

    has_gpm_compute = any(s.source_compute == "nvml_gpm" for s in samples)
    conf = _confidence_from_sources(coverage, has_gpm_compute, has_gpm_dram, bool(offline_hints))
    if label == "mixed_bound":
        conf *= 0.92
    if limitations:
        conf *= 0.95

    return {
        "label": label,
        "confidence": round(min(0.97, max(0.05, conf)), 4),
        "primary_evidence": primary_evidence,
        "secondary_evidence": secondary_evidence,
        "metric_coverage": coverage,
        "limitations": limitations,
        "scores": {
            "compute": c_score,
            "memory": m_score,
            "pcie_io": p_score,
            "latency": l_score,
        },
    }


def classify_event_online(samples: List[GpuSample], duration_ms: float) -> Dict[str, Any]:
    """Per-event classification from samples falling inside that event window."""
    if not samples or duration_ms <= 0:
        return {
            "label": "unknown",
            "confidence": 0.1,
            "evidence": ["No samples in event window"],
        }
    sub = build_classification(samples, offline_hints=None, duration_s=duration_ms / 1000.0)
    return {
        "label": sub["label"],
        "confidence": sub["confidence"] * 0.92,
        "evidence": sub["primary_evidence"][:6],
    }
