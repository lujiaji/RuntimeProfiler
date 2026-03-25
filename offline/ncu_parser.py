"""Parse Nsight Compute CSV exports into kernel-level tables and bound hints."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_ncu_csv_to_kernels(csv_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    path = Path(csv_path)
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}
            rows.append(row)
    return rows


def _f(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def _duration_to_ms(dur: Optional[float]) -> Optional[float]:
    if dur is None:
        return None
    x = float(dur)
    if x > 1e9:
        return x / 1e6
    if x > 1e6:
        return x / 1e3
    if x > 1e3:
        return x / 1000.0
    return x


def summarize_kernels(kernel_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Heuristic kernel_bound_summary from parsed NCU rows (column names vary by driver)."""
    kernels: List[Dict[str, Any]] = []
    for r in kernel_rows:
        name = r.get("Kernel Name") or r.get("Name") or r.get("kernel_name") or "kernel"
        raw_dur = _f(r.get("Duration")) or _f(r.get("gpu__time_duration.sum")) or _f(r.get("GPU Time Duration"))
        dur_ms = _duration_to_ms(raw_dur)
        sm_thr = _f(r.get("sm__throughput.avg.pct_of_peak_sustained_elapsed"))
        dram_thr = _f(r.get("dram__throughput.avg.pct_of_peak_sustained_elapsed"))
        occ = _f(r.get("smsp__warps_active.avg.pct_of_peak_sustained_active"))
        stall_cols = [k for k in r if "stall" in k.lower() and "pcsampler" in k.lower()]
        top_stall = None
        if stall_cols:
            best = None
            for k in stall_cols:
                v = _f(r.get(k))
                if v is None:
                    continue
                if best is None or v > best[1]:
                    best = (k, v)
            if best:
                top_stall = f"{best[0]}={best[1]:.2f}"
        kernels.append(
            {
                "kernel_name": name,
                "total_time_ms": dur_ms or 0.0,
                "sm_throughput_pct": sm_thr,
                "dram_throughput_pct": dram_thr,
                "occupancy_proxy_pct": occ,
                "top_stall_reason": top_stall,
            }
        )

    dominant_label = "mixed_bound"
    if kernels:
        top = max(kernels, key=lambda k: float(k.get("total_time_ms") or 0.0))
        smt = top.get("sm_throughput_pct")
        dmt = top.get("dram_throughput_pct")
        if smt is not None and dmt is not None:
            if smt >= (dmt + 12):
                dominant_label = "compute_bound"
            elif dmt >= (smt + 12):
                dominant_label = "memory_bound"
        elif smt is not None and smt >= 70:
            dominant_label = "compute_bound"
        elif dmt is not None and dmt >= 70:
            dominant_label = "memory_bound"

    return {
        "dominant_label": dominant_label,
        "kernel_coverage_fraction": 1.0 if kernels else 0.0,
        "kernels": kernels[:200],
    }


def write_kernel_artifacts(
    kernel_rows: List[Dict[str, Any]],
    out_dir: str,
) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_out = out / "kernel_level_metrics.csv"
    json_out = out / "kernel_level_metrics.json"
    summ_out = out / "kernel_bound_summary.json"

    if kernel_rows:
        fieldnames = list(kernel_rows[0].keys())
        with csv_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in kernel_rows:
                w.writerow(row)
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(kernel_rows, f, indent=2, ensure_ascii=False)

    summary = summarize_kernels(kernel_rows)
    with summ_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {
        "kernel_level_metrics_csv": str(csv_out),
        "kernel_level_metrics_json": str(json_out),
        "kernel_bound_summary_json": str(summ_out),
    }
