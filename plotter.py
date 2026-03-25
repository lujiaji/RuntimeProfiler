"""Multi-panel diagnostic plots for runtime profiler."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .event_aggregate import EventMetricRollup
from .event_tracer import TraceEvent
from .metrics_backends import GpuSample


def _event_category(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("attention", "attn", "kv", "cache")):
        return "attention_kv"
    if any(k in n for k in ("decode", "prefill", "generate")):
        return "llm_phase"
    if any(k in n for k in ("load", "offload", "reload", "prefetch", "io")):
        return "io_stage"
    if any(k in n for k in ("dit", "vae", "encoder", "decoder")):
        return "gen_model"
    if any(k in n for k in ("data", "preprocess", "postprocess", "dataloader")):
        return "host_pipeline"
    return "other"


def _color_for_event(name: str) -> Tuple[float, float, float]:
    h = hashlib.sha256(name.encode("utf-8")).digest()
    return (h[0] / 255.0, h[1] / 255.0, h[2] / 255.0)


def _any_values(seq: Sequence[Optional[float]]) -> bool:
    return any(v is not None for v in seq)


def _series_peak(xs: List[float], ys: List[Optional[float]]) -> Optional[Tuple[float, float]]:
    best: Optional[Tuple[float, float]] = None
    for x, y in zip(xs, ys):
        if y is None:
            continue
        yv = float(y)
        if best is None or yv > best[1]:
            best = (x, yv)
    return best


def plot_main_figure(
    samples: List[GpuSample],
    events: List[TraceEvent],
    out_png: str,
    detail_mode: bool = False,
) -> None:
    """Six stacked panels sharing time axis (seconds from first sample)."""
    if not samples:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    ts0 = samples[0].ts_ns
    xs = [(s.ts_ns - ts0) / 1e9 for s in samples]

    has_sm = _any_values([s.sm_util_pct for s in samples])
    has_occ = _any_values([s.sm_occupancy_pct for s in samples])
    has_ten = _any_values([s.tensor_util_pct for s in samples])
    has_dram_m = _any_values([s.dram_bw_util_pct for s in samples])
    has_dram_p = _any_values([s.dram_bw_proxy_gbps for s in samples])
    has_mem_u = _any_values([s.mem_util_pct for s in samples])
    has_alloc = _any_values([s.alloc_mem_mb for s in samples])
    has_pcie = _any_values([s.pcie_rx_mib_s for s in samples]) or _any_values(
        [s.pcie_tx_mib_s for s in samples]
    )
    has_pwr = _any_values([s.power_w for s in samples])

    degraded: List[str] = []
    if not has_sm and not has_occ:
        degraded.append("compute")
    if not has_dram_m and not has_dram_p:
        degraded.append("device_memory")
    if not has_pcie:
        degraded.append("pcie")
    if not has_alloc:
        degraded.append("capacity")
    if not has_pwr:
        degraded.append("power_clocks")

    subtitle = ""
    if degraded:
        subtitle = " (degraded: " + ", ".join(degraded) + ")"

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(13, 14),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.3, 1.2, 1.2, 1.15, 1.0]},
    )
    fig.suptitle("Runtime Profiler — diagnostic panels" + subtitle, fontsize=12)

    ax0, ax1, ax2, ax3, ax4, ax5 = axes

    # --- Panel 1: event timeline (gantt) ---
    ax0.set_ylabel("Events")
    ax0.set_title("Stage / Event timeline (semantic alignment)")
    if events:
        sorted_ev = sorted(events, key=lambda e: (e.start_ns, e.depth))
        yticks: List[int] = []
        ylabels: List[str] = []
        y = 0
        for ev in sorted_ev:
            t0 = (ev.start_ns - ts0) / 1e9
            w = max(1e-9, (ev.end_ns - ev.start_ns) / 1e9)
            c = _color_for_event(ev.name)
            ax0.barh(y, w, left=t0, height=0.85, color=c, alpha=0.85, edgecolor="none")
            ax0.text(
                t0 + w * 0.02,
                y,
                ev.name[:48],
                va="center",
                fontsize=7,
                color="black",
                clip_on=True,
            )
            yticks.append(y)
            ylabels.append(f"d{ev.depth}")
            y += 1
        ax0.set_yticks([float(i) for i in yticks])
        ax0.set_yticklabels(ylabels, fontsize=7)
        ax0.set_ylim(-0.5, max(0.5, y - 0.5))
    else:
        ax0.text(0.5, 0.5, "No trace events", ha="center", va="center", transform=ax0.transAxes)
    ax0.grid(True, axis="x", alpha=0.2)

    # --- Panel 2: compute ---
    ax1.set_title("Compute pressure (SM / occupancy / tensor)")
    ax1.set_ylabel("%")
    if has_sm:
        ax1.plot(xs, [s.sm_util_pct for s in samples], label="sm_util_pct", linewidth=1.2)
    if has_occ:
        ax1.plot(
            xs,
            [s.sm_occupancy_pct for s in samples],
            label="sm_occupancy_pct",
            linewidth=1.0,
            linestyle="--",
        )
    if has_ten:
        ax1.plot(xs, [s.tensor_util_pct for s in samples], label="tensor_util_pct", linewidth=1.0)
    if detail_mode:
        if _any_values([s.fp16_util_pct for s in samples]):
            ax1.plot(xs, [s.fp16_util_pct for s in samples], label="fp16_util_pct", alpha=0.7, linewidth=0.9)
        if _any_values([s.fp32_util_pct for s in samples]):
            ax1.plot(xs, [s.fp32_util_pct for s in samples], label="fp32_util_pct", alpha=0.7, linewidth=0.9)
        if _any_values([s.int_util_pct for s in samples]):
            ax1.plot(xs, [s.int_util_pct for s in samples], label="int_util_pct", alpha=0.7, linewidth=0.9)
    if ax1.get_legend_handles_labels()[0]:
        ax1.legend(loc="upper right", fontsize=7)
    ax1.grid(True, alpha=0.25)
    ax1.set_ylim(0, 105)

    # --- Panel 3: device memory pressure ---
    ax2.set_title("Device memory pressure (DRAM BW vs coarse mem_util)")
    ax2.set_ylabel("% / proxy Gb/s")
    if has_dram_m:
        ax2.plot(
            xs,
            [s.dram_bw_util_pct for s in samples],
            label="dram_bw_util_pct",
            linewidth=1.3,
        )
    ax2_mem = ax2.twinx() if has_dram_p or has_mem_u else None
    if has_dram_p and ax2_mem is not None:
        ax2_mem.plot(
            xs,
            [s.dram_bw_proxy_gbps for s in samples],
            label="dram_bw_proxy_gbps",
            linestyle=":",
            linewidth=1.1,
            alpha=0.85,
        )
        ax2_mem.set_ylabel("Proxy Gb/s")
    if has_mem_u:
        target_ax = ax2_mem if ax2_mem is not None else ax2
        target_ax.plot(
            xs,
            [s.mem_util_pct for s in samples],
            label="mem_util_pct (secondary)",
            linewidth=0.9,
            linestyle="--",
            alpha=0.65,
        )
    lines, labels = ax2.get_legend_handles_labels()
    if ax2_mem is not None:
        l2, lb2 = ax2_mem.get_legend_handles_labels()
        lines += l2
        labels += lb2
    if lines:
        ax2.legend(lines, labels, loc="upper right", fontsize=7)
    ax2.grid(True, alpha=0.25)

    # --- Panel 4: capacity ---
    ax3.set_title("Capacity / residency (allocator)")
    ax3.set_ylabel("MiB")
    if has_alloc:
        ax3.plot(xs, [s.alloc_mem_mb for s in samples], label="alloc_mem_mb", linewidth=1.1)
    if _any_values([s.reserved_mem_mb for s in samples]):
        ax3.plot(xs, [s.reserved_mem_mb for s in samples], label="reserved_mem_mb", linewidth=1.0, alpha=0.85)
    tot = samples[0].total_mem_mb if samples else None
    if tot is not None:
        ax3.axhline(y=float(tot), linestyle="--", linewidth=1.0, label="total_mem_mb", alpha=0.6)
    if detail_mode and _any_values([s.active_mem_mb for s in samples]):
        ax3.plot(xs, [s.active_mem_mb for s in samples], label="active_mem_mb", alpha=0.7)
    if ax3.get_legend_handles_labels()[0]:
        ax3.legend(loc="upper right", fontsize=7)
    ax3.grid(True, alpha=0.25)

    # --- Panel 5: PCIe ---
    ax4.set_title("PCIe / external IO (host ↔ device)")
    ax4.set_ylabel("MiB/s")
    if _any_values([s.pcie_rx_mib_s for s in samples]):
        ax4.plot(xs, [s.pcie_rx_mib_s for s in samples], label="pcie_rx_mib_s", linewidth=1.1)
    if _any_values([s.pcie_tx_mib_s for s in samples]):
        ax4.plot(xs, [s.pcie_tx_mib_s for s in samples], label="pcie_tx_mib_s", linewidth=1.1)
    if detail_mode:
        if _any_values([s.nvlink_rx_mib_s for s in samples]):
            ax4.plot(xs, [s.nvlink_rx_mib_s for s in samples], label="nvlink_rx_mib_s", alpha=0.7)
        if _any_values([s.nvlink_tx_mib_s for s in samples]):
            ax4.plot(xs, [s.nvlink_tx_mib_s for s in samples], label="nvlink_tx_mib_s", alpha=0.7)
    if ax4.get_legend_handles_labels()[0]:
        ax4.legend(loc="upper right", fontsize=7)
    ax4.grid(True, alpha=0.25)

    # --- Panel 6: power / clocks / thermal ---
    ax5.set_title("Power / clocks / thermal (throttle hints)")
    ax5.set_ylabel("W / MHz")
    if has_pwr:
        ax5.plot(xs, [s.power_w for s in samples], label="power_w", color="tab:red", linewidth=1.0)
    ax5c = ax5.twinx()
    if _any_values([s.sm_clock_mhz for s in samples]):
        ax5c.plot(xs, [s.sm_clock_mhz for s in samples], label="sm_clock_mhz", color="tab:blue", alpha=0.85)
    if _any_values([s.mem_clock_mhz for s in samples]):
        ax5c.plot(xs, [s.mem_clock_mhz for s in samples], label="mem_clock_mhz", color="tab:green", alpha=0.85)
    if detail_mode and _any_values([s.temperature_c for s in samples]):
        ax5c.plot(
            xs,
            [s.temperature_c for s in samples],
            label="temperature_c",
            color="tab:orange",
            alpha=0.75,
        )
    l1, lb1 = ax5.get_legend_handles_labels()
    l2, lb2 = ax5c.get_legend_handles_labels()
    if l1 or l2:
        ax5.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=7)
    ax5.set_xlabel("Time (s)")
    ax5.grid(True, alpha=0.25)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_event_summary_bars(rollups: List[EventMetricRollup], out_png: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rollups:
        return
    names = [f"{r.event_name}"[:40] for r in rollups]
    dur = [r.duration_ms for r in rollups]
    sm = [r.avg_sm_util_pct or 0.0 for r in rollups]
    dram = [r.avg_dram_bw_util_pct or 0.0 for r in rollups]
    pcie = [r.avg_pcie_total_mib_s or 0.0 for r in rollups]

    x = list(range(len(names)))
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(names) * 0.35), 8), sharex=True)
    axes[0].bar(x, dur, color="tab:purple", alpha=0.85)
    axes[0].set_ylabel("duration_ms")
    axes[0].set_title("Event duration")
    axes[0].grid(True, axis="y", alpha=0.25)

    w = 0.25
    axes[1].bar([xi - w for xi in x], sm, width=w, label="avg sm_util%", color="tab:blue")
    axes[1].bar(x, dram, width=w, label="avg dram_bw%", color="tab:orange")
    axes[1].bar([xi + w for xi in x], pcie, width=w, label="avg pcie MiB/s", color="tab:green")
    axes[1].set_ylabel("metric")
    axes[1].legend(fontsize=7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_bound_evidence_heatmap(rollups: List[EventMetricRollup], out_png: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    if not rollups:
        return
    rows = [r.event_name[:32] for r in rollups]
    cols = ["sm_util", "occupancy", "tensor", "dram_bw", "alloc_mb", "pcie_rx", "pcie_tx"]
    M = np.zeros((len(rollups), len(cols)), dtype=float)
    for i, r in enumerate(rollups):
        M[i, 0] = r.avg_sm_util_pct or 0.0
        M[i, 1] = r.avg_sm_occupancy_pct or 0.0
        M[i, 2] = r.avg_tensor_util_pct or 0.0
        M[i, 3] = r.avg_dram_bw_util_pct or 0.0
        M[i, 4] = r.avg_alloc_mem_mb or 0.0
        M[i, 5] = r.avg_pcie_rx_mib_s or 0.0
        M[i, 6] = r.avg_pcie_tx_mib_s or 0.0
    col_max = np.maximum(M.max(axis=0), 1e-6)
    Mn = M / col_max

    fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.35)))
    im = ax.imshow(Mn, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7)
    ax.set_title("Normalized event-level pressure (avg metrics)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_kernel_drilldown(kernel_rows: List[Dict[str, Any]], out_png: str, topk: int = 20) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not kernel_rows:
        return
    rows = sorted(
        kernel_rows,
        key=lambda r: float(r.get("total_time_ms") or 0.0),
        reverse=True,
    )[:topk]
    names = [str(r.get("kernel_name", "kernel"))[:50] for r in rows]
    times = [float(r.get("total_time_ms") or 0.0) for r in rows]

    fig, ax = plt.subplots(figsize=(11, max(4, len(rows) * 0.32)))
    y = list(range(len(names)))
    ax.barh(y, times, color="tab:cyan", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("total_time_ms")
    ax.set_title("Kernel drilldown (offline)")
    for i, r in enumerate(rows):
        note = r.get("top_stall_reason") or r.get("stall_top")
        thr = r.get("sm_throughput_pct")
        if note or thr is not None:
            tail = f"{thr}%" if thr is not None else ""
            if note:
                tail = f"{tail} | {note}" if tail else str(note)
            ax.text(float(times[i]), float(i), "  " + tail[:40], va="center", fontsize=6, color="dimgray")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# Backward compatibility
def plot_timeline(
    samples: List[GpuSample],
    out_png: str,
    events: Optional[List[TraceEvent]] = None,
    torch_timeline: Optional[Dict[str, List[float]]] = None,
    detail_mode: bool = False,
) -> None:
    ev = events or []
    plot_main_figure(samples, ev, out_png, detail_mode=detail_mode)


def plot_event_timeline(events: List[TraceEvent], out_png: str) -> None:
    if not events:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    sorted_events = sorted(events, key=lambda e: e.start_ns)
    t0 = sorted_events[0].start_ns
    names = [f"{e.name} (d={e.depth})" for e in sorted_events]
    starts = [(e.start_ns - t0) / 1e9 for e in sorted_events]
    durations = [(e.end_ns - e.start_ns) / 1e9 for e in sorted_events]

    fig, ax = plt.subplots(figsize=(12, max(4, len(events) * 0.25)))
    y = list(range(len(sorted_events)))
    ax.barh(y, durations, left=starts, color="tab:purple", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Time (s)")
    ax.set_title("Function Event Timeline (legacy)")
    ax.grid(True, axis="x", alpha=0.25)

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def samples_to_csv_rows(samples: List[GpuSample]) -> List[Dict[str, Any]]:
    return [s.to_dict() for s in samples]
