from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .event_tracer import TraceEvent
from .metrics_backends import GpuSample


def _series_peak(xs: List[float], ys: List[Optional[float]]) -> Optional[Tuple[float, float]]:
    best: Optional[Tuple[float, float]] = None
    for x, y in zip(xs, ys):
        if y is None:
            continue
        yv = float(y)
        if best is None or yv > best[1]:
            best = (x, yv)
    return best


def _annotate_peak(ax, x: float, y: float, label: str, color: str) -> None:
    ax.scatter([x], [y], color=color, s=24, zorder=4)
    ax.axvline(x=x, color=color, linestyle="--", linewidth=1.0, alpha=0.35)
    ax.axhline(y=y, color=color, linestyle="--", linewidth=1.0, alpha=0.35)
    ax.annotate(
        f"{label} peak={y:.2f} @ {x:.3f}s",
        xy=(x, y),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=8,
        color=color,
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": color, "alpha": 0.8},
    )


def plot_timeline(
    samples: List[GpuSample],
    out_png: str,
    torch_timeline: Optional[Dict[str, List[float]]] = None,
) -> None:
    if not samples:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    ts0 = samples[0].ts_ns
    xs = [(s.ts_ns - ts0) / 1e9 for s in samples]
    sm = [s.sm_util for s in samples]
    mem_util = [s.mem_util for s in samples]
    mem_allocated = [s.mem_allocated_mb for s in samples]
    mem_reserved = [s.mem_reserved_mb for s in samples]

    bandwidth = [s.bandwidth_gbps for s in samples]
    bandwidth_tx = [s.bandwidth_tx_gbps for s in samples]
    bandwidth_rx = [s.bandwidth_rx_gbps for s in samples]

    has_torch_timeline = bool(torch_timeline)
    if has_torch_timeline:
        fig, (ax_top, ax_bw, ax_ops) = plt.subplots(
            3,
            1,
            figsize=(12, 9),
            sharex=True,
            gridspec_kw={"height_ratios": [3.0, 1.5, 1.7]},
        )
    else:
        fig, (ax_top, ax_bw) = plt.subplots(
            2,
            1,
            figsize=(12, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [3.2, 1.6]},
        )
        ax_ops = None
    fig.suptitle("Runtime Profile Timeline")

    ax_top.set_ylabel("Utilization (%)")
    ax_top.plot(xs, sm, label="sm_util(%)", color="tab:blue", linewidth=1.2)
    ax_top.plot(xs, mem_util, label="mem_util(%)", color="tab:orange", linewidth=1.2)
    ax_top.set_ylim(0, 100)
    ax_top.grid(True, alpha=0.25)

    ax_top_mem = ax_top.twinx()
    ax_top_mem.set_ylabel("Memory (MB)")
    ax_top_mem.plot(xs, mem_allocated, label="mem_allocated_mb", color="tab:green", linewidth=1.2)
    ax_top_mem.plot(xs, mem_reserved, label="mem_reserved_mb", color="tab:cyan", linewidth=1.2)

    sm_peak = _series_peak(xs, sm)
    if sm_peak is not None:
        _annotate_peak(ax_top, sm_peak[0], sm_peak[1], "sm_util", "tab:blue")
    mem_util_peak = _series_peak(xs, mem_util)
    if mem_util_peak is not None:
        _annotate_peak(ax_top, mem_util_peak[0], mem_util_peak[1], "mem_util", "tab:orange")
    mem_alloc_peak = _series_peak(xs, mem_allocated)
    if mem_alloc_peak is not None:
        _annotate_peak(ax_top_mem, mem_alloc_peak[0], mem_alloc_peak[1], "mem_allocated", "tab:green")
    mem_reserved_peak = _series_peak(xs, mem_reserved)
    if mem_reserved_peak is not None:
        _annotate_peak(ax_top_mem, mem_reserved_peak[0], mem_reserved_peak[1], "mem_reserved", "tab:cyan")

    ax_bw.set_xlabel("Time (s)")
    ax_bw.set_ylabel("Bandwidth (Gb/s)")
    bw_any = any(v is not None for v in bandwidth)
    bw_tx_any = any(v is not None for v in bandwidth_tx)
    bw_rx_any = any(v is not None for v in bandwidth_rx)
    if bw_any or bw_tx_any or bw_rx_any:
        if bw_any:
            ax_bw.plot(xs, bandwidth, label="bandwidth_total_gbps", color="tab:red", linewidth=1.2)
            bw_peak = _series_peak(xs, bandwidth)
            if bw_peak is not None:
                _annotate_peak(ax_bw, bw_peak[0], bw_peak[1], "bw_total", "tab:red")
        if bw_tx_any:
            ax_bw.plot(xs, bandwidth_tx, label="bandwidth_tx_gbps", color="tab:pink", linewidth=1.0, alpha=0.9)
            bw_tx_peak = _series_peak(xs, bandwidth_tx)
            if bw_tx_peak is not None:
                _annotate_peak(ax_bw, bw_tx_peak[0], bw_tx_peak[1], "bw_tx", "tab:pink")
        if bw_rx_any:
            ax_bw.plot(xs, bandwidth_rx, label="bandwidth_rx_gbps", color="tab:purple", linewidth=1.0, alpha=0.9)
            bw_rx_peak = _series_peak(xs, bandwidth_rx)
            if bw_rx_peak is not None:
                _annotate_peak(ax_bw, bw_rx_peak[0], bw_rx_peak[1], "bw_rx", "tab:purple")

        y_candidates = [float(v) for arr in (bandwidth, bandwidth_tx, bandwidth_rx) for v in arr if v is not None]
        observed_peak = max(y_candidates) if y_candidates else None
        if observed_peak is not None:
            ax_bw.set_ylim(0, observed_peak * 1.15 if observed_peak > 0 else 1.0)
    else:
        ax_bw.text(
            0.5,
            0.5,
            "No bandwidth samples available",
            ha="center",
            va="center",
            transform=ax_bw.transAxes,
            fontsize=9,
            color="gray",
        )
    ax_bw.grid(True, alpha=0.25)

    lines_1, labels_1 = ax_top.get_legend_handles_labels()
    lines_2, labels_2 = ax_top_mem.get_legend_handles_labels()
    ax_top.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    lines_3, labels_3 = ax_bw.get_legend_handles_labels()
    if lines_3:
        ax_bw.legend(lines_3, labels_3, loc="upper right")

    if ax_ops is not None and torch_timeline:
        t_xs = torch_timeline.get("bin_centers_s", [])
        active_ratio = torch_timeline.get("cuda_active_ratio", [])
        mem_rate = torch_timeline.get("cuda_mem_event_mb_per_s", [])
        if t_xs and active_ratio:
            ax_ops.set_ylabel("CUDA Active Ratio")
            ax_ops.plot(t_xs, active_ratio, color="tab:purple", label="cuda_active_ratio", linewidth=1.2)
            peak_active = _series_peak(t_xs, active_ratio)
            if peak_active is not None:
                _annotate_peak(ax_ops, peak_active[0], peak_active[1], "cuda_active", "tab:purple")
            ax_ops.set_ylim(bottom=0)
            ax_ops.grid(True, alpha=0.25)

            if mem_rate and len(mem_rate) == len(t_xs):
                ax_ops_mem = ax_ops.twinx()
                ax_ops_mem.set_ylabel("Mem Event Rate (MB/s)")
                ax_ops_mem.plot(
                    t_xs,
                    mem_rate,
                    color="tab:brown",
                    label="cuda_mem_event_mb_per_s",
                    linewidth=1.0,
                    alpha=0.85,
                )
                peak_mem = _series_peak(t_xs, mem_rate)
                if peak_mem is not None:
                    _annotate_peak(
                        ax_ops_mem,
                        peak_mem[0],
                        peak_mem[1],
                        "cuda_mem_event",
                        "tab:brown",
                    )
                l4, lb4 = ax_ops.get_legend_handles_labels()
                l5, lb5 = ax_ops_mem.get_legend_handles_labels()
                ax_ops.legend(l4 + l5, lb4 + lb5, loc="upper right")
        else:
            ax_ops.text(
                0.5,
                0.5,
                "Torch profiler enabled, but no timeline bins were parsed",
                ha="center",
                va="center",
                transform=ax_ops.transAxes,
                fontsize=9,
                color="gray",
            )
        ax_ops.set_xlabel("Time (s)")

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


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
    ax.set_title("Function Event Timeline")
    ax.grid(True, axis="x", alpha=0.25)

    if durations:
        best_idx = max(range(len(durations)), key=lambda i: durations[i])
        best_start = starts[best_idx]
        best_dur = durations[best_idx]
        best_end = best_start + best_dur
        ax.axvline(best_start, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.4)
        ax.axvline(best_end, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.4)
        ax.annotate(
            f"max duration={best_dur:.3f}s",
            xy=(best_end, best_idx),
            xytext=(6, 4),
            textcoords="offset points",
            color="tab:red",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "tab:red", "alpha": 0.8},
        )

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def samples_to_csv_rows(samples: List[GpuSample]) -> List[Dict[str, float]]:
    return [s.to_dict() for s in samples]
