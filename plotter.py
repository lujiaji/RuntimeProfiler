from pathlib import Path
from typing import Dict, List

from .event_tracer import TraceEvent
from .metrics_backends import GpuSample


def plot_timeline(samples: List[GpuSample], out_png: str) -> None:
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
    mem_used = [s.mem_used_mb for s in samples]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.set_title("Runtime Profile Timeline")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Utilization (%)")
    ax1.plot(xs, sm, label="sm_util(%)", color="tab:blue", linewidth=1.2)
    ax1.plot(xs, mem_util, label="mem_util(%)", color="tab:orange", linewidth=1.2)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Memory (MB)")
    ax2.plot(xs, mem_used, label="mem_used_mb", color="tab:green", linewidth=1.2)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

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

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def samples_to_csv_rows(samples: List[GpuSample]) -> List[Dict[str, float]]:
    return [s.to_dict() for s in samples]
