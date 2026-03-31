import json
import os
import os.path as osp
import shutil
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch


MIB = 1024.0 * 1024.0


def cuda_memory_stats_mib() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    try:
        device = torch.cuda.current_device()
        alloc = float(torch.cuda.memory_allocated(device)) / MIB
        reserved = float(torch.cuda.memory_reserved(device)) / MIB
        max_alloc = float(torch.cuda.max_memory_allocated(device)) / MIB
        max_reserved = float(torch.cuda.max_memory_reserved(device)) / MIB
        return {
            "cuda_device": int(device),
            "alloc_mem_mb": alloc,
            "reserved_mem_mb": reserved,
            "max_alloc_mem_mb": max_alloc,
            "max_reserved_mem_mb": max_reserved,
        }
    except Exception:
        return {}


def emit_profile_memory_marker(
    tracer,
    marker_name: str,
    component: str = "custom",
    value_mb=None,
    marker_ts_ns=None,
    is_peak: bool = False,
    **extra,
) -> None:
    if tracer is None:
        return
    metadata = {
        "marker_type": "memory",
        "marker_name": marker_name,
        "label": marker_name,
        "value_mb": value_mb,
        "component": component,
        "is_peak": bool(is_peak),
    }
    if marker_ts_ns is not None:
        metadata["marker_ts_ns"] = int(marker_ts_ns)
    metadata.update(cuda_memory_stats_mib())
    metadata.update(extra)
    with tracer.trace(f"mem_marker:{marker_name}", metadata=metadata):
        pass


def mark_mem(tracer, name: str, component: str = "custom", value_mb=None, **extra) -> None:
    if value_mb is None:
        stats = cuda_memory_stats_mib()
        value_mb = stats.get("alloc_mem_mb")
    emit_profile_memory_marker(
        tracer=tracer,
        marker_name=name,
        component=component,
        value_mb=value_mb,
        **extra,
    )


def reset_peak_and_get_baseline_alloc_mb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    try:
        torch.cuda.reset_peak_memory_stats()
        return float(torch.cuda.memory_allocated()) / MIB
    except Exception:
        return None


def get_activation_peak_mb(baseline_alloc_mb: Optional[float]) -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    try:
        peak_alloc_mb = float(torch.cuda.max_memory_allocated()) / MIB
        if baseline_alloc_mb is not None:
            return max(0.0, peak_alloc_mb - baseline_alloc_mb)
        return peak_alloc_mb
    except Exception:
        return None


def emit_runtime_mem_trace_markers(
    tracer,
    mem_trace: List[Dict[str, Any]],
    baseline_alloc_mb: Optional[float],
    activation_peak_mb: Optional[float],
    kv_cache_live_mb: float,
) -> None:
    if mem_trace:
        base_alloc = baseline_alloc_mb if baseline_alloc_mb is not None else 0.0
        for idx, p in enumerate(mem_trace):
            alloc_mb = float(p.get("alloc_mem_mb") or 0.0)
            kv_mb = float(p.get("kv_cache_mb") or 0.0)
            activation_mb = max(0.0, alloc_mb - base_alloc - kv_mb)
            point = str(p.get("point") or f"step_{idx}")
            ts_ns = p.get("ts_ns")
            scale_ind = p.get("scale_ind")
            repeat_idx = p.get("repeat_idx")

            emit_profile_memory_marker(
                tracer,
                "activation_step",
                value_mb=activation_mb,
                marker_ts_ns=ts_ns,
                component="activation",
                point=point,
                scale_ind=scale_ind,
                repeat_idx=repeat_idx,
            )
            emit_profile_memory_marker(
                tracer,
                "kv_cache_step",
                value_mb=kv_mb,
                marker_ts_ns=ts_ns,
                component="kv_cache",
                point=point,
                scale_ind=scale_ind,
                repeat_idx=repeat_idx,
            )

        max_act = max(
            max(
                0.0,
                float(p.get("alloc_mem_mb") or 0.0)
                - base_alloc
                - float(p.get("kv_cache_mb") or 0.0),
            )
            for p in mem_trace
        )
        p_act = max(
            mem_trace,
            key=lambda p: max(
                0.0,
                float(p.get("alloc_mem_mb") or 0.0)
                - base_alloc
                - float(p.get("kv_cache_mb") or 0.0),
            ),
        )
        max_kv = max(float(p.get("kv_cache_mb") or 0.0) for p in mem_trace)
        p_kv = max(mem_trace, key=lambda p: float(p.get("kv_cache_mb") or 0.0))

        emit_profile_memory_marker(
            tracer,
            "activation_max",
            value_mb=max_act,
            marker_ts_ns=p_act.get("ts_ns"),
            component="activation",
            is_peak=True,
            point=str(p_act.get("point") or ""),
            scale_ind=p_act.get("scale_ind"),
            repeat_idx=p_act.get("repeat_idx"),
        )
        emit_profile_memory_marker(
            tracer,
            "kv_cache_max",
            value_mb=max_kv,
            marker_ts_ns=p_kv.get("ts_ns"),
            component="kv_cache",
            is_peak=True,
            point=str(p_kv.get("point") or ""),
            scale_ind=p_kv.get("scale_ind"),
            repeat_idx=p_kv.get("repeat_idx"),
        )
        return

    emit_profile_memory_marker(
        tracer,
        "activation_peak",
        value_mb=activation_peak_mb,
        component="activation",
        kv_cache_live_mb=kv_cache_live_mb,
        is_peak=True,
    )
    emit_profile_memory_marker(
        tracer,
        "kv_cache_live",
        value_mb=kv_cache_live_mb,
        component="kv_cache",
        is_peak=True,
    )


def plot_infinitystar_profiles(
    latency_trace: List[Dict[str, Any]],
    weights_baseline_mb: float,
    save_dir: str,
) -> Tuple[Optional[str], Optional[str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not latency_trace:
        return None, None

    all_steps = latency_trace
    visual_steps = [s for s in all_steps if isinstance(s["scale_ind"], int)]

    total_attn = sum(s["attn_time_s"] for s in all_steps)
    total_ffn = sum(s["ffn_time_s"] for s in all_steps)
    total_others = sum(s["others_time_s"] for s in all_steps)
    grand_total = total_attn + total_ffn + total_others

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sizes = [total_attn, total_ffn, total_others]
    labels = [
        f"Attention\n{total_attn:.2f}s ({total_attn / grand_total * 100:.1f}%)",
        f"FFN\n{total_ffn:.2f}s ({total_ffn / grand_total * 100:.1f}%)",
        f"Others\n{total_others:.2f}s ({total_others / grand_total * 100:.1f}%)",
    ]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    ax1.pie(
        sizes,
        labels=labels,
        colors=colors,
        startangle=90,
        textprops={"fontsize": 11},
        labeldistance=1.15,
    )
    ax1.set_title(
        f"InfinityStar End-to-End Latency Breakdown\nTotal: {grand_total:.2f}s",
        fontsize=14,
        fontweight="bold",
    )
    fig1.tight_layout()
    chart1_path = osp.join(save_dir, "infinitystar_latency_breakdown.png")
    fig1.savefig(chart1_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"[InfinityStar Profile] Latency breakdown chart saved: {chart1_path}")

    if not visual_steps:
        return chart1_path, None

    scale_agg = defaultdict(
        lambda: {
            "attn_time": 0.0,
            "ffn_time": 0.0,
            "others_time": 0.0,
            "total_time": 0.0,
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "kv_cache_mb": 0.0,
        }
    )
    for s in visual_steps:
        si = s["scale_ind"]
        scale_agg[si]["attn_time"] += s["attn_time_s"]
        scale_agg[si]["ffn_time"] += s["ffn_time_s"]
        scale_agg[si]["others_time"] += s["others_time_s"]
        scale_agg[si]["total_time"] += s["total_time_s"]
        scale_agg[si]["allocated_mb"] = s["allocated_mb"]
        scale_agg[si]["reserved_mb"] = s["reserved_mb"]
        scale_agg[si]["kv_cache_mb"] = s["kv_cache_mb"]

    scale_inds = sorted(scale_agg.keys())
    n = len(scale_inds)

    attn_mem, ffn_mem, others_mem, reserved_mem = [], [], [], []
    latency_vals = []
    for si in scale_inds:
        d = scale_agg[si]
        kv = d["kv_cache_mb"]
        alloc = d["allocated_mb"]
        resv = d["reserved_mb"]
        act = max(0.0, alloc - kv - weights_baseline_mb)
        attn_mem.append(kv)
        ffn_mem.append(act)
        others_mem.append(weights_baseline_mb)
        reserved_mem.append(resv)
        latency_vals.append(d["total_time"])

    x = list(range(n))
    fig2, (ax_mem, ax_lat) = plt.subplots(
        2, 1, figsize=(max(14, n * 0.45), 10), sharex=True
    )

    ax_mem.bar(x, others_mem, width=0.7, label="Others (Weights)", color="#45B7D1")
    bottom1 = [o for o in others_mem]
    ax_mem.bar(
        x,
        attn_mem,
        width=0.7,
        bottom=bottom1,
        label="Attention (KV Cache)",
        color="#FF6B6B",
    )
    bottom2 = [o + a for o, a in zip(others_mem, attn_mem)]
    ax_mem.bar(
        x,
        ffn_mem,
        width=0.7,
        bottom=bottom2,
        label="FFN (Activations)",
        color="#4ECDC4",
    )
    ax_mem.plot(x, reserved_mem, "k--", linewidth=1.5, label="CUDA Reserved")
    ax_mem.set_ylabel("Memory (MiB)", fontsize=12)
    ax_mem.set_title("Per-Scale Memory Breakdown", fontsize=13, fontweight="bold")
    ax_mem.legend(loc="upper left", fontsize=9)
    ax_mem.grid(axis="y", alpha=0.3)

    ax_lat.plot(
        x,
        latency_vals,
        "o-",
        color="#FF6B6B",
        linewidth=1.5,
        markersize=4,
        label="Total Latency",
    )
    ax_lat.set_ylabel("Latency (s)", fontsize=12)
    ax_lat.set_xlabel("Scale Index", fontsize=12)
    ax_lat.set_title("Per-Scale Latency", fontsize=13, fontweight="bold")
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(
        [str(si) for si in scale_inds], fontsize=8, rotation=45 if n > 20 else 0
    )
    ax_lat.legend(loc="upper left", fontsize=9)
    ax_lat.grid(axis="y", alpha=0.3)

    fig2.tight_layout()
    chart2_path = osp.join(save_dir, "infinitystar_per_scale_profile.png")
    fig2.savefig(chart2_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[InfinityStar Profile] Per-scale profile chart saved: {chart2_path}")

    return chart1_path, chart2_path


def runtime_profiler_output_dir(tracer: Any) -> Optional[str]:
    """Absolute ``output_dir`` when ``tracer`` is a :class:`~runtime_profiler.core.RuntimeProfiler` instance."""
    if tracer is None:
        return None
    cfg = getattr(tracer, "config", None)
    if cfg is None:
        return None
    out = getattr(cfg, "output_dir", None)
    if not out:
        return None
    return osp.abspath(str(out))


def _latency_trace_json_rows(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in trace:
        d = dict(row)
        ss = d.get("scale_shape")
        if ss is not None and hasattr(ss, "__iter__") and not isinstance(ss, (str, bytes)):
            try:
                d["scale_shape"] = list(ss)
            except TypeError:
                d["scale_shape"] = str(ss)
        rows.append(d)
    return rows


def mirror_infinitystar_artifacts_to_runtime_profiler(
    tracer: Any,
    charts_dir: str,
    latency_trace: Optional[List[Dict[str, Any]]] = None,
    perform_inference_wall_s: Optional[float] = None,
) -> None:
    """Copy InfinityStar latency PNGs and write JSON into RuntimeProfiler ``output_dir`` when profiling."""
    rdir = runtime_profiler_output_dir(tracer)
    if not rdir:
        return
    os.makedirs(rdir, exist_ok=True)
    copied = False
    for name in ("infinitystar_latency_breakdown.png", "infinitystar_per_scale_profile.png"):
        src = osp.join(charts_dir, name)
        if osp.isfile(src):
            shutil.copy2(src, osp.join(rdir, name))
            copied = True
    timing: Dict[str, Any] = {}
    if perform_inference_wall_s is not None:
        timing["perform_inference_wall_s"] = perform_inference_wall_s
    if latency_trace:
        total_blocks = sum(float(x.get("total_time_s") or 0.0) for x in latency_trace)
        timing["latency_trace_step_count"] = len(latency_trace)
        timing["latency_trace_summed_block_time_s"] = total_blocks
        timing["notes"] = (
            "perform_inference_wall_s: wall clock around gen_one_example in infer script (text encode + "
            "autoregressive infer + in-script work). latency_trace_summed_block_time_s: sum of timed blocks "
            "inside ar_infer_infinity_elegant; may differ slightly from wall time."
        )
        with open(osp.join(rdir, "infinitystar_latency_trace.json"), "w", encoding="utf-8") as f:
            json.dump(_latency_trace_json_rows(latency_trace), f, indent=2)
        copied = True
    elif perform_inference_wall_s is not None:
        timing["notes"] = (
            "perform_inference_wall_s only (no runtime_latency_trace; use infinity_elegant schedule path for per-step trace)."
        )
    if timing:
        with open(osp.join(rdir, "infinitystar_model_timing.json"), "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=2)
        copied = True
    if copied:
        print(f"[InfinityStar Profile] Mirrored latency artifacts to RuntimeProfiler dir: {rdir}")
