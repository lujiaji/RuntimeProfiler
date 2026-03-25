from typing import Any, Dict


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_profile_summary(summary: Dict[str, Any]) -> None:
    bound = summary.get("bound_classification", {})
    label = bound.get("label", "unknown")
    confidence = bound.get("confidence", 0.0)

    print("=== Runtime Profile Summary ===")
    print(f"backend: {_fmt(summary.get('backend'))}")
    print(f"samples: {_fmt(summary.get('sample_count'))}")
    print(f"events: {_fmt(summary.get('event_count'))}")
    print(f"max_throughput: {_fmt(summary.get('max_throughput'))}")
    print(f"max_bandwidth_gbps: {_fmt(summary.get('max_bandwidth_gbps'))}")
    print(f"max_bandwidth_tx_gbps: {_fmt(summary.get('max_bandwidth_tx_gbps'))}")
    print(f"max_bandwidth_rx_gbps: {_fmt(summary.get('max_bandwidth_rx_gbps'))}")
    print(f"max_mem_util_percent: {_fmt(summary.get('max_mem_util_percent'))}")
    print(f"max_memory_allocated_mb: {_fmt(summary.get('max_memory_allocated_mb'))}")
    print(f"max_memory_reserved_mb: {_fmt(summary.get('max_memory_reserved_mb'))}")
    run_window = summary.get("run_window", {})
    print(f"run_duration_s: {_fmt(run_window.get('duration_s'))}")
    torch_prof = summary.get("torch_profiler", {})
    print(f"torch_profiler_enabled: {_fmt(torch_prof.get('enabled'))}")
    print(f"bound: {label} (confidence={_fmt(confidence)})")
