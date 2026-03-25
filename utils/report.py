from typing import Any, Dict, List


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_evidence(title: str, items: List[Any]) -> None:
    print(title)
    if not items:
        print("  (none)")
        return
    for line in items[:12]:
        print(f"  - {line}")


def print_profile_summary(summary: Dict[str, Any]) -> None:
    bound = summary.get("bound_classification", {})
    label = bound.get("label", "unknown")
    confidence = bound.get("confidence", 0.0)
    cls = summary.get("classification") or {}

    print("=== Runtime Profile Summary ===")
    print(f"backend: {_fmt(summary.get('backend'))}")
    print(f"samples: {_fmt(summary.get('sample_count'))}")
    print(f"events: {_fmt(summary.get('event_count'))}")
    print(f"max_sm_util_pct: {_fmt(summary.get('max_sm_util_pct'))}")
    print(f"max_dram_bw_util_pct: {_fmt(summary.get('max_dram_bw_util_pct'))}")
    print(f"max_pcie_total_mib_s: {_fmt(summary.get('max_pcie_total_mib_s'))}")
    print(f"max_mem_util_percent: {_fmt(summary.get('max_mem_util_percent'))}")
    print(f"max_memory_allocated_mb: {_fmt(summary.get('max_memory_allocated_mb'))}")
    print(f"max_memory_reserved_mb: {_fmt(summary.get('max_memory_reserved_mb'))}")
    run_window = summary.get("run_window", {})
    print(f"run_duration_s: {_fmt(run_window.get('duration_s'))}")
    torch_prof = summary.get("torch_profiler", {})
    print(f"torch_profiler_enabled: {_fmt(torch_prof.get('enabled'))}")
    print(f"classification.label: {cls.get('label', label)} (confidence={_fmt(cls.get('confidence', confidence))})")
    cov = cls.get("metric_coverage") or {}
    if cov:
        print(
            "metric_coverage: "
            f"compute={cov.get('compute')} "
            f"device_memory={cov.get('device_memory')} "
            f"io={cov.get('io')} "
            f"alloc_mem={cov.get('alloc_mem')}"
        )
    _print_evidence("primary_evidence:", list(cls.get("primary_evidence") or []))
    _print_evidence("limitations:", list(cls.get("limitations") or []))

    for key in (
        "top_compute_pressure_events",
        "top_memory_pressure_events",
        "top_pcie_io_events",
        "top_latency_bound_events",
    ):
        block = summary.get(key)
        if not block:
            continue
        print(f"{key}:")
        for row in block[:5]:
            name = row.get("event_name", "?")
            dur = row.get("duration_ms")
            print(f"  - {name} (duration_ms={_fmt(dur)})")

    merged = summary.get("merged_diagnosis_preview") or {}
    if merged.get("global_label"):
        print(
            f"merged.global_label: {merged.get('global_label')} "
            f"(confidence={_fmt(merged.get('global_confidence'))})"
        )

    print(f"(compat) bound_classification: {label} (confidence={_fmt(confidence)})")
