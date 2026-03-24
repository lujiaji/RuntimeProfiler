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
    print(f"max_bandwidth: {_fmt(summary.get('max_bandwidth'))}")
    print(f"max_memory_used_mb: {_fmt(summary.get('max_memory_used_mb'))}")
    print(f"bound: {label} (confidence={_fmt(confidence)})")
