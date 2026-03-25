"""Merge online summary with offline kernel profiler output."""

from __future__ import annotations

from typing import Any, Dict, Optional


def merge_online_offline(
    online_classification: Dict[str, Any],
    event_phase_labels: Dict[str, Dict[str, Any]],
    offline_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    offline_hints = None
    if offline_summary and isinstance(offline_summary, dict):
        offline_hints = {
            "dominant_label": offline_summary.get("dominant_label"),
            "kernel_coverage_fraction": offline_summary.get("kernel_coverage_fraction"),
        }

    global_label = online_classification.get("label", "mixed_bound")
    confidence = float(online_classification.get("confidence", 0.5))
    primary = list(online_classification.get("primary_evidence", []))
    secondary = list(online_classification.get("secondary_evidence", []))

    if offline_hints and offline_hints.get("dominant_label"):
        global_label = str(offline_hints["dominant_label"])
        confidence = min(0.95, confidence + 0.12)
        primary.insert(0, "Offline hotspot kernel analysis (ncu) preferred for global label when coverage is high.")

    merged_phases: Dict[str, Any] = {}
    for name, evc in event_phase_labels.items():
        merged_phases[name] = dict(evc)
        if offline_hints and name in merged_phases:
            merged_phases[name]["offline_note"] = "Phase label kept online unless kernel names map to this range (manual review)."

    return {
        "global_label": global_label,
        "global_confidence": round(confidence, 4),
        "phases": merged_phases,
        "primary_evidence": primary,
        "secondary_evidence": secondary,
        "offline": offline_summary or {},
        "rules_applied": [
            "If offline kernel profiler covers dominant GPU time, prefer its compute vs DRAM throughput story.",
            "Otherwise keep online stage-level labels and cite limited offline coverage in evidence.",
        ],
    }
