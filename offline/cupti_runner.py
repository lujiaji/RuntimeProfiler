"""
CUPTI range profiler runner (Phase 2 / advanced).

Reserved interface: will attach CUPTI callbacks around EventTracer ranges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class CUPTIConfig:
    enabled: bool = False
    output_dir: str = "cupti_profile"
    metrics: List[str] = field(default_factory=list)
    enable_pc_sampling: bool = False

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CUPTIConfig":
        return CUPTIConfig(
            enabled=bool(d.get("enabled", False)),
            output_dir=str(d.get("output_dir", "cupti_profile")),
            metrics=list(d.get("metrics", [])),
            enable_pc_sampling=bool(d.get("enable_pc_sampling", False)),
        )


def run_cupti_range_profile(_cfg: CUPTIConfig) -> Dict[str, Any]:
    return {
        "ok": False,
        "message": "CUPTI runner not implemented in this version; interface reserved.",
    }
