from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .metrics_backends import GpuSample


@dataclass
class BoundClassification:
    label: str
    confidence: float
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def classify_bound(samples: List[GpuSample]) -> BoundClassification:
    sm_utils = [float(s.sm_util) for s in samples if s.sm_util is not None]
    mem_utils = [float(s.mem_util) for s in samples if s.mem_util is not None]
    avg_sm = _mean(sm_utils)
    avg_mem = _mean(mem_utils)

    if avg_sm is None and avg_mem is None:
        return BoundClassification(
            label="unknown",
            confidence=0.0,
            evidence={"reason": "No utilization metrics available"},
        )

    avg_sm = avg_sm if avg_sm is not None else 0.0
    avg_mem = avg_mem if avg_mem is not None else 0.0

    compute_score = max(0.0, avg_sm - 0.6 * avg_mem)
    memory_score = max(0.0, avg_mem - 0.6 * avg_sm)
    mixed_score = min(avg_sm, avg_mem)
    total = compute_score + memory_score + mixed_score + 1e-6

    if compute_score >= memory_score and compute_score >= mixed_score:
        label = "compute_bound"
        confidence = compute_score / total
    elif memory_score >= compute_score and memory_score >= mixed_score:
        label = "memory_bound"
        confidence = memory_score / total
    else:
        label = "mixed_bound"
        confidence = mixed_score / total

    return BoundClassification(
        label=label,
        confidence=float(confidence),
        evidence={
            "avg_sm_util": avg_sm,
            "avg_mem_util": avg_mem,
            "compute_score": compute_score,
            "memory_score": memory_score,
            "mixed_score": mixed_score,
            "note": "mem_util may be proxy when backend lacks bandwidth counters",
        },
    )
