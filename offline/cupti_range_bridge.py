"""
Bridge EventTracer range names to CUPTI profiling windows (reserved).

First release: schema + no-op hooks for future integration.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


def register_range_bridge(_on_enter: Callable[[str], None], _on_exit: Callable[[str], None]) -> None:
    """Placeholder for future CUPTI correlation IDs tied to trace event names."""
    return None


def correlation_payload(range_name: str, range_id: int) -> Dict[str, Any]:
    return {"range_name": range_name, "range_id": range_id, "backend": "cupti_reserved"}
