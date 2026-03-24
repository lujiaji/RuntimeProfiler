"""
Minimal template for your own model runner.

Usage:
  python runtime_profiler/scripts/profile_any_model.py \
    --pythonpath . \
    --target-module runtime_profiler.scripts.template_target \
    --target-fn run_target \
    --target-kwargs-json '{"steps": 30, "size": 1024}'
"""

from typing import Dict

import torch


def run_target(steps: int = 20, size: int = 1024) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    for _ in range(steps):
        _ = a @ b
    return {"steps": float(steps), "size": float(size)}
