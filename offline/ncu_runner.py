"""Nsight Compute CLI wrapper — spawns `ncu` with metric presets."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class NCUConfig:
    enabled: bool = False
    target_executable: str = "python"
    target_python_entry: str = ""
    kernel_name_regex: str = ".*"
    event_name_regex: str = ""
    section_names: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    launch_skip: int = 0
    launch_count: int = 0
    output_dir: str = "ncu_profile"
    preset: str = "bound_basic"
    extra_args: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "NCUConfig":
        return NCUConfig(
            enabled=bool(d.get("enabled", False)),
            target_executable=str(d.get("target_executable", "python")),
            target_python_entry=str(d.get("target_python_entry", "")),
            kernel_name_regex=str(d.get("kernel_name_regex", ".*")),
            event_name_regex=str(d.get("event_name_regex", "")),
            section_names=list(d.get("section_names", [])),
            metrics=list(d.get("metrics", [])),
            launch_skip=int(d.get("launch_skip", 0)),
            launch_count=int(d.get("launch_count", 0)),
            output_dir=str(d.get("output_dir", "ncu_profile")),
            preset=str(d.get("preset", "bound_basic")),
            extra_args=list(d.get("extra_args", [])),
        )


def ncu_presets() -> Dict[str, List[str]]:
    """Return recommended `--metrics` groups (strings passed to ncu)."""
    return {
        "bound_basic": [
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "smsp__warps_active.avg.pct_of_peak_sustained_active",
        ],
        "stall_debug": [
            "smsp__pcsampler_warps_issue_stalled_long_scoreboard.avg.pct",
            "smsp__pcsampler_warps_issue_stalled_short_scoreboard.avg.pct",
            "smsp__pcsampler_warps_issue_stalled_memory_dependency.avg.pct",
            "smsp__pcsampler_warps_issue_stalled_not_selected.avg.pct",
            "smsp__pcsampler_warps_issue_stalled_barrier.avg.pct",
            "smsp__warps_active.avg.pct_of_peak_sustained_active",
        ],
        "roofline": [
            "sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained",
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        ],
    }


def build_ncu_command(cfg: NCUConfig, report_base: str) -> List[str]:
    exe = shutil.which("ncu") or "ncu"
    presets = ncu_presets()
    metrics = list(cfg.metrics)
    if not metrics and cfg.preset in presets:
        metrics = presets[cfg.preset]
    cmd: List[str] = [
        exe,
        "--target-processes",
        "all",
        "--csv",
        "--force-overwrite",
        "-o",
        report_base,
    ]
    if metrics:
        cmd.append("--metrics")
        cmd.append(",".join(metrics))
    for sec in cfg.section_names:
        cmd.extend(["--section", sec])
    if cfg.kernel_name_regex:
        cmd.extend(["--kernel-name-base", "mangled"])
        cmd.extend(["--kernel-regex", cfg.kernel_name_regex])
    if cfg.launch_skip > 0:
        cmd.extend(["--launch-skip", str(cfg.launch_skip)])
    if cfg.launch_count > 0:
        cmd.extend(["--launch-count", str(cfg.launch_count)])
    cmd.extend(cfg.extra_args)
    if cfg.target_python_entry:
        cmd.append(cfg.target_executable)
        cmd.append(cfg.target_python_entry)
    return cmd


def run_ncu(cfg: NCUConfig) -> Dict[str, Any]:
    """Execute ncu; returns paths and status. Requires `ncu` on PATH."""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_base = str(out / "ncu_report")
    cmd = build_ncu_command(cfg, report_base)
    if not cfg.target_python_entry:
        return {"ok": False, "error": "target_python_entry empty", "command": cmd}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        ok = proc.returncode == 0
        return {
            "ok": ok,
            "returncode": proc.returncode,
            "command": cmd,
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-4000:],
            "report_base": report_base,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "command": cmd}


def save_ncu_config(cfg: NCUConfig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
