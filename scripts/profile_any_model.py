#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
RUNTIME_PROFILER_ROOT = CURRENT_FILE.parents[1]
PROJECT_PARENT = RUNTIME_PROFILER_ROOT.parent
sys.path.insert(0, str(PROJECT_PARENT))

from runtime_profiler import RuntimeProfiler, RuntimeProfilerConfig  # noqa: E402
from runtime_profiler.utils import (  # noqa: E402
    load_callable_from_module,
    parse_json_list,
    parse_json_object,
    print_profile_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generic runtime profiler launcher for any model/function.",
    )
    parser.add_argument(
        "--target-module",
        type=str,
        required=True,
        help="Python module path, e.g. my_pkg.my_runner",
    )
    parser.add_argument(
        "--target-fn",
        type=str,
        required=True,
        help="Callable name in target module, e.g. run_inference",
    )
    parser.add_argument(
        "--target-args-json",
        type=str,
        default="[]",
        help='JSON list for positional args, e.g. \'["/tmp/input", 4]\'',
    )
    parser.add_argument(
        "--target-kwargs-json",
        type=str,
        default="{}",
        help='JSON object for keyword args, e.g. \'{"batch_size":8}\'',
    )
    parser.add_argument(
        "--pythonpath",
        type=str,
        default="",
        help="Extra path prepended to sys.path for loading your project code",
    )
    parser.add_argument("--output-dir", type=str, default="./runtime_profile_any_model")
    parser.add_argument("--interval-ms", type=int, default=20)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--inject-profiler-kwarg",
        type=str,
        default="",
        help="Inject profiler object into target kwargs with this key, e.g. tracer",
    )
    parser.add_argument(
        "--backend-order",
        type=str,
        default="nvml_gpm,nvml_legacy,torch_cuda",
        help="Comma separated backend preference, e.g. nvml_gpm,nvml_legacy,torch_cuda (alias: nvml -> nvml_legacy)",
    )
    parser.add_argument(
        "--detail-plot",
        action="store_true",
        help="Enable detail lines on main figure (fp16/fp32/int util, nvlink, temperature, etc.)",
    )
    parser.add_argument(
        "--enable-torch-profiler",
        action="store_true",
        help="Enable optional torch.profiler export (op-level time/memory)",
    )
    parser.add_argument(
        "--torch-profiler-record-shapes",
        action="store_true",
        help="Record tensor shapes in torch.profiler",
    )
    parser.add_argument(
        "--no-torch-profiler-memory",
        action="store_true",
        help="Disable memory profiling in torch.profiler",
    )
    parser.add_argument(
        "--torch-profiler-with-stack",
        action="store_true",
        help="Enable stack capture in torch.profiler",
    )
    parser.add_argument(
        "--torch-profiler-with-flops",
        action="store_true",
        help="Enable flops estimation in torch.profiler (if supported)",
    )
    parser.add_argument(
        "--no-torch-profiler-trace",
        action="store_true",
        help="Disable chrome trace export from torch.profiler",
    )
    parser.add_argument(
        "--torch-profiler-topk-ops",
        type=int,
        default=200,
        help="How many top ops to keep in torch_profiler_ops.csv",
    )
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="After the online run, invoke Nsight Compute (ncu) on a Python entry script (second full run; requires ncu on PATH)",
    )
    parser.add_argument(
        "--ncu-entry",
        type=str,
        default="",
        help="Python script path passed to ncu (e.g. /path/to/run_flux_schnell_q4.py). Required when --ncu is set.",
    )
    parser.add_argument(
        "--ncu-python",
        type=str,
        default="",
        help="Interpreter for ncu (default: python3 on PATH)",
    )
    parser.add_argument(
        "--ncu-preset",
        type=str,
        default="bound_basic",
        help="Metric preset: bound_basic | stall_debug | roofline",
    )
    parser.add_argument(
        "--ncu-launch-skip",
        type=int,
        default=0,
        help="ncu --launch-skip",
    )
    parser.add_argument(
        "--ncu-launch-count",
        type=int,
        default=0,
        help="ncu --launch-count (0 = driver default / no limit)",
    )
    parser.add_argument(
        "--ncu-kernel-regex",
        type=str,
        default=".*",
        help="ncu --kernel-regex filter",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.pythonpath:
        sys.path.insert(0, os.path.abspath(args.pythonpath))

    backend_order = [x.strip() for x in args.backend_order.split(",") if x.strip()]
    target_args = parse_json_list(args.target_args_json)
    target_kwargs = parse_json_object(args.target_kwargs_json)
    target_fn = load_callable_from_module(args.target_module, args.target_fn)

    ncu_cfg: dict = {}
    if args.ncu:
        if not args.ncu_entry.strip():
            print("ERROR: --ncu requires --ncu-entry /path/to/script.py", file=sys.stderr)
            sys.exit(2)
        py = args.ncu_python.strip() or "python3"
        ncu_cfg = {
            "enabled": True,
            "target_executable": py,
            "target_python_entry": os.path.abspath(args.ncu_entry),
            "preset": args.ncu_preset,
            "launch_skip": int(args.ncu_launch_skip),
            "launch_count": int(args.ncu_launch_count),
            "kernel_name_regex": args.ncu_kernel_regex,
        }

    profiler = RuntimeProfiler(
        RuntimeProfilerConfig(
            output_dir=args.output_dir,
            sample_interval_ms=args.interval_ms,
            gpu_index=args.gpu_index,
            backend_preference=backend_order,
            detail_plot_mode=args.detail_plot,
            enable_torch_profiler=args.enable_torch_profiler,
            torch_profiler_record_shapes=args.torch_profiler_record_shapes,
            torch_profiler_profile_memory=not args.no_torch_profiler_memory,
            torch_profiler_with_stack=args.torch_profiler_with_stack,
            torch_profiler_with_flops=args.torch_profiler_with_flops,
            torch_profiler_export_chrome_trace=not args.no_torch_profiler_trace,
            torch_profiler_topk_ops=args.torch_profiler_topk_ops,
            ncu=ncu_cfg,
        )
    )
    if args.inject_profiler_kwarg:
        target_kwargs[args.inject_profiler_kwarg] = profiler
    result = profiler.run(target_fn, *target_args, **target_kwargs)
    print("Profile finished:", result.output_dir)
    print_profile_summary(result.summary)


if __name__ == "__main__":
    main()
