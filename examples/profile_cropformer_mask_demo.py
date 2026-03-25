import argparse
import os
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Package layout: <parent>/runtime_profiler/__init__.py — Python needs <parent> on sys.path.
_RUNTIME_PROFILER_REPO = os.path.abspath(os.path.join(THIS_DIR, ".."))
_RUNTIME_PROFILER_PARENT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, _RUNTIME_PROFILER_PARENT)


def main():
    # CropFormer's run_mask2d_via_root() calls get_parser().parse_args() and reads global sys.argv.
    # We must consume profiler-only flags here and forward the rest (e.g. --root, --opts) unchanged.
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./runtime_profile_cropformer")
    parser.add_argument("--interval-ms", type=int, default=20)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--enable-torch-profiler", action="store_true")
    parser.add_argument(
        "--cropformer-root",
        type=str,
        default=os.environ.get("CROPFORMER_ROOT", ""),
        help="Path to CropFormer project root (or set CROPFORMER_ROOT env var).",
    )
    args, forward_argv = parser.parse_known_args()

    if not args.cropformer_root:
        raise ValueError(
            "Missing CropFormer path. Provide --cropformer-root or set CROPFORMER_ROOT."
        )
    demo_root = os.path.join(os.path.abspath(args.cropformer_root), "demo_cropformer")
    if not os.path.isdir(demo_root):
        raise FileNotFoundError(
            f"Invalid CropFormer root: {args.cropformer_root}. "
            "Expected demo folder at <root>/demo_cropformer."
        )
    sys.path.insert(0, demo_root)

    saved_argv = sys.argv[:]
    sys.argv = [saved_argv[0]] + forward_argv

    from runtime_profiler import RuntimeProfiler, RuntimeProfilerConfig  # noqa: E402
    from mask_predict_single_seq_w_semantic import run_mask2d_via_root  # noqa: E402

    config = RuntimeProfilerConfig(
        output_dir=args.output_dir,
        sample_interval_ms=args.interval_ms,
        gpu_index=args.gpu_index,
        enable_torch_profiler=args.enable_torch_profiler,
    )
    profiler = RuntimeProfiler(config=config)
    try:
        result = profiler.run(run_mask2d_via_root, None)
    finally:
        sys.argv = saved_argv
    print("Profile finished:", result.output_dir)
    print("Summary:", result.summary.get("bound_classification", {}))


if __name__ == "__main__":
    main()
