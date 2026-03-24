import argparse
import os
import sys


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_PROFILER_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

sys.path.insert(0, RUNTIME_PROFILER_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./runtime_profile_cropformer")
    parser.add_argument("--interval-ms", type=int, default=20)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--cropformer-root",
        type=str,
        default=os.environ.get("CROPFORMER_ROOT", ""),
        help="Path to CropFormer project root (or set CROPFORMER_ROOT env var).",
    )
    args = parser.parse_args()

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

    from runtime_profiler import RuntimeProfiler, RuntimeProfilerConfig  # noqa: E402
    from mask_predict_single_seq_w_semantic import run_mask2d_via_root  # noqa: E402

    config = RuntimeProfilerConfig(
        output_dir=args.output_dir,
        sample_interval_ms=args.interval_ms,
        gpu_index=args.gpu_index,
    )
    profiler = RuntimeProfiler(config=config)
    result = profiler.run(run_mask2d_via_root, None)
    print("Profile finished:", result.output_dir)
    print("Summary:", result.summary.get("bound_classification", {}))


if __name__ == "__main__":
    main()
