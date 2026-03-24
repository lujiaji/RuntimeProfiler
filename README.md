# Runtime Profiler

Lightweight, model-agnostic runtime profiler for GPU bottleneck analysis.

It helps answer:
- Is the workload `compute_bound` or `memory_bound`?
- What are peak memory / bandwidth-proxy metrics?
- Which internal function was active when the peak happened?

The core API is function-based:

```python
from runtime_profiler import RuntimeProfiler, RuntimeProfilerConfig

profiler = RuntimeProfiler(
    RuntimeProfilerConfig(output_dir="./runtime_profile", sample_interval_ms=20)
)
result = profiler.run(target_fn, *args, **kwargs)
print(result.summary["bound_classification"])
```

## Features

- Model-agnostic profiling via `profiler.run(target_fn, *args, **kwargs)`
- Time-series GPU sampling (NVML first, Torch CUDA fallback)
- Function-level event tracing (`trace` context manager and decorator)
- Peak attribution (maps peak timestamps to innermost active function)
- Bound classification (`compute_bound` / `memory_bound` / `mixed_bound`)
- Export to CSV/JSON and timeline plots

## Quick Start

### 1) Generic CLI (recommended)

Use the reusable script for any project by passing a module + function:

```bash
python runtime_profiler/scripts/profile_any_model.py \
  --pythonpath /path/to/your/project \
  --target-module your_module.runner \
  --target-fn run_inference \
  --target-args-json '[]' \
  --target-kwargs-json '{"batch_size":8}' \
  --output-dir ./runtime_profile \
  --interval-ms 20
```

If your target function accepts the profiler object (for stage-level tracing), inject it:

```bash
--inject-profiler-kwarg tracer
```

Example target signature:

```python
def run_inference(..., tracer=None):
    if tracer is not None:
        with tracer.trace("load_model"):
            ...
        with tracer.trace("forward"):
            ...
```

### 2) Python API

```python
from runtime_profiler import RuntimeProfiler, RuntimeProfilerConfig

def target_fn():
    ...

profiler = RuntimeProfiler(
    RuntimeProfilerConfig(
        output_dir="./runtime_profile",
        sample_interval_ms=20,
        gpu_index=0,
    )
)
result = profiler.run(target_fn)
```

## Output Artifacts

- `profile_samples.csv`: sampled GPU metrics over time
- `profile_events.json`: traced function event windows
- `profile_summary.json`: peaks, peak locations, bound classification
- `profile_timeline.png`: utilization/memory timeline (if matplotlib is available)
- `profile_events_timeline.png`: event Gantt timeline (if matplotlib is available)

## Project Layout

```text
runtime_profiler/
  __init__.py
  core.py
  metrics_backends.py
  event_tracer.py
  classifiers.py
  plotter.py
  adapters/
    __init__.py
    function_adapter.py
    pytorch_adapter.py
  utils/
    __init__.py
    loader.py
    parsing.py
    report.py
  scripts/
    __init__.py
    profile_any_model.py
    template_target.py
  examples/
    profile_cropformer_mask_demo.py
    profile_toy_models.py
```

## CLI Reference (`scripts/profile_any_model.py`)

- `--target-module` (required): module path containing the target callable
- `--target-fn` (required): callable name inside the module
- `--pythonpath`: prepend this path to `sys.path` before loading module
- `--target-args-json`: JSON list for positional args
- `--target-kwargs-json`: JSON object for keyword args
- `--inject-profiler-kwarg`: inject profiler object into kwargs with this key
- `--output-dir`: output directory for artifacts
- `--interval-ms`: sampling interval in milliseconds
- `--gpu-index`: GPU index for sampling
- `--backend-order`: backend preference, e.g. `nvml,torch_cuda`

## Included Examples

- `examples/profile_toy_models.py`: tiny CNN/Transformer/functional workloads
- `examples/profile_cropformer_mask_demo.py`: CropFormer integration example

For CropFormer example:

```bash
python runtime_profiler/examples/profile_cropformer_mask_demo.py \
  --cropformer-root /path/to/CropFormer \
  --output-dir ./runtime_profile_cropformer
```

## Notes

- If NVML is unavailable, the profiler falls back to `torch_cuda`; some metrics become proxies.
- Classification quality improves with richer backend metrics and stable workloads.
- For kernel-level analysis, integrate with lower-level profilers separately (e.g., CUPTI/Nsight).
