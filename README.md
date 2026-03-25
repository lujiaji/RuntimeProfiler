# Runtime Profiler

面向**瓶颈归因**的个人分析工具：在线轻量总览 + 可选离线精剖（Nsight Compute / CUPTI 预留），统一输出「运行时间主要被什么资源限制」的结论与证据链。

## 目标与结论标签

最终问题不是「显存高不高」，而是**时间主要被谁限制**。标签包括：

- `compute_bound` — SM / Tensor / FP / INT 等算力管线成为主矛盾  
- `memory_bound` — **设备端** DRAM（HBM/GDDR）带宽或相关压力为主（需 measured DRAM 或离线 kernel 证据才有高置信度）  
- `pcie_io_bound` — **主机–设备** PCIe 搬运（H2D/D2H、paging、CPU KV 等）与运行时间强相关  
- `mixed_bound` — 不同阶段或多种资源同时显著  
- `latency_bound` — 利用率整体不高但耗时长：同步空洞、kernel 过碎、stall、occupancy 过低等  

### 三类资源域（必须区分）

| 域 | 含义 |
|----|------|
| **Compute** | SM / Tensor Core / FP / INT |
| **Device memory** | 片上 HBM/GDDR 带宽与设备内存子系统（不是 PCIe） |
| **Host–device IO** | PCIe（后续可扩展 NVLink） |

### 术语与常见误区

1. **PCIe 吞吐 ≠ 设备 DRAM 带宽**：前者是 CPU↔GPU 链路，后者是 GPU 上 HBM/GDDR。  
2. **`dram_bw_proxy_gbps`**：由 legacy NVML 等推导的近似，**不是**硬件实测 DRAM 吞吐。  
3. **高置信度 `memory_bound`**：需要 NVML GPM 的 DRAM BW 类指标，或 Nsight Compute / CUPTI 的 kernel 级证据。  
4. **高置信度 `compute_bound`**：需要 SM/Tensor/管线利用率或离线 throughput 指标支撑。  
5. **「显存占用高」**：多为 **capacity / residency**（容量压力），**不等于** `memory_bound`（吞吐/访存受限）。  
6. **KV cache / attention 类问题**：同时看 **DRAM BW**（片上供给）、**PCIe**（与 CPU 间搬运）、**SM/Tensor/occupancy**（是否算力吃满）。

## Python API

```python
from runtime_profiler import RuntimeProfiler, RuntimeProfilerConfig

profiler = RuntimeProfiler(
    RuntimeProfilerConfig(
        output_dir="./runtime_profile",
        sample_interval_ms=20,
        backend_preference=["nvml_gpm", "nvml_legacy", "torch_cuda"],
        detail_plot_mode=False,
    )
)
result = profiler.run(target_fn, *args, **kwargs)
print(result.summary["classification"])
print(result.summary["merged_diagnosis_preview"])
```

- `backend_preference` 默认优先 **NVML GPM + legacy 补齐**；`nvml` 等价于 `nvml_legacy`。  
- `classification`：`label`, `confidence`, `primary_evidence`, `secondary_evidence`, `metric_coverage`, `limitations` 等。  
- `bound_classification`：兼容旧字段（内含 `structured` 全量分类）。  
- 可选离线 NCU：`RuntimeProfilerConfig(ncu={"enabled": True, "target_python_entry": "...", ...})`（会单独跑 `ncu`，耗时长）。

## 输出文件

| 文件 | 说明 |
|------|------|
| `profile_samples.csv` | 在线采样（含时间窗、活跃 event、各域指标、source_*） |
| `profile_events.json` | 带 `metrics_rollup` 与 `event_classification` 的 trace 事件 |
| `profile_event_summary.csv` | 按事件的 duration / avg / p95 / max 聚合 |
| `profile_summary.json` | 全局分类、top 事件列表、能力声明等 |
| `profile_main_figure.png` | **六面板**主诊断图（时间轴对齐） |
| `profile_event_bars.png` | 事件时长 + 平均 sm / dram_bw / pcie |
| `profile_event_heatmap.png` | 事件 × 指标归一化热力图 |
| `profile_events_timeline.png` | 事件甘特（legacy 单列图） |
| `merged_diagnosis_report.json` | 在线 + 离线合并预览 |
| `kernel_level_metrics.csv/json` | 离线 NCU 解析结果（若存在） |
| `kernel_bound_summary.json` | 离线 kernel 级粗结论 |
| `kernel_drilldown.png` | 离线 kernel 条形图（若有 kernel 摘要） |

主图六面板：**事件时间线** | **Compute** | **Device memory 压力** | **Capacity/显存占用** | **PCIe** | **功耗/频率/温度**。

## 通用 CLI

```bash
python runtime_profiler/scripts/profile_any_model.py \
  --pythonpath /path/to/your/project \
  --target-module your_module.runner \
  --target-fn run_inference \
  --target-args-json '[]' \
  --target-kwargs-json '{"batch_size":8}' \
  --output-dir ./runtime_profile \
  --interval-ms 20 \
  --backend-order nvml_gpm,nvml_legacy,torch_cuda
```

阶段级 trace：向目标函数注入 profiler，例如 `--inject-profiler-kwarg tracer`，在代码里 `with tracer.trace("prefill"): ...`。

## 离线精剖（路线 B）

- **Nsight Compute**：`offline/ncu_runner.py`、`offline/ncu_parser.py`；配置见 `NCUConfig` / `RuntimeProfilerConfig.ncu`。预设：`bound_basic`、`stall_debug`、`roofline`。  
- **CUPTI**：`offline/cupti_runner.py`、`offline/cupti_range_bridge.py` 为预留接口，第一版未完全打通。

## 实现优先级对照

- **P0**：NvmlGpmBackend（与 legacy 组合的在线后端）、采样 schema、六面板主图、事件聚合、含 `pcie_io_bound` 的分类逻辑 — **已落地**。  
- **P1**：ncu runner/parser、kernel drilldown、合并诊断 — **框架与解析已接好；完整 ncu 工作流依赖本机 `ncu` 与驱动**。  
- **P2**：CUPTI 打通、roofline 专用图、更细 stall/PC — **接口预留**。

## 项目布局

```text
runtime_profiler/
  core.py
  metrics_backends.py
  event_tracer.py
  event_aggregate.py
  classifiers.py
  plotter.py
  merge_diagnosis.py
  offline/
    ncu_runner.py
    ncu_parser.py
    cupti_runner.py
    cupti_range_bridge.py
  adapters/ ...
  utils/ ...
  scripts/ ...
  examples/ ...
```

## 依赖说明

- **NVML**：`pynvml` 或 `nvidia-ml-py`；GPM API 随驱动/绑定版本变化，不可用时自动退化为 legacy + torch。  
- **Torch**：无 NVML 时仅能做 allocator 与有限结论（低置信度）。  
- **matplotlib / numpy**：用于出图（可选）。
