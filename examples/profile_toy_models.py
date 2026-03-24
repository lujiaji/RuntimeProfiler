import argparse
import os
import sys

import torch
import torch.nn as nn


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, TOOLS_DIR)

from runtime_profiler import RuntimeProfiler, RuntimeProfilerConfig  # noqa: E402
from runtime_profiler.adapters import trace_module_forward  # noqa: E402


class TinyCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


class TinyTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.encoder(x)
        return self.proj(x)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def profile_case(name: str, target_fn, output_root: str):
    cfg = RuntimeProfilerConfig(output_dir=os.path.join(output_root, name), sample_interval_ms=10)
    profiler = RuntimeProfiler(config=cfg)
    result = profiler.run(target_fn)
    print(name, "=>", result.summary.get("bound_classification", {}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, default="./runtime_profile_toys")
    args = parser.parse_args()

    device = _device()
    torch.manual_seed(0)

    cnn = TinyCnn().to(device).eval()
    x_img = torch.randn(16, 3, 256, 256, device=device)

    def run_cnn():
        with torch.no_grad():
            for _ in range(20):
                _ = cnn(x_img)

    profile_case("cnn", run_cnn, args.output_root)

    transformer = TinyTransformer().to(device).eval()
    x_seq = torch.randn(8, 256, 256, device=device)

    def run_transformer():
        with torch.no_grad():
            for _ in range(20):
                _ = transformer(x_seq)

    profile_case("transformer", run_transformer, args.output_root)

    def run_functional():
        a = torch.randn(2048, 2048, device=device)
        b = torch.randn(2048, 2048, device=device)
        for _ in range(20):
            _ = a @ b

    profile_case("functional", run_functional, args.output_root)

    # Optional module-level events on transformer
    cfg = RuntimeProfilerConfig(output_dir=os.path.join(args.output_root, "transformer_with_hooks"))
    profiler = RuntimeProfiler(config=cfg)
    with trace_module_forward(profiler, transformer, include_types=["Linear", "MultiheadAttention"], max_modules=32):
        result = profiler.run(run_transformer)
    print("transformer_with_hooks =>", result.summary.get("bound_classification", {}))


if __name__ == "__main__":
    main()
