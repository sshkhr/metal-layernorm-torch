"""Capture a Metal GPU trace for analysis in Xcode.

Produces a .gputrace file that can be opened in Xcode to inspect
per-line shader costs, register usage, memory bandwidth, and GPU counters.

Usage:
    METAL_CAPTURE_ENABLED=1 python benchmarks/gpu_capture.py
    METAL_CAPTURE_ENABLED=1 python benchmarks/gpu_capture.py --kernel naive

Then open the .gputrace file:
    open /tmp/layernorm.gputrace
"""

import argparse
import torch
from layernorm_metal import layernorm_forward, start_gpu_capture, stop_gpu_capture

# ----- Config ----------------------------------------------------------------
B, N = 32, 768
EPS = 1e-5

parser = argparse.ArgumentParser(description="Capture Metal GPU trace.")
parser.add_argument("--kernel", default="vectorized",
                    choices=["naive", "shared", "simd", "vectorized",
                             "fused", "robust", "regtiled"],
                    help="Kernel variant to capture (default: vectorized)")
parser.add_argument("-o", "--output", default="/tmp/layernorm.gputrace",
                    help="Output .gputrace path")
args = parser.parse_args()

# ----- Setup -----------------------------------------------------------------
device = torch.device("mps")
x = torch.randn(B, N, device=device)
gamma = torch.ones(N, device=device)
beta = torch.zeros(N, device=device)

# Warm-up (compiles the shader so the capture only contains dispatch)
_ = layernorm_forward(x, gamma, beta, EPS, kernel=args.kernel)
torch.mps.synchronize()

# ----- Capture ---------------------------------------------------------------
print(f"Capturing kernel='{args.kernel}' → {args.output}")
start_gpu_capture(args.output)

for _ in range(10):
    output = layernorm_forward(x, gamma, beta, EPS, kernel=args.kernel)

torch.mps.synchronize()
stop_gpu_capture()

print(f"GPU trace saved to {args.output}")
print(f"Open with: open {args.output}")
