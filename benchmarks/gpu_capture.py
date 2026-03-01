"""Capture a Metal GPU trace for analysis in Xcode.

Produces a .gputrace file that can be opened in Xcode to inspect
per-line shader costs, register usage, memory bandwidth, and GPU counters.

Supports any combination of custom Metal kernels, with configurable
batch size, hidden dimension, warmup, and iteration count.

Usage:
    METAL_CAPTURE_ENABLED=1 python benchmarks/gpu_capture.py
    METAL_CAPTURE_ENABLED=1 python benchmarks/gpu_capture.py -k all
    METAL_CAPTURE_ENABLED=1 python benchmarks/gpu_capture.py -k naive fused k7
    METAL_CAPTURE_ENABLED=1 python benchmarks/gpu_capture.py --B 64 -N 1024
    METAL_CAPTURE_ENABLED=1 python benchmarks/gpu_capture.py -k k4 --num-iters 20

Then open the .gputrace file:
    open /tmp/layernorm_vectorized.gputrace

Requirements:
    pip install torch
    Must run on macOS with MPS-capable GPU (Apple Silicon).
    Set METAL_CAPTURE_ENABLED=1 in the environment.
"""

from __future__ import annotations

import argparse
import torch
from layernorm_metal import layernorm_forward, start_gpu_capture, stop_gpu_capture

from utils import (
    KERNEL_META, KERNEL_ORDER,
    resolve_kernels, output_path_for_kernel, EPS, DTYPE, N_DEFAULT,
)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
B_DEFAULT = 32
WARMUP_DEFAULT = 5
NUM_ITERS_DEFAULT = 10

# Custom kernels only (no "pytorch" — GPU capture requires Metal shaders)
CUSTOM_KERNEL_ORDER = [k for k in KERNEL_ORDER if k != "pytorch"]

assert torch.backends.mps.is_available(), "MPS backend not available"
device = torch.device("mps")


# ──────────────────────────────────────────────
# GPU capture
# ──────────────────────────────────────────────

def capture_kernel(kernel: str, B: int, N: int, num_iters: int,
                   warmup: int, output_path: str):
    """Run a GPU capture for a single kernel."""
    label = KERNEL_META[kernel]["label"]

    x = torch.randn(B, N, device=device, dtype=DTYPE)
    gamma = torch.ones(N, device=device, dtype=DTYPE)
    beta = torch.zeros(N, device=device, dtype=DTYPE)

    # Warmup (compiles the shader so the capture only contains dispatch)
    for _ in range(warmup):
        _ = layernorm_forward(x, gamma, beta, EPS, kernel=kernel)
    torch.mps.synchronize()

    # Capture
    print(f"  Capturing: {label}  →  {output_path}")
    start_gpu_capture(output_path)

    for _ in range(num_iters):
        _ = layernorm_forward(x, gamma, beta, EPS, kernel=kernel)

    torch.mps.synchronize()
    stop_gpu_capture()
    print(f"  Done.\n")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Capture Metal GPU traces for LayerNorm kernels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Kernel names: naive (k1), shared (k2), simd (k3),\n"
            "              vectorized (k4), fused (k5), robust (k6), "
            "regtiled (k7)\n"
            "              naive_1024 (k1b)\n"
            "Use 'all' to select every kernel."
        ))
    parser.add_argument(
        "-k", "--kernels", nargs="+", default=["vectorized"],
        help="Kernels to capture (default: vectorized)")
    parser.add_argument(
        "-N", type=int, default=N_DEFAULT,
        help=f"Hidden dimension (default: {N_DEFAULT})")
    parser.add_argument(
        "--B", type=int, default=B_DEFAULT,
        help=f"Batch size (default: {B_DEFAULT})")
    parser.add_argument(
        "--num-iters", type=int, default=NUM_ITERS_DEFAULT,
        help=f"Iterations inside capture (default: {NUM_ITERS_DEFAULT})")
    parser.add_argument(
        "--warmup", type=int, default=WARMUP_DEFAULT,
        help=f"Warmup iterations before capture (default: {WARMUP_DEFAULT})")
    parser.add_argument(
        "-o", "--output", type=str,
        default="/tmp/layernorm_{kernel}.gputrace",
        help="Output .gputrace path; {kernel} is replaced with kernel name "
             "(default: /tmp/layernorm_{kernel}.gputrace)")
    args = parser.parse_args()

    kernels = resolve_kernels(args.kernels, valid_kernels=CUSTOM_KERNEL_ORDER)
    B, N = args.B, args.N

    print(f"GPU Capture — shape=[{B}, {N}], kernels: {kernels}")
    print(f"  Warmup: {args.warmup} iters, Captured: {args.num_iters} iters\n")

    trace_files = []

    for kernel in kernels:
        print("=" * 60)
        output_path = output_path_for_kernel(args.output, kernel)
        capture_kernel(kernel, B, N, args.num_iters, args.warmup, output_path)
        trace_files.append((kernel, output_path))

    # Summary
    print("=" * 60)
    print("  Trace files generated:")
    print("=" * 60)
    for kernel, path in trace_files:
        label = KERNEL_META[kernel]["label"]
        print(f"  {label:<30s}  {path}")

    print(f"\nOpen traces with: open <path>")


if __name__ == "__main__":
    main()
