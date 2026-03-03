"""Profile LayerNorm using torch.profiler with CPU activity tracing.

Captures CPU-side dispatch timing for MPS operations and exports a
Chrome trace JSON that can be visualized at https://ui.perfetto.dev/

Supports PyTorch's native LayerNorm and any combination of custom Metal
kernels, with configurable batch size, hidden dimension, and iteration count.

Note: ProfilerActivity.MPS does not exist in PyTorch (as of 2.10).
CPU-only profiling still captures MPS op dispatch overhead — useful for
identifying whether the CPU dispatch or the GPU kernel is the bottleneck.
For GPU-side timing, use bench_mps_events.py (MPS events) or gpu_capture.py
(Xcode GPU trace).

Usage:
    python benchmarks/profile_torch.py                        # pytorch only
    python benchmarks/profile_torch.py -k all                 # all kernels
    python benchmarks/profile_torch.py -k pytorch fused k7    # specific set
    python benchmarks/profile_torch.py --B 256 -N 1024        # custom size
    python benchmarks/profile_torch.py --seq-len 128          # 3D [B,S,N]

Outputs:
    - Console table with per-op CPU time breakdown
    - layernorm_profile_{kernel}.json  (Chrome trace per kernel)

Requirements:
    pip install torch
    Must run on macOS with MPS-capable GPU (Apple Silicon).
"""

from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

from utils import (
    KERNEL_META,
    resolve_kernels, output_path_for_kernel, ensure_parent_dir,
    EPS, DTYPE, N_DEFAULT,
)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
B_DEFAULT = 1024             # batch size
WARMUP_DEFAULT = 10
NUM_ITERS_DEFAULT = 100

assert torch.backends.mps.is_available(), "MPS backend not available"
device = torch.device("mps")


# ──────────────────────────────────────────────
# Profiling
# ──────────────────────────────────────────────

def profile_kernel(kernel: str, B: int, N: int, num_iters: int,
                   warmup: int, seq_len: int | None = None) -> profile:
    """Profile a single kernel. Returns the profiler object."""
    if seq_len is not None:
        x = torch.randn(B, seq_len, N, device=device, dtype=DTYPE)
    else:
        x = torch.randn(B, N, device=device, dtype=DTYPE)

    gamma = torch.ones(N, device=device, dtype=DTYPE)
    beta = torch.zeros(N, device=device, dtype=DTYPE)

    if kernel == "pytorch":
        def run():
            F.layer_norm(x, (N,), gamma, beta, EPS)
    else:
        from layernorm_metal import layernorm_forward
        # Metal kernels expect 2D [rows, N] input
        x_2d = x.view(-1, N)

        def run():
            layernorm_forward(x_2d, gamma, beta, EPS, kernel=kernel)

    # Warmup
    for _ in range(warmup):
        run()
    torch.mps.synchronize()

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function(f"layernorm_profile_{kernel}"):
            for _ in range(num_iters):
                run()
        torch.mps.synchronize()

    return prof


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Profile LayerNorm kernels on MPS with torch.profiler.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Kernel names: pytorch, naive (k1), shared (k2), simd (k3),\n"
            "              vectorized (k4), fused (k5), robust (k6), "
            "regtiled (k7)\n"
            "              naive_1024 (k1b)\n"
            "Use 'all' to select every kernel."
        ))
    parser.add_argument(
        "-k", "--kernels", nargs="+", default=["pytorch"],
        help="Kernels to profile (default: pytorch)")
    parser.add_argument(
        "-N", type=int, default=N_DEFAULT,
        help=f"Hidden dimension (default: {N_DEFAULT})")
    parser.add_argument(
        "--B", type=int, default=B_DEFAULT,
        help=f"Batch size (default: {B_DEFAULT})")
    parser.add_argument(
        "--num-iters", type=int, default=NUM_ITERS_DEFAULT,
        help=f"Number of profiled iterations (default: {NUM_ITERS_DEFAULT})")
    parser.add_argument(
        "--warmup", type=int, default=WARMUP_DEFAULT,
        help=f"Warmup iterations (default: {WARMUP_DEFAULT})")
    parser.add_argument(
        "--seq-len", type=int, default=None,
        help="Sequence length for 3D input [B, S, N] (default: 2D)")
    parser.add_argument(
        "-o", "--output", type=str, default="traces/layernorm_profile_{kernel}.json",
        help="Output trace filename; {kernel} is replaced with kernel name "
             "(default: traces/layernorm_profile_{kernel}.json)")
    args = parser.parse_args()

    kernels = resolve_kernels(args.kernels)
    B, N = args.B, args.N
    if args.seq_len:
        shape_str = f"[{B}, {args.seq_len}, {N}]"
    else:
        shape_str = f"[{B}, {N}]"

    print(f"LayerNorm Profiler — shape={shape_str}, kernels: {kernels}")
    print(f"  Warmup: {args.warmup} iters, Profiled: {args.num_iters} iters\n")

    trace_files = []

    for kernel in kernels:
        label = KERNEL_META[kernel]["label"]
        print("=" * 60)
        print(f"  Profiling: {label}")
        print("=" * 60 + "\n")

        prof = profile_kernel(kernel, B, N, args.num_iters,
                              args.warmup, args.seq_len)

        # Print per-op breakdown
        print(prof.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=15))

        # Export Chrome trace
        trace_path = output_path_for_kernel(args.output, kernel)
        ensure_parent_dir(trace_path)
        prof.export_chrome_trace(trace_path)
        trace_files.append((kernel, trace_path))
        print(f"\nChrome trace exported to {trace_path}\n")

    # Summary
    if len(kernels) > 1:
        print("\n" + "=" * 60)
        print("  Trace files generated:")
        print("=" * 60)
        for kernel, path in trace_files:
            label = KERNEL_META[kernel]["label"]
            print(f"  {label:<30s}  {path}")

    print("\nOpen traces at https://ui.perfetto.dev/ for visual timelines.")


if __name__ == "__main__":
    main()
