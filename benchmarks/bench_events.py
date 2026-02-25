"""Micro-benchmark all LayerNorm kernel variants using MPS events.

Compares PyTorch's built-in LayerNorm against the seven progressive
Metal kernels (naive → shared → simd → vectorized → fused → robust →
register-tiled), reporting kernel time, achieved memory bandwidth, and
GFLOPS for each.

Sweeps across batch sizes to distinguish dispatch overhead from kernel
throughput.  At small B the fixed per-dispatch cost dominates; at large B
the memory bus saturates and bandwidth differences between kernels emerge.

Usage:
    pip install -e .            # build the extension first
    python benchmarks/bench_events.py              # default sweep
    python benchmarks/bench_events.py --B 4096     # single batch size
    python benchmarks/bench_events.py --sweep-N    # also sweep hidden dim
"""

import argparse
import torch
from layernorm_metal import layernorm_forward, KERNELS

# ----- Config ----------------------------------------------------------------
NUM_WARMUP = 10
NUM_ITERS = 100
EPS = 1e-5

device = torch.device("mps")

# Batch sizes to sweep — from dispatch-dominated to bandwidth-saturated
B_SWEEP = [32, 128, 512, 2048, 8192]

# Additional (B, N) configs for hidden-dim sweep
N_SWEEP = [
    (4096, 768),    # GPT-2 small
    (4096, 1024),   # GPT-2 medium
    (4096, 2048),   # GPT-2 large / LLaMA-7B
    (4096, 4096),   # LLaMA-13B / larger models
]

# Memory traffic per element (bytes) for each kernel class.
# K1-K4: 3 reads of X (12) + gamma (4) + beta (4) + write (4) = 24
# K5/K6: 2 reads of X  (8) + gamma (4) + beta (4) + write (4) = 20
# K7:    1 read of X   (4) + gamma (4) + beta (4) + write (4) = 16
BYTES_PER_ELEM = {
    "naive": 24, "shared": 24, "simd": 24, "vectorized": 24,
    "fused": 20, "robust": 20, "regtiled": 16,
}
PYTORCH_BYTES_PER_ELEM = 20  # PyTorch native kernel reads X twice

KERNEL_ORDER = ["naive", "shared", "simd", "vectorized",
                "fused", "robust", "regtiled"]
LABELS = {
    "naive":      "K1: Naive (1 thread/row)",
    "shared":     "K2: Threadgroup reduction",
    "simd":       "K3: SIMD reduction",
    "vectorized": "K4: Vectorized float4",
    "fused":      "K5: Fused single-pass",
    "robust":     "K6: Robust (tail+precise)",
    "regtiled":   "K7: Register-tiled",
}


def bench_pytorch_layernorm(x: torch.Tensor) -> float:
    """Benchmark PyTorch's built-in nn.LayerNorm. Returns avg ms."""
    model = torch.nn.LayerNorm(x.shape[-1]).to(device)

    for _ in range(NUM_WARMUP):
        _ = model(x)
    torch.mps.synchronize()

    start = torch.mps.event.Event(enable_timing=True)
    end = torch.mps.event.Event(enable_timing=True)

    start.record()
    for _ in range(NUM_ITERS):
        _ = model(x)
    end.record()
    torch.mps.synchronize()

    return start.elapsed_time(end) / NUM_ITERS


def bench_custom_kernel(x: torch.Tensor, gamma: torch.Tensor,
                        beta: torch.Tensor, kernel: str) -> float:
    """Benchmark a custom Metal LayerNorm kernel variant. Returns avg ms."""
    for _ in range(NUM_WARMUP):
        _ = layernorm_forward(x, gamma, beta, EPS, kernel=kernel)
    torch.mps.synchronize()

    start = torch.mps.event.Event(enable_timing=True)
    end = torch.mps.event.Event(enable_timing=True)

    start.record()
    for _ in range(NUM_ITERS):
        _ = layernorm_forward(x, gamma, beta, EPS, kernel=kernel)
    end.record()
    torch.mps.synchronize()

    return start.elapsed_time(end) / NUM_ITERS


def compute_bandwidth(B: int, N: int, time_ms: float,
                      bytes_per_elem: int = 20) -> float:
    """Compute achieved memory bandwidth in GB/s."""
    bytes_total = B * bytes_per_elem * N
    time_s = time_ms / 1000.0
    return bytes_total / time_s / 1e9


def compute_gflops(B: int, N: int, time_ms: float) -> float:
    """Compute achieved GFLOPS (~8 FLOPs per element)."""
    flops = B * 8 * N
    time_s = time_ms / 1000.0
    return flops / time_s / 1e9


def run_config(B: int, N: int):
    """Run all kernels for a single (B, N) configuration."""
    x = torch.randn(B, N, device=device)
    gamma = torch.ones(N, device=device)
    beta = torch.zeros(N, device=device)

    total_bytes_k5 = B * N * 20
    print(f"\n{'='*72}")
    print(f"  B={B}, N={N}, FP32  "
          f"(working set: {total_bytes_k5 / 1024:.0f} KB @ 20 B/elem)")
    print(f"  Warmup: {NUM_WARMUP} iters, Measured: {NUM_ITERS} iters")
    print(f"{'='*72}")

    # Header
    print(f"  {'Kernel':<30s} {'Time (ms)':>10s}  {'BW (GB/s)':>10s}"
          f"  {'GFLOPS':>8s}")
    print(f"  {'-'*68}")

    # PyTorch built-in
    pytorch_ms = bench_pytorch_layernorm(x)
    pytorch_bw = compute_bandwidth(B, N, pytorch_ms,
                                   bytes_per_elem=PYTORCH_BYTES_PER_ELEM)
    pytorch_gf = compute_gflops(B, N, pytorch_ms)
    print(f"  {'PyTorch nn.LayerNorm':<30s} {pytorch_ms:>10.4f}"
          f"  {pytorch_bw:>10.2f}  {pytorch_gf:>8.4f}")

    # Custom kernels
    results = {}
    for k in KERNEL_ORDER:
        ms = bench_custom_kernel(x, gamma, beta, kernel=k)
        bw = compute_bandwidth(B, N, ms, bytes_per_elem=BYTES_PER_ELEM[k])
        gf = compute_gflops(B, N, ms)
        results[k] = ms
        print(f"  {LABELS[k]:<30s} {ms:>10.4f}  {bw:>10.2f}  {gf:>8.4f}")

    # Speedup summary
    print(f"  {'-'*68}")
    naive_ms = results["naive"]
    for k in KERNEL_ORDER:
        speedup = naive_ms / results[k] if results[k] > 0 else float('inf')
        vs_pytorch = pytorch_ms / results[k] if results[k] > 0 else float('inf')
        print(f"  {LABELS[k]:<30s}  {speedup:>5.1f}x vs K1"
              f"  {vs_pytorch:>6.2f}x vs PyTorch")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LayerNorm kernel variants.")
    parser.add_argument(
        "--B", type=int, default=None,
        help="Single batch size (skips sweep)")
    parser.add_argument(
        "--N", type=int, default=768,
        help="Hidden dimension (default: 768)")
    parser.add_argument(
        "--sweep-N", action="store_true",
        help="Also sweep across hidden dimensions at B=4096")
    args = parser.parse_args()

    print("LayerNorm Benchmark Suite")
    print(f"  Arithmetic intensity (FLOP/byte):")
    print(f"    K1-K4: 8 / 24 = {8/24:.2f}  (3 reads of X)")
    print(f"    K5/K6: 8 / 20 = {8/20:.2f}  (2 reads of X)")
    print(f"    K7:    8 / 16 = {8/16:.2f}  (1 read of X)")

    if args.B is not None:
        # Single configuration
        run_config(args.B, args.N)
    else:
        # Batch size sweep at fixed N
        print(f"\n--- Batch size sweep (N={args.N}) ---")
        for B in B_SWEEP:
            run_config(B, args.N)

        # Optional hidden dim sweep
        if args.sweep_N:
            print(f"\n--- Hidden dimension sweep ---")
            for B, N in N_SWEEP:
                run_config(B, N)


if __name__ == "__main__":
    main()
