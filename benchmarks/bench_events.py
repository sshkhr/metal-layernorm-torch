"""Micro-benchmark all LayerNorm kernel variants using MPS events.

Compares PyTorch's built-in LayerNorm against the four progressive
Metal kernels (naive → shared → simd → vectorized), reporting kernel
time, achieved memory bandwidth, and GFLOPS for each.

Usage:
    pip install -e .   # build the extension first
    python benchmarks/bench_events.py
"""

import torch
from layernorm_metal import layernorm_forward, KERNELS

# ----- Config ----------------------------------------------------------------
B, N = 32, 768
NUM_WARMUP = 10
NUM_ITERS = 100
EPS = 1e-5

device = torch.device("mps")


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


def compute_bandwidth(B: int, N: int, time_ms: float) -> float:
    """Compute achieved memory bandwidth in GB/s.

    LayerNorm reads x twice (two-pass), reads gamma and beta once each,
    and writes output once: 20 bytes per element in FP32.
    """
    bytes_total = B * 20 * N  # 20 bytes/element for two-pass LayerNorm
    time_s = time_ms / 1000.0
    return bytes_total / time_s / 1e9


def compute_gflops(B: int, N: int, time_ms: float) -> float:
    """Compute achieved GFLOPS.

    LayerNorm requires ~8 FLOPs per element.
    """
    flops = B * 8 * N
    time_s = time_ms / 1000.0
    return flops / time_s / 1e9


def main():
    x = torch.randn(B, N, device=device)
    gamma = torch.ones(N, device=device)
    beta = torch.zeros(N, device=device)

    print(f"LayerNorm benchmark: B={B}, N={N}, FP32")
    print(f"  Warmup: {NUM_WARMUP} iters, Measured: {NUM_ITERS} iters")
    print("=" * 70)

    # Header
    print(f"  {'Kernel':<28s} {'Time (ms)':>10s}  {'BW (GB/s)':>10s}"
          f"  {'GFLOPS':>8s}")
    print("-" * 70)

    # PyTorch built-in
    pytorch_ms = bench_pytorch_layernorm(x)
    pytorch_bw = compute_bandwidth(B, N, pytorch_ms)
    pytorch_gf = compute_gflops(B, N, pytorch_ms)
    print(f"  {'PyTorch nn.LayerNorm':<28s} {pytorch_ms:>10.4f}"
          f"  {pytorch_bw:>10.2f}  {pytorch_gf:>8.4f}")

    # All 4 custom kernels in progression order
    kernel_order = ["naive", "shared", "simd", "vectorized"]
    labels = {
        "naive":      "K1: Naive (1 thread/row)",
        "shared":     "K2: Threadgroup reduction",
        "simd":       "K3: SIMD reduction",
        "vectorized": "K4: Vectorized float4",
    }

    results = {}
    for k in kernel_order:
        ms = bench_custom_kernel(x, gamma, beta, kernel=k)
        bw = compute_bandwidth(B, N, ms)
        gf = compute_gflops(B, N, ms)
        results[k] = ms
        print(f"  {labels[k]:<28s} {ms:>10.4f}  {bw:>10.2f}  {gf:>8.4f}")

    # Summary
    print("=" * 70)
    naive_ms = results["naive"]
    for k in kernel_order:
        speedup = naive_ms / results[k] if results[k] > 0 else float('inf')
        print(f"  {labels[k]:<28s}  {speedup:>5.1f}x vs naive")

    ai = 0.4  # 8 FLOPs / 20 bytes per element
    print(f"\n  Arithmetic intensity: {ai:.1f} FLOP/byte (deeply memory-bound)")


if __name__ == "__main__":
    main()
