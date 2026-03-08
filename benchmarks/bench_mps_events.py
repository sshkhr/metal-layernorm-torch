"""
bench_mps_events.py — Benchmark LayerNorm kernels on MPS using GPU events.

Benchmarks PyTorch's native LayerNorm and any combination of custom Metal
kernels, sweeping across batch sizes and optionally hidden dimensions.
Produces console tables with speedup summaries, CSV output, and roofline plots.

Usage:
    python bench_mps_events.py                           # pytorch + naive
    python bench_mps_events.py -k all                    # all kernels
    python bench_mps_events.py -k pytorch fused k7       # specific set
    python bench_mps_events.py --B 4096                  # single batch size
    python bench_mps_events.py --sweep-N                 # sweep hidden dims
    python bench_mps_events.py --two-panel               # dual-panel plot
    python bench_mps_events.py --dram-ceiling            # show DRAM line

Outputs:
    - Console table with timing, GFLOP/s, bandwidth, speedup summary
    - benchmark-roofline.png  (plot, optional)
    - benchmark-results.csv   (raw data, optional)

Requirements:
    pip install torch matplotlib
    Must run on macOS with MPS-capable GPU (Apple Silicon).
"""

import argparse
import csv
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils import (
    KERNEL_META as _KERNEL_META,
    resolve_kernels, ensure_parent_dir, EPS, DTYPE, N_DEFAULT,
)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BATCH_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
WARMUP = 10
NUM_ITERS = 100

# ── M1 Max hardware specs ──
PEAK_BW_GB_S = 400.0        # LPDDR5 peak memory bandwidth (GB/s)
PEAK_FLOPS = 10.6e3         # FP32 GFLOP/s (10.6 TFLOPS)

# ── LayerNorm arithmetic constants ──
FLOPS_PER_ELEM = 8          # ~8 FLOPs per element

# Memory traffic per element (bytes) for each kernel class.
# Theoretical: assumes all reads/writes hit DRAM.
THEORETICAL_BYTES = {
    "pytorch":    20,  # 2 reads of X (8) + gamma (4) + beta (4) + write (4)
    "naive":      24,  # 3 reads of X (12) + gamma (4) + beta (4) + write (4)
    "naive_32":   24,
    "shared":     24,
    "simd":       24,
    "vectorized": 24,
    "fused":      20,  # 2 reads of X (8) + gamma (4) + beta (4) + write (4)
    "robust":     20,
    "regtiled":   16,  # 1 read of X (4) + gamma (4) + beta (4) + write (4)
}

# Estimated DRAM: accounts for SLC/L1 cache hits on repeated reads.
# Effective for large B: read X once (4B) + write Y (4B) = 8B.
ESTIMATED_DRAM_BYTES = {k: 8 for k in THEORETICAL_BYTES}

# Plot styles (marker + color) per kernel — merged with shared labels below
PLOT_STYLES = {
    "pytorch":    {"marker": "*", "color": "#f59e0b"},
    "naive":      {"marker": "v", "color": "#6b7280"},
    "naive_32":   {"marker": "v", "color": "#d97706"},
    "shared":     {"marker": "^", "color": "#8b5cf6"},
    "simd":       {"marker": "D", "color": "#ec4899"},
    "vectorized": {"marker": "o", "color": "#2563eb"},
    "fused":      {"marker": "s", "color": "#059669"},
    "robust":     {"marker": "P", "color": "#dc2626"},
    "regtiled":   {"marker": "X", "color": "#0891b2"},
}

# Merge shared labels with plot styles so KERNEL_META[k] has label+marker+color
KERNEL_META = {k: {**v, **PLOT_STYLES.get(k, {})} for k, v in _KERNEL_META.items()}

# Hidden-dim sweep configs (B, N)
N_SWEEP = [
    (4096, 768),    # GPT-2 small
    (4096, 1024),   # GPT-2 medium
    (4096, 2048),   # GPT-2 large / LLaMA-7B
    (4096, 4096),   # LLaMA-13B / larger models
]

assert torch.backends.mps.is_available(), "MPS backend not available"
device = torch.device("mps")


# ──────────────────────────────────────────────
# Benchmarking
# ──────────────────────────────────────────────

def bench_pytorch(B: int, N: int) -> float:
    """Benchmark PyTorch native LayerNorm. Returns average time in seconds."""
    x = torch.randn(B, N, device=device, dtype=DTYPE)
    gamma = torch.ones(N, device=device, dtype=DTYPE)
    beta = torch.zeros(N, device=device, dtype=DTYPE)

    for _ in range(WARMUP):
        _ = F.layer_norm(x, (N,), gamma, beta, EPS)
    torch.mps.synchronize()

    start = torch.mps.event.Event(enable_timing=True)
    end = torch.mps.event.Event(enable_timing=True)

    start.record()
    for _ in range(NUM_ITERS):
        _ = F.layer_norm(x, (N,), gamma, beta, EPS)
    end.record()
    torch.mps.synchronize()

    return start.elapsed_time(end) / NUM_ITERS / 1000.0


def bench_custom(B: int, N: int, kernel: str) -> float:
    """Benchmark a custom Metal kernel. Returns average time in seconds."""
    from layernorm_metal import layernorm_forward

    x = torch.randn(B, N, device=device, dtype=DTYPE)
    gamma = torch.ones(N, device=device, dtype=DTYPE)
    beta = torch.zeros(N, device=device, dtype=DTYPE)

    for _ in range(WARMUP):
        _ = layernorm_forward(x, gamma, beta, EPS, kernel=kernel)
    torch.mps.synchronize()

    start = torch.mps.event.Event(enable_timing=True)
    end = torch.mps.event.Event(enable_timing=True)

    start.record()
    for _ in range(NUM_ITERS):
        _ = layernorm_forward(x, gamma, beta, EPS, kernel=kernel)
    end.record()
    torch.mps.synchronize()

    return start.elapsed_time(end) / NUM_ITERS / 1000.0


def bench_kernel(B: int, N: int, kernel: str) -> float:
    """Dispatch to the right benchmark function. Returns avg seconds."""
    if kernel == "pytorch":
        return bench_pytorch(B, N)
    return bench_custom(B, N, kernel)


# ──────────────────────────────────────────────
# Main logic
# ──────────────────────────────────────────────

def run_sweep(kernels: list[str], batch_sizes: list[int],
              N: int) -> list[dict]:
    """Run selected kernels across batch sizes. Returns list of result dicts."""
    header = (f"  {'B':>6}  {'Kernel':<30s}  {'Time (ms)':>10}  "
              f"{'GFLOP/s':>10}  {'Theo BW':>10}  {'Est DRAM BW':>12}  "
              f"{'% Peak':>8}")
    print(header)
    print(f"  {'-' * 100}")

    results = []

    for B in batch_sizes:
        total_elems = B * N

        for kernel in kernels:
            avg_s = bench_kernel(B, N, kernel)
            avg_ms = avg_s * 1000.0
            total_flops = total_elems * FLOPS_PER_ELEM

            gflops = total_flops / avg_s / 1e9
            theo_bw = (total_elems * THEORETICAL_BYTES[kernel]
                       / avg_s / 1e9)
            dram_bw = (total_elems * ESTIMATED_DRAM_BYTES[kernel]
                       / avg_s / 1e9)
            pct_peak = dram_bw / PEAK_BW_GB_S * 100.0

            label = KERNEL_META[kernel]["label"]
            print(f"  {B:>6}  {label:<30s}  {avg_ms:>10.4f}  "
                  f"{gflops:>10.2f}  {theo_bw:>10.2f}  "
                  f"{dram_bw:>12.2f}  {pct_peak:>7.1f}%")

            results.append({
                "batch_size": B,
                "hidden_dim": N,
                "kernel": kernel,
                "time_ms": avg_ms,
                "gflops": gflops,
                "theoretical_bw_gbs": theo_bw,
                "estimated_dram_bw_gbs": dram_bw,
                "pct_peak_dram_bw": pct_peak,
            })

        if len(kernels) > 1:
            print(f"  {'-' * 100}")

    return results


def print_speedup_summary(results: list[dict], kernels: list[str]):
    """Print speedup summary for the last batch size in results."""
    if len(kernels) < 2:
        return

    # Group by batch size, summarize each
    batch_sizes = sorted(set(r["batch_size"] for r in results))
    for B in batch_sizes:
        br = {r["kernel"]: r for r in results if r["batch_size"] == B}

        # Find reference kernels
        pytorch_ms = br["pytorch"]["time_ms"] if "pytorch" in br else None
        naive_ms = br["naive"]["time_ms"] if "naive" in br else None

        parts = []
        if naive_ms is not None:
            parts.append("vs K1")
        if pytorch_ms is not None:
            parts.append("vs PyTorch")
        if not parts:
            continue

        print(f"\n  Speedup @ B={B}:")
        for kernel in kernels:
            if kernel not in br:
                continue
            label = KERNEL_META[kernel]["label"]
            ms = br[kernel]["time_ms"]
            cols = []
            if naive_ms is not None:
                cols.append(f"{naive_ms / ms:>6.2f}x vs K1")
            if pytorch_ms is not None:
                cols.append(f"{pytorch_ms / ms:>6.2f}x vs PyTorch")
            print(f"    {label:<30s}  {'  '.join(cols)}")


def save_csv(results: list[dict], path: str):
    """Write results to CSV."""
    ensure_parent_dir(path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {path}")


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_single(results: list[dict], kernels: list[str],
                show_dram_ceiling: bool, output_path: str):
    """Single-panel plot: GFLOP/s vs batch size for all kernels."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for kernel in kernels:
        kr = [r for r in results if r["kernel"] == kernel]
        bs = [r["batch_size"] for r in kr]
        gf = [r["gflops"] for r in kr]
        meta = KERNEL_META[kernel]
        ax.plot(bs, gf, marker=meta["marker"], color=meta["color"],
                linewidth=2, markersize=7, label=meta["label"], zorder=3)

    if show_dram_ceiling:
        min_bytes = min(THEORETICAL_BYTES[k] for k in kernels)
        ai = FLOPS_PER_ELEM / min_bytes
        ceiling = ai * PEAK_BW_GB_S
        ax.axhline(y=ceiling, color='#dc2626', linestyle='--',
                   linewidth=1.5,
                   label=f'DRAM-bound ceiling ({ceiling:.0f} GFLOP/s)',
                   zorder=2)

        y_max = ax.get_ylim()[1]
        all_gflops = [r["gflops"] for r in results]
        if max(all_gflops) > ceiling:
            ax.fill_between(
                [min(r["batch_size"] for r in results),
                 max(r["batch_size"] for r in results)],
                ceiling, max(max(all_gflops) * 1.15, y_max),
                alpha=0.06, color='#dc2626', zorder=0)

    N = results[0]["hidden_dim"]
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Batch Size (B)', fontsize=11)
    ax.set_ylabel('GFLOP/s', fontsize=11)
    ax.set_title(f'LayerNorm Throughput (N={N})', fontsize=12)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


def plot_two_panel(results: list[dict], kernels: list[str],
                   show_dram_ceiling: bool, output_path: str):
    """Two-panel plot: GFLOP/s (left) + bandwidth comparison (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left panel: GFLOP/s ──
    for kernel in kernels:
        kr = [r for r in results if r["kernel"] == kernel]
        bs = [r["batch_size"] for r in kr]
        gf = [r["gflops"] for r in kr]
        meta = KERNEL_META[kernel]
        ax1.plot(bs, gf, marker=meta["marker"], color=meta["color"],
                 linewidth=2, markersize=7, label=meta["label"], zorder=3)

    if show_dram_ceiling:
        min_bytes = min(THEORETICAL_BYTES[k] for k in kernels)
        ai = FLOPS_PER_ELEM / min_bytes
        ceiling = ai * PEAK_BW_GB_S
        ax1.axhline(y=ceiling, color='#dc2626', linestyle='--',
                    linewidth=1.5,
                    label=f'DRAM-bound ceiling ({ceiling:.0f} GFLOP/s)',
                    zorder=2)

    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Batch Size (B)', fontsize=11)
    ax1.set_ylabel('GFLOP/s', fontsize=11)
    ax1.set_title('Throughput', fontsize=12)
    ax1.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # ── Right panel: Bandwidth (theoretical vs estimated DRAM) ──
    for kernel in kernels:
        kr = [r for r in results if r["kernel"] == kernel]
        bs = [r["batch_size"] for r in kr]
        meta = KERNEL_META[kernel]

        theo = [r["theoretical_bw_gbs"] for r in kr]
        ax2.plot(bs, theo, marker=meta["marker"], color=meta["color"],
                 linewidth=2, markersize=6,
                 label=f'{meta["label"]} (theo)', zorder=3)

        dram = [r["estimated_dram_bw_gbs"] for r in kr]
        ax2.plot(bs, dram, marker=meta["marker"], color=meta["color"],
                 linewidth=1.5, markersize=5, linestyle='--', alpha=0.6,
                 label=f'{meta["label"]} (est DRAM)', zorder=2)

    ax2.axhline(y=PEAK_BW_GB_S, color='#dc2626', linestyle='--',
                linewidth=1.5,
                label=f'Peak DRAM BW ({PEAK_BW_GB_S:.0f} GB/s)', zorder=2)

    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Batch Size (B)', fontsize=11)
    ax2.set_ylabel('Memory Bandwidth (GB/s)', fontsize=11)
    ax2.set_title('Theoretical vs Estimated DRAM Bandwidth', fontsize=12)
    ax2.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
    ax2.legend(fontsize=7, loc='upper left', ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LayerNorm kernels on MPS with roofline plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Kernel names: pytorch, naive (k1), shared (k2), simd (k3),\n"
            "              vectorized (k4), fused (k5), robust (k6), "
            "regtiled (k7)\n"
            "              naive_32 (k1b)\n"
            "Use 'all' to select every kernel."
        ))
    parser.add_argument(
        "-k", "--kernels", nargs="+", default=["pytorch", "naive"],
        help="Kernels to benchmark (default: pytorch naive)")
    parser.add_argument(
        "-N", type=int, default=N_DEFAULT,
        help=f"Hidden dimension (default: {N_DEFAULT})")
    parser.add_argument(
        "--B", type=int, default=None,
        help="Single batch size (skips sweep)")
    parser.add_argument(
        "--sweep-N", action="store_true",
        help="Sweep across hidden dimensions at B=4096")
    parser.add_argument(
        "--two-panel", action="store_true",
        help="Show two-panel plot (throughput + bandwidth)")
    parser.add_argument(
        "--dram-ceiling", action="store_true",
        help="Show DRAM-bound ceiling line on throughput plot")
    parser.add_argument(
        "-o", "--output", type=str, default="figures/benchmark-roofline.png",
        help="Output plot filename (default: figures/benchmark-roofline.png)")
    parser.add_argument(
        "--csv", type=str, default="results/benchmark-results.csv",
        help="Output CSV filename (default: results/benchmark-results.csv)")
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation (console + CSV only)")
    args = parser.parse_args()

    kernels = resolve_kernels(args.kernels)
    N = args.N

    # Determine batch sizes
    if args.B is not None:
        batch_sizes = [args.B]
    else:
        batch_sizes = BATCH_SIZES

    print(f"LayerNorm Benchmark — N={N}, kernels: {kernels}")
    print(f"  Warmup: {WARMUP} iters, Measured: {NUM_ITERS} iters")
    print(f"  Arithmetic intensity (FLOP/byte):")
    print(f"    K1-K4: 8/24 = {8/24:.2f}  (3 reads of X)")
    print(f"    K5/K6: 8/20 = {8/20:.2f}  (2 reads of X)")
    print(f"    K7:    8/16 = {8/16:.2f}  (1 read of X)\n")

    all_results = []

    # Main sweep
    results = run_sweep(kernels, batch_sizes, N)
    all_results.extend(results)
    print_speedup_summary(results, kernels)

    # Optional hidden-dim sweep
    if args.sweep_N:
        print(f"\n{'='*60}")
        print(f"  Hidden dimension sweep")
        print(f"{'='*60}")
        for B_n, N_n in N_SWEEP:
            print(f"\n  --- B={B_n}, N={N_n} ---\n")
            nr = run_sweep(kernels, [B_n], N_n)
            all_results.extend(nr)
            print_speedup_summary(nr, kernels)

    # Save CSV
    save_csv(all_results, args.csv)

    # Plot (only for batch-size sweeps, not single-B or N-sweep)
    if not args.no_plot and args.B is None:
        # Use only the main sweep results for plotting
        plot_results = [r for r in all_results if r["hidden_dim"] == N]
        if args.two_panel:
            plot_two_panel(plot_results, kernels,
                           args.dram_ceiling, args.output)
        else:
            plot_single(plot_results, kernels,
                        args.dram_ceiling, args.output)


if __name__ == "__main__":
    main()
