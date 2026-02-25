"""Plot a roofline model for Apple Silicon and overlay LayerNorm kernels.

Generates a log-log roofline chart showing the memory-bandwidth and
compute ceilings for a given Apple Silicon chip, with measured kernel
performance overlaid.  If no measured data is provided, example points
from the PLAN.md are used.

Usage:
    python benchmarks/roofline.py                         # default M4 Pro
    python benchmarks/roofline.py --chip m1               # pick a chip
    python benchmarks/roofline.py -o roofline.png         # save to file
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


# Apple Silicon GPU specs: (FP32 TFLOPS, Memory BW GB/s)
CHIPS = {
    "m1":       {"gflops": 2600,  "bw": 68.25, "label": "Apple M1 (8-core GPU)"},
    "m1_max":   {"gflops": 10600, "bw": 400,   "label": "Apple M1 Max (32-core GPU)"},
    "m2":       {"gflops": 3600,  "bw": 100,   "label": "Apple M2 (10-core GPU)"},
    "m2_ultra": {"gflops": 27200, "bw": 800,   "label": "Apple M2 Ultra (76-core GPU)"},
    "m3":       {"gflops": 3500,  "bw": 100,   "label": "Apple M3 (10-core GPU)"},
    "m3_max":   {"gflops": 14100, "bw": 400,   "label": "Apple M3 Max (40-core GPU)"},
    "m4":       {"gflops": 3800,  "bw": 120,   "label": "Apple M4 (10-core GPU)"},
    "m4_pro":   {"gflops": 7500,  "bw": 273,   "label": "Apple M4 Pro (20-core GPU)"},
    "m4_max":   {"gflops": 15050, "bw": 546,   "label": "Apple M4 Max (40-core GPU)"},
}


def plot_roofline(peak_gflops, peak_bw_gbs, kernels=None,
                  title="Roofline Model"):
    """Plot a roofline model and optionally overlay kernel measurements.

    Args:
        peak_gflops: Peak FP32 GFLOPS of the chip.
        peak_bw_gbs: Peak memory bandwidth in GB/s.
        kernels: List of dicts with keys 'name', 'ai' (FLOP/byte),
                 'gflops' (achieved).
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    ai = np.logspace(-2, 4, 1000)
    ridge = peak_gflops / peak_bw_gbs
    roofline = np.minimum(peak_bw_gbs * ai, peak_gflops)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.loglog(ai, roofline, 'b-', lw=2.5,
              label=f'Roofline ({peak_gflops:.0f} GFLOPS, '
                    f'{peak_bw_gbs:.0f} GB/s)')
    ax.axvline(ridge, color='blue', ls=':', alpha=0.4)
    ax.annotate(f'Ridge: {ridge:.1f} FLOP/B',
                xy=(ridge, peak_gflops), fontsize=9, color='blue')

    if kernels:
        for k in kernels:
            roof_at_ai = min(peak_bw_gbs * k['ai'], peak_gflops)
            eff = k['gflops'] / roof_at_ai * 100
            ax.plot(k['ai'], k['gflops'], 'o', ms=12, mec='black',
                    label=f"{k['name']} ({eff:.0f}% eff)")
            ax.plot([k['ai']] * 2, [k['gflops'], roof_at_ai],
                    ':', lw=1)

    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)')
    ax.set_ylabel('Performance (GFLOPS)')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.set_xlim(0.01, 10000)
    ax.set_ylim(1, peak_gflops * 2)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot a roofline model for Apple Silicon.")
    parser.add_argument(
        "--chip", choices=list(CHIPS.keys()), default="m4_pro",
        help="Apple Silicon chip variant (default: m4_pro)")
    parser.add_argument(
        "-o", "--output", default="roofline.png",
        help="Output file path (default: roofline.png)")
    parser.add_argument(
        "--show", action="store_true",
        help="Show the plot interactively instead of just saving")
    args = parser.parse_args()

    chip = CHIPS[args.chip]
    peak_gflops = chip["gflops"]
    peak_bw = chip["bw"]

    # Example kernel data points — replace with measured values
    kernels = [
        {
            'name': 'LayerNorm (naive)',
            'ai': 0.4,
            'gflops': 0.4 * peak_bw * 0.60,
        },
        {
            'name': 'LayerNorm (optimized)',
            'ai': 0.4,
            'gflops': 0.4 * peak_bw * 0.85,
        },
    ]

    fig = plot_roofline(peak_gflops, peak_bw, kernels,
                        title=chip["label"])
    fig.savefig(args.output, dpi=150)
    print(f"Roofline plot saved to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
