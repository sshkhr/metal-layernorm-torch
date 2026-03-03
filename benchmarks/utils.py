"""Shared constants and utilities for LayerNorm benchmark scripts.

Centralises kernel metadata, aliases, resolution logic, and common
constants used by bench_mps_events.py, profile_torch.py, and gpu_capture.py.
"""

from __future__ import annotations

import os
import torch

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
N_DEFAULT = 768              # hidden dimension (GPT-2)
EPS = 1e-5
DTYPE = torch.float32

# ──────────────────────────────────────────────
# Kernel metadata
# ──────────────────────────────────────────────

# Display labels for every kernel (base metadata).
# Scripts that need extra fields (e.g. plot marker/color) can layer them on.
KERNEL_META = {
    "pytorch":    {"label": "PyTorch nn.LayerNorm"},
    "naive":      {"label": "K1: Naive (tg=256)"},
    "naive_1024": {"label": "K1: Naive (tg=1024)"},
    "shared":     {"label": "K2: Threadgroup reduction"},
    "simd":       {"label": "K3: SIMD reduction"},
    "vectorized": {"label": "K4: Vectorized float4"},
    "fused":      {"label": "K5: Fused single-pass"},
    "robust":     {"label": "K6: Robust (tail+precise)"},
    "regtiled":   {"label": "K7: Register-tiled"},
}

# Canonical ordering for consistent display
KERNEL_ORDER = ["pytorch", "naive", "naive_1024", "shared", "simd",
                "vectorized", "fused", "robust", "regtiled"]

# Aliases for convenience (e.g. -k k1 k4 k7)
ALIASES = {
    "k1": "naive", "k1b": "naive_1024",
    "k2": "shared", "k3": "simd", "k4": "vectorized",
    "k5": "fused", "k6": "robust", "k7": "regtiled",
}


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def resolve_kernels(names: list[str],
                    valid_kernels: list[str] | None = None) -> list[str]:
    """Resolve kernel names/aliases to a canonical ordered list.

    Parameters
    ----------
    names : list[str]
        Raw kernel names or aliases from the CLI (e.g. ["k4", "k7", "all"]).
    valid_kernels : list[str] | None
        If provided, overrides KERNEL_ORDER as both the set of accepted
        kernels and the ordering of the result.  Useful for scripts that
        should not accept "pytorch" (e.g. gpu_capture.py).
    """
    order = valid_kernels if valid_kernels is not None else KERNEL_ORDER
    valid = set(order)

    if "all" in names:
        return list(order)

    resolved = []
    for name in names:
        canonical = ALIASES.get(name.lower(), name.lower())
        if canonical not in valid:
            raise ValueError(
                f"Unknown kernel '{name}'. "
                f"Choose from: {list(valid)} "
                f"or aliases {list(ALIASES.keys())} or 'all'")
        if canonical not in resolved:
            resolved.append(canonical)
    # Sort by canonical order
    return [k for k in order if k in resolved]


def output_path_for_kernel(template: str, kernel: str) -> str:
    """Generate output path, substituting {kernel} if present."""
    if "{kernel}" in template:
        return template.replace("{kernel}", kernel)
    return template


def ensure_parent_dir(path: str) -> str:
    """Create parent directories for *path* if they don't exist. Returns path unchanged."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path
