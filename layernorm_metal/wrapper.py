import torch
import pkg_resources
from functools import lru_cache

# Import the compiled C++ extension (.so lives inside the package as _C)
from . import _C

# Kernel variants: maps short name → (shader filename, Metal function name)
KERNELS = {
    "naive":      ("layernorm_naive.metal",      "layernorm_naive"),
    "naive_1024": ("layernorm_naive.metal",      "layernorm_naive_1024"),
    "shared":     ("layernorm_shared.metal",     "layernorm_shared"),
    "simd":       ("layernorm_simd.metal",       "layernorm_simd"),
    "vectorized": ("layernorm_vectorized.metal", "layernorm_vectorized"),
    "fused":      ("layernorm_fused.metal",      "layernorm_fused"),
    "robust":     ("layernorm_robust.metal",     "layernorm_robust"),
    "regtiled":   ("layernorm_regtiled.metal",   "layernorm_regtiled"),
}


@lru_cache(maxsize=None)
def _shader_path(filename: str) -> str:
    return pkg_resources.resource_filename(
        'layernorm_metal', f'kernels/{filename}'
    )


def layernorm_forward(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
    kernel: str = "vectorized",
) -> torch.Tensor:
    """Run LayerNorm forward pass using a custom Metal kernel.

    Args:
        input: [B, N] tensor on MPS device.
        gamma: [N] scale parameter on MPS device.
        beta:  [N] bias parameter on MPS device.
        eps:   Small constant for numerical stability.
        kernel: Which kernel variant to use. One of:
            "naive"      — K1: one thread per row, 3 serial passes.
            "shared"     — K2: threadgroup tree reduction, coalesced access.
            "simd"       — K3: simd_sum reduction, minimal barriers.
            "vectorized" — K4: float4 loads + simd reduction (default).
            "fused"      — K5: single-pass fused stats + float4 + SIMD.
            "robust"     — K6: fused + tail handling + precise rsqrt.
            "regtiled"   — K7: register-tiled, single device memory read.

    Returns:
        Normalized output tensor with same shape as input.
    """
    if kernel not in KERNELS:
        raise ValueError(
            f"Unknown kernel '{kernel}'. "
            f"Choose from: {list(KERNELS.keys())}"
        )
    filename, kernel_name = KERNELS[kernel]
    return _C.layernorm_forward(
        input, gamma, beta, eps, _shader_path(filename), kernel_name
    )


def start_gpu_capture(output_path: str = "/tmp/layernorm.gputrace") -> None:
    """Start a Metal GPU trace capture.

    Requires METAL_CAPTURE_ENABLED=1 environment variable to be set
    before launching Python. The resulting .gputrace file can be opened
    in Xcode for shader-level profiling.
    """
    _C.start_gpu_capture(output_path)


def stop_gpu_capture() -> None:
    """Stop the active Metal GPU trace capture."""
    _C.stop_gpu_capture()
