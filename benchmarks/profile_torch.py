"""Profile LayerNorm using torch.profiler with CPU activity tracing.

Captures CPU-side dispatch timing for MPS operations and exports a
Chrome trace JSON that can be visualized at https://ui.perfetto.dev/

Note: ProfilerActivity.MPS does not exist in PyTorch (as of 2.10).
CPU-only profiling still captures MPS op dispatch overhead — useful for
identifying whether the CPU dispatch or the GPU kernel is the bottleneck.
For GPU-side timing, use bench_events.py (MPS events) or gpu_capture.py
(Xcode GPU trace).

Usage:
    python benchmarks/profile_torch.py
"""

import torch
from torch.profiler import profile, record_function, ProfilerActivity

# ----- Config ----------------------------------------------------------------
B, N = 32, 768
NUM_ITERS = 100
TRACE_FILE = "layernorm_mps_trace.json"

# ----- Setup -----------------------------------------------------------------
device = torch.device("mps")

# PyTorch built-in LayerNorm
model = torch.nn.LayerNorm(N).to(device)
x = torch.randn(B, 128, N, device=device)

# Warm-up (first call compiles shaders)
_ = model(x)
torch.mps.synchronize()

# ----- Profile ---------------------------------------------------------------
with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with record_function("layernorm_bench"):
        for _ in range(NUM_ITERS):
            output = model(x)
    torch.mps.synchronize()

# Sort by CPU time (shows MPS dispatch overhead per op)
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))

# Export Chrome trace for visual timeline
prof.export_chrome_trace(TRACE_FILE)
print(f"\nChrome trace exported to {TRACE_FILE}")
print("Open at https://ui.perfetto.dev/ for a visual timeline.")
