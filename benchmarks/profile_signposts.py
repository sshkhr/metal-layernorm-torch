"""Profile LayerNorm with OS Signposts for Apple Instruments.

Emits PyTorchMPS signposts that Instruments captures.  Run this script,
note the PID it prints, then attach Instruments (Metal System Trace
template + os_signpost instrument) to that PID.

Usage:
    python benchmarks/profile_signposts.py

Instruments workflow:
    1. Run this script — it prints its PID and pauses.
    2. Open Instruments → Metal System Trace template.
    3. Attach to Process → select the python3 PID.
    4. Add the os_signpost instrument (click +, search "Logging").
    5. Click Record, then press Enter in the terminal to start the workload.
    6. After the script finishes, stop recording in Instruments.
    7. Expand the os_signpost track → look for the PyTorchMPS subsystem.
"""

import os
import torch

# ----- Config ----------------------------------------------------------------
B, N = 32, 768
NUM_ITERS = 100

# ----- Setup -----------------------------------------------------------------
device = torch.device("mps")
model = torch.nn.LayerNorm(N).to(device)
x = torch.randn(B, 128, N, device=device)

# Warm-up
_ = model(x)
torch.mps.synchronize()

# ----- Wait for Instruments attachment ---------------------------------------
print(f"PID: {os.getpid()}")
input("Attach Instruments to this PID, then press Enter to start profiling...")

# ----- Profile with signposts ------------------------------------------------
# mode="interval" emits begin/end markers for each op duration.
# wait_until_completed=True serializes GPU work for clean timelines.
torch.mps.profiler.start(mode="interval", wait_until_completed=True)

for _ in range(NUM_ITERS):
    output = model(x)

torch.mps.synchronize()
torch.mps.profiler.stop()

print("Signpost profiling complete. Check Instruments for results.")
