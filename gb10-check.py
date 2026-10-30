#!/usr/bin/env python3
"""Compact GB10 low-power-state check: 8192^3 bf16 burn + median SM clock/power.
Read-only. Run inside the vLLM container (needs torch + nvidia-smi)."""
import subprocess, threading, time

def smi():
    q = "clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
    if not out:
        return None
    f = [x.strip() for x in out[0].split(",")]
    num = lambda s: float(s) if s.replace(".", "", 1).isdigit() else None
    return num(f[0]), num(f[1]), num(f[2]), f[3]

import torch
n = 8192
x = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
y = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
for _ in range(10):
    torch.mm(x, y)
torch.cuda.synchronize()

samples, stop = [], threading.Event()
def sampler():
    time.sleep(3)
    while not stop.is_set():
        s = smi()
        if s:
            samples.append(s)
        stop.wait(2)
th = threading.Thread(target=sampler, daemon=True); th.start()

t0, iters = time.time(), 0
while time.time() - t0 < 45:
    for _ in range(20):
        torch.mm(x, y); torch.cuda.synchronize()
    iters += 20
dt = time.time() - t0
stop.set(); th.join(timeout=5)

tflops = 2 * n ** 3 * iters / dt / 1e12
clk = sorted(s[0] for s in samples if s[0] is not None)
pw = sorted(s[1] for s in samples if s[1] is not None)
tp = sorted(s[2] for s in samples if s[2] is not None)
med = lambda v: v[len(v) // 2] if v else None
print(f"RESULT tflops={tflops:.1f} sm_median={med(clk):.0f}MHz power_median={med(pw):.1f}W "
      f"temp_max={max(tp) if tp else 0:.0f}C throttle={sorted({s[3] for s in samples})} "
      f"iters={iters} in {dt:.1f}s")
print("VERDICT:", "STUCK low-power state" if tflops < 60 and (med(clk) or 9999) < 1400
      else ("OK (lower-power chassis range)" if tflops < 80 else "HEALTHY"))
