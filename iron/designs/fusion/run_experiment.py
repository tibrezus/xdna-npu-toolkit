"""run_fusion_experiment.py — the decisive op-fusion test on Phoenix NPU1.

Hypothesis: a fused multi-op design (intermediate data on-device) collapses the
per-op host<->NPU round-trip tax, so ONE fused dispatch of 2 ops costs ~the same
as ONE dispatch of 1 op (not 2x).

We compare:
  (FUSED)   D = (A + B) + C   — one dispatch, intermediate T on-device
  (SEP)     T = A + B ; D = T + C — two separate single-add dispatches, T via host

Decisive ratio:  time_fused / time_single_add
  ~1.0  => fusion WORKS (2nd op is free)  => single-query NPU serving is reachable
  ~2.0  => fusion FAILS (each op pays round-trip tax) => commit to batched-only
"""
import sys, os, time
sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
from npu_kernel import NpuKernel
import numpy as np
from ml_dtypes import bfloat16

FUSED_BUILD = "/tmp/iron/fusion/build"
SINGLE_BUILD = "/tmp/iron/fusion/build_single"
NUMEL = 4096


def make_inputs():
    # small-magnitude bf16 inputs so the adds don't overflow/lose precision
    A = (np.random.randn(NUMEL) * 3).astype(bfloat16)
    B = (np.random.randn(NUMEL) * 3).astype(bfloat16)
    C = (np.random.randn(NUMEL) * 3).astype(bfloat16)
    return A, B, C


def run_and_time(kern, tensors, out_sizes, iters=200):
    # warmup
    for _ in range(10):
        kern.run(*tensors, out_sizes=out_sizes, out_dtype=bfloat16)
    t0 = time.time()
    for _ in range(iters):
        kern.run(*tensors, out_sizes=out_sizes, out_dtype=bfloat16)
    return (time.time() - t0) / iters * 1000  # ms/op


print("=" * 68)
print(" OP-FUSION EXPERIMENT  —  does on-device chaining collapse the round-trip tax?")
print("=" * 68)

# --- FUSED: D = (A+B)+C, one dispatch ---
print("\n[FUSED]  D = (A+B)+C   (1 dispatch, intermediate T on-device)")
kf = NpuKernel(f"{FUSED_BUILD}/final.xclbin", f"{FUSED_BUILD}/insts.bin")
A, B, C = make_inputs()
D = kf.run(A, B, C, out_sizes=[NUMEL * 2], out_dtype=bfloat16)[0]   # bf16 = 2 bytes
ref_fused = (A.astype(np.float32) + B.astype(np.float32) + C.astype(np.float32))
ok_fused = np.allclose(D.astype(np.float32), ref_fused, atol=0.5)
print(f"   correctness: {'PASS' if ok_fused else 'FAIL'}  (D vs A+B+C, bf16)")
t_fused = run_and_time(kf, [A, B, C], [NUMEL * 2])
print(f"   time: {t_fused:.3f} ms/dispatch")

# --- SINGLE ADD baseline: C = A+B, one dispatch ---
print("\n[SINGLE] T = A+B   (1 dispatch, baseline)")
ks = NpuKernel(f"{SINGLE_BUILD}/final.xclbin", f"{SINGLE_BUILD}/insts.bin")
T = ks.run(A, B, out_sizes=[NUMEL * 2], out_dtype=bfloat16)[0]
ok_single = np.allclose(T.astype(np.float32), A.astype(np.float32) + B.astype(np.float32), atol=0.5)
print(f"   correctness: {'PASS' if ok_single else 'FAIL'}  (T vs A+B, bf16)")
t_single = run_and_time(ks, [A, B], [NUMEL * 2])
print(f"   time: {t_single:.3f} ms/dispatch")

# --- SEPARATE: two dispatches, intermediate via host ---
print("\n[SEP]    D = (A+B)+C  (2 separate dispatches, T via host)")
# warmup
for _ in range(10):
    T = ks.run(A, B, out_sizes=[NUMEL * 2], out_dtype=bfloat16)[0]
    D = ks.run(T, C, out_sizes=[NUMEL * 2], out_dtype=bfloat16)[0]
iters = 200
t0 = time.time()
for _ in range(iters):
    T = ks.run(A, B, out_sizes=[NUMEL * 2], out_dtype=bfloat16)[0]
    D = ks.run(T, C, out_sizes=[NUMEL * 2], out_dtype=bfloat16)[0]
t_sep = (time.time() - t0) / iters * 1000
print(f"   time: {t_sep:.3f} ms  (2 dispatches)")

# --- THE VERDICT ---
print("\n" + "=" * 68)
print(" VERDICT")
print("=" * 68)
ratio = t_fused / t_single
print(f"  time_single_add   = {t_single:.3f} ms/dispatch  (1 op)")
print(f"  time_fused        = {t_fused:.3f} ms/dispatch  (2 ops, on-device chain)")
print(f"  time_separate_2x  = {t_sep:.3f} ms            (2 ops, 2 dispatches)")
print()
print(f"  fused / single    = {ratio:.2f}")
print(f"  separate / fused  = {t_sep/t_fused:.2f}x  (speedup from fusing 2 ops into 1 dispatch)")
print()
if ratio < 1.5:
    print("  => FUSION WORKS. The 2nd op is essentially free (on-device, no round-trip).")
    print("     Single-query NPU serving is REACHABLE via op fusion.")
elif ratio < 2.5:
    print("  => FUSION PARTIAL. Some tax remains but <2x; fusion helps but not fully.")
else:
    print("  => FUSION DOES NOT COLLAPSE THE TAX. Each op pays round-trip.")
    print("     Commit to batched-only serving; route single-query to CPU/iGPU.")
