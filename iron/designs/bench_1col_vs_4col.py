"""Compare 1-column vs 4-column int16 GEMM on the Phoenix NPU.

(A) of the roadmap: prove multi-column works + quantify the speedup.
Uses npu_kernel.NpuKernel — the serving-layer execution primitive.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from npu_kernel import NpuKernel, bench_gemm
import numpy as np

BUILD = "/tmp/mliraie-v132/programming_examples/basic/matrix_multiplication"

print("=" * 70)
print(" int16 GEMM on Phoenix NPU: single-column vs whole-array (4-col)")
print("=" * 70)

# --- single core (built earlier): 256x256x256 ---
print("\n[1-col] 256x256x256, tile 32x32x32, i16->i32, 1 core")
k1 = NpuKernel.from_build(f"{BUILD}/single_core/build", "256x256x256_32x32x32")
ms1, g1 = bench_gemm(k1, 256, 256, 256, iters=100)
print(f"   {ms1:7.2f} ms/op   {g1:6.1f} GOPS   (verified exact vs numpy)")

# --- whole array (4-col): 512x512x512 ---
print("\n[4-col] 512x512x512, tile 64x64x32, i16->i32, 16 cores (4 cols x 4 rows)")
k4 = NpuKernel.from_build(f"{BUILD}/whole_array/build", "512x512x512_64x64x32_4c")
ms4, g4 = bench_gemm(k4, 512, 512, 512, iters=100)
print(f"   {ms4:7.2f} ms/op   {g4:6.1f} GOPS   (verified exact vs numpy)")

# --- normalize: per-FLOP efficiency (different shapes, so compare GOPS) ---
print("\n" + "-" * 70)
print(" THROUGHPUT (GOPS, apples-to-apples):")
print(f"   1-col: {g1:6.1f} GOPS")
print(f"   4-col: {g4:6.1f} GOPS   ({g4/g1:.1f}x the single-core throughput)")
print("-" * 70)

# also a same-shape comparison so speedup isn't confounded by shape
print("\n[same-shape check] 512x512x512 on BOTH 1-col and 4-col would need a")
print("   1-col build at 512; the GOPS comparison above is the fair metric.")
