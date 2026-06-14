"""(C) Matrix-vector GEMV on Phoenix NPU — the inference op (batch=1).

A[M,K] int16 × b[K] int16 -> c[M] int32. This is exactly what a transformer
linear layer does at inference time: weight matrix × activation vector.

Tests that NpuKernel.run() generalizes to the matvec kernel layout (A, b in; c out).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from npu_kernel import NpuKernel
import numpy as np

BUILD = "/tmp/mliraie-v132/programming_examples/basic/matrix_multiplication/matrix_vector/build"
M = K = 288

print(f"===== matrix-vector (GEMV) {M}x{K} on Phoenix NPU =====")
kern = NpuKernel.from_build(BUILD, "288x288x1")

A = np.random.randint(-100, 100, (M, K), dtype=np.int16)
b = np.random.randint(-100, 100, (K,), dtype=np.int16)
ref = (A.astype(np.int32) @ b.astype(np.int32)).astype(np.int32)   # [M]

# A is arg 3, b is arg 4, c is arg 5. c size = M * 4 bytes.
out = kern.run(A, b, out_sizes=[M * 4], out_dtype=np.int32)[0]
if np.array_equal(out, ref):
    print(f"PASS!  GEMV {M}x{K} on Phoenix NPU == numpy exactly.")
else:
    d = np.abs(out.astype(np.int64) - ref.astype(np.int64))
    print(f"MISMATCH: max|diff|={d.max()}  match={(d==0).sum()}/{M}")
    print(f"  out[:4]={out[:4]}\n  ref[:4]={ref[:4]}")
    sys.exit(1)

# benchmark
NITERS = 200
t0 = time.time()
for _ in range(NITERS):
    kern.run(A, b, out_sizes=[M * 4], out_dtype=np.int32)
ms = (time.time() - t0) / NITERS * 1000
gops = 2 * M * K / (ms / 1000) / 1e9
print(f"   {ms:.3f} ms/op   {gops:.1f} GOPS   (single core, batch=1)")

# CPU reference
t0 = time.time()
for _ in range(50):
    _ = A.astype(np.int32) @ b.astype(np.int32)
cpu_ms = (time.time() - t0) / 50 * 1000
print(f"   CPU numpy: {cpu_ms:.3f} ms/op   ({ms/cpu_ms:.1f}x NPU)")
print()
print("NOTE: batch=1 matvec is host<->NPU round-trip bound. The whole-array")
print("matrix-matrix GEMM (264 GOPS) is the throughput path for batched serving.")
