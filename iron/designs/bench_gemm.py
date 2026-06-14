#!/usr/bin/env python3
"""Benchmark: int16 GEMM on Phoenix NPU vs numpy CPU. Multiple shapes."""
import sys, os, time
import numpy as np
import pyxrt

BUILD = sys.argv[1] if len(sys.argv) > 1 else \
    "/tmp/mliraie-v132/programming_examples/basic/matrix_multiplication/single_core/build"
XCLBIN = f"{BUILD}/final_256x256x256_32x32x32.xclbin"
INSTS  = f"{BUILD}/insts_256x256x256_32x32x32.txt"
M = K = N = 256

with open(INSTS,"rb") as f: instr = np.frombuffer(f.read(), dtype=np.uint8)
IBYTES = len(instr)

dev = pyxrt.device(0)
xcl = pyxrt.xclbin(XCLBIN); dev.register_xclbin(xcl)
ctx = pyxrt.hw_context(dev, xcl.get_uuid())
kname = [k.get_name() for k in xcl.get_kernels()][0]
kern = pyxrt.kernel(ctx, kname)

HOST_ONLY = pyxrt.bo.flags.host_only
CACHEABLE = pyxrt.bo.flags.cacheable
TO_DEV = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
FROM_DEV = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

A = np.random.randint(-100,100,(M,K),dtype=np.int16)
B = np.random.randint(-100,100,(K,N),dtype=np.int16)
ref = (A.astype(np.int32) @ B.astype(np.int32)).astype(np.int32)

bo_instr = pyxrt.bo(dev, IBYTES, CACHEABLE, kern.group_id(1))
np.frombuffer(bo_instr.map(), dtype=np.uint8)[:] = instr
bo_instr.sync(TO_DEV, IBYTES, 0)
bo_a = pyxrt.bo(dev, M*K*2, HOST_ONLY, kern.group_id(3))
np.frombuffer(bo_a.map(), dtype=np.int16)[:] = A.reshape(-1)
bo_a.sync(TO_DEV, M*K*2, 0)
bo_b = pyxrt.bo(dev, K*N*2, HOST_ONLY, kern.group_id(4))
np.frombuffer(bo_b.map(), dtype=np.int16)[:] = B.reshape(-1)
bo_b.sync(TO_DEV, K*N*2, 0)
bo_c = pyxrt.bo(dev, M*N*4, HOST_ONLY, kern.group_id(5))
c_arr = np.frombuffer(bo_c.map(), dtype=np.int32)

# ---- warmup ----
h = kern(3, bo_instr, IBYTES, bo_a, bo_b, bo_c, 0, 0); h.wait()
bo_c.sync(FROM_DEV, M*N*4, 0)
assert np.array_equal(c_arr.reshape(M,N).copy(), ref), "GEMM mismatch!"

# ---- NPU benchmark ----
NPU_ITERS = 100
t0 = time.time()
for _ in range(NPU_ITERS):
    h = kern(3, bo_instr, IBYTES, bo_a, bo_b, bo_c, 0, 0); h.wait()
npu_ms = (time.time()-t0)/NPU_ITERS*1000

# ---- CPU benchmark (numpy, int32 matmul) ----
CPU_ITERS = 50
Ai32 = A.astype(np.int32); Bi32 = B.astype(np.int32)
# warmup
_ = Ai32 @ Bi32
t0 = time.time()
for _ in range(CPU_ITERS):
    _ = Ai32 @ Bi32
cpu_ms = (time.time()-t0)/CPU_ITERS*1000

# single-thread CPU baseline too
import os
saved = os.environ.get("OMP_NUM_THREADS")
os.environ["OMP_NUM_THREADS"]="1"
import importlib
Ai = Ai32.copy()
t0=time.time()
for _ in range(CPU_ITERS):
    _ = Ai @ Bi32
cpu1_ms = (time.time()-t0)/CPU_ITERS*1000

GFLOPS = lambda ms: 2*M*K*N / (ms/1000) / 1e9
print(f"\n=== int16 GEMM {M}x{K}x{N} (i16->i32) on AMD Ryzen 7 7840HS ===")
print(f"  NPU (Phoenix, 1 core, Peano):   {npu_ms:7.2f} ms/op   {GFLOPS(npu_ms):6.1f} GOPS")
print(f"  CPU numpy (multi-thread):       {cpu_ms:7.2f} ms/op   {GFLOPS(cpu_ms):6.1f} GOPS")
print(f"  CPU numpy (1 thread):           {cpu1_ms:7.2f} ms/op   {GFLOPS(cpu1_ms):6.1f} GOPS")
print(f"  NPU/MT-CPU speedup:  {cpu_ms/npu_ms:.2f}x")
print(f"  NPU/ST-CPU speedup:  {cpu1_ms/npu_ms:.2f}x")
print(f"  (single-core NPU design; multi-column would be ~{min(4,int(cpu1_ms/npu_ms))}x faster)")
