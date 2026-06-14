"""profile_dispatch.py — break down where NPU dispatch time goes.

Times a single GEMM at increasing M to separate:
  - fixed per-dispatch overhead (BO alloc/map/sync, kernel launch)
  - M-proportional cost (compute + transfer)

If the per-dispatch overhead is large and flat, BO pooling is the win.
"""
import sys, time
sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
import numpy as np
from npu_kernel import NpuKernel
import pyxrt

k = NpuKernel("/tmp/iron/minilm-gemms/qkv-4col.xclbin", "/tmp/iron/minilm-gemms/qkv-4col.insts.txt")

print("=== single GEMM (512x384x384 4-col) timing breakdown ===\n")

def time_full_run(M, iters=50):
    A = np.random.randint(-100,100,(M,384),np.int16)
    B = np.random.randint(-100,100,(384,384),np.int16)
    out_sz = M*384*4
    k.run(A,B,out_sizes=[out_sz])  # warmup
    t0=time.time()
    for _ in range(iters): k.run(A,B,out_sizes=[out_sz])
    return (time.time()-t0)/iters*1000

def time_bo_alloc_only(iters=200):
    """Just the BO allocation + map + sync, no kernel run."""
    dev = k._dev; kern = k._kern
    t = np.random.randint(-100,100,(512,384),np.int16)
    HO = pyxrt.bo.flags.host_only
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    t0=time.time()
    for _ in range(iters):
        bo = pyxrt.bo(dev, t.nbytes, HO, kern.group_id(3))
        np.frombuffer(bo.map(), dtype=t.dtype)[:] = t.reshape(-1)
        bo.sync(TO, t.nbytes, 0)
    return (time.time()-t0)/iters*1000

def time_reuse_bo(iters=200):
    """Same data movement but BO allocated ONCE and reused."""
    dev = k._dev; kern = k._kern
    t = np.random.randint(-100,100,(512,384),np.int16)
    HO = pyxrt.bo.flags.host_only
    TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    bo = pyxrt.bo(dev, t.nbytes, HO, kern.group_id(3))
    arr = np.frombuffer(bo.map(), dtype=t.dtype)
    t0=time.time()
    for _ in range(iters):
        arr[:] = t.reshape(-1)
        bo.sync(TO, t.nbytes, 0)
    return (time.time()-t0)/iters*1000

alloc_ms = time_bo_alloc_only()
reuse_ms = time_reuse_bo()
print(f"  BO alloc+map+sync (per input, fresh):  {alloc_ms:.3f} ms")
print(f"  BO copy+sync (per input, reused):      {reuse_ms:.3f} ms")
print(f"  -> BO pooling saves {alloc_ms-reuse_ms:.3f} ms/input/dispatch (3 inputs = {(alloc_ms-reuse_ms)*3:.2f} ms)\n")

for M in [512, 1024, 2048, 4096]:
    ms = time_full_run(M)
    gops = 2*M*384*384/(ms/1000)/1e9
    print(f"  M={M:5} (batch {M//64:3}): {ms:6.2f} ms/GEMM  ({gops:6.1f} GOPS)")

print("\n=== cost model ===")
import numpy as np
t512 = time_full_run(512); t4096 = time_full_run(4096)
compute_per_M = (t4096 - t512) / (4096-512)
fixed = t512 - compute_per_M*512
print(f"  fixed overhead per dispatch ~ {fixed:.2f} ms")
print(f"  compute ~ {compute_per_M*1000:.3f} us per row (M)")
print(f"  => at 36 dispatches: {fixed*36:.0f} ms fixed + compute")
print(f"  => with BO reuse, fixed could drop ~{alloc_ms*3:.1f} ms -> ~{max(0,fixed-alloc_ms*3)*36:.0f} ms")
