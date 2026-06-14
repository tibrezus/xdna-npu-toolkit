"""measure_bandwidth.py — size the fusion prize precisely.

How long does it take to move the [M,1536] intermediate host->NPU and back?
The fusion win = eliminating 2x that transfer per layer (×6 layers).
"""
import sys, time, numpy as np
sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
import pyxrt
from ml_dtypes import bfloat16

dev = pyxrt.device(0)
HO = pyxrt.bo.flags.host_only
TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
FR = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

M = 4096
for kbytes, kname in [(1536, "intermediate [M,1536]"), (384, "activation [M,384]"), (1152, "qkv [M,1152]")]:
    sz = M * kbytes * 2  # bf16
    bo = pyxrt.bo(dev, sz, HO, 0)   # dummy group id 0
    arr = np.frombuffer(bo.map(), bfloat16)
    data = np.random.randn(M, kbytes).astype(bfloat16).reshape(-1)
    # warmup
    for _ in range(20):
        arr[:] = data; bo.sync(TO, sz, 0); bo.sync(FR, sz, 0)
    # H->D
    t0=time.time()
    for _ in range(100):
        arr[:] = data; bo.sync(TO, sz, 0)
    h2d = (time.time()-t0)/100*1000
    # D->H
    t0=time.time()
    for _ in range(100):
        bo.sync(FR, sz, 0)
    d2h = (time.time()-t0)/100*1000
    gbps_h2d = sz/h2d/1e6; gbps_d2h = sz/d2h/1e6
    print(f"  {kname}: {sz/1e6:.1f}MB  H->D {h2d:.3f}ms ({gbps_h2d:.1f}GB/s)  D->H {d2h:.3f}ms ({gbps_d2h:.1f}GB/s)")

print(f"\n=== fusion prize (FFN1->gelu->FFN2 keeps [M,1536] on-chip) ===")
# re-measure intermediate precisely
sz = M*1536*2
bo = pyxrt.bo(dev, sz, HO, 0); arr = np.frombuffer(bo.map(), bfloat16)
data = np.random.randn(M,1536).astype(bfloat16).reshape(-1)
for _ in range(20): arr[:]=data; bo.sync(TO,sz,0); bo.sync(FR,sz,0)
t0=time.time()
for _ in range(100): arr[:]=data; bo.sync(TO,sz,0)
h2d=(time.time()-t0)/100*1000
t0=time.time()
for _ in range(100): bo.sync(FR,sz,0)
d2h=(time.time()-t0)/100*1000
per_layer = h2d + d2h   # what fusion saves (out of FFN1 + into FFN2)
total = per_layer * 6
print(f"  per layer: {per_layer:.2f} ms (H->D {h2d:.2f} + D->H {d2h:.2f})")
print(f"  ×6 layers: {total:.1f} ms saved")
print(f"  forward is 210ms -> fusion saves ~{total/210*100:.0f}% (if intermediate stays on-chip)")
