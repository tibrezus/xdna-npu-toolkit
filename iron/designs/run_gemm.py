#!/usr/bin/env python3
"""Strategy A — execute the compiled int16 GEMM on the Phoenix NPU + verify.

Mirrors IRON's own proven XRT host-runtime pattern (hostruntime/xrtruntime/tensor.py):
  device(0) -> load_xclbin -> kernel -> bo(...).map() -> np.frombuffer
  kernel(3, insts_bo, insts_bytes, *data_bos)
"""
import sys, os, time
import numpy as np
import pyxrt

M = K = N = 256
BUILD = sys.argv[1] if len(sys.argv) > 1 else \
    "/tmp/mliraie-v132/programming_examples/basic/matrix_multiplication/single_core/build"
XCLBIN = f"{BUILD}/final_256x256x256_32x32x32.xclbin"
INSTS  = f"{BUILD}/insts_256x256x256_32x32x32.txt"

print(f"xclbin: {XCLBIN} ({os.path.getsize(XCLBIN)} bytes)")
print(f"insts:  {INSTS}  ({os.path.getsize(INSTS)} bytes)")

with open(INSTS, "rb") as f:
    raw = f.read()
instr = np.frombuffer(raw, dtype=np.uint8)        # byte stream
instr_bytes = len(instr)
print(f"instruction bytes: {instr_bytes}")

# ---- device + xclbin + kernel (NEW xrt API: register_xclbin, NOT load_xclbin) ----
# IRON's proven path: pyxrt.xclbin -> device.register_xclbin -> hw_context -> kernel(ctx,...)
# (the old dev.load_xclbin(path) returns EOPNOTSUPP on amdxdna firmware)
dev = pyxrt.device(0)
xcl = pyxrt.xclbin(XCLBIN)
dev.register_xclbin(xcl)
ctx = pyxrt.hw_context(dev, xcl.get_uuid())
kname = [k.get_name() for k in xcl.get_kernels()][0]
kern = pyxrt.kernel(ctx, kname)
print(f"kernel: {kname}")
print(f"kernel group_ids: 1={kern.group_id(1)} 3={kern.group_id(3)} 4={kern.group_id(4)} 5={kern.group_id(5)}")

# ---- data ----
A = np.random.randint(-100, 100, (M, K), dtype=np.int16)
B = np.random.randint(-100, 100, (K, N), dtype=np.int16)
ref = (A.astype(np.int32) @ B.astype(np.int32)).astype(np.int32)

HOST_ONLY = pyxrt.bo.flags.host_only
CACHEABLE = pyxrt.bo.flags.cacheable
TO_DEV    = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
FROM_DEV  = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

def make_bo(size, dtype, gid, flags=HOST_ONLY, fill=None):
    bo = pyxrt.bo(dev, size, flags, kern.group_id(gid))
    arr = np.frombuffer(bo.map(), dtype=dtype)
    if fill is not None:
        arr[:] = fill
        bo.sync(TO_DEV, size, 0)
    return bo, arr

bo_instr, _ = make_bo(instr_bytes, np.uint8, 1, flags=CACHEABLE, fill=instr)
bo_a, _     = make_bo(M*K*2, np.int16, 3, fill=A.reshape(-1))
bo_b, _     = make_bo(K*N*2, np.int16, 4, fill=B.reshape(-1))
bo_c, c_arr = make_bo(M*N*4, np.int32, 5)   # output, zero-init

# ---- run ----
print("running GEMM on Phoenix NPU...")
t0 = time.time()
h = kern(3, bo_instr, instr_bytes, bo_a, bo_b, bo_c, 0, 0)
state = h.wait()
dt = time.time() - t0
print(f"kernel state: {state} (COMPLETED={pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED})  time={dt*1000:.1f} ms")

# ---- read back + verify ----
bo_c.sync(FROM_DEV, M*N*4, 0)
out = c_arr.reshape(M, N).copy()

if np.array_equal(out, ref):
    print(f"PASS!  int16 GEMM {M}x{K}x{N} on Phoenix NPU == numpy exactly.")
    print(f"   max|C|={np.abs(out).max()}   C[0,:4]={out[0,:4]}")
    print(f"   ref[0,:4]={ref[0,:4]}")
else:
    d = np.abs(out.astype(np.int64) - ref.astype(np.int64))
    print(f"MISMATCH: max|diff|={d.max()}  exact matches={(d==0).sum()}/{M*N}")
    print(f"   out[0,:4]={out[0,:4]}\n   ref[0,:4]={ref[0,:4]}")
