"""profile_single_gemm.py — break down ONE GEMM call into its cost components.

5.77ms/GEMM in the forward. Transfer is only ~0.8ms (measured). Where's the rest?
Hypothesis: host-side numpy memcpy into the BO (arr[:]=data) dominates.
"""
import sys, time, numpy as np
sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
import pyxrt
from ml_dtypes import bfloat16

dev = pyxrt.device(0); HO = pyxrt.bo.flags.host_only
TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
FR = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

# load the qkvfused kernel (4096x384x1152) like FastNpuKernel does
xcl = pyxrt.xclbin("/tmp/iron/minilm-gemms-bf16-M4096/qkvfused.xclbin"); dev.register_xclbin(xcl)
ctx = pyxrt.hw_context(dev, xcl.get_uuid())
kn = [k.get_name() for k in xcl.get_kernels()][0]; kern = pyxrt.kernel(ctx, kn)
with open("/tmp/iron/minilm-gemms-bf16-M4096/qkvfused.insts.txt","rb") as f:
    instr = np.frombuffer(f.read(), np.uint8)
ib=len(instr); ibo=pyxrt.bo(dev,ib,pyxrt.bo.flags.cacheable,kern.group_id(1))
np.frombuffer(ibo.map(),np.uint8)[:]=instr; ibo.sync(TO,ib,0)

M,K,N=4096,384,1152
a_bo=pyxrt.bo(dev,M*K*2,HO,kern.group_id(3)); a=np.frombuffer(a_bo.map(),bfloat16)
b_bo=pyxrt.bo(dev,K*N*2,HO,kern.group_id(4)); b=np.frombuffer(b_bo.map(),bfloat16)
o_bo=pyxrt.bo(dev,M*N*2,HO,kern.group_id(5)); o=np.frombuffer(o_bo.map(),bfloat16)

A=np.random.randn(M,K).astype(bfloat16); B=np.random.randn(K,N).astype(bfloat16)
asz=M*K*2; bsz=K*N*2; osz=M*N*2

def t(fn,n=200):
    for _ in range(20): fn()
    t0=time.time()
    for _ in range(n): fn()
    return (time.time()-t0)/n*1000

# isolate each step
t_copyA = t(lambda: a.__setitem__(slice(None), A.reshape(-1)))
t_copyB = t(lambda: b.__setitem__(slice(None), B.reshape(-1)))
t_syncA = t(lambda: a_bo.sync(TO,asz,0))
t_syncB = t(lambda: b_bo.sync(TO,bsz,0))
def launch():
    h=kern(3,ibo,ib,a_bo,b_bo,o_bo,0,0); h.wait()
t_launch = t(launch)
t_syncO = t(lambda: o_bo.sync(FR,osz,0))
t_copyO = t(lambda: np.frombuffer(o_bo.map(),bfloat16).copy())

# full pooled run
def full():
    a[:]=A.reshape(-1); b[:]=B.reshape(-1)
    a_bo.sync(TO,asz,0); b_bo.sync(TO,bsz,0)
    h=kern(3,ibo,ib,a_bo,b_bo,o_bo,0,0); h.wait()
    o_bo.sync(FR,osz,0)
t_full = t(full)

print(f"=== qkvfused GEMM (4096x384x1152) breakdown ===")
print(f"  copy A into BO (numpy {asz/1e6:.1f}MB):  {t_copyA:.3f} ms")
print(f"  copy B into BO (numpy {bsz/1e6:.1f}MB):  {t_copyB:.3f} ms")
print(f"  sync A (H->D):                  {t_syncA:.3f} ms")
print(f"  sync B (H->D):                  {t_syncB:.3f} ms")
print(f"  launch + wait (NPU compute):    {t_launch:.3f} ms")
print(f"  sync O (D->H):                  {t_syncO:.3f} ms")
print(f"  copy O out:                     {t_copyO:.3f} ms")
print(f"  sum of parts:                   {t_copyA+t_copyB+t_syncA+t_syncB+t_launch+t_syncO+t_copyO:.3f} ms")
print(f"  full pooled run:                {t_full:.3f} ms")
print(f"  pure compute estimate (1010GOPS): {2*M*K*N/1010e9*1000:.3f} ms")
print(f"\n  => overhead = {t_full - 2*M*K*N/1010e9*1000:.3f} ms, dominated by: ", end="")
parts=[("copyA",t_copyA),("copyB",t_copyB),("syncA",t_syncA),("syncB",t_syncB),("launch",t_launch),("syncO",t_syncO),("copyO",t_copyO)]
parts.sort(key=lambda x:-x[1])
print(f"{parts[0][0]} ({parts[0][1]:.2f}), {parts[1][0]} ({parts[1][1]:.2f})")
