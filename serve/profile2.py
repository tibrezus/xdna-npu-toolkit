"""profile2.py — clean targeted measurements (no lambda-instrumentation bug)."""
import sys, time
sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
sys.path.insert(0, "/tmp/xdna-npu-toolkit/serve")
import numpy as np
from scipy.special import erf

# 1. is gelu (scipy erf) actually fast?
x = np.random.randn(8,64,1536).astype(np.float32)
for _ in range(3): _ = 0.5*x*(1.0+erf(x/np.sqrt(2)))
t0=time.time()
for _ in range(50): _ = 0.5*x*(1.0+erf(x/np.sqrt(2)))
print(f"gelu (scipy erf) on [8,64,1536]: {(time.time()-t0)/50*1000:.3f} ms")

# 2. NpuGemmPool.run wrapper overhead vs raw NpuKernel.run
from npu_kernel import NpuKernel
from npu_backend import NpuGemmPool
k = NpuKernel("/tmp/iron/minilm-gemms/qkv-4col.xclbin", "/tmp/iron/minilm-gemms/qkv-4col.insts.txt")
A = np.random.randint(-100,100,(512,384),np.int16); B = np.random.randint(-100,100,(384,384),np.int16)

# raw kernel
for _ in range(3): k.run(A,B,out_sizes=[512*384*4])
t0=time.time()
for _ in range(100): k.run(A,B,out_sizes=[512*384*4])
raw = (time.time()-t0)/100*1000
print(f"raw NpuKernel.run (qkv 512): {raw:.3f} ms")

# pool wrapper
for _ in range(3): NpuGemmPool.run(A,B)
t0=time.time()
for _ in range(100): NpuGemmPool.run(A,B)
pool = (time.time()-t0)/100*1000
print(f"NpuGemmPool.run (qkv 512):   {pool:.3f} ms  (wrapper adds {pool-raw:.3f} ms)")

# 3. quantize/dequantize cost
from minilm_forward import quantize_i16, quantize_weight_i16
xf = np.random.randn(512,384).astype(np.float32)
Wf = np.random.randn(384,384).astype(np.float32)
Wq,ws = quantize_weight_i16(Wf)
t0=time.time()
for _ in range(100): _ = quantize_i16(xf)
print(f"quantize_i16 [512,384]:       {(time.time()-t0)/100*1000:.3f} ms")
acc = np.random.randint(-1<<30,1<<30,(512,384)).astype(np.int32)
t0=time.time()
for _ in range(100): _ = acc.astype(np.float32)*(0.01*ws[None,:])
print(f"dequantize [512,384]:         {(time.time()-t0)/100*1000:.3f} ms")

# 4. ascontiguousarray cost (called in pool)
t0=time.time()
for _ in range(1000): _ = np.ascontiguousarray(A)
print(f"ascontiguousarray (no-op):    {(time.time()-t0)/1000*1000:.3f} ms")

# 5. the attention matmuls
q = np.random.randn(8,12,64,32).astype(np.float32)
t0=time.time()
for _ in range(50): _ = (q @ q.transpose(0,1,3,2))
print(f"attn scores [8,12,64,32]@[8,12,32,64]: {(time.time()-t0)/50*1000:.3f} ms/layer")
