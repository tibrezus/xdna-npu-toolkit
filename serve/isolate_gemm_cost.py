"""isolate_gemm_cost.py — time JUST the 24 backend.run() calls, no torch glue.

If 24 raw GEMM calls with real weights ≈ 152ms (the in-forward number), the gap
is inherent. If they're much faster, the forward has torch contention/overhead
that's the real lever (before fusion).
"""
import sys, time
sys.path.insert(0, "/tmp/xdna-npu-toolkit/serve"); sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
import numpy as np, torch
from safetensors import safe_open
from ml_dtypes import bfloat16
from bf16_backend import Bf16Backend

WDIR = "/tmp/voe-inspect/minilm"; M, S = 4096, 64
W = {}
with safe_open(f"{WDIR}/model.safetensors", framework="numpy") as f:
    for k in f.keys(): W[k] = f.get_tensor(k)

npu = Bf16Backend()
# build the 4 weight matrices per layer as the forward does (bf16, contiguous, transposed)
def mkWT(name):  # weight [out,in] -> [in,out] bf16 contig
    return np.ascontiguousarray(W[f"encoder.layer.0.{name}.weight"].astype(np.float32).astype(bfloat16).T)
def mkWTqkv(i):
    Wq=W[f"encoder.layer.{i}.attention.self.query.weight"]; Wk=W[f"encoder.layer.{i}.attention.self.key.weight"]; Wv=W[f"encoder.layer.{i}.attention.self.value.weight"]
    return np.ascontiguousarray(np.concatenate([Wq,Wk,Wv],0).astype(np.float32).astype(bfloat16).T)

# activation: bf16 [M, in]
def act(k): return np.random.randn(M,k).astype(np.float32).astype(bfloat16)

# the 24 GEMM calls per forward (one layer's pattern, ×6)
layers = []
for i in range(6):
    layers.append({
        "qkv": (mkWTqkv(i), 384, 1152),
        "o":   (mkWT(f"attention.output.dense"), 384, 384),
        "f1":  (mkWT("intermediate.dense"), 384, 1536),
        "f2":  (mkWT("output.dense"), 1536, 384),
    })

# pre-make activations of the right sizes
A384 = act(384); A1536 = act(1536)

def run_24_gemms():
    for ly in layers:
        npu.run(A384, ly["qkv"][0])    # [M,384]x[384,1152]
        npu.run(A384, ly["o"][0])      # [M,384]x[384,384]
        npu.run(A384, ly["f1"][0])     # [M,384]x[384,1536]
        npu.run(A1536, ly["f2"][0])    # [M,1536]x[1536,384]

for _ in range(3): run_24_gemms()
t0=time.time()
for _ in range(8): run_24_gemms()
ms=(time.time()-t0)/8*1000
print(f"=== 24 raw GEMM calls (real weights, no torch): {ms:.1f} ms ===")
print(f"  per GEMM: {ms/24:.2f} ms")
print(f"  (in-forward GEMM total was 152.7 ms)")
print(f"  -> raw GEMM is {'FASTER' if ms<152.7 else 'same/slower'} than in-forward; gap = {152.7-ms:.1f} ms = torch/glue contention")

# also time each shape separately
for name,(WT,kin,kout) in [("qkvfused", (layers[0]["qkv"][0],384,1152)),
                            ("o", (layers[0]["o"][0],384,384)),
                            ("ffn1", (layers[0]["f1"][0],384,1536)),
                            ("ffn2", (layers[0]["f2"][0],1536,384))]:
    A = act(kin)
    for _ in range(5): npu.run(A, WT)
    t0=time.time()
    for _ in range(20): npu.run(A, WT)
    m=(time.time()-t0)/20*1000
    gops=2*M*kin*kout/(m/1000)/1e9
    print(f"  {name:9} ({M}x{kin}x{kout}): {m:5.2f} ms  {gops:5.0f} GOPS")
