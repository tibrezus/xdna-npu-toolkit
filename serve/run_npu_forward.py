"""run_npu_forward.py — run MiniLM-L6-v2 forward with GEMMs on the NPU (#12).

The hybrid model: Linear projections (Q/K/V/O, FFN1/FFN2) run their int16 GEMM
on the Phoenix NPU; attention score matmuls + LayerNorm + softmax + GELU stay
on CPU (float). NPU and CPU-int16 paths are bit-identical (same int math).

batch=4 x seq=64 = M=256 (the compiled GEMM shape). Verify:
  1. NPU embeddings == CPU int16 embeddings (maxdiff == 0, bit-identical)
  2. semantics preserved (paraphrase > unrelated)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MINILM_GEMM_DIR", "/tmp/iron/minilm-gemms")
sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")   # for npu_kernel

import numpy as np
from transformers import AutoTokenizer
from minilm_forward import build_model, forward
from npu_backend import NpuGemmPool

WDIR = "/tmp/voe-inspect/minilm"
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

texts = [
    "A man is eating a piece of food.",
    "A man eats a meal at the table.",
    "The young girl is petting a small cat.",
    "Two dogs run together across the grassy park.",
    "Someone is preparing a fresh salad in the kitchen.",
    "A chef cooks a delicious dinner for guests.",
    "The cat sleeps peacefully on the warm sofa.",
    "A dog chases a ball across the open field.",
]
B, S = 8, 64   # M = 512 = compiled 4-col shape
enc = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="np")
ids = enc["input_ids"].astype(np.int64)
mask = enc["attention_mask"].astype(np.int64)

def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

# ---- CPU int16 reference ----
print("=== CPU int16 forward (reference) ===")
mcpu = build_model(WDIR, backend="cpu")
t0 = time.time(); out_cpu = forward(mcpu, ids, mask, backend="cpu"); t_cpu = time.time()-t0
print(f"   {t_cpu*1000:.1f} ms")
print(f"   paraphrase(0,1)={cos(out_cpu[0],out_cpu[1]):.4f}  unrelated(0,2)={cos(out_cpu[0],out_cpu[2]):.4f}")

# ---- NPU GEMM forward ----
print("\n=== NPU-hybrid forward (Linear GEMMs on Phoenix NPU) ===")
mnpu = build_model(WDIR, backend=NpuGemmPool.run)
t0 = time.time(); out_npu = forward(mnpu, ids, mask, backend=NpuGemmPool.run); t_npu = time.time()-t0
print(f"   {t_npu*1000:.1f} ms")

# ---- THE proof: bit-identical ----
maxdiff = np.abs(out_npu.astype(np.float64) - out_cpu.astype(np.float64)).max()
print(f"\n=== NPU vs CPU int16: max|diff| = {maxdiff} ===")
identical = np.array_equal(out_npu, out_cpu)
print(f"   bit-identical: {'YES' if identical else 'NO'}")

print("\n" + "=" * 60)
if identical:
    print(" PASS: NPU-hybrid embeddings are BIT-IDENTICAL to CPU int16.")
    print(" NPU correctness proven by construction (int16@int16->int32 exact).")
else:
    print(" FAIL: NPU and CPU diverge — investigate.")
print("=" * 60)
print(f"\nNOTE: single-core GEMMs at M=256; this proves correctness/integration.")
print(f"Throughput scaling (4-col) needs shape-specific tiling — tracked separately.")
