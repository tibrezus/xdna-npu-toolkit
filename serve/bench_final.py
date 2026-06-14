"""bench_final.py — the decisive sweep: pooled-NPU vs torch-CPU across batches.

After fixes (BO pooling + torch glue): does the NPU beat CPU? At what batch?
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
os.environ.setdefault("MINILM_GEMM_DIR", "/tmp/iron/minilm-gemms")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from forward_fast import build_fast_model, forward_fast
from pooled_backend import MultiMBackend

WDIR = "/tmp/voe-inspect/minilm"
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
S = 64
cpu_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"); cpu_model.eval()

def time_it(fn, warmup=4, iters=8):
    for _ in range(warmup): fn()
    t0=time.time()
    for _ in range(iters): fn()
    return (time.time()-t0)/iters

# correctness check at batch8 first
texts = ["A man is eating food.", "A man eats a meal.", "A cat sleeps on the sofa."]
enc = tok(texts+[""]*5, padding="max_length", truncation=True, max_length=S, return_tensors="np")
ids=enc["input_ids"].astype(np.int64); mask=enc["attention_mask"].astype(np.int64)
npu = MultiMBackend()
fm = build_fast_model(WDIR, npu.run)
fast = forward_fast(fm, ids, mask, npu.run)
enc2 = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="pt")
with torch.no_grad(): o = cpu_model(**enc2).last_hidden_state
m2 = enc2["attention_mask"][:,:,None].float()
ref = torch.nn.functional.normalize((o*m2).sum(1)/m2.sum(1).clamp(min=1e-9), dim=1).numpy()
print("=== correctness (pooled NPU vs torch) ===")
for i in range(3):
    print(f"  cos={float(fast[i]@ref[i]/(np.linalg.norm(fast[i])*np.linalg.norm(ref[i]))):.4f}")

print("\n=== pooled-NPU vs torch-CPU ===")
print(f"{'batch':>6} {'M':>6} {'torch-CPU':>14} {'pooled-NPU':>14} {'NPU/CPU':>9}")
print("-"*54)
for B in [8, 64]:
    M = B*S
    texts = ["dogs and cats play in the park"] * B
    enc = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="pt")
    with torch.no_grad(): t_cpu = time_it(lambda: cpu_model(**enc), iters=8)
    enc_np = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="np")
    ids=enc_np["input_ids"].astype(np.int64); mask=enc_np["attention_mask"].astype(np.int64)
    fm = build_fast_model(WDIR, npu.run)
    t_npu = time_it(lambda: forward_fast(fm, ids, mask, npu.run), iters=5)
    spd = t_cpu/t_npu
    win = "<<< NPU WINS" if spd>1 else "(cpu)"
    print(f"{B:>6} {M:>6} {t_cpu*1000:7.1f} ({t_cpu*1000/B:5.1f}) {t_npu*1000:7.1f} ({t_npu*1000/B:5.1f}) {spd:6.2f}x  {win}")
