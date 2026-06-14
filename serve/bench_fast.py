"""bench_fast.py — optimized NPU forward vs torch CPU. Can we beat CPU now?"""
import sys, time, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
os.environ.setdefault("MINILM_GEMM_DIR", "/tmp/iron/minilm-gemms")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from forward_fast import build_fast_model, forward_fast
from npu_backend import NpuGemmPool

WDIR = "/tmp/voe-inspect/minilm"
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
S = 64

# correctness reference
ref_texts = ["A man is eating food.", "A man eats a meal.", "A cat sleeps on the sofa."]
ref_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"); ref_model.eval()

def torch_embed(texts):
    enc = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="pt")
    with torch.no_grad():
        o = ref_model(**enc).last_hidden_state
    m = enc["attention_mask"][:, :, None].float()
    return torch.nn.functional.normalize((o*m).sum(1)/m.sum(1).clamp(min=1e-9), dim=1).numpy()

def time_it(fn, warmup=3, iters=10):
    for _ in range(warmup): fn()
    t0=time.time()
    for _ in range(iters): fn()
    return (time.time()-t0)/iters

print("=== correctness (3 reference texts) ===")
ref = torch_embed(ref_texts)
# pad to batch 8 for NPU (M=512)
enc = tok(ref_texts + [""]*5, padding="max_length", truncation=True, max_length=S, return_tensors="np")
ids = enc["input_ids"].astype(np.int64); mask = enc["attention_mask"].astype(np.int64)
fm = build_fast_model(WDIR, NpuGemmPool.run)
fast = forward_fast(fm, ids, mask, NpuGemmPool.run)
for i in range(3):
    cos = float(fast[i]@ref[i]/(np.linalg.norm(fast[i])*np.linalg.norm(ref[i])))
    print(f"  text[{i}]: cos(opt-NPU vs torch) = {cos:.5f}")

print("\n=== batch sweep: optimized-NPU vs torch-CPU (ms/batch, ms/text) ===")
print(f"{'batch':>6} {'torch-CPU':>14} {'opt-NPU':>14} {'speedup':>9}")
print("-"*48)
cpu_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"); cpu_model.eval()
for B in [8]:
    texts = ["dogs and cats play in the park"] * B
    enc = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="pt")
    with torch.no_grad():
        t_cpu = time_it(lambda: cpu_model(**enc), iters=8)
    enc_np = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="np")
    ids = enc_np["input_ids"].astype(np.int64); mask = enc_np["attention_mask"].astype(np.int64)
    fmn = build_fast_model(WDIR, NpuGemmPool.run)
    t_npu = time_it(lambda: forward_fast(fmn, ids, mask, NpuGemmPool.run), iters=5)
    spd = t_cpu/t_npu
    win = "NPU WINS" if spd>1 else "cpu"
    print(f"{B:>6} {t_cpu*1000:7.1f} ({t_cpu*1000/B:5.1f}) {t_npu*1000:7.1f} ({t_npu*1000/B:5.1f}) {spd:6.2f}x  {win}")
