"""bench_crossover.py — the decisive crossover curve: pooled-NPU vs torch-CPU."""
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
cpu = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"); cpu.eval()
npu = MultiMBackend()

def time_it(fn, warmup=4, iters=8):
    for _ in range(warmup): fn()
    t0=time.time()
    for _ in range(iters): fn()
    return (time.time()-t0)/iters

print(f"{'batch':>6} {'M':>6} {'torch-CPU':>14} {'pooled-NPU':>14} {'NPU/CPU':>9}")
print("-"*54)
results=[]
for B in [8, 16, 32, 64]:
    M = B*S
    texts = ["dogs and cats play in the park"] * B
    enc = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="pt")
    with torch.no_grad(): t_cpu = time_it(lambda: cpu(**enc), iters=8)
    enc_np = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="np")
    ids=enc_np["input_ids"].astype(np.int64); mask=enc_np["attention_mask"].astype(np.int64)
    fm = build_fast_model(WDIR, npu.run)
    t_npu = time_it(lambda: forward_fast(fm, ids, mask, npu.run), iters=5)
    spd = t_cpu/t_npu
    win = "<<< NPU WINS" if spd>1 else ""
    print(f"{B:>6} {M:>6} {t_cpu*1000:7.1f} ({t_cpu*1000/B:5.1f}) {t_npu*1000:7.1f} ({t_npu*1000/B:5.1f}) {spd:6.2f}x  {win}")
    results.append((B,t_cpu,t_npu,spd))

crossover = [b for b,tc,tn,s in results if s>1]
print(f"\nCrossover (NPU>CPU) at batch >= {crossover[0] if crossover else 'NONE'}")
if crossover:
    b=crossover[-1]; _,tc,tn,s=[r for r in results if r[0]==b][0]
    print(f"Best win: batch {b}: NPU {tn*1000/b:.1f} ms/text vs CPU {tc*1000/b:.1f} ms/text = {s:.2f}x")
