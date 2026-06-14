"""bench_one.py — benchmark a single batch size (run as subprocess).

Each batch loads only ITS 3 kernels (matching real serving). Prints JSON.
Usage: python bench_one.py <B>
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
os.environ.setdefault("MINILM_GEMM_DIR", "/tmp/iron/minilm-gemms")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from forward_fast import build_fast_model, forward_fast
from pooled_backend import MultiMBackend

B = int(sys.argv[1]); S = 64; M = B*S
WDIR = "/tmp/voe-inspect/minilm"
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
cpu = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"); cpu.eval()
npu = MultiMBackend()

def time_it(fn, warmup=4, iters=8):
    for _ in range(warmup): fn()
    t0=time.time()
    for _ in range(iters): fn()
    return (time.time()-t0)/iters

texts = ["dogs and cats play in the park"] * B
enc = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="pt")
with torch.no_grad(): t_cpu = time_it(lambda: cpu(**enc), iters=8)
enc_np = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="np")
ids=enc_np["input_ids"].astype(np.int64); mask=enc_np["attention_mask"].astype(np.int64)
fm = build_fast_model(WDIR, npu.run)
t_npu = time_it(lambda: forward_fast(fm, ids, mask, npu.run), iters=5)
print(json.dumps({"B":B,"M":M,"cpu_ms":t_cpu*1000,"npu_ms":t_npu*1000,
                   "cpu_per_text":t_cpu*1000/B,"npu_per_text":t_npu*1000/B,
                   "speedup":t_cpu/t_npu}))
