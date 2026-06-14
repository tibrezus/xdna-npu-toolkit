"""bench_fused.py — QKV-fused vs non-fused vs CPU, single batch (subprocess).

Usage: python bench_fused.py <B> <mode>   where mode in {cpu, npu, npu_fused}
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
os.environ.setdefault("MINILM_GEMM_DIR", "/tmp/iron/minilm-gemms")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from forward_fast import build_fast_model, forward_fast
from forward_fused import build_fused_model, forward_fused
from pooled_backend import MultiMBackend

B = int(sys.argv[1]); mode = sys.argv[2]; S = 64; M = B*S
WDIR = "/tmp/voe-inspect/minilm"
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
texts = ["dogs and cats play in the park"] * B

def time_it(fn, warmup=4, iters=8):
    for _ in range(warmup): fn()
    t0=time.time()
    for _ in range(iters): fn()
    return (time.time()-t0)/iters

if mode == "cpu":
    cpu = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"); cpu.eval()
    enc = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="pt")
    with torch.no_grad(): t = time_it(lambda: cpu(**enc))
elif mode == "npu":
    npu = MultiMBackend()
    enc = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="np")
    ids=enc["input_ids"].astype(np.int64); mask=enc["attention_mask"].astype(np.int64)
    fm = build_fast_model(WDIR, npu.run)
    t = time_it(lambda: forward_fast(fm, ids, mask, npu.run))
elif mode == "npu_fused":
    npu = MultiMBackend()
    enc = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="np")
    ids=enc["input_ids"].astype(np.int64); mask=enc["attention_mask"].astype(np.int64)
    fm = build_fused_model(WDIR, npu.run)
    t = time_it(lambda: forward_fused(fm, ids, mask, npu.run))
print(json.dumps({"B":B,"mode":mode,"ms":t*1000,"per_text":t*1000/B}))
