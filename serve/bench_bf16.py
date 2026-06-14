"""bench_bf16.py — bf16 NPU vs torch-CPU vs int16 NPU baseline. The Phase-1 test."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from forward_bf16 import build_bf16_model, forward_bf16
from bf16_backend import Bf16Backend

B = int(sys.argv[1]); mode = sys.argv[2]; S = 64
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
elif mode == "npu_bf16":
    npu = Bf16Backend()
    enc = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="np")
    ids=enc["input_ids"].astype(np.int64); mask=enc["attention_mask"].astype(np.int64)
    fm = build_bf16_model(WDIR, npu.run)
    t = time_it(lambda: forward_bf16(fm, ids, mask, npu.run))
print(json.dumps({"B":B,"mode":mode,"ms":t*1000,"per_text":t*1000/B}))
