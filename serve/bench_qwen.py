"""bench_qwen.py — Qwen3-0.6B: NPU bf16 vs torch-CPU. The bigger-model test.

Usage: python bench_qwen.py <mode>   (mode: cpu | npu | verify)
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
import numpy as np, torch
from glob import glob
from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
from qwen_forward import load_model, forward, load_weights

M = "Qwen/Qwen3-Embedding-0.6B"
SPATH = glob("/home/tib/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*/model.safetensors")[0]
tok = AutoTokenizer.from_pretrained(M)
S, B = 64, 64
mode = sys.argv[1] if len(sys.argv) > 1 else "npu"
texts = ["dogs and cats play together in the sunny park"] * B
enc = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="np")
ids = enc["input_ids"].astype(np.int64); mask = enc["attention_mask"].astype(np.int64)

def time_it(fn, warmup=2, iters=4):
    for _ in range(warmup): fn()
    t0 = time.time()
    for _ in range(iters): fn()
    return (time.time()-t0)/iters

if mode == "cpu":
    model = AutoModel.from_pretrained(M); model.eval()
    enc2 = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="pt")
    with torch.no_grad():
        t = time_it(lambda: model(**enc2), iters=4)
    print(json.dumps({"mode":"cpu","ms":t*1000,"per_text":t*1000/B}))

elif mode == "npu":
    from qwen_backend import QwenBf16Backend
    nb = QwenBf16Backend()
    model = load_model(SPATH, nb.run)
    W = load_weights(SPATH)
    t = time_it(lambda: forward(model, ids, mask, nb.run, W), warmup=2, iters=4)
    print(json.dumps({"mode":"npu","ms":t*1000,"per_text":t*1000/B}))

elif mode == "verify":
    from qwen_backend import QwenBf16Backend
    nb = QwenBf16Backend()
    model = load_model(SPATH, nb.run)
    W = load_weights(SPATH)
    emb_npu = forward(model, ids, mask, nb.run, W)
    # reference
    texts4 = ["A man is eating food.", "Someone eats a meal.", "A cat sleeps.", "Two teams play soccer."]
    enc4 = tok(texts4, padding="max_length", truncation=True, max_length=S, return_tensors="np")
    ids4 = enc4["input_ids"].astype(np.int64); mask4 = enc4["attention_mask"].astype(np.int64)
    # pad to batch 64
    pad = 64 - 4
    ids64 = np.concatenate([ids4, np.zeros((pad,S),np.int64)]); mask64 = np.concatenate([mask4, np.zeros((pad,S),np.int64)])
    emb = forward(load_model(SPATH, nb.run), ids64, mask64, nb.run, W)[:4]
    ref = AutoModel.from_pretrained(M); ref.eval()
    enc2 = tok(texts4, padding="max_length", truncation=True, max_length=S, return_tensors="pt")
    with torch.no_grad(): o = ref(**enc2).last_hidden_state
    sl = enc2["attention_mask"].sum(1)-1
    emb_ref = F.normalize(o[torch.arange(4),sl].float(), dim=-1).numpy()
    print("=== Qwen3-0.6B NPU bf16 vs transformers fp32 ===")
    for i in range(4):
        c = float(emb[i]@emb_ref[i]/(np.linalg.norm(emb[i])*np.linalg.norm(emb_ref[i])))
        print(f"  text[{i}] cos(npu, transformers) = {c:.5f}")
    print(f"  paraphrase(0,1)={float(emb[0]@emb[1]/(np.linalg.norm(emb[0])*np.linalg.norm(emb[1]))):.3f}  unrelated(0,3)={float(emb[0]@emb[3]/(np.linalg.norm(emb[0])*np.linalg.norm(emb[3]))):.3f}")
