"""verify_qwen.py — verify Qwen3 forward (CPU vs transformers) + NPU benchmark.

backend="cpu"  -> numpy f32 matmul (architecture correctness)
backend="npu"  -> NPU bf16 GEMMs (perf + accuracy within bf16)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
import numpy as np, torch
from glob import glob
from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
from qwen_forward import load_model, forward, load_weights

M = "Qwen/Qwen3-Embedding-0.6B"
SPATH = glob("/home/tib/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*/model.safetensors")[0]
tok = AutoTokenizer.from_pretrained(M)
ref = AutoModel.from_pretrained(M); ref.eval()

def last_token_pool(h, mask):
    sl = mask.sum(1) - 1
    return h[torch.arange(h.shape[0]), sl]

def cpu_backend(A, WT):  # f32 matmul -> bf16 out (so _n2t byte-view works)
    from ml_dtypes import bfloat16
    return (A.astype(np.float32) @ WT.astype(np.float32)).astype(bfloat16)

texts = ["A man is eating food.", "Someone eats a meal.", "A cat sleeps on the sofa.", "Two teams play soccer."]
S = 64
enc = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="np")
ids = enc["input_ids"].astype(np.int64); mask = enc["attention_mask"].astype(np.int64)

# transformers reference
enc2 = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="pt")
with torch.no_grad(): o = ref(**enc2).last_hidden_state
emb_ref = F.normalize(last_token_pool(o, enc2["attention_mask"]).float(), dim=-1).numpy()

# our forward (CPU f32)
model = load_model(SPATH, cpu_backend)
emb_cpu = forward(model, ids, mask, cpu_backend, load_weights(SPATH))

print("=== Qwen3-0.6B CPU forward vs transformers ===")
for i in range(4):
    c = float(emb_cpu[i] @ emb_ref[i] / (np.linalg.norm(emb_cpu[i]) * np.linalg.norm(emb_ref[i])))
    print(f"  text[{i}] cos(ours, transformers) = {c:.5f}")
def cos(a,b): return float(emb_cpu[a]@emb_ref[b]/(np.linalg.norm(emb_cpu[a])*np.linalg.norm(emb_ref[b])))
print(f"  semantics: paraphrase(0,1)={cos(0,1):.3f}  unrelated(0,2)={cos(0,2):.3f}")
ok = all(float(emb_cpu[i]@emb_ref[i]/(np.linalg.norm(emb_cpu[i])*np.linalg.norm(emb_ref[i]))) > 0.95 for i in range(4))
print(f"  {'PASS' if ok else 'FAIL'}: architecture (RoPE/GQA/SwiGLU/pooling) correct")
