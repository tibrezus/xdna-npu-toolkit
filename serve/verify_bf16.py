"""verify_bf16.py — bf16 NPU forward: correctness vs fp32 torch + semantics."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from forward_bf16 import build_bf16_model, forward_bf16
from bf16_backend import Bf16Backend

WDIR = "/tmp/voe-inspect/minilm"
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
S = 64; B = 64
npu = Bf16Backend()
cpu = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"); cpu.eval()

texts = ["A man is eating food.", "Someone eats a meal.", "A cat sleeps on the sofa.",
         "Two teams play soccer.", "The dog runs in the park."] * 13  # 65 -> use 64
texts = texts[:64]
enc = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="np")
ids = enc["input_ids"].astype(np.int64); mask = enc["attention_mask"].astype(np.int64)

# NPU bf16
fm = build_bf16_model(WDIR, npu.run)
emb_npu = forward_bf16(fm, ids, mask, npu.run)

# fp32 torch reference
enc2 = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="pt")
with torch.no_grad(): o = cpu(**enc2).last_hidden_state
m2 = enc2["attention_mask"][:, :, None].float()
emb_ref = torch.nn.functional.normalize((o*m2).sum(1)/m2.sum(1).clamp(min=1e-9), dim=1).numpy()

# correlation + per-text cos
n = 5  # the 5 distinct texts repeat
print(f"=== bf16 NPU vs fp32 torch ({len(texts)} texts, batch=64) ===")
coses = []
for i in range(5):
    c = float(emb_npu[i] @ emb_ref[i] / (np.linalg.norm(emb_npu[i]) * np.linalg.norm(emb_ref[i])))
    coses.append(c)
print(f"  per-text cos(npu,ref): {[f'{c:.4f}' for c in coses]}")
print(f"  mean cos: {np.mean(coses):.4f}")
print(f"  embedding-matrix corr: {np.corrcoef(emb_npu[:5].ravel(), emb_ref[:5].ravel())[0,1]:.4f}")

# semantics: paraphrase vs unrelated
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
print(f"\n  paraphrase (0,1) food: {cos(emb_npu[0],emb_npu[1]):.3f}")
print(f"  unrelated (0,3) sport: {cos(emb_npu[0],emb_npu[3]):.3f}")
print(f"  unrelated (2,4) cat/park: {cos(emb_npu[2],emb_npu[4]):.3f}")
ok = np.mean(coses) > 0.95 and cos(emb_npu[0],emb_npu[1]) > cos(emb_npu[0],emb_npu[3])
print(f"\n  {'PASS' if ok else 'FAIL'}: bf16 path preserves semantics")
