"""qwen_ref.py — establish transformers CPU reference + inspect weight shapes.

Confirms GEMM shapes for the NPU plan and verifies the embedding recipe.
"""
import torch, numpy as np, time
from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F

M = "Qwen/Qwen3-Embedding-0.6B"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModel.from_pretrained(M); model.eval()

# inspect weight shapes
print("=== weight shapes (GEMM plan) ===")
sd = model.state_dict()
for k in sd:
    if k.startswith("model.layers.0.") and ("proj.weight" in k):
        print(f"  {k.replace('model.layers.0.','')}: {tuple(sd[k].shape)}  [out, in]")
print(f"  embed_tokens: {tuple(sd['model.embed_tokens.weight'].shape)}")
print(f"  num layers: {sum(1 for k in sd if k.endswith('.o_proj.weight'))}")

def last_token_pool(h, mask):
    sl = mask.sum(dim=1) - 1
    return h[torch.arange(h.shape[0]), sl]

# reference embedding
texts = ["A man is eating food.", "Someone eats a meal.", "A cat sleeps on the sofa.", "Two teams play soccer."]
inputs = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
with torch.no_grad():
    out = model(**inputs).last_hidden_state
emb = F.normalize(last_token_pool(out, inputs["attention_mask"]), p=2, dim=1)
print("\n=== reference embeddings (CPU, fp32) ===")
print(f"  shape {tuple(emb.shape)}")
def cos(a,b): return float(F.cosine_similarity(emb[a:a+1], emb[b:b+1])[0])
print(f"  paraphrase(0,1) food: {cos(0,1):.4f}")
print(f"  unrelated(0,2):       {cos(0,2):.4f}")
print(f"  unrelated(0,3) sport: {cos(0,3):.4f}")

# time it at batch 64
texts64 = ["dogs and cats play in the sunny park"] * 64
inp64 = tok(texts64, padding=True, truncation=True, max_length=64, return_tensors="pt")
for _ in range(3):
    with torch.no_grad(): model(**inp64)
t0=time.time()
for _ in range(8):
    with torch.no_grad(): model(**inp64)
print(f"\n  CPU fp32 batch64: {(time.time()-t0)/8*1000:.1f} ms ({(time.time()-t0)/8*1000/64:.2f} ms/text)")
