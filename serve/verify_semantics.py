"""verify_semantics.py — does the pooled int16 NPU path preserve embedding quality?

Tests: paraphrase cosine >> unrelated cosine, and NPU embeddings cluster correctly.
Compares discrimination against torch-CPU reference.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
os.environ.setdefault("MINILM_GEMM_DIR", "/tmp/iron/minilm-gemms")
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from forward_fast import build_fast_model, forward_fast
from pooled_backend import MultiMBackend

WDIR = "/tmp/voe-inspect/minilm"
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
S = 64
npu = MultiMBackend()
cpu = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"); cpu.eval()

groups = [
    ("food",  ["A man is eating food.", "Someone eats a meal.", "A person is having dinner."]),
    ("pets",  ["A cat sleeps on the sofa.", "A kitten rests on the couch.", "The cat is napping."]),
    ("sport", ["Two teams play soccer.", "Players compete in a football match.", "Athletes play a game."]),
]
texts = [t for _, ts in groups for t in ts]
B = 8  # pad to M=512
while len(texts) % B: texts.append("zzz padding")

def cpu_embed(ts):
    enc = tok(ts, padding="max_length", truncation=True, max_length=S, return_tensors="pt")
    with torch.no_grad(): o = cpu(**enc).last_hidden_state
    m = enc["attention_mask"][:,:,None].float()
    return torch.nn.functional.normalize((o*m).sum(1)/m.sum(1).clamp(min=1e-9), dim=1).numpy()[:len(ts)]

def npu_embed(ts):
    enc = tok(ts, padding="max_length", truncation=True, max_length=S, return_tensors="np")
    fm = build_fast_model(WDIR, npu.run)
    return forward_fast(fm, enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64), npu.run)[:len(ts)]

ce = cpu_embed(texts); ne = npu_embed(texts)

print("=== embedding quality: within-group (paraphrase) vs cross-group (unrelated) ===")
print(f"{'':16} {'within-grp avg':>15} {'cross-grp avg':>15} {'discrim':>9}")
for name, (a, b, c) in zip([n for n,_ in groups], [(0,1,3),(3,4,6),(6,7,9)]):
    for emb,label in [(ce,"cpu"),(ne,"npu")]:
        within = np.mean([emb[a]@emb[b]/(np.linalg.norm(emb[a])*np.linalg.norm(emb[b])),
                          emb[a]@emb[c]/(np.linalg.norm(emb[a])*np.linalg.norm(emb[c]))])
        cross = np.mean([emb[a]@emb[3]/(np.linalg.norm(emb[a])*np.linalg.norm(emb[3])) if label=="npu" else emb[a]@emb[3]/(np.linalg.norm(emb[a])*np.linalg.norm(emb[3]))])
        cross_real = emb[a] @ emb[(a+3)%9] / (np.linalg.norm(emb[a])*np.linalg.norm(emb[(a+3)%9]))
    pass

# cleaner: matrix of all pairwise
import itertools
def pairwise(embs):
    n=len(embs); M=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            M[i,j]=embs[i]@embs[j]/(np.linalg.norm(embs[i])*np.linalg.norm(embs[j]))
    return M
Mc, Mn = pairwise(ce[:9]), pairwise(ne[:9])
within_c = np.mean([Mc[i,j] for g in range(3) for i,j in itertools.combinations(range(g*3,g*3+3),2)])
cross_c = np.mean([Mc[i,j] for i in range(9) for j in range(9) if i//3 != j//3])
within_n = np.mean([Mn[i,j] for g in range(3) for i,j in itertools.combinations(range(g*3,g*3+3),2)])
cross_n = np.mean([Mn[i,j] for i in range(9) for j in range(9) if i//3 != j//3])
print(f"\n  CPU: within-group={within_c:.3f}  cross-group={cross_c:.3f}  gap={within_c-cross_c:.3f}")
print(f"  NPU: within-group={within_n:.3f}  cross-group={cross_n:.3f}  gap={within_n-cross_n:.3f}")
print(f"\n  NPU vs CPU embedding matrix correlation: {np.corrcoef(Mc.ravel(), Mn.ravel())[0,1]:.4f}")
verdict = (within_n > 0.6) and (cross_n < within_n) and (within_n-cross_n > 0.3)
print(f"\n  {'PASS' if verdict else 'FAIL'}: NPU path preserves semantic discrimination")
