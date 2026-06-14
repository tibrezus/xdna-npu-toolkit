"""validate_forward.py — validate the numpy MiniLM forward vs transformers.

1. float backend == transformers BertModel output (verifies the forward is correct)
2. int16-cpu backend cos-sim vs float (measures quantization impact)
3. semantic sanity: paraphrase cos > unrelated cos (from issue #3 pattern)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from minilm_forward import build_model, forward

WDIR = "/tmp/voe-inspect/minilm"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

texts = [
    "A man is eating food.",
    "A man eats a meal.",
    "The girl is petting a cat.",
    "Two dogs are running in the park.",
]
paraphrase = (0, 1)      # should be high
unrelated = (0, 2)       # should be lower

def encode(texts, max_len=64):
    enc = tok(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="np")
    return enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)

ids, mask = encode(texts)
print("=== forward shapes:", ids.shape)

def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

# ---- 1. float forward ----
mf = build_model(WDIR, backend="float")
out_f = forward(mf, ids, mask, backend="float")
print("\n[float] embeddings shape:", out_f.shape)
print(f"  paraphrase(0,1) cos: {cos(out_f[0], out_f[1]):.4f}  (want >0.5)")
print(f"  unrelated(0,2)  cos: {cos(out_f[0], out_f[2]):.4f}  (want lower)")
ok_sem = cos(out_f[0], out_f[1]) > cos(out_f[0], out_f[2])

# ---- 2. transformers ground truth ----
import torch
from transformers import AutoModel
tmodel = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tmodel.eval()
with torch.no_grad():
    tm = tmodel(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))
    tpool = (tm.last_hidden_state * torch.tensor(mask)[:,:,None]).sum(1)
    tpool = tpool / torch.tensor(mask).sum(1)[:,None].clip(min=1e-9)
    tout = torch.nn.functional.normalize(tpool, dim=1).numpy()
maxdiff = np.abs(out_f - tout).max()
cossim_mean = np.mean([cos(out_f[i], tout[i]) for i in range(len(texts))])
print(f"\n[float vs transformers] max|diff|={maxdiff:.2e}  mean cos={cossim_mean:.5f}  (want cos>0.9999)")
ok_match = cossim_mean > 0.9999

# ---- 3. int16-cpu (the quantized reference the NPU must match) ----
mi = build_model(WDIR, backend="cpu")
out_i = forward(mi, ids, mask, backend="cpu")
icossim_mean = np.mean([cos(out_i[i], tout[i]) for i in range(len(texts))])
idiff = np.abs(out_i.astype(np.float64) - out_f.astype(np.float64)).max()
print(f"\n[int16-cpu vs float]    max|diff|={idiff:.2e}  vs-transformers cos={icossim_mean:.5f}")
print(f"  paraphrase cos: {cos(out_i[0], out_i[1]):.4f}  unrelated: {cos(out_i[0], out_i[2]):.4f}")
ok_quant = cos(out_i[0], out_i[1]) > cos(out_i[0], out_f[2])

print("\n" + "="*60)
print(f"  forward correct (float==transformers):  {'PASS' if ok_match else 'FAIL'}")
print(f"  semantics (paraphrase>unrelated):        {'PASS' if ok_sem else 'FAIL'}")
print(f"  int16 quantization preserves meaning:    {'PASS' if ok_quant else 'FAIL'}")
print("="*60)
