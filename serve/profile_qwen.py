"""profile_qwen.py — where does Qwen3-0.6B NPU forward spend time?

112 GEMMs (28 layers x 4) + heavy CPU glue (RoPE, rmsnorm, float32 attention).
Hypothesis: the CPU glue (float32 attention over 28 layers) dominates, not GEMMs.
"""
import sys, time
sys.path.insert(0, "/tmp/xdna-npu-toolkit/serve"); sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
import numpy as np, torch
from glob import glob
from transformers import AutoTokenizer
from qwen_forward import load_model, forward, load_weights, _t2n, _n2t, rmsnorm
from qwen_backend import QwenBf16Backend

M = "Qwen/Qwen3-Embedding-0.6B"
SPATH = glob("/home/tib/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*/model.safetensors")[0]
tok = AutoTokenizer.from_pretrained(M)
S, B = 64, 64
texts = ["dogs and cats play in the park"] * B
enc = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="np")
ids = enc["input_ids"].astype(np.int64); mask = enc["attention_mask"].astype(np.int64)
nb = QwenBf16Backend()

# instrument: time GEMM calls vs the rest
from qwen_forward import _FusedLinear, Linear
state = {"gemm": 0, "ngemm": 0}
orig_f = _FusedLinear.__call__; orig_l = Linear.__call__
def tf(self, x):
    t0=time.time(); r=orig_f(self,x); state["gemm"]+=time.time()-t0; state["ngemm"]+=1; return r
def tl(self, x):
    t0=time.time(); r=orig_l(self,x); state["gemm"]+=time.time()-t0; state["ngemm"]+=1; return r
_FusedLinear.__call__ = tf; Linear.__call__ = tl

model = load_model(SPATH, nb.run); W = load_weights(SPATH)
def fwd():
    state["gemm"]=0; state["ngemm"]=0
    t0=time.time(); forward(model, ids, mask, nb.run, W); return time.time()-t0
for _ in range(2): fwd()
tot, gem = [], []
for _ in range(3):
    t=fwd(); tot.append(t); gem.append(state["gemm"])
T=np.mean(tot); G=np.mean(gem)
print(f"=== Qwen3-0.6B NPU forward batch64 ===")
print(f"  total:        {T*1000:.0f} ms")
print(f"  GEMMs ({state['ngemm']} calls): {G*1000:.0f} ms  ({G/T*100:.0f}%)")
print(f"  glue (rest):  {(T-G)*1000:.0f} ms  ({(T-G)/T*100:.0f}%)")
print(f"  per GEMM avg: {G/state['ngemm']*1000:.2f} ms")
