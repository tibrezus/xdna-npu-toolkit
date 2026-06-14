"""profile_bf16.py — where does the bf16 forward spend time at batch 64?

After Phase 1 (bf16+tiled+pooled), what's the real bottleneck? If GEMMs are
only ~10ms of ~210ms, fusion won't help and we need to find the real cost.
"""
import sys, time
sys.path.insert(0, "/tmp/xdna-npu-toolkit/serve"); sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
import numpy as np, torch
from transformers import AutoTokenizer
from forward_bf16 import build_bf16_model, forward_bf16
from bf16_backend import Bf16Backend

WDIR = "/tmp/voe-inspect/minilm"; tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
texts = ["dogs and cats play in the park"] * 64; S = 64
enc = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="np")
ids = enc["input_ids"].astype(np.int64); mask = enc["attention_mask"].astype(np.int64)
npu = Bf16Backend()

# instrument the Bf16Linear call to separate: torch->numpy (conv) + GEMM + numpy->torch (conv) + bias
from forward_bf16 import Bf16Linear, Bf16QKV, _t2n, _n2t
orig_lin = Bf16Linear.__call__
orig_qkv = Bf16QKV.__call__
state = {"conv_in": 0, "gemm": 0, "conv_out": 0, "bias": 0, "ncalls": 0}

def timed_lin(self, x):
    t0=time.time(); A = _t2n(x.reshape(-1, x.shape[-1])); state["conv_in"]+=time.time()-t0
    t0=time.time(); out = self.backend(A, self.WT); state["gemm"]+=time.time()-t0
    t0=time.time(); t = _n2t(out).reshape(*x.shape[:-1], -1); state["conv_out"]+=time.time()-t0
    t0=time.time(); r = t + self.b; state["bias"]+=time.time()-t0; state["ncalls"]+=1
    return r
def timed_qkv(self, x):
    t0=time.time(); A = _t2n(x.reshape(-1, x.shape[-1])); state["conv_in"]+=time.time()-t0
    t0=time.time(); out = self.backend(A, self.WT); state["gemm"]+=time.time()-t0
    t0=time.time(); t = _n2t(out).reshape(*x.shape[:-1], 1156+16) if False else _n2t(out).reshape(*x.shape[:-1], 1152); state["conv_out"]+=time.time()-t0
    t0=time.time(); t = t + self.b; state["bias"]+=time.time()-t0; state["ncalls"]+=1
    q,k,v = t[...,:384], t[...,384:768], t[...,768:]
    return q,k,v
Bf16Linear.__call__ = timed_lin
Bf16QKV.__call__ = timed_qkv

m = build_bf16_model(WDIR, npu.run)
def fwd():
    state.update(conv_in=0,gemm=0,conv_out=0,bias=0,ncalls=0)
    t0=time.time(); forward_bf16(m, ids, mask, npu.run); tot=time.time()-t0
    return tot
for _ in range(3): fwd()
tots=[]; d={}
for _ in range(8):
    t=fwd(); tots.append(t)
    for k in state:
        d[k] = d.get(k,0)+state[k]
tot=np.mean(tots)
print(f"=== bf16 forward batch64: {tot*1000:.1f} ms ===")
print(f"  linears ({state['ncalls']} calls):")
for k in ["conv_in","gemm","conv_out","bias"]:
    print(f"    {k:10}: {d[k]/8*1000:7.1f} ms  ({d[k]/8/tot*100:4.1f}%)")
glue = tot - sum(d[k]/8 for k in ["conv_in","gemm","conv_out","bias"])
print(f"    {'glue(rest)':10}: {glue*1000:7.1f} ms  ({glue/tot*100:4.1f}%)")
print(f"  (linears total: {sum(d[k]/8 for k in state if k!='ncalls')*1000:.1f} ms)")
