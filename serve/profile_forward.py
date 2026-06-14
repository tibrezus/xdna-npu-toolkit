"""profile_forward.py — where does the MiniLM forward actually spend time?

Hypothesis (from profile_dispatch.py): the NPU GEMMs are only ~32ms of the 226ms
forward. The rest is CPU glue. Find the real bottleneck.
"""
import sys, time
sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
sys.path.insert(0, "/tmp/xdna-npu-toolkit/serve")
import numpy as np
from transformers import AutoTokenizer
from minilm_forward import build_model, layernorm, softmax, gelu, quantize_i16, N_HEADS, HEAD_DIM
from npu_backend import NpuGemmPool

WDIR = "/tmp/voe-inspect/minilm"
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
texts = ["a sentence about cats and dogs"] * 8
enc = tok(texts, padding="max_length", truncation=True, max_length=64, return_tensors="np")
ids = enc["input_ids"].astype(np.int64); mask = enc["attention_mask"].astype(np.int64)

mnpu = build_model(WDIR, backend=NpuGemmPool.run)

# instrumented forward
from safetensors.numpy import load_file
B, S = ids.shape
x = mnpu["word_emb"][ids] + mnpu["pos_emb"][np.arange(S)[None,:]] + mnpu["tok_emb"][np.zeros_like(ids)]
x = layernorm(x, mnpu["emb_ln_w"], mnpu["emb_ln_b"])
ext = mask[:,None,None,:].astype(np.float32)

T = {"quant":0, "gemm":0, "attn_scores":0, "softmax":0, "attn_ctx":0, "layernorm":0, "gelu":0, "dequant":0, "reshape":0}

def tick(cat, fn, *a):
    t0=time.time(); r=fn(*a); T[cat]+=time.time()-t0; return r

for lyr in mnpu["layers"]:
    # QKV via NPU (instrumented)
    def do_linear(L, xin):
        lead = xin.shape[:-1]; xf = xin.reshape(-1, xin.shape[-1])
        xq = tick("quant", lambda: quantize_i16(xf))
        acc = tick("gemm", lambda: L.backend(xq[0], L.Wq.T))
        out = tick("dequant", lambda: acc.astype(np.float32)*(xq[1]*L.wscale[None,:])+L.b[None,:])
        return out.reshape(*lead, L.out_dim)
    q=do_linear(lyr["q"],x); k=do_linear(lyr["k"],x); v=do_linear(lyr["v"],x)
    def split(t): return t.reshape(B,S,N_HEADS,HEAD_DIM).transpose(0,2,1,3)
    q,k,v = split(q),split(k),split(v)
    scores = tick("attn_scores", lambda: (q @ k.transpose(0,1,3,2))/np.sqrt(HEAD_DIM))
    scores = scores + (1.0-ext)*-1e9
    attn = tick("softmax", lambda: softmax(scores,-1))
    ctx = tick("attn_ctx", lambda: attn @ v)
    ctx = tick("reshape", lambda: ctx.transpose(0,2,1,3).reshape(B,S,384))
    ctx = do_linear(lyr["o"], ctx)
    x = tick("layernorm", lambda: layernorm(x+ctx, lyr["attn_ln_w"], lyr["attn_ln_b"]))
    h = do_linear(lyr["f2"], tick("gelu", lambda: gelu(do_linear(lyr["f1"], x))))
    x = tick("layernorm", lambda: layernorm(x+h, lyr["out_ln_w"], lyr["out_ln_b"]))

total = sum(T.values())
print(f"=== batch8 x seq64 instrumented forward: {total*1000:.1f} ms ===\n")
for cat, t in sorted(T.items(), key=lambda x:-x[1]):
    print(f"  {cat:14} {t*1000:7.1f} ms  ({t/total*100:4.1f}%)")
print(f"  {'TOTAL':14} {total*1000:7.1f} ms")
