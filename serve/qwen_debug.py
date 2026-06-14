"""qwen_debug.py — pure float32 torch forward, compare to transformers.

Isolates architecture (RoPE/GQA/SwiGLU/norm) from bf16 plumbing. If this matches
transformers but bf16 path doesn't -> bf16 issue. If this doesn't match -> arch bug.
"""
import torch, numpy as np
from glob import glob
from transformers import AutoTokenizer, AutoModel
from safetensors import safe_open

M = "Qwen/Qwen3-Embedding-0.6B"
SPATH = glob("/home/tib/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*/model.safetensors")[0]
tok = AutoTokenizer.from_pretrained(M)
ref = AutoModel.from_pretrained(M); ref.eval()

W = {}
with safe_open(SPATH, framework="numpy") as f:
    for k in f.keys(): W[k] = torch.from_numpy(f.get_tensor(k)).float()

H=1024; NL=28; NQ=16; NKV=8; HD=128; INTER=3072; EPS=1e-6; THETA=1e6

def rms(x, w): return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+EPS) * w
def rope(x, cos, sin):  # x[B,S,nh,hd], cos/sin [S,hd]
    x1=x[...,:HD//2]; x2=x[...,HD//2:]; rot=torch.cat([-x2,x1],-1)
    return x*cos[None,:,None,:]+rot*sin[None,:,None,:]
inv=1.0/(THETA**(torch.arange(0,HD,2).float()/HD))
fr=torch.outer(torch.arange(64).float(),inv)
cos_full=torch.cat([fr.cos(),fr.cos()],-1); sin_full=torch.cat([fr.sin(),fr.sin()],-1)

texts=["A man is eating food.","Someone eats a meal.","A cat sleeps.","Two teams play soccer."]
enc=tok(texts,padding="max_length",truncation=True,max_length=64,return_tensors="pt")
ids=enc["input_ids"]; mask=enc["attention_mask"]

def fwd():
    B,S=ids.shape
    h=W["embed_tokens.weight"][ids]
    ext=mask[:,None,None,:].float()
    for i in range(NL):
        p=f"layers.{i}."
        hin=rms(h,W[p+"input_layernorm.weight"])
        q=W[p+"self_attn.q_proj.weight"]@hin.reshape(-1,H).T  # testing approach
        return q.reshape(B,S,NQ*HD)  # just first op, to compare incrementally
    return h

# instead, get transformers layer0 outputs by hooking
acts={}
def hook(name):
    def f(m,i,o): acts[name]=o[0] if isinstance(o,tuple) else o
    return f
hm=[]
for name,m in ref.named_modules():
    if name.endswith("layers.0"): hm.append(m.register_forward_hook(hook("layer0")))
with torch.no_grad(): ref_out=ref(**enc)
for h in hm: h.remove()

print("layer0 output (transformers):", acts["layer0"].shape)
# our layer0
with torch.no_grad():
    ours_layer0=fwd()
print("ours q (first op):", ours_layer0.shape)
# This fwd returns just q after q_proj - not the full layer. Just a smoke test.
print("transformers layer0[0,0,:5]:", acts["layer0"][0,0,:5].tolist())
