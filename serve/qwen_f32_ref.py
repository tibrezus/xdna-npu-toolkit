"""qwen_f32_ref.py — clean float32 torch forward, verified vs transformers.

The canonical Llama-style decoder forward. If cos(this, transformers) > 0.999,
the architecture is correct and I port this EXACT logic to the NPU bf16 path.
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
with safe_open(SPATH, framework="torch") as f:
    for k in f.keys(): W[k] = f.get_tensor(k).float()

H=1024; NL=28; NQ=16; NKV=8; HD=128; EPS=1e-6; THETA=1e6; SQHD=HD**0.5

def rms(x,w): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+EPS)*w
def make_rope(S):
    inv=1.0/(THETA**(torch.arange(0,HD,2).float()/HD))
    fr=torch.outer(torch.arange(S).float(),inv)
    return torch.cat([fr.cos(),fr.cos()],-1), torch.cat([fr.sin(),fr.sin()],-1)  # [S,HD]
def rope(x,cos,sin):  # x:[B,S,nh,HD]
    x1=x[...,:HD//2]; x2=x[...,HD//2:]
    return x*cos[None,:,None,:]+torch.cat([-x2,x1],-1)*sin[None,:,None,:]

def lin(x,w):  # x[..,in] w[out,in] -> [..,out]
    return x@w.T

texts=["A man is eating food.","Someone eats a meal.","A cat sleeps.","Two teams play soccer."]
enc=tok(texts,padding="max_length",truncation=True,max_length=64,return_tensors="pt")
ids=enc["input_ids"]; mask=enc["attention_mask"]
B,S=ids.shape
cos,sin=make_rope(S)
ext=mask[:,None,None,:].float()
causal=torch.triu(torch.full((S,S),-1e9),diagonal=1)  # [S,S]: j>i = -inf

def fwd():
    h=W["embed_tokens.weight"][ids]
    for i in range(NL):
        p=f"layers.{i}."
        hin=rms(h,W[p+"input_layernorm.weight"])
        q=lin(hin,W[p+"self_attn.q_proj.weight"]).reshape(B,S,NQ,HD)
        k=lin(hin,W[p+"self_attn.k_proj.weight"]).reshape(B,S,NKV,HD)
        v=lin(hin,W[p+"self_attn.v_proj.weight"]).reshape(B,S,NKV,HD)
        q=rms(q,W[p+"self_attn.q_norm.weight"]); k=rms(k,W[p+"self_attn.k_norm.weight"])
        q=rope(q,cos,sin); k=rope(k,cos,sin)
        q=q.transpose(1,2); k=k.transpose(1,2); v=v.transpose(1,2)  # [B,heads,S,HD]
        rep=NQ//NKV; k=k.repeat_interleave(rep,1); v=v.repeat_interleave(rep,1)
        sc=(q@k.transpose(-2,-1))/SQHD
        sc=sc+ causal + (1.0-ext)*-1e9
        at=torch.softmax(sc,-1)
        ctx=(at@v).transpose(1,2).reshape(B,S,NQ*HD)
        h=h+lin(ctx,W[p+"self_attn.o_proj.weight"])
        h2=rms(h,W[p+"post_attention_layernorm.weight"])
        g=lin(h2,W[p+"mlp.gate_proj.weight"]); u=lin(h2,W[p+"mlp.up_proj.weight"])
        mlp=lin(torch.nn.functional.silu(g)*u,W[p+"mlp.down_proj.weight"])
        h=h+mlp
    h=rms(h,W["norm.weight"])
    sl=mask.sum(1)-1
    return torch.nn.functional.normalize(h[torch.arange(B),sl],dim=-1)

with torch.no_grad():
    ours=fwd()
    refo=ref(**enc).last_hidden_state
    ref_emb=torch.nn.functional.normalize(refo[torch.arange(B),mask.sum(1)-1].float(),dim=-1)
for i in range(4):
    c=float(ours[i]@ref_emb[i]/(ours[i].norm()*ref_emb[i].norm()))
    print(f"  text[{i}] cos(ours f32, transformers) = {c:.6f}")
oc=float(ours[0]@ours[1]/(ours[0].norm()*ours[1].norm()))
print(f"  paraphrase(0,1) ours={oc:.3f}")
