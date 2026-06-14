"""qwen_forward.py — Qwen3-Embedding-0.6B forward (bf16, NPU GEMMs + torch glue).

Architecture: decoder LLM, RMSNorm, q/k-norm (RMSNorm over head_dim=128),
RoPE (theta=1e6), GQA (16 q-heads, 8 kv-heads), SwiGLU (silu). Last-token pool.

Fused GEMM plan (4 distinct shapes, fits amdxdna 4-context limit):
  qkv_fused : [1024 -> 4096]  (q=2048, k=1024, v=1024)
  o_proj    : [2048 -> 1024]
  gate_up   : [1024 -> 6144]  (gate=3072, up=3072)
  down_proj : [3072 -> 1024]

backend: callable (A_bf16[M,K], WT_bf16[K,N]) -> bf16 [M,N]  (NPU)
"""
from __future__ import annotations
import numpy as np, torch
from ml_dtypes import bfloat16
from safetensors import safe_open

H = 1024; NL = 28; NQ = 16; NKV = 8; HD = 128; INTER = 3072; EPS = 1e-6; THETA = 1e6


def _t2n(t): return t.view(torch.int16).numpy().view(bfloat16)
def _n2t(a): return torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).view(torch.bfloat16)


class Linear:
    """y = x @ W^T  (no bias; Qwen3 attention_bias=False, MLP no bias). bf16."""
    def __init__(self, W_np, backend):
        self.WT = np.ascontiguousarray(W_np.astype(np.float32).astype(bfloat16).T)
        self.out_dim = W_np.shape[0]; self.backend = backend
    def __call__(self, x):  # x torch bf16 [*,in] -> torch bf16 [*,out]
        lead = x.shape[:-1]
        A = _t2n(x.reshape(-1, x.shape[-1]))
        return _n2t(self.backend(A, self.WT)).reshape(*lead, -1)


class FusedLinear2:
    """Two weights concatenated -> one GEMM, split after. (qkv, gate_up)"""
    def __init__(self, Wa, Wb, backend):
        self.WT = np.ascontiguousarray(np.concatenate([Wa, Wb], 0).astype(np.float32).astype(bfloat16).T)
        self.split = Wa.shape[0]; self.backend = backend
    def __call__(self, x):
        lead = x.shape[:-1]
        out = _n2t(self.backend(_t2n(x.reshape(-1, x.shape[-1])), self.WT)).reshape(*lead, -1)
        return out[..., :self.split], out[..., self.split:]


def rmsnorm(x, w, eps=EPS):
    o = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * w.float()
    return o.to(x.dtype)


def _rope_cos_sin(S, theta=THETA, head_dim=HD):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(S).float()
    freqs = torch.outer(pos, inv)                          # [S, hd/2]
    return torch.cos(freqs), torch.sin(freqs)              # [S, hd/2]

def apply_rope(x, cos, sin):
    # x: [B, S, n_heads, hd]; cos,sin: [S, hd/2]
    cos = cos[None, :, None, :]; sin = sin[None, :, None, :]   # [1,S,1,hd/2]
    x1 = x[..., :x.shape[-1]//2]; x2 = x[..., x.shape[-1]//2:]
    rot = torch.cat([-x2, x1], -1)
    cos_full = torch.cat([cos, cos], -1); sin_full = torch.cat([sin, sin], -1)  # [1,S,1,hd]
    return x * cos_full + rot * sin_full


def build_model(W, backend):
    layers = []
    for i in range(NL):
        qkv = np.concatenate([W[f"layers.{i}.self_attn.q_proj.weight"],
                              W[f"layers.{i}.self_attn.k_proj.weight"],
                              W[f"layers.{i}.self_attn.v_proj.weight"]], 0)  # [4096,1024]
        gu = np.concatenate([W[f"layers.{i}.mlp.gate_proj.weight"],
                             W[f"layers.{i}.mlp.up_proj.weight"]], 0)      # [6144,1024]
        layers.append({
            "qkv": _FusedLinear(qkv, backend, [2048,1024,1024]),
            "o": Linear(W[f"layers.{i}.self_attn.o_proj.weight"], backend),
            "gate_up": _FusedLinear(gu, backend, [3072,3072]),
            "down": Linear(W[f"layers.{i}.mlp.down_proj.weight"], backend),
            "in_ln": torch.from_numpy(W[f"layers.{i}.input_layernorm.weight"].astype(np.float32)).bfloat16(),
            "post_ln": torch.from_numpy(W[f"layers.{i}.post_attention_layernorm.weight"].astype(np.float32)).bfloat16(),
            "q_norm": torch.from_numpy(W[f"layers.{i}.self_attn.q_norm.weight"].astype(np.float32)).bfloat16(),
            "k_norm": torch.from_numpy(W[f"layers.{i}.self_attn.k_norm.weight"].astype(np.float32)).bfloat16(),
        })
    return {
        "embed": torch.from_numpy(W["embed_tokens.weight"].astype(np.float32)).bfloat16(),
        "final_ln": torch.from_numpy(W["norm.weight"].astype(np.float32)).bfloat16(),
        "layers": layers,
    }


def _cat3(W, i):
    return None  # kept for compat; unused


class _FusedLinear:
    """N-way concatenated weights -> one GEMM, split into parts after."""
    def __init__(self, W, backend, splits):
        self.WT = np.ascontiguousarray(W.astype(np.float32).astype(bfloat16).T)
        self.splits = splits; self.backend = backend
    def __call__(self, x):
        lead = x.shape[:-1]
        out = _n2t(self.backend(_t2n(x.reshape(-1, x.shape[-1])), self.WT)).reshape(*lead, -1)
        idx = np.cumsum(self.splits)[:-1]
        return torch.split(out, list(self.splits), dim=-1)


def load_weights(path):
    W = {}
    with safe_open(path, framework="numpy") as f:
        for k in f.keys(): W[k] = f.get_tensor(k)
    return W


def forward(model, ids, mask, backend, weights):
    B, S = ids.shape
    ids_t = torch.from_numpy(ids); mask_t = torch.from_numpy(mask)
    cos, sin = _rope_cos_sin(S)
    h = model["embed"][ids_t]
    ext = mask_t[:, None, None, :].to(torch.bfloat16)
    causal = torch.triu(torch.full((S, S), -1e4, dtype=torch.bfloat16), diagonal=1)  # causal mask
    sqrt_hd = float(np.sqrt(HD))
    for ly in model["layers"]:
        h_in = rmsnorm(h, ly["in_ln"])
        q, k, v = ly["qkv"](h_in)                          # [B,S,2048],[B,S,1024]x2
        q = q.reshape(B, S, NQ, HD); k = k.reshape(B, S, NKV, HD); v = v.reshape(B, S, NKV, HD)
        q = rmsnorm(q, ly["q_norm"]); k = rmsnorm(k, ly["k_norm"])
        q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
        q = q.permute(0,2,1,3)                             # [B,NQ,S,HD]
        k = k.permute(0,2,1,3); v = v.permute(0,2,1,3)     # [B,NKV,S,HD]
        # GQA: repeat KV
        rep = NQ // NKV
        k = k.repeat_interleave(rep, 1); v = v.repeat_interleave(rep, 1)
        scores = (q.float() @ k.float().transpose(-2,-1)) / sqrt_hd
        scores = scores + causal + (1.0 - ext) * -1e4
        attn = torch.softmax(scores, -1).to(torch.bfloat16)
        ctx = (attn @ v).permute(0,2,1,3).reshape(B, S, NQ*HD)   # [B,S,2048]
        h = h + ly["o"](ctx)
        h2 = rmsnorm(h, ly["post_ln"])
        gate, up = ly["gate_up"](h2)
        mlp = ly["down"](torch.nn.functional.silu(gate) * up)
        h = h + mlp
    h = rmsnorm(h, model["final_ln"])
    # last-token pool
    sl = mask_t.sum(1) - 1
    pooled = h[torch.arange(B), sl].float()
    return torch.nn.functional.normalize(pooled, dim=-1).numpy()


def load_model(weights_path, backend):
    return build_model(load_weights(weights_path), backend)
