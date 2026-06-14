"""forward_bf16.py — bf16 MiniLM-L6-v2 forward (NPU GEMMs, torch glue). O1.

The bf16 path: no quantize/dequantize. Activations are bf16 end to end.
  - Linear GEMMs run on the NPU as bf16 x bf16 -> bf16 (922 GOPS, 1.2x over i16).
  - Glue (layernorm/softmax/gelu/attention) runs on torch (bf16 tensors; torch
    upcasts internally for accuracy).

torch bf16 <-> numpy bf16 (ml_dtypes) at each Linear boundary is a zero-copy
int16-bit-view round-trip (verified bit-identical).

QKV is fused: one 384->1152 GEMM, then split.
"""
from __future__ import annotations
import numpy as np
import torch
from ml_dtypes import bfloat16
from safetensors import safe_open

N_LAYERS, HIDDEN, N_HEADS, HEAD_DIM, FFN = 6, 384, 12, 32, 1536
EPS = 1e-12


def _t2n(t):
    """torch bf16 -> numpy bf16, zero-copy."""
    return t.view(torch.int16).numpy().view(bfloat16)

def _n2t(a):
    """numpy bf16 -> torch bf16, zero-copy."""
    return torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).view(torch.bfloat16)


def _ln(x, w, b):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), w, b, EPS)


class Bf16Linear:
    """y = x @ W^T + b, GEMM on the NPU (bf16). x: torch bf16 -> torch bf16."""
    def __init__(self, Wf, bf, backend):
        # weight stored as bf16 numpy [in, out] (transposed) for the NPU GEMM
        self.WT = np.ascontiguousarray(
            Wf.astype(np.float32).astype(bfloat16).T)            # [in, out] bf16
        self.b = torch.from_numpy(np.ascontiguousarray(bf.astype(np.float32))).bfloat16()
        self.out_dim = Wf.shape[0]
        self.backend = backend

    def __call__(self, x):   # x: torch bf16 [*, in]
        lead = x.shape[:-1]
        A = _t2n(x.reshape(-1, x.shape[-1]))                      # [M, in] bf16 numpy
        out = self.backend(A, self.WT)                            # [M, out] bf16 numpy
        return _n2t(out).reshape(*lead, -1) + self.b


class Bf16QKV:
    """Fused 384->1152 GEMM producing [q;k;v]. x: torch bf16 -> (q,k,v) bf16."""
    def __init__(self, Wq, bq, Wk, bk, Wv, bv, backend):
        W = np.concatenate([Wq, Wk, Wv], axis=0)                  # [1152, 384]
        b = np.concatenate([bq, bk, bv], axis=0)                  # [1152]
        self.WT = np.ascontiguousarray(W.astype(np.float32).astype(bfloat16).T)  # [384,1152]
        self.b = torch.from_numpy(np.ascontiguousarray(b.astype(np.float32))).bfloat16()
        self.backend = backend

    def __call__(self, x):
        lead = x.shape[:-1]
        A = _t2n(x.reshape(-1, x.shape[-1]))
        out = self.backend(A, self.WT)                            # [M,1152] bf16
        t = _n2t(out).reshape(*lead, 1152) + self.b
        return t[..., :384], t[..., 384:768], t[..., 768:]


def build_bf16_model(weights_dir, backend):
    W = {}
    with safe_open(f"{weights_dir}/model.safetensors", framework="numpy") as f:
        for k in f.keys():
            W[k] = f.get_tensor(k)
    def L(n): return Bf16Linear(W[f"{n}.weight"], W[f"{n}.bias"], backend)
    return {
        "word_emb": torch.from_numpy(W["embeddings.word_embeddings.weight"].astype(np.float32)).bfloat16(),
        "pos_emb": torch.from_numpy(W["embeddings.position_embeddings.weight"].astype(np.float32)).bfloat16(),
        "tok_emb": torch.from_numpy(W["embeddings.token_type_embeddings.weight"].astype(np.float32)).bfloat16(),
        "emb_ln_w": torch.from_numpy(W["embeddings.LayerNorm.weight"].astype(np.float32)).bfloat16(),
        "emb_ln_b": torch.from_numpy(W["embeddings.LayerNorm.bias"].astype(np.float32)).bfloat16(),
        "layers": [{
            "qkv": Bf16QKV(
                W[f"encoder.layer.{i}.attention.self.query.weight"], W[f"encoder.layer.{i}.attention.self.query.bias"],
                W[f"encoder.layer.{i}.attention.self.key.weight"], W[f"encoder.layer.{i}.attention.self.key.bias"],
                W[f"encoder.layer.{i}.attention.self.value.weight"], W[f"encoder.layer.{i}.attention.self.value.bias"],
                backend),
            "o": L(f"encoder.layer.{i}.attention.output.dense"),
            "aln_w": torch.from_numpy(W[f"encoder.layer.{i}.attention.output.LayerNorm.weight"].astype(np.float32)).bfloat16(),
            "aln_b": torch.from_numpy(W[f"encoder.layer.{i}.attention.output.LayerNorm.bias"].astype(np.float32)).bfloat16(),
            "f1": L(f"encoder.layer.{i}.intermediate.dense"),
            "f2": L(f"encoder.layer.{i}.output.dense"),
            "oln_w": torch.from_numpy(W[f"encoder.layer.{i}.output.LayerNorm.weight"].astype(np.float32)).bfloat16(),
            "oln_b": torch.from_numpy(W[f"encoder.layer.{i}.output.LayerNorm.bias"].astype(np.float32)).bfloat16(),
        } for i in range(N_LAYERS)],
    }


def forward_bf16(model, ids, mask, backend):
    B, S = ids.shape
    ids_t = torch.from_numpy(ids); mask_t = torch.from_numpy(mask)
    x = model["word_emb"][ids_t] + model["pos_emb"][torch.arange(S)[None]] \
        + model["tok_emb"][torch.zeros_like(ids_t)]
    x = _ln(x, model["emb_ln_w"], model["emb_ln_b"])
    ext = mask_t[:, None, None, :].to(torch.bfloat16)
    sqrt_hd = float(np.sqrt(HEAD_DIM))
    for ly in model["layers"]:
        q, k, v = ly["qkv"](x)                                     # ONE fused GEMM
        def sp(t): return t.reshape(B, S, N_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        q, k, v = sp(q), sp(k), sp(v)
        scores = (q @ k.transpose(-2, -1)) / sqrt_hd + (1.0 - ext) * -1e4   # bf16: -1e9 clamps
        attn = torch.softmax(scores.float(), -1).to(torch.bfloat16)
        ctx = (attn @ v).permute(0, 2, 1, 3).reshape(B, S, HIDDEN)
        ctx = ly["o"](ctx)
        x = _ln(x + ctx, ly["aln_w"], ly["aln_b"])
        h = ly["f2"](torch.nn.functional.gelu(ly["f1"](x)))
        x = _ln(x + h, ly["oln_w"], ly["oln_b"])
    m = mask_t[:, :, None].to(torch.bfloat16)
    pooled = (x * m).sum(1) / m.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled.float(), dim=1).numpy()
