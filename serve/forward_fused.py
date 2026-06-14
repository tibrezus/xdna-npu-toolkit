"""forward_fused.py — QKV-fused forward (fewer dispatches).

Concatenate Q/K/V weights -> one [1152,384] GEMM -> split into q,k,v.
Reduces 36 dispatches -> 24, and quantizes x once for q,k,v instead of 3x.
"""
from __future__ import annotations
import numpy as np
import torch
from safetensors.numpy import load_file
from forward_fast import FastLinear, _ln, N_LAYERS, HIDDEN, N_HEADS, HEAD_DIM, EPS


class QKVFusedLinear:
    """One 384->1152 GEMM producing [q;k;v] concatenated. Splits after."""
    def __init__(self, Wq, bq, Wk, bk, Wv, bv, backend):
        W = np.concatenate([Wq, Wk, Wv], axis=0)   # [1152, 384]
        b = np.concatenate([bq, bk, bv], axis=0)   # [1152]
        amax = np.abs(W).max(axis=1)
        wscale = amax / 32767.0; wscale[wscale == 0] = 1.0
        self.WqT = np.round(W / wscale[:, None]).astype(np.int16).T.copy()  # [384,1152]
        self.wscale = torch.from_numpy(wscale.astype(np.float32))
        self.b = torch.from_numpy(b.astype(np.float32))
        self.backend = backend

    def __call__(self, x):  # x: torch [*,384] -> (q,k,v) each [*,384]
        lead = x.shape[:-1]
        xf_np = x.reshape(-1, x.shape[-1]).numpy()
        amax = np.abs(xf_np).max()
        xscale = np.float32(amax / 32767.0 if amax > 0 else 1.0)
        xq = np.round(xf_np / xscale).astype(np.int16)
        acc = self.backend(xq, self.WqT)                                  # [M,1152]
        out = torch.from_numpy(np.ascontiguousarray(acc)).float() * (xscale * self.wscale[None,:]) + self.b[None,:]
        out = out.reshape(*lead, 1152)
        return out[..., :384], out[..., 384:768], out[..., 768:]


def build_fused_model(weights_dir, backend):
    W = load_file(f"{weights_dir}/model.safetensors")
    def L(n): return FastLinear(W[f"{n}.weight"], W[f"{n}.bias"], backend)
    return {
        "word_emb": torch.from_numpy(W["embeddings.word_embeddings.weight"]),
        "pos_emb": torch.from_numpy(W["embeddings.position_embeddings.weight"]),
        "tok_emb": torch.from_numpy(W["embeddings.token_type_embeddings.weight"]),
        "emb_ln_w": W["embeddings.LayerNorm.weight"], "emb_ln_b": W["embeddings.LayerNorm.bias"],
        "layers": [{
            "qkv": QKVFusedLinear(
                W[f"encoder.layer.{i}.attention.self.query.weight"], W[f"encoder.layer.{i}.attention.self.query.bias"],
                W[f"encoder.layer.{i}.attention.self.key.weight"], W[f"encoder.layer.{i}.attention.self.key.bias"],
                W[f"encoder.layer.{i}.attention.self.value.weight"], W[f"encoder.layer.{i}.attention.self.value.bias"],
                backend),
            "o": L(f"encoder.layer.{i}.attention.output.dense"),
            "aln_w": W[f"encoder.layer.{i}.attention.output.LayerNorm.weight"],
            "aln_b": W[f"encoder.layer.{i}.attention.output.LayerNorm.bias"],
            "f1": L(f"encoder.layer.{i}.intermediate.dense"),
            "f2": L(f"encoder.layer.{i}.output.dense"),
            "oln_w": W[f"encoder.layer.{i}.output.LayerNorm.weight"],
            "oln_b": W[f"encoder.layer.{i}.output.LayerNorm.bias"],
        } for i in range(N_LAYERS)],
    }


def forward_fused(model, ids, mask, backend):
    B, S = ids.shape
    ids_t = torch.from_numpy(ids); mask_t = torch.from_numpy(mask)
    x = model["word_emb"][ids_t] + model["pos_emb"][torch.arange(S)[None]] + model["tok_emb"][torch.zeros_like(ids_t)]
    x = _ln(x, model["emb_ln_w"], model["emb_ln_b"])
    ext = mask_t[:, None, None, :].float(); sqrt_hd = float(np.sqrt(HEAD_DIM))
    for ly in model["layers"]:
        q, k, v = ly["qkv"](x)                       # ONE fused dispatch (was 3)
        def sp(t): return t.reshape(B, S, N_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        q, k, v = sp(q), sp(k), sp(v)
        scores = (q @ k.transpose(-2, -1)) / sqrt_hd + (1.0 - ext) * -1e9
        attn = torch.softmax(scores, -1)
        ctx = (attn @ v).permute(0, 2, 1, 3).reshape(B, S, HIDDEN)
        ctx = ly["o"](ctx)
        x = _ln(x + ctx, ly["aln_w"], ly["aln_b"])
        h = ly["f2"](torch.nn.functional.gelu(ly["f1"](x)))
        x = _ln(x + h, ly["oln_w"], ly["oln_b"])
    m = mask_t[:, :, None].float()
    pooled = (x * m).sum(1) / m.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, dim=1).numpy()
