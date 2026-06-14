"""forward_fast.py — optimized MiniLM-L6-v2 forward (NPU GEMMs + torch glue).

The profiling insight: numpy/scipy element-wise ops are 100-1000x slower than
torch's C kernels (scipy erf GELU = 8.4ms; torch F.gelu = 0.05ms). So the glue
(layernorm/softmax/gelu/attention) runs on torch (zero-copy via from_numpy),
and the Linear GEMMs run on the NPU. This is the legitimate hybrid.

Still int16 around GEMMs (NPU correctness), but the float glue is now fast.

API: build_fast_model(weights_dir) -> dict; forward_fast(model, ids, mask, backend)
backend = a callable (xq_i16, WqT_i16) -> int32 [M,N]  (e.g. NpuGemmPool.run)
"""
from __future__ import annotations
import numpy as np
import torch
from safetensors.numpy import load_file

N_LAYERS, HIDDEN, N_HEADS, HEAD_DIM, FFN, EPS = 6, 384, 12, 32, 1536, 1e-12


class FastLinear:
    """y = x @ W^T + b. GEMM on `backend` (NPU); quantize/dequant around it."""
    def __init__(self, Wf, bf, backend):
        # static per-channel int16 weight quant (cached once)
        amax = np.abs(Wf).max(axis=1)
        wscale = amax / 32767.0; wscale[wscale == 0] = 1.0
        self.WqT = np.round(Wf / wscale[:, None]).astype(np.int16).T.copy()  # [in,out]
        self.wscale = torch.from_numpy(wscale.astype(np.float32))           # [out]
        self.b = torch.from_numpy(bf.astype(np.float32))
        self.backend = backend

    def __call__(self, x):  # x: torch [*, in] float32
        lead = x.shape[:-1]
        xf = x.reshape(-1, x.shape[-1])                                   # [M,in]
        xf_np = xf.numpy() if isinstance(xf, torch.Tensor) else xf
        amax = np.abs(xf_np).max()
        xscale = np.float32(amax / 32767.0 if amax > 0 else 1.0)
        xq = np.round(xf_np / xscale).astype(np.int16)                    # [M,in]
        acc = self.backend(xq, self.WqT)                                  # int32 [M,out]
        acc_t = torch.from_numpy(np.ascontiguousarray(acc)).float()
        out = acc_t * (xscale * self.wscale[None, :]) + self.b[None, :]
        return out.reshape(*lead, -1)


def build_fast_model(weights_dir, backend):
    W = load_file(f"{weights_dir}/model.safetensors")
    def L(n): return FastLinear(W[f"{n}.weight"], W[f"{n}.bias"], backend)
    return {
        "word_emb": torch.from_numpy(W["embeddings.word_embeddings.weight"]),
        "pos_emb": torch.from_numpy(W["embeddings.position_embeddings.weight"]),
        "tok_emb": torch.from_numpy(W["embeddings.token_type_embeddings.weight"]),
        "emb_ln_w": W["embeddings.LayerNorm.weight"], "emb_ln_b": W["embeddings.LayerNorm.bias"],
        "layers": [{
            "q": L(f"encoder.layer.{i}.attention.self.query"),
            "k": L(f"encoder.layer.{i}.attention.self.key"),
            "v": L(f"encoder.layer.{i}.attention.self.value"),
            "o": L(f"encoder.layer.{i}.attention.output.dense"),
            "aln_w": W[f"encoder.layer.{i}.attention.output.LayerNorm.weight"],
            "aln_b": W[f"encoder.layer.{i}.attention.output.LayerNorm.bias"],
            "f1": L(f"encoder.layer.{i}.intermediate.dense"),
            "f2": L(f"encoder.layer.{i}.output.dense"),
            "oln_w": W[f"encoder.layer.{i}.output.LayerNorm.weight"],
            "oln_b": W[f"encoder.layer.{i}.output.LayerNorm.bias"],
        } for i in range(N_LAYERS)],
    }


def _ln(x, w, b):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],),
        torch.from_numpy(w) if isinstance(w, np.ndarray) else w,
        torch.from_numpy(b) if isinstance(b, np.ndarray) else b, EPS)


def forward_fast(model, ids, mask, backend):
    B, S = ids.shape
    ids_t = torch.from_numpy(ids); mask_t = torch.from_numpy(mask)
    pos = torch.arange(S)[None]
    x = model["word_emb"][ids_t] + model["pos_emb"][pos] + model["tok_emb"][torch.zeros_like(ids_t)]
    x = _ln(x, model["emb_ln_w"], model["emb_ln_b"])
    ext = mask_t[:, None, None, :].float()
    sqrt_hd = float(np.sqrt(HEAD_DIM))
    for ly in model["layers"]:
        q = ly["q"](x); k = ly["k"](x); v = ly["v"](x)
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
