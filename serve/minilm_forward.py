"""minilm_forward.py — pure-numpy MiniLM-L6-v2 forward pass.

Goal (issue #11): a transformer forward whose GEMMs can run on EITHER the NPU
or CPU with bit-identical results. We keep activations float, and quantize to
int16 *only around each GEMM* (dynamic input scale, static weight scale). The
glue (LayerNorm, softmax, GELU, residuals) stays float on CPU.

  NPU path:  X_f -> [quant int16] -> NPU GEMM int16@int16->int32 -> [dequant float]
  CPU path:  X_f -> [quant int16] -> numpy GEMM int16@int16->int32     -> [dequant float]
Both produce the SAME int32 -> identical floats -> identical embeddings. The NPU
correctness is therefore guaranteed by construction (the int math is exact).

Weight layout note: safetensors stores Linear.weight as [out, in], so a forward
on x[in] is  x @ W.T.
"""
from __future__ import annotations
import numpy as np
from safetensors.numpy import load_file

# ---- architecture (all-MiniLM-L6-v2) ----
N_LAYERS = 6
HIDDEN = 384
N_HEADS = 12
HEAD_DIM = HIDDEN // N_HEADS   # 32
FFN = 1536
MAX_POS = 512
EPS = 1e-12


def gelu(x):
    # BERT uses the exact erf gelu. scipy.special.erf is a fast vectorized C impl
    # (np.vectorize was the slow path — 30x slower, unsuitable for serving).
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def _erf(x):
    # Abramowitz-Stegun 7.1.26 (numpy has no native erf without scipy)
    s = 1.0 if x >= 0 else -1.0
    x = abs(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return s * y


def layernorm(x, gamma, beta, eps=EPS):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * gamma + beta


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def quantize_i16(x_f, scale=None):
    """Dynamic per-tensor int16 quantization. Returns (q_i16, scale)."""
    if scale is None:
        amax = np.abs(x_f).max()
        scale = amax / 32767.0 if amax > 0 else 1.0
    q = np.round(x_f / scale).astype(np.int16)
    return q, scale


def quantize_weight_i16(W_f):
    """Static per-output-channel int16 quantization for a [out,in] weight.
    Returns (q_i16, scale[out]) so dequant = (q@in) * scale[out]."""
    amax = np.abs(W_f).max(axis=1)                      # [out]
    scale = amax / 32767.0
    scale[scale == 0] = 1.0
    q = np.round(W_f / scale[:, None]).astype(np.int16)
    return q, scale


class Linear:
    """A quantized linear layer: y = x @ W^T + b, with int16 GEMM.

    backend: 'cpu' (numpy) or a function (M,K_int16, N_int16)->int32 for NPU.
    Both do identical int16@int16->int32 math, so results match bit-for-bit.
    """
    def __init__(self, W_f, b_f, backend="cpu"):
        self._Wf = W_f.astype(np.float32)
        self.Wq, self.wscale = quantize_weight_i16(W_f)   # [out,in] int16, [out]
        self.b = b_f
        self.out_dim = W_f.shape[0]
        self.backend = backend

    def __call__(self, x_f):
        # x_f: [..., in] float -> flatten leading dims
        lead = x_f.shape[:-1]
        xin = x_f.reshape(-1, x_f.shape[-1])              # [M, in]
        if self.backend == "float":
            # pure float reference (no quantization) — matches transformers
            out = xin @ self._Wf.T + self.b[None, :]
            return out.reshape(*lead, self.out_dim)
        xq, xscale = quantize_i16(xin)                    # [M,in] int16, scalar
        # GEMM: int16 [M,in] @ int16 [in,out] -> int32 [M,out]
        if self.backend == "cpu":
            acc = xq.astype(np.int32) @ self.Wq.T.astype(np.int32)
        else:
            acc = self.backend(xq, self.Wq.T)             # NPU: returns int32 [M,out]
        # dequantize: out = acc * (xscale * wscale[out])  + bias
        out = acc.astype(np.float32) * (xscale * self.wscale[None, :]) + self.b[None, :]
        return out.reshape(*lead, self.out_dim)


def build_model(weights_dir, backend="cpu"):
    W = load_file(f"{weights_dir}/model.safetensors")

    def L(name, backend=backend):
        return Linear(W[f"{name}.weight"], W[f"{name}.bias"], backend)

    model = {
        "word_emb": W["embeddings.word_embeddings.weight"],
        "pos_emb": W["embeddings.position_embeddings.weight"],
        "tok_emb": W["embeddings.token_type_embeddings.weight"],
        "emb_ln_w": W["embeddings.LayerNorm.weight"],
        "emb_ln_b": W["embeddings.LayerNorm.bias"],
        "layers": [
            {
                "q": L(f"encoder.layer.{i}.attention.self.query", backend),
                "k": L(f"encoder.layer.{i}.attention.self.key", backend),
                "v": L(f"encoder.layer.{i}.attention.self.value", backend),
                "o": L(f"encoder.layer.{i}.attention.output.dense", backend),
                "attn_ln_w": W[f"encoder.layer.{i}.attention.output.LayerNorm.weight"],
                "attn_ln_b": W[f"encoder.layer.{i}.attention.output.LayerNorm.bias"],
                "f1": L(f"encoder.layer.{i}.intermediate.dense", backend),
                "f2": L(f"encoder.layer.{i}.output.dense", backend),
                "out_ln_w": W[f"encoder.layer.{i}.output.LayerNorm.weight"],
                "out_ln_b": W[f"encoder.layer.{i}.output.LayerNorm.bias"],
            }
            for i in range(N_LAYERS)
        ],
    }
    return model


def forward(model, input_ids, attention_mask, backend="cpu"):
    """input_ids: [B,S] int. attention_mask: [B,S] (1=keep). Returns [B,HIDDEN]."""
    B, S = input_ids.shape
    pos = np.arange(S)[None, :]
    tok = np.zeros_like(input_ids)
    x = model["word_emb"][input_ids] + model["pos_emb"][pos] + model["tok_emb"][tok]
    x = layernorm(x, model["emb_ln_w"], model["emb_ln_b"])

    ext = attention_mask[:, None, None, :].astype(np.float32)
    for lyr in model["layers"]:
        q = lyr["q"](x); k = lyr["k"](x); v = lyr["v"](x)         # [B,S,384]
        def split(t):
            t = t.reshape(B, S, N_HEADS, HEAD_DIM)
            return t.transpose(0, 2, 1, 3)                          # [B,H,S,dh]
        q, k, v = split(q), split(k), split(v)
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(HEAD_DIM)  # [B,H,S,S]
        scores = scores + (1.0 - ext) * -1e9
        attn = softmax(scores, -1)
        ctx = attn @ v                                               # [B,H,S,dh]
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, S, HIDDEN)
        ctx = lyr["o"](ctx)
        x = layernorm(x + ctx, lyr["attn_ln_w"], lyr["attn_ln_b"])
        h = lyr["f2"](gelu(lyr["f1"](x)))
        x = layernorm(x + h, lyr["out_ln_w"], lyr["out_ln_b"])

    # mean pooling over masked tokens + L2 normalize (sentence-transformers style)
    mask = attention_mask[:, :, None].astype(np.float32)
    summed = (x * mask).sum(1)
    counts = mask.sum(1).clip(min=1e-9)
    pooled = summed / counts
    pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-12)
    return pooled
