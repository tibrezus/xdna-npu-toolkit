"""bf16_backend.py — pooled bf16 NPU GEMM backend (O1).

Routes a bf16 Linear's GEMM to the right compiled xclbin by (K,N), pooled BOs.
This is the bf16 sibling of pooled_backend.MultiMBackend. No quant/dequant —
activations stay bf16 end to end.
"""
from __future__ import annotations
import os
import numpy as np
from ml_dtypes import bfloat16
from fast_kernel import FastNpuKernel

# (K, N) -> compiled design name
SHAPES = {
    (384, 384): "qkv",
    (384, 1152): "qkvfused",
    (384, 1536): "ffn1",
    (1536, 384): "ffn2",
}

# M -> directory of compiled bf16 xclbins
ROOTS = {
    512: "/tmp/iron/minilm-gemms-bf16-M512",
    1024: "/tmp/iron/minilm-gemms-bf16-M1024",
    2048: "/tmp/iron/minilm-gemms-bf16-M2048",
    4096: "/tmp/iron/minilm-gemms-bf16-M4096",
    8192: "/tmp/iron/minilm-gemms-bf16-M8192",
}


class Bf16Backend:
    """Pooled bf16 GEMM pool. run(A_i16view, B_i16view) -> bf16 [M,N].

    Inputs A:[M,K] bf16 numpy, B:[K,N] bf16 numpy. Returns bf16 numpy [M,N].
    Kernels cached per (M,K,N). Weight passed per call (resident-weights = O5,
    a follow-up — needs more device contexts).
    """
    def __init__(self):
        self._kernels = {}

    def _get(self, M, K, N):
        key = (M, K, N)
        if key not in self._kernels:
            name = SHAPES.get((K, N))
            root = ROOTS.get(M)
            if name is None or root is None:
                raise KeyError(f"no compiled bf16 GEMM for M={M} K={K} N={N}")
            self._kernels[key] = FastNpuKernel(
                f"{root}/{name}.xclbin", f"{root}/{name}.insts.txt",
                M, K, N, dtype_in=bfloat16, dtype_out=bfloat16)
        return self._kernels[key]

    def run(self, A, B):
        """A: bf16 [M,K], B: bf16 [K,N] -> bf16 [M,N]."""
        M, K = A.shape
        N = B.shape[1]
        return self._get(M, K, N).run(A, B)
