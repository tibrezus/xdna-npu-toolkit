"""qwen_backend.py — pooled bf16 NPU GEMM backend for Qwen3-Embedding-0.6B.

4 distinct fused GEMM shapes (fits amdxdna 4-context limit):
  qkv     : [1024 -> 4096]   (m=128 tile)
  o       : [2048 -> 1024]   (m=128 tile)
  gate_up : [1024 -> 6144]   (m=64 tile; m=128 failed NPU lowering)
  down    : [3072 -> 1024]   (m=64 tile)
All bf16->bf16, M=4096 (batch 64 x seq 64).
"""
from __future__ import annotations
import os
import numpy as np
from ml_dtypes import bfloat16
from fast_kernel import FastNpuKernel

SHAPES = {
    (1024, 4096): "qkv",
    (2048, 1024): "o",
    (1024, 6144): "gate_up",
    (3072, 1024): "down",
}
# Compiled Qwen GEMM xclbins under XDNA_HOME/iron/gemms (relocate via XDNA_HOME
# or XDNA_QWEN_ROOT). Default ~/source/NPU.
_XDNA_HOME = os.path.expanduser(os.environ.get("XDNA_HOME", "~/source/NPU"))
ROOT = os.environ.get("XDNA_QWEN_ROOT", os.path.join(_XDNA_HOME, "iron", "gemms", "qwen-gemms-bf16"))


class QwenBf16Backend:
    def __init__(self, M=4096):
        self.M = M; self._kernels = {}

    def _get(self, K, N):
        key = (K, N)
        if key not in self._kernels:
            name = SHAPES.get(key)
            if name is None:
                raise KeyError(f"no compiled qwen GEMM for K={K} N={N}")
            self._kernels[key] = FastNpuKernel(
                f"{ROOT}/{name}.xclbin", f"{ROOT}/{name}.insts.txt",
                self.M, K, N, dtype_in=bfloat16, dtype_out=bfloat16)
        return self._kernels[key]

    def run(self, A, WT):
        M, K = A.shape
        N = WT.shape[1]
        assert M == self.M, f"compiled for M={self.M} (batch64 x seq64); got M={M}"
        return self._get(K, N).run(A, WT)
