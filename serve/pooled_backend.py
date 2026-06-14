"""pooled_backend.py — FastNpuKernel-based GEMM pool (the perf-optimized backend).

Selects a pooled NPU kernel by (K,N) shape. Compiles once, reuses BOs.
Multiple compiled M values supported (batch routing).
"""
from __future__ import annotations
import os
import numpy as np
from fast_kernel import FastNpuKernel

# shape -> (name). Multiple M variants live in subdirs.
SHAPES = {
    (384, 384): "qkv",
    (384, 1152): "qkvfused",   # QKV-fused
    (384, 1536): "ffn1",
    (1536, 384): "ffn2",
}


class PooledBackend:
    """Caches FastNpuKernel instances per (M,K,N). Call .run(xq, WqT) -> int32."""
    def __init__(self):
        self._kernels = {}   # (M,K,N) -> FastNpuKernel
        self._base = os.environ.get("MINILM_GEMM_DIR", "/tmp/iron/minilm-gemms")

    def _get(self, M, K, N):
        key = (M, K, N)
        if key not in self._kernels:
            name = SHAPES.get((K, N))
            if name is None:
                raise KeyError(f"no compiled GEMM for (K={K},N={N})")
            # try current dir, then M4096 subdir
            xc = f"{self._base}/{name}-4col.xclbin"
            ins = f"{self._base}/{name}-4col.insts.txt"
            self._kernels[key] = FastNpuKernel(xc, ins, M, K, N)
        return self._kernels[key]

    def run(self, xq, WqT):
        M, K = xq.shape
        K2, N = WqT.shape
        assert K == K2, f"K mismatch {K} vs {K2}"
        return self._get(M, K, N).run(xq, WqT)


# directory mapping for different M (batch routing)
class MultiMBackend:
    """Routes to different compiled-M xclbins based on actual M (batch x seq)."""
    def __init__(self):
        self._kernels = {}
        self._roots = {
            512: "/tmp/iron/minilm-gemms",
            1024: "/tmp/iron/minilm-gemms-M1024",
            2048: "/tmp/iron/minilm-gemms-M2048",
            4096: "/tmp/iron/minilm-gemms-M4096",
            8192: "/tmp/iron/minilm-gemms-M8192",
        }
        self._base = os.environ.get("MINILM_GEMM_DIR", "/tmp/iron/minilm-gemms")

    def _pick_M(self, M):
        # smallest compiled M >= requested (pad up). Here: exact match only.
        for cm in sorted(self._roots):
            if cm == M:
                return cm, self._roots[cm]
        return None, None

    def _get(self, M, K, N):
        key = (M, K, N)
        if key not in self._kernels:
            name = SHAPES.get((K, N))
            cm, root = self._pick_M(M)
            if name is None or cm is None:
                raise KeyError(f"no kernel for M={M} K={K} N={N}")
            xc = f"{root}/{name}-4col.xclbin"
            ins = f"{root}/{name}-4col.insts.txt"
            self._kernels[key] = FastNpuKernel(xc, ins, M, K, N)
        return self._kernels[key]

    def run(self, xq, WqT):
        M, K = xq.shape
        N = WqT.shape[1]
        return self._get(M, K, N).run(xq, WqT)
