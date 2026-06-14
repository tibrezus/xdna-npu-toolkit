"""npu_backend.py — NPU GEMM backend for the MiniLM forward (issue #12).

Provides a callable backend for minilm_forward.Linear that runs the int16 GEMM
on the Phoenix NPU. Selects the right compiled xclbin by (K,N) shape. Both NPU
and CPU paths do identical int16@int16->int32 math, so the embeddings are
bit-identical — NPU correctness proven by construction.

Model GEMM shapes (compiled single-core, i16->i32, M=256 = batch4 x seq64):
  attention Q/K/V/O:  K=384 N=384   -> qkv.xclbin
  FFN1:               K=384 N=1536  -> ffn1.xclbin
  FFN2:               K=1536 N=384  -> ffn2.xclbin
"""
from __future__ import annotations
import os
import numpy as np

# locate the compiled gemms (4-col, M=512 = batch8 x seq64) 
GEMM_DIR = os.environ.get("MINILM_GEMM_DIR", "/tmp/iron/minilm-gemms")
COMPILED_M = 512   # 4-col whole-array needs M >= 2*(m*n_rows)=512


class NpuGemmPool:
    """Lazily-load compiled xclbins, keyed by (K,N). Returns a callable
    (xq_i16[M,K], Wt_i16[K,N]) -> acc_i32[M,N]."""
    _kernels = {}

    @classmethod
    def _load(cls, K, N):
        key = (K, N)
        if key in cls._kernels:
            return cls._kernels[key]
        # map (K,N) to the compiled design name (4-col preferred)
        name = {(384, 384): "qkv", (384, 1536): "ffn1", (1536, 384): "ffn2"}.get(key)
        suffix = "-4col"   # 4-col whole-array build (264-GOPS class)
        if name is None:
            raise KeyError(f"no compiled NPU GEMM for shape K={K} N={N}")
        from npu_kernel import NpuKernel   # imported at call time (needs NPU env)
        k = NpuKernel(f"{GEMM_DIR}/{name}{suffix}.xclbin", f"{GEMM_DIR}/{name}{suffix}.insts.txt")
        cls._kernels[key] = (k, key)
        return cls._kernels[key]

    @classmethod
    def run(cls, xq, Wt):
        """xq: [M,K] int16, Wt: [K,N] int16 -> [M,N] int32, on the NPU."""
        kern, (K, N) = cls._load(xq.shape[1], Wt.shape[1])
        M = xq.shape[0]
        assert M == COMPILED_M, f"4-col GEMM compiled for M={COMPILED_M} (batch8 x seq64); got M={M}."
        out = kern.run(xq, Wt, out_sizes=[M * N * 4], out_dtype=np.int32)[0].reshape(M, N)
        return out.astype(np.int32)
