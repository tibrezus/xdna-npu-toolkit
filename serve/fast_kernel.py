"""fast_kernel.py — pooled-BO, dtype-generic kernel runner for the NPU.

Three perf wins over NpuKernel.run():
  - POOLED BOs: allocate input/output buffers once, reuse (2x/GEMM).
  - RESIDENT WEIGHT (O5): for a Linear, the weight matrix is STATIC — stage it
    once as a device BO and never copy it again. Only the activation moves per
    dispatch. ~1-3 ms/forward saved.
  - DTYPE-GENERIC: works for i16->i32 AND bf16->bf16 (O1 bf16 path).
  - ASYNC submit()/wait() (O2): kern() is non-blocking; pipeline host work.

Usage:
    # generic: weight supplied every call
    k = FastNpuKernel(xc, ins, M, K, N, dtype_in=np.int16, dtype_out=np.int32)
    out = k.run(A, B)

    # resident weight (Linear): stage once, run(A) only moves activation
    k = FastNpuKernel(xc, ins, M, K, N, dtype_in=bfloat16, dtype_out=bfloat16, weight=W)
    out = k.run(A)
"""
from __future__ import annotations
import numpy as np
import pyxrt

# bfloat16 lives in ml_dtypes; resolve lazily so import never hard-fails
def _bf16():
    from ml_dtypes import bfloat16
    return bfloat16


class FastNpuKernel:
    def __init__(self, xclbin_path, insts_path, M, K, N,
                 dtype_in=np.int16, dtype_out=np.int32, weight=None):
        self.M, self.K, self.N = M, K, N
        self._dtype_in, self._dtype_out = dtype_in, dtype_out
        self._has_weight = weight is not None

        self._dev = pyxrt.device(0)
        xcl = pyxrt.xclbin(xclbin_path)
        self._dev.register_xclbin(xcl)
        ctx = pyxrt.hw_context(self._dev, xcl.get_uuid())
        kname = [k.get_name() for k in xcl.get_kernels()][0]
        self._kern = pyxrt.kernel(ctx, kname)

        # instructions BO (once)
        with open(insts_path, "rb") as f:
            instr = np.frombuffer(f.read(), dtype=np.uint8)
        self._insts_bytes = len(instr)
        self._insts_bo = pyxrt.bo(self._dev, self._insts_bytes,
                                  pyxrt.bo.flags.cacheable, self._kern.group_id(1))
        np.frombuffer(self._insts_bo.map(), dtype=np.uint8)[:] = instr
        self._insts_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                            self._insts_bytes, 0)

        HO = pyxrt.bo.flags.host_only
        TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        FR = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        self._TO, self._FR = TO, FR

        # input activation BO (always moved per dispatch)
        self._in0_sz = M * K * np.dtype(dtype_in).itemsize
        self._in0 = pyxrt.bo(self._dev, self._in0_sz, HO, self._kern.group_id(3))
        self._a = np.frombuffer(self._in0.map(), dtype=dtype_in)

        # weight BO: resident (staged once) OR per-call
        self._in1_sz = K * N * np.dtype(dtype_in).itemsize
        self._in1 = pyxrt.bo(self._dev, self._in1_sz, HO, self._kern.group_id(4))
        self._b = np.frombuffer(self._in1.map(), dtype=dtype_in)
        if self._has_weight:
            self._b[:] = np.ascontiguousarray(weight).reshape(-1)
            self._in1.sync(TO, self._in1_sz, 0)   # stage once, never again

        # output BO
        self._out_sz = M * N * np.dtype(dtype_out).itemsize
        self._out = pyxrt.bo(self._dev, self._out_sz, HO, self._kern.group_id(5))
        self._c = np.frombuffer(self._out.map(), dtype=dtype_out)

    def run(self, A, B=None):
        """A:[M,K] activation -> [M,N] output. B required unless weight resident."""
        self._a[:] = np.ascontiguousarray(A).reshape(-1)
        self._in0.sync(self._TO, self._in0_sz, 0)
        if not self._has_weight:
            if B is None:
                raise ValueError("B required (no resident weight staged)")
            self._b[:] = np.ascontiguousarray(B).reshape(-1)
            self._in1.sync(self._TO, self._in1_sz, 0)
        self._wait(self._kern(3, self._insts_bo, self._insts_bytes,
                              self._in0, self._in1, self._out, 0, 0))
        self._out.sync(self._FR, self._out_sz, 0)
        return self._c.copy().reshape(self.M, self.N)

    def submit(self, A, B=None):
        """Async: start the dispatch, return a run handle. Call wait(h) to collect.

        NOTE: pooled output BO means you must wait() before the next submit()
        (no double-buffering yet). Useful to overlap small host work with NPU compute.
        """
        self._a[:] = np.ascontiguousarray(A).reshape(-1)
        self._in0.sync(self._TO, self._in0_sz, 0)
        if not self._has_weight:
            self._b[:] = np.ascontiguousarray(B).reshape(-1)
            self._in1.sync(self._TO, self._in1_sz, 0)
        return self._kern(3, self._insts_bo, self._insts_bytes,
                          self._in0, self._in1, self._out, 0, 0)

    def collect(self, h):
        """Collect output for a submitted run handle (blocks until done)."""
        self._wait(h)
        self._out.sync(self._FR, self._out_sz, 0)
        return self._c.copy().reshape(self.M, self.N)

    def _wait(self, h):
        st = h.wait()
        if str(st) != str(pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED):
            raise RuntimeError(f"kernel failed: {st}")
