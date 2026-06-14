"""fast_kernel.py — pooled-BO kernel runner for the NPU (the perf fix).

NpuKernel.run() allocates a fresh pyxrt.bo() + .map() per call — 36x/forward.
That alloc/map is ~0.4ms/call = ~14ms/forward, and worse under contention.

FastNpuKernel pre-allocates input + output BOs ONCE (sized to the compiled shape)
and reuses them: each run() just copies new data in + sync + run + sync back.
No per-call allocation. This is the standard pattern for high-throughput serving.

API matches NpuKernel for drop-in use: kern.run(A_i16, B_i16) -> int32 [M,N].
"""
from __future__ import annotations
import numpy as np
import pyxrt


class FastNpuKernel:
    def __init__(self, xclbin_path, insts_path, M, K, N, dtype_in=np.int16, dtype_out=np.int32):
        self.M, self.K, self.N = M, K, N
        self._dtype_in, self._dtype_out = dtype_in, dtype_out
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

        # POOLED input/output BOs (the win): allocate once, reuse
        HO = pyxrt.bo.flags.host_only
        self._in0 = pyxrt.bo(self._dev, M*K*np.dtype(dtype_in).itemsize, HO, self._kern.group_id(3))
        self._in1 = pyxrt.bo(self._dev, K*N*np.dtype(dtype_in).itemsize, HO, self._kern.group_id(4))
        self._out = pyxrt.bo(self._dev, M*N*np.dtype(dtype_out).itemsize, HO, self._kern.group_id(5))
        self._a = np.frombuffer(self._in0.map(), dtype=dtype_in)   # host view (zero-copy)
        self._b = np.frombuffer(self._in1.map(), dtype=dtype_in)
        self._c = np.frombuffer(self._out.map(), dtype=dtype_out)
        TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        FR = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        self._TO, self._FR = TO, FR
        self._in0_sz = M*K*np.dtype(dtype_in).itemsize
        self._in1_sz = K*N*np.dtype(dtype_in).itemsize
        self._out_sz = M*N*np.dtype(dtype_out).itemsize

    def run(self, A, B):
        """A: [M,K] i16, B: [K,N] i16 -> int32 [M,N]. Reuses pooled BOs."""
        self._a[:] = A.reshape(-1)
        self._b[:] = B.reshape(-1)
        self._in0.sync(self._TO, self._in0_sz, 0)
        self._in1.sync(self._TO, self._in1_sz, 0)
        h = self._kern(3, self._insts_bo, self._insts_bytes, self._in0, self._in1, self._out, 0, 0)
        st = h.wait()
        if str(st) != str(pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED):
            raise RuntimeError(f"kernel failed: {st}")
        self._out.sync(self._FR, self._out_sz, 0)
        return self._c.copy().reshape(self.M, self.N)


if __name__ == "__main__":
    import sys, time
    # A/B: pooled vs alloc-per-call, single GEMM
    sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
    from npu_kernel import NpuKernel
    path = "/tmp/iron/minilm-gemms/qkv-4col.xclbin"
    insts = "/tmp/iron/minilm-gemms/qkv-4col.insts.txt"
    A = np.random.randint(-100,100,(512,384),np.int16); B = np.random.randint(-100,100,(384,384),np.int16)

    k_old = NpuKernel(path, insts)
    k_new = FastNpuKernel(path, insts, 512, 384, 384)
    o_old = k_old.run(A,B,out_sizes=[512*384*4])[0].reshape(512,384)
    o_new = k_new.run(A,B)
    print("correctness (pooled==alloc):", "PASS" if np.array_equal(o_old,o_new) else "FAIL")

    def bench(k, fn, n=200):
        for _ in range(20): fn(k)
        t0=time.time()
        for _ in range(n): fn(k)
        return (time.time()-t0)/n*1000
    t_old = bench(k_old, lambda k: k.run(A,B,out_sizes=[512*384*4]))
    t_new = bench(k_new, lambda k: k.run(A,B))
    print(f"alloc-per-call: {t_old:.3f} ms")
    print(f"pooled BOs:     {t_new:.3f} ms  ({t_old/t_new:.2f}x faster)")
