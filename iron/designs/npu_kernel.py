"""npu_kernel.py — a thin, reusable runner for compiled IRON/Peano designs on the Phoenix NPU.

This is the execution primitive that the future embedding-serving layer is built on:
load a compiled (xclbin, insts) pair once, then run() it on int16 tensors repeatedly.

Design principle: "easily servable" means the caller never sees XRT. They hand us numpy
arrays and get numpy arrays back.

Usage:
    kern = NpuKernel.from_build(build_dir, shape_tag="512x512x512_64x64x32_4c")
    out = kern.run(A, B)              # A:[M,K] i16, B:[K,N] i16 -> out:[M,N] i32
"""
from __future__ import annotations
import os, re, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pyxrt


def _find_files(build_dir: str, shape_tag: str):
    bd = Path(build_dir)
    xc = list(bd.glob(f"final_{shape_tag}*.xclbin"))
    inst = list(bd.glob(f"insts_{shape_tag}*.txt")) + list(bd.glob(f"*{shape_tag}*/insts.bin"))
    if not xc or not inst:
        raise FileNotFoundError(
            f"no xclbin/insts for tag '{shape_tag}' in {bd}\n"
            f"  xclbins: {[p.name for p in bd.glob('*.xclbin')]}"
        )
    return str(xc[0]), str(inst[0])


@dataclass
class NpuKernel:
    """A loaded, ready-to-run compiled AIE kernel on the NPU."""
    xclbin_path: str
    insts_path: str
    _dev: object = None
    _kern: object = None
    _insts_bo: object = None
    _insts_bytes: int = 0

    @classmethod
    def from_build(cls, build_dir: str, shape_tag: str) -> "NpuKernel":
        xc, inst = _find_files(build_dir, shape_tag)
        return cls(xc, inst)

    def __post_init__(self):
        self._dev = pyxrt.device(0)
        xcl = pyxrt.xclbin(self.xclbin_path)
        self._dev.register_xclbin(xcl)
        ctx = pyxrt.hw_context(self._dev, xcl.get_uuid())
        kname = [k.get_name() for k in xcl.get_kernels()][0]
        self._kern = pyxrt.kernel(ctx, kname)

        with open(self.insts_path, "rb") as f:
            instr = np.frombuffer(f.read(), dtype=np.uint8)
        self._insts_bytes = len(instr)
        bo = pyxrt.bo(self._dev, self._insts_bytes, pyxrt.bo.flags.cacheable, self._kern.group_id(1))
        np.frombuffer(bo.map(), dtype=np.uint8)[:] = instr
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, self._insts_bytes, 0)
        self._insts_bo = bo

    def run(self, *tensors: np.ndarray, out_sizes: list[int] | None = None,
            out_dtype=np.int32) -> list[np.ndarray]:
        """Run the kernel. tensors are inputs (numpy). Returns output array(s).

        out_sizes: byte sizes of each output buffer (default: one output = product of its shape).
        Kernel arg layout (IRON convention): (3, insts_bo, insts_bytes, *in_bos, *out_bos, 0...)
        Inputs occupy group ids 3,4,...; outputs follow. We allocate output BOs and return them.
        """
        HOST_ONLY = pyxrt.bo.flags.host_only
        TO_DEV = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        FROM_DEV = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE

        in_bos = []
        gid = 3
        for t in tensors:
            t = np.ascontiguousarray(t)
            bo = pyxrt.bo(self._dev, t.nbytes, HOST_ONLY, self._kern.group_id(gid))
            np.frombuffer(bo.map(), dtype=t.dtype)[:] = t.reshape(-1)
            bo.sync(TO_DEV, t.nbytes, 0)
            in_bos.append(bo)
            gid += 1

        out_bos = []
        results = []
        for sz in (out_sizes or []):
            bo = pyxrt.bo(self._dev, sz, HOST_ONLY, self._kern.group_id(gid))
            arr = np.frombuffer(bo.map(), dtype=out_dtype)
            out_bos.append(bo)
            results.append(arr)
            gid += 1

        h = self._kern(3, self._insts_bo, self._insts_bytes, *in_bos, *out_bos, 0, 0)
        state = h.wait()
        if str(state) != str(pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED):
            raise RuntimeError(f"NPU kernel did not complete: {state}")

        for bo, arr in zip(out_bos, results):
            bo.sync(FROM_DEV, arr.nbytes, 0)
        return [a.copy() for a in results]


# ---------- benchmark helpers ----------
def bench_gemm(kern: NpuKernel, M, K, N, dtype_in=np.int16, dtype_out=np.int32,
               iters=100, verify=True):
    A = np.random.randint(-100, 100, (M, K), dtype=dtype_in)
    B = np.random.randint(-100, 100, (K, N), dtype=dtype_in)
    ref = (A.astype(np.int32) @ B.astype(np.int32)).astype(dtype_out)

    out_size = M * N * np.dtype(dtype_out).itemsize
    out = kern.run(A, B, out_sizes=[out_size], out_dtype=dtype_out)[0].reshape(M, N)
    if verify and not np.array_equal(out, ref):
        d = np.abs(out.astype(np.int64) - ref.astype(np.int64))
        raise AssertionError(f"GEMM mismatch: max|diff|={d.max()} match={(d==0).sum()}/{M*N}")

    # warmup already done above; benchmark
    t0 = time.time()
    for _ in range(iters):
        kern.run(A, B, out_sizes=[out_size], out_dtype=dtype_out)
    ms = (time.time() - t0) / iters * 1000
    gops = 2 * M * K * N / (ms / 1000) / 1e9
    return ms, gops
