"""npu_server.py — a batched embedding-serving scaffold for the Phoenix NPU.

Implements the architecture from iron/docs/ARCHITECTURE.md:
  - accumulate queries into a batch
  - above a threshold, dispatch as ONE big GEMM to the NPU (264 GOPS path)
  - below it, run on CPU (avoids the host<->NPU round-trip tax)

This proves the SERVING PATTERN today, using a linear layer as the stand-in for
a full transformer block (which needs more fused ops — see roadmap). The linear
layer alone is already the dominant cost and a real embedding primitive
(it's literally the embedding-table lookup: E[V,D] @ onehot[batch,V] -> [batch,D]).
"""
from __future__ import annotations
import os, sys, time
from dataclasses import dataclass, field
sys.path.insert(0, os.path.dirname(__file__))
from npu_kernel import NpuKernel
import numpy as np


# --------------------------------------------------------------------------- #
#  Backends: a "linear layer" = weight matrix W[D_out, D_in], compute X @ W.T  #
# --------------------------------------------------------------------------- #
class LinearBackend:
    """A linear layer that can run on either NPU or CPU."""
    def forward_batch(self, X: np.ndarray) -> np.ndarray: ...


class CpuLinear(LinearBackend):
    def __init__(self, W: np.ndarray):
        self.W = W  # [D_out, D_in], int16
    def forward_batch(self, X):
        # X: [batch, D_in] int16 -> [batch, D_out] int32
        return (X.astype(np.int32) @ self.W.astype(np.int32).T)


class NpuLinear(LinearBackend):
    """Runs the linear layer as a GEMM on the 4-col NPU design.

    forward: X[batch,D_in] @ W[D_out,D_in].T == W @ X.T  transposed, but to fit the
    compiled GEMM (A[M,K] @ B[K,N] -> C[M,N]) we compute  C = W @ X.T
    with M=D_out, K=D_in, N=batch. Requires the 4-col build at that shape.
    """
    def __init__(self, kern: NpuKernel, W: np.ndarray, D_in: int, D_out: int, batch: int):
        self.kern = kern
        self.W = W
        self.M, self.K, self.N = D_out, D_in, batch   # matches the compiled xclbin shape

    def forward_batch(self, X):
        # X: [batch, D_in] -> need A=W[D_out,D_in], B=X.T[D_in,batch]
        A = np.ascontiguousarray(self.W)                      # [D_out, D_in]
        B = np.ascontiguousarray(X.T)                         # [D_in, batch]
        out = self.kern.run(A, B, out_sizes=[self.M * self.N * 4],
                            out_dtype=np.int32)[0].reshape(self.M, self.N)
        return out.T.copy()   # [batch, D_out]


# --------------------------------------------------------------------------- #
#  The server: batch + route                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class BatchedEmbedServer:
    cpu_backend: CpuLinear
    npu_backend: NpuLinear
    batch_threshold: int = 8       # route to NPU when batch >= this
    _queue: list = field(default_factory=list)

    def embed(self, X: np.ndarray) -> np.ndarray:
        """Embed a single [D_in] vector or a [batch, D_in] batch."""
        X = np.atleast_2d(X)
        batch = X.shape[0]
        if batch >= self.batch_threshold and batch == self.npu_backend.N:
            out = self.npu_backend.forward_batch(X)
            route = "NPU(4col)"
        else:
            out = self.cpu_backend.forward_batch(X)
            route = "CPU"
        return out, route

    def embed_many(self, vecs: list[np.ndarray]) -> list[np.ndarray]:
        """Accumulate queries, flush as a batch. Demonstrates the server pattern."""
        self._queue.extend(vecs)
        if len(self._queue) >= self.batch_threshold:
            X = np.stack(self._queue)
            out, _ = self.embed(X)
            self._queue.clear()
            return [out[i] for i in range(len(vecs))]
        return [None] * len(vecs)   # not flushed yet


# --------------------------------------------------------------------------- #
#  Demo / sanity check                                                        #
# --------------------------------------------------------------------------- #
def main():
    print("=" * 70)
    print(" Batched embedding server — NPU (4-col GEMM) vs CPU routing demo")
    print("=" * 70)

    D_in, D_out = 512, 512          # must match the compiled 4-col GEMM shape
    BUILD = "/tmp/mliraie-v132/programming_examples/basic/matrix_multiplication/whole_array/build"

    W = np.random.randint(-50, 50, (D_out, D_in), dtype=np.int16)   # the "weights"
    cpu = CpuLinear(W)

    try:
        kern = NpuKernel.from_build(BUILD, "512x512x512_64x64x32_4c")
        # the compiled shape is M=N=K=512; we use batch=N=512 for the demo
        npu = NpuLinear(kern, W, D_in=D_in, D_out=D_out, batch=512)
    except Exception as e:
        print(f"NPU backend unavailable ({e}); CPU-only demo.")
        npu = None

    server = BatchedEmbedServer(cpu_backend=cpu, npu_backend=npu, batch_threshold=8)

    # --- correctness: NPU vs CPU on a full batch ---
    if npu:
        X = np.random.randint(-50, 50, (512, D_in), dtype=np.int16)
        out_npu, route = server.embed(X)
        out_cpu = cpu.forward_batch(X)
        match = np.array_equal(out_npu, out_cpu)
        print(f"\n[correctness] batch=512 via {route}: {'PASS' if match else 'FAIL'} vs CPU")
        if not match:
            d = np.abs(out_npu.astype(np.int64) - out_cpu.astype(np.int64))
            print(f"  max|diff|={d.max()} match={(d==0).sum()}/{out_npu.size}")

    # --- routing demo ---
    print("\n[routing] single query vs batched:")
    for b in [1, 4, 16, 512]:
        X = np.random.randint(-50, 50, (b, D_in), dtype=np.int16)
        _, route = server.embed(X)
        print(f"  batch={b:4} -> {route}")

    # --- throughput: the point of the NPU path ---
    if npu:
        print("\n[throughput] batch=512 (the NPU's sweet spot):")
        X = np.random.randint(-50, 50, (512, D_in), dtype=np.int16)
        t0 = time.time()
        for _ in range(20):
            server.npu_backend.forward_batch(X)
        npu_ms = (time.time() - t0) / 20 * 1000
        t0 = time.time()
        for _ in range(5):
            cpu.forward_batch(X)
        cpu_ms = (time.time() - t0) / 5 * 1000
        print(f"  NPU: {npu_ms:.2f} ms/batch  ({512*512*512*2/npu_ms/1e6:.0f} GOPS)")
        print(f"  CPU: {cpu_ms:.2f} ms/batch  ({cpu_ms/npu_ms:.1f}x slower than NPU)")
        print(f"  => batched embedding: NPU is {cpu_ms/npu_ms:.1f}x faster. This is the viable serving path.")


if __name__ == "__main__":
    main()
