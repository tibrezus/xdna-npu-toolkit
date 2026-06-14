# Strategy A — Embeddings on Phoenix (NPU1) via IRON/Peano

> **Status: PROVEN.** The transformer-core op (int16 GEMM) compiles AND executes
> correctly on the Phoenix NPU, fully on Linux, fully open-source. See
> [`results/gemm-256-benchmark.md`](results/gemm-256-benchmark.md) and
> [issue #8](https://github.com/tibrezus/xdna-npu-toolkit/issues/8).

The AMD Ryzen 7 7840HS has a Phoenix NPU (XDNA 1 / AIE-ML / AIE2). AMD's own
**open-source** toolchain — `mlir-aie` (the IRON Python API) and `llvm-aie`
(Peano, the LLVM-based AIE compiler) — targets it natively on Linux with
**no license, no Windows, no AMD account, no Early Access**. This is the
unblocked path to running transformer/embedding kernels on the NPU.

## Why this exists

The VitisAI-EP "ONNX-consuming" compile path is deployment-gated on Linux
(see [issue #5](https://github.com/tibrezus/xdna-npu-toolkit/issues/5) and
[Strategy B, #9](https://github.com/tibrezus/xdna-npu-toolkit/issues/9)).
Strategy A sidesteps that entirely by composing kernels directly in IRON and
compiling them with Peano.

## Quick start (Arch Linux, Phoenix / NPU1)

```bash
# prerequisites: xrt + xrt-plugin-amdxdna (pacman), unlimited memlock via pam_limits
# (run designs in a login shell: `sudo -i -u $USER`)
cd iron && ./bootstrap.sh        # venv py3.14 + mlir_aie 1.3.2 + llvm-aie nightly + mlir-aie repo
source setup-env.sh              # activate env, set PEANO/MLIR_AIE_INSTALL_DIR, XRT paths
```

Build the GEMM design for Phoenix:

```bash
cd mlir-aie/programming_examples/basic/matrix_multiplication/single_core
make all use_iron=1 dtype_in=i16 dtype_out=i32 M=256 K=256 N=256 m=32 k=32 n=32
# → build/final_256x256x256_32x32x32.xclbin  (Peano compiles AIE2 kernel → ELF → xclbin)
```

Run + verify on the NPU:

```bash
python3 ../../../../../../iron/designs/run_gemm.py $(pwd)/build
# PASS!  int16 GEMM 256x256x256 on Phoenix NPU == numpy exactly.
```

Benchmark NPU vs CPU:

```bash
python3 ../../../../../../iron/designs/bench_gemm.py $(pwd)/build
# NPU (1 core): 0.82 ms/op, 41.2 GOPS   |   CPU numpy: 13.70 ms/op, 2.4 GOPS   (16.8x)
```

## Result

int16 GEMM 256×256×256 on Phoenix, single AIE core, verified exact vs numpy:

| | time/op | GOPS |
|---|---|---|
| **NPU (1 core, Peano)** | **0.82 ms** | **41.2** |
| CPU numpy (multi-thread) | 13.70 ms | 2.4 |
| **NPU speedup** | **16.8×** | |

Single-core design. A 4-column Phoenix design projects to ~0.2 ms (~160 GOPS).

## Two non-obvious technical facts

1. **Use the new XRT load path.** `device.load_xclbin(path)` returns `EOPNOTSUPP`
   on amdxdna firmware. The working path is:
   ```python
   xcl = pyxrt.xclbin(path)
   dev.register_xclbin(xcl)
   ctx = pyxrt.hw_context(dev, xcl.get_uuid())
   kern = pyxrt.kernel(ctx, "MLIR_AIE")
   # run: kern(3, insts_bo, insts_bytes, *data_bos)
   ```
2. **Phoenix (NPU1) = 4 compute columns × 4 compute rows = 16 tiles** in IRON
   (one column reserved from the firmware's 5). i16 MMUL geometry is (4,4,4).

## Roadmap to a real embedding model (tracked in issue #8)

The GEMM proves compute. Remaining work is kernel composition — all have IRON
examples — stitched into a transformer:

- [ ] 4-column GEMM (adapt `whole_array` NPU2→NPU1) — ~4× throughput
- [ ] LayerNorm, Softmax, GELU, quantized-embedding-gather kernels
- [ ] Compose one transformer layer (QKV → attention → FFN)
- [ ] MiniLM-L6-v2 equivalent (6 layers, 384-dim, i16) → publish compiled xclbins

## Honest caveats

- Peano is **single-core**; multi-column needs more design work (not a license).
- This is **kernel composition**, not "feed an ONNX" — more engineering, but
  it's the genuine Linux/open-source answer (Strategy B is the lower-effort
  ONNX path if its gate lifts on Linux).
- The CPU baseline above is numpy int32 matmul (generic, not BLAS). A fairer
  comparison for a full pipeline is the Radeon 780M iGPU or float BLAS.
