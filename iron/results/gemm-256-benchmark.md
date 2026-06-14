# int16 GEMM 256×256×256 on Phoenix NPU — NPU vs CPU

Date: 2026-06-13
Machine: AMD Ryzen 7 7840HS, Phoenix NPU (XDNA1 / AIE2 / NPU1, firmware 1.5.5.391)
Toolchain: mlir-aie v1.3.2 (IRON) + llvm-aie/Peano nightly, Arch Linux, Python 3.14
Design: single_core_iron.py, i16×i16→i32, tile 32×32×32, MMUL (4,4,4), 1 AIE core

## Result
```
NPU (Phoenix, 1 core, Peano):   0.82 ms/op   41.2 GOPS
CPU numpy (multi-thread):      13.70 ms/op    2.4 GOPS
CPU numpy (1 thread):          13.92 ms/op    2.4 GOPS
NPU/MT-CPU speedup:  16.80x
NPU/ST-CPU speedup:  17.08x
```

## Proven execution (strace on /dev/accel/accel0)
```
AMDXDNA_CREATE_HWCTX   1   (created hardware context)
AMDXDNA_CONFIG_HWCTX   1   (loaded xclbin/overlay)
AMDXDNA_CREATE_BO      7   (A, B, C, insts, ...)
AMDXDNA_EXEC_CMD       1   (dispatched GEMM to NPU hardware)
AMDXDNA_DESTROY_HWCTX  1
```
Correctness: exact match vs numpy (i32 accumulator), zero error.

## Notes
- Single-core NPU design. Phoenix has 4 usable compute columns → a 4-column design
  projects to ~4× faster (≈0.2 ms, ≈160 GOPS).
- CPU baseline is numpy int32 matmul (not BLAS — int matmul is generic). A fairer CPU
  comparison for a real embedding pipeline would be the Radeon 780M iGPU or float BLAS;
  the point here is the NPU is genuinely fast at int16 GEMM.
