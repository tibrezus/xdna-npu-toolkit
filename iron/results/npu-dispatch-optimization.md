# NPU dispatch-overhead optimization: pooled BOs + pipelined submission

Date: 2026-10-31. Machine: Ryzen 7 7840HS, Phoenix NPU1, fw 1.5.5.391, XRT 2.21.75.
Design: `gemv-288` (int16 GEMV, batch=1, 1 core). All results **exact vs numpy** (i32).

## Results

| path | ms/op | GOPS | vs baseline |
|---|---|---|---|
| baseline (per-call BO alloc + both syncs) | 0.479 | 0.35 | 1× |
| pooled BOs — `run()` | 0.223 | 0.74 | **2.1×** |
| pipelined, 50 execs in flight — `run_many()` | 0.120 | 1.38 | **4.0×** |

## Findings

- Per-call `pyxrt.bo()` allocation is the single largest dispatch cost on amdxdna;
  pooling buffers removes it (~0.25 ms saved per op).
- BO syncs on `host_only` buffers are nearly free (~20 µs combined; results stay
  exact even with syncs skipped) — not worth chasing.
- Pipelined submission (enqueue N execs, wait once) drops the per-op floor to
  **~0.12 ms** — the current driver-level dispatch wall (XRT 2.21.75).
- Even at the floor, batch=1 GEMV remains ~8× behind CPU numpy. Dispatch
  engineering cannot rescue batch=1; batching→GEMM (922 GOPS bf16 at M=4096) or
  whole-block fusion remains the only path to NPU wins (see ARCHITECTURE.md).

## Implementation

`iron/designs/npu_kernel.py`: `run()` now uses pooled BOs (2.1×, drop-in);
`run_many(jobs, out_size)` adds pipelined execution with per-slot buffers
(safe against in-flight overwrite; verified with distinct inputs per job).
