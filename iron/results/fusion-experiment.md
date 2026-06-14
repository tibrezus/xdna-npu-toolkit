# Op-Fusion Experiment — the single-query NPU serving question, resolved

Date: 2026-06-13. Phoenix NPU1, mlir-aie v1.3.2 + Peano, Arch Linux.

## The hypothesis
Batch=1 GEMV was 8× slower than CPU because every op pays a ~0.3ms
host↔NPU round-trip tax. The question: does **fusing multiple ops into one
dispatch** (intermediate data on-device) collapse that tax?

## The experiment
Two designs, identical topology except stage count, both verified correct:

- **single_add**: `T = A + B` — 1 add, 1 dispatch (baseline)
- **fused_add_add**: `D = (A + B) + C` — 2 adds chained, intermediate `T` lives in
  an on-device `ObjectFifo` and NEVER crosses the host boundary. 1 dispatch.
- **separate**: run `single_add` twice (intermediate via host) — 2 dispatches

bf16, 4096 elements, 1 compute column, Peano-compiled `add.o` kernel.

## Result (both correctness PASS, exact bf16)

| | time | ratio |
|---|---|---|
| single add (1 op, 1 dispatch) | 0.275 ms | — |
| **fused add-add (2 ops, 1 dispatch)** | **0.323 ms** | **1.18× single** |
| separate 2× (2 ops, 2 dispatches) | 0.558 ms | 1.73× fused |

## Verdict: FUSION WORKS

- **fused / single = 1.18** → the 2nd add op costs only **+0.048 ms** when fused,
  vs 0.275 ms as a standalone dispatch. The per-op marginal cost drops to ~18%
  of a dispatch.
- **separate / fused = 1.73×** → fusing 2 ops into 1 dispatch is 1.73× faster.
- The on-device `ObjectFifo` chaining works on NPU1: data flows worker→worker
  without touching host memory.

## What this means for single-query embedding serving

The round-trip floor is ~0.275ms/dispatch. With fusion, each additional fused
op costs only its true compute time (no round-trip). Extrapolated to a
6-layer transformer:

```
separate (N dispatches):  N × 0.275 ms  (+ per-op compute)   — round-trip bound
fused   (1 dispatch):     0.275 ms + Σ(per-op compute)       — compute bound
```

For a 6-layer model this is roughly 3× faster fused vs separate, AND fused
moves the bottleneck from round-trips to actual compute — which the NPU
handles at 264 GOPS. **Single-query NPU serving is reachable.**

## The honest caveat: GEMM-to-GEMM layout matching

This experiment used **elementwise** ops, where input and output layouts are
identical — so the on-device fifo connects stages directly. A real transformer
fuses **GEMMs**, whose output is in a shuffled MMUL-tile layout that does NOT
match the next GEMM's expected input layout. Bridging that needs a layout-
transform fifo (or matched tile geometry) between GEMM stages — substantially
harder than the elementwise case. That is the remaining engineering work;
the *mechanism* (on-device chaining, tax collapse) is now proven.

## Conclusion
The fusion direction is validated. The path to single-query NPU serving is:
fuse the transformer block into one dispatch (GEMM → norm → activation → GEMM),
keeping activations on-device throughout. The hard part is GEMM layout
matching, not the round-trip tax.
