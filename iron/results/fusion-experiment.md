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

---

## Appendix: the GEMM-to-GEMM fusion attempt (and why it's a different order of problem)

I attempted to extend the fusion result to `C = (A@B)@D` with the intermediate
on-device. It does NOT drop out of the elementwise proof. Two distinct obstacles:

### 1. The dtype/requantization wall (the deeper one)
AIE2 i16 MMUL is `i16 × i16 → i32` (accumulator wider than input). So GEMM1's
output T is **i32**, but GEMM2's MMUL input must be **i16**. You cannot feed
T directly into GEMM2 — you need a **requantization op** (rescale i32 → i16)
on-device between the stages. That requant itself has to be fused, and its
scale/zero-point must be computed per the quantization scheme. Real
transformers do exactly this between every linear layer; it's not optional.

This is the single biggest reason "fuse a transformer block by hand in IRON"
is weeks of work, not hours: every GEMM→GEMM boundary needs a fused requant,
and getting the numerics right (vs a float reference) is the hard part.

### 2. Layout matching + runtime dataflow
GEMM1 writes C in MMUL-tile order; GEMM2 expects its A-input in a (different)
MMUL-tile order. Plus the runtime sequence must correctly: fill A,B tiles,
loop over K (accumulation) in GEMM1, stream T tiles to GEMM2, loop over GEMM2's
K (= GEMM1's N), fill D tiles, drain E tiles. Hand-authoring this correctly is
the body of a small compiler.

### Strategic implication
These two obstacles are **precisely what AMD's VitisAI EP (Strategy B, #9)
solves automatically** — it takes an ONNX, inserts the requants, matches
layouts, and emits one fused dispatch. That is the value of the gated toolchain.
Hand-rolling it in IRON (Strategy A) for one model = reinventing that compiler.

### So what's the honest path to servable embeddings?
- **Batched serving**: proven viable TODAY (34.7× over CPU), no fusion needed.
- **Single-query via fusion**: mechanism proven (elementwise), but full-model
  fusion is real compiler work. Two honest routes:
  - (B) VitisAI EP — automatic, gated. Best leverage if the Linux gate lifts.
  - (A-deep) Hand-build the fused transformer in IRON — feasible but weeks of
    careful dataflow+requant work per model.
- **Pragmatic single-query**: route to CPU/iGPU (Radeon 780M). Honest and correct
  given the above.
