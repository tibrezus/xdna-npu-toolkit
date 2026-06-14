# Serving benchmark — the honest verdict (MiniLM-L6-v2 on 7840HS)

The decisive test: does the Phoenix NPU actually beat CPU for serving MiniLM-L6-v2?

## Result (seq=64, i16 NPU path, float CPU paths)

| batch | torch-CPU | numpy-float | NPU-4col hybrid |
|---|---|---|---|
| 8 | **79 ms (9.9/text)** | 153 ms (19.1) | 226 ms (28.2) |
| 16 | 145 ms (9.1) | 286 ms (17.8) | — |
| 32 | 285 ms (8.9) | 554 ms (17.3) | — |
| 64 | 611 ms (9.5) | 1115 ms (17.4) | — |

**torch-CPU is the fastest at every batch size.** The NPU-4col hybrid is slower
than both torch-CPU and even our numpy-float reference.

## Why (honest analysis)

Three compounding reasons the NPU loses for *this* model:

1. **The model is small.** MiniLM-L6-v2 is 22M params with 384/1536-dim GEMMs.
   The NPU's 264-GOPS advantage needs *big* GEMMs to amortize; 384-dim GEMMs are
   too small for the throughput to dominate.
2. **24 separate dispatches (no fusion).** The hybrid design sends each of the
   24 Linear GEMMs (Q/K/V/O ×6, FFN ×6×2) as a *separate* NPU dispatch. Each
   dispatch pays a host↔NPU round trip (~5–10 ms). 24 × ~8 ms ≈ 190 ms of pure
   dispatch overhead — that's most of the 226 ms. (This is exactly the
   [fusion problem](../iron/results/fusion-experiment.md): without fusing the
   block into one dispatch, per-op round trips dominate.)
3. **torch CPU is extremely optimized.** AVX-512 + batched BLAS + fused ops.
   Hard to beat for small models on a fast Zen 4 core.

## What this means for the goal

> **For MiniLM-L6-v2 specifically, the NPU is NOT a serving speedup over CPU.
> The correct path is torch-on-CPU (or the Radeon 780M iGPU).**

The NPU's value proposition requires one of:
- **A larger model** (768/1024/4096-dim GEMMs) where compute ≫ dispatch overhead.
  Untested here — would need compiling those shapes. Likely the crossover point.
- **Op fusion** (one dispatch for the whole block) — collapses the 24-dispatch
  overhead. This is Strategy B's domain (VitisAI EP automatic fusion), or the
  hard IRON-fusion work.

## What IS proven (don't lose this)

- The NPU executes MiniLM's GEMMs **bit-identically** to CPU (correctness, by
  construction). The integration works.
- The hybrid architecture (batch + route) is sound.
- The 4-col GEMM at 512³ runs at 264 GOPS — that's real, just not enough headroom
  to beat optimized CPU at MiniLM's tiny dims.

## Recommendation

For a *servable* MiniLM embedder today: **use torch-CPU** (9.9 ms/text). Route
the NPU path to larger models or wait for fusion. This v0 ships the correct
NPU execution + the honest benchmark so the decision is evidence-based, not
wishful.
