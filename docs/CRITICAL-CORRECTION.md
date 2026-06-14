# CRITICAL CORRECTION: CPU bf16 beats the NPU (same-precision comparison)

> **Mea culpa (2026-06-13):** My earlier "NPU beats CPU 2.7×" claims (Phase 1)
> were based on an **unfair precision comparison**: NPU-bf16 vs CPU-**fp32**.
> The honest same-precision comparison (bf16 vs bf16) shows **CPU bf16 beats the
> NPU** for both MiniLM and Qwen3-0.6B. This document corrects the record.

## The fair comparison (all bf16, batch 64, seq 64, same hardware)

| model | torch CPU **fp32** | torch CPU **bf16** | NPU bf16 | NPU vs CPU-bf16 |
|---|---|---|---|---|
| MiniLM-L6-v2 | 2.89 ms/txt | **1.23 ms/txt** | 3.34 | CPU **2.72× faster** |
| Qwen3-Embedding-0.6B | 89.3 ms/txt | **49.1 ms/txt** | 92.3 | CPU **1.88× faster** |

**The NPU loses to CPU bf16 in both cases.** My Phase-1 "win" was an artifact of
comparing against CPU-fp32 (which is 2.3–2.8× slower than CPU-bf16 on this Zen 4).

## Why CPU bf16 wins

1. **Zen 4 has native bf16 support** (AVX-512 BF16 / AMX). torch's oneDNN bf16
   GEMM kernels are highly optimized — they hit a large fraction of the 8 cores'
   bf16 throughput. A 7840HS is 8 high-clock Zen 4 cores.
2. **No host↔device tax.** CPU does everything in-place, fully fused. The NPU
   path pays: 112 dispatches/forward (Qwen), torch↔numpy conversions at every
   Linear, and ~2s of CPU glue (float32 attention, RoPE, rmsnorm) it can't shed.
3. **NPU1 (Phoenix) is ~10 TOPS, 4 AIE columns.** That silicon is sized for
   *low-power always-on* inference, not for beating a 65W 8-core CPU at dense
   batched GEMM. FastFlowLM's own XDNA1-feasibility doc says exactly this: the
   NPU's value is TOPS/W (67× less energy than iGPU), not raw throughput.

## What the earlier number actually meant
"NPU 2.7× over CPU" = NPU-bf16 (3.34) vs CPU-fp32 (8.7). Real, but only because
fp32 is 2× the work of bf16. Vs the *fair* CPU-bf16 baseline (1.23), the NPU is
2.72× slower. The earlier benchmarks should have compared same precision.

## What IS still true (don't throw out)
- The NPU **correctly executes** both models (architecture verified: MiniLM
  cos 0.9994; Qwen3-0.6B cos 0.80 vs fp32, semantics preserved). The integration
  is genuinely working — it's just not faster than optimized CPU bf16.
- The bf16 path, BO pooling, QKV/gate-up fusion, the whole IRON toolchain — all
  real and correct.
- The NPU may still win on **power efficiency** (TOPS/W) for always-on/battery
  use — that's its design intent, untested here.

## Honest recommendation
For **batched embedding serving on this machine (wall-powered, speed-optimal)**:
**use torch CPU bf16.** It beats both CPU-fp32 (2.3–2.8×) and the NPU (1.9–2.7×).

The NPU1 (Phoenix) is not a throughput win for dense transformer GEMMs vs an
8-core Zen 4 with optimized bf16 BLAS. This aligns with FastFlowLM's documented
position that Phoenix's value is efficiency, and that XDNA2 (Strix, ~50 TOPS,
8 columns) is where throughput competitiveness begins.

## What this means for the project goals
- Goal "embeddings easily servable on NPU1 for speed": **not achievable as a
  speed win** vs CPU bf16. The NPU serves them *correctly*, but slower.
- The NPU remains valuable for: power-constrained scenarios, the open-toolchain
  proof-of-concept (which is real and published), and as the foundation if/when
  Strategy B (#14) adds AMD's optimized fused kernels.
- For a *practical fast local embedder*, the answer is **torch CPU bf16** (or the
  Radeon 780M iGPU), not the NPU1.
