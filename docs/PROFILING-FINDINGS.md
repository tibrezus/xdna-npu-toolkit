# Profiling: the forward is NPU-COMPUTE-BOUND (fusion is NOT the lever)

> **Disciplined finding (2026-06-13):** Detailed profiling of the Phase-1 bf16
> forward reveals it is **NPU-compute-bound**, not dispatch/overhead-bound.
> This **overturns the fusion plan's premise** (Phases 2-4 estimated 2.3-4×).
> Fusion's real prize is ~3-5%, not 2-4×. Phase 1 (bf16) was the architectural win.

## The measurements that changed the conclusion

### 1. Single-GEMM breakdown (qkvfused 4096×384×1152, pooled bf16+tiled)
| component | time |
|---|---|
| copy A into BO (numpy, 3.1MB) | 0.056 ms |
| copy B into BO (numpy, 0.9MB) | 0.016 ms |
| sync A (H→D) | 0.036 ms |
| sync B (H→D) | 0.012 ms |
| **launch + wait (NPU compute)** | **2.638 ms** |
| sync O (D→H) | 0.112 ms |
| copy O out | 0.371 ms |
| **full pooled run** | **3.140 ms** |

**The GEMM call is ~99% NPU compute.** Transfer/copy overhead is ~0.15ms total — negligible. The 2.638ms launch+wait IS the matmul compute (1377 GOPS).

### 2. Host↔NPU bandwidth is excellent
| tensor | size | H→D | D→H |
|---|---|---|---|
| [M,1536] intermediate | 12.6MB | 0.77ms (16GB/s) | 0.11ms (114GB/s) |
| [M,384] activation | 3.1MB | 0.16ms (20GB/s) | 0.03ms (123GB/s) |

**The fusion "prize" (keeping [M,1536] on-chip) = 0.88ms/layer × 6 = 5.3ms = ~3% of the 210ms forward.** Transfer was never the bottleneck.

### 3. Where the 210ms forward actually goes (batch 64)
| | ms | % |
|---|---|---|
| **24 GEMMs (NPU compute)** | **~140** | **67%** |
| glue (softmax 8, gelu 4, attn 3, LN 2...) | ~18-30 | ~12% |
| torch↔numpy conversions + framework | ~30 | ~14% |
| bias-add | ~13 | 6% |

The forward is **compute-bound on the GEMMs at ~1100-1400 GOPS** (~34-55% of NPU1's ~4096 GOPS bf16 peak).

## Why this overturns the fusion plan

The OPTIMIZATION-PLAN.md Phases 2-4 estimated:
- Phase 2 (FFN fusion): 2.3-2.8× → **reality: ~3-5%** (only transfer + gelu host cost saved; the 2 GEMMs' compute is identical whether fused or separate)
- Phase 3 (attention fusion): 3.0-3.5× → **reality: ~3% more** (softmax 7.9ms kept on-chip)
- Phase 4 (full-layer): 4×+ → **reality: ~10-12%** (all glue kept on-chip)

The estimates assumed dispatch overhead dominated (true for the **v0 int16 path** where quant/dequant + per-call BO alloc made each dispatch expensive). **bf16 + BO pooling eliminated that overhead** — so the GEMM time is now genuine compute, which fusion doesn't reduce.

## What the earlier "fusion works" result actually showed
The elementwise fusion experiment (`fused_add_add`) showed fused/single = 1.18 — **but elementwise ops are dispatch-bound, not compute-bound.** GEMMs are compute-bound, so fusion saves only the (tiny) dispatch overhead + transfer, not the compute.

## The real remaining lever: GEMM kernel throughput
The forward is at ~34-55% of NPU1's bf16 peak (4096 GOPS). Headroom exists, but it lives in **kernel efficiency**, not architecture/fusion:
- Better MMUL utilization (AMD's whole_array kernel is what it is)
- int8 MMUL = 256 MAC/cycle vs bf16 128 (2×), but needs int8 quantization (accuracy regression) — and no native mixed i8/bf16 precision in this design
- These are kernel-author tasks, not integration tasks

## Recommendation (honest)
1. **Phase 1 (bf16) was the win** — 2.7-3.1× over CPU, already delivered. The forward is near the efficient frontier for this hand-rolled-hybrid approach.
2. **Fusion (#20/#21/#22) is NOT worth the compiler-grade effort** — ~3-12% for weeks of work. Demote/defer.
3. **Real next levers**, in order:
   - **Larger model** (768/1024-dim): the NPU's compute advantage scales with GEMM size; MiniLM's 384-dim is small. Untested but the natural fit.
   - **Strategy B (#14, VitisAI EP)**: automatic fusion + AMD's optimized kernels — does for free what hand-rolling takes weeks. Still the highest-leverage path, blocked on Early Access.
   - **Accept the efficient Phase-1 result** for MiniLM; 2.7× over CPU is a real, shippable win for batched RAG indexing.

This is the disciplined outcome: **measure before building**. The profiling corrected a wrong assumption and redirected effort away from a low-value fusion toward the actually-promising directions.
