# Serving benchmark — NPU BEATS CPU for batched embeddings (v1, optimized)

> **Verdict (updated):** After performance optimization, the Phoenix NPU beats
> torch-on-CPU by **1.2–1.37×** for batched embedding workloads (batch ≥ 32),
> and the auto-routing embedder delivers **1.43×** at batch=64. For single-query
> / small batches, CPU still wins — the embedder auto-routes accordingly.
>
> This overturns the v0 finding (where the unoptimized NPU path lost). The v0
> loss was almost entirely harness overhead, not the NPU — see below.

## Definitive result (MiniLM-L6-v2, seq=64, 7840HS, optimized path)

| batch | M | torch-CPU | NPU (optimized) | speedup |
|------:|--:|----------:|----------------:|--------:|
| 8 | 512 | 11.31 ms/txt | 17.05 ms/txt | 0.66× (CPU) |
| 16 | 1024 | 10.81 | 11.45 | 0.94× (CPU) |
| 32 | 2048 | 10.03 | 8.33 | **1.20× (NPU)** |
| 64 | 4096 | 8.72 | 6.38 | **1.37× (NPU)** |
| 128 | 8192 | 8.99 | 6.55 | **1.37× (NPU)** |

**Auto-routing embedder** (`backend="auto"`): 8 texts → CPU, 64 texts → NPU → **1.43× faster** than CPU-only.

## What changed v0 → v1 (the four optimizations)

The v0 NPU path lost (28 ms/text vs CPU 9.9). Profiling revealed **the NPU GEMMs
were never the bottleneck** — a single 512³ GEMM is 0.89ms (169 GOPS), scaling to
2.18ms at 4096³ (553 GOPS). The cost was all harness overhead:

1. **BO pooling** (`fast_kernel.py`): `NpuKernel.run()` allocated a fresh
   `pyxrt.bo()` + `.map()` per call (36×/forward). Pre-allocating pooled BOs cut
   per-GEMM time 2× (0.89 → 0.43ms).

2. **torch glue** (`forward_fast.py`): the *hidden* killer was `scipy.special.erf`
   GELU — 8.4ms per layer (scipy is inexplicably slow). `torch.nn.functional.gelu`
   is 0.05ms (**170× faster**). All glue (gelu/layernorm/softmax/attention) moved
   to torch C kernels via zero-copy `from_numpy`.

3. **QKV fusion** (`forward_fused.py`): concatenate Q/K/V weights → one 384→1152
   GEMM (36 → 24 dispatches, shared input quantization). Bit-identical; helps at
   batch≥32 (6.75 → 6.29 ms/text @batch64).

4. **Batch scaling** (`pooled_backend.py`): compile GEMMs at M = 512/1024/2048/
   4096/8192; route by actual batch. The NPU's 553 GOPS amortizes the fixed
   ~0.5ms/dispatch overhead → wins once compute ≫ overhead (batch ≥ 32).

## Why the NPU wins at batch ≥ 32

Per-dispatch fixed overhead is ~0.5ms. At batch=8 (M=512) the 24–36 dispatches'
overhead (~12–18ms) dominates a small total. At batch=64 the same dispatches
carry 8× more compute, so the NPU's 553-GOPS throughput pulls ahead of CPU's
~linear scaling. The crossover is batch ≈ 16–24.

## Why CPU still wins for small batch

For batch ≤ 16, the dispatch overhead isn't amortized. The embedder auto-routes
these to torch-CPU. Single-query serving is CPU's domain until op-fusion (one
dispatch per block) collapses the overhead — that's Strategy B (#14, VitisAI EP).

## Correctness

The int16 NPU path is bit-identical to the CPU int16 path by construction
(int16×int16→int32 is exact). Semantics verified: NPU-vs-torch embedding
correlation **0.99**, within-group 0.64 vs cross-group 0.10 (strong
discrimination), paraphrase-vs-unrelated PASS.

## Files
- `embed.py` — `Embedder(backend="auto"|"torch"|"npu")` with smart routing + CLI
- `forward_fast.py` / `forward_fused.py` — torch-glue + QKV-fused forwards
- `fast_kernel.py` — `FastNpuKernel` (pooled-BO runner, 2×/GEMM)
- `pooled_backend.py` — `MultiMBackend` (shape/M routing to pooled kernels)
- `bench_crossover.py`, `bench_fused.py` — the benchmarks behind this table
