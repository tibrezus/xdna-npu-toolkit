# Serving benchmark — bf16 NPU beats CPU 2.2–3.2× (Phase 1, v2)

> **Verdict (v2):** Phase-1 optimization (bf16 throughout) dropped the crossover
> from batch 32 → batch 8, and nearly doubled the win. The Phoenix NPU now beats
> torch-on-CPU by **2.2–3.2×** for batched embeddings. Auto-router delivers
> **2.77–2.80×** on real serving workloads.
>
> v1 (int16) was 1.2–1.37×. v2 (bf16) is 2.2–3.2× — a ~2× per-path improvement.

## Definitive result (MiniLM-L6-v2, seq=64, 7840HS, bf16 NPU)

| batch | M | torch-CPU | bf16 NPU | speedup | (v1 int16 was) |
|------:|--:|----------:|---------:|--------:|------:|
| 8 | 512 | 9.70 ms/txt | 9.78 | 0.99× | 0.84× |
| 16 | 1024 | 8.92 | 6.22 | **1.43×** | 1.00× |
| 32 | 2048 | 8.95 | 4.07 | **2.20×** | 1.20× |
| 64 | 4096 | 8.70 | 3.34 | **2.60×** | 1.37× |
| 128 | 8192 | 8.76 | 2.72 | **3.22×** | 1.37× |

**Auto-routing embedder** (`backend="auto"`): 8 texts → CPU; 64 texts → bf16 NPU → **2.80× faster**; 128 texts → 2.77×.

## What changed v1→v2 (Phase 1 = bf16, issue #17)

The int16 path quantized activations to int16 around each GEMM (36×/forward),
ran the GEMM, then dequantized back to float. bf16 removes ALL of that:

1. **Faster GEMM**: NPU1 bf16 MMUL (4,8,4)=128 MAC/cycle vs i16 (4,4,4)=64.
   Measured bf16 GEMM = **922 GOPS** vs i16 771 (1.20×). NPU1 bf16 is native.
2. **No quant/dequant**: the per-Linear quantize→GEMM→dequantize round-trip is
   gone. Activations stay bf16 end to end. This is the bigger win — it removes
   host-side per-op overhead AND the int32-accumulator dequant math.
3. **Better accuracy**: bf16 path cos(npu, fp32-ref) = **0.9994** vs int16's ~0.78.
   bf16 is more accurate than int16 quantization. No quality regression.
4. **torch bf16 glue**: layernorm/softmax/gelu/attention run as torch bf16 ops
   (torch upcasts internally for accuracy). torch↔numpy bf16 conversion is a
   zero-copy int16-bit-view round-trip (verified bit-identical).

## The QKV fusion (kept from v1)
Q/K/V weights concatenated → one 384→1152 GEMM (24→ dispatches, shared input).
Bit-identical to separate. This is orthogonal to dtype; works in bf16.

## Critical constraint discovered: 4-context limit
The `amdxdna` driver allows only **4 simultaneous hw_contexts** (max 2 per xclbin).
Each compiled (M, shape) kernel needs its own context. Consequences:
- The serving embedder uses **ONE compiled M (4096, batch 64)** — exactly 4
  shape-kernels = the limit. Inputs chunk/pad to batch 64. Small batches → CPU.
- **Resident-weight optimization (O5) is blocked**: per-layer weight baking would
  need 24 contexts. Documented; the weight copy (~0.03ms) is negligible anyway.
- **Fusion (O3/O4/O9) is not just a perf win but a NECESSITY**: a fused design is
  one xclbin = one context, freeing budget for richer pipelines.

## Why the NPU wins at batch ≥ 16 (bf16)
Per-dispatch fixed overhead ~0.5ms. At batch 64 (M=4096) the bf16 GEMMs hit 922
GOPS with no quant tax, so the 24 dispatches' overhead is dwarfed by compute.
The crossover dropped to batch ~8 (was ~24 with int16). The win grows with batch.

## Correctness (bf16)
- per-text cos(npu, fp32-torch): **0.9992–0.9995** (mean 0.9994)
- semantics: paraphrase 0.77 > unrelated 0.03 — strong discrimination, PASS
- bf16 GEMM single-kernel rel-err 0.017 (within bf16's ~3-digit precision)

## Files
- `embed.py` — `Embedder(backend="auto"|"npu"|"torch")`, auto-routes, single-M NPU
- `forward_bf16.py` — bf16 forward (NPU GEMMs + torch glue + QKV fusion)
- `bf16_backend.py` — `Bf16Backend` (pooled bf16 kernels, routes by shape)
- `fast_kernel.py` — `FastNpuKernel` (dtype-generic, pooled BOs, async-ready)
- `verify_bf16.py` — correctness (cos 0.9994, PASS)
- `bench_bf16.py` — the benchmark behind this table

## Open (Phase 1+)
- **O6 tiling sweep (#18)**: push 922→~1300 GOPS per GEMM (currently ~45% of peak)
- **O3 FFN fusion (#20)**: bf16 dissolves the requant wall → first real fusion
- **O4 attention fusion (#21)**, **O9 full-layer xclbin (#22)**
