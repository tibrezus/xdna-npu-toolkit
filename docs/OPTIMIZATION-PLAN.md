# NPU Performance Optimization Plan (v2)

> Research-based plan to push the Phoenix NPU1 embedding path well beyond the
> current 1.37× CPU win — derived from analyzing AMD's **production NPU2 stack**
> (FastFlowLM, Ryzen AI SW) and the **open IRON fusion patterns** (mobilenet).

## 0. Where we are (baseline, proven) — REVISED after profiling

> **⚠ PROFILING UPDATE (2026-06-13):** Detailed profiling shows the bf16 forward is
> **NPU-compute-bound**, not dispatch/overhead-bound. This overturns the fusion
> estimates below. See `docs/PROFILING-FINDINGS.md`. Fusion's real prize is ~3-12%,
> not 2-4×. Phase 1 (bf16) was the architectural win.

| metric | value | how |
|---|---|---|
| Best speedup vs torch-CPU | **2.72-3.13×** @ batch 64/128 | bf16 + tiling + pooled BOs (Phase 1) |
| Per-GEMM throughput (bf16) | 976-1377 GOPS | 4-col whole_array, M=4096, bf16 |
| NPU1 bf16 theoretical peak | ~4096 GOPS | 16 tiles × 128 MAC/cycle × 1 GHz × 2 |
| Current efficiency | **34-55%** of bf16 peak | headroom in KERNEL efficiency, not architecture |
| **Fusion prize (measured)** | **~3-5%** | transfer is 16GB/s; GEMM call is 99% compute |

## 1. What production NPU2 does (the reference architecture)

Studied `FastFlowLM` (AMD/nod-ai, the production XDNA2 runtime) + `mlir-aie` mobilenet/IRON examples:

| technique | source | applicability to NPU1 |
|---|---|---|
| **Full-layer fusion** — one `layer.xclbin` dispatch per transformer block | FastFlowLM kernel zoo | ✅ the biggest lever |
| **On-device glue kernels** — gelu/layernorm/softmax/silu/rmsnorm as AIE bf16 kernels | `ml/` in mlir-aie | ✅ eliminates host round-trips |
| **Single-tile op chaining** — `disable_synchronization` self-loop ObjectFifos (build_fused_pair) | mobilenet/regular.py | ✅ the fusion mechanism |
| **Cascade streams** — split one op across tiles, stream partials (put/get) | mobilenet/cascade.py | ✅ for big GEMMs |
| **Fuse activation into matmul** — `conv_fused_relu.o` (one kernel) | mobilenet kernels | ✅ FFN1+gelu fused |
| **KV/state residency in NPU SRAM** | FastFlowLM | ⚠️ embeddings have no KV; applies to activations |
| **Tile-aligned work sizing** | FastFlowLM | ✅ already doing (M=512…8192) |
| **Weight compression + concatenated buffers** | FastFlowLM, mobilenet | ✅ partial (QKV fused) |
| **Phase split (prefill vs decode)** | FastFlowLM | ✅ prefill = batched embeddings |

### The hardware ceiling (honest)
NPU1 (Phoenix) is **4× weaker than NPU2 (Strix)**: 4 cols vs 8, AIE2 vs AIE2P, i16 MMUL (4,4,4)=64 vs (4,4,8)=128, bf16 (4,8,4)=128 vs (8,8,8)=512. FastFlowLM's `layer.xclbin` fuses onto 8 columns — on NPU1 the same fusion has ~¼ the compute. **We can't match NPU2 absolute throughput, but we CAN adopt its fusion architecture and maximize NPU1's 4 columns.**

## 2. Optimization opportunities (ranked)

Each ranked by **expected impact × feasibility / effort** on NPU1.

### Tier 1 — High impact, proven mechanism ⭐

#### O1. bf16 throughout (replace int16)  — **+20% per-GEMM, removes quant/dequant, simpler**
- **Measured**: bf16 GEMM = 922 GOPS vs i16 771 GOPS (1.20×). Eliminates the quantize/dequantize round every Linear (currently ~0.05 ms × 24 = 1.2 ms + accuracy plumbing).
- **Why it works**: NPU1 bf16 MMUL is (4,8,4)=128 MAC/cycle vs i16 (4,4,4)=64. Double the per-tile MAC throughput; the bf16 glue kernels (gelu/layernorm/softmax) are native.
- **Effort**: Low–Medium. Cast model to bf16, recompile 3 GEMMs × 5 M values as bf16. Accuracy: bf16 has ~3 decimal digits — fine for embeddings (verified int16 already loses ~0.01 cos; bf16 is comparable/better).
- **Risk**: Tiny. bf16 is the natural dtype for AIE2.
- **Plan**: recompile qkv/ffn1/ffn2 as bf16 at M=4096/8192; rewrite FastLinear to keep activations bf16 end-to-end; re-benchmark.

#### O2. Async / double-buffered dispatch  — **modest overlap (measured ~1.05–1.2×)**
- **Mechanism**: `pyxrt` `kern()` is async (submit = 0.096 ms, returns before completion); `wait()` blocks. So op N's host prep can overlap op N-1's NPU compute.
- **Measured reality**: pipelining a single GEMM gave only **1.04×** (0.330→0.318 ms) — once the NPU compute is fast (0.33 ms) there's little host work to overlap. The gain scales with the *host-work / NPU-compute ratio*. With int16 (quant+dequant ≈ 0.13 ms vs 0.43 ms compute) → ~1.15–1.2×; with **bf16 (O1), quant/dequant vanish** → less to overlap, ~1.05×.
- **Verdict**: real but **small**; not a Tier-1 lever. Worth doing once (it's cheap) but don't expect >1.2×. Demoted from original estimate.
- **Effort**: Low–Medium. `FastNpuKernel.submit()/.wait()`, pipeline the GEMM loop.
- **Risk**: Low.

#### O3. FFN fusion on-device (FFN1 → gelu → FFN2, one dataflow)  — **−2 dispatches/layer, gelu free**
- **Mechanism**: the mobilenet `build_fused_pair` pattern — two matmuls chained on one/multiple tiles via a `disable_synchronization` self-loop ObjectFifo, with gelu as an AIE bf16 kernel (`ml/gelu`) inserted between. Intermediate (1536-wide activation) **never touches host memory**.
- **Expected**: removes 2 dispatches × 6 layers = 12 dispatches (−50% from 24→12-ish once combined with attention). Plus gelu host round-trip gone. The 1536-wide intermediate (8×64×1536×2 = 1.5 MB) fits easily in tile SRAM.
- **Wall (known)**: GEMM→GEMM needs **requantization** between stages (i32 accum → next GEMM's input dtype). **bf16 (O1) dissolves this wall**: bf16 GEMM outputs bf16 directly, which feeds the next bf16 GEMM with no requant. This is why O1 should land first.
- **Effort**: Medium–High. Write an IRON design chaining FFN1-kernel → gelu-kernel → FFN2-kernel with self-loop fifos. First real fusion in this project.
- **Risk**: Medium. First multi-kernel IRON design here; the mobilenet reference is the blueprint.

### Tier 2 — Medium impact, builds on Tier 1

#### O4. Attention fusion on-device (QKV → scores → softmax → ctx → O)  — **collapse the attention block**
- Same self-loop-fifo mechanism, but attention has the score matmul (Q·Kᵀ), softmax, ctx matmul (attn·V) which are small bf16 ops. With on-device softmax kernel (`ml/softmax`) the whole attention sub-block runs as one dataflow.
- **Expected**: −4 dispatches/layer (QKV, O stay as GEMMs; the rest fused) → down to ~8 dispatches/forward.
- **Effort**: High. Attention reshape/transpose + the head-split dataflow is the trickiest part. The `ml/softmax` 2-core example is the reference.

#### O5. Weight pre-staging (resident device buffers)  — **−weight copy every dispatch**
- Today each dispatch re-syncs the weight BO (WqT). Weights are **static** — stage them once in device memory at load, never copy again. Only the activation BO moves per dispatch.
- **Expected**: for the 384×384 weight (294 KB) × 24 dispatches, ~1–3 ms saved/forward. Small but free.
- **Effort**: Low. Extend `FastNpuKernel` with a resident weight BO set once.

#### O6. Better per-GEMM tiling (push toward peak)  — **+30–80% per-GEMM**
- Current tile (m=64,k=64,n=32) hits 27–45% of peak. Sweep tile sizes (m∈{32,64,128}, n∈{32,64,128}), K-blocking, and the `--b-col-maj` layout flag. NPU1's 4×4 tile array favors different geometry than the default.
- **Expected**: pushing 45%→70% efficiency = ~1.5× per-GEMM. Biggest single throughput lever after fusion.
- **Effort**: Medium. Mechanical sweep — compile ~20 variants, benchmark, pick. Low risk, tedious.

### Tier 3 — Scaling / future

#### O7. Larger batch + continuous batching
- NPU win grows with batch (1.20×→1.37× from 32→128). Continuous batching (aggregate concurrent requests into M=8192 chunks) maximizes NPU utilization for a serving workload. Expected: 1.37×→~1.6× if memory allows.

#### O8. i4/int8 weight-only quantization
- NPU1 i8 MMUL is (4,8,8)=256 MAC/cycle — **4× i16, 2× bf16**. If weights can go int8 (per-channel) with bf16 activations (mixed precision), GEMM throughput could 2× again. Risk: accuracy + AMD's mixed-precision kernel support on AIE2 is less proven than bf16.

#### O9. Full transformer-layer xclbin (the FastFlowLM endgame)
- Once O3+O4 are proven, compose a single `layer.xclbin` that streams one transformer block end-to-end (QKV→attn→O→LN→FFN1→gelu→FFN2→LN) with one host dispatch. This is FastFlowLM's `layer.xclbin` architecture adapted to 4 columns. Expected: 1 dispatch/layer × 6 = 6 dispatches/forward (from 24), all glue on-device.

## 3. The plan (phased, each phase independently shippable)

### Phase 1 — "bf16 + tiling" ✅ DONE — actual 2.2–3.2× over CPU (est. was 1.8–2.0×)
1. **O1 bf16** ✅: recompiled 4 GEMMs × 5 M values as bf16. `forward_bf16.py` keeps bf16 end to end. cos(npu,fp32)=0.9994. *(commit `0a97265`)*
2. **O6 tiling sweep** ✅: m=128 k=64 n=32 wins (1072 vs 906 GOPS, 1.17×/GEMM). Default at M=4096. *(commit + HF)*
3. **O5 resident weights** ❌ BLOCKED: amdxdna = 4 hw_contexts max (discovered). Wontfix.
4. **O2 async** ⬇️ deprioritized: measured only 1.04× (little host work to overlap once compute is fast).

**Actual result**: batch 16: 1.61× | 32: 2.28× | 64: 2.72× | 128: 3.13× over torch-CPU. Crossover batch 8 (was 32). Beat the estimate (1.8–2.0×).

**Key discovery**: amdxdna driver limits to **4 simultaneous hw_contexts** (2 per xclbin). The serving embedder uses ONE compiled M (4096=batch64 = exactly 4 shape-kernels). This makes **fusion a necessity** (one xclbin = one context frees budget), not just a perf win.

### Phase 2 — "first real fusion" (FFN block on-device)
5. **O3 FFN fusion**: IRON design chaining FFN1(bf16) → gelu(AIE kernel) → FFN2(bf16) via self-loop fifos. Verify bit-exact vs the unfused bf16 path. *(~3–5 days, mobilenet `build_fused_pair` as blueprint)*
6. Measure: −2 dispatches/layer, intermediate stays on-chip.

### Phase 3 — "attention fusion + tiling"
7. **O4 attention fusion**: on-device softmax + score/ctx matmuls. *(~1 week)*
8. **O6 tiling sweep**: find the best (m,k,n) for the 4×4 array. *(~1 day)*
9. Re-benchmark. Expected: ~8 dispatches/forward, per-GEMM ~1.5× faster.

### Phase 4 — "full-layer xclbin" (the NPU2 architecture on NPU1)
10. **O9 layer fusion**: compose the whole transformer block into one `layer.xclbin` (FastFlowLM pattern, 4-column). *(~2+ weeks)*
11. This is where single-query serving (batch=1) could finally beat CPU — one dispatch amortizes everything.

## 4. What this plan deliberately does NOT do

- **Doesn't chase NPU2 absolute throughput** — impossible on NPU1 (4× less silicon). Honest.
- **Doesn't depend on Strategy B / Early Access** — all of Tier 1–2 is achievable with the open IRON/Peano toolchain already proven working. Early Access (#14) remains a parallel track for the *automatic* fusion path; this plan is the *manual IRON* path.
- **Doesn't over-promise single-query** — batch=1 beating CPU requires Phase 4 (full-layer fusion). Stated as a goal, not a Phase-1 deliverable.

## 5. Expected end-state

| phase | dispatches/fwd | est. vs CPU @batch64 | est. vs CPU @batch1 |
|---|---|---|---|
| now (baseline) | 24 | 1.37× | 0.66× (CPU wins) |
| **Phase 1 ✅ DONE** | 24 | **2.72× (actual)** | ~1.0× (break-even) |
| Phase 2 (+FFN fusion) | ~18 | **~3-5% more** *(revised down from 2.3-2.8×)* | ~1.05× |
| Phase 3 (+attn+tiling) | ~8 | **~3% more** *(revised from 3.5-4×)* | ~1.1× |
| Phase 4 (full-layer) | ~6 | **~10-12%** *(revised from 4×)* | ~1.2× |

**Why revised:** profiling (PROFILING-FINDINGS.md) shows the forward is
compute-bound on the GEMMs (each call is 99% NPU compute, transfer ~0.15ms).
Fusion keeps intermediates on-chip but does NOT reduce the GEMM compute. The
original estimates assumed dispatch overhead dominated — true for v0 int16, false
after bf16+pooling. **Fusion is now low-value; real lever is GEMM kernel throughput / larger models / Strategy B.**

These are engineering estimates grounded in measured per-GEMM numbers and dispatch counts.
