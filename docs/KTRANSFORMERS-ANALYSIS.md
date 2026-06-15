# ktransformers analysis — does it help our NPU embedding case?

> Research note, 2026-06-15. Source: github.com/kvcache-ai/ktransformers (v0.6.x,
> the kt-kernel era). Analysed to decide whether adopting an inference framework
> would accelerate the xdna-embed engine.

## How ktransformers works (the four core mechanisms)

KTransformers (Tsinghua MADSys / Approaching.AI) is a **CPU-GPU heterogeneous
inference framework** that made its name running **DeepSeek-V3/R1 (671B MoE) on
a single 24 GB consumer GPU + lots of DRAM**, at usable decode speeds. It rests
on four pillars:

### 1. YAML op-injection (the extensibility model)
A `config.yaml` lists rules: `match: {name: regex, class: torch.nn.Linear}` →
`replace: {class: <optimized op>, kwargs: {...}}`. KT walks the HuggingFace
model's module tree, regex-matches each submodule, and **swaps it for a custom
operator** (Marlin GPU linear, llamafile/AMX/AVX512 CPU linear, fused MLA
attention, CPU expert list...). The model's Python `forward()` is unchanged;
only the leaf operators are replaced. This is clean and is genuinely a good
design for a multi-backend engine.

### 2. Arithmetic-intensity-guided placement (the headline idea)
Every op is ranked by **arithmetic intensity** (FLOPs / byte moved):
- **MLA attention** has intensity ~512 → compute-bound → put on **GPU**.
- **MoE experts** have intensity ~0.075 at batch 1 (a GEMV), and they're 96%
  of total params but only **3.75% are activated per token** (6/160 experts) →
  memory-bound + sparse → keep them all in **CPU DRAM** (136 GB is cheap) and
  only fetch the few needed per token.

This is why "671B model on 24 GB VRAM" works: the GPU never holds the experts.

### 3. Expert prefetch + overlap (the decode speedup)
During autoregressive **decode**, the router picks which experts a token needs.
KT **prefetches those experts to GPU (or pin-memory)** while the previous
token's attention runs on the GPU, overlapping the CPU expert GEMV with GPU
attention. NUMA-aware thread pools + quantized kernels (below) make the CPU
side fast enough to feed decode.

### 4. Quantized kernels that operate *directly* on INT4/INT8 (no dequant)
- **GPU**: Marlin (near-ideal GPTQ INT4 GEMM).
- **CPU**: llamafile, and (newer) **AMX / AVX512-BF16 / AVX-VNNI** native
  kernels that run GEMM straight on quantized weights — no dequant pass.
  kt-kernel ships pre-built wheels that **auto-detect the CPU** and pick the
  best variant (AMX > AVX512+BF16 > AVX512+VNNI > AVX512 > AVX2). **Zen 4
  (our 7840HS) auto-selects the AVX512+BF16 variant.**

(Plus SGLang integration for production serving, prefix caching, multi-
concurrency "balance-serve", and an SFT path via LLaMA-Factory. Out of scope
here.)

## Honest assessment: does any of this help *us*?

Our case: **small dense embedding models** (MiniLM 22M, BGE/E5 33M, Qwen3-0.6B),
**single forward pass** (no decode), on a **Ryzen 7 7840HS** (Zen 4, AVX-512 BF16,
**no discrete NVIDIA GPU**), **XDNA1 NPU** (~10 TOPS, 4 cols), **Radeon 780M**
iGPU. Goal: serve embeddings on the NPU.

| KT pillar | Applies to us? | Why |
|---|---|---|
| **MoE expert offload** (the marquee win) | ❌ **Irrelevant** | Our models are **dense** — every param is used every forward. There are no "cold experts" to park in DRAM. The entire 671B-on-24GB trick depends on MoE sparsity (3.75% active) that dense models don't have. |
| **CPU-GPU heterogeneous split** | ❌ **No GPU to offload to** | KT's GPU side needs **NVIDIA CUDA SM ≥ 8.0** (Ampere+). We have **no NVIDIA GPU**. ROCm support exists but targets **discrete** Radeon (7900xtx on EPYC), not our RDNA3 *integrated* 780M. So the "hot ops on GPU" half of the split is unavailable — we'd be CPU-only via KT. |
| **Decode / expert-prefetch** | ❌ **Wrong workload** | KT's overlap magic targets **autoregressive token-by-token decode** (batch 1, memory-bound GEMV). Embedding inference is a **single dense forward over the whole sequence** — prefill-like, compute-bound at batch ≥ 1. There is no decode loop to overlap into. |
| **Quantized CPU GEMM kernels** | ⚠️ **One transferable idea** | kt-kernel's AVX512-BF16/AMX INT4-INT8 kernels auto-select on Zen 4 and skip dequant. If we quantize our models to **INT8**, KT's CPU path *might* beat our torch bf16 CPU path. **But**: (a) it's a CPU optimization, not an NPU one — it does nothing for the NPU goal; (b) it reintroduces the quant/dequant accuracy tradeoff we deliberately removed by going bf16 (our cos 0.999); (c) it targets **MoE** kernels primarily. |
| **YAML op-injection architecture** | ✅ **Good design, already have it** | Elegant multi-backend model. But our engine already has a cleaner, purpose-built version (model registry + adapter pattern + per-op backend routing in `engine/backends.py`). |

### The NPU question, specifically
**ktransformers has zero AMD XDNA / AIE / Phoenix support** — confirmed by full
grep (no `xdna`/`aie`/`phoenix` anywhere in the repo). The only "NPU" it supports
is **Huawei Ascend** (via CANN), unrelated silicon. So:

- KT cannot drive our NPU. Adopting it would not move *any* compute to the NPU.
- KT *could* be **extended** to add an XDNA backend (its YAML injection is a
  clean extension point: a `KLinearXdna` operator). But we measured that the
  NPU1 **loses** to CPU bf16 for dense batched GEMM (see
  `docs/CRITICAL-CORRECTION.md`), so routing *more* work to the NPU via KT
  would **slow us down**, not speed us up. There's no prize to chase.

## Verdict

**Adopting ktransformers would not help our NPU embedding goal.** Its marquee
capability (giant MoE on a tiny GPU via expert offload) presumes a discrete
NVIDIA GPU and MoE sparsity — we have neither, and we run small *dense* models.
Its CPU quantized kernels are the only transferable piece, and they're a CPU
optimization that trades accuracy and doesn't touch the NPU.

**What KT's design does validate:** our own architecture. KT's "regex-match a
module, swap its backend operator" is exactly our registry+adapter pattern; and
KT's "measure arithmetic intensity, place accordingly" is exactly what our
`-b auto` calibration does (it measured CPU winning and routed there). We
reached the same right design independently, sized for our actual hardware.

**What would actually move the needle (unchanged from AGENTS.md):**
1. **Strategy B (#14):** AMD Early Access for `vitis_aie_essentials` → automatic
   NPU fusion (the one thing that could lift NPU utilization 34%→peak). KT
   can't substitute for this.
2. **Better NPU GEMM kernels** (int8 MMUL = 2× MAC, accuracy tradeoff) — kernel
   work, not a framework.
3. **iGPU (Radeon 780M via ROCm/HIP)** — likely 2-4× over CPU for dense batched
   bf16, no host tax. The realistic "use this APU's silicon for AI" path. KT's
   ROCm path is for *discrete* Radeon and doesn't target the 780M, so even here
   it's a roll-your-own ROCm job, not a KT win.

**Bottom line:** ktransformers solves a problem we don't have (huge MoE, tiny
GPU) using hardware we don't own (discrete NVIDIA/ROCm GPU), for a workload we
don't run (autoregressive decode). It's a great framework for its niche; our
niche is different. Keep our purpose-built engine.
