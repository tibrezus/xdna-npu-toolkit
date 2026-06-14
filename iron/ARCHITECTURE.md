# Architecture: Embeddings on the Phoenix NPU1

> Honest engineering analysis of the path to **easily servable embeddings on the
> AMD Ryzen 7 7840HS NPU (Phoenix / XDNA 1 / AIE2)**, based on measured results.

## What's proven (all on Linux, open-source IRON/Peano)

| Op | Config | Throughput | vs CPU | Status |
|---|---|---|---|---|
| int16 GEMM | 256³, 1 core | 41 GOPS | **16.8× faster** | ✅ verified exact |
| int16 GEMM | 512³, 4 cols (16 cores) | **264 GOPS** | — | ✅ verified exact |
| int16 GEMV | 288², batch=1, 1 core | 0.4 GOPS | **8× SLOWER** | ✅ verified exact |

## The defining constraint: data-movement vs compute

The NPU is a **discrete-ish coprocessor on a PCIe-like path**. Every op pays a
fixed host→NPU→host round trip (BO allocation, sync, dispatch). Measured:
**~0.37 ms per dispatch** regardless of how tiny the compute is.

```
                 compute time         round-trip overhead
                 ↓                    ↓
big batched GEMM: small fraction      amortized over 134M MACs  → NPU WINS (264 GOPS)
batch=1 GEMV:     ~0 (166K MACs)      dominates entirely          → NPU LOSES (8× slower)
```

This is NOT a Phoenix weakness — it's the fundamental shape of any
discrete-accelerator serving model. It has a direct, hard consequence:

> **For single-query (batch=1) serving, the NPU only wins if an entire model
> (or a large fused block) executes in ONE dispatch** — so data stays on-device
> and there's only one round trip for the whole forward pass.

## What this means for embedding serving (two regimes)

RAG embedding has two workloads:

### 1. Offline index building — NPU WINS (proven)
Embed thousands of documents. **Batch them.** Each layer becomes a big
matrix-matrix GEMM → the 4-col design (264 GOPS) wins decisively. This works
**today** with what we've built.

### 2. Online single-query — NPU LOSES (today), needs fusion
Embed one live query (batch=1). Per-op round trips kill it. Two ways out:
- **(a) Op fusion** — compile the whole transformer block into ONE xclbin so
  data never returns to host mid-forward. This is exactly what AMD's VitisAI EP
  does (Strategy B, #9) and exactly why it matters despite Strategy A working.
  IRON hand-composition *can* fuse, but it's the hard part.
- **(b) Route batch=1 to CPU/iGPU** — pragmatic. The NPU handles the batched
  indexing; the live query goes to the Radeon 780M or CPU.

## The realistic serving architecture

```
                    ┌─────────────────────────────────────┐
   embed(texts) ───▶│  EmbeddingServer                     │
                    │   ├─ tokenize                         │
                    │   ├─ if batch ≥ B_thresh:             │
                    │   │    → NpuBatchedLinear (4-col GEMM)│  ← 264 GOPS, proven
                    │   └─ else (batch=1):                  │
                    │       → CpuLinear / iGPU             │  ← avoids round trip
                    └─────────────────────────────────────┘
```

The server's job is **batching + routing**: accumulate queries until a batch
threshold, then dispatch to the NPU as one big GEMM; send lonely single queries
to CPU. This makes the NPU genuinely useful for serving without needing the
hard op-fusion work.

## Where this leaves the full model

A real embedding model (MiniLM-L6-v2) is more than GEMMs: it needs LayerNorm,
softmax, GELU, residual adds, and an embedding-gather, all stitched together.
Three honest options:

1. **NPU-GEMM + CPU-glue (hybrid)** — keep the model in Python/numpy, offload
   only each GEMM to the NPU. Simple, servable, but the per-GEMM round trips
   make single-query slow (the finding above). Best for batched workloads.
2. **IRON full fusion (Strategy A, hard)** — hand-compose the whole transformer
   in IRON so it's one dispatch. Viable for single-query, but weeks of careful
   kernel + dataflow work. Not "easily servable" in the near term.
3. **ONNX fusion (Strategy B, gated)** — feed the model's ONNX to the VitisAI
   compiler, which fuses automatically. The "easy" path IF the Linux gate lifts.

## Current recommendation

Build the **batched serving layer** now (option 1, proven primitives) — it makes
the NPU useful for the indexing workload today. Pursue op fusion (option 2/3) in
parallel for the single-query workload. Do NOT pretend batch=1 NPU serving is
fast today — it isn't, and routing those to CPU is the honest, correct call.
