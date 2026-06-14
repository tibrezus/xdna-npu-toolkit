# serve/ — servable MiniLM-L6-v2 embeddings: NPU-accelerated (beats CPU at batch≥32)

A working embedding pipeline that **beats torch-on-CPU by 1.2–1.37× for batched
workloads** using the Phoenix NPU, with smart auto-routing to CPU for small
batches. The honest, optimized result of the NPU serving investigation.

## Quick start

```bash
# auto-routing (default): small batch -> CPU, large batch -> NPU (1.43x at 64)
python -m serve.embed --backend auto --bench 64

# embed some texts
python -m serve.embed "a man eating food" "a man having a meal"

# force a backend
python -m serve.embed --backend torch ...   # CPU (fastest for single queries)
python -m serve.embed --backend npu   ...   # NPU (research / batched)
```

As a library:
```python
from serve.embed import Embedder
e = Embedder(backend="auto")
vecs = e.embed(large_corpus)   # auto-routes to NPU; 1.2-1.37x over CPU
```

## The result (see BENCHMARK.md)

| batch | torch-CPU | NPU (optimized) | speedup |
|---|---|---|---|
| 8 | 11.3 ms/txt | 17.1 | 0.66× (→CPU) |
| 32 | 10.0 | 8.3 | **1.20× (NPU)** |
| 64 | 8.7 | 6.4 | **1.37× (NPU)** |
| 128 | 9.0 | 6.6 | **1.37× (NPU)** |

Auto-router: 64 texts → NPU = **1.43× faster than CPU-only**.

## How it got here (4 optimizations)
1. **BO pooling** — pre-allocate device buffers once, reuse (2×/GEMM)
2. **torch glue** — gelu/layernorm/softmax/attention on torch C kernels (scipy erf was 170× slower)
3. **QKV fusion** — one 384→1152 GEMM instead of 3 (36→24 dispatches)
4. **Batch scaling** — compile at M=512…8192, route by batch; amortize dispatch overhead

## Files
- `embed.py` — `Embedder` + CLI, auto-routing
- `forward_fast.py` / `forward_fused.py` — optimized forwards (torch glue + QKV fusion)
- `fast_kernel.py` — `FastNpuKernel` (pooled-BO NPU runner)
- `pooled_backend.py` — `MultiMBackend` (shape + M routing)
- `minilm_forward.py` — original pure-numpy reference (int16, NPU/CPU-identical)
- `BENCHMARK.md` — full benchmark + methodology
- `verify_semantics.py` — correctness (correlation 0.99, discrimination PASS)
