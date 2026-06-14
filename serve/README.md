# serve/ — servable MiniLM-L6-v2 embeddings on the 7840HS

A working embedding pipeline with two backends, plus the honest benchmark that
tells you which to use.

## Quick start

```bash
# default: torch CPU (fast + correct, ~10 ms/text)
python -m serve.embed "a man eating food" "a man having a meal"

# NPU backend (Linear GEMMs on the Phoenix NPU; correct but slower for MiniLM — see BENCHMARK.md)
python -m serve.embed --backend npu "a man eating food" "a man having a meal"
```

As a library:
```python
from serve.embed import Embedder
e = Embedder(backend="torch")
vecs = e.embed(["query text", "document text"])   # -> np.ndarray [N, 384]
```

## Files
- `embed.py` — `Embedder` class + CLI (backends: `torch`, `npu`)
- `minilm_forward.py` — pure-numpy MiniLM forward (float-verified vs transformers;
  int16 path where GEMMs run identically on CPU or NPU)
- `npu_backend.py` — `NpuGemmPool`: routes Linear GEMMs to compiled NPU xclbins
- `validate_forward.py` — correctness: float==transformers, int16 semantics
- `run_npu_forward.py` — run the model with NPU GEMMs, verify bit-identical to CPU
- `bench_serving.py` / `BENCHMARK.md` — **the honest perf verdict**

## The honest verdict (read this)

For MiniLM-L6-v2 on this machine, **torch-on-CPU is the fastest** (~10 ms/text).
The NPU-4col hybrid is slower (28 ms/text) — not because the NPU is slow, but
because MiniLM's GEMMs are small (384-dim) and the hybrid design pays 24 separate
host↔NPU round trips (no fusion). See [`BENCHMARK.md`](BENCHMARK.md).

**The NPU's value is for larger models** (768/1024/4096-dim GEMMs, where compute
≫ dispatch overhead) **or with op fusion** (one dispatch for the whole block —
Strategy B / VitisAI EP). For MiniLM-sized models, CPU/iGPU wins.

## What's proven here
- ✅ The NPU executes MiniLM's GEMMs **bit-identically** to CPU (correctness by
  construction: int16×int16→int32 is exact).
- ✅ The hybrid architecture (numpy forward + NPU GEMM backend + batch routing) works.
- ✅ 4-col GEMMs at model shapes (384/1536) compile and run on NPU1.
- ⚠️  But for MiniLM, it's not a speedup over optimized CPU. Honest.
