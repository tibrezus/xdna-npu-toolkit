# xdna-embed — a llama.cpp-style embedding engine for the AMD XDNA NPU

A command-line embedding inference engine that runs sentence-embedding models on
the **AMD XDNA1 NPU** (Phoenix, Ryzen 7 7840HS) via the open-source IRON/PEANO
toolchain, with a CPU fallback that works for **any** HuggingFace model. Think
`llama-embedding` / `llama-server --embedding`, but for the NPU.

```bash
# one-shot embeddings
xdna-embed embed -m minilm -b npu "a dog runs in the park"
xdna-embed embed -m qwen3-0.6b -b cpu --input corpus.txt --output vecs.npy -f numpy

# OpenAI-compatible HTTP server
xdna-embed server -m minilm -b auto --port 8080
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"minilm","input":["a dog runs","a puppy sprints"]}'

# benchmark / inspect
xdna-embed bench -m minilm -b all
xdna-embed list        # which models have compiled NPU kernels
xdna-embed info        # NPU device + backend status
```

## Subcommands

| command  | like llama.cpp's           | what it does                                        |
|----------|----------------------------|-----------------------------------------------------|
| `embed`  | `llama-embedding`          | embed text(s); compact / json / numpy output        |
| `server` | `llama-server --embedding` | OpenAI-compatible `/v1/embeddings` + `/v1/models`   |
| `bench`  | `llama-bench`              | min-of-K timing across backends & batch sizes       |
| `list`   | —                          | registered models + which run on the NPU            |
| `info`   | —                          | NPU device, perms, memlock, compiled models         |

## Backends (`-b`)

| backend | runs on       | scope                                    |
|---------|---------------|------------------------------------------|
| `npu`   | XDNA1 NPU     | **registered, compiled models only**     |
| `cpu`   | torch CPU bf16| **any HuggingFace model**                |
| `auto`  | either        | calibrates both on warmup, picks winner  |

**The NPU is not a universal JIT.** Unlike llama.cpp's GGUF loader, running a
model on the NPU requires bf16 GEMM xclbins compiled for that model's exact
Linear shapes (the amdxdna driver further caps at 4 simultaneous contexts).
`minilm` and `qwen3-0.6b` are compiled and ready. Adding a model = compile its
GEMM shapes with IRON + write an adapter in `engine/backends.py` (see the
`_MinilmAdapter` / `_QwenAdapter` patterns).

## The honest performance picture (this machine)

`xdna-embed bench -m minilm -b all` (batch 64, bf16, venv `torch+cpu`):

```
backend     batch   ms/text  ms/batch   dim
cpu             64     1.34       87.6   384
npu             64     3.03      193.6   384
auto->cpu       64     1.26       80.9   384   <- auto calibrated, picked CPU
```

On this 8-core Zen 4 with native AVX-512 BF16 + oneDNN, **torch CPU bf16 beats
the ~10 TOPS NPU1 for dense embedding GEMM** (see `docs/CRITICAL-CORRECTION.md`).
`auto` measures this itself and picks the winner. The NPU remains valuable for
**power efficiency / CPU offload** — use `-b npu` explicitly when you want to
free the CPU or run on battery, even if wall-clock is slower.

> **Gotcha that cost real debugging time:** the Arch system `torch` (in
> `/usr/lib/python3.14/site-packages`, where `pyxrt` also lives) is ~5× slower
> than the venv's `torch+cpu`. If you put that dir on `PYTHONPATH` to reach
> `pyxrt`, Python loads the slow system torch and everything crawls. The engine
> avoids this by importing the venv torch *first*, then appending the system
> path for `pyxrt`. **Do not set `PYTHONPATH=/usr/lib/python3.14/site-packages`.**

## Setup

The engine needs the IRON env (Python 3.14 venv with torch, transformers,
mlir-aie, ml_dtypes) and the NPU stack (`xrt` + `xrt-plugin-amdxdna`, memlock
unlimited — run under a login shell so `pam_limits` applies).

```bash
source /tmp/iron/env/bin/activate
# pyxrt + IRON are wired up internally; no PYTHONPATH needed.
xdna-embed info     # confirm: NPU device present, memlock OK, models ready
```

Install (editable, no deps — runtime deps come from the IRON env):
```bash
uv pip install -e . --no-deps
```

Weights: MiniLM at `$MINILM_WEIGHTS` (default `/tmp/voe-inspect/minilm`),
Qwen3-0.6B at `$QWEN_WEIGHTS` or the HF cache. Compiled GEMM xclbins at
`/tmp/iron/{minilm-gemms-bf16-M*,qwen-gemms-bf16}/`.

## Files

```
engine/
  cli.py        argparse subcommands + llama.cpp-style banner/timings
  backends.py   CpuBackend (any HF), NpuBackend (adapter dispatch), AutoBackend (calibrate)
  server.py     stdlib OpenAI-compatible /v1/embeddings
  models.py     ModelSpec registry (alias, hf_id, dim, pooling, npu adapter)
```
