# Embeddings on the AMD XDNA 1 NPU (Phoenix) — Complete Walkthrough

This is the canonical, step-by-step record of the entire investigation and
implementation: **what works, what's gated, and exactly how to reproduce it**.
It mirrors the GitHub issues/wikis on `tibrezus/xdna-npu-toolkit` and is kept
in the repo so it's always available offline.

> **The one-line status:** unlike LLMs (whose runtimes reject Phoenix outright),
> the AMD **VitisAI runtime supports XDNA 1 for embedding/transformer inference**.
> The EP initializes on the NPU. The only remaining gate is *model compilation*
> (the public Linux wheels are deployment-only; the compiler is account-gated).

---

## 0. Prerequisites

- An AMD XDNA 1 NPU: Ryzen 7040 (Phoenix, e.g. **7840HS**) or 8040 (Hawk Point).
- Kernel 6.11+ with `amdxdna` (best on 7.x), `linux-firmware` for `amdnpu/*`.
- `xrt` + `xrt-plugin-amdxdna` (Arch: `sudo pacman -S xrt xrt-plugin-amdxdna`).
- Run `xdna-npu enable` first (memlock fix + NPU performance power mode).
- Verify with `xdna-npu doctor` — all checks PASS, and it should report
  `XDNA 1`, `5 cols`, `AIE-ML (AIE2)`, firmware e.g. `1.5.5.391`.

## 1. Why embeddings are different from LLMs on this NPU

LLM runtimes (FastFlowLM, Lemonade 10) ship precompiled `.xclbin` overlays
compiled for **XDNA 2** (Strix: 8 columns, AIE2P tiles). A 5-column Phoenix
AIE-ML array cannot load them, and there's no open AIE overlay compiler to
rebuild them. See `docs/LLM-FEASIBILITY.md` → §1–5.

Embeddings/CNNs/transformers take a **different, public, Phoenix-supported
path**: AMD's VitisAI Execution Provider stack. Evidence (all from the installed
wheels):

| Signal | Where | What it shows |
|---|---|---|
| `voe` branches on `device in ["phx","stx"]` | `voe/passes/fuse_MATMULINTEGER.py` | Phoenix is a first-class target |
| `4x4` design param selected | `voe/passes/op_fusion.py` | `4x4` = the PHX/HPT partition |
| AIE-ML (Phoenix) + AIE2P (Strix) kernels | `ryzenai_dynamic_dispatch/include/xaiengine/` | both gens shipped |
| Public pip index, **no account gate** | `pypi.amd.com/ryzenai_llm/1.7.1/linux/simple/` | the whole runtime is downloadable |

## 2. Install the VitisAI stack

```bash
uv python install 3.12          # the AMD wheels are cp312
xdna-npu embed-setup            # installs from pypi.amd.com, then probes the EP
```

This installs: `onnxruntime-vitisai`, `voe`, `ryzenai-dynamic-dispatch`,
`ryzenai-onnx-utils`, `onnxruntime-providers-ryzenai`.

## 3. Prove the EP initializes on your NPU

```bash
xdna-npu embed-check
```

Expected on a 7840HS:

```
✓ EP listed by onnxruntime : True
✓ EP initializes on NPU   : True
providers                   : VitisAIExecutionProvider, CPUExecutionProvider
at-runtime compile available: False (deployment-only build)
```

**This is the headline result.** The NPU execution provider initializes on
XDNA 1. The runtime plumbing is real.

## 4. The two embedding backends in this toolkit

### 4a. CPU reference backend — works TODAY

Runs the same ONNX embedding model on CPU via stock onnxruntime. Produces the
ground-truth vectors the NPU run must match — and is itself a usable RAG
embedding backend right now.

```bash
# export a small, RAG-grade model correctly (a naive export saturates outputs)
xdna-npu embed-export sentence-transformers/all-MiniLM-L6-v2

# run it
xdna-npu embed-run "a cat sat on the mat" --cpu --model ./all-MiniLM-L6-v2/model.onnx --tokenizer ./all-MiniLM-L6-v2
```

Verified numerically (all-MiniLM-L6-v2, 384-d):
- paraphrase cosine **0.525** ("a cat sat on the mat" vs "the feline rested on the rug")
- unrelated cosine **0.346** (vs "how to install linux arch")
- matches the PyTorch reference (0.549 / 0.06) — masked mean-pooling is correct.

> **Export gotcha (recorded so you don't repeat it):** `torch.onnx.export(model,
> dict(inputs), ...)` produces a graph whose outputs saturate (cosine ~0.99
> between any two inputs) because the kwargs aren't wired positionally. Export a
> thin **positional wrapper** (see `xdna_npu/export_model.py`).

### 4b. NPU backend — ready, blocked only on a compiled PHX model

```bash
xdna-npu embed-run "hello world" --model <phx-compiled-model-name>
```

The runner wires the full VitisAI runtime: `dd_root`, `dd_cache`, the
`libcustom_op_library.so` custom-ops registration, and a `vaip.json` pinned to
`device: phx`. It will run the moment a compiled PHX embedding model is placed
under `~/.local/share/ryzen-ai-models/<name>/`.

## 5. The one gate: model compilation

The public Linux wheels are a **deployment-only** build. Running a *pre-compiled*
model works; compiling an ONNX graph to the NPU at runtime emits:

```
F... vaiml_compile.cpp:633] Model compilation is not supported in a
deployment only installation. Please compile the model with a full installation.
```

The full compiler ships only in AMD's **account-gated** Ryzen AI Software
installer (`account.amd.com/.../xef.html?filename=ryzen-ai-lt-1.7.1.exe`).
Two ways forward:

- **(Recommended) Publish a pre-compiled PHX embedding model.** Once one exists
  on HuggingFace (e.g. `amd/MiniLM-L6-v2-phx`), the NPU backend runs it with
  zero extra setup. The lone existing HF NPU embedding model
  (`amd/NPU-Nomic-embed-text-v1.5-ryzen-strix-cpp`) is Strix/XDNA2-only.
- **Compile your own** using AMD's installer (see §6).

## 6. Compiling an embedding ONNX to the Phoenix NPU (needs the gated installer)

1. Obtain AMD Ryzen AI Software 1.7.1 from `account.amd.com` (free AMD account).
2. Install it; it provides the full `voe` compiler + the VAIML flow.
3. Quantize the embedding ONNX to int8/bf16 with `quark` (bundled):
   ```bash
   quark --model_dir ./all-MiniLM-L6-v2 --target phx --quant_int8 ...
   ```
4. Partition + compile to the PHX `4x4` overlay; this produces
   `model_compiled.onnx` + a `cache/` dir of compiled subgraphs.
5. Place the output under `~/.local/share/ryzen-ai-models/<name>/` and run:
   ```bash
   xdna-npu embed-run "hello" --model <name>
   ```
6. **Validate against the CPU reference** (§4a): the NPU vectors must match the
   CPU vectors within quantization tolerance (cosine > 0.98 typical).

## 7. Reproducing this whole investigation from scratch

```bash
git clone https://github.com/tibrezus/xdna-npu-toolkit
xdna-npu doctor          # 1. detect + validate the NPU stack
xdna-npu enable          # 2. memlock + performance power mode
xdna-npu embed-setup     # 3. install the AMD VitisAI stack
xdna-npu embed-check     # 4. prove the EP initializes on the NPU
xdna-npu embed-export sentence-transformers/all-MiniLM-L6-v2
xdna-npu embed-run "test" --cpu --model ./all-MiniLM-L6-v2/model.onnx --tokenizer ./all-MiniLM-L6-v2
# 5. NPU path: blocked only on a compiled PHX model (§5/§6)
```

## 8. Open work / where to contribute

- **Publish a PHX-compiled embedding model** (the single highest-value unblock).
- Wire the compile step into `xdna-npu embed-compile` once the compiler is
  obtainable without an account.
- Add more architectures (E5/GTE/BGE) to the export/compile matrix.
- Benchmark NPU vs CPU vs iGPU latency for RAG-scale batch embedding.

See the issues on `tibrezus/xdna-npu-toolkit` for tracking.
