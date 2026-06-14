# xdna-npu-toolkit wiki

Detect, validate, enable, and run inference on the **AMD XDNA NPU on Linux** —
with an honest, machine-specific verdict on what's feasible.

## Start here

- **[Embeddings walkthrough](https://github.com/tibrezus/xdna-npu-toolkit/blob/main/docs/EMBEDDINGS-WALKTHROUGH.md)** — the complete step-by-step record of getting embedding/transformer inference running on a Ryzen 7 7840HS (Phoenix, XDNA 1).
- **[LLM feasibility](https://github.com/tibrezus/FastFlowLM/blob/docs/xdna1-feasibility/docs/XDNA1-FEASIBILITY.md)** — why LLMs don't run on XDNA 1 (FastFlowLM fork).

## The headline

| Workload | XDNA 1 (Phoenix) | XDNA 2 (Strix) |
|---|---|---|
| **LLMs** (FastFlowLM/Lemonade) | ❌ not supported by any runtime | ✅ |
| **Embeddings / CNN / transformer** (VitisAI EP) | ✅ runtime supports it | ✅ |
| **Compiling your own model** | ⚠️ compiler is AMD-account-gated | ⚠️ same |

For embeddings, the only gate is *model compilation*. See
[[Compile-Embeddings]].

## Quick commands

```bash
xdna-npu doctor          # detect + validate the NPU stack
xdna-npu enable          # memlock fix + NPU performance power mode
xdna-npu embed-setup     # install the AMD VitisAI stack (Python 3.12)
xdna-npu embed-check     # prove the VitisAI EP initializes on your NPU
xdna-npu embed-export sentence-transformers/all-MiniLM-L6-v2
xdna-npu embed-run "hello" --cpu --model ./all-MiniLM-L6-v2/model.onnx --tokenizer ./all-MiniLM-L6-v2
```

## See also

- Issues track every step of the investigation: https://github.com/tibrezus/xdna-npu-toolkit/issues
- The companion LLM analysis: https://github.com/tibrezus/FastFlowLM
