# LLM feasibility on XDNA 1

The full technical analysis lives in the **FastFlowLM fork**, where it's most
discoverable for people chasing the LLM path:

→ https://github.com/tibrezus/FastFlowLM/blob/docs/xdna1-feasibility/docs/XDNA1-FEASIBILITY.md

**Summary:** full LLM support is **not implementable** from FastFlowLM on XDNA 1
(Phoenix). All 227 committed `.xclbin` overlays are `*-NPU2` (Strix/XDNA2),
loaded as opaque XRT binaries; there is no kernel source or open AIE compiler
in the repo. The `cols < 8` guard is a symptom, not the blocker.

**Important nuance (§7 of that doc):** the LLM verdict applies only to *LLM
runtimes*. Embedding/CNN/transformer inference on XDNA 1 *is* supported via
AMD's VitisAI EP — see [docs/EMBEDDINGS-WALKTHROUGH.md](EMBEDDINGS-WALKTHROUGH.md).
