"""Model registry for the embedding engine.

A ModelSpec captures everything the engine needs to embed with a given model on
a given backend:

  - hf_id     HuggingFace id (for tokenizer + CPU weights)
  - dim       embedding dimensionality
  - max_seq   sequence length (NPU path is compiled for a FIXED seq*batch = M)
  - pooling   "mean" | "last_token" | "cls"
  - normalize L2-normalize the pooled vector (True for all sentence-embedders)
  - npu       name of the NPU adapter in backends._NPU_ADAPTERS, or None
              (None => CPU-only; the NPU can't run arbitrary uncompiled models)

Adding NPU support for a NEW model = compile its bf16 GEMM xclbins (see
`xdna-embed compile <alias>`) + write an adapter in backends.py. This is the
one place the engine differs from llama.cpp: the NPU is not a universal JIT,
it needs per-model-shape compiled kernels.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    hf_id: str
    dim: int
    max_seq: int
    pooling: str            # "mean" | "last_token" | "cls"
    normalize: bool
    npu: str | None         # adapter name, or None if not NPU-supported


# Registry. `npu` is set only for models with compiled xclbins on THIS machine.
REGISTRY: dict[str, ModelSpec] = {
    "minilm": ModelSpec(
        alias="minilm",
        hf_id="sentence-transformers/all-MiniLM-L6-v2",
        dim=384, max_seq=64, pooling="mean", normalize=True,
        npu="minilm",
    ),
    "qwen3-0.6b": ModelSpec(
        alias="qwen3-0.6b",
        hf_id="Qwen/Qwen3-Embedding-0.6B",
        dim=1024, max_seq=64, pooling="last_token", normalize=True,
        npu="qwen",
    ),
    # CPU-only entries (no compiled NPU kernels yet; listed for convenience):
    "bge-small": ModelSpec(
        alias="bge-small", hf_id="BAAI/bge-small-en-v1.5",
        dim=384, max_seq=512, pooling="cls", normalize=True, npu=None,
    ),
    "e5-small": ModelSpec(
        alias="e5-small", hf_id="intfloat/e5-small-v2",
        dim=384, max_seq=512, pooling="mean", normalize=True, npu=None,
    ),
}


def resolve(model: str) -> tuple[ModelSpec, bool]:
    """Resolve a `-m` argument to (ModelSpec, is_arbitrary_hf_id).

    If `model` matches a registry alias, return that spec. Otherwise treat it as
    an arbitrary HuggingFace id (CPU-only, mean pooling, seq 512).
    """
    if model in REGISTRY:
        return REGISTRY[model], False
    # arbitrary HF id -> CPU-only spec
    return ModelSpec(
        alias=model, hf_id=model, dim=0, max_seq=512,
        pooling="mean", normalize=True, npu=None,
    ), True
