"""xdna-npu-toolkit inference engine — llama.cpp-style embedding serving.

Subcommands (see `xdna-embed --help`):
    embed     one-shot / batch embeddings (like `llama-embedding`)
    server    OpenAI-compatible /v1/embeddings HTTP server (like `llama-server --embedding`)
    bench     benchmark backends (npu / cpu / auto)
    list      show models compiled/supported on this NPU
    info      show backend availability + NPU device status

Backends:
    npu   AMD XDNA1 NPU via IRON bf16 GEMMs (requires pre-compiled model)
    cpu   torch CPU bf16 (works for ANY HuggingFace model; the speed winner here)
    auto  pick per-batch (small->cpu, large->npu when available)

The NPU path requires GEMM xclbins compiled for a model's specific Linear shapes
(unlike llama.cpp's universal GGUF loader). See `xdna-embed list` for what is
compiled, and `xdna-embed info` for device/backend status.
"""
__version__ = "0.1.0"
