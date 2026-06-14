# Compile Embeddings for the XDNA 1 (Phoenix) NPU

This page is the definitive guide for the **one remaining gate** on the
embedding path: turning a HuggingFace embedding model into a Phoenix-runnable
compiled model. It's referenced from `xdna-npu embed-run`'s error message.

## Status

🟡 **The compiler is AMD-account-gated.** The public Linux wheels
(`pypi.amd.com`) are *deployment-only* — they run a compiled model but cannot
compile one:

```
F... vaiml_compile.cpp:633] Model compilation is not supported in a
deployment only installation. Please compile the model with a full installation.
```

## Path A (preferred): use a pre-compiled PHX model

Once a PHX-compiled embedding model exists on HuggingFace, **no compiler is
needed** — `xdna-npu embed-run --model <name>` runs it directly.

**There is no such model published yet.** The lone HF NPU embedding model
(`amd/NPU-Nomic-embed-text-v1.5-ryzen-strix-cpp`) is Strix/XDNA2-only.
Publishing `amd/<model>-phx` is the single highest-value unblock. See issue #5.

## Path B: compile your own with the gated installer

1. **Get the installer.** Sign in to `account.amd.com` (free AMD account) and
   download the Ryzen AI Software 1.7.1 package
   (`ryzen-ai-lt-1.7.1.exe` / the Linux tarball). This provides the full `voe`
   compiler + `quark` quantizer + VAIML flow.

2. **Export the model to ONNX correctly** (do NOT skip — a naive export
   saturates outputs; see [[Home]] / the walkthrough §4a):
   ```bash
   xdna-npu embed-export sentence-transformers/all-MiniLM-L6-v2
   ```

3. **Quantize** to int8 (weights/activations) with `quark`, targeting `phx`:
   ```bash
   quark --model_dir ./all-MiniLM-L6-v2 --target phx --quant_int8 ...
   ```

4. **Partition + compile** to the PHX `4x4` overlay. The VAIML flow produces
   `model_compiled.onnx` + a `cache/` directory of compiled subgraphs.

5. **Place the output** under the models dir:
   ```
   ~/.local/share/ryzen-ai-models/<name>/
     model_compiled.onnx
     cache/
     tokenizer/          # copy the HF tokenizer here
     vitisai_config.json # device: "phx"
   ```

6. **Run + validate against the CPU reference:**
   ```bash
   xdna-npu embed-run "hello" --model <name>        # NPU
   xdna-npu embed-run "hello" --cpu --model ./all-MiniLM-L6-v2/model.onnx --tokenizer ./all-MiniLM-L6-v2
   ```
   The NPU vector must match the CPU vector within quantization tolerance
   (cosine > ~0.98 typical). This validation is exactly why we built the CPU
   reference backend first.

## Reference: the runtime API the runner uses

```python
so = ort.SessionOptions()
so.add_session_config_entry("dd_root", <ryzenai_dynamic_dispatch pkg>)   # 4x4/PHX kernels
so.add_session_config_entry("dd_cache", <cache dir>)
so.register_custom_ops_library(<voe/lib/libcustom_op_library.so>)
s = ort.InferenceSession(model, so,
    providers=["VitisAIExecutionProvider"],
    provider_options=[{"config_file": <vaip.json>, "cache_dir": <cache>, "enable_preemption":"0"}])
```
where `vaip.json` pins `device: "phx"` for this hardware.
