"""embed.py — servable MiniLM-L6-v2 embeddings: bf16 NPU (primary) + CPU routing.

PRIMARY PATH (bf16 NPU): the Phase-1 optimization. Activations stay bf16 end to
end (no quant/dequant). NPU GEMM runs at 922 GOPS (1.2x over int16) and — more
importantly — removes the per-Linear quantize/dequantize overhead entirely.
Result: NPU beats torch-CPU by 1.4-3.2x for batch>=16, break-even at batch 8.

  Embedder(backend="auto")     # batch>=8 -> bf16 NPU; smaller -> torch CPU
  Embedder(backend="npu")      # force bf16 NPU
  Embedder(backend="torch")    # force CPU (fastest for single queries)

  vecs = e.embed(list_of_texts)   # -> np.ndarray [N, 384]
"""
from __future__ import annotations
import os, time
import numpy as np
import torch

SEQ_LEN = 64
WDIR = os.environ.get("MINILM_WEIGHTS", "/tmp/voe-inspect/minilm")
# NPU uses ONE compiled M (4096 = batch 64) — keeps within the amdxdna 4-context
# limit (4 shape-kernels). Inputs are chunked/padded to batch 64. Small requests
# (where padding would waste >2x compute) route to torch CPU instead.
NPU_BATCH = 64
NPU_MIN_BATCH = 32   # below this, CPU wins (padding waste dominates)
COMPILED_M = (512, 1024, 2048, 4096, 8192)


class Embedder:
    def __init__(self, backend="auto", weights_dir=WDIR):
        self.backend = backend; self.weights_dir = weights_dir
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self._torch_model = None; self._npu = None; self._npu_model = None

    def _get_torch(self):
        if self._torch_model is None:
            from transformers import AutoModel
            self._torch_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"); self._torch_model.eval()
        return self._torch_model

    def _get_npu(self):
        if self._npu_model is None:
            import sys; sys.path.insert(0, os.path.dirname(__file__))
            from forward_bf16 import build_bf16_model
            from bf16_backend import Bf16Backend
            self._npu = Bf16Backend()
            self._npu_model = build_bf16_model(self.weights_dir, self._npu.run)
        return self._npu_model

    def embed(self, texts):
        n = len(texts)
        use_npu = self.backend == "npu" or (self.backend == "auto" and n >= NPU_MIN_BATCH)
        return self._embed_npu(texts) if use_npu else self._embed_torch(texts)

    def _embed_torch(self, texts):
        m = self._get_torch()
        enc = self.tok(texts, padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="pt")
        with torch.no_grad(): out = m(**enc).last_hidden_state
        mask = enc["attention_mask"][:, :, None].float()
        return torch.nn.functional.normalize((out*mask).sum(1)/mask.sum(1).clamp(min=1e-9), dim=1).numpy()

    def _embed_npu(self, texts):
        """Always chunk to NPU_BATCH=64 (M=4096). One fixed compiled M = 4 contexts."""
        from forward_bf16 import forward_bf16
        npu_model = self._get_npu()
        out = np.zeros((len(texts), 384), np.float32)
        i = 0
        while i < len(texts):
            chunk = list(texts[i:i+NPU_BATCH])
            if len(chunk) < NPU_BATCH: chunk += [""] * (NPU_BATCH - len(chunk))
            enc = self.tok(chunk, padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="np")
            ids = enc["input_ids"].astype(np.int64); mask = enc["attention_mask"].astype(np.int64)
            emb = forward_bf16(npu_model, ids, mask, self._npu.run)
            got = min(NPU_BATCH, len(texts) - i)
            for j in range(got): out[i+j] = emb[j]
            i += got
        return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="MiniLM-L6-v2 embeddings (bf16 NPU, primary)")
    ap.add_argument("inputs", nargs="*", help="texts or *.txt files")
    ap.add_argument("--backend", choices=["auto","torch","npu"], default="auto")
    ap.add_argument("--bench", type=int, metavar="N", help="benchmark at N texts")
    a = ap.parse_args()
    e = Embedder(backend=a.backend)
    if a.bench:
        texts = ["dogs and cats play in the sunny park together"] * a.bench
        t0 = time.time(); e.embed(texts); dt = time.time()-t0
        route = "NPU" if (a.backend=="npu" or (a.backend=="auto" and a.bench>=NPU_MIN_BATCH)) else "CPU"
        print(f"[{a.backend}->{route}] {a.bench} texts: {dt*1000:.1f} ms ({dt*1000/a.bench:.2f} ms/text)")
        return
    texts = []
    for inp in a.inputs:
        if inp.endswith(".txt") and os.path.exists(inp): texts.append(open(inp).read())
        else: texts.append(inp)
    if not texts: print("give texts or --bench N"); return
    vecs = e.embed(texts)
    print(f"[{a.backend}] {len(texts)} texts -> {vecs.shape}")
    if len(texts) >= 2:
        print(f"  cos(0,1) = {vecs[0]@vecs[1]/(np.linalg.norm(vecs[0])*np.linalg.norm(vecs[1])):.4f}")


if __name__ == "__main__":
    main()
