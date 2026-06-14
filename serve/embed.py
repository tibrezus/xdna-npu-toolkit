"""embed.py — servable MiniLM-L6-v2 embeddings: smart NPU/CPU routing.

AUTO-ROUTING (the win): for batched workloads (RAG indexing), the NPU beats
torch-CPU by 1.2-1.37x at batch>=32. For small batches it loses. So the "auto"
backend routes by batch size and chunks large requests into NPU-optimal batches.

  Embedder(backend="auto")     # routes: small batch -> CPU, large -> NPU (fused)
  Embedder(backend="torch")    # force CPU (fastest for single queries)
  Embedder(backend="npu")      # force NPU (for research / batched)

  vecs = e.embed(list_of_texts)

CLI:
  python -m serve.embed --backend auto *.txt
  python -m serve.embed --bench N      # benchmark routing at N texts
"""
from __future__ import annotations
import os, time
import numpy as np
import torch

SEQ_LEN = 64
WDIR = os.environ.get("MINILM_WEIGHTS", "/tmp/voe-inspect/minilm")
# NPU beats CPU at batch >= ~24; sweet spot ~64. Process large jobs in chunks of 64.
NPU_BATCH_CHUNK = 64
NPU_MIN_BATCH = 24


class Embedder:
    def __init__(self, backend="auto", weights_dir=WDIR):
        self.backend = backend; self.weights_dir = weights_dir
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self._torch_model = None
        self._npu_model = None

    def _get_torch(self):
        if self._torch_model is None:
            from transformers import AutoModel
            self._torch_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self._torch_model.eval()
        return self._torch_model

    def _get_npu(self):
        if self._npu_model is None:
            import sys; sys.path.insert(0, os.path.dirname(__file__))
            sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
            os.environ.setdefault("MINILM_GEMM_DIR", "/tmp/iron/minilm-gemms")
            from forward_fused import build_fused_model
            from pooled_backend import MultiMBackend
            self._npu = MultiMBackend()
            self._npu_model = build_fused_model(self.weights_dir, self._npu.run)
        return self._npu_model

    def embed(self, texts):
        n = len(texts)
        use_npu = self.backend == "npu" or (self.backend == "auto" and n >= NPU_MIN_BATCH)
        if use_npu:
            return self._embed_npu(texts)
        return self._embed_torch(texts)

    def _embed_torch(self, texts):
        m = self._get_torch()
        enc = self.tok(texts, padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="pt")
        with torch.no_grad(): out = m(**enc).last_hidden_state
        mask = enc["attention_mask"][:, :, None].float()
        pooled = torch.nn.functional.normalize((out*mask).sum(1)/mask.sum(1).clamp(min=1e-9), dim=1)
        return pooled.numpy()

    def _embed_npu(self, texts):
        from forward_fused import forward_fused
        npu_model = self._get_npu()
        out = np.zeros((len(texts), 384), np.float32)
        # chunk into NPU_BATCH_CHUNK (pad last chunk to compile cleanly at a known M)
        for s in range(0, len(texts), NPU_BATCH_CHUNK):
            chunk = texts[s:s+NPU_BATCH_CHUNK]
            # round up to nearest compiled M that's >= chunk size (512,1024,...,8192)
            need = len(chunk) * SEQ_LEN
            cM = min([M for M in (512,1024,2048,4096,8192) if M >= need], default=8192)
            cb = cM // SEQ_LEN
            if len(chunk) < cb:
                chunk = list(chunk) + [""] * (cb - len(chunk))
            enc = self.tok(chunk, padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="np")
            ids = enc["input_ids"].astype(np.int64); mask = enc["attention_mask"].astype(np.int64)
            emb = forward_fused(npu_model, ids, mask, self._npu.run)
            for i in range(min(cb, len(texts)-s)):
                out[s+i] = emb[i]
        return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="MiniLM-L6-v2 embeddings (auto NPU/CPU routing)")
    ap.add_argument("inputs", nargs="*", help="texts or *.txt files")
    ap.add_argument("--backend", choices=["auto","torch","npu"], default="auto")
    ap.add_argument("--bench", type=int, metavar="N", help="benchmark routing at N texts")
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
    if not texts:
        print("give some texts, or --bench N"); return
    vecs = e.embed(texts)
    print(f"[{a.backend}] {len(texts)} texts -> {vecs.shape}")
    if len(texts) >= 2:
        print(f"  cos(0,1) = {vecs[0]@vecs[1]/(np.linalg.norm(vecs[0])*np.linalg.norm(vecs[1])):.4f}")


if __name__ == "__main__":
    main()
