"""embed.py — servable MiniLM-L6-v2 embeddings on the 7840HS.

Two backends:
  - "torch"  (default): fast, correct, ~10 ms/text. The practical choice.
  - "npu":   hybrid (Linear GEMMs on Phoenix NPU, glue on CPU). Correct
             (bit-identical to int16 CPU) but currently SLOWER than torch for
             this small model — see BENCHMARK.md. Provided for NPU research and
             as the foundation for larger models / fusion work.

Usage:
  from serve.embed import Embedder
  e = Embedder(backend="torch")        # or "npu"
  vecs = e.embed(["hello world", "another text"])   # -> np.ndarray [N, 384]

CLI:
  python -m serve.embed --backend torch *.txt
"""
from __future__ import annotations
import os
import numpy as np

SEQ_LEN = 64
WDIR = os.environ.get("MINILM_WEIGHTS", "/tmp/voe-inspect/minilm")


class Embedder:
    def __init__(self, backend="torch", weights_dir=WDIR):
        self.backend = backend
        self.weights_dir = weights_dir
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        if backend == "torch":
            import torch
            from transformers import AutoModel
            self._dev = "cpu"
            self._model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self._model.eval()
        elif backend == "npu":
            import sys
            sys.path.insert(0, os.path.dirname(__file__))
            sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
            os.environ.setdefault("MINILM_GEMM_DIR", "/tmp/iron/minilm-gemms")
            from minilm_forward import build_model
            from npu_backend import NpuGemmPool
            self._model = build_model(weights_dir, backend=NpuGemmPool.run)
            self._M_compiled = 512   # 4-col: batch must satisfy B*SEQ_LEN == 512 -> B=8
        else:
            raise ValueError(f"unknown backend {backend!r}; use 'torch' or 'npu'")

    def embed(self, texts):
        if self.backend == "torch":
            return self._embed_torch(texts)
        return self._embed_npu(texts)

    def _embed_torch(self, texts):
        import torch
        enc = self.tok(texts, padding="max_length", truncation=True,
                       max_length=SEQ_LEN, return_tensors="pt")
        with torch.no_grad():
            out = self._model(**enc).last_hidden_state          # [N,S,H]
        mask = enc["attention_mask"][:, :, None].float()
        pooled = (out * mask).sum(1) / mask.sum(1).clip(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, dim=1)
        return pooled.numpy()

    def _embed_npu(self, texts):
        # NPU GEMM compiled for M = B*SEQ_LEN == 512 -> batch exactly 8 (pad/truncate)
        B = self._M_compiled // SEQ_LEN
        from minilm_forward import forward
        results = [None] * len(texts)
        for chunk_start in range(0, len(texts), B):
            chunk = texts[chunk_start:chunk_start + B]
            if len(chunk) < B:
                chunk = chunk + [""] * (B - len(chunk))          # pad to full batch
            enc = self.tok(chunk, padding="max_length", truncation=True,
                           max_length=SEQ_LEN, return_tensors="np")
            ids = enc["input_ids"].astype(np.int64)
            mask = enc["attention_mask"].astype(np.int64)
            emb = forward(self._model, ids, mask)
            for i in range(min(B, len(texts) - chunk_start)):
                results[chunk_start + i] = emb[i]
        return np.stack(results)


def main():
    import argparse, glob, time
    ap = argparse.ArgumentParser(description="Embed texts with MiniLM-L6-v2 (torch or NPU backend)")
    ap.add_argument("inputs", nargs="+", help="text strings or *.txt files")
    ap.add_argument("--backend", choices=["torch", "npu"], default="torch")
    args = ap.parse_args()

    texts = []
    for inp in args.inputs:
        if inp.endswith(".txt") and os.path.exists(inp):
            texts.append(open(inp).read())
        else:
            texts.append(inp)
    e = Embedder(backend=args.backend)
    t0 = time.time()
    vecs = e.embed(texts)
    dt = time.time() - t0
    print(f"[{args.backend}] {len(texts)} texts in {dt*1000:.1f} ms ({dt*1000/len(texts):.1f} ms/text)")
    print(f"  shape={vecs.shape}  |vec[0]|={np.linalg.norm(vecs[0]):.4f}")
    if len(texts) >= 2:
        cos = vecs[0] @ vecs[1] / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]))
        print(f"  cos(text[0], text[1]) = {cos:.4f}")


if __name__ == "__main__":
    main()
