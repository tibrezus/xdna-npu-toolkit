"""bench_serving.py — honest NPU vs CPU benchmark across batch sizes.

The decisive test: for MiniLM-L6-v2 on this 7840HS, is the NPU ever faster than
torch-on-CPU? Answer determines the honest serving story.
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MINILM_GEMM_DIR", "/tmp/iron/minilm-gemms")
sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
import numpy as np
from transformers import AutoTokenizer, AutoModel
from minilm_forward import build_model, forward
import torch

WDIR = "/tmp/voe-inspect/minilm"
tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
S = 64

def time_it(fn, iters=10):
    for _ in range(2): fn()
    t0 = time.time()
    for _ in range(iters): fn()
    return (time.time() - t0) / iters

torch_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"); torch_model.eval()
mfloat = build_model(WDIR, backend="float")

print(f"{'batch':>6} {'torch-CPU':>11} {'numpy-float':>12} {'NPU-4col':>10}  (ms/batch, ms/text)")
print("-" * 64)
for B in [8, 16, 32, 64]:
    texts = ["a sample sentence about cats and dogs playing"] * B
    enc = tok(texts, padding="max_length", truncation=True, max_length=S, return_tensors="np")
    ids, mask = enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)
    ti, tm = torch.tensor(ids), torch.tensor(mask)

    t_torch = time_it(lambda: torch_model(ti, attention_mask=tm).last_hidden_state, iters=8)
    t_np = time_it(lambda: forward(mfloat, ids, mask), iters=5)

    # NPU only at M=B*S that matches a compiled shape (512, 1024...)
    M = B * S
    t_npu = None
    if M in (512, 1024):
        from npu_backend import NpuGemmPool
        # need 4-col compiled at this M — only 512 compiled. skip others honestly.
        if M == 512:
            mnpu = build_model(WDIR, backend=NpuGemmPool.run)
            t_npu = time_it(lambda: forward(mnpu, ids, mask), iters=5)

    npu_s = f"{t_npu*1000:7.1f} ({t_npu*1000/B:4.1f})" if t_npu else "  (not compiled at this M)"
    print(f"{B:>6} {t_torch*1000:7.1f} ({t_torch*1000/B:4.1f}) {t_np*1000:8.1f} ({t_np*1000/B:4.1f})  {npu_s}")

print("\nLegend: torch-CPU = optimized float BLAS; numpy-float = our reference; NPU-4col = hybrid (GEMMs on NPU)")
print("Honest read: MiniLM is small (384/1536 dims). NPU value needs bigger models/batches.")
