"""profile_fast.py — where does forward_fast spend time now?"""
import sys, time, os
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "/home/tib/projects/xdna-iron/designs")
os.environ.setdefault("MINILM_GEMM_DIR", "/tmp/iron/minilm-gemms")
import numpy as np, torch
from transformers import AutoTokenizer
from forward_fast import build_fast_model, N_HEADS, HEAD_DIM, _ln
from npu_backend import NpuGemmPool

WDIR="/tmp/voe-inspect/minilm"; tok=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
enc=tok(["cats and dogs"]*8, padding="max_length", truncation=True, max_length=64, return_tensors="np")
ids=enc["input_ids"].astype(np.int64); mask=enc["attention_mask"].astype(np.int64)
m=build_fast_model(WDIR, NpuGemmPool.run)

# time individual glue ops at the sizes they run
print("=== glue op costs at batch8 sizes ===")
x384 = torch.randn(8,64,384); x1536 = torch.randn(8,64,1536)
def bench(name, fn, n=50):
    for _ in range(5): fn()
    t0=time.time()
    for _ in range(n): fn()
    print(f"  {name:28} {(time.time()-t0)/n*1000:6.3f} ms  (x6 layers = {(time.time()-t0)/n*1000*6:.1f} ms)")

bench("gelu [8,64,1536]", lambda: torch.nn.functional.gelu(x1536))
bench("layernorm [8,64,384]", lambda: _ln(x384, np.ones(384,np.float32), np.zeros(384,np.float32)))
q=torch.randn(8,12,64,32)
bench("attn scores matmul", lambda: q@q.transpose(-2,-1))
attn=torch.softmax(q@q.transpose(-2,-1),-1)
bench("attn ctx matmul", lambda: attn@q)

# the Linear path (quant+NPU+dequant) - time one layer's 6 linears
print("\n=== FastLinear (quant+NPU GEMM+dequant) per op ===")
ly=m["layers"][0]
def bench2(name, lin, xin, n=50):
    for _ in range(5): lin(xin)
    t0=time.time()
    for _ in range(n): lin(xin)
    print(f"  {name:28} {(time.time()-t0)/n*1000:6.3f} ms")
bench2("q/k/v/o [8,64,384->384]", ly["q"], x384)
bench2("f1 [8,64,384->1536]", ly["f1"], x384)
ctx=torch.randn(8,64,1536)
bench2("f2 [8,64,1536->384]", ly["f2"], ctx)

# isolated: quant overhead
def quant_cost():
    xf=x384.reshape(-1,384).numpy()
    amax=np.abs(xf).max(); xs=np.float32(amax/32767.0)
    return np.round(xf/xs).astype(np.int16)
bench("quant_i16 [512,384]", quant_cost)
