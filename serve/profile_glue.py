"""profile_glue.py — break down the 49ms glue (attention/layernorm/softmax/gelu).

If attention matmuls are a fat target, moving them to NPU is the lever.
If glue is spread thin, the forward is genuinely compute-bound and Phase 1 was the win.
"""
import sys, time
sys.path.insert(0, "/tmp/xdna-npu-toolkit/serve")
import numpy as np, torch
from forward_bf16 import _ln

B, S, H, NH, HD, FFN = 64, 64, 384, 12, 32, 1536
device = torch.device("cpu")

x384 = torch.randn(B,S,H).bfloat16(); x1536 = torch.randn(B,S,FFN).bfloat16()
q = torch.randn(B,NH,S,HD).bfloat16()
ext = torch.zeros(1,1,S,S).bfloat16()
w = torch.ones(H).bfloat16(); bb = torch.zeros(H).bfloat16()

def bench(name, fn, n=50):
    for _ in range(10): fn()
    t0=time.time()
    for _ in range(n): fn()
    print(f"  {name:32} {(time.time()-t0)/n*1000:6.3f} ms  (x6 = {(time.time()-t0)/n*1000*6:.1f} ms/forward)")

print(f"=== glue op costs at batch{B} (×6 layers) ===")
bench("layernorm [B,S,384]", lambda: _ln(x384, w, bb))
scores = (q @ q.transpose(-2,-1))
bench("attn scores [B,12,S,32]@[B,12,32,S]", lambda: q @ q.transpose(-2,-1))
attn = torch.softmax(scores.float(),-1).bfloat16()
bench("softmax [B,12,S,S]", lambda: torch.softmax(scores.float(),-1).bfloat16())
bench("attn ctx [B,12,S,S]@[B,12,S,32]", lambda: attn @ q)
bench("gelu [B,S,1536]", lambda: torch.nn.functional.gelu(x1536))
bench("reshape/permute (split heads)", lambda: x384.reshape(B,S,NH,HD).permute(0,2,1,3))
bench("bias-add [B,S,384]", lambda: x384 + w)
# pooling
m = torch.ones(B,S,1).bfloat16()
bench("pooling (sum+norm)", lambda: torch.nn.functional.normalize((x384*m).sum(1),dim=1))
