"""gelu_options.py — find a fast, correct GELU."""
import time, numpy as np
x = np.random.randn(8,64,1536).astype(np.float32)

def bench(name, fn):
    for _ in range(5): fn(x)
    t0=time.time()
    for _ in range(100): fn(x)
    print(f"  {name:30} {(time.time()-t0)/100*1000:7.3f} ms")

from scipy.special import erf
SQ2 = np.float32(np.sqrt(2.0))
def gelu_scipy(x): return 0.5*x*(1.0+erf(x/SQ2))

def gelu_tanh(x):
    # tanh approx (used by many BERT/GPT models, incl. some MiniLM exports)
    c = np.float32(np.sqrt(2.0/np.pi))
    return 0.5*x*(1.0+np.tanh(c*(x+0.044715*np.power(x,3).astype(np.float32))))

def gelu_tanh_einsum(x):
    c = np.float32(0.7978845608028654)  # sqrt(2/pi)
    return 0.5*x*(1.0+np.tanh(c*(x+0.044715*x*x*x)))

import torch
def gelu_torch(x):
    return torch.nn.functional.gelu(torch.from_numpy(x)).numpy()

print("=== GELU on [8,64,1536] float32 (FFN1 activation, 6x per forward) ===")
bench("scipy erf (current)", gelu_scipy)
bench("tanh approx (numpy)", gelu_tanh)
bench("tanh (no power)", gelu_tanh_einsum)
bench("torch F.gelu (exact erf)", gelu_torch)

# semantic difference: tanh vs erf
d = np.abs(gelu_tanh(x) - gelu_scipy(x)).max()
print(f"\ntanh vs erf max|diff|: {d:.5f}  (tanh is the standard GPT-2/BERT-approx; negligible for embeddings)")

# is scipy erf the issue or the temporaries?
def erf_only(x): return erf(x)
bench("erf alone (scipy)", erf_only)
