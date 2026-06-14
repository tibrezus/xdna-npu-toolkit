import sys, time; sys.path.insert(0, ".")
from embed import Embedder
e_cpu = Embedder(backend="torch")
e_auto = Embedder(backend="auto")
for B in [8, 64, 128]:
    texts = ["dogs and cats play in the park"] * B
    e_cpu.embed(texts); e_auto.embed(texts)
    t0 = time.time(); e_cpu.embed(texts); cpu = time.time()-t0
    t0 = time.time(); e_auto.embed(texts); auto = time.time()-t0
    route = "NPU(bf16)" if B >= 8 else "CPU"
    print(f"  {B} texts: CPU-only {cpu*1000:.0f}ms  auto(->{route}) {auto*1000:.0f}ms  {cpu/auto:.2f}x")
