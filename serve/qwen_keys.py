from transformers import AutoModel
m = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
sd = m.state_dict()
print("ALL keys (sample):")
proj = [k for k in sd if "proj.weight" in k][:10]
for k in proj: print("  ", k, tuple(sd[k].shape))
print("norm keys:", [k for k in sd if "norm" in k][:8])
print("embed:", [k for k in sd if "embed" in k])
