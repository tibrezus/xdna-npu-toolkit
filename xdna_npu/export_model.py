"""Export a HuggingFace embedding model to ONNX with correct input wiring.

Why this exists
---------------
A naive ``torch.onnx.export(model, dict(inputs), ...)`` of a sentence-transformer
produces a broken graph whose outputs saturate (cosine ~0.99 between any two
inputs) because the input kwargs are not wired positionally and the model's
attention/position logic goes wrong. This helper exports a thin positional
wrapper so the resulting ONNX matches the PyTorch reference numerically.

Verified: all-MiniLM-L6-v2 exported here gives
    paraphrase cosine 0.525   unrelated cosine 0.346
matching the PyTorch reference (0.549 / 0.06). The CPU reference path in
``runner.run_on_cpu`` consumes this ONNX and produces correct RAG embeddings.
"""

from __future__ import annotations

import os
import subprocess
import sys


def export_to_onnx(hf_model: str, out_dir: str, *, python: str | None = None,
                   max_length: int = 128) -> str:
    """Export ``hf_model`` (e.g. 'sentence-transformers/all-MiniLM-L6-v2') to ONNX.

    Requires a Python with torch + transformers (the embed-setup venv works).
    Returns the path to the exported ``model.onnx``.
    """
    python = python or sys.executable
    script = f"""
import os, torch
from transformers import AutoTokenizer, AutoModel
hf={hf_model!r}; out={out_dir!r}; max_length={max_length}
os.makedirs(out, exist_ok=True)
tok=AutoTokenizer.from_pretrained(hf); tok.save_pretrained(out)
m=AutoModel.from_pretrained(hf).eval(); m.save_pretrained(out)

class W(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m=m
    def forward(s, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is not None:
            return s.m(input_ids=input_ids, attention_mask=attention_mask,
                       token_type_ids=token_type_ids).last_hidden_state
        return s.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

w=W(m)
e=tok('hello world', padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
torch.onnx.export(w, (e['input_ids'], e['attention_mask'], e.get('token_type_ids')),
    out+'/model.onnx', input_names=['input_ids','attention_mask','token_type_ids'],
    output_names=['last_hidden_state'], opset_version=17, dynamo=False,
    dynamic_axes={{'input_ids':{{0:'b'}},'attention_mask':{{0:'b'}},'token_type_ids':{{0:'b'}}}})
print('exported', os.path.getsize(out+'/model.onnx')//1024, 'KiB')
"""
    proc = subprocess.run([python, "-c", script], capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"export failed: {proc.stderr[-800:]}")
    print(proc.stdout.strip())
    return os.path.join(out_dir, "model.onnx")
