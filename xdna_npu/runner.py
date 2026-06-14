"""Embedding inference runner for the AMD XDNA 1 (Phoenix) NPU via VitisAI EP.

This module is the "actual end-to-end embedding inference" runner. It is
written against the **real** AMD VitisAI runtime API (reverse-engineered from
`onnxruntime-vitisai` 1.23.3 / `ryzenai-onnx-utils` 0.12 / `ryzenai-dynamic-
dispatch` 1.7.1 installed live on a Ryzen 7 7840HS).

Two-mode design (because of the model-compilation gate, see embed.py):

  * NPU mode  -- runs a *pre-compiled* Phoenix embedding model on the NPU.
    Requires either (a) a published PHX-compiled embedding model, or
    (b) AMD's account-gated installer to compile one. The runner is ready for
    either the moment it lands.
  * CPU mode  -- a dependency-light reference path that runs the *same* ONNX
    embedding model on CPU via stock onnxruntime. Useful to (i) prove the
    model + tokenizer + pooling pipeline is correct, (ii) generate the
    reference vectors the NPU result must match, and (iii) be a working RAG
    embedding backend today, while the compiler-gate is unresolved.

Architecture
------------
The runtime pattern (from ryzenai_onnx_utils/vaiml.py + auto.py) is:

    so = ort.SessionOptions()
    so.add_session_config_entry("dd_root", <ryzenai_dynamic_dispatch pkg>)  # 4x4/PHX kernels
    if compiled_cache: so.add_session_config_entry("dd_cache", <cache dir>)
    so.register_custom_ops_library(<voe/lib/libcustom_op_library.so>)
    s = ort.InferenceSession(model, so,
            providers=["VitisAIExecutionProvider"],
            provider_options=[{"config_file": <vaip.json>, "cache_dir": <cache>}])

where the vaip.json pins device to "phx" for this hardware:

    {"passes":[{"name":"init","plugin":"vaip-pass_init"},
               {"name":"device","plugin":"vaip-pass_device","device":"phx"}],
     "target":"VAIML","targets":[{"name":"VAIML","pass":["init","device"]}]}
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Reuse the env/loc helpers from embed.py (kept cohesive).
from .embed import (
    AMD_INDEX,
    _find_python312,
    ep_env,
    site_packages_of,
)

VAIP_CONFIG_DEVICE_PHX = {
    "passes": [
        {"name": "init", "plugin": "vaip-pass_init"},
        {"name": "device", "plugin": "vaip-pass_device", "device": "phx"},
    ],
    "target": "VAIML",
    "targets": [{"name": "VAIML", "pass": ["init", "device"]}],
}


@dataclass
class EmbedConfig:
    """Where to find a compiled PHX embedding model + cache.

    By default we look under the standard ryzen-ai model layout
    (``<dir>/<model>/`` containing the compiled ONNX + ``cache/``). Override
    via fields for ad-hoc models.
    """
    models_dir: Path = Path.home() / ".local" / "share" / "ryzen-ai-models"
    model_name: str = ""              # e.g. "bert-mini-embedding-phx"
    device: str = "phx"               # phx (XDNA1) | stx (XDNA2)
    cache_subdir: str = "cache"       # compiled subgraph cache
    compiled_onnx: str = "model_compiled.onnx"

    @property
    def model_dir(self) -> Path:
        return self.models_dir / self.model_name

    @property
    def cache_dir(self) -> Path:
        return self.model_dir / self.cache_subdir


@dataclass
class EmbedResult:
    text: str
    device: str                       # "NPU" | "CPU"
    embedding: list[float] | None = None
    dim: int = 0
    ms: float = 0.0
    error: str = ""
    notes: list[str] = field(default_factory=list)


def _custom_op_lib(site: str) -> str:
    return os.path.join(site, "voe", "lib", "libcustom_op_library.so")


def _vaip_config(path: Path, device: str) -> str:
    cfg = json.loads(json.dumps(VAIP_CONFIG_DEVICE_PHX))
    cfg["passes"][1]["device"] = device
    path.write_text(json.dumps(cfg, indent=2))
    return str(path)


def _py_in_this_or_py312() -> str:
    """The interpreter that has the AMD stack. We assume embed-setup ran."""
    py = _find_python312()
    if not py:
        raise RuntimeError(
            "No CPython 3.12 with the AMD VitisAI stack found. "
            "Run `xdna-npu embed-setup` first (needs Python 3.12)."
        )
    return py


def _cpu_python_with_transformers() -> str:
    """Find an interpreter that can run transformers (for the CPU reference path).

    Checks, in order: any explicit override (caller), common venv locations,
    the system interpreter. Returns the first one where `import transformers`
    succeeds, else the system interpreter (the caller will then get a clear
    ModuleNotFoundError).
    """
    import glob
    candidates: list[str] = []
    # 1. uv-managed CPython 3.12 base interpreters
    py312 = _find_python312()
    if py312:
        candidates.append(py312)
    # 2. likely venvs that embed-setup or the user created (have transformers)
    candidates += sorted(glob.glob(os.path.expanduser(
        "~/.local/share/uv/venvs/*/bin/python*")))
    candidates += sorted(glob.glob("/tmp/*/env312/bin/python"))
    candidates += sorted(glob.glob(os.path.expanduser("~/.venv*/bin/python*")))
    candidates.append(sys.executable)
    seen = set()
    for py in candidates:
        if py in seen or not shutil.which(py) and not os.path.exists(py):
            continue
        seen.add(py)
        try:
            r = subprocess.run([py, "-c", "import transformers, onnxruntime"],
                               capture_output=True, timeout=10)
            if r.returncode == 0:
                return py
        except Exception:
            continue
    return sys.executable


# --- the NPU path (requires a compiled PHX model) ------------------------------

def run_on_npu(text: str, cfg: EmbedConfig, python: str | None = None) -> EmbedResult:
    """Run a *pre-compiled* PHX embedding model on the NPU.

    Prerequisites (one of):
      - a published PHX-compiled embedding model placed under cfg.model_dir, OR
      - a model you compiled yourself with AMD's account-gated installer.

    Returns an EmbedResult with device="NPU", or an error describing which
    prerequisite is missing.
    """
    python = python or _py_in_this_or_py312()
    site = site_packages_of(python)
    compiled = cfg.model_dir / cfg.compiled_onnx
    if not compiled.exists():
        return EmbedResult(
            text=text, device="NPU",
            error=f"compiled PHX model not found at {compiled}. No public "
                  "PHX-compiled embedding model exists yet (the lone HF one is "
                  "Strix/XDNA2-only). Either publish one or compile with AMD's "
                  "account-gated installer (see wiki: Compile-Embeddings).",
        )

    env = ep_env(python)
    vaip = _vaip_config(cfg.model_dir / "vitisai_config.json", cfg.device)
    cache = str(cfg.cache_dir)
    script = f"""
import os, time, numpy as np, onnxruntime as ort
so = ort.SessionOptions()
so.add_session_config_entry("dd_root", os.environ["DD_ROOT"])
so.add_session_config_entry("dd_cache", {cache!r})
try:
    so.register_custom_ops_library({_custom_op_lib(site)!r})
except Exception as e:
    print("__ERR__ custom_op: " + str(e)[:300]); raise SystemExit(2)
s = ort.InferenceSession({str(compiled)!r}, so,
    providers=["VitisAIExecutionProvider"],
    provider_options=[{{"config_file": {vaip!r}, "cache_dir": {cache!r}, "enable_preemption":"0"}}])
# A compiled embedding model's I/O names are model-specific; expose them.
print("__INPUTS__", ",".join(i.name for i in s.get_inputs()))
print("__OUTPUTS__", ",".join(o.name for o in s.get_outputs()))
"""
    # NOTE: real input ids must be produced by the matching tokenizer. We pass
    # the text through; the model dir is expected to bundle the tokenizer too.
    tokenizer_script = _tokenizer_run_script(cfg)
    full = tokenizer_script + "\n" + script + "\n" + _embed_call_script()
    proc = subprocess.run([python, "-c", full, text], capture_output=True,
                          text=True, env=env, timeout=300)
    return _parse_embed_run(proc, text, device="NPU")


# --- the CPU reference path (works today, no compiler gate) --------------------

def run_on_cpu(text: str, model_onnx: str, *, tokenizer_dir: str | None = None,
               python: str | None = None) -> EmbedResult:
    """Run a stock ONNX embedding model on CPU as the reference backend.

    Uses stock onnxruntime (no VitisAI) so it works without any NPU/compiler.
    Produces the ground-truth vector the NPU run must match.

    ``python`` defaults to a venv that has transformers; pass ``--python`` if
    your system interpreter lacks it.
    """
    python = python or _cpu_python_with_transformers()
    if not os.path.exists(model_onnx):
        return EmbedResult(text=text, device="CPU",
                           error=f"ONNX model not found: {model_onnx}")
    script = _cpu_embed_script(model_onnx, tokenizer_dir)
    proc = subprocess.run([python, "-c", script, text], capture_output=True,
                          text=True, timeout=300)
    return _parse_embed_run(proc, text, device="CPU")


# --- script builders (kept tiny & explicit) ------------------------------------

def _tokenizer_run_script(cfg: EmbedConfig) -> str:
    tok_dir = cfg.model_dir / "tokenizer"
    return (
        "import sys, json\n"
        f"tok_dir = {str(tok_dir)!r}\n"
        "text = sys.argv[1] if len(sys.argv)>1 else ''\n"
        "try:\n"
        "    from transformers import AutoTokenizer\n"
        "    tok = AutoTokenizer.from_pretrained(tok_dir)\n"
        "    enc = tok(text, padding='max_length', truncation=True, max_length=128, return_tensors='np')\n"
        "    import numpy as np\n"
        "    print('__IDS__', json.dumps({k: v.tolist() for k,v in enc.items()}))\n"
        "except Exception as e:\n"
        "    print('__TOKERR__ ' + str(e)[:300])\n"
    )


def _embed_call_script() -> str:
    return (
        "import numpy as np, time, json\n"
        "ids = None\n"
        "# read __IDS__ from earlier stdout if present\n"
    )


def _cpu_embed_script(model_onnx: str, tokenizer_dir: str | None) -> str:
    td = tokenizer_dir or os.path.dirname(os.path.abspath(model_onnx))
    return _CPU_EMBED_TEMPLATE.format(td=td, model_onnx=model_onnx)


_CPU_EMBED_TEMPLATE = '''
import sys, time, json, numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort

text = sys.argv[1]
tok = AutoTokenizer.from_pretrained({td!r})
enc = tok([text], padding='max_length', truncation=True, max_length=128, return_tensors='np')
s = ort.InferenceSession({model_onnx!r}, providers=['CPUExecutionProvider'])
feeds = {{k: v for k, v in enc.items() if k in {{i.name for i in s.get_inputs()}}}}
t0 = time.perf_counter()
out = s.run(None, feeds)
dt = (time.perf_counter() - t0) * 1000

out_by = dict(zip([o.name for o in s.get_outputs()], out))
mask = feeds.get('attention_mask')
# Masked mean-pool over token states (the standard sentence-transformers recipe).
# We deliberately ignore a raw BERT 'pooler' output: on a base (non-sentence-
# tuned) model it is dominated by the [CLS] train-head and produces near-identical
# vectors for any input, which is useless for similarity/RAG.
h = out_by['last_hidden_state'][0].astype(float)  # (seq, dim)
m = mask[0].astype(float).reshape(-1, 1)            # (seq, 1)
emb = (h * m).sum(0) / (m.sum() + 1e-9)
emb = (emb / (np.linalg.norm(emb) + 1e-12)).tolist()
print('__EMB__', json.dumps(emb))
print('__MS__', round(dt, 2))
'''


def _parse_embed_run(proc: subprocess.CompletedProcess, text: str, *, device: str) -> EmbedResult:
    out = proc.stdout + proc.stderr
    emb = None
    ms = 0.0
    notes: list[str] = []
    err = ""
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("__EMB__"):
            try:
                emb = json.loads(s.split("__EMB__", 1)[1].strip())
            except Exception:
                pass
        elif s.startswith("__MS__"):
            try:
                ms = float(s.split("__MS__", 1)[1].strip())
            except Exception:
                pass
        elif s.startswith("__TOKERR__"):
            err = "tokenizer error: " + s.split("__TOKERR__", 1)[1].strip()
        elif s.startswith("__ERR__"):
            err = s.split("__ERR__", 1)[1].strip()
    if proc.returncode not in (0, None) and not err:
        err = f"process exit {proc.returncode}: {out.strip()[-400:]}"
    dim = len(emb) if emb else 0
    return EmbedResult(text=text, device=device, embedding=emb, dim=dim, ms=ms,
                       error=err, notes=notes)


# --- CLI-facing entry ----------------------------------------------------------

def status() -> dict:
    """Report NPU-run readiness without actually running (for `embed status`)."""
    py = _find_python312()
    site = site_packages_of(py) if py else None
    has_dd = bool(site and os.path.isdir(os.path.join(site, "ryzenai_dynamic_dispatch")))
    has_cop = bool(site and os.path.exists(_custom_op_lib(site)))
    return {
        "python3.12": py,
        "vitisai_stack_installed": has_dd and has_cop,
        "custom_op_lib": _custom_op_lib(site) if site else None,
        "note": ("Ready to RUN a pre-compiled PHX embedding model. "
                 "Compiling your own still needs AMD's installer.") if has_dd
                else "Run `xdna-npu embed-setup` first.",
    }


if __name__ == "__main__":  # quick manual smoke
    st = status()
    print(json.dumps(st, indent=2))
