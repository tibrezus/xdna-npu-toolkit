"""Backends for the embedding engine.

Three backends behind one interface (`embed(texts) -> np.ndarray [N, dim]`):

  CpuBackend   torch CPU bf16, ANY HuggingFace model. Correct per-model pooling
               (mean / last_token / cls). This is the speed winner on this
               Zen 4 box vs the NPU1 (see docs/CRITICAL-CORRECTION.md).
  NpuBackend   AMD XDNA1 NPU via IRON bf16 GEMMs. Dispatches to a model-specific
               adapter (minilm / qwen) that wraps the hand-written forward. Only
               works for registered, pre-compiled models — the NPU is not a
               universal JIT.
  make_backend()  factory: resolves "npu"/"cpu"/"auto" + a ModelSpec.

Honest defaults: `auto` picks NPU when the model is compiled for it AND the batch
is large enough to amortise the host/dispatch overhead; otherwise CPU. For pure
speed on this wall-powered machine, `--backend cpu` is usually fastest.
"""
from __future__ import annotations
import os
import sys
import time
from glob import glob
from typing import Protocol

import numpy as np
import torch
from torch.nn import functional as F

import sys as _sys
import os as _os

# IMPORTANT: pyxrt (the NPU/XRT runtime) ships in the Arch SYSTEM site-packages,
# not in our venv. We must NOT put that dir on PYTHONPATH before importing torch,
# because it contains Arch's slower `torch` build which would SHADOW our venv's
# fast `torch+cpu` (a ~5x slowdown measured). torch is already imported above
# (locked to the venv build), so appending the system dir now is safe: pyxrt
# resolves, but torch does not get re-resolved/replaced.
_PY = f"{_sys.version_info.major}.{_sys.version_info.minor}"
for _sp in (_os.environ.get("XDNA_SYSTEM_SITE"), f"/usr/lib/python{_PY}/site-packages"):
    if _sp and _os.path.isdir(_sp) and _sp not in _sys.path:
        _sys.path.append(_sp)

from .models import ModelSpec

# Process-wide singleton NPU bf16 GEMM backend shared across all bert384 models
# (they use identical GEMM shapes, so identical kernels/contexts). Keeps the
# amdxdna 4-hw_context budget satisfied while letting minilm+bge+e5 coexist.
_BF16_SINGLETON = None

# The NPU is a single serial resource: pooled device BOs + at most one in-flight
# batch. When multiple models share the singleton backend (or qwen uses its own),
# concurrent worker threads would race on the device. This lock serialises ALL
# NPU inference regardless of how many backends/workers exist.
import threading as _threading
_NPU_LOCK = _threading.Lock()

# NPU forward code lives in the repo's serve/ dir; add it to the path.
_SERVE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "serve")
if _SERVE not in sys.path:
    sys.path.insert(0, _SERVE)

# ─── device / env helpers ────────────────────────────────────────────────────

NPU_DEVICE_NODE = "/dev/accel/accel0"


def npu_available() -> bool:
    """True iff the amdxdna device node exists (best-effort, no XRT import)."""
    return os.path.exists(NPU_DEVICE_NODE)


# ─── pooling ─────────────────────────────────────────────────────────────────

def pool(last_hidden: torch.Tensor, mask: torch.Tensor, spec: ModelSpec) -> torch.Tensor:
    """Pool [B,S,H] hidden states -> [B,H] per spec.pooling."""
    if spec.pooling == "mean":
        m = mask[:, :, None].to(last_hidden.dtype)
        return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
    if spec.pooling == "cls":
        return last_hidden[:, 0]
    if spec.pooling == "last_token":
        seq_lens = mask.sum(1) - 1            # last non-pad index
        return last_hidden[torch.arange(last_hidden.shape[0]), seq_lens]
    raise ValueError(f"unknown pooling '{spec.pooling}'")


# ─── backend interface ───────────────────────────────────────────────────────

class EmbedBackend(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...
    def warmup(self) -> None: ...
    @property
    def dim(self) -> int: ...


# ─── CPU backend (torch bf16, general) ───────────────────────────────────────

class CpuBackend:
    """torch CPU bf16 — works for any HF model. The fair-precision speed winner."""

    name = "cpu"

    def __init__(self, spec: ModelSpec, dtype=torch.bfloat16):
        from transformers import AutoModel, AutoTokenizer
        self.spec = spec
        self.dtype = dtype
        self.tok = AutoTokenizer.from_pretrained(spec.hf_id)
        self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModel
            t0 = time.time()
            self._model = AutoModel.from_pretrained(self.spec.hf_id).to(self.dtype)
            self._model.eval()
            self._load_ms = (time.time() - t0) * 1000
        return self._model

    @property
    def dim(self) -> int:
        return self.spec.dim or self._load().config.hidden_size

    def warmup(self):
        self._load()
        with torch.no_grad():
            self.embed(["warmup"])

    def embed(self, texts: list[str]) -> np.ndarray:
        m = self._load()
        enc = self.tok(
            texts, padding="max_length", truncation=True,
            max_length=self.spec.max_seq, return_tensors="pt",
        )
        with torch.no_grad():
            out = m(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        pooled = pool(out, enc["attention_mask"], self.spec).float()
        if self.spec.normalize:
            pooled = F.normalize(pooled, dim=-1)
        return pooled.numpy()


# ─── NPU backend (registry dispatch) ─────────────────────────────────────────

class NpuBackend:
    """AMD XDNA1 NPU via IRON bf16 GEMMs. Dispatches to a registered adapter."""

    name = "npu"

    def __init__(self, spec: ModelSpec):
        if not npu_available():
            raise RuntimeError(
                f"NPU device {NPU_DEVICE_NODE} not found. Run `xdna-npu doctor` "
                f"and `xdna-npu enable` (needs root).")
        if spec.npu is None:
            raise RuntimeError(
                f"model '{spec.alias}' has no compiled NPU kernels. The NPU is not "
                f"a universal JIT — see `xdna-embed compile {spec.alias}`. "
                f"Use --backend cpu for this model.")
        if spec.npu not in _NPU_ADAPTERS:
            raise RuntimeError(f"NPU adapter '{spec.npu}' not implemented.")
        self.spec = spec
        self._adapter = _NPU_ADAPTERS[spec.npu](spec)
        self._ready = False

    @property
    def dim(self) -> int:
        return self.spec.dim

    def warmup(self):
        self._adapter.load()
        self._ready = True
        with torch.no_grad():
            self.embed(["warmup"] * 4)

    def embed(self, texts: list[str]) -> np.ndarray:
        self._adapter.load(); self._ready = True
        # Serialise ALL NPU inference: the device has pooled BOs and can run one
        # batch at a time. Multiple model workers share the singleton backend.
        with _NPU_LOCK:
            return self._adapter.embed(self.spec, texts)


# ─── NPU adapters (one per compiled model family) ────────────────────────────

class _NpuAdapter(Protocol):
    def __init__(self, spec: ModelSpec) -> None: ...
    def load(self) -> None: ...
    def embed(self, spec: ModelSpec, texts: list[str]) -> np.ndarray: ...


class _Bert384Adapter:
    """Any 384-dim BERT sentence model on NPU (MiniLM / BGE-small / E5-small).

    These all share the SAME bf16 GEMM shapes (qkv 384->1152, o 384->384,
    ffn1 384->1536, ffn2 1536->384), so one set of compiled xclbins serves
    the whole family. Layer count + pooling come from the ModelSpec / weights.
    The Bf16Backend is a process-wide SINGLETON, so minilm + bge + e5 can all be
    resident at once using only 4 NPU contexts (the amdxdna driver limit).
    """
    BATCH = 64  # compiled M=4096 = batch64 x seq64

    def __init__(self, spec):
        self._spec = spec
        self._model = self._tok = None

    @staticmethod
    def _shared_backend():
        """Process-wide singleton Bf16Backend (shared kernels/contexts)."""
        global _BF16_SINGLETON
        if _BF16_SINGLETON is None:
            from bf16_backend import Bf16Backend
            _BF16_SINGLETON = Bf16Backend()
        return _BF16_SINGLETON

    def _resolve_weights(self, spec):
        """spec.weights_dir -> env XDNA_WEIGHTS_<ALIAS> -> HF cache download."""
        env_key = "XDNA_WEIGHTS_" + spec.alias.upper().replace("-", "_")
        for cand in (spec.weights_dir, os.environ.get(env_key)):
            if cand and os.path.exists(os.path.join(cand, "model.safetensors")):
                return cand
        # fall back to HF cache (download if needed)
        from huggingface_hub import snapshot_download
        p = snapshot_download(repo_id=spec.hf_id, allow_patterns=[
            "config.json", "model.safetensors", "tokenizer.json",
            "tokenizer_config.json", "vocab.txt"])
        return p

    def load(self):
        if self._model is not None:
            return
        from forward_bf16 import build_bf16_model
        from transformers import AutoTokenizer
        wdir = self._resolve_weights(self._spec)
        self._tok = AutoTokenizer.from_pretrained(self._spec.hf_id)
        self._model = build_bf16_model(
            wdir, self._shared_backend().run, pooling=self._spec.pooling)

    def embed(self, spec, texts):
        from forward_bf16 import forward_bf16
        self.load()
        out = np.zeros((len(texts), spec.dim), np.float32)
        i = 0
        while i < len(texts):
            chunk = list(texts[i:i + self.BATCH])
            if len(chunk) < self.BATCH:
                chunk += [""] * (self.BATCH - len(chunk))   # pad to compiled M
            enc = self._tok(chunk, padding="max_length", truncation=True,
                            max_length=spec.max_seq, return_tensors="np")
            ids = enc["input_ids"].astype(np.int64)
            mask = enc["attention_mask"].astype(np.int64)
            emb = forward_bf16(self._model, ids, mask, self._shared_backend().run)
            got = min(self.BATCH, len(texts) - i)
            out[i:i + got] = emb[:got]
            i += got
        return out


class _QwenAdapter:
    """Qwen3-Embedding-0.6B on NPU: decoder forward (RMSNorm/GQA/SwiGLU, last-token pool)."""
    BATCH = 64  # compiled M=4096

    def __init__(self, spec):
        self._spec = spec
        self._model = self._weights = self._npu = self._spath = self._tok = None

    def _find_safetensors(self):
        if self._spath:
            return self._spath
        env = os.environ.get("QWEN_WEIGHTS")
        if env and os.path.exists(env):
            self._spath = env; return env
        hits = glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/"
            "snapshots/*/model.safetensors"))
        if hits:
            self._spath = hits[0]; return self._spath
        raise RuntimeError(
            "Qwen3-Embedding-0.6B weights not found. Set QWEN_WEIGHTS=<model.safetensors> "
            "or `huggingface-cli download Qwen/Qwen3-Embedding-0.6B`.")

    def load(self):
        if self._model is not None:
            return
        from qwen_backend import QwenBf16Backend
        from qwen_forward import load_model, load_weights
        from transformers import AutoTokenizer
        sp = self._find_safetensors()
        self._tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        self._npu = QwenBf16Backend()
        self._model = load_model(sp, self._npu.run)
        self._weights = load_weights(sp)

    def embed(self, spec, texts):
        from qwen_forward import forward
        out = np.zeros((len(texts), spec.dim), np.float32)
        i = 0
        while i < len(texts):
            chunk = list(texts[i:i + self.BATCH])
            if len(chunk) < self.BATCH:
                chunk += [""] * (self.BATCH - len(chunk))
            enc = self._tok(chunk, padding="max_length", truncation=True,
                            max_length=spec.max_seq, return_tensors="np")
            ids = enc["input_ids"].astype(np.int64)
            mask = enc["attention_mask"].astype(np.int64)
            emb = forward(self._model, ids, mask, self._npu.run, self._weights)
            got = min(self.BATCH, len(texts) - i)
            out[i:i + got] = emb[:got]
            i += got
        return out


_NPU_ADAPTERS: dict[str, type[_NpuAdapter]] = {
    "bert384": _Bert384Adapter,
    "qwen": _QwenAdapter,
}


# ─── factory ─────────────────────────────────────────────────────────────────

# auto-routes to NPU only at/above this batch (padding waste dominates below).
AUTO_NPU_MIN_BATCH = 32


def make_backend(backend: str, spec: ModelSpec) -> EmbedBackend:
    """Resolve 'npu' | 'cpu' | 'auto' against a ModelSpec and build the backend.

    Note: 'auto' cannot decide at construction time (batch unknown yet), so it
    returns an AutoBackend that picks per call.
    """
    if backend == "cpu":
        return CpuBackend(spec)
    if backend == "npu":
        return NpuBackend(spec)
    if backend == "auto":
        return AutoBackend(spec)
    raise ValueError(f"unknown backend '{backend}'")


class AutoBackend:
    """Picks CPU vs NPU per-machine/model via a one-time calibration probe.

    On warmup, if the model is NPU-compiled and the device is present, it times a
    small probe batch on BOTH backends and caches the faster one for the process
    lifetime. This is correct per (machine, model) rather than a hardcoded
    threshold — on this box CPU bf16 usually wins for small models, so auto will
    pick CPU; on hardware/a model where the NPU is faster, it picks NPU.
    """
    name = "auto"

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self._cpu = None
        self._npu = None
        self._choice = None          # cached winner after calibration
        self._last_choice = None

    @property
    def dim(self) -> int:
        return self.spec.dim or self._get_cpu().dim

    def _get_cpu(self):
        if self._cpu is None:
            self._cpu = CpuBackend(self.spec)
        return self._cpu

    def warmup(self):
        self._get_cpu().warmup()
        # calibrate if NPU is available for this model
        if self.spec.npu is not None and npu_available():
            self._choice = self._calibrate()
        else:
            self._choice = "cpu"

    def _calibrate(self) -> str:
        """Time a probe batch on both backends; return the name of the faster."""
        probe = [
            "the quick brown fox jumps over the lazy dog in the park",
            "machine learning models process text into dense vectors",
            "a cat sleeps quietly on the windowsill all afternoon",
        ]
        import time as _t
        try:
            nb = NpuBackend(self.spec); nb.warmup()
        except Exception as e:
            sys.stderr.write(f"[auto] NPU unavailable ({e}); using CPU\n")
            return "cpu"
        # min-of-3 on each
        def best(fn):
            ts = []
            for _ in range(3):
                t0 = _t.time(); fn(); ts.append(_t.time() - t0)
            return min(ts)
        cpu_t = best(lambda: self._get_cpu().embed(probe))
        npu_t = best(lambda: nb.embed(probe))
        self._npu = nb if npu_t < cpu_t else None   # keep winner loaded
        winner = "npu" if npu_t < cpu_t else "cpu"
        sys.stderr.write(
            f"[auto] calibrated: cpu={cpu_t*1000:.1f}ms npu={npu_t*1000:.1f}ms "
            f"-> {winner}\n")
        return winner

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._choice is None:      # not warmed (e.g. direct use); default cpu
            self._choice = "cpu"
        if self._choice == "npu" and self._npu is not None:
            self._last_choice = "npu"
            try:
                return self._npu.embed(texts)
            except Exception as e:
                sys.stderr.write(f"[auto] NPU failed ({e}); falling back to CPU\n")
                self._choice = "cpu"
        self._last_choice = "cpu"
        return self._get_cpu().embed(texts)
