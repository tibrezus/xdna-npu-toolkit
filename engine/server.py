"""OpenAI-compatible embedding server, hardened.

Endpoints:
    POST /v1/embeddings   {"model":alias,"input":"..."|[..],"encoding_format":"float"|"base64",
                           "dimensions":N?}  -> OpenAI-shaped response
    GET  /v1/models       -> registered models (with NPU-availability flags)
    GET  /health[?deep=1] -> shallow ok, or deep (runs a probe embed)
    GET  /metrics         -> request/text/error counts, latency, avg batch, per-backend

Hardening over the v0 server:
  - Thread-safety: each (model,backend) runs on a SINGLE worker thread. HTTP
    handler threads enqueue and block on a future; the worker is the only thing
    that touches the backend -> no concurrent embed() calls, no BO corruption.
  - Dynamic micro-batching: the worker coalesces concurrent requests into one
    embed() call (up to --max-batch texts within --batch-window-ms). For the NPU
    this turns a burst of single-query RAG lookups into one batch-64 forward,
    amortising the padding that otherwise wastes 64x compute per lone query.
  - Validation: input size/count limits with proper OpenAI-style error codes.
  - Graceful shutdown: SIGTERM/SIGINT stop workers + close the socket.

The NPU's 4-hw_context budget is honoured because all bert384 models share one
singleton Bf16Backend (see backends.py). A different-shape NPU family (qwen) can't
coexist with bert384 at once; the server loads lazily and reports a clear 503 if
the context budget is exhausted.
"""
from __future__ import annotations
import base64
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, Queue

from .backends import make_backend, npu_available, EmbedBackend
from .models import REGISTRY, resolve

# defaults tunable via CLI
DEFAULT_MAX_BATCH = 64          # one NPU batch-64 forward; CPU benefits too
DEFAULT_BATCH_WINDOW_MS = 5.0   # coalesce window; 0 = no batching (still serialised)
DEFAULT_MAX_INPUT_ITEMS = 4096  # max texts per request
DEFAULT_MAX_TEXT_CHARS = 32768  # per-text char cap (tokenizer truncates anyway)


# ─── metrics ─────────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    requests: int = 0
    texts: int = 0
    errors: int = 0
    inference_ms: float = 0.0
    batches: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, n_texts: int, ms: float, ok: bool):
        with self._lock:
            self.requests += 1
            self.texts += n_texts
            self.inference_ms += ms
            if ok:
                self.batches += 1
            else:
                self.errors += 1

    def snapshot(self) -> dict:
        with self._lock:
            avg_batch = (self.texts / self.batches) if self.batches else 0.0
            avg_ms = (self.inference_ms / self.batches) if self.batches else 0.0
            return {
                "requests": self.requests, "texts": self.texts,
                "errors": self.errors, "batches": self.batches,
                "avg_texts_per_batch": round(avg_batch, 2),
                "avg_inference_ms": round(avg_ms, 2),
                "total_inference_ms": round(self.inference_ms, 1),
            }


# ─── single-worker micro-batcher (thread-safe by construction) ───────────────

class BatchWorker:
    """One worker thread per backend. Serialises embed() AND coalesces requests.

    submit(texts) blocks until the worker has run (possibly batched with others).
    """

    def __init__(self, eng: EmbedBackend, metrics: Metrics,
                 max_batch: int, window_ms: float, name: str):
        self.eng = eng
        self.metrics = metrics
        self.max_batch = max_batch
        self.window_ms = window_ms
        self.name = name
        self.q: Queue = Queue()
        self._stop = False
        self.thread = threading.Thread(target=self._loop, name=f"worker-{name}", daemon=True)
        self.thread.start()

    def submit(self, texts: list[str]) -> tuple:
        """Return (vecs, ms, batch_size). Raises if the embed failed."""
        fut: Future = Future()
        self.q.put((texts, fut))
        t0 = time.monotonic()
        vecs = fut.result()                      # propagates worker exceptions
        wall_ms = (time.monotonic() - t0) * 1000
        return vecs, wall_ms, fut._xdna_batch_size  # type: ignore[attr-defined]

    def stop(self):
        self._stop = True
        self.q.put(([], None))                   # nudge the blocking get()

    def _loop(self):
        while not self._stop:
            try:
                first = self.q.get(timeout=1.0)
            except Empty:
                continue
            if first[1] is None:
                continue                          # stop nudge
            items = [first]
            total = len(first[0])
            # window_ms is milliseconds; monotonic() is seconds -> divide.
            deadline = time.monotonic() + self.window_ms / 1000.0
            while total < self.max_batch:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    it = self.q.get(timeout=timeout)
                except Empty:
                    break
                if it[1] is None:
                    continue
                items.append(it)
                total += len(it[0])
            # run once for the whole coalesced batch
            all_texts = [t for ts, _ in items for t in ts]
            sizes = [len(ts) for ts, _ in items]
            t0 = time.monotonic()
            try:
                vecs = self.eng.embed(all_texts)
                ms = (time.monotonic() - t0) * 1000
                self.metrics.observe(len(all_texts), ms, ok=True)
            except Exception as e:                # whole batch fails
                self.metrics.observe(len(all_texts), 0.0, ok=False)
                for _, fut in items:
                    if fut is not None:
                        fut.set_exception(e)
                sys.stderr.write(f"[worker:{self.name}] batch failed: {e}\n")
                continue
            # split results back to each request's future
            off = 0
            for (_, fut), sz in zip(items, sizes):
                fut._xdna_batch_size = len(all_texts)   # type: ignore[attr-defined]
                fut.set_result(vecs[off:off + sz])
                off += sz


# ─── engine pool (lazy, cached workers) ──────────────────────────────────────

class EnginePool:
    """Lazily builds + caches one BatchWorker per (alias, backend)."""

    def __init__(self, default_alias, default_backend, max_batch, window_ms):
        self.default_alias = default_alias
        self.default_backend = default_backend
        self.max_batch = max_batch
        self.window_ms = window_ms
        self.metrics = Metrics()
        self._workers: dict[tuple, BatchWorker] = {}
        self._lock = threading.Lock()

    def get(self, alias: str, backend: str) -> tuple[BatchWorker, object]:
        key = (alias, backend)
        if key not in self._workers:
            with self._lock:
                if key not in self._workers:
                    spec, _ = resolve(alias)
                    eng = make_backend(backend, spec)
                    eng.warmup()
                    self._workers[key] = BatchWorker(
                        eng, self.metrics, self.max_batch, self.window_ms,
                        name=f"{alias}:{backend}")
                    sys.stderr.write(
                        f"[pool] loaded {alias} ({backend}) -> worker started\n")
        return self._workers[key], resolve(alias)[0]

    def stop_all(self):
        for w in self._workers.values():
            w.stop()


# ─── HTTP plumbing ───────────────────────────────────────────────────────────

def _send(h: BaseHTTPRequestHandler, code: int, obj):
    body = json.dumps(obj).encode()
    h.send_response(code)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(body)))
    h.end_headers()
    h.wfile.write(body)


def _err(h, code, msg, etype="invalid_request_error"):
    _send(h, code, {"error": {"message": msg, "type": etype, "code": code}})


def make_handler(pool: EnginePool, max_items: int, max_chars: int):
    class Handler(BaseHTTPRequestHandler):
        server_version = "xdna-embed/0.1"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            sys.stderr.write(f"[http] {self.address_string()} {fmt % args}\n")

        def _read_json(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
            except ValueError:
                return None
            if n == 0:
                return {}
            if n > 16 * 1024 * 1024:
                return None
            raw = self.rfile.read(n)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None

        # ── GET routes ──
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            if path == "/health":
                deep = "deep=1" in qs or "deep=true" in qs
                if not deep:
                    return _send(self, 200, {"status": "ok"})
                # deep: probe the default backend end-to-end
                try:
                    w, _ = pool.get(pool.default_alias, pool.default_backend)
                    w.submit(["health probe"])
                    return _send(self, 200, {"status": "ok", "deep": True,
                                             "model": pool.default_alias})
                except Exception as e:
                    return _send(self, 503, {"status": "degraded", "error": str(e)})
            if path == "/metrics":
                snap = pool.metrics.snapshot()
                snap["backends"] = [
                    {"model": a, "backend": b, "running": True}
                    for (a, b) in pool._workers]
                return _send(self, 200, snap)
            if path == "/v1/models":
                data = []
                for s in REGISTRY.values():
                    data.append({
                        "id": s.alias, "object": "model",
                        "owned_by": "xdna-embed",
                        "npu": bool(s.npu and npu_available()),
                        "dim": s.dim, "pooling": s.pooling,
                    })
                return _send(self, 200, {"object": "list", "data": data})
            return _err(self, 404, f"unknown path {path}", "not_found")

        # ── POST /v1/embeddings ──
        def do_POST(self):
            if self.path.split("?", 1)[0] != "/v1/embeddings":
                return _err(self, 404, "unknown path", "not_found")
            body = self._read_json()
            if body is None:
                return _err(self, 400, "invalid or missing JSON body")
            if not isinstance(body, dict):
                return _err(self, 400, "request body must be a JSON object")

            inp = body.get("input")
            if inp is None:
                return _err(self, 400, "missing required field: 'input'")
            texts = [inp] if isinstance(inp, str) else list(inp)
            if not isinstance(inp, (str, list)):
                return _err(self, 400, "'input' must be a string or array of strings")
            if not texts:
                return _err(self, 400, "'input' is empty")
            if len(texts) > max_items:
                return _err(self, 413, f"too many inputs: {len(texts)} > {max_items}",
                            "request_too_large")
            for i, t in enumerate(texts):
                if not isinstance(t, str):
                    return _err(self, 400, f"input[{i}] is not a string")
                if len(t) > max_chars:
                    return _err(self, 413,
                                f"input[{i}] too long: {len(t)} > {max_chars} chars",
                                "request_too_large")

            alias = body.get("model") or pool.default_alias
            backend = body.get("backend") or pool.default_backend
            if backend not in ("npu", "cpu", "auto"):
                return _err(self, 400, f"unknown backend '{backend}'")

            try:
                worker, spec = pool.get(alias, backend)
            except Exception as e:
                return _err(self, 503,
                            f"cannot load model/backend '{alias}/{backend}': {e}",
                            "service_unavailable")

            try:
                vecs, wall_ms, batch_size = worker.submit(texts)
            except Exception as e:
                return _err(self, 500, f"inference failed: {e}", "inference_error")

            fmt = body.get("encoding_format", "float")
            if fmt not in ("float", "base64"):
                return _err(self, 400, "encoding_format must be 'float' or 'base64'")
            dims = body.get("dimensions")
            data = []
            for i, v in enumerate(vecs):
                vv = v
                if dims is not None and isinstance(dims, int) and dims > 0:
                    vv = v[:dims]
                emb = (base64.b64encode(vv.astype("float32").tobytes()).decode()
                       if fmt == "base64" else vv.tolist())
                data.append({"object": "embedding", "embedding": emb, "index": i})

            approx_tokens = int(sum(len(t.split()) * 1.3 for t in texts))
            out = {
                "object": "list", "data": data, "model": alias,
                "usage": {"prompt_tokens": approx_tokens,
                          "total_tokens": approx_tokens},
                "xdna_meta": {                       # extra, non-OpenAI; clients ignore
                    "backend": getattr(worker.eng, "name", backend),
                    "batch_size": batch_size, "wall_ms": round(wall_ms, 2),
                    "dim": int(spec.dim or vecs.shape[1]),
                },
            }
            return _send(self, 200, out)

    return Handler


# ─── entrypoint ──────────────────────────────────────────────────────────────

def serve(host: str, port: int, alias: str, backend: str,
          max_batch: int = DEFAULT_MAX_BATCH,
          window_ms: float = DEFAULT_BATCH_WINDOW_MS,
          max_items: int = DEFAULT_MAX_INPUT_ITEMS,
          max_chars: int = DEFAULT_MAX_TEXT_CHARS):
    pool = EnginePool(alias, backend, max_batch, window_ms)
    Handler = make_handler(pool, max_items, max_chars)
    httpd = ThreadingHTTPServer((host, port), Handler)

    shutting = {"v": False}

    def _shutdown(signum, frame):
        if shutting["v"]:
            return
        shutting["v"] = True
        sys.stderr.write(f"\n[server] caught signal {signum}; draining workers...\n")
        pool.stop_all()
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _shutdown)

    sys.stderr.write(
        f"\n  xdna-embed server\n"
        f"  ────────────────────────────────────────────────\n"
        f"  model    : {alias}   (override per-request: \"model\")\n"
        f"  backend  : {backend} (override per-request: \"backend\")\n"
        f"  batching : max_batch={max_batch}, window={window_ms}ms\n"
        f"  limits   : {max_items} texts/req, {max_chars} chars/text\n"
        f"  listen   : http://{host}:{port}\n\n"
        f"  curl -X POST http://{host}:{port}/v1/embeddings \\\n"
        f"    -H 'Content-Type: application/json' \\\n"
        f"    -d '{{\"model\":\"{alias}\",\"input\":\"hello world\"}}'\n\n"
        f"  GET /health?deep=1   /metrics   /v1/models\n\n")
    try:
        httpd.serve_forever()
    finally:
        pool.stop_all()
        httpd.server_close()
        sys.stderr.write("[server] stopped\n")
