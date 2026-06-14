"""OpenAI-compatible embedding server (stdlib only — no extra deps).

    POST /v1/embeddings
      {"model": "<alias>", "input": "text" | ["t1","t2"], "encoding_format": "float"}
    -> {"data": [{"object":"embedding","embedding":[...],"index":0}, ...],
        "model": "<alias>", "usage": {"prompt_tokens": N, "total_tokens": N}}

    GET  /v1/models   -> {"data":[{"id":alias,"object":"model"}]}
    GET  /health      -> {"status":"ok"}

Mirrors the OpenAI embeddings endpoint so any OpenAI client (LangChain,
chromadb, llama-index, openai-python with base_url override, ...) can use the
NPU/CPU engine as a drop-in local backend, exactly like `llama-server`.
"""
from __future__ import annotations
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .backends import make_backend, EmbedBackend
from .models import resolve


def _json_response(handler, code: int, obj):
    body = json.dumps(obj).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(default_alias: str, default_backend: str, cache: dict):
    """Build a request handler bound to the chosen model/backend (lazy-loaded)."""

    def get_engine(alias: str, backend: str) -> tuple[EmbedBackend, object]:
        key = (alias, backend)
        if key not in cache:
            spec, _ = resolve(alias)
            eng = make_backend(backend, spec)
            eng.warmup()
            cache[key] = (eng, spec)
        return cache[key]

    class Handler(BaseHTTPRequestHandler):
        server_version = "xdna-embed/0.1"

        def log_message(self, fmt, *args):
            sys_write = self.server  # ThreadingHTTPServer
            # concise access log like llama-server
            import sys
            sys.stderr.write(f"[server] {self.address_string()} {fmt % args}\n")

        def _read_json(self):
            n = int(self.headers.get("Content-Length", 0))
            if n == 0:
                return {}
            raw = self.rfile.read(n)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return None

        def do_GET(self):
            if self.path == "/health":
                return _json_response(self, 200, {"status": "ok"})
            if self.path == "/v1/models":
                from .models import REGISTRY
                data = [{"id": a, "object": "model"} for a in REGISTRY]
                return _json_response(self, 200, {"object": "list", "data": data})
            return _json_response(self, 404, {"error": {"message": "not found", "type": "not_found"}})

        def do_POST(self):
            if self.path != "/v1/embeddings":
                return _json_response(self, 404, {"error": {"message": "not found"}})
            body = self._read_json()
            if body is None:
                return _json_response(self, 400, {"error": {"message": "invalid JSON"}})
            inp = body.get("input")
            if inp is None:
                return _json_response(self, 400, {"error": {"message": "missing 'input'"}})
            alias = body.get("model", default_alias)
            backend = body.get("backend", default_backend)
            texts = [inp] if isinstance(inp, str) else list(inp)
            if not texts:
                return _json_response(self, 400, {"error": {"message": "empty input"}})

            try:
                eng, spec = get_engine(alias, backend)
            except Exception as e:
                return _json_response(self, 404, {"error": {
                    "message": f"cannot load model/backend: {e}", "type": "model_error"}})

            t0 = time.time()
            try:
                vecs = eng.embed(texts)
            except Exception as e:
                return _json_response(self, 500, {"error": {
                    "message": f"inference failed: {e}", "type": "inference_error"}})
            dt_ms = (time.time() - t0) * 1000

            fmt = body.get("encoding_format", "float")
            data = []
            for i, v in enumerate(vecs):
                emb = v.tolist() if fmt == "float" else _b64(v)
                data.append({"object": "embedding", "embedding": emb, "index": i})

            # rough token estimate (words*1.3); precise counting would need the tok
            approx_tokens = int(sum(len(t.split()) * 1.3 for t in texts))
            out = {
                "object": "list",
                "data": data,
                "model": alias,
                "usage": {"prompt_tokens": approx_tokens, "total_tokens": approx_tokens},
            }
            import sys
            sys.stderr.write(
                f"[server] embedded {len(texts)} text(s) via "
                f"{getattr(eng,'name','?')} in {dt_ms:.1f} ms\n")
            return _json_response(self, 200, out)

    return Handler


def _b64(vec):
    import base64
    return base64.b64encode(vec.astype("float32").tobytes()).decode("ascii")


def serve(host: str, port: int, alias: str, backend: str):
    cache: dict = {}
    Handler = make_handler(alias, backend, cache)
    httpd = ThreadingHTTPServer((host, port), Handler)
    import sys
    sys.stderr.write(
        f"\n  xdna-embed server\n"
        f"  model:   {alias}   (change per-request with \"model\" in body)\n"
        f"  backend: {backend} (change per-request with \"backend\" in body)\n"
        f"  listen:  http://{host}:{port}\n\n"
        f"  curl -X POST http://{host}:{port}/v1/embeddings \\\n"
        f"    -H 'Content-Type: application/json' \\\n"
        f"    -d '{{\"model\":\"{alias}\",\"input\":\"hello world\"}}'\n\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\n[server] shutting down\n")
        httpd.shutdown()
