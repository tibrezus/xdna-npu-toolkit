"""xdna-embed — llama.cpp-style CLI for embedding models on the AMD XDNA NPU.

    xdna-embed embed -m minilm -b npu "dogs play in the park"
    xdna-embed embed -m qwen3-0.6b -b cpu --input corpus.txt --output vecs.npy
    xdena-embed server -m minilm -b auto --port 8080
    xdna-embed bench -m qwen3-0.6b
    xdna-embed list
    xdna-embed info
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

from . import __version__
from .models import REGISTRY, resolve
from .backends import make_backend, npu_available, CpuBackend, NpuBackend, AutoBackend


# ─── formatting ──────────────────────────────────────────────────────────────

C = {
    "R": "\033[31m", "G": "\033[32m", "Y": "\033[33m",
    "B": "\033[34m", "M": "\033[35m", "C": "\033[36m",
    "dim": "\033[2m", "bold": "\033[1m", "rst": "\033[0m",
}


def _c(code: str, s: str) -> str:
    return f"{C[code]}{s}{C['rst']}" if sys.stdout.isatty() else s


def banner(model: str, backend: str):
    sys.stderr.write(
        f"\n{_c('bold', 'xdna-embed')} {_c('dim', f'v{__version__}')}  "
        f"AMD XDNA NPU embedding engine\n"
        f"{_c('dim', '─' * 52)}\n"
        f"  model    : {model}\n"
        f"  backend  : {backend}\n\n")


# ─── helpers ─────────────────────────────────────────────────────────────────

def _read_inputs(args) -> list[str]:
    if args.input:
        if args.input == "-":
            return sys.stdin.read().splitlines()
        with open(args.input, "r") as f:
            return [ln.rstrip("\n") for ln in f if ln.strip()]
    if not args.texts:
        # read from stdin if piped
        if not sys.stdin.isatty():
            return [ln.rstrip("\n") for ln in sys.stdin if ln.strip()]
        sys.stderr.write("error: no input. Pass texts, --input FILE, or pipe stdin.\n")
        sys.exit(2)
    return list(args.texts)


def _backend_for(backend: str, spec):
    """Construct backend, with a friendly error if NPU requested but unsupported."""
    if backend == "npu" and spec.npu is None:
        sys.stderr.write(
            f"warn: model '{spec.alias}' has no compiled NPU kernels; "
            f"falling back to cpu. (The NPU is not a universal JIT — "
            f"see `xdna-embed list`.)\n")
        return make_backend("cpu", spec)
    try:
        return make_backend(backend, spec)
    except RuntimeError as e:
        sys.stderr.write(f"error: {e}\n")
        sys.exit(1)


# ─── subcommands ─────────────────────────────────────────────────────────────

def cmd_embed(args):
    spec, _ = resolve(args.model)
    banner(args.model, args.backend)
    eng = _backend_for(args.backend, spec)
    texts = _read_inputs(args)
    sys.stderr.write(f"  input    : {len(texts)} text(s)\n")
    sys.stderr.write(f"{_c('dim', '─' * 52)}\n")

    t0 = time.time()
    eng.warmup()
    load_ms = (time.time() - t0) * 1000
    sys.stderr.write(f"  load     : {load_ms:>8.1f} ms\n")

    t0 = time.time()
    vecs = eng.embed(texts)
    eval_ms = (time.time() - t0) * 1000
    chosen = getattr(eng, "name", args.backend)
    if isinstance(eng, AutoBackend):
        chosen = f"auto->{eng._last_choice}"
    sys.stderr.write(
        f"  eval     : {eval_ms:>8.1f} ms  ({_c('C', chosen)})  "
        f"[{eval_ms/max(len(texts),1)*1000:.1f} us/text]\n")
    sys.stderr.write(f"  dim      : {vecs.shape[1]}\n\n")

    _emit(args, texts, vecs)


def _emit(args, texts, vecs):
    fmt = args.format
    if fmt == "numpy":
        if args.output:
            np.save(args.output, vecs)
            sys.stderr.write(f"  wrote {args.output}  shape={vecs.shape}\n")
        else:
            sys.stderr.write("  (use --output FILE with --format numpy)\n")
        return
    if fmt == "json":
        obj = [{"text": t, "embedding": v.tolist()} for t, v in zip(texts, vecs)]
        print(json.dumps(obj, indent=2) if args.pretty else json.dumps(obj))
        return
    # "compact": one vector per line, space-separated floats (like llama-embedding)
    for v in vecs:
        print(" ".join(f"{x:.6f}" for x in v))


def cmd_server(args):
    from .server import serve
    banner(args.model, args.backend)
    spec, _ = resolve(args.model)
    if args.backend == "npu" and spec.npu is None:
        sys.stderr.write(
            f"warn: default model '{args.model}' has no NPU kernels; "
            f"requests for it will use cpu unless the body sets a compiled alias.\n")
    serve(args.host, args.port, args.model, args.backend,
          max_batch=args.max_batch, window_ms=args.batch_window_ms,
          max_items=args.max_input_items, max_chars=args.max_text_chars)


def _bench_one_subprocess(model: str, backend: str, sizes_csv: str) -> list[str]:
    """Run one (model, backend) combo in a fresh subprocess for context isolation.

    The amdxdna driver caps at 4 simultaneous hw_contexts, so running multiple
    NPU backends in one process is unsafe. Each combo gets its own process.
    """
    import subprocess
    py = sys.executable
    cmd = [py, "-m", "engine.cli", "bench",
           "-m", model, "-b", backend, "--sizes", sizes_csv, "--internal"]
    # inherit the same env the parent runs with (sys.path etc. set by wrappers)
    env = os.environ.copy()
    out = subprocess.run(cmd, env=env, capture_output=True, text=True,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rows = [ln for ln in out.stdout.splitlines()
            if ln.strip() and not ln.startswith(("backend", "---"))]
    if not rows and out.returncode != 0:
        avail = f"(stderr: {out.stderr.strip().splitlines()[-1][:50]})"
        rows = [f"{backend:<10} {'-':>6} {'-':>9} {'-':>9}  unavailable {avail}"]
    return rows


def cmd_bench(args):
    spec, _ = resolve(args.model)
    banner(args.model, args.backend)
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else [1, 8, 32, 64]
    sample = "dogs and cats play together in the sunny park on a warm afternoon"
    print(f"{'backend':<14} {'batch':>6} {'ms/text':>9} {'ms/batch':>9} {'dim':>5}")
    print("-" * 47)

    if args.internal:
        # single-backend worker invoked by the parent; runs in-process.
        backends = [args.backend]
    elif args.backend == "all":
        backends = ["cpu"] + (["npu"] if spec.npu is not None else []) + ["auto"]
    else:
        backends = [args.backend]

    sizes_csv = ",".join(str(s) for s in sizes)
    for b in backends:
        if args.internal or args.backend not in ("all",):
            # in-process path (single backend)
            for row in _bench_one_inproc(b, spec, sizes, sample):
                print(row)
        else:
            # subprocess for isolation
            for row in _bench_one_subprocess(args.model, b, sizes_csv):
                print(row)


def _bench_one_inproc(b, spec, sizes, sample):
    rows = []
    try:
        eng = make_backend(b, spec)
        eng.warmup()
    except Exception as e:
        return [f"{b:<14} {'-':>6} {'-':>9} {'-':>9}  (unavailable: {str(e)[:30]})"]
    for n in sizes:
        texts = [sample] * n
        warm = 4 if n <= 8 else 2
        for _ in range(warm):
            eng.embed(texts)
        iters = max(3, 40 // max(n, 1))
        samples = []
        for _ in range(iters):
            t0 = time.time(); eng.embed(texts); samples.append(time.time() - t0)
        dt = min(samples)
        chosen = getattr(eng, "_last_choice", None) or getattr(eng, "name", b)
        tag = f"{b}->{chosen}" if b == "auto" and chosen else b
        rows.append(f"{tag:<14} {n:>6} {dt*1000/n:>9.3f} {dt*1000:>9.1f} {eng.dim:>5}")
    return rows


def cmd_list(args):
    print(f"{'alias':<14} {'hf_id':<46} {'dim':>5} {'pool':<11} {'NPU':<5}")
    print("-" * 84)
    npu_ok = npu_available()
    for s in REGISTRY.values():
        npu_state = (s.npu or "-")
        if s.npu and not npu_ok:
            npu_state += " (no device)"
        print(f"{s.alias:<14} {s.hf_id:<46} {s.dim:>5} {s.pooling:<11} {npu_state:<5}")
    print()
    print(_c("dim", "Models with an NPU adapter name have compiled bf16 GEMM kernels "
                     "and run on the NPU. Others are CPU-only (use --backend cpu). "
                     "Arbitrary HuggingFace ids also work on CPU, e.g. "
                     "-m intfloat/multilingual-e5-small."))


def cmd_info(args):
    print(f"{_c('bold','xdna-embed')} device / backend status\n{'─'*44}")
    # NPU device
    present = npu_available()
    print(f"  NPU device   : {NPU_DEVICE_NODE}")
    print(f"  present      : {_c('G','yes') if present else _c('R','no')}")
    if present:
        import stat
        st = os.stat(NPU_DEVICE_NODE)
        print(f"  perms        : {oct(st.st_mode & 0o777)}")
    # memlock (needed for NPU access)
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        ok = (soft == resource.RLIM_INFINITY or soft == -1)
        print(f"  memlock      : {soft}  {_c('G','OK') if ok else _c('Y','UNLIMITED needed (run under login shell / pam_limits)')}")
    except Exception:
        pass
    print()
    print(f"  compiled NPU models:")
    for s in REGISTRY.values():
        if s.npu:
            print(f"    - {s.alias:<12} (adapter: {s.npu})  {_c('G','ready') if present else _c('Y','device missing')}")
    print()
    print(_c("dim", "Tip: run `xdna-npu doctor` for the full stack diagnostic, "
                     "`xdna-npu enable` (root) to fix memlock/power."))


# ─── arg parser ──────────────────────────────────────────────────────────────

NPU_DEVICE_NODE = "/dev/accel/accel0"  # mirror backends for cmd_info


def build_parser():
    p = argparse.ArgumentParser(
        prog="xdna-embed",
        description="llama.cpp-style embedding engine for the AMD XDNA NPU.",
    )
    p.add_argument("--version", action="version", version=f"xdna-embed {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    # embed
    pe = sub.add_parser("embed", help="embed text(s) and print/store vectors")
    pe.add_argument("texts", nargs="*", help="text(s) to embed")
    pe.add_argument("-m", "--model", default="minilm", help="model alias or HF id (default: minilm)")
    pe.add_argument("-b", "--backend", default="auto", choices=["npu", "cpu", "auto"])
    pe.add_argument("-i", "--input", help="input file (one text per line), or '-' for stdin")
    pe.add_argument("-o", "--output", help="output file (with --format numpy)")
    pe.add_argument("-f", "--format", default="compact", choices=["compact", "json", "numpy"])
    pe.add_argument("--pretty", action="store_true", help="pretty-print JSON")
    pe.set_defaults(func=cmd_embed)

    # server
    ps = sub.add_parser("server", help="OpenAI-compatible /v1/embeddings HTTP server")
    ps.add_argument("-m", "--model", default="minilm")
    ps.add_argument("-b", "--backend", default="auto", choices=["npu", "cpu", "auto"])
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8080)
    ps.add_argument("--max-batch", type=int, default=64,
                    help="max texts coalesced into one embed() call (default 64 = one NPU forward)")
    ps.add_argument("--batch-window-ms", type=float, default=5.0,
                    help="coalesce window in ms (0 disables batching; calls still serialised)")
    ps.add_argument("--max-input-items", type=int, default=4096)
    ps.add_argument("--max-text-chars", type=int, default=32768)
    ps.set_defaults(func=cmd_server)

    # bench
    pb = sub.add_parser("bench", help="benchmark backends across batch sizes")
    pb.add_argument("-m", "--model", default="qwen3-0.6b")
    pb.add_argument("-b", "--backend", default="cpu", choices=["npu", "cpu", "auto", "all"])
    pb.add_argument("--sizes", help="comma-separated batch sizes (default 1,8,32,64)")
    pb.add_argument("--internal", action="store_true", help=argparse.SUPPRESS)
    pb.set_defaults(func=cmd_bench)

    # list
    pl = sub.add_parser("list", help="show models + which have NPU kernels")
    pl.set_defaults(func=cmd_list)

    # info
    pi = sub.add_parser("info", help="show NPU device + backend status")
    pi.set_defaults(func=cmd_info)

    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
