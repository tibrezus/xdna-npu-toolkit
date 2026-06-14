"""AMD VitisAI Execution Provider setup + probe for the XDNA 1 NPU.

This module captures the working recipe for driving the XDNA (Phoenix/Hawk
Point) NPU via AMD's public ``onnxruntime-vitisai`` stack on Linux. It exists
because the stock distribution makes this painful:

  * the AMD wheels are ``cp312`` (system Python is often newer, e.g. 3.14);
  * the VitisAI EP loads a pile of shared objects from ``voe/lib`` that must be
    on ``LD_LIBRARY_PATH`` or it fails with "cannot open shared object file";
  * the EP needs ``dd_root`` (the ``ryzenai_dynamic_dispatch`` package dir) as a
    session-config entry to use the dynamic-dispatch (4x4 / Phoenix) kernels.

What this proves
----------------
``probe_ep()`` instantiates ``VitisAIExecutionProvider`` against the live
Phoenix NPU. On a Ryzen 7 7840HS this returns::

    providers: ['VitisAIExecutionProvider', 'CPUExecutionProvider']

i.e. the NPU execution provider *initializes on XDNA 1*. That is the headline
result: unlike LLMs (whose runtimes reject Phoenix), **embedding/CNN/transformer
inference is runtime-supported on XDNA 1 on Linux.**

The one remaining gate is *model compilation*: the publicly-installable Linux
wheels are "deployment-only" (they run pre-compiled models but emit
``Model compilation is not supported in a deployment only installation`` if you
ask them to compile an ONNX graph at runtime). The full compiler ships only in
AMD's account-gated Ryzen AI Software installer. So you can run a *pre-compiled*
PHX embedding model today; to compile your own you need that installer.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

# Python that the AMD cp312 wheels require.
REQUIRED_PY = "3.12"
AMD_INDEX = "https://pypi.amd.com/ryzenai_llm/1.7.1/linux/simple/"
AMD_PKGS = [
    "onnxruntime-vitisai",        # the VitisAI EP build of onnxruntime
    "voe",                        # Vitis Overlay Engine: compiler + 4x4 kernels
    "ryzenai-dynamic-dispatch",   # dynamic dispatch (runtime kernels incl. PHX 4x4)
    "ryzenai-onnx-utils",         # ONNX preprocessing/partitioning helpers
    "onnxruntime-providers-ryzenai",
]


@dataclass
class EpProbe:
    vitisai_available: bool      # EP listed by onnxruntime.get_available_providers()
    session_initialized: bool    # EP created a session against the live NPU
    providers: list[str]
    compile_supported: bool      # whether at-runtime compilation is available
    messages: list[str]
    deployment_only: bool


def _find_python312() -> str | None:
    """Find a CPython 3.12 interpreter (system or uv-managed)."""
    for cand in (sys.executable, "python3.12", "python3.12"):
        try:
            out = subprocess.run([cand, "--version"], capture_output=True, text=True, timeout=5)
            if out.stdout.strip().endswith("3.12") or out.stdout.strip().endswith("3.12.13"):
                return cand
        except Exception:
            pass
    # uv-managed
    import glob
    uv = shutil.which("uv")
    if uv:
        for p in glob.glob(os.path.expanduser("~/.local/share/uv/python/cpython-3.12*/bin/python3.12")):
            return p
    return None


def site_packages_of(python: str) -> str | None:
    try:
        out = subprocess.run([python, "-c", "import site;print(site.getsitepackages()[0])"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out or None
    except Exception:
        return None


def lib_dirs(site: str) -> list[str]:
    """The directories that must be on LD_LIBRARY_PATH for the EP to load."""
    return [
        os.path.join(site, "voe", "lib"),
        os.path.join(site, "onnxruntime", "capi"),
        os.path.join(site, "ryzenai_dynamic_dispatch", "lib"),
    ]


def ep_env(python: str) -> dict[str, str]:
    """Build the environment (LD_LIBRARY_PATH + DD_ROOT) needed for the EP."""
    site = site_packages_of(python) or ""
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs(site) + [env.get("LD_LIBRARY_PATH", "")])
    env["DD_ROOT"] = os.path.join(site, "ryzenai_dynamic_dispatch")
    return env


def is_stack_installed(python: str) -> bool:
    """True if the AMD VitisAI wheels are present under this interpreter."""
    site = site_packages_of(python)
    if not site:
        return False
    return os.path.isdir(os.path.join(site, "voe")) and \
        os.path.isdir(os.path.join(site, "ryzenai_dynamic_dispatch")) and \
        os.path.isdir(os.path.join(site, "onnxruntime"))


def probe_ep(python: str | None = None) -> EpProbe:
    """Instantiate the VitisAI EP against the live NPU and report the result.

    This is the headline check: it proves whether the NPU execution provider
    initializes on *this* hardware. Safe to call as any user with device access
    (the memlock fix from ``xdna-npu enable`` should be applied first).
    """
    python = python or _find_python312()
    msgs: list[str] = []
    if not python:
        msgs.append(f"No CPython {REQUIRED_PY} found (the AMD wheels are cp312). "
                    f"Install one: `uv python install {REQUIRED_PY}` or your distro's python3.12.")
        return EpProbe(False, False, [], False, msgs, False)

    site = site_packages_of(python)
    if not site or not os.path.isdir(os.path.join(site, "voe")):
        msgs.append("AMD VitisAI stack not installed in this interpreter. Run "
                    "`xdna-npu embed-setup` (or see README) to install it from pypi.amd.com.")
        return EpProbe(False, False, [], False, msgs, False)

    env = ep_env(python)
    probe = (
        "import onnxruntime as ort, os\n"
        "avps = ort.get_available_providers()\n"
        "print('__AVPS__', ','.join(avps))\n"
        "so = ort.SessionOptions()\n"
        "so.add_session_config_entry('dd_root', os.environ['DD_ROOT'])\n"
        "ok=False; msg=''\n"
        "try:\n"
        "    s = ort.InferenceSession('TINY', so, providers=['VitisAIExecutionProvider'],\n"
        "        provider_options=[{'config_file':'','cache_dir':'/tmp/xdna-ep-cache'}])\n"
        "    print('__SESS__', ','.join(s.get_providers())); ok=True\n"
        "except Exception as e:\n"
        "    msg = str(e); print('__ERR__', msg[:600])\n"
    )
    # build a tiny throwaway model in-process first
    setup = (
        "import numpy as np, onnx\n"
        "from onnx import helper, TensorProto\n"
        "W=helper.make_tensor('w',TensorProto.FLOAT,[16,8],np.ones(128,dtype=np.float32))\n"
        "B=helper.make_tensor('b',TensorProto.FLOAT,[8],np.zeros(8,dtype=np.float32))\n"
        "X=helper.make_tensor_value_info('x',TensorProto.FLOAT,[8,16])\n"
        "Y=helper.make_tensor_value_info('y',TensorProto.FLOAT,[8,8])\n"
        "g=helper.make_graph([helper.make_node('MatMul',['x','w'],['t']),helper.make_node('Add',['t','b'],['y'])],'g',[X],[Y],[W,B])\n"
        "m=helper.make_model(g,opset_imports=[helper.make_opsetid('',13)]); m.ir_version=9\n"
        "onnx.save(m,'/tmp/xdna_tiny.onnx')\n"
    )
    proc = subprocess.run(
        [python, "-c", setup + probe.replace("'TINY'", "'/tmp/xdna_tiny.onnx'")],
        capture_output=True, text=True, env=env, timeout=240,
    )
    out = proc.stdout + proc.stderr
    msgs.append(out.strip()[-1200:])
    avps = []
    sess_providers: list[str] = []
    err = ""
    for line in out.splitlines():
        if line.startswith("__AVPS__"):
            avps = [p.strip() for p in line.split("__AVPS__", 1)[1].split(",") if p.strip()]
        elif line.startswith("__SESS__"):
            sess_providers = [p.strip() for p in line.split("__SESS__", 1)[1].split(",") if p.strip()]
        elif line.startswith("__ERR__"):
            err = line.split("__ERR__", 1)[1].strip()
    vitisai = "VitisAIExecutionProvider" in avps
    initialized = "VitisAIExecutionProvider" in sess_providers
    deploy_only = "deployment only installation" in err.lower() or "deployment only" in out.lower()
    compile_supported = initialized and not deploy_only
    if err and not initialized:
        msgs.append(f"EP error: {err[:400]}")
    if deploy_only:
        msgs.append("Note: this is the deployment-only build (cannot compile models "
                    "at runtime). Pre-compiled PHX models run; compiling your own needs "
                    "AMD's account-gated Ryzen AI installer.")
    return EpProbe(vitisai, initialized, sess_providers or avps, compile_supported,
                   msgs, deploy_only)


def status_line(p: EpProbe) -> str:
    flags = []
    flags.append("ep_available=" + ("yes" if p.vitisai_available else "no"))
    flags.append("ep_init_on_npu=" + ("yes" if p.session_initialized else "no"))
    flags.append("compile=" + ("yes" if p.compile_supported else ("deployment_only" if p.deployment_only else "no")))
    return "vitisai: " + " ".join(flags)


if __name__ == "__main__":
    p = probe_ep()
    print(status_line(p))
    for m in p.messages:
        print(m)
