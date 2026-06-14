# xdna-npu-toolkit

**Detect, validate, enable and assess the AMD XDNA NPU on Linux.** Dependency-free (stdlib Python ≥3.8).

This toolkit answers the question every Ryzen-AI laptop owner eventually asks:
*"Can I run LLMs on my NPU on Linux?"* It gives a **machine-specific, evidence-based
verdict** by reading the live AIE tile-array geometry straight from the `amdxdna`
driver — and it fixes the two common blockers that keep a stock NPU from being usable
at all (the memlock limit and the NPU power mode).

It does **not** run LLMs. On XDNA 1 (Phoenix / Hawk Point, e.g. the Ryzen 7 7840HS) no
open tool *can* — see [Why can't I run LLMs on my XDNA 1 NPU?](#why-cant-i-run-llms-on-my-xdna-1-npu)
below. This toolkit is the honest, working layer *underneath* that: it gets the NPU
fully enabled and tells you exactly what is and isn't possible on your silicon.

---

## What it does

| Command | What it does | Needs root? |
|---|---|---|
| `xdna-npu doctor` | Full hardware detection + stack validation + feasibility verdict (the main command) | no |
| `xdna-npu detect` | NPU hardware detection (PCI, driver, firmware, live AIE cols/rows) | no |
| `xdna-npu validate` | Stack checks (kernel driver, device node, firmware, XRT, plugin, memlock) | no |
| `xdna-npu enable` | Install the memlock drop-in + set NPU power mode to performance | **yes** |
| `xdna-npu embed-check` | Probe whether the AMD VitisAI EP initializes on the XDNA 1 NPU | no |
| `xdna-npu embed-setup` | Install the AMD VitisAI stack (onnxruntime-vitisai, voe, dynamic-dispatch) | no |
| `xdna-npu status` | One-line machine-readable status | no |

`doctor`, `detect`, `validate` and `status` issue the `DRM_IOCTL_AMDXDNA_GET_INFO`
ioctl directly to `/dev/accel/accelN` to read the **live** AIE metadata (columns,
rows, AIE generation) and firmware version. This is a *metadata-only* query — it does
not allocate DMA buffers — so it works for any user with access to the device node,
with no raised memlock limit and no root.

## Quick start

```bash
git clone https://github.com/tibrezus/xdna-npu-toolkit
cd xdna-npu-toolkit
python3 -m xdna_npu doctor
```

Or one-shot:

```bash
curl -fsSL https://raw.githubusercontent.com/tibrezus/xdna-npu-toolkit/main/install.sh | bash
```

## Example output

On a **Ryzen 7 7840HS** (XDNA 1, Phoenix) after `sudo xdna-npu enable`:

```
── NPU hardware ────────────────────────────────────────────────
  PCI address     : 0000:c9:00.1
  Vendor:Device   : 0x1022:0x1502
  Driver          : amdxdna
  Device node     : /dev/accel/accel0
  Codename        : Phoenix / Hawk Point
  XDNA generation : XDNA 1
  AIE family      : AIE-ML (AIE2)
  AIE array (live): 5 cols x 6 rows, col_size=504, AIE v1.1
  Firmware (live) : 1.5.5.391

── Stack validation ────────────────────────────────────────────
  ✓ [PASS] kernel driver 'amdxdna' bound
  ✓ [PASS] device node accessible
  ✓ [PASS] NPU firmware loaded (1.5.5.391)
  ✓ [PASS] XRT userspace runtime
  ✓ [PASS] XRT amdxdna plugin
  ✓ [PASS] memlock (RLIMIT_MEMLOCK): unlimited

── LLM feasibility verdict ─────────────────────────────────────
  LLMs NOT FEASIBLE on this NPU (XDNA 1, 5 cols)
  This NPU has only 5 AIE columns (AIE-ML (AIE2), XDNA 1, ~10 TOPS). Every public LLM
  runtime (FastFlowLM, Lemonade 10) ships overlays compiled for XDNA 2 (8 columns,
  AIE2P) and hard-requires them; there is no open AIE overlay compiler and no
  redistributable Phoenix overlay. ...
```

JSON is available for all read-only commands (`--json`), and `status` is one line:

```
npu=/dev/accel/accel0 gen=XDNA 1 cols=5 fw=1.5.5.391 llm_capable=no
```

## Why can't I run LLMs on my XDNA 1 NPU?

The XDNA NPU is a **spatial dataflow accelerator**, not a programmable GPU. It cannot
run arbitrary code — every op must be *offline compiled, place-and-routed onto the AIE
tile array*, and shipped as an `.xclbin` overlay bitstream. Running a transformer
therefore needs four proprietary layers stacked on top of the open kernel driver:

1. an **operator compiler** (VAIP / Vitis AI),
2. **precompiled overlays** (`.xclbin` binaries),
3. an **ONNX-RT execution provider**, and
4. a **tuned, quantized model zoo**.

That entire stack is built Windows-first, and the open turnkey Linux runners
(FastFlowLM, Lemonade 10) compile their overlays for **XDNA 2** (Strix Point:
8 columns, AIE2P tiles, ~50 TOPS). Your XDNA 1 chip is 5 columns of AIE-ML (~10 TOPS);
an 8-column AIE2P overlay simply cannot load on it. There is no open AIE overlay
compiler and no redistributable Phoenix overlay, so it cannot be rebuilt from source.

This is **not** fixable by removing the `cols < 8` guard in those tools — that just
changes the error from a clean message to an XRT overlay-load failure. See the
[full feasibility analysis](https://github.com/tibrezus/FastFlowLM/blob/docs/xdna1-feasibility/docs/XDNA1-FEASIBILITY.md)
in the FastFlowLM fork.

### The realistic local-LLM path on a 7840HS

Use the **integrated GPU (Radeon 780M) via ROCm/HIP**, which shares system memory:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH="gfx1100"
# llama.cpp with: -DLLAMA_HIPBLAS=ON -DLLAMA_HIP_UMA=ON
```

The NPU stays enabled and ready for classic ONNX vision/NLP workloads once a
redistributable Phoenix overlay exists.

## What about embeddings / CNNs / transformers (not LLMs)?

**This is a different — and more positive — story than LLMs.** Smaller models
(BERT/MiniLM/E5/GTE-class embeddings, ResNet/YOLO vision) are exactly what AMD's
VitisAI stack was built for, and that stack *does* support Phoenix.

`xdna-npu embed-check` probes it live. On a Ryzen 7 7840HS:

```
── AMD VitisAI Execution Provider probe (XDNA 1) ───────────────
  ✓ EP listed by onnxruntime : True
  ✓ EP initializes on NPU   : True
  providers                   : VitisAIExecutionProvider, CPUExecutionProvider
  at-runtime compile available: False (deployment-only build)
```

i.e. **the NPU execution provider initializes on XDNA 1**. Unlike LLMs (whose
runtimes reject Phoenix outright), transformer/CNN inference is *runtime-
supported* on XDNA 1 on Linux. The whole public runtime stack —
`onnxruntime-vitisai`, `voe` (with `4x4`/Phoenix kernels), `ryzenai-dynamic-
dispatch`, `ryzenai-onnx-utils` — installs from AMD's public pip index
(`pypi.amd.com`) with **no account gate**, and the voe compiler code branches
explicitly on `device in ["phx", "stx"]`.

The one remaining gate is *model compilation*. The publicly-installable Linux
wheels are a **deployment-only** build: they *run* a pre-compiled model but emit
`Model compilation is not supported in a deployment only installation` if you
ask them to compile an ONNX graph to the NPU at runtime. The full compiler ships
only inside AMD's account-gated Ryzen AI Software installer. So:

- **You can run a pre-compiled Phoenix embedding model today** (none ships
  publicly yet — the lone HF embedding model, `amd/NPU-Nomic-embed-text-v1.5-ryzen-
  strix-cpp`, is Strix/XDNA2-only). This is a real gap, and a good first target.
- **To compile your own** embedding ONNX to the NPU, you need AMD's installer.

To set up the stack:

```bash
uv python install 3.12          # the AMD wheels are cp312
xdna-npu embed-setup            # installs the AMD wheels, then probes the EP
xdna-npu embed-check            # re-probe anytime
```

## Requirements

- An AMD XDNA NPU and the `amdxdna` kernel driver (mainline since 6.11; best on 7.x).
- For the XRT / power-mode checks: the `xrt` and `xrt-plugin-amdxdna` packages.
  On Arch: `sudo pacman -S xrt xrt-plugin-amdxdna`.
- `linux-firmware` (provides `/lib/firmware/amdnpu/<devid>_00/`).

## Project layout

```
xdna_npu/
  ioctl.py      ctypes mirror of the amdxdna GET_INFO ioctl (AIE metadata + fw version)
  detect.py     PCI/sysfs NPU detection + XDNA-generation mapping
  validate.py   stack checks (driver/node/fw/XRT/plugin/memlock)
  verdict.py    column-count -> feasibility verdict + iGPU fallback path
  enable.py     memlock limits.d drop-in + NPU power mode (root)
  embed.py      AMD VitisAI EP setup + live probe (proves EP initializes on XDNA 1)
  cli.py        argparse front-end
scripts/
  enable-npu.sh standalone shell enablement (no Python needed)
```

## License

MIT — see [LICENSE](LICENSE).

This project is independent and not affiliated with AMD. "Ryzen" and "XDNA" are
trademarks of Advanced Micro Devices, Inc.
