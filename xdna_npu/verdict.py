"""Feasibility verdict: what can actually run on *this* NPU.

The key insight (see the FastFlowLM feasibility analysis) is that an LLM
runtime needs precompiled ``.xclbin`` overlays sized for the chip's tile
geometry. There is no open AIE overlay compiler and no redistributable
Phoenix overlay, so the column count is the practical gate:

    cols >= 8  -> XDNA 2 (Strix/Kraken/Strix Halo): LLM-capable today
    cols <  8  -> XDNA 1 (Phoenix/Hawk Point):     LLMs NOT feasible on Linux

For XDNA 1 the realistic local-LLM path is the integrated GPU (Radeon 780M
on a 7840HS) via ROCm/HIP, not the NPU.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from dataclasses import dataclass

from .detect import NpuDevice

MIN_COLS_FOR_LLM = 8


@dataclass
class Verdict:
    can_run_llm: bool
    summary: str
    detail: str
    recommendation: str


def _igpu() -> tuple[str | None, str | None]:
    """Detect the AMD integrated GPU (model, gfx arch string) for the fallback path."""
    name = None
    try:
        out = subprocess.run(
            ["lspci", "-nn"], capture_output=True, text=True, check=False
        ).stdout
        for line in out.splitlines():
            if "VGA compatible controller" in line and "AMD" in line:
                # 'c8:00.0 VGA compatible controller [0300]: <NAME> [1002:15bf] (rev c2)'
                desc = line.split("]:", 1)[-1].strip() if "]:" in line else line
                desc = re.sub(r"\[[0-9a-fA-F]{4}:[0-9a-fA-F]{4}\]", "", desc)  # drop ids
                desc = re.sub(r"\(rev[^)]*\)", "", desc).strip(" []")
                name = desc or None
    except Exception:
        pass
    # gfx arch guess for known RDNA3 APUs (Phoenix / Hawk Point)
    gfx = None
    if name and any(k in name for k in ("780M", "760M", "Phoenix", "Radeon 700M")):
        gfx = "gfx1100"
    return name, gfx


def assess(dev: NpuDevice) -> Verdict:
    cols = dev.aie_info.cols if dev.aie_info else None

    if cols is None:
        return Verdict(
            can_run_llm=False,
            summary="UNDETERMINED",
            detail="Could not read the AIE tile-array geometry from the driver, "
                   "so the feasibility verdict cannot be computed.",
            recommendation="Resolve the ioctl / device-node errors above first.",
        )

    if cols >= MIN_COLS_FOR_LLM:
        return Verdict(
            can_run_llm=True,
            summary="LLM-CAPABLE (XDNA 2)",
            detail=f"This NPU has {cols} AIE columns ({dev.aie_family}, {dev.xdna_gen}). "
                   "Open turnkey runtimes (FastFlowLM, Lemonade) target this geometry "
                   "and can run LLMs on the NPU on Linux.",
            recommendation="Install `flm` / Lemonade and run e.g. `flm run llama3.2:1b`.",
        )

    igpu, gfx = _igpu()
    igpu_line = ""
    rec = (
        "For local LLMs on this machine, use the integrated GPU (Radeon) via ROCm/HIP, "
        "not the NPU. Example:\n"
        "    export HSA_OVERRIDE_GFX_VERSION=11.0.0\n"
        "    export PYTORCH_ROCM_ARCH=\"gfx1100\"\n"
        "    # llama.cpp: -DLLAMA_HIPBLAS=ON -DLLAMA_HIP_UMA=ON  (UMA shares system RAM)\n"
        "The NPU stays usable for classic ONNX vision/NLP once a redistributable "
        "Phoenix overlay exists; it is not usable for FastFlowLM-style LLMs today."
    )
    if igpu:
        igpu_line = f"Detected iGPU: {igpu}" + (f" ({gfx})" if gfx else "") + ". "
    return Verdict(
        can_run_llm=False,
        summary=f"LLMs NOT FEASIBLE on this NPU (XDNA 1, {cols} cols)",
        detail=(
            f"This NPU has only {cols} AIE columns ({dev.aie_family}, {dev.xdna_gen}, "
            f"~{'10' if dev.device_id == 0x1502 else '?'} TOPS). Every public LLM runtime "
            "(FastFlowLM, Lemonade 10) ships overlays compiled for XDNA 2 (8 columns, "
            "AIE2P) and hard-requires them; there is no open AIE overlay compiler and no "
            "redistributable Phoenix overlay. Removing their column guard does not help -- "
            "the XDNA2 overlay cannot load on this tile geometry. " + igpu_line
        ),
        recommendation=rec,
    )
