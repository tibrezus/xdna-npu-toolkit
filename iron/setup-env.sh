#!/bin/bash
# Strategy A — IRON/Peano env for Phoenix (NPU1) on Arch. Source this.
# Requires: run bootstrap.sh first (creates env/ + mlir-aie/).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SP="$HERE/env/lib/python3.14/site-packages"

export PEANO_INSTALL_DIR="$SP/llvm-aie"
export MLIR_AIE_INSTALL_DIR="$SP/mlir_aie"
export XRT_INC_DIR=/usr/include
export XRT_LIB_DIR=/usr/lib
export PATH="$MLIR_AIE_INSTALL_DIR/bin:$PEANO_INSTALL_DIR/bin:$PATH"

source "$HERE/env/bin/activate"
export PYTHONPATH=/usr/lib/python3.14/site-packages   # system pyxrt (Arch 'xrt' pkg)

echo "[iron] PEANO=$PEANO_INSTALL_DIR"
echo "[iron] MLIR_AIE=$MLIR_AIE_INSTALL_DIR  aiecc=$(command -v aiecc 2>/dev/null || echo '?')"
