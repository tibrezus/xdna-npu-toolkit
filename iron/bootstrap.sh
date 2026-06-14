#!/usr/bin/env bash
# Strategy A — IRON/Peano setup for AMD Ryzen 7 7840HS (Phoenix / NPU1 / AIE2) on Arch Linux.
# Fully open-source: mlir-aie (IRON) + llvm-aie (Peano) pip wheels. No license, no Windows.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "[1/4] venv (python 3.14 — matches system pyxrt from Arch 'xrt' pkg)"
if [ ! -d env ]; then
  uv venv --python /usr/bin/python3.14 env
fi
source env/bin/activate

echo "[2/4] install mlir_aie (v1.3.2) + llvm-aie/Peano (nightly)"
uv pip install --python env/bin/python "mlir_aie==1.3.2" \
  -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.3.2/
uv pip install --python env/bin/python "llvm-aie" \
  -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly/
uv pip install --python env/bin/python numpy pyyaml

echo "[3/4] mlir-aie repo (for programming_examples + aie_kernels)"
if [ ! -d mlir-aie ]; then
  git clone --depth 1 --branch v1.3.2 https://github.com/Xilinx/mlir-aie
fi

echo "[4/4] environment helper written to setup-env.sh"
echo "Done. Next: source setup-env.sh; then build/run a design."
