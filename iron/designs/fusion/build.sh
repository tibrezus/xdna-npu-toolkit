#!/usr/bin/env bash
# Build the fused 2-stage add-add design for Phoenix (NPU1) via Peano + aiecc.
set -e
source /tmp/iron/iron-env.sh >/dev/null 2>&1

HERE=/tmp/iron/fusion
ELTWISE=/tmp/mliraie-v132/programming_examples/ml/eltwise_add
mkdir -p "$HERE/build"

# 1. emit MLIR
echo "[1/3] emit MLIR..."
cd "$HERE" && python3 fused_add_add.py > build/aie.mlir

# 2. reuse the already-built add.o kernel (copy into build dir for aiecc)
cp "$ELTWISE/build/add.o" "$HERE/build/add.o"

# 3. aiecc -> xclbin + insts
echo "[2/3] aiecc compile (Peano) -> xclbin..."
cd "$HERE/build" && aiecc --aie-generate-xclbin --aie-generate-npu-insts --no-compile-host \
    --no-xchesscc --no-xbridge \
    --xclbin-name=final.xclbin --npu-insts-name=insts.bin aie.mlir 2>&1 | tail -3

echo "[3/3] done."
ls -la "$HERE/build/final.xclbin" "$HERE/build/insts.bin"
