#!/usr/bin/env bash
# Apply m1pack + RISC-V fp16 RVV dispatch (HGEMM, softmax, eltwise, GQA) to ORT on X60.
set -euo pipefail

ORT_ROOT="${ORT_ROOT:-$HOME/x60-work/build/ONNXRuntime/1.29.0/foss-2025b-xsmtvdot}"
ORT_TOP="$ORT_ROOT/onnxruntime-361184e61957410f19153754f325806972546d5b"
ORT_SRC="$ORT_TOP/onnxruntime"
PATCH_DIR="${PATCH_DIR:-$HOME/onnx-mlas-bench}"
REPO_VENDOR="$(cd "$(dirname "${BASH_SOURCE[0]}")/vendor" && pwd)"

stamp=$(date +%Y%m%d-%H%M%S)
bak="$ORT_SRC/core/mlas/lib/.pre-fp16-rvv-$stamp"
mkdir -p "$bak"

echo "=== m1pack ==="
bash "$PATCH_DIR/apply-ime-m1pack.sh"

echo "=== enable MLAS RVV Zvfh in cmake ==="
ORT_BUILD="${ORT_BUILD:-$ORT_ROOT/easybuild_obj}"
cmake -S "$ORT_TOP/cmake" -B "$ORT_BUILD" -Donnxruntime_USE_RVV_ZVFH=ON

echo "=== backup + apply fp16 RVV ==="
cp -a "$ORT_SRC/core/mlas/lib/platform.cpp" \
      "$ORT_SRC/core/mlas/lib/mlasi.h" \
      "$ORT_SRC/core/mlas/lib/halfgemm.h" \
      "$ORT_SRC/core/mlas/lib/riscv64/halfgemm_kernel_rvv.cpp" \
      "$ORT_SRC/contrib_ops/cpu/bert/gqa_attention_base.h" \
      "$bak/" 2>/dev/null || true

cp -a "$REPO_VENDOR/ort-src/core/mlas/lib/platform.cpp" "$ORT_SRC/core/mlas/lib/"
cp -a "$REPO_VENDOR/ort-src/core/mlas/lib/mlasi.h" "$ORT_SRC/core/mlas/lib/"
cp -a "$REPO_VENDOR/ort-src/core/mlas/lib/halfgemm.h" "$ORT_SRC/core/mlas/lib/"
cp -a "$REPO_VENDOR/ort-src/core/mlas/lib/riscv64/halfgemm_kernel_rvv.cpp" \
      "$ORT_SRC/core/mlas/lib/riscv64/"
cp -a "$REPO_VENDOR/fp16/hgemm_kernel_rvv.cpp" \
      "$REPO_VENDOR/fp16/softmax_kernel_rvv_fp16.cpp" \
      "$REPO_VENDOR/fp16/eltwise_kernel_rvv.cpp" \
      "$ORT_SRC/core/mlas/lib/riscv64/"
cp -a "$REPO_VENDOR/ort-src/contrib_ops/cpu/bert/gqa_attention_base.h" \
      "$ORT_SRC/contrib_ops/cpu/bert/"
cp -a "$REPO_VENDOR/ort-src/cmake/onnxruntime_mlas.cmake" "$ORT_TOP/cmake/onnxruntime_mlas.cmake"

echo "backup: $bak"
echo "fp16 RVV applied to $ORT_SRC"
