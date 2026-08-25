#!/usr/bin/env bash
# Apply m1pack + fp16 MatMulNBits native path (skip bulk fp16↔fp32 on CompInt8).
set -euo pipefail

ORT_ROOT="${ORT_ROOT:-$HOME/x60-work/build/ONNXRuntime/1.29.0/foss-2025b-xsmtvdot}"
ORT_TOP="$ORT_ROOT/onnxruntime-361184e61957410f19153754f325806972546d5b"
ORT_SRC="$ORT_TOP/onnxruntime"
PATCH_DIR="${PATCH_DIR:-$HOME/onnx-mlas-bench}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_VENDOR="${REPO_VENDOR:-$(cd "$SCRIPT_DIR/vendor" 2>/dev/null && pwd || echo "$PATCH_DIR")}"

stamp=$(date +%Y%m%d-%H%M%S)
bak="$ORT_SRC/core/mlas/lib/.pre-matmul-fp16-$stamp"
mkdir -p "$bak"

echo "=== m1pack + fp16 RVV base ==="
bash "$PATCH_DIR/apply-fp16-rvv.sh"

cp -a "$ORT_SRC/core/mlas/inc/mlas_qnbit.h" \
      "$ORT_SRC/contrib_ops/cpu/quantization/matmul_nbits.cc" \
      "$bak/" 2>/dev/null || true

if [[ -f "$PATCH_DIR/mlas_qnbit.h" ]]; then
  cp -a "$PATCH_DIR/mlas_qnbit.h" "$ORT_SRC/core/mlas/inc/mlas_qnbit.h"
elif [[ -f "$REPO_VENDOR/mlas_qnbit.h" ]]; then
  cp -a "$REPO_VENDOR/mlas_qnbit.h" "$ORT_SRC/core/mlas/inc/mlas_qnbit.h"
else
  echo "missing mlas_qnbit.h"; exit 2
fi

if [[ -f "$PATCH_DIR/matmul_nbits.cc" ]]; then
  cp -a "$PATCH_DIR/matmul_nbits.cc" "$ORT_SRC/contrib_ops/cpu/quantization/matmul_nbits.cc"
elif [[ -f "$REPO_VENDOR/ort-src/contrib_ops/cpu/quantization/matmul_nbits.cc" ]]; then
  cp -a "$REPO_VENDOR/ort-src/contrib_ops/cpu/quantization/matmul_nbits.cc" \
        "$ORT_SRC/contrib_ops/cpu/quantization/"
else
  echo "missing matmul_nbits.cc"; exit 2
fi

echo "backup: $bak"
echo "matmul fp16-native + m1pack FromFp16 applied to $ORT_SRC"
