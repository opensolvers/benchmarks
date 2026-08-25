#!/usr/bin/env bash
# Apply m1pack ship set (driver + kernel + .inc) to ORT on RV2.
set -euo pipefail

ORT_ROOT="${ORT_ROOT:-$HOME/x60-work/build/ONNXRuntime/1.29.0/foss-2025b-xsmtvdot}"
ORT_SRC="$ORT_ROOT/onnxruntime-361184e61957410f19153754f325806972546d5b/onnxruntime"
PATCH_DIR="${PATCH_DIR:-$HOME/onnx-mlas-bench}"

MLAS_LIB="$ORT_SRC/core/mlas/lib"
IME="$MLAS_LIB/riscv64"

need() { test -f "$PATCH_DIR/$1" || { echo "missing $PATCH_DIR/$1"; exit 2; }; }
for f in qnbitgemm.h qnbitgemm.cpp sqnbitgemm_kernel_ime.m1pack.cpp \
         sqnbitgemm_ime_m1_panel.inc sqnbitgemm_ime_quantize.inc; do
  need "$f"
done

stamp=$(date +%Y%m%d-%H%M%S)
bak="$MLAS_LIB/.pre-m1pack-$stamp"
mkdir -p "$bak"
cp -a "$MLAS_LIB/qnbitgemm.h" "$MLAS_LIB/qnbitgemm.cpp" \
      "$IME/sqnbitgemm_kernel_ime.cpp" "$bak/"
test -f "$IME/sqnbitgemm_ime_m1_panel.inc" && \
  cp -a "$IME/sqnbitgemm_ime_m1_panel.inc" "$bak/" || true
test -f "$IME/sqnbitgemm_ime_quantize.inc" && \
  cp -a "$IME/sqnbitgemm_ime_quantize.inc" "$bak/" || true
echo "backup: $bak"

cp -a "$PATCH_DIR/qnbitgemm.h" "$PATCH_DIR/qnbitgemm.cpp" "$MLAS_LIB/"
cp -a "$PATCH_DIR/sqnbitgemm_kernel_ime.m1pack.cpp" "$IME/sqnbitgemm_kernel_ime.cpp"
cp -a "$PATCH_DIR/sqnbitgemm_ime_m1_panel.inc" "$PATCH_DIR/sqnbitgemm_ime_quantize.inc" "$IME/"

echo "m1pack applied to $ORT_SRC"
