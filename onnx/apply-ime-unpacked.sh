#!/usr/bin/env bash
# Apply IME unpacked-B driver hook + kernel to the ORT tree on RV2, rebuild MLAS, A/B bench.
set -euo pipefail

ORT_ROOT="${ORT_ROOT:-$HOME/x60-work/build/ONNXRuntime/1.29.0/foss-2025b-xsmtvdot}"
ORT_SRC="$ORT_ROOT/onnxruntime-361184e61957410f19153754f325806972546d5b/onnxruntime"
ORT_BUILD="$ORT_ROOT/easybuild_obj"
PATCH_DIR="${PATCH_DIR:-$HOME/onnx-mlas-bench}"
BENCH_DIR="${BENCH_DIR:-$HOME/onnx-mlas-bench}"

for f in qnbitgemm.h qnbitgemm.cpp sqnbitgemm_kernel_ime.unpacked.cpp; do
  test -f "$PATCH_DIR/$f" || { echo "missing $PATCH_DIR/$f"; exit 2; }
done

MLAS_LIB="$ORT_SRC/core/mlas/lib"
IME_CPP="$MLAS_LIB/riscv64/sqnbitgemm_kernel_ime.cpp"

cp -a "$MLAS_LIB/qnbitgemm.h" "${MLAS_LIB}/qnbitgemm.h.pre-unpacked.bak"
cp -a "$MLAS_LIB/qnbitgemm.cpp" "${MLAS_LIB}/qnbitgemm.cpp.pre-unpacked.bak"
cp -a "$IME_CPP" "${IME_CPP}.pre-unpacked.bak"

cp -a "$PATCH_DIR/qnbitgemm.h" "$MLAS_LIB/qnbitgemm.h"
cp -a "$PATCH_DIR/qnbitgemm.cpp" "$MLAS_LIB/qnbitgemm.cpp"
cp -a "$PATCH_DIR/sqnbitgemm_kernel_ime.unpacked.cpp" "$IME_CPP"

GCC14="${GCC14:-$HOME/eessi-x60/versions/2025.06-001/software/linux/riscv64/generic/software/GCCcore/14.3.0}"
SMT_AS="${SMT_AS:-$HOME/eessi-x60/versions/2025.06-001/software/linux/riscv64/generic/software/binutils/2.46.1-xsmtvdot/bin}"
export PATH="$SMT_AS:$GCC14/bin:$PATH"
export LD_LIBRARY_PATH="$GCC14/lib64:${LD_LIBRARY_PATH:-}"

cd "$ORT_BUILD"
if command -v ninja >/dev/null 2>&1; then
  ninja onnxruntime_mlas
else
  make -j"$(nproc)" onnxruntime_mlas
fi

ABSEIL=$(find "$ORT_BUILD/_deps/abseil_cpp-build" -name 'libabsl_*.a' | tr '\n' ' ')
g++ -O2 -std=c++17 -Wall -march=rv64gcv_zvl256b_zfh_zvfh \
  -I"$ORT_SRC/core/mlas/inc" -I"$ORT_SRC" \
  "$BENCH_DIR/bench_qnbit_mlas.cpp" \
  -Wl,--start-group "$ORT_BUILD/libonnxruntime_mlas.a" "$ORT_BUILD/libonnxruntime_common.a" $ABSEIL -Wl,--end-group \
  -pthread -o "$BENCH_DIR/qnbit-mlas-bench-unpacked"

echo '=== unpacked (driver hook + pack-time int8 B) ==='
taskset -c 0 "$BENCH_DIR/qnbit-mlas-bench-unpacked" 1 4096 11008 12
taskset -c 0 "$BENCH_DIR/qnbit-mlas-bench-unpacked" 4 4096 11008 6
taskset -c 0 "$BENCH_DIR/qnbit-mlas-bench-unpacked" 1 4096 4096 12
echo DONE
