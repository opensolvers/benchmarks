#!/usr/bin/env bash
# End-to-end ONNX validation: apply m1pack, rebuild ORT, microbench + perf_test.
set -euo pipefail

ORT_ROOT="${ORT_ROOT:-$HOME/x60-work/build/ONNXRuntime/1.29.0/foss-2025b-xsmtvdot}"
ORT_SRC="$ORT_ROOT/onnxruntime-361184e61957410f19153754f325806972546d5b/onnxruntime"
ORT_BUILD="$ORT_ROOT/easybuild_obj"
PATCH_DIR="${PATCH_DIR:-$HOME/onnx-mlas-bench}"
BENCH_DIR="${BENCH_DIR:-$HOME/onnx-mlas-bench}"
MODEL_DIR="${MODEL_DIR:-$HOME/onnx-bench}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"
REPS="${REPS:-8}"

mkdir -p "$MODEL_DIR" "$LOG_DIR"

EESSI_ROOT="${EESSI_ROOT:-/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software}"
GCC14="${GCC14:-$HOME/eessi-x60/versions/2025.06-001/software/linux/riscv64/generic/software/GCCcore/14.3.0}"
SMT_AS="${SMT_AS:-$HOME/eessi-x60/versions/2025.06-001/software/linux/riscv64/generic/software/binutils/2.46.1-xsmtvdot/bin}"
ICONV="${ICONV:-$HOME/eessi-x60/versions/2025.06-001/software/linux/riscv64/generic/software/libiconv/1.18-GCCcore-14.3.0/lib}"
if [[ ! -x "$GCC14/bin/g++" ]]; then
  GCC14="$EESSI_ROOT/GCCcore/14.3.0"
fi
if [[ ! -x "$SMT_AS/as" ]]; then
  SMT_AS="$EESSI_ROOT/binutils/2.46.1-xsmtvdot/bin"
fi
if [[ ! -f "$ICONV/libiconv.so.2" ]]; then
  ICONV="$EESSI_ROOT/libiconv/1.18-GCCcore-14.3.0/lib"
fi
export PATH="$SMT_AS:$GCC14/bin:$PATH"
export LD_LIBRARY_PATH="$ICONV:$GCC14/lib64:${LD_LIBRARY_PATH:-}"

stamp=$(date +%Y%m%d-%H%M%S)
log="$LOG_DIR/onnx-m1pack-e2e-$stamp.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$log") 2>&1

echo "=== m1pack e2e validation $stamp ==="
echo "log: $log"

# 1) Apply patches
bash "$PATCH_DIR/apply-ime-m1pack.sh"

# 2) Rebuild MLAS + relink ORT (perf_test uses libonnxruntime.so)
cd "$ORT_BUILD"
if command -v ninja >/dev/null 2>&1; then
  ninja -j"$(nproc)" onnxruntime_mlas onnxruntime onnxruntime_perf_test
else
  make -j"$(nproc)" onnxruntime_mlas onnxruntime onnxruntime_perf_test
fi
# Ensure shared lib picks up fresh libonnxruntime_mlas.a
rm -f "$ORT_BUILD/libonnxruntime.so" "$ORT_BUILD/libonnxruntime.so.1.29.0"
if command -v ninja >/dev/null 2>&1; then
  ninja -j"$(nproc)" onnxruntime onnxruntime_perf_test
else
  make -j"$(nproc)" onnxruntime onnxruntime_perf_test
fi

PERF="$ORT_BUILD/onnxruntime_perf_test"
test -x "$PERF" || PERF="$ORT_BUILD/Release/onnxruntime_perf_test"
test -x "$PERF" || { echo "onnxruntime_perf_test not found under $ORT_BUILD"; exit 3; }
echo "perf_test: $PERF"

# 3) Microbench sanity (M=1 decode shape)
ABSEIL=$(find "$ORT_BUILD/_deps/abseil_cpp-build" -name 'libabsl_*.a' | tr '\n' ' ')
g++ -O2 -std=c++17 -Wall -march=rv64gcv_zvl256b_zfh_zvfh \
  -I"$ORT_SRC/core/mlas/inc" -I"$ORT_SRC" \
  "$BENCH_DIR/bench_qnbit_mlas.cpp" \
  -Wl,--start-group "$ORT_BUILD/libonnxruntime_mlas.a" \
    "$ORT_BUILD/libonnxruntime_common.a" $ABSEIL -Wl,--end-group \
  -pthread -o "$BENCH_DIR/qnbit-mlas-bench-m1pack-e2e"

echo "=== microbench M=1×4096×11008 ==="
taskset -c 0 "$BENCH_DIR/qnbit-mlas-bench-m1pack-e2e" 1 4096 11008 12

# 4) Model: ensure accuracy_level=4
RAW=""
for cand in "$MODEL_DIR/int4_ffn.onnx" "$HOME/ort-build/int4_ffn.onnx" \
            "$HOME/onnx-bench/int4_ffn.onnx"; do
  if test -f "$cand"; then RAW="$cand"; break; fi
done
test -n "$RAW" || { echo "no int4_ffn.onnx found"; exit 4; }
ACC4="$MODEL_DIR/int4_ffn_acc4.onnx"
echo "model: $RAW -> $ACC4"

if ! grep -q accuracy_level "$ACC4" 2>/dev/null; then
  python3 "$PATCH_DIR/patch_accuracy_level.py" "$RAW" "$ACC4"
fi
acc4_count=$(grep -c accuracy_level "$ACC4" || true)
echo "accuracy_level attrs in $ACC4: $acc4_count (expect 16)"

# 5) E2E perf (stock CompInt8 baseline ~3522 ms x1, ~590 ms x8)
echo "=== onnxruntime_perf_test x1 (M=1 decode) ==="
taskset -c 0 "$PERF" -e cpu -I -m times -r "$REPS" -x 1 "$ACC4"

echo "=== onnxruntime_perf_test x8 ==="
taskset -c 0 "$PERF" -e cpu -I -m times -r "$REPS" -x 8 "$ACC4"

echo "=== DONE — log saved to $log ==="
