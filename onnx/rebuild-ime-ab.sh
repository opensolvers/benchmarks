#!/usr/bin/env bash
# Rebuild ORT MLAS with patched IME kernel and A/B qnbit-mlas-bench on RV2.
set -euo pipefail

ORT_ROOT="${ORT_ROOT:-$HOME/x60-work/build/ONNXRuntime/1.29.0/foss-2025b-xsmtvdot}"
ORT_SRC="$ORT_ROOT/onnxruntime-361184e61957410f19153754f325806972546d5b/onnxruntime"
ORT_BUILD="$ORT_ROOT/easybuild_obj"
IME_CPP="$ORT_SRC/core/mlas/lib/riscv64/sqnbitgemm_kernel_ime.cpp"
BENCH_DIR="${BENCH_DIR:-$HOME/onnx-mlas-bench}"
PATCHED="${1:?usage: $0 <patched-sqnbitgemm_kernel_ime.cpp> [bench-suffix]}"
OUT_SUFFIX="${2:-}"

GCC14="${GCC14:-$HOME/eessi-x60/versions/2025.06-001/software/linux/riscv64/generic/software/GCCcore/14.3.0}"
if [[ ! -x "$GCC14/bin/g++" ]]; then
  # shellcheck disable=SC1091
  source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash 2>/dev/null || true
  module use /cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/modules/all 2>/dev/null || true
  module load GCCcore/14.3.0 2>/dev/null || true
  GCC14="$(dirname "$(dirname "$(command -v g++)")")"
fi
export PATH="$GCC14/bin:$PATH"
export LD_LIBRARY_PATH="$GCC14/lib64:${LD_LIBRARY_PATH:-}"

SMT_AS="${SMT_AS:-$HOME/eessi-x60/versions/2025.06-001/software/linux/riscv64/generic/software/binutils/2.46.1-xsmtvdot/bin}"
if [[ -x "$SMT_AS/as" ]]; then
  export PATH="$SMT_AS:$PATH"
fi

cp -a "$IME_CPP" "${IME_CPP}.pre-hoist.bak"
cp -a "$PATCHED" "$IME_CPP"

# Rebuild just the IME object + refresh libonnxruntime_mlas.a
OBJ="$ORT_BUILD/CMakeFiles/onnxruntime_mlas.dir$IME_CPP.o"
# cmake uses full path as part of object path:
OBJ=$(find "$ORT_BUILD/CMakeFiles/onnxruntime_mlas.dir" -name 'sqnbitgemm_kernel_ime.cpp.o' | head -1)
echo "OBJ=$OBJ"
if command -v ninja >/dev/null 2>&1; then
  ninja -C "$ORT_BUILD" onnxruntime_mlas
else
  make -C "$ORT_BUILD" -j"$(nproc)" onnxruntime_mlas
fi

mkdir -p "$BENCH_DIR"
# Prefer local bench sources if present
if [[ -f "$BENCH_DIR/bench_qnbit_mlas.cpp" ]]; then
  SRC="$BENCH_DIR/bench_qnbit_mlas.cpp"
elif [[ -f "$HOME/opensolvers-benchmarks/onnx/bench_qnbit_mlas.cpp" ]]; then
  SRC="$HOME/opensolvers-benchmarks/onnx/bench_qnbit_mlas.cpp"
else
  echo "need bench_qnbit_mlas.cpp in $BENCH_DIR"; exit 2
fi

ABSEIL=$(find "$ORT_BUILD/_deps/abseil_cpp-build" -name 'libabsl_*.a' | tr '\n' ' ')
g++ -O2 -std=c++17 -Wall -march=rv64gcv_zvl256b_zfh_zvfh \
  -I"$ORT_SRC/core/mlas/inc" -I"$ORT_SRC" \
  "$SRC" \
  -Wl,--start-group "$ORT_BUILD/libonnxruntime_mlas.a" "$ORT_BUILD/libonnxruntime_common.a" $ABSEIL -Wl,--end-group \
  -pthread -o "$BENCH_DIR/qnbit-mlas-bench${OUT_SUFFIX}"

echo "=== baseline shapes (qnbit-mlas-bench${OUT_SUFFIX}) ==="
taskset -c 0 "$BENCH_DIR/qnbit-mlas-bench${OUT_SUFFIX}" 1 4096 11008 20
taskset -c 0 "$BENCH_DIR/qnbit-mlas-bench${OUT_SUFFIX}" 4 4096 11008 10
taskset -c 0 "$BENCH_DIR/qnbit-mlas-bench${OUT_SUFFIX}" 1 4096 4096 20
echo DONE
