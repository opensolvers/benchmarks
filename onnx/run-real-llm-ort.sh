#!/usr/bin/env bash
# Build + run real int4 ONNX LLM (Qwen2.5-0.5B) through m1pack ORT on X60.
set -euo pipefail

EESSI_ROOT="${EESSI_ROOT:-/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software}"
GCC14="${GCC14:-$EESSI_ROOT/GCCcore/14.3.0}"
ICONV="${ICONV:-$EESSI_ROOT/libiconv/1.18-GCCcore-14.3.0/lib}"
ORT_ROOT="${ORT_ROOT:-$HOME/x60-work/build/ONNXRuntime/1.29.0/foss-2025b-xsmtvdot}"
ORT_BUILD="$ORT_ROOT/easybuild_obj"
ORT_INC="$ORT_ROOT/onnxruntime-361184e61957410f19153754f325806972546d5b/include/onnxruntime/core/session"
MODEL_DIR="${MODEL_DIR:-$HOME/onnx-models/qwen25-0.5b-int4}"
SRC="${SRC:-$HOME/onnx-mlas-bench/run_real_llm_ort.cpp}"
OUT="${OUT:-$HOME/onnx-mlas-bench/run_real_llm_ort}"
NEW_TOKENS="${NEW_TOKENS:-12}"
THREADS="${THREADS:-1}"

export PATH="$GCC14/bin:$PATH"
export LD_LIBRARY_PATH="$ICONV:$GCC14/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$GCC14/lib64:${LIBRARY_PATH:-}"

echo "=== MatMulNBits / accuracy_level in model.onnx ==="
python3 - <<PY
from pathlib import Path
p = Path("$MODEL_DIR/model.onnx")
b = p.read_bytes()
print("size", len(b))
print("MatMulNBits substrings", b.count(b"MatMulNBits"))
print("accuracy_level substrings", b.count(b"accuracy_level"))
PY

echo "=== build $OUT ==="
g++ -O2 -std=c++17 -Wall -I"$ORT_INC" "$SRC" \
  -L"$ORT_BUILD" -lonnxruntime -Wl,-rpath,"$ORT_BUILD" -o "$OUT"

echo "=== run real LLM decode (taskset -c 0, ${THREADS} thread) ==="
taskset -c 0 "$OUT" "$MODEL_DIR/model.onnx" "$NEW_TOKENS" "$THREADS"
echo "=== DONE ==="
