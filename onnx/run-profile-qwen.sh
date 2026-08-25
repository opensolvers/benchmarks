#!/usr/bin/env bash
# Profile one Qwen decode step with ORT chrome trace + summarize op mix.
set -euo pipefail

EESSI_ROOT="${EESSI_ROOT:-/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software}"
GCC14="${GCC14:-$EESSI_ROOT/GCCcore/14.3.0}"
ICONV="${ICONV:-$EESSI_ROOT/libiconv/1.18-GCCcore-14.3.0/lib}"
ORT_ROOT="${ORT_ROOT:-$HOME/x60-work/build/ONNXRuntime/1.29.0/foss-2025b-xsmtvdot}"
ORT_BUILD="$ORT_ROOT/easybuild_obj"
ORT_INC="$ORT_ROOT/onnxruntime-361184e61957410f19153754f325806972546d5b/include/onnxruntime/core/session"
MODEL_DIR="${MODEL_DIR:-$HOME/onnx-models/qwen25-0.5b-int4}"
PATCH_DIR="${PATCH_DIR:-$HOME/onnx-mlas-bench}"
SRC="${SRC:-$PATCH_DIR/run_real_llm_ort.cpp}"
OUT="${OUT:-$PATCH_DIR/run_real_llm_ort}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"
stamp=$(date +%Y%m%d-%H%M%S)
PROFILE_PREFIX="${PROFILE_PREFIX:-$LOG_DIR/qwen-profile-$stamp}"

export PATH="$GCC14/bin:$PATH"
export LD_LIBRARY_PATH="$ICONV:$GCC14/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$GCC14/lib64:${LIBRARY_PATH:-}"

mkdir -p "$LOG_DIR"
log="$LOG_DIR/qwen-profile-run-$stamp.log"

echo "=== build runner ===" | tee "$log"
g++ -O2 -std=c++17 -Wall -I"$ORT_INC" "$SRC" \
  -L"$ORT_BUILD" -lonnxruntime -Wl,-rpath,"$ORT_BUILD" -o "$OUT" 2>&1 | tee -a "$log"

echo "=== profile prefill + 1 decode token ===" | tee -a "$log"
taskset -c 0 "$OUT" "$MODEL_DIR/model.onnx" 2 1 "$PROFILE_PREFIX" 2>&1 | tee -a "$log"

profile_json=$(grep -o 'profile_json=.*' "$log" | tail -1 | cut -d= -f2-)
if [[ -z "$profile_json" || ! -f "$profile_json" ]]; then
  profile_json=$(ls -t "${PROFILE_PREFIX}"*.json 2>/dev/null | head -1 || true)
fi
echo "profile_json=$profile_json" | tee -a "$log"

if [[ -z "$profile_json" || ! -f "$profile_json" ]]; then
  echo "ERROR: no profile JSON found" | tee -a "$log"
  exit 1
fi

echo "" | tee -a "$log"
echo "=== OP MIX: decode (1 token) ===" | tee -a "$log"
python3 "$PATCH_DIR/parse_ort_profile.py" "$profile_json" decode 30 | tee -a "$log"

echo "" | tee -a "$log"
echo "=== OP MIX: prefill ===" | tee -a "$log"
python3 "$PATCH_DIR/parse_ort_profile.py" "$profile_json" prefill 20 | tee -a "$log"

echo "" | tee -a "$log"
echo "=== OP MIX: all runs ===" | tee -a "$log"
python3 "$PATCH_DIR/parse_ort_profile.py" "$profile_json" all 15 | tee -a "$log"

echo "=== DONE log=$log profile=$profile_json ===" | tee -a "$log"
