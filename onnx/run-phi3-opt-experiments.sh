#!/usr/bin/env bash
# Phi-3 optimization experiments: profile + IntraOp thread sweep.
set -euo pipefail

EESSI_ROOT="${EESSI_ROOT:-/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software}"
GCC14="${GCC14:-$EESSI_ROOT/GCCcore/14.3.0}"
ICONV="${ICONV:-$EESSI_ROOT/libiconv/1.18-GCCcore-14.3.0/lib}"
ORT_ROOT="${ORT_ROOT:-$HOME/x60-work/build/ONNXRuntime/1.29.0/foss-2025b-xsmtvdot}"
ORT_BUILD="$ORT_ROOT/easybuild_obj"
ORT_INC="$ORT_ROOT/onnxruntime-361184e61957410f19153754f325806972546d5b/include/onnxruntime/core/session"
MODEL_DIR="${MODEL_DIR:-$HOME/onnx-models/phi3-mini-4k-int4}"
MODEL_ONNX="${MODEL_ONNX:-$MODEL_DIR/model_acc4.onnx}"
PATCH_DIR="${PATCH_DIR:-$HOME/onnx-mlas-bench}"
SRC="${SRC:-$PATCH_DIR/run_real_llm_ort.cpp}"
OUT="${OUT:-$PATCH_DIR/run_real_llm_ort}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"
stamp=$(date +%Y%m%d-%H%M%S)
LOG="$LOG_DIR/phi3-opt-$stamp.log"

export PATH="$GCC14/bin:$PATH"
export LD_LIBRARY_PATH="$ICONV:$GCC14/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$GCC14/lib64:${LIBRARY_PATH:-}"

export LLM_KV_HEADS=32 LLM_HEAD_DIM=96 LLM_LAYERS=32 LLM_MAX_SEQ=256
export LLM_PROMPT_TOKENS=32010,1724,338,29871,29906,29974,29906,29973,32007,32001
export LLM_EOS_IDS=-1

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "=== build runner ==="
g++ -O2 -std=c++17 -Wall -I"$ORT_INC" "$SRC" \
  -L"$ORT_BUILD" -lonnxruntime -Wl,-rpath,"$ORT_BUILD" -o "$OUT"

echo "=== 1) PROFILE prefill + 1 decode (1 thread, core 0) ==="
PROFILE_PREFIX="$LOG_DIR/phi3-profile-$stamp"
taskset -c 0 "$OUT" "$MODEL_ONNX" 2 1 "$PROFILE_PREFIX"
profile_json=$(ls -t "${PROFILE_PREFIX}"*.json 2>/dev/null | head -1 || true)
echo "profile_json=$profile_json"
if [[ -n "$profile_json" && -f "$profile_json" ]]; then
  echo "--- decode op mix ---"
  python3 "$PATCH_DIR/parse_ort_profile.py" "$profile_json" decode 25
  echo "--- prefill op mix ---"
  python3 "$PATCH_DIR/parse_ort_profile.py" "$profile_json" prefill 15
fi

echo ""
echo "=== 2) THREAD SWEEP (8 decode tokens, no early EOS) ==="
for th in 1 2 4 8; do
  if [[ "$th" -eq 1 ]]; then
    cpus="0"
  else
    cpus="0-$((th - 1))"
  fi
  echo "--- threads=$th taskset -c $cpus ---"
  # average decode ms from runner lines with cur=1
  out=$(taskset -c "$cpus" "$OUT" "$MODEL_ONNX" 8 "$th")
  echo "$out" | grep -E 'cur=|loaded|prefill|decode|OK|FATAL' || true
  echo "$out" | awk '
    /cur=1 / {
      n++; sum+=$4; if(n==1||$4<min)min=$4; if(n==1||$4>max)max=$4
    }
    END {
      if(n>0) printf("decode_ms avg=%.1f min=%.1f max=%.1f n=%d\n", sum/n, min, max, n)
    }'
done

echo "=== DONE log=$LOG ==="
