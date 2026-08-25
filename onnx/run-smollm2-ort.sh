#!/usr/bin/env bash
# Build + run SmolLM2-360M Instruct int4 through m1pack ORT on X60.
set -euo pipefail

EESSI_ROOT="${EESSI_ROOT:-/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software}"
GCC14="${GCC14:-$EESSI_ROOT/GCCcore/14.3.0}"
ICONV="${ICONV:-$EESSI_ROOT/libiconv/1.18-GCCcore-14.3.0/lib}"
ORT_ROOT="${ORT_ROOT:-$HOME/x60-work/build/ONNXRuntime/1.29.0/foss-2025b-xsmtvdot}"
ORT_BUILD="$ORT_ROOT/easybuild_obj"
ORT_INC="$ORT_ROOT/onnxruntime-361184e61957410f19153754f325806972546d5b/include/onnxruntime/core/session"
MODEL_DIR="${MODEL_DIR:-$HOME/onnx-models/smollm2-360m-int4}"
MODEL_ONNX="${MODEL_ONNX:-$MODEL_DIR/model_acc4.onnx}"
SRC="${SRC:-$HOME/onnx-mlas-bench/run_real_llm_ort.cpp}"
OUT="${OUT:-$HOME/onnx-mlas-bench/run_real_llm_ort}"
NEW_TOKENS="${NEW_TOKENS:-16}"
THREADS="${THREADS:-4}"
PATCH_PY="${PATCH_PY:-$HOME/onnx-mlas-bench/patch_accuracy_level.py}"

export PATH="$GCC14/bin:$PATH"
export LD_LIBRARY_PATH="$ICONV:$GCC14/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$GCC14/lib64:${LIBRARY_PATH:-}"

# SmolLM2-360M: 32 layers, GQA 5 KV heads, head_dim 64
export LLM_KV_HEADS="${LLM_KV_HEADS:-5}"
export LLM_HEAD_DIM="${LLM_HEAD_DIM:-64}"
export LLM_LAYERS="${LLM_LAYERS:-32}"
export LLM_MAX_SEQ="${LLM_MAX_SEQ:-512}"
# Chat: user "What is 2+2?" + assistant prompt
export LLM_PROMPT_TOKENS="${LLM_PROMPT_TOKENS:-1,4093,198,1780,314,216,34,27,34,47,2,198,1,520,9531,198}"
export LLM_EOS_IDS="${LLM_EOS_IDS:-2,0}"

if [[ ! -f "$MODEL_DIR/model.onnx" || ! -f "$MODEL_DIR/model.onnx.data" ]]; then
  echo "missing model under $MODEL_DIR" >&2
  exit 1
fi

if [[ ! -f "$MODEL_ONNX" ]]; then
  echo "=== patch accuracy_level -> $MODEL_ONNX ==="
  python3 "$PATCH_PY" "$MODEL_DIR/model.onnx" "$MODEL_ONNX"
fi

echo "=== MatMulNBits / accuracy_level in $(basename "$MODEL_ONNX") ==="
python3 - <<PY
from pathlib import Path
p = Path("$MODEL_ONNX")
b = p.read_bytes()
print("size", len(b))
print("MatMulNBits substrings", b.count(b"MatMulNBits"))
print("accuracy_level substrings", b.count(b"accuracy_level"))
# block_size hint
print("bits/block_size attrs present via MatMulNBits count only")
PY

echo "=== build $OUT ==="
g++ -O2 -std=c++17 -Wall -I"$ORT_INC" "$SRC" \
  -L"$ORT_BUILD" -lonnxruntime -Wl,-rpath,"$ORT_BUILD" -o "$OUT"

echo "=== run SmolLM2 decode (taskset for ${THREADS} threads) ==="
if [[ "$THREADS" -eq 1 ]]; then CPUS=0; else CPUS="0-$((THREADS - 1))"; fi
taskset -c "$CPUS" "$OUT" "$MODEL_ONNX" "$NEW_TOKENS" "$THREADS"
echo "=== DONE ==="
