#!/usr/bin/env bash
# Full deploy: m1pack + fp16 RVV patches, rebuild ORT, benchmark Qwen real LLM.
set -euo pipefail

EESSI_ROOT="${EESSI_ROOT:-/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software}"
GCC14="${GCC14:-$EESSI_ROOT/GCCcore/14.3.0}"
ICONV="${ICONV:-$EESSI_ROOT/libiconv/1.18-GCCcore-14.3.0/lib}"
ORT_ROOT="${ORT_ROOT:-$HOME/x60-work/build/ONNXRuntime/1.29.0/foss-2025b-xsmtvdot}"
ORT_BUILD="$ORT_ROOT/easybuild_obj"
PATCH_DIR="${PATCH_DIR:-$HOME/onnx-mlas-bench}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"
stamp=$(date +%Y%m%d-%H%M%S)
log="$LOG_DIR/onnx-fp16-rvv-$stamp.log"

export PATH="$GCC14/bin:$PATH"
export LD_LIBRARY_PATH="$ICONV:$GCC14/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$GCC14/lib64:${LIBRARY_PATH:-}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$log") 2>&1

echo "=== fp16 RVV deploy $stamp ==="
bash "$PATCH_DIR/apply-fp16-rvv.sh"

cd "$ORT_BUILD"
rm -f libonnxruntime.so libonnxruntime.so.1.29.0
if command -v ninja >/dev/null 2>&1; then
  ninja -j"$(nproc)" onnxruntime_mlas onnxruntime onnxruntime_perf_test
  ninja -j"$(nproc)" onnxruntime onnxruntime_perf_test
else
  make -j"$(nproc)" onnxruntime_mlas onnxruntime onnxruntime_perf_test
  make -j"$(nproc)" onnxruntime onnxruntime_perf_test
fi

echo "=== rebuild real LLM runner ==="
bash "$PATCH_DIR/run-real-llm-ort.sh"

echo "=== DONE log=$log ==="
