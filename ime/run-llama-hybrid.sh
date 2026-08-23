#!/bin/bash
# Run llama-bench with SPACEMIT hybrid: IME M4 prefill + stock ggml RVV decode (M=1).
# Requires hybrid libggml-cpu from apply-hybrid.py + run-llama-pipe-build.sh on the board.
#
# Usage (on RV2):
#   SPACEMIT_HYBRID=1 bash run-llama-hybrid.sh [llama-bench args...]
#
# Example:
#   SPACEMIT_HYBRID=1 bash run-llama-hybrid.sh \
#     ~/llama-x60/models/qwen2.5-0.5b-q8_0.gguf -p 512 -n 32 -t 4 -r 3
set -eo pipefail

Q8="${Q8_ROOT:-$HOME/eessi-x60/versions/2025.06-001/software/linux/riscv64/generic/software/llama.cpp/ad8d821-foss-2025b-x60-ime-q8_0}"
HY_LIB="${HYBRID_GGML_CPU:-$HOME/llama-pipe/build/bin/libggml-cpu.so.0.16.0}"
BENCH="${LLAMA_BENCH:-$Q8/bin/llama-bench}"

export SPACEMIT_HYBRID="${SPACEMIT_HYBRID:-1}"
export SPACEMIT_IME_MIN_M="${SPACEMIT_IME_MIN_M:-4}"
export LD_PRELOAD="${HY_LIB}${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH="${Q8}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

taskset -c 0-3 "$BENCH" "$@"
