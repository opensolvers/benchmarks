#!/usr/bin/env bash
# i8i8 kernel microbench + RVV baseline on RV2 (cluster 0).
set -euo pipefail
LOG="${LOG:-$HOME/logs/i8i8-selftest-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1
echo "START $(date -Iseconds) log=$LOG"

Q8=~/eessi-x60/versions/2025.06-001/software/linux/riscv64/generic/software/llama.cpp/ad8d821-foss-2025b-x60-ime-q8_0
PIN="taskset -c 0"
REPS="${REPS:-25}"

if [[ ! -x ~/bench_i8i8 ]]; then
  echo "missing ~/bench_i8i8 — build with libggml-cpu from $Q8"
  exit 1
fi

run_i8i8() {
  local tag=$1 M=$2 N=$3 K=$4
  echo ""
  echo "==== i8i8 $tag ${M}x${N}x${K} ===="
  LD_LIBRARY_PATH="$Q8/lib:$Q8/lib64:${LD_LIBRARY_PATH:-}" \
    $PIN ~/bench_i8i8 "$M" "$N" "$K" "$REPS"
}

run_rvv() {
  local tag=$1 M=$2 N=$3 K=$4
  echo ""
  echo "==== ime-bench RVV $tag ${M}x${N}x${K} ===="
  $PIN ~/ime-bench/ime-bench "$M" "$N" "$K" 2>/dev/null | grep -E 'rvv|ime|ref|ok'
}

echo "==== correctness chk 512^3 ===="
LD_LIBRARY_PATH="$Q8/lib" $PIN ~/bench_i8i8 512 512 512 0 0 1 0 || echo "chk FAILED"

for sz in 512:512:512 768:768:512 1024:1024:512; do
  IFS=: read -r M N K <<<"$sz"
  run_i8i8 "$sz" "$M" "$N" "$K"
  run_rvv "$sz" "$M" "$N" "$K"
done

echo "DONE $(date -Iseconds)"
