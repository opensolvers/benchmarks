#!/usr/bin/env bash
# Synthetic IME raw throughput: 1 core vs 4 cores (OpenMP M-split on cluster 0).
set -euo pipefail
LOG="${LOG:-$HOME/logs/ime-mc-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1
echo "START $(date -Iseconds) log=$LOG"

BIN=${BIN:-$HOME/ime-bench/ime-bench-mc}
REPS=${REPS:-25}
SIZES=${SIZES:-"512 512 512 768 768 512 1024 1024 512"}

if [[ ! -x "$BIN" ]]; then
  echo "missing $BIN — run: cd ~/ime-bench && make board-mc"
  exit 1
fi

run() {
  local tag=$1 nth=$2 pin=$3 M=$4 N=$5 K=$6
  echo ""
  echo "==== $tag threads=$nth core=$pin ${M}x${N}x${K} ===="
  taskset -c "$pin" "$BIN" "$M" "$N" "$K" "$REPS" "$nth" || true
}

read -r -a sz <<< "$SIZES"
i=0
while (( i + 2 < ${#sz[@]} )); do
  M=${sz[i]} N=${sz[i+1]} K=${sz[i+2]}
  i=$((i + 3))
  run "1c" 1 0 "$M" "$N" "$K"
  run "4c" 4 0-3 "$M" "$N" "$K"
done

echo ""
echo "==== 4× independent 1c (same GEMM, no split) ===="
M=768 N=768 K=512
echo "launch 4 parallel full GEMMs on cores 0-3..."
t0=$(date +%s.%N)
for c in 0 1 2 3; do
  taskset -c "$c" "$HOME/ime-bench/ime-bench" "$M" "$N" "$K" "$REPS" >/tmp/ime-mc-p$$-$c.log &
done
wait
t1=$(date +%s.%N)
wall=$(awk -v t0="$t0" -v t1="$t1" 'BEGIN{printf "%.4f", t1-t0}')
ops=$((2 * M * N * K * REPS * 4))
agg=$(awk -v o="$ops" -v w="$wall" 'BEGIN{printf "%.2f", o/w/1e9}')
echo "4× independent wall aggregate: ${agg} GOP/s (${wall}s wall, ${REPS} reps each, ${M}x${N}x${K})"
for c in 0 1 2 3; do
  grep 'ime-peak\|ime ' /tmp/ime-mc-p$$-$c.log | tail -1 || grep ime /tmp/ime-mc-p$$-$c.log | tail -1
done
rm -f /tmp/ime-mc-p$$-*.log

echo "DONE $(date -Iseconds)"
