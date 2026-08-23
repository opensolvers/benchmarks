#!/usr/bin/env bash
# Isolated vmadot peak (cpufp-style) + GEMM aliasing A/B (pad vs unpadded).
set -euo pipefail
LOG="${LOG:-$HOME/logs/ime-peak-alias-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1
echo "START $(date -Iseconds) log=$LOG"

PEAK=${PEAK:-$HOME/ime-bench/ime-bench-peak}
BENCH=${BENCH:-$HOME/ime-bench/ime-bench}
REPS=${REPS:-25}

for b in "$PEAK" "$BENCH"; do
  [[ -x "$b" ]] || { echo "missing $b — run: cd ~/ime-bench && make board board-peak"; exit 1; }
done

echo ""
echo "==== (a) cpufp-style vmadot peak — kloop only ===="
for K in 64 128 256; do
  taskset -c 0 "$PEAK" "$K" 500000 0
done

echo ""
echo "==== (a) full 8×16 block (+ store) ===="
taskset -c 0 "$PEAK" 64 200000 1

echo ""
echo "==== (b) GEMM aliasing A/B — unpadded vs anti-alias (ldc=N+16, buf+2048) ===="
for sz in "512 512 512" "768 768 512"; do
  read -r M N K <<< "$sz"
  echo ""
  echo "--- ${M}x${N}x${K} ---"
  echo "unpadded (ldc=N, malloc):"
  taskset -c 0 "$BENCH" "$M" "$N" "$K" "$REPS" 0 | grep ime-peak
  echo "anti-alias (ldc=N+16, GEMM_BUF_PAD):"
  taskset -c 0 "$BENCH" "$M" "$N" "$K" "$REPS" 1 | grep ime-peak
done

echo ""
echo "==== (b) full ime path (5 reps) unpadded vs anti-alias @768³ ===="
taskset -c 0 "$BENCH" 768 768 512 5 0 | grep -E '^ime|^M='
taskset -c 0 "$BENCH" 768 768 512 5 1 | grep -E '^ime|^M='

echo "DONE $(date -Iseconds)"
