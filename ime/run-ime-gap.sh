#!/usr/bin/env bash
# Waterfall: insn peak → kloop → block → pack → compute → full GEMM.
set -euo pipefail
LOG="${LOG:-$HOME/logs/ime-gap-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1
echo "START $(date -Iseconds) log=$LOG"

GAP=${GAP:-$HOME/ime-bench/ime-bench-gap}
[[ -x "$GAP" ]] || { echo "missing $GAP — run: cd ~/ime-bench && make board-gap"; exit 1; }

REPS=${REPS:-25}

echo ""
echo "==== anti-alias (ldc=N+16) — good malloc tier ===="
for sz in "512 512 512" "768 768 512"; do
  read -r M N K <<< "$sz"
  echo ""
  taskset -c 0 "$GAP" "$M" "$N" "$K" "$REPS" 1
done

echo ""
echo "==== unpadded malloc (ldc=N) — aliasing lottery ===="
taskset -c 0 "$GAP" 768 768 512 "$REPS" 0

echo "DONE $(date -Iseconds)"
