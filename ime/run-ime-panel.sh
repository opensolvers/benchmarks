#!/usr/bin/env bash
# Step-2 panel tuning: nc sweep, megakernel touches, offline-B.
set -euo pipefail
LOG="${LOG:-$HOME/logs/ime-panel-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1
echo "START $(date -Iseconds) log=$LOG"

PANEL=${PANEL:-$HOME/ime-bench/ime-bench-panel}
BENCH=${BENCH:-$HOME/ime-bench/ime-bench}
REPS=${REPS:-25}

[[ -x "$PANEL" ]] || { echo "missing $PANEL — run: cd ~/ime-bench && make board-panel"; exit 1; }

echo ""
echo "==== panel sweep @768³ anti-alias (TCM via group tcm) ===="
sg tcm -c "taskset -c 0 \"$PANEL\" 768 768 512 \"$REPS\" 1" 2>/dev/null ||
    sudo taskset -c 0 "$PANEL" 768 768 512 "$REPS" 1

echo ""
echo "==== baseline ime-bench peak ===="
taskset -c 0 "$BENCH" 768 768 512 "$REPS" 1 | grep -E "ime-peak|M="

echo "DONE $(date -Iseconds)"
