#!/bin/bash
# Serial benchmark AFTER all downloads complete (avoids IO/RAM contention).
# HARDENED: force-killed timeouts (-k), -r 1, short coherence gen, per-model log.
BIN=$HOME/eessi-x60/versions/2025.06-001/software/linux/riscv64/generic/software/llama.cpp/ad8d821-foss-2025b-x60-ime-m1gemv/bin
M=$HOME/llama-x60/models
RES=$HOME/llama-x60/results.md
MAN=$M/dl_manifest.txt
DLLOG=$HOME/llama-x60/dl_suite.log
THREADS=8
PROMPT="Give a short factual answer. Q: What is the capital of France? A:"

# 1) wait until downloader signals completion (marker printed at end of dl_suite.sh)
echo ">>> waiting for ALL-DOWNLOADS-COMPLETE ..."
while ! grep -q "ALL-DOWNLOADS-COMPLETE" "$DLLOG" 2>/dev/null; do sleep 20; done
echo ">>> downloads finished, starting serial benchmark"
sync; sleep 3

echo "# Suite benchmark results (m1gemv build, IME1) — $(date -u)" > "$RES"
echo "" >> "$RES"
echo "| label | GB | pp512 t/s | tg64 t/s | coherence |" >> "$RES"
echo "|-------|----|-----------|----------|-----------|" >> "$RES"

# NOTE: read manifest on fd 3 so llama-* child processes cannot drain the loop's stdin
while IFS='|' read -r -u 3 label repo fn; do
  [ -z "$label" ] && continue
  f="$M/$label.gguf"
  if [ ! -s "$f" ]; then echo "| $label | - | DL-FAIL | - | - |" >> "$RES"; echo "MISS $label"; continue; fi
  gb=$(awk "BEGIN{printf \"%.2f\", $(stat -c %s "$f")/1073741824}")
  echo ">>> benchmarking $label ($gb GB)  $(date -u +%H:%M:%S)"
  # llama-bench: single repeat; hard-kill (-k 30 SIGKILL) so a slow model cannot wedge the suite
  bench=$(timeout -k 30 -s KILL 600 "$BIN/llama-bench" -m "$f" -p 512 -n 64 -t $THREADS -r 1 2>/dev/null)
  # t/s is the LAST float on the row (col before the "± stddev"); size (373.71 MiB) is the first
  pp=$(echo "$bench" | grep -E 'pp512' | grep -oE '[0-9]+\.[0-9]+' | tail -2 | head -1)
  tg=$(echo "$bench" | grep -E 'tg64'  | grep -oE '[0-9]+\.[0-9]+' | tail -2 | head -1)
  # coherence: short gen (8 tokens is enough to emit "Paris"); hard-killed at 120s
  raw=$(timeout -k 15 -s KILL 120 "$BIN/llama-cli" -m "$f" -p "$PROMPT" -n 8 -no-cnv -t $THREADS 2>/dev/null | tr '\n' ' ')
  echo "$raw" | grep -qiE 'paris' && coh="OK(Paris)" || coh="?"
  echo "| $label | $gb | ${pp:-?} | ${tg:-?} | $coh |" >> "$RES"
  echo "RESULT $label pp=${pp:-?} tg=${tg:-?} coh=$coh  $(date -u +%H:%M:%S)"
done 3< "$MAN"
echo "ALL-BENCH-COMPLETE"
