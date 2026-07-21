#!/bin/bash
# RVV control-arm benchmark: same 17 models as the m1gemv IME run, plain RVV build
# (no smt.vmadot). Mirrors bench_suite.sh hardening: fd-3 manifest read, force-killed
# timeouts, -r 1, short coherence gen. Writes results_rvv.md (does NOT touch results.md).
RVV=$HOME/x60-rvv
M=$HOME/llama-x60/models
RES=$HOME/llama-x60/results_rvv.md
MAN=$M/dl_manifest.txt
THREADS=8
PROMPT="Give a short factual answer. Q: What is the capital of France? A:"
export LD_LIBRARY_PATH="$RVV"

echo "# Suite benchmark results (RVV control build, no-IME) — $(date -u)" > "$RES"
echo "" >> "$RES"
echo "| label | GB | pp512 t/s | tg64 t/s | coherence |" >> "$RES"
echo "|-------|----|-----------|----------|-----------|" >> "$RES"

# manifest on fd 3 so llama-* child procs cannot drain the loop's stdin
while IFS='|' read -r -u 3 label repo fn; do
  [ -z "$label" ] && continue
  f="$M/$label.gguf"
  if [ ! -s "$f" ]; then echo "| $label | - | DL-FAIL | - | - |" >> "$RES"; echo "MISS $label"; continue; fi
  gb=$(awk "BEGIN{printf \"%.2f\", $(stat -c %s "$f")/1073741824}")
  echo ">>> RVV benchmarking $label ($gb GB)  $(date -u +%H:%M:%S)"
  bench=$(timeout -k 30 -s KILL 600 "$RVV/llama-bench" -m "$f" -p 512 -n 64 -t $THREADS -r 1 2>/dev/null)
  # t/s is second-to-last float on the row (last is ± stddev); size col is first float
  pp=$(echo "$bench" | grep -E 'pp512' | grep -oE '[0-9]+\.[0-9]+' | tail -2 | head -1)
  tg=$(echo "$bench" | grep -E 'tg64'  | grep -oE '[0-9]+\.[0-9]+' | tail -2 | head -1)
  raw=$(timeout -k 15 -s KILL 120 "$RVV/llama-cli" -m "$f" -p "$PROMPT" -n 8 -no-cnv -t $THREADS 2>/dev/null | tr '\n' ' ')
  echo "$raw" | grep -qiE 'paris' && coh="OK(Paris)" || coh="?"
  echo "| $label | $gb | ${pp:-?} | ${tg:-?} | $coh |" >> "$RES"
  echo "RVV-RESULT $label pp=${pp:-?} tg=${tg:-?} coh=$coh  $(date -u +%H:%M:%S)"
done 3< "$MAN"
echo "ALL-RVV-BENCH-COMPLETE"
