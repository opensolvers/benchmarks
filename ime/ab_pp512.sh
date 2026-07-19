#!/usr/bin/env bash
# Isolated interleaved A/B: STOCKEB vs SCALEOPT libggml-cpu.so, pinned to cluster-0
# (cores 0-3), performance governor. Each round runs BOTH variants back-to-back so
# any slow thermal/freq drift cancels in the per-round paired delta. Emits raw
# per-round median GOP/s for both variants to a TSV for offline paired analysis.
set -u
cd ~/x60-ime
LIB=libggml-cpu.so.0.16.0
STOCK=libggml-cpu.so.0.16.0.STOCKEB
SCALE=libggml-cpu.so.0.16.0.SCALEOPT
M=${M:-512}; N=${N:-512}; K=${K:-512}
ROUNDS=${ROUNDS:-30}
INNER=${INNER:-25}          # reps inside bench_i8i4 -> it reports min/median/max
PIN="taskset -c 0-3"
OUT=~/ab_pp512.tsv
echo -e "round\tstock_med\tscale_med" > "$OUT"

# median GOP/s = 2nd of the 3 gemm values in the bench summary line
med() { LD_LIBRARY_PATH=~/x60-ime $PIN ~/bench_i8i4 "$M" "$N" "$K" "$INNER" 0 0 2>/dev/null \
        | awk '/^i8i4/{split($0,a,"gemm="); split(a[2],b," "); split(b[1],c,"/"); print c[2]}'; }

# warmup (discard) to spin cores to Fmax
cp -f "$STOCK" "$LIB"; med >/dev/null; cp -f "$SCALE" "$LIB"; med >/dev/null

for r in $(seq 1 "$ROUNDS"); do
  cp -f "$STOCK" "$LIB"; s=$(med)
  cp -f "$SCALE" "$LIB"; c=$(med)
  echo -e "${r}\t${s}\t${c}" | tee -a "$OUT"
done
echo "AB_DONE rows=$(($(wc -l < "$OUT")-1))"
