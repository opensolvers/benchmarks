#!/bin/bash
# Planner-aware A/B of the r5v (RVV) vs scalar FFTW libs built by
# build-fftw-r5v.sh. For each size x planner x lib, run FFTW's own tests/bench
# (1D complex-to-complex) and record the MEDIAN of the 4 reported MFLOPS values.
#
# Planner is chosen via bench's -o<word> (see tests/fftw-bench.c): valid words
# are estimate|patient|estimatepat|exhaustive. The DEFAULT (no -o) is
# FFTW_MEASURE -- there is NO -omeasure (it prints "unknown user option:
# measure. Ignoring." and silently runs the MEASURE default). We therefore use
# planner="default" to mean FFTW_MEASURE.
#
# Use -t 1.0 (>=1 s): short timings (<=0.3 s) are noisy on the X60 and can
# invert individual points. NOTE: patient planning is pathological at large N
# (>35 min at N=262144); drop it from SIZES>=262144 if you value your afternoon.

ROOT=${ROOT:-$HOME/fftwbuild}
LOG=${LOG:-$HOME/fftw-proper.log}
: > "$LOG"

# median of a whitespace-separated list of floats
med() { echo "$1" | tr ' ' '\n' | grep -E '^[0-9]' | sort -n | \
        awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }

bench_one() {
  name=$1; dir=$2; planner=$3; s=$4
  cd "$dir/tests" || return 1
  if [ "$planner" = default ]; then          # default == FFTW_MEASURE
    raw=$(LD_LIBRARY_PATH="$dir/.libs" ./bench --report-mflops -t 1.0 -s "$s" 2>&1)
  else
    raw=$(LD_LIBRARY_PATH="$dir/.libs" ./bench --report-mflops -o"$planner" -t 1.0 -s "$s" 2>&1)
  fi
  vals=$(echo "$raw" | tr -d '()' | grep -oE '[0-9]+\.[0-9]+' | tr '\n' ' ')
  echo "[$name/$planner] size=$s med=$(med "$vals") | raw=$vals" >> "$LOG"
}

SIZES=${SIZES:-'256 1024 4096 16384 65536 262144 1048576'}
echo "=== FFTW r5v vs scalar A/B start $(date -u +%FT%TZ) ===" >> "$LOG"
for s in $SIZES; do
  for p in estimate default patient; do
    bench_one r5v    "$ROOT/src-r5v"    "$p" "$s"
    bench_one scalar "$ROOT/src-scalar" "$p" "$s"
  done
  echo "--- size $s complete $(date -u +%TZ) ---" >> "$LOG"
done
echo '=== DONE ===' >> "$LOG"
