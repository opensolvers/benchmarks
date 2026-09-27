#!/bin/bash
set -eo pipefail
source "$HOME/.eessi_init"
module load foss/2025b HPL/2.3-foss-2025b
HPLDIR="$HOME/hpl-baseline"
cd "$HPLDIR"
export LD_LIBRARY_PATH="$HOME/openblas-dual/lib:$HOME/openblas-zvl1024b/lib:${LD_LIBRARY_PATH:-}"
export HPL_X100_BLAS=OPENBLAS_DUAL
export HPL_A100_BLAS=OPENBLAS_ZVL1024B
export FLEXIBLAS=OPENBLAS_DUAL
export OMP_NUM_THREADS=1
LOG="$HPLDIR/hpl-hetero-zvl1024b.log"

run_hetero() {
  local LABEL=$1 N=$2 NB=$3 NX=$4 NA=$5 ATH=$6 P=$7 Q=$8
  local NP=$((NX + NA))
  export HPL_NX100=$NX HPL_A100_THREADS=$ATH
  cat > "$HPLDIR/HPL.dat" <<EOF
HPLinpack benchmark input file
OpenSolvers k3 hetero ${LABEL}
HPL.out
6
1
${N}
1
${NB}
0
1
${P}
${Q}
16.0
1
1
1
4
1
2
1
1
1
0
1
1
2
64
0
0
1
8
EOF
  local OUT="$HPLDIR/xhpl-hetero-${LABEL}-N${N}.out"
  echo "=== hetero ${LABEL} N=${N} NX=${NX} NA=${NA} ATH=${ATH} grid=${P}x${Q} $(date -Is) ===" | tee "$OUT" | tee -a "$LOG"
  mpirun -np "$NP" --bind-to none \
    -x FLEXIBLAS -x OMP_NUM_THREADS -x LD_LIBRARY_PATH \
    -x HPL_NX100 -x HPL_A100_THREADS -x HPL_X100_BLAS -x HPL_A100_BLAS \
    "$HPLDIR/hpl-hetero-rank.sh" xhpl >>"$OUT" 2>&1 || true
  tee -a "$LOG" < "$OUT" >/dev/null
  grep -aE "WR[0-9]|PASSED|FAILED|TRUNCATE" "$OUT" | tee -a "$LOG" || true
}

# remaining + ref
run_hetero "8plus8x1" 12000 192 8 8 1 4 4
run_hetero "8plus4x2" 12000 192 8 4 2 3 4
run_hetero "x100only" 12000 192 8 0 1 2 4

echo "ALL DONE hetero $(date -Is)" | tee -a "$LOG"
