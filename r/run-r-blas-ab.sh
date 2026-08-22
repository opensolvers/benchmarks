#!/bin/bash
# R FlexiBLAS A/B on Orange Pi RV2 (EESSI riscv 20240402 / gfbf-2023b).
#
# Usage: bash run-r-blas-ab.sh
set +e
LOG="${LOG:-$HOME/logs/r-blas-ab-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "START $(date -Iseconds) log=$LOG"

export LMOD_IGNORE_CACHE=yes
export LMOD_CACHED_LOADS=no
# shellcheck disable=SC1091
source /cvmfs/riscv.eessi.io/versions/20240402/init/bash
module --ignore_cache load R/4.4.1-gfbf-2023b
echo "R=$(command -v Rscript)"
echo "EBROOTR=$EBROOTR"
echo "EBROOTFLEXIBLAS=$EBROOTFLEXIBLAS"

HERE=$(cd "$(dirname "$0")" && pwd)
RBENCH=${R_BENCH:-$HERE/bench_r.R}
if [[ ! -f "$RBENCH" ]]; then
  RBENCH=$HOME/r-bench/bench_r.R
fi
DGEMM_N=${DGEMM_N:-2048}
EIG_N=${EIG_N:-1024}
THREADS=${THREADS:-8}
RVV_LIB=${RVV_LIB:-$HOME/libopenblas_x60_eb_fixed.so}

export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

run_one() {
  local tag=$1
  shift
  echo ""
  echo "========== [$tag] N_dgemm=$DGEMM_N N_eig=$EIG_N thr=$THREADS env: $* =========="
  local t0 t1 rc
  t0=$(date +%s.%N)
  env "$@" Rscript "$RBENCH" "$DGEMM_N" "$EIG_N"
  rc=$?
  t1=$(date +%s.%N)
  python3 -c "print('WALL_S=%.3f RC=%d tag=%s' % ($t1-$t0, $rc, '$tag'))"
}

run_one scalar FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC
run_one stock  FLEXIBLAS=OPENBLAS
if [[ -f "$RVV_LIB" ]]; then
  run_one patched FLEXIBLAS="$RVV_LIB"
else
  echo "SKIP patched: no $RVV_LIB"
fi

echo "DONE $(date -Iseconds)"
