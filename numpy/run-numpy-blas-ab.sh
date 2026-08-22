#!/bin/bash
# NumPy FlexiBLAS A/B on Orange Pi RV2 (EESSI dev riscv 2025.06 / foss-2025b).
#
# Usage: bash run-numpy-blas-ab.sh
set +e
LOG="${LOG:-$HOME/logs/numpy-blas-ab-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "START $(date -Iseconds) log=$LOG"

export LMOD_IGNORE_CACHE=yes
export LMOD_CACHED_LOADS=no
export EESSI_VERSION_OVERRIDE=2025.06-001
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module --ignore_cache load SciPy-bundle/2025.07-gfbf-2025b
echo "python=$(command -v python3)"
echo "EBROOTFLEXIBLAS=$EBROOTFLEXIBLAS"
python3 -c "import numpy; print('numpy', numpy.__version__)"

HERE=$(cd "$(dirname "$0")" && pwd)
PY=${NUMPY_BENCH:-$HERE/bench_blas.py}
if [[ ! -f "$PY" ]]; then
  PY=$HOME/numpy-bench/bench_blas.py
fi
DGEMM_N=${DGEMM_N:-4096}
EIG_N=${EIG_N:-2048}
THREADS=${THREADS:-8}
RVV_LIB=${RVV_LIB:-$HOME/libopenblas_x60_eb_fixed.so}

export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

run_one() {
  local tag=$1
  shift
  echo ""
  echo "========== [$tag] dgemm_N=$DGEMM_N eigh_N=$EIG_N thr=$THREADS env: $* =========="
  local t0 t1 rc
  t0=$(date +%s.%N)
  env "$@" python3 "$PY" "$DGEMM_N" "$EIG_N"
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
