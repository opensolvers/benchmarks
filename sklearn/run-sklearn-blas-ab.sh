#!/bin/bash
# scikit-learn FlexiBLAS A/B on Orange Pi RV2 (EESSI riscv 20240402 / gfbf-2023b).
#
# Usage: bash run-sklearn-blas-ab.sh
set +e
LOG="${LOG:-$HOME/logs/sklearn-blas-ab-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "START $(date -Iseconds) log=$LOG"

export LMOD_IGNORE_CACHE=yes
export LMOD_CACHED_LOADS=no
# shellcheck disable=SC1091
source /cvmfs/riscv.eessi.io/versions/20240402/init/bash
module --ignore_cache load scikit-learn/1.4.0-gfbf-2023b
echo "python=$(command -v python3)"
echo "EBROOTSCIKITLEARN=${EBROOTSCIKITLEARN:-}"
echo "EBROOTFLEXIBLAS=$EBROOTFLEXIBLAS"
python3 -c "import sklearn,numpy; print('sklearn',sklearn.__version__,'numpy',numpy.__version__)"

HERE=$(cd "$(dirname "$0")" && pwd)
PY=${SKLEARN_BENCH:-$HERE/bench_sklearn.py}
if [[ ! -f "$PY" ]]; then
  PY=$HOME/sklearn-bench/bench_sklearn.py
fi
N=${SK_N:-8000}
D=${SK_D:-512}
K=${SK_K:-64}
THREADS=${THREADS:-8}
RVV_LIB=${RVV_LIB:-$HOME/libopenblas_x60_eb_fixed.so}

export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

run_one() {
  local tag=$1
  shift
  echo ""
  echo "========== [$tag] N=$N D=$D K=$K thr=$THREADS env: $* =========="
  local t0 t1 rc
  t0=$(date +%s.%N)
  env "$@" python3 "$PY" "$N" "$D" "$K"
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
