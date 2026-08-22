#!/bin/bash
# Armadillo FlexiBLAS A/B on Orange Pi RV2 (EESSI riscv 20240402 / foss-2023b).
#
# Usage: bash run-armadillo-blas-ab.sh
# Optional: DGEMM_N=2048 EIG_N=1024 THREADS=8
set +e
LOG="${LOG:-$HOME/logs/armadillo-blas-ab-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "START $(date -Iseconds) log=$LOG"

export LMOD_IGNORE_CACHE=yes
export LMOD_CACHED_LOADS=no
# shellcheck disable=SC1091
source /cvmfs/riscv.eessi.io/versions/20240402/init/bash
module --ignore_cache load Armadillo/12.8.0-foss-2023b
echo "g++=$(command -v g++)"
echo "EBROOTARMADILLO=$EBROOTARMADILLO"
echo "EBROOTFLEXIBLAS=$EBROOTFLEXIBLAS"
export LD_LIBRARY_PATH="${EBROOTARMADILLO}/lib64:${EBROOTARMADILLO}/lib:${LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${ARMA_SRC:-$HERE/bench_arma.cpp}
# allow board-local copy
if [[ ! -f "$SRC" ]]; then
  SRC=$HOME/arma-bench/bench_arma.cpp
fi
BIN=${ARMA_BIN:-$HOME/arma-bench/bench_arma}
DGEMM_N=${DGEMM_N:-2048}
EIG_N=${EIG_N:-1024}
THREADS=${THREADS:-8}
RVV_LIB=${RVV_LIB:-$HOME/libopenblas_x60_eb_fixed.so}

mkdir -p "$(dirname "$BIN")"
cp -f "$SRC" "$(dirname "$BIN")/bench_arma.cpp"
echo "Building $BIN from $SRC"
g++ -O2 -std=c++17 "$(dirname "$BIN")/bench_arma.cpp" -o "$BIN" \
  -Wl,-rpath,"${EBROOTARMADILLO}/lib64" -Wl,-rpath,"${EBROOTARMADILLO}/lib" \
  -larmadillo
brc=$?
echo "build_rc=$brc"
if [[ $brc -ne 0 || ! -x "$BIN" ]]; then
  echo "FATAL: build failed"
  exit 1
fi
ldd "$BIN" | grep -iE "arma|blas|flexi" || true
"$BIN" 64 32 | head -5 || true

export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

run_one() {
  local tag=$1
  shift
  echo ""
  echo "========== [$tag] N_dgemm=$DGEMM_N N_eig=$EIG_N thr=$THREADS env: $* =========="
  local t0 t1 rc
  t0=$(date +%s.%N)
  env "$@" "$BIN" "$DGEMM_N" "$EIG_N"
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
