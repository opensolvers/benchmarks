#!/bin/bash
# MetalWalls A/B on Orange Pi RV2 (EESSI riscv 20240402 / foss-2023b):
#   1) FFT axis: LD_PRELOAD scalar vs r5v libfftw3 (BLAS pinned scalar)
#   2) BLAS axis: OPENBLAS_CORETYPE=RISCV64_GENERIC vs stock OpenBLAS
#      (optional FLEXIBLAS=$RVV_LIB if a compatible patched .so is set)
#
# Workload: tip4p-water example (bulk TIP4P; Ewald LR ~10–15% of wall).
# Electrode matrix_inversion / CG examples are too heavy for a short A/B at
# stock sizes (~3k electrode atoms).
#
# Usage: MW_STEPS=200 bash run-metalwalls-ab.sh
set +e
LOG="${LOG:-$HOME/logs/metalwalls-ab-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "START $(date -Iseconds) log=$LOG"

export LMOD_IGNORE_CACHE=yes
export LMOD_CACHED_LOADS=no
# shellcheck disable=SC1091
source /cvmfs/riscv.eessi.io/versions/20240402/init/bash
module --ignore_cache load MetalWalls/21.06.1-foss-2023b
echo "mw=$(command -v mw)"
echo "EBROOTMETALWALLS=$EBROOTMETALWALLS"
echo "EBROOTFLEXIBLAS=$EBROOTFLEXIBLAS"
echo "EBROOTOPENBLAS=$EBROOTOPENBLAS"
echo "EBROOTFFTW=${EBROOTFFTW:-}"

HERE=$(cd "$(dirname "$0")" && pwd)
EX=${MW_EXAMPLE:-$HOME/mw-bench/metalwalls/example/tip4p-water}
WORK=${WORK:-$HOME/mw-bench/ab-tip4p}
STEPS=${MW_STEPS:-200}
NP=${MW_NP:-1}
SCALAR_FFTW=${SCALAR_FFTW:-$HOME/fftwbuild/src-scalar/.libs/libfftw3.so.3.6.10}
R5V_FFTW=${R5V_FFTW:-$HOME/fftwbuild/src-r5v/.libs/libfftw3.so.3.6.10}
RVV_LIB=${RVV_LIB:-$HOME/libopenblas_x60_eb_fixed.so}

if [[ ! -f "$EX/runtime.inpt" ]]; then
  echo "FATAL: example not found at $EX (clone gitlab.com/ampere2/metalwalls example/tip4p-water)"
  exit 1
fi

mkdir -p "$WORK"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

prep_case() {
  local tag=$1
  local dir="$WORK/run-$tag"
  rm -rf "$dir"
  mkdir -p "$dir"
  cp "$EX"/* "$dir"/
  # strip trajectories / quiet-ish output for short benches
  sed -i "s/num_steps.*/num_steps       $STEPS/" "$dir/runtime.inpt"
  # disable frequent traj if present
  sed -i "s/^ *step .*/  step $STEPS/" "$dir/runtime.inpt" || true
  echo "$dir"
}

run_one() {
  local tag=$1
  shift
  local dir
  dir=$(prep_case "$tag")
  echo ""
  echo "========== [$tag] STEPS=$STEPS NP=$NP env: $* =========="
  echo "dir=$dir"
  local t0 t1 rc
  t0=$(date +%s.%N)
  (
    cd "$dir" || exit 99
    # shellcheck disable=SC2086
    env "$@" mpirun -np "$NP" mw >mw.stdout 2>mw.stderr
  )
  rc=$?
  t1=$(date +%s.%N)
  python3 -c "print('WALL_S=%.3f RC=%d tag=%s' % ($t1-$t0, $rc, '$tag'))"
  if [[ -f "$dir/run.out" ]]; then
    grep -E "Total elapsed time:|Ions Coulomb forces|long range|vdW  forces|Rattle" "$dir/run.out" | head -20
    # checksum: last temperature line if present
    if [[ -f "$dir/temperature.out" ]]; then
      echo -n "TEMP_LAST="; tail -1 "$dir/temperature.out"
    fi
  fi
  grep -iE "error|Error|NaN|FATAL" "$dir/mw.stderr" "$dir/run.out" 2>/dev/null | head -10
  echo "RC=$rc"
}

echo ""
echo "######## FFT A/B (BLAS pinned scalar) ########"
if [[ -f "$SCALAR_FFTW" && -f "$R5V_FFTW" ]]; then
  run_one fft_scalar \
    FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC \
    LD_PRELOAD="$SCALAR_FFTW"
  run_one fft_r5v \
    FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC \
    LD_PRELOAD="$R5V_FFTW"
else
  echo "SKIP FFT A/B: missing $SCALAR_FFTW or $R5V_FFTW"
fi

echo ""
echo "######## BLAS A/B (stock FFTW) ########"
run_one blas_scalar FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC
run_one blas_stock  FLEXIBLAS=OPENBLAS
# stock may equal scalar if OpenBLAS has no RVV path; still records the pair
if [[ -f "$RVV_LIB" ]]; then
  run_one blas_patched FLEXIBLAS="$RVV_LIB"
else
  echo "SKIP blas_patched: RVV_LIB missing"
fi

echo "DONE $(date -Iseconds)"
