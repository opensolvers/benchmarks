#!/bin/bash
# FFT-axis A/B for QE pw.x: hold BLAS constant (FlexiBLAS scalar OpenBLAS),
# swap ONLY libfftw3.so.3 via LD_PRELOAD (scalar vs r5v/RVV). Serial (np=1).
# Captures QE's own WALL timers for fftw / vloc_psi / PWSCF total + total energy.
#
# Usage: bash run-qe-fft-ab.sh <input.in> <label>
IN=${1:-si-super-64.in}
LABEL=${2:-run}
BENCH=$HOME/qe-bench
PW=$HOME/qe-serial/pw.x

SCALAR=$HOME/fftwbuild/src-scalar/.libs/libfftw3.so.3.6.10
R5V=$HOME/fftwbuild/src-r5v/.libs/libfftw3.so.3.6.10

source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash >/dev/null 2>&1
module use /cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/modules/all >/dev/null 2>&1
module load GCC/14.3.0 FFTW/3.3.10-GCC-14.3.0 FlexiBLAS/3.4.5-GCC-14.3.0 >/dev/null 2>&1
set -eo pipefail
export LD_LIBRARY_PATH=$EBROOTFFTW/lib:$EBROOTFLEXIBLAS/lib:$EBROOTOPENBLAS/lib:$EBROOTGCCCORE/lib64:$LD_LIBRARY_PATH
# pin BLAS to scalar so only the FFT axis varies
export FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC OMP_NUM_THREADS=1

cd "$BENCH"
test -f "$SCALAR" || { echo "FATAL: scalar fftw missing: $SCALAR"; exit 1; }
test -f "$R5V"    || { echo "FATAL: r5v fftw missing: $R5V"; exit 1; }

run_one() {
  local tag=$1 lib=$2 out=$3
  echo "===== [$tag] LD_PRELOAD=$(basename "$lib") ====="
  rm -rf ./out
  env LD_PRELOAD="$lib" "$PW" -in "$IN" > "$out" 2> "$out.err" || {
    echo "  RUN FAILED ($tag)"; tail -20 "$out" "$out.err"; return 1; }
  # which fftw actually served the run
  echo -n "  loaded libfftw3: "
  grep -c . "$out" >/dev/null && echo "(size $(stat -c%s "$lib") bytes)"
  local etot fftw vloc pwscf
  etot=$(grep -E '! *total energy' "$out" | tail -1 | awk '{print $(NF-1)}')
  fftw=$(grep -E '^ *fftw ' "$out" | tail -1 | grep -oE '[0-9.]+s WALL' | head -1)
  vloc=$(grep -E '^ *vloc_psi ' "$out" | tail -1 | grep -oE '[0-9.]+s WALL' | head -1)
  pwscf=$(grep -E '^ *PWSCF ' "$out" | tail -1 | grep -oE '[0-9.]+s WALL' | head -1)
  echo "  total energy = ${etot:-?} Ry"
  echo "  fftw      WALL = ${fftw:-?}"
  echo "  vloc_psi  WALL = ${vloc:-?}"
  echo "  PWSCF     WALL = ${pwscf:-?}"
}

echo "#################### QE FFT-axis A/B : $IN ($LABEL) ####################"
echo "pw.x = $PW"
echo "BLAS = FlexiBLAS->OPENBLAS RISCV64_GENERIC (constant)"
echo
run_one scalar "$SCALAR" "$BENCH/qe-fft-${LABEL}-scalar.out"
echo
run_one r5v    "$R5V"    "$BENCH/qe-fft-${LABEL}-r5v.out"
echo
echo "===== full timer tables saved: qe-fft-${LABEL}-{scalar,r5v}.out ====="
