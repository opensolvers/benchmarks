#!/bin/bash
# FFT-axis A/B for ScaFaCoS P3M: swap libfftw3 via LD_PRELOAD.
set +e
LOG="${LOG:-$HOME/logs/scafacos-p3m-ab-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")" "$HOME/scafacos-bench"
exec >>"$LOG" 2>&1
echo "START $(date -Iseconds) log=$LOG"

export EESSI_VERSION_OVERRIDE=2025.06-001
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load ScaFaCoS/1.0.4-foss-2025b

HERE=${HERE:-$HOME/scafacos}
BENCH=$HOME/scafacos-bench
cp -f "$HERE/scafacos_bench.c" "$HERE/armci_stubs.c" "$HERE/Makefile" "$BENCH/"
cd "$BENCH"
make -B scafacos_bench
echo "binary=$(pwd)/scafacos_bench"

SCALAR=${SCALAR:-$HOME/fftwbuild/src-scalar/.libs/libfftw3.so.3.6.10}
R5V=${R5V:-$HOME/fftwbuild/src-r5v/.libs/libfftw3.so.3.6.10}
NP=${NP:-4}
NSIDE=${NSIDE:-24}
REPS=${REPS:-10}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENBLAS_CORETYPE=RISCV64_GENERIC
export LD_LIBRARY_PATH=$EBROOTFFTW/lib:$EBROOTGSL/lib:$LD_LIBRARY_PATH

run_one() {
  local tag=$1 lib=$2
  echo ""
  echo "========== [$tag] LD_PRELOAD=$(basename "$lib") method=p3m np=$NP n_side=$NSIDE reps=$REPS =========="
  LD_PRELOAD="$lib" mpirun --bind-to core -np "$NP" ./scafacos_bench p3m "$NSIDE" "$REPS"
  echo "RC=$?"
}

run_one scalar "$SCALAR"
run_one r5v "$R5V"
echo "DONE $(date -Iseconds)"
