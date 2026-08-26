#!/bin/bash
# run-hpl-034.sh - HPL with OpenBLAS 0.3.34 ZVL256B via FlexiBLAS (no rebuild).
#
# Usage:  ./run-hpl-034.sh [config.dat ...]
# Env:    NP           MPI ranks (default 8)
#         OB034_LIB    OpenBLAS 0.3.34 .so (default ~/ob-0.3.34/libopenblas_riscv64_zvl256bp-r0.3.34.so)
#         ALSO_PATCHED also run patched 0.3.30 for comparison (default 1)
#         PATCHED_LIB  patched 0.3.30 .so (default ~/libopenblas_x60_eb_fixed.so)
set -u
NP=${NP:-8}
OB034_LIB=${OB034_LIB:-$HOME/ob-0.3.34/libopenblas_riscv64_zvl256bp-r0.3.34.so}
PATCHED_LIB=${PATCHED_LIB:-$HOME/libopenblas_x60_eb_fixed.so}
ALSO_PATCHED=${ALSO_PATCHED:-1}
LOGDIR=${LOGDIR:-$HOME/logs}
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d-%H%M%S)
LOG="$LOGDIR/hpl-034-$STAMP.log"

set +u
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module use /cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/modules/all
module load HPL/2.3-foss-2025b
set -u

command -v xhpl   >/dev/null || { echo "ERROR: xhpl not on PATH"; exit 1; }
command -v mpirun >/dev/null || { echo "ERROR: mpirun not on PATH"; exit 1; }
[ -f "$OB034_LIB" ] || { echo "ERROR: 0.3.34 lib missing: $OB034_LIB"; exit 1; }

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

run_one() {
  local tag=$1 lib=$2
  echo "=== [$tag] FLEXIBLAS=$lib | np=$NP | $DAT ==="
  FLEXIBLAS="$lib" mpirun -np "$NP" xhpl | tee -a "$LOG.tmp" | grep -E '^W|PASSED|FAILED|Ax-b'
}

{
  echo "=== HPL OpenBLAS 0.3.34 | $STAMP | host=$(hostname) ==="
  echo "OB034_LIB=$OB034_LIB"
  echo "PATCHED_LIB=$PATCHED_LIB"
  echo "xhpl=$(command -v xhpl)"
  echo

  if [ "$#" -eq 0 ]; then
    set -- HPL.dat HPL-sweep.dat
  fi

  for DAT in "$@"; do
    [ -f "$DAT" ] || { echo "ERROR: config '$DAT' not found"; exit 1; }
    [ "$DAT" = "HPL.dat" ] || cp -f "$DAT" HPL.dat
    echo
    echo "######## $DAT ########"
    run_one "0.3.34" "$OB034_LIB"
    if [ "$ALSO_PATCHED" = "1" ] && [ -f "$PATCHED_LIB" ]; then
      run_one "patched-0.3.30" "$PATCHED_LIB"
    fi
  done
} 2>&1 | tee "$LOG"

# Keep a full xhpl capture if we wrote one
[ -f "$LOG.tmp" ] && cat "$LOG.tmp" >> "$LOG" && rm -f "$LOG.tmp"
echo "Log: $LOG"
