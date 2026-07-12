#!/bin/bash
# run-hpl-ab.sh - HPL scalar-vs-RVV A/B via FlexiBLAS backend swap (no HPL rebuild).
#
# The same xhpl binary is run twice, changing only which BLAS backend FlexiBLAS
# loads: a forced-scalar OpenBLAS vs a vector (RVV) OpenBLAS .so. This isolates
# the BLAS backend's effect on an end-to-end Linpack run.
#
# Usage:  RVV_LIB=/path/to/vector/libopenblas.so ./run-hpl-ab.sh [config.dat]
# Env:    NP       number of MPI ranks (default 8)
#         RVV_LIB  vector OpenBLAS .so; if unset, only the scalar run is done
# Requires a loaded HPL module providing xhpl + mpirun + FlexiBLAS, e.g.:
#   module load HPL/2.3-foss-2025b
#
# xhpl reads ./HPL.dat, so the chosen config is copied there first.
set -u
DAT=${1:-HPL.dat}
NP=${NP:-8}
RVV_LIB=${RVV_LIB:-}
command -v xhpl   >/dev/null || { echo "ERROR: xhpl not on PATH (module load HPL/...)"; exit 1; }
command -v mpirun >/dev/null || { echo "ERROR: mpirun not on PATH"; exit 1; }
[ -f "$DAT" ] || { echo "ERROR: config '$DAT' not found"; exit 1; }
[ "$DAT" = "HPL.dat" ] || cp -f "$DAT" HPL.dat
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "=== [scalar] OPENBLAS_CORETYPE=RISCV64_GENERIC | np=$NP | $DAT ==="
OPENBLAS_CORETYPE=RISCV64_GENERIC mpirun -np "$NP" xhpl | grep -E '^W|PASSED|FAILED'

if [ -n "$RVV_LIB" ]; then
  echo "=== [RVV] FLEXIBLAS=$RVV_LIB | np=$NP | $DAT ==="
  FLEXIBLAS="$RVV_LIB" mpirun -np "$NP" xhpl | grep -E '^W|PASSED|FAILED'
else
  echo "(set RVV_LIB=/path/to/vector/libopenblas.so to also run the vector backend)"
fi
