#!/bin/bash
# run-hpl-blis.sh - run the BLIS-linked xhpl end-to-end and report GFLOP/s +
# PASSED/residual. This is the BLIS analogue of run-hpl-ab.sh, but there is NO
# FlexiBLAS swap here: the BLAS provider (BLIS) is baked into this xhpl at link
# time (see build-hpl-blis.sh), so we just run that dedicated binary.
#
# Usage:  ./run-hpl-blis.sh [config.dat]
# Env:    NP        MPI ranks (default 8 = one per X60 core)
#         XHPL      path to the BLIS-linked xhpl (default $HOME/hpl-2.3/bin/rv64_blis/xhpl)
#         BLIS_THREADS  BLIS threads per rank (default 1 - pure-MPI, avoids
#                       oversubscribing 8 cores with 8 ranks x N threads)
#
# xhpl reads ./HPL.dat, so the chosen config is copied there first.

GCC14=/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software/GCCcore/14.3.0
export LD_LIBRARY_PATH="$GCC14/lib64:$GCC14/lib:$LD_LIBRARY_PATH"

set -u
DAT=${1:-HPL.dat}
NP=${NP:-8}
XHPL=${XHPL:-$HOME/hpl-2.3/bin/rv64_blis/xhpl}
BLIS_THREADS=${BLIS_THREADS:-1}

[ -x "$XHPL" ]     || { echo "ERROR: xhpl not found/executable: $XHPL (run ./build-hpl-blis.sh)"; exit 1; }
command -v mpirun >/dev/null || { echo "ERROR: mpirun not on PATH (module load OpenMPI / foss)"; exit 1; }
[ -f "$DAT" ]      || { echo "ERROR: config '$DAT' not found"; exit 1; }
[ "$DAT" = "HPL.dat" ] || cp -f "$DAT" HPL.dat

# pure-MPI: one BLAS thread per rank so 8 ranks fill 8 cores exactly
export OMP_NUM_THREADS="$BLIS_THREADS" BLIS_NUM_THREADS="$BLIS_THREADS" OPENBLAS_NUM_THREADS="$BLIS_THREADS"

echo "=== [BLIS] xhpl=$XHPL | np=$NP | blis_threads=$BLIS_THREADS | $DAT ==="
mpirun -np "$NP" "$XHPL" | grep -E '^W|PASSED|FAILED|Ax-b'
