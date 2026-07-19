#!/bin/bash
# run-ab.sh - DGEMM performance A/B: BLIS vs OpenBLAS on the SpaceMiT X60 / K1.
#
# No FlexiBLAS on this board, so the A/B is done by LINKING the same
# backend-agnostic bench_dgemm.c against each library in turn - one variable
# changed (the BLAS implementation), everything else held constant. Both builds
# use the same compiler flags and the same source.
#
# Usage:
#   BLIS_PREFIX=$HOME/blis-install \
#   OPENBLAS_LIB=$HOME/trsm-pr5830/libopenblas.a \
#   ./run-ab.sh
#
# Env:
#   BLIS_PREFIX   BLIS install prefix (expects $BLIS_PREFIX/lib/libblis.a)   [required]
#   OPENBLAS_LIB  path to an OpenBLAS static lib or .so                       [required]
#   SIZES         space-separated N list (default: 512 1024 2048 4096)
#   THREADS       space-separated thread counts (default: 1 8)

# --- EESSI GCC 14.3.0 (force ahead of EESSI compat gcc 13; see ../README.md) ---
GCC14=/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software/GCCcore/14.3.0
export PATH="$GCC14/bin:$PATH"
export LD_LIBRARY_PATH="$GCC14/lib64:$GCC14/lib:$LD_LIBRARY_PATH"

set -euo pipefail

BLIS_PREFIX=${BLIS_PREFIX:-$HOME/blis-install}
OPENBLAS_LIB=${OPENBLAS_LIB:-$HOME/trsm-pr5830/libopenblas.a}
SIZES=${SIZES:-"512 1024 2048 4096"}
THREADS=${THREADS:-"1 8"}
CFLAGS="-O3 -march=rv64imafdcv_zvl256b -fopenmp"
# GCC14's runtime libs (libgcc_s, libatomic, libgomp.spec) live in $GCC14/lib,
# off the default search path; -L finds the libs, -B lets the driver read
# libgomp.spec for -fopenmp. Without these the OpenMP-threaded link fails.
LDPATH="-L$GCC14/lib -B$GCC14/lib -Wl,-rpath,$GCC14/lib"

BLIS_LIB="$BLIS_PREFIX/lib/libblis.a"
[ -f "$BLIS_LIB" ]     || { echo "ERROR: BLIS lib not found: $BLIS_LIB (run ./build-blis.sh)"; exit 1; }
[ -f "$OPENBLAS_LIB" ] || { echo "ERROR: OpenBLAS lib not found: $OPENBLAS_LIB"; exit 1; }

echo "=== toolchain ==="; gcc --version | head -1
echo "BLIS     : $BLIS_LIB"
echo "OpenBLAS : $OPENBLAS_LIB"

# --- build two binaries from the SAME source, differing only in the BLAS lib ---
# BLIS static lib needs -lm -lpthread; OpenBLAS static likewise (+ gfortran runtime).
gcc $CFLAGS bench_dgemm.c -o /tmp/bench_blis     $LDPATH "$BLIS_LIB"     -lm -lpthread
gcc $CFLAGS bench_dgemm.c -o /tmp/bench_openblas $LDPATH "$OPENBLAS_LIB" -lm -lpthread -lgfortran

echo
printf "%-8s %-8s | %-14s | %-14s | %-8s\n" threads N BLIS_GFLOPs OpenBLAS_GFLOPs ratio
echo "-----------------------------------------------------------------------"
for th in $THREADS; do
  export OPENBLAS_NUM_THREADS=$th BLIS_NUM_THREADS=$th OMP_NUM_THREADS=$th
  for n in $SIZES; do
    bo=$(/tmp/bench_blis     "$n" 2>&1 | grep -oE "DGEMM=[0-9.]+" | cut -d= -f2)
    oo=$(/tmp/bench_openblas "$n" 2>&1 | grep -oE "DGEMM=[0-9.]+" | cut -d= -f2)
    ratio=$(awk -v b="$bo" -v o="$oo" 'BEGIN{if(o>0)printf "%.2fx", b/o; else print "n/a"}')
    printf "%-8s %-8s | %-14s | %-14s | %-8s\n" "$th" "$n" "$bo" "$oo" "$ratio"
  done
done
echo
echo "ratio = BLIS / OpenBLAS  (>1 means BLIS faster). Check C[0] in raw output for NaN sanity."
