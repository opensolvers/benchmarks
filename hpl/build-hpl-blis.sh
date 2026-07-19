#!/bin/bash
# build-hpl-blis.sh - build HPL 2.3 (xhpl) linked DIRECTLY against a static
# BLIS libblis.a, for the SpaceMiT X60 / K1 (rv64gcv, RVV 1.0, VLEN=256).
#
# There is no FlexiBLAS on the board, so the BLAS provider is baked in at link
# time (contrast run-hpl-ab.sh, which swaps OpenBLAS backends at run time via
# FlexiBLAS under a prebuilt xhpl). This builds a *dedicated* BLIS-linked xhpl.
#
# Prereqs:
#   - BLIS built + installed (see ../BLIS/build-blis.sh) -> $BLIS_HOME
#   - an MPI toolchain providing mpicc + mpirun (OpenMPI). The repo's HPL runs
#     use the HPL/2.3-foss-* module; here we only need its MPI compilers, so:
#         module load OpenMPI          # or the foss/HPL module that pulls it in
#
# Usage:  BLIS_HOME=$HOME/blis-install ./build-hpl-blis.sh
# Output: $HPL_SRC/bin/rv64_blis/xhpl
#
# NOTE: run ON the Orange Pi RV2 (native build). Modules are loaded BEFORE
# `set -e` because on the RV2 the lmod `module` function returns nonzero / reads
# unbound vars and would kill a strict shell (see ../BLIS/README.md gotcha).

# --- EESSI GCC 14.3.0: module load does NOT repath gcc on the RV2, so force it ---
GCC14=/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software/GCCcore/14.3.0
export GCC14
export PATH="$GCC14/bin:$PATH"
export LD_LIBRARY_PATH="$GCC14/lib64:$GCC14/lib:$LD_LIBRARY_PATH"

set -euo pipefail

BLIS_HOME=${BLIS_HOME:-$HOME/blis-install}
export BLIS_HOME
HPL_SRC=${HPL_SRC:-$HOME/hpl-2.3}
JOBS=${JOBS:-$(nproc)}
HERE=$(cd "$(dirname "$0")" && pwd)

echo "=== toolchain ==="
command -v mpicc  >/dev/null || { echo "ERROR: mpicc not on PATH (module load OpenMPI / foss)"; exit 1; }
mpicc --version | head -1
echo "underlying cc: $(mpicc -show 2>/dev/null | awk '{print $1}')"
echo "BLIS_HOME : $BLIS_HOME"
echo "HPL_SRC   : $HPL_SRC"

# --- sanity: BLIS built with cblas + carries cblas_dgemm ---
[ -f "$BLIS_HOME/lib/libblis.a" ] || { echo "ERROR: $BLIS_HOME/lib/libblis.a missing (build BLIS first)"; exit 1; }
[ -f "$BLIS_HOME/include/blis/cblas.h" ] || { echo "ERROR: $BLIS_HOME/include/blis/cblas.h missing (BLIS needs --enable-cblas)"; exit 1; }
if [ "$(nm -g "$BLIS_HOME/lib/libblis.a" 2>/dev/null | grep -c ' T cblas_dgemm$')" = 0 ]; then
  echo "ERROR: libblis.a has no cblas_dgemm symbol (rebuild BLIS with --enable-cblas --enable-blas)"; exit 1
fi

# --- fetch HPL 2.3 if needed ---
if [ ! -d "$HPL_SRC" ]; then
  echo "=== fetching HPL 2.3 ==="
  tmp=$(mktemp -d)
  ( cd "$tmp" && curl -fsSLO https://netlib.org/benchmark/hpl/hpl-2.3.tar.gz && tar xzf hpl-2.3.tar.gz )
  mv "$tmp/hpl-2.3" "$HPL_SRC"
  rm -rf "$tmp"
fi

# --- drop in the BLIS make config (TOPdir must match HPL_SRC) ---
echo "=== install Make.rv64_blis ==="
sed "s|^TOPdir .*|TOPdir       = $HPL_SRC|" "$HERE/Make.rv64_blis" > "$HPL_SRC/Make.rv64_blis"

# --- build ---
cd "$HPL_SRC"
echo "=== make arch=rv64_blis ==="
make arch=rv64_blis clean_arch_all >/dev/null 2>&1 || true
# HPL 2.3's top Makefile is NOT -j-safe for its startup_dir/refresh_src phase
# (parallel `cp` races to create src/*/rv64_blis before the dir exists). Do the
# directory scaffolding serially first, then compile in parallel.
make arch=rv64_blis startup_dir >/dev/null 2>&1 || true
make arch=rv64_blis build_src -j"$JOBS" || make arch=rv64_blis
make arch=rv64_blis build_tst

# --- result ---
XHPL="$HPL_SRC/bin/rv64_blis/xhpl"
if [ -x "$XHPL" ]; then
  echo "=== OK: $XHPL ==="
  echo "BLAS symbols resolved to BLIS?"
  { nm "$XHPL" 2>/dev/null || true; } | grep -E "cblas_dgemm|bli_" | head -3 || true
  echo "Next: ./run-hpl-blis.sh HPL.dat   (from $HERE)"
else
  echo "=== BUILD FAILED: $XHPL not produced ==="; exit 1
fi
