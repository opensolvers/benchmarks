#!/bin/bash
# build-blis.sh - build BLIS (flame/blis) for the SpaceMiT X60 / K1 (rv64gcv,
# RVV 1.0, VLEN=256) under the EESSI GCC 14.3.0 toolchain, with the CBLAS/BLAS
# compatibility layer so it links against the repo's backend-agnostic harnesses.
#
# BLIS target `rv64iv` = 64-bit RVV 1.0 with dynamic VLEN detection (correct for
# the X60's VLEN=256); it selects the hand-written RVV assembly gemm microkernels
# under kernels/rviv/3/. (The `sifive_rvv` config defaults to VLEN=128 and is the
# wrong choice here.)
#
# Usage:  ./build-blis.sh [install_prefix]     # default prefix: $HOME/blis-install
# Output: $PREFIX/lib/libblis.a  (+ CBLAS/BLAS symbols) , $PREFIX/include/blis/
#
# NOTE: run ON the Orange Pi RV2 (native build). Modules are loaded BEFORE
# `set -e` because on the RV2 the lmod `module` function returns nonzero / reads
# unbound vars and would kill a strict shell (see ../README.md gotcha).

# --- EESSI GCC 14.3.0: module load does NOT repath gcc on the RV2, so force it ---
# (identical workaround to fftw/README.md - EESSI compat GCC 13.4.0 otherwise wins)
GCC14=/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software/GCCcore/14.3.0
export PATH="$GCC14/bin:$PATH"
export LD_LIBRARY_PATH="$GCC14/lib64:$GCC14/lib:$LD_LIBRARY_PATH"
# GCC14's libgcc_s / libatomic live in $GCC14/lib, off the default linker search
# path; without this the shared libblis.so link fails on -lgcc_s / -latomic.
export LDFLAGS="-L$GCC14/lib -B$GCC14/lib -Wl,-rpath,$GCC14/lib ${LDFLAGS:-}"

set -euo pipefail

PREFIX=${1:-$HOME/blis-install}
SRC=${BLIS_SRC:-$HOME/blis}
JOBS=${JOBS:-$(nproc)}

echo "=== toolchain ==="
gcc --version | head -1
gcc -dumpmachine
echo "prefix : $PREFIX"
echo "src    : $SRC"
echo "jobs   : $JOBS"

# --- clone if needed ---
if [ ! -d "$SRC/.git" ]; then
  echo "=== cloning flame/blis into $SRC ==="
  git clone --depth 1 https://github.com/flame/blis.git "$SRC"
fi
cd "$SRC"
echo "=== BLIS commit ==="; git rev-parse --short HEAD

# --- configure for RVV 1.0, dynamic VLEN, with CBLAS + BLAS compat ---
echo "=== configure rv64iv ==="
./configure --prefix="$PREFIX" --enable-cblas --enable-blas --enable-threading=openmp rv64iv

echo "=== make -j$JOBS ==="
make -j"$JOBS"

echo "=== make install ==="
make install

echo "=== result ==="
ls -la "$PREFIX"/lib/libblis.* 2>/dev/null || true
echo "cblas symbols present?"
nm -g "$PREFIX"/lib/libblis.a 2>/dev/null | grep -c -E "cblas_dgemm|cblas_dtrsm" || true
echo
echo "BLIS built. Next: ./run-ab.sh (bench vs OpenBLAS) and ./verify (correctness)."
