#!/bin/bash
# Apply the RISC-V RVV SIMD backend patch to a pristine GROMACS 2026.2 tree and
# build it on a SpaceMiT X60 / K1 board (RVV 1.0, VLEN=256) under EESSI GCC 14.3.
#
# This produces a `gmx` that reports `SIMD instructions: RISCV_RVV` instead of the
# stock `SIMD: None`, vectorizing the nonbonded `Force` kernels (float / mixed
# precision). See ../README.md for the measured speedup and rvv-backend/README.md
# for what the patch changes.
#
# Usage:
#   ./apply-and-build.sh /path/to/gromacs-2026.2   # a pristine v2026.2 checkout
#
# Prereqs on the board:
#   - a git checkout of GROMACS at tag v2026.2 (commit da9e013)
#   - EESSI 2025.06 (or dev.eessi.io riscv overlay): GCC 14.3.0 + CMake
set -uo pipefail

SRC=${1:?usage: apply-and-build.sh /path/to/gromacs-2026.2}
PATCH="$(cd "$(dirname "$0")" && pwd)/gromacs-2026.2-riscv-rvv-float.patch"
BUILD="$SRC/build-rvv"
JOBS=${JOBS:-8}
RVV_LEN=${RVV_LEN:-256}   # VLEN in bits; X60 = 256

test -f "$PATCH" || { echo "FATAL: patch not found: $PATCH"; exit 1; }
test -d "$SRC/.git" || { echo "FATAL: $SRC is not a git checkout of GROMACS"; exit 1; }

# --- EESSI toolchain (load modules BEFORE any set -e; lmod returns nonzero) ----
export EESSI_VERSION_OVERRIDE=2025.06-001
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash >/dev/null 2>&1
module load GCC/14.3.0 CMake >/dev/null 2>&1
# GCC 14.3 runtime libstdc++ carries CXXABI_1.3.15 required by the built binaries;
# the EESSI *compat* libstdc++ (GCC 13) does NOT — prepend the real one.
GCCLIB=$(dirname "$(gcc -print-file-name=libstdc++.so.6)")
export LD_LIBRARY_PATH="$GCCLIB:${LD_LIBRARY_PATH:-}"

cd "$SRC"
echo "=== base commit ==="; git rev-parse HEAD
echo "=== applying $PATCH ==="
# The patch creates the new impl_riscv_rvv/ headers AND edits 7 tracked files.
if git apply --check "$PATCH" 2>/dev/null; then
    git apply "$PATCH"
    echo "APPLY=OK"
else
    echo "FATAL: patch does not apply cleanly to this tree (expected pristine v2026.2)."
    echo "       If the backend dir already exists, remove it first:"
    echo "       rm -rf src/gromacs/simd/include/gromacs/simd/impl_riscv_rvv"
    exit 1
fi

echo "=== configuring (GMX_SIMD=RISCV_RVV, VLEN=$RVV_LEN, single precision) ==="
rm -rf "$BUILD" && mkdir -p "$BUILD" && cd "$BUILD"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGMX_SIMD=RISCV_RVV \
    -DGMX_SIMD_RISCV_RVV_LENGTH=$RVV_LEN \
    -DGMX_DOUBLE=OFF \
    -DGMX_MPI=OFF \
    -DGMX_GPU=OFF \
    -DGMX_FFT_LIBRARY=fftpack \
    -DGMX_BUILD_OWN_FFTW=OFF \
    -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
    || { echo "CMAKE_RC=$?"; exit 1; }

echo "=== building gmx + simd-test (-j$JOBS; slow on X60) ==="
make -j"$JOBS" gmx simd-test
RC=$?
echo "MAKE_RC=$RC"
[ $RC -eq 0 ] || exit $RC

echo "=== verify SIMD banner ==="
./bin/gmx --quiet --version 2>&1 | grep -iE "GROMACS version|SIMD instructions"
echo
echo "Done. Run the float SIMD unit tests with:"
echo "  LD_LIBRARY_PATH=$GCCLIB:\$LD_LIBRARY_PATH \\"
echo "  $BUILD/bin/simd-test --gtest_filter='SimdFloatingpoint*:SimdMath*:Simd4Floatingpoint*:Simd4Math*:SimdVectorOperations*'"
