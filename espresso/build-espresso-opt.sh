#!/bin/bash
# Build ESPResSo 4.2.2 with slim myconfig + ForceKernelRef + P3M pair opts → ~/espresso-opt
set -e
PS1=${PS1:-}
SRC=${SRC:-$HOME/espresso-src}
BUILD=${BUILD:-$HOME/espresso-build-opt}
PREFIX=${PREFIX:-$HOME/espresso-opt}
BENCH=${BENCH:-$HOME/espresso-bench}
JOBS=${JOBS:-$(nproc)}
# Set REUSE_BUILD=1 to skip cmake wipe (faster incremental after source patches).
REUSE_BUILD=${REUSE_BUILD:-0}

export EESSI_VERSION_OVERRIDE=2025.06-001
export EESSI_USER_INSTALL="${EESSI_USER_INSTALL:-$HOME/eessi-overlay}"
export EESSI_NO_MODULE_PURGE_ON_INIT=1
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load ESPResSo/4.2.2-foss-2025b
module load CMake/4.0.3-GCCcore-14.3.0
module load Cython/3.1.2-GCCcore-14.3.0 2>/dev/null || \
  export PYTHONPATH="/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software/Cython/3.1.2-GCCcore-14.3.0/lib/python3.13/site-packages:${PYTHONPATH:-}"
# Runtime libs for installed pypresso smoke check
# shellcheck disable=SC1090
[ -f "$HOME/espresso-opt-env.sh" ] && . "$HOME/espresso-opt-env.sh"
python3 -c "import cython; print('cython', cython.__version__)"

echo "START $(date -Iseconds)"
echo "g++=$(command -v g++) ver=$(g++ -dumpversion)"
echo "SRC=$SRC PREFIX=$PREFIX JOBS=$JOBS REUSE_BUILD=$REUSE_BUILD"

cp -f "$BENCH/myconfig-slim.hpp" "$SRC/myconfig.hpp"
cp -f "$BENCH/patches/coulomb_inline.hpp" \
  "$SRC/src/core/electrostatics/coulomb_inline.hpp"
cp -f "$BENCH/patches/unit_test.cmake" "$SRC/cmake/unit_test.cmake"
cp -f "$BENCH/patches/forces_inline.hpp" "$SRC/src/core/forces_inline.hpp"
cp -f "$BENCH/patches/forces.cpp" "$SRC/src/core/forces.cpp"
cp -f "$BENCH/patches/p3m.hpp" "$SRC/src/core/electrostatics/p3m.hpp"
cp -f "$BENCH/patches/p3m.cpp" "$SRC/src/core/electrostatics/p3m.cpp"
echo "Applied slim myconfig + ForceKernelRef + P3M monomorph + SR force table"

if [ "$REUSE_BUILD" != 1 ] || [ ! -f "$BUILD/CMakeCache.txt" ]; then
  rm -rf "$BUILD"
  mkdir -p "$BUILD" "$PREFIX"
  cd "$BUILD"
  cmake "$SRC" \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_CUDA=OFF \
    -DWITH_TESTS=OFF \
    -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" \
    -DCMAKE_C_FLAGS="-O3 -DNDEBUG" \
    -DPYTHON_EXECUTABLE="$(command -v python3)"
else
  mkdir -p "$PREFIX"
  cd "$BUILD"
  echo "Reusing existing build dir"
fi

echo "=== building ==="
cmake --build . -j "$JOBS"
cmake --install .

echo "=== feature check ==="
export PATH="$PREFIX/bin:$PATH"
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
export PYTHONPATH="$PREFIX/lib/python${PYVER}/site-packages:${PYTHONPATH:-}"
"$PREFIX/bin/pypresso" -c "import espressomd; print(sorted(espressomd.features()))"
echo "DONE $(date -Iseconds) PREFIX=$PREFIX"
