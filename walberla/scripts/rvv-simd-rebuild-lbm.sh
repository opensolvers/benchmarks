#!/usr/bin/env bash
# Rebuild only 01_BasicLBM in build-gcv after RVV SIMD patch, then short timing.
set -eo pipefail
export PS1="${PS1:-}"
export EESSI_VERSION_OVERRIDE=2025.06-001
export EESSI_USER_INSTALL=$HOME/eessi-overlay
export EESSI_NO_MODULE_PURGE_ON_INIT=1
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load EasyBuild/5.3.1 EESSI-extend/2025.06-easybuild foss/2025b CMake/4.0.3 Boost.MPI/1.88.0-gompi-2025b
export LD_LIBRARY_PATH="$EBROOTGCCCORE/lib64:${LD_LIBRARY_PATH:-}"

BUILD=/home/orangepi/walberla-bench/build-gcv
PRM=/home/orangepi/walberla-bench/prm/01_BasicLBM_bench.prm
# Prefer repo-synced prm if present on board
if [ -f "$HOME/walberla-bench/prm/01_BasicLBM_bench.prm" ]; then
  PRM=$HOME/walberla-bench/prm/01_BasicLBM_bench.prm
fi
OUT=/tmp/rvv_simd_lbm
mkdir -p "$OUT"

cd "$BUILD"
echo "=== rebuild 01_BasicLBM (make) ==="
# Force rebuild of anything depending on simd headers by touching them
touch "$HOME/walberla-bench/src/walberla-7.2/src/simd/RVV.h" \
      "$HOME/walberla-bench/src/walberla-7.2/src/simd/SIMD.h"
(cd "$BUILD/apps/tutorials/lbm" && make -j4 01_BasicLBM) 2>&1 | tee "$OUT/rebuild.log" | tail -60

BIN=$BUILD/apps/tutorials/lbm/01_BasicLBM
echo "=== binary ==="
ls -la "$BIN"
readelf -A "$BIN" | head -12 || true

# Check whether CellwiseSweep / binary references usedInstructionSet / RVV symbols
echo "=== strings / nm hints ==="
(strings "$BIN" | grep -E 'RVV|SSE|Scalar|usedInstruction' || true) | head -20

echo "=== short LBM run (np=1) ==="
if [ -f "$PRM" ]; then
  /usr/bin/time -f 'WALL=%e' "$BIN" "$PRM" >"$OUT/lbm_rvv.out" 2>"$OUT/lbm_rvv.time"
  tail -30 "$OUT/lbm_rvv.out"
  cat "$OUT/lbm_rvv.time"
else
  echo "PRM missing: $PRM"
fi
