#!/usr/bin/env bash
# Build + A/B run: stock CellwiseSweep vs stream-then-SoA-collide on Orange Pi (rv64gcv).
# Installs app sources into walberla-7.2 tutorials/lbm, rebuilds one target in build-gcv.
set -e
export PS1="${PS1:-}"

ROOT="${ROOT:-/home/orangepi/walberla-bench}"
SRC_APP="${SRC_APP:-$ROOT/apps/collide_stream_split}"
WALBERLA_LBM="$ROOT/src/walberla-7.2/apps/tutorials/lbm"
BUILD="${BUILD:-$ROOT/build-gcv}"
OUT_DIR="${OUT_DIR:-$ROOT/results/collide-stream-split}"
PRM_NP1="${PRM_NP1:-$ROOT/prm/01_BasicLBM_collide_stream_split.prm}"
PRM_NP4="${PRM_NP4:-$ROOT/prm/01_BasicLBM_collide_stream_split_mpi4.prm}"
DO_NP4="${DO_NP4:-1}"

export EESSI_VERSION_OVERRIDE=2025.06-001
export EESSI_USER_INSTALL=$HOME/eessi-overlay
export EESSI_NO_MODULE_PURGE_ON_INIT=1
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load EasyBuild/5.3.1 EESSI-extend/2025.06-easybuild foss/2025b CMake/4.0.3 Boost.MPI/1.88.0-gompi-2025b
export LD_LIBRARY_PATH="$EBROOTGCCCORE/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUT_DIR"

echo "=== INSTALL sources into $WALBERLA_LBM ==="
cp -f "$SRC_APP/01_BasicLBM_CollideStreamSplit.cpp" "$WALBERLA_LBM/"
cp -f "$SRC_APP/SoaCollideKernels.cpp" "$WALBERLA_LBM/"
cp -f "$SRC_APP/SoaCollideKernels.h" "$WALBERLA_LBM/"

CMAKE_LBM="$WALBERLA_LBM/CMakeLists.txt"
MARKER="01_BasicLBM_CollideStreamSplit"
if ! grep -q "$MARKER" "$CMAKE_LBM"; then
  echo "=== PATCH CMakeLists.txt ==="
  cat >> "$CMAKE_LBM" <<'EOF'

waLBerla_add_executable ( NAME 01_BasicLBM_CollideStreamSplit
                          FILES 01_BasicLBM_CollideStreamSplit.cpp SoaCollideKernels.cpp
      DEPENDS walberla::blockforest walberla::core walberla::field walberla::lbm walberla::geometry walberla::timeloop )

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set_source_files_properties(SoaCollideKernels.cpp PROPERTIES
    COMPILE_FLAGS "-fopt-info-vec-optimized")
endif()
EOF
else
  echo "CMakeLists already contains $MARKER"
fi

BIN="$BUILD/apps/tutorials/lbm/01_BasicLBM_CollideStreamSplit"
SKIP_BUILD="${SKIP_BUILD:-0}"

if [[ "$SKIP_BUILD" != "1" ]]; then
  echo "=== RECONFIGURE / BUILD target ==="
  cd "$BUILD"
  # Re-run cmake so new target is picked up (cheap if cache exists)
  cmake . >"$OUT_DIR/cmake_reconfigure.log" 2>&1
  # Capture vec opt-info from compiling SoaCollideKernels.cpp
  set +e
  make 01_BasicLBM_CollideStreamSplit -j4 >"$OUT_DIR/build.log" 2>"$OUT_DIR/build_stderr.log"
  BUILD_RC=$?
  set -e
  if [[ $BUILD_RC -ne 0 ]]; then
    echo "BUILD_FAIL rc=$BUILD_RC" >&2
    tail -80 "$OUT_DIR/build_stderr.log" >&2
    tail -40 "$OUT_DIR/build.log" >&2
    exit $BUILD_RC
  fi
  echo "BUILD_OK"
else
  echo "=== SKIP_BUILD=1 (using existing $BIN) ==="
  [[ -x "$BIN" ]] || { echo "Missing binary $BIN" >&2; exit 1; }
fi

ls -la "$BIN"
readelf -A "$BIN" | head -12 | tee "$OUT_DIR/readelf.txt"

# Extract vectorization notes for the collide TU
if [[ "$SKIP_BUILD" != "1" ]]; then
  grep -E "SoaCollideKernels|vectorized|LOOP VECTORIZED" "$OUT_DIR/build_stderr.log" \
    >"$OUT_DIR/vec_optimized_collide.txt" || true
  if [[ ! -s "$OUT_DIR/vec_optimized_collide.txt" ]]; then
    grep -E "SoaCollideKernels|vectorized|LOOP VECTORIZED" "$OUT_DIR/build.log" \
      >"$OUT_DIR/vec_optimized_collide.txt" || true
  fi
fi
echo "=== VEC REPORT (collide TU) ==="
cat "$OUT_DIR/vec_optimized_collide.txt" || true

SUMMARY="$OUT_DIR/../collide-stream-split.txt"
{
  echo "======== collide-SoA-then-stream A/B $(date -Iseconds) ========"
  echo "board: $(uname -a)"
  echo "binary: $BIN"
  echo "CXX_FLAGS: $(grep CMAKE_CXX_FLAGS: $BUILD/CMakeCache.txt | head -1)"
  echo
  echo "--- vectorize notes (SoaCollideKernels.cpp) ---"
  cat "$OUT_DIR/vec_optimized_collide.txt" 2>/dev/null || echo "(none captured)"
  echo
} >"$SUMMARY"

run_one() {
  local name="$1"
  local np="$2"
  local prm="$3"
  local mode="$4"
  local log="$OUT_DIR/run_${name}.log"
  local work="$OUT_DIR/work_${name}"
  rm -rf "$work"
  mkdir -p "$work"
  # Materialize prm with sweepMode so we do not rely on CLI unknown-arg handling.
  sed -E "s/sweepMode[[:space:]]+[A-Za-z_]+/sweepMode ${mode}/" "$prm" >"$work/bench.prm"
  echo "==== RUN $name np=$np mode=$mode $(date -Iseconds) ====" | tee -a "$SUMMARY"
  cd "$work"
  local t0 t1 rc
  t0=$(date +%s)
  set +e
  if [[ "$np" -eq 1 ]]; then
    "$BIN" "$work/bench.prm" >"$log" 2>&1
  else
    mpirun -np "$np" "$BIN" "$work/bench.prm" >"$log" 2>&1
  fi
  rc=$?
  set -e
  t1=$(date +%s)
  echo "RUN_RC=$rc name=$name" | tee -a "$SUMMARY"
  if [[ $rc -ne 0 ]]; then
    echo "FAILED $name" | tee -a "$SUMMARY"
    tail -40 "$log" | tee -a "$SUMMARY"
    echo >>"$SUMMARY"
    return 0
  fi
  # Prefer app chrono WALL (fractional); fall back to date seconds.
  local wall
  wall=$(grep -E "WALL " "$log" | grep -oE "[0-9]+\.[0-9]+|[0-9]+" | tail -1)
  if [[ -z "$wall" ]]; then
    wall=$((t1 - t0))
  fi
  local mlups checksum
  mlups=$(grep -E "MLUPS " "$log" | tail -1)
  checksum=$(grep -E "checksum_density_sum=" "$log" | tail -1)
  echo "WALL $wall" | tee -a "$SUMMARY"
  echo "WALL_date_s $((t1 - t0))" | tee -a "$SUMMARY"
  echo "$mlups" | tee -a "$SUMMARY"
  echo "$checksum" | tee -a "$SUMMARY"
  grep -E "Estimated Remaining|sweepMode=|WALL |MLUPS |checksum_" "$log" | tail -20 >>"$SUMMARY" || true
  echo >>"$SUMMARY"
}

run_one stock_np1 1 "$PRM_NP1" stock
run_one split_np1 1 "$PRM_NP1" split

if [[ "$DO_NP4" == "1" ]]; then
  run_one stock_np4 4 "$PRM_NP4" stock
  run_one split_np4 4 "$PRM_NP4" split
fi

# Speedup summary
python3 - <<'PY' "$SUMMARY" "$OUT_DIR" || true
import re, sys
summary = open(sys.argv[1]).read()
walls = dict(re.findall(r"==== RUN (\w+) .*?\nWALL ([0-9.]+)", summary, re.S))
# fallback: sequential WALL after RUN headers
walls = {}
name = None
for line in summary.splitlines():
    m = re.match(r"==== RUN (\w+)", line)
    if m:
        name = m.group(1)
    m = re.match(r"WALL ([0-9.]+)", line)
    if m and name and name not in walls:
        walls[name] = float(m.group(1))

def spd(a, b):
    if a in walls and b in walls and walls[b] > 0:
        return walls[a] / walls[b]
    return float("nan")

lines = []
lines.append("--- speedup (stock_WALL / split_WALL; >1 => split faster) ---")
for tag in ("np1", "np4"):
    s, p = f"stock_{tag}", f"split_{tag}"
    if s in walls and p in walls:
        lines.append(f"{tag}: stock={walls[s]:.3f}s split={walls[p]:.3f}s speedup={spd(s,p):.3f}x")
open(sys.argv[1], "a").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo "======== DONE $(date -Iseconds) ========" | tee -a "$SUMMARY"
echo "Summary: $SUMMARY"
echo "Details: $OUT_DIR/"
