#!/usr/bin/env bash
# A/B: 02_HeatEquation stock Jacobi — rv64gc vs rv64gcv (+ optional true novec) on Orange Pi.
# Installs prm-driven bench fork over tutorial source, rebuilds in build-gc / build-gcv.
# Novec: recompiles the HeatEquation TU with -fno-tree-vectorize (cmake CXXFLAGS override
# does NOT work; must invoke c++ directly on the .cpp).
set -eo pipefail
export PS1="${PS1:-}"
export FPATH="${FPATH:-}"

ROOT="${ROOT:-/home/orangepi/walberla-bench}"
SRC_APP="${SRC_APP:-$ROOT/apps/heat_equation_bench}"
WALBERLA_PDE="$ROOT/src/walberla-7.2/apps/tutorials/pde"
BUILD_GC="${BUILD_GC:-$ROOT/build-gc}"
BUILD_GCV="${BUILD_GCV:-$ROOT/build-gcv}"
OUT_DIR="${OUT_DIR:-$ROOT/results/heat-equation}"
SUMMARY="${SUMMARY:-$ROOT/results/heat-equation-rvv-ab.txt}"
PRM_NP1="${PRM_NP1:-$ROOT/prm/02_HeatEquation_bench.prm}"
PRM_NP4="${PRM_NP4:-$ROOT/prm/02_HeatEquation_bench_mpi4.prm}"
DO_NP4="${DO_NP4:-1}"
DO_NOVEC="${DO_NOVEC:-1}"
SKIP_BUILD="${SKIP_BUILD:-0}"

export EESSI_VERSION_OVERRIDE=2025.06-001
export EESSI_USER_INSTALL=$HOME/eessi-overlay
export EESSI_NO_MODULE_PURGE_ON_INIT=1
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load EasyBuild/5.3.1 EESSI-extend/2025.06-easybuild foss/2025b CMake/4.0.3 Boost.MPI/1.88.0-gompi-2025b
export LD_LIBRARY_PATH="$EBROOTGCCCORE/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUT_DIR" "$SRC_APP" "$(dirname "$SUMMARY")"

{
  echo "======== HeatEquation A/B $(date -Is) (gc vs gcv[+novec]) ========"
  echo "board: $(uname -a)"
  lscpu | grep -E 'Model name|CPU\(s\)|MHz|Vector' | head -12 || true
  echo "prm np1:"; cat "$PRM_NP1"
  echo "prm np4:"; cat "$PRM_NP4"
} | tee "$SUMMARY"

STOCK_CPP="$WALBERLA_PDE/02_HeatEquation.cpp"
BENCH_CPP="$SRC_APP/02_HeatEquation.cpp"
[[ -f "$BENCH_CPP" ]] || { echo "Missing $BENCH_CPP" >&2; exit 1; }

if [[ ! -f "$STOCK_CPP.stock" ]]; then
  cp -f "$STOCK_CPP" "$STOCK_CPP.stock"
  echo "Saved stock tutorial to $STOCK_CPP.stock" | tee -a "$SUMMARY"
fi
echo "=== INSTALL bench 02_HeatEquation.cpp ===" | tee -a "$SUMMARY"
cp -f "$BENCH_CPP" "$STOCK_CPP"

CXX_BIN="${CXX_BIN:-$(command -v c++)}"
COMMON_INCLUDES=(-I"$BUILD_GCV/src" -I"$ROOT/src/walberla-7.2/src")
COMMON_WARN=(-Wall -Wconversion -Wshadow -Wfloat-equal -Wextra -pedantic -Wno-maybe-uninitialized -Wno-array-bounds -pthread)
OBJ_GCV="$BUILD_GCV/apps/tutorials/pde/CMakeFiles/02_HeatEquation.dir/02_HeatEquation.cpp.o"

compile_tu() {
  local bdir=$1 flags=$2 obj=$3 logbase=$4
  "$CXX_BIN" -I"$bdir/src" -I"$ROOT/src/walberla-7.2/src" \
    $flags "${COMMON_WARN[@]}" -O3 -DNDEBUG -std=c++20 \
    -fopt-info-vec-optimized -fopt-info-vec-missed \
    -c "$STOCK_CPP" -o "$obj" \
    >"$OUT_DIR/${logbase}.log" 2>"$OUT_DIR/${logbase}_stderr.log"
}

build_one() {
  local bdir=$1 tag=$2
  local bin="$bdir/apps/tutorials/pde/02_HeatEquation"
  echo "==== BUILD $tag $(date -Is) ====" | tee -a "$SUMMARY"
  cd "$bdir"
  rm -f "apps/tutorials/pde/CMakeFiles/02_HeatEquation.dir/02_HeatEquation.cpp.o"
  set +e
  cmake --build . -j4 --target 02_HeatEquation \
    >"$OUT_DIR/build_${tag}.log" 2>"$OUT_DIR/build_${tag}_stderr.log"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "BUILD_FAIL $tag rc=$rc" | tee -a "$SUMMARY"
    tail -50 "$OUT_DIR/build_${tag}_stderr.log" | tee -a "$SUMMARY"
    exit $rc
  fi
  echo "BUILD_OK $tag" | tee -a "$SUMMARY"
  ls -la "$bin" | tee -a "$SUMMARY"
  echo "ISA $tag:" | tee -a "$SUMMARY"
  readelf -A "$bin" | head -12 | tee "$OUT_DIR/readelf_${tag}.txt" | tee -a "$SUMMARY"
}

build_novec_and_restore_gcv() {
  local bin_out="$OUT_DIR/02_HeatEquation_novec"
  local flags_gcv="-O2 -ftree-vectorize -march=rv64gcv -mabi=lp64d -fno-math-errno -fPIC"
  local flags_novec="-O2 -fno-tree-vectorize -march=rv64gcv -mabi=lp64d -fno-math-errno -fPIC"

  echo "==== BUILD true novec (direct c++ -fno-tree-vectorize) $(date -Is) ====" | tee -a "$SUMMARY"
  compile_tu "$BUILD_GCV" "$flags_novec" "$OBJ_GCV" build_novec_true
  grep -E 'optimized:|vectorized' "$OUT_DIR/build_novec_true_stderr.log" \
    | head -20 | tee "$OUT_DIR/vec_novec.txt" | tee -a "$SUMMARY" || true
  cd "$BUILD_GCV"
  cmake --build . -j4 --target 02_HeatEquation >"$OUT_DIR/relink_novec.log" 2>&1
  cp -f apps/tutorials/pde/02_HeatEquation "$bin_out"
  ls -la "$bin_out" | tee -a "$SUMMARY"
  readelf -A "$bin_out" | head -12 | tee "$OUT_DIR/readelf_novec.txt" | tee -a "$SUMMARY"
  objdump -d --disassemble="_ZN8walberla15JacobiIterationclEv" "$bin_out" >"$OUT_DIR/jacobi_novec.s"

  echo "==== RESTORE gcv with opt-info $(date -Is) ====" | tee -a "$SUMMARY"
  compile_tu "$BUILD_GCV" "$flags_gcv" "$OBJ_GCV" build_gcv_true
  grep -E 'optimized:|02_HeatEquation.cpp' "$OUT_DIR/build_gcv_true_stderr.log" \
    | head -100 | tee "$OUT_DIR/vec_optimized_gcv.txt" | tee -a "$SUMMARY" || true
  cmake --build . -j4 --target 02_HeatEquation >"$OUT_DIR/relink_gcv.log" 2>&1
  objdump -d --disassemble="_ZN8walberla15JacobiIterationclEv" apps/tutorials/pde/02_HeatEquation >"$OUT_DIR/jacobi_gcv.s"
  objdump -d --disassemble="_ZN8walberla15JacobiIterationclEv" \
    "$BUILD_GC/apps/tutorials/pde/02_HeatEquation" >"$OUT_DIR/jacobi_gc.s" || true

  echo "Jacobi RVV insn counts:" | tee -a "$SUMMARY"
  for tag in gc gcv novec; do
    echo -n "  $tag: " | tee -a "$SUMMARY"
    for i in vle64 vse64 vfdiv vsetvli; do
      c=$(grep -c "[[:space:]]$i" "$OUT_DIR/jacobi_${tag}.s" 2>/dev/null || echo 0)
      echo -n "$i=$c " | tee -a "$SUMMARY"
    done
    echo | tee -a "$SUMMARY"
  done
}

run_bin() {
  local tag=$1 bin=$2 np=$3 prm=$4
  local log="$OUT_DIR/run-${tag}-np${np}.log"
  local work="$OUT_DIR/work-${tag}-np${np}"
  rm -rf "$work"
  mkdir -p "$work"
  cp "$prm" "$work/bench.prm"
  echo "==== RUN $tag np=$np $(date -Is) ====" | tee -a "$SUMMARY"
  cd "$work"
  local t0 t1 wall
  t0=$(date +%s)
  set +e
  if [[ "$np" = "1" ]]; then
    "$bin" bench.prm >"$log" 2>&1
  else
    mpirun -np "$np" "$bin" bench.prm >"$log" 2>&1
  fi
  local rc=$?
  set -e
  t1=$(date +%s)
  wall=$((t1 - t0))
  echo "WALL $wall  rc=$rc" | tee -a "$SUMMARY"
  if [[ $rc -ne 0 ]]; then
    echo "FAILED $tag" | tee -a "$SUMMARY"
    tail -40 "$log" | tee -a "$SUMMARY"
    return $rc
  fi
  grep -E 'HeatEquation|checksum' "$log" | tee -a "$SUMMARY" || true
  echo "$wall" >"$OUT_DIR/wall_${tag}_np${np}.txt"
}

if [[ "$SKIP_BUILD" != "1" ]]; then
  build_one "$BUILD_GC" gc
  build_one "$BUILD_GCV" gcv
  if [[ "$DO_NOVEC" = "1" ]]; then
    build_novec_and_restore_gcv
  fi
else
  echo "=== SKIP_BUILD=1 ===" | tee -a "$SUMMARY"
fi

BIN_GC="$BUILD_GC/apps/tutorials/pde/02_HeatEquation"
BIN_GCV="$BUILD_GCV/apps/tutorials/pde/02_HeatEquation"
BIN_NOVEC="$OUT_DIR/02_HeatEquation_novec"

run_bin gc "$BIN_GC" 1 "$PRM_NP1"
run_bin gcv "$BIN_GCV" 1 "$PRM_NP1"
if [[ "$DO_NOVEC" = "1" && -x "$BIN_NOVEC" ]]; then
  run_bin novec "$BIN_NOVEC" 1 "$PRM_NP1"
fi

if [[ "$DO_NP4" = "1" ]]; then
  run_bin gc "$BIN_GC" 4 "$PRM_NP4"
  run_bin gcv "$BIN_GCV" 4 "$PRM_NP4"
  if [[ "$DO_NOVEC" = "1" && -x "$BIN_NOVEC" ]]; then
    run_bin novec "$BIN_NOVEC" 4 "$PRM_NP4"
  fi
fi

{
  echo "---- speedup summary ----"
  for np in 1 4; do
    [[ -f "$OUT_DIR/wall_gc_np${np}.txt" && -f "$OUT_DIR/wall_gcv_np${np}.txt" ]] || continue
    gc=$(cat "$OUT_DIR/wall_gc_np${np}.txt")
    gcv=$(cat "$OUT_DIR/wall_gcv_np${np}.txt")
    python3 -c "gc=float('$gc'); gcv=float('$gcv'); print(f'np={$np}: gc={gc:.0f}s gcv={gcv:.0f}s speedup={gc/gcv:.3f}x ({(gc-gcv)/gc*100:.2f}% faster gcv)')"
    if [[ -f "$OUT_DIR/wall_novec_np${np}.txt" ]]; then
      nov=$(cat "$OUT_DIR/wall_novec_np${np}.txt")
      python3 -c "gc=float('$gc'); gcv=float('$gcv'); nov=float('$nov'); print(f'np={$np}: novec={nov:.0f}s  gcv_vs_novec={nov/gcv:.3f}x  novec_vs_gc={gc/nov:.3f}x')"
    fi
  done
  echo "======== HeatEquation A/B DONE $(date -Is) ========"
} | tee -a "$SUMMARY"
