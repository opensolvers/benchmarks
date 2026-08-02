#!/usr/bin/env bash
# A/B: UniformGridBenchmark — rv64gc vs rv64gcv on Orange Pi.
# Focus: --not-fused (SoA SplitPureSweep collide separate from stream).
# Optional: fused default control; np=4.
# No CODEGEN/PYTHON; no hand SIMD; GCC foss only.
set -eo pipefail
export PS1="${PS1:-}"
export FPATH="${FPATH:-}"

ROOT="${ROOT:-/home/orangepi/walberla-bench}"
SRC="${SRC:-$ROOT/src/walberla-7.2}"
BUILD_GC="${BUILD_GC:-$ROOT/build-gc}"
BUILD_GCV="${BUILD_GCV:-$ROOT/build-gcv}"
OUT_DIR="${OUT_DIR:-$ROOT/results/uniform-grid}"
SUMMARY="${SUMMARY:-$ROOT/results/uniform-grid-rvv-ab.txt}"
PRM_NP1="${PRM_NP1:-$ROOT/prm/UniformGrid_bench.prm}"
PRM_NP4="${PRM_NP4:-$ROOT/prm/UniformGrid_bench_mpi4.prm}"
DO_NP4="${DO_NP4:-1}"
DO_FUSED="${DO_FUSED:-1}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_OBJDUMP="${SKIP_OBJDUMP:-0}"

export EESSI_VERSION_OVERRIDE=2025.06-001
export EESSI_USER_INSTALL=$HOME/eessi-overlay
export EESSI_NO_MODULE_PURGE_ON_INIT=1
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load EasyBuild/5.3.1 EESSI-extend/2025.06-easybuild foss/2025b CMake/4.0.3 Boost.MPI/1.88.0-gompi-2025b
export LD_LIBRARY_PATH="$EBROOTGCCCORE/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$OUT_DIR" "$(dirname "$SUMMARY")"

BIN_REL="apps/benchmarks/UniformGrid/UniformGridBenchmark"

{
  echo "======== UniformGrid A/B $(date -Is) (gc vs gcv, focus --not-fused) ========"
  echo "board: $(uname -a)"
  lscpu | grep -E 'Model name|CPU\(s\)|MHz|Vector' | head -12 || true
  echo "binary: UniformGridBenchmark"
  echo "CLI: <prm> [--not-fused] [--not-split] [--not-pure] [--zyxf] [--full-comm] [--direct-comm] [--trt|--mrt] [--comp]"
  echo "defaults: split+pure+fzyx+fused (SRT incompressible)"
  echo "prm np1:"; cat "$PRM_NP1"
  echo "prm np4:"; cat "$PRM_NP4"
} | tee "$SUMMARY"

reconfigure_benchmarks() {
  local bdir=$1 tag=$2
  echo "==== RECONFIGURE $tag WALBERLA_BUILD_BENCHMARKS=ON $(date -Is) ====" | tee -a "$SUMMARY"
  cd "$bdir"
  cmake -DWALBERLA_BUILD_BENCHMARKS=ON . \
    >"$OUT_DIR/cmake_${tag}.log" 2>"$OUT_DIR/cmake_${tag}_stderr.log"
  grep -E 'WALBERLA_BUILD_BENCHMARKS|CMAKE_CXX_FLAGS:STRING' CMakeCache.txt | tee -a "$SUMMARY"
}

build_one() {
  local bdir=$1 tag=$2
  local bin="$bdir/$BIN_REL"
  echo "==== BUILD $tag UniformGridBenchmark $(date -Is) ====" | tee -a "$SUMMARY"
  cd "$bdir"
  set +e
  cmake --build . -j4 --target UniformGridBenchmark \
    >"$OUT_DIR/build_${tag}.log" 2>"$OUT_DIR/build_${tag}_stderr.log"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "BUILD_FAIL $tag rc=$rc" | tee -a "$SUMMARY"
    tail -80 "$OUT_DIR/build_${tag}_stderr.log" | tee -a "$SUMMARY"
    exit $rc
  fi
  echo "BUILD_OK $tag" | tee -a "$SUMMARY"
  ls -la "$bin" | tee -a "$SUMMARY"
  echo "ISA $tag:" | tee -a "$SUMMARY"
  readelf -A "$bin" | head -14 | tee "$OUT_DIR/readelf_${tag}.txt" | tee -a "$SUMMARY"
}

# Rebuild gcv TU with -fopt-info-vec to confirm collide auto-vec, then restore.
rebuild_gcv_with_vecinfo() {
  local bdir="$BUILD_GCV"
  local objdir="$bdir/apps/benchmarks/UniformGrid/CMakeFiles/UniformGridBenchmark.dir"
  local cpp="$SRC/apps/benchmarks/UniformGrid/UniformGrid.cpp"
  local obj="$objdir/UniformGrid.cpp.o"
  local flags="-O2 -ftree-vectorize -march=rv64gcv -mabi=lp64d -fno-math-errno -fPIC"
  local warns="-Wall -Wconversion -Wshadow -Wfloat-equal -Wextra -pedantic -Wno-maybe-uninitialized -Wno-array-bounds -pthread"
  local CXX_BIN
  CXX_BIN="$(command -v c++)"

  echo "==== VEC-INFO rebuild UniformGrid.cpp (gcv) $(date -Is) ====" | tee -a "$SUMMARY"
  mkdir -p "$objdir"
  "$CXX_BIN" -I"$bdir/src" -I"$SRC/src" $flags $warns -O3 -DNDEBUG -std=c++20 \
    -fopt-info-vec-optimized -fopt-info-vec-missed \
    -c "$cpp" -o "$obj" \
    >"$OUT_DIR/build_gcv_vecinfo.log" 2>"$OUT_DIR/build_gcv_vecinfo_stderr.log"
  grep -E 'optimized:|vectorized|SplitPure|collide|UniformGrid' \
    "$OUT_DIR/build_gcv_vecinfo_stderr.log" \
    | head -120 | tee "$OUT_DIR/vec_optimized_gcv.txt" | tee -a "$SUMMARY" || true
  cd "$bdir"
  cmake --build . -j4 --target UniformGridBenchmark >"$OUT_DIR/relink_gcv_vecinfo.log" 2>&1
}

objdump_collide() {
  local bin=$1 out=$2
  # Try common mangled symbols for SplitPureSweep collide / CollideSweep
  local syms
  syms=$(nm -C "$bin" 2>/dev/null | grep -E 'SplitPureSweep|CollideSweep|collide' | head -40 || true)
  echo "$syms" | tee "$OUT_DIR/nm_collide_${out}.txt" | tee -a "$SUMMARY" || true
  objdump -d "$bin" >"$OUT_DIR/ug_${out}.s" 2>/dev/null || true
  echo "RVV insn counts in full UniformGridBenchmark ($out):" | tee -a "$SUMMARY"
  for i in vle64 vse64 vfadd vfmul vfmacc vfmadd vsetvli; do
    c=$(grep -c "[[:space:]]$i" "$OUT_DIR/ug_${out}.s" 2>/dev/null || echo 0)
    echo "  $i=$c" | tee -a "$SUMMARY"
  done
  # Narrow: sections mentioning collide / SplitPure if present
  if grep -q 'split pure LB sweep (collide)\|CollideSweep\|SplitPureSweep' "$OUT_DIR/ug_${out}.s" 2>/dev/null; then
    :
  fi
  # Prefer disassembling first matching collide-related symbol
  local demangled
  demangled=$(nm -C "$bin" 2>/dev/null | awk '/SplitPureSweep.*collide|CollideSweep.*operator|::collide\(/ {print $NF; exit}')
  if [[ -n "${demangled:-}" ]]; then
    local mangled
    mangled=$(nm "$bin" 2>/dev/null | awk -v d="$demangled" 'index($0,d){print $3; exit}')
    # fallback: pick first SplitPureSweep collide-ish mangled
    if [[ -z "${mangled:-}" ]]; then
      mangled=$(nm "$bin" 2>/dev/null | grep -i collide | head -1 | awk '{print $NF}')
    fi
  fi
  mangled=$(nm "$bin" 2>/dev/null | grep -E 'CollideSweep|SplitPure' | head -1 | awk '{print $NF}' || true)
  if [[ -n "${mangled:-}" ]]; then
    echo "objdump symbol: $mangled" | tee -a "$SUMMARY"
    objdump -d --disassemble="$mangled" "$bin" >"$OUT_DIR/collide_${out}.s" 2>/dev/null || \
      objdump -d "$bin" --disassemble="$mangled" >"$OUT_DIR/collide_${out}.s" 2>/dev/null || true
    if [[ -s "$OUT_DIR/collide_${out}.s" ]]; then
      echo -n "  collide-symbol RVV: " | tee -a "$SUMMARY"
      for i in vle64 vse64 vfadd vfmul vfmacc vsetvli; do
        c=$(grep -c "[[:space:]]$i" "$OUT_DIR/collide_${out}.s" 2>/dev/null || echo 0)
        echo -n "$i=$c " | tee -a "$SUMMARY"
      done
      echo | tee -a "$SUMMARY"
    fi
  fi
}

run_one() {
  local tag=$1 np=$2 mode=$3  # mode: not-fused | fused
  local bdir
  if [[ "$tag" == "gc" ]]; then bdir="$BUILD_GC"; else bdir="$BUILD_GCV"; fi
  local bin="$bdir/$BIN_REL"
  local prm="$PRM_NP1"
  [[ "$np" == "4" ]] && prm="$PRM_NP4"
  local extra=()
  [[ "$mode" == "not-fused" ]] && extra=(--not-fused)
  local label="${tag}_np${np}_${mode}"
  local log="$OUT_DIR/run_${label}.log"
  local wallf="$OUT_DIR/wall_${label}.txt"

  echo "==== RUN $label $(date -Is) ====" | tee -a "$SUMMARY"
  cd /tmp
  set +e
  if [[ "$np" == "1" ]]; then
    /usr/bin/time -f 'WALL %e' -o "$wallf" "$bin" "$prm" "${extra[@]}" \
      >"$log" 2>"$OUT_DIR/run_${label}_stderr.log"
  else
    /usr/bin/time -f 'WALL %e' -o "$wallf" mpirun -np "$np" "$bin" "$prm" "${extra[@]}" \
      >"$log" 2>"$OUT_DIR/run_${label}_stderr.log"
  fi
  local rc=$?
  set -e
  echo "rc=$rc" | tee -a "$SUMMARY"
  cat "$wallf" | tee -a "$SUMMARY"
  # TimingPool / collide / stream lines
  grep -E 'collide|stream|WALL|Timing|MLUPS|Performance|split pure|fused|cells|time step' "$log" \
    | head -80 | tee "$OUT_DIR/timing_${label}.txt" | tee -a "$SUMMARY" || true
  # Also append stderr progress if useful
  grep -E 'collide|stream|TimingPool|WcTiming|min |avg |max ' "$OUT_DIR/run_${label}_stderr.log" \
    | head -60 | tee -a "$OUT_DIR/timing_${label}.txt" | tee -a "$SUMMARY" || true
}

# ---- main ----
if [[ "$SKIP_BUILD" != "1" ]]; then
  reconfigure_benchmarks "$BUILD_GC" gc
  reconfigure_benchmarks "$BUILD_GCV" gcv
  build_one "$BUILD_GC" gc
  build_one "$BUILD_GCV" gcv
  if [[ "$SKIP_OBJDUMP" != "1" ]]; then
    rebuild_gcv_with_vecinfo
    objdump_collide "$BUILD_GC/$BIN_REL" gc
    objdump_collide "$BUILD_GCV/$BIN_REL" gcv
  fi
else
  echo "SKIP_BUILD=1 — using existing binaries" | tee -a "$SUMMARY"
  ls -la "$BUILD_GC/$BIN_REL" "$BUILD_GCV/$BIN_REL" | tee -a "$SUMMARY"
fi

# Primary: not-fused
run_one gc 1 not-fused
run_one gcv 1 not-fused

# Optional fused control
if [[ "$DO_FUSED" == "1" ]]; then
  run_one gc 1 fused
  run_one gcv 1 fused
fi

# Optional np=4
if [[ "$DO_NP4" == "1" ]]; then
  run_one gc 4 not-fused
  run_one gcv 4 not-fused
fi

{
  echo
  echo "======== SUMMARY TABLE $(date -Is) ========"
  echo "label | WALL(s)"
  for f in "$OUT_DIR"/wall_*.txt; do
    [[ -f "$f" ]] || continue
    b=$(basename "$f" .txt | sed 's/^wall_//')
    w=$(grep WALL "$f" | awk '{print $2}')
    echo "$b | $w"
  done
  echo
  echo "TimingPool collide/stream (from logs):"
  for f in "$OUT_DIR"/timing_*.txt; do
    [[ -f "$f" ]] || continue
    echo "--- $(basename "$f") ---"
    grep -E 'collide|stream' "$f" | head -20 || true
  done
} | tee -a "$SUMMARY"

echo "DONE $(date -Is) summary=$SUMMARY" | tee -a "$SUMMARY"
