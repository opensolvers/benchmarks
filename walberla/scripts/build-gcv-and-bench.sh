#!/usr/bin/env bash
set -eo pipefail
export EESSI_VERSION_OVERRIDE=2025.06-001
export EESSI_USER_INSTALL=$HOME/eessi-overlay
export EESSI_NO_MODULE_PURGE_ON_INIT=1
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load EasyBuild/5.3.1 EESSI-extend/2025.06-easybuild foss/2025b
module load CMake/4.0.3 Boost.MPI/1.88.0-gompi-2025b
export LD_LIBRARY_PATH="$EBROOTGCCCORE/lib64:${LD_LIBRARY_PATH:-}"

SRC=$HOME/walberla-bench/src/walberla-7.2
OUT=$HOME/walberla-bench/results
PRM=$HOME/walberla-bench/prm
STOCK=/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software/waLBerla/7.2-foss-2025b/build/apps/tutorials/lbm/01_BasicLBM
SUMMARY=$OUT/walberla-ab-summary.txt
mkdir -p "$OUT"
{
  echo "======== waLBerla A/B $(date -Is) (stock-gc vs local-gcv) ========"
  echo "board: $(uname -a)"
  lscpu | grep -E 'Model name|CPU\(s\)|MHz' | head -10
  echo "stock ISA:"; readelf -A "$STOCK" | head -8
} | tee "$SUMMARY"

bdir=$HOME/walberla-bench/build-gcv
rm -rf "$bdir"; mkdir -p "$bdir"; cd "$bdir"
echo "==== CONFIGURE gcv $(date -Is) ====" | tee -a "$SUMMARY"
cmake "$SRC" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O2 -ftree-vectorize -march=rv64gcv -mabi=lp64d -fno-math-errno -fPIC" \
  -DCMAKE_C_FLAGS="-O2 -ftree-vectorize -march=rv64gcv -mabi=lp64d -fno-math-errno -fPIC" \
  -DWALBERLA_BUILD_WITH_MPI=ON \
  -DWALBERLA_BUILD_WITH_OPENMP=OFF \
  -DWALBERLA_BUILD_WITH_PYTHON=OFF \
  -DWALBERLA_BUILD_SHOWCASES=OFF \
  -DWALBERLA_BUILD_DOC=OFF \
  -DWALBERLA_BUILD_BENCHMARKS=OFF \
  -DWALBERLA_BUILD_TESTS=OFF \
  -DWALBERLA_BUILD_TOOLS=OFF \
  -DWALBERLA_BUILD_TUTORIALS=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
echo "==== BUILD gcv $(date -Is) ====" | tee -a "$SUMMARY"
cmake --build . -j4 --target 01_BasicLBM
BIN=$bdir/apps/tutorials/lbm/01_BasicLBM
ls -la "$BIN"
echo "gcv ISA:" | tee -a "$SUMMARY"
readelf -A "$BIN" | tee -a "$SUMMARY" | head -8

run_bin() {
  local tag=$1 bin=$2 np=$3 prm=$4
  local log=$OUT/run-${tag}-np${np}.log
  local work=$OUT/work-${tag}-np${np}
  rm -rf "$work"; mkdir -p "$work"
  cp "$prm" "$work/bench.prm"
  echo "==== RUN $tag np=$np $(date -Is) ====" | tee -a "$SUMMARY"
  cd "$work"
  local t0 t1; t0=$(date +%s)
  if [ "$np" = 1 ]; then
    "$bin" bench.prm >"$log" 2>&1 || { echo FAILED $tag; tail -50 "$log" | tee -a "$SUMMARY"; return 1; }
  else
    mpirun -np "$np" "$bin" bench.prm >"$log" 2>&1 || { echo FAILED $tag; tail -50 "$log" | tee -a "$SUMMARY"; return 1; }
  fi
  t1=$(date +%s)
  echo "WALL $((t1-t0))" | tee -a "$SUMMARY"
  grep -iE 'MLUPS|MFLUPS|Performance|Cells|timestep|finished|Duration|remaining' "$log" | tee -a "$SUMMARY" | tail -40
  tail -15 "$log" | tee -a "$SUMMARY"
}

run_bin stock-gc "$STOCK" 1 $PRM/01_BasicLBM_bench.prm
run_bin gcv "$BIN" 1 $PRM/01_BasicLBM_bench.prm
run_bin stock-gc "$STOCK" 4 $PRM/01_BasicLBM_bench_mpi4.prm
run_bin gcv "$BIN" 4 $PRM/01_BasicLBM_bench_mpi4.prm

echo "======== waLBerla A/B DONE $(date -Is) ========" | tee -a "$SUMMARY"
