#!/bin/bash
# Part A re-verify v2 — fixed make targets + LD_LIBRARY_PATH.
set +e
LOG="${LOG:-$HOME/logs/part-a-v2-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "V2 START $(date -Iseconds) log=$LOG"

export LMOD_IGNORE_CACHE=yes
export EESSI_VERSION_OVERRIDE=2025.06-001
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module --ignore_cache load foss/2025b

REPO=${REPO:-$HOME/part-a-bench}
RVV=${RVV_LIB:-$HOME/libopenblas_x60_eb_fixed.so}
export LD_LIBRARY_PATH="${EBROOTFLEXIBLAS}/lib:${EBROOTFLEXIBLAS}/lib64:${LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

run_dgemm() {
  local tag=$1; shift
  echo "--- DGEMM $tag ---"
  env "$@" "$OB/bench_dgemm" 2048
}

run_elpa() {
  local tag=$1; shift
  echo "--- ELPA $tag ---"
  timeout 180 env OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 "$@" \
    mpirun --bind-to none -np 1 "$EB" 3000
}

run_scalapack() {
  local tag=$1; shift
  echo "--- ScaLAPACK $tag ---"
  timeout 300 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$@" \
    mpirun --bind-to core -np 8 "$SB" 3000 64 2
}

echo "=== OpenBLAS ==="
OB=~/openblas-bench2
mkdir -p "$OB"
cp "$REPO/OpenBLAS/bench_dgemm.c" "$OB/"
gcc -O2 "$OB/bench_dgemm.c" -o "$OB/bench_dgemm" -lflexiblas -L"$EBROOTFLEXIBLAS/lib" -Wl,-rpath,"$EBROOTFLEXIBLAS/lib"
run_dgemm scalar FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC
run_dgemm stock  FLEXIBLAS=OPENBLAS
run_dgemm patched FLEXIBLAS="$RVV"

echo "=== BLIS ==="
cd "$REPO/BLIS"
BLIS_PREFIX=$HOME/blis-install OPENBLAS_LIB=$RVV SIZES="2048" THREADS="1 8" bash run-ab.sh

echo "=== ELPA ==="
module --ignore_cache load ELPA/2025.06.002-foss-2025b
make -C "$REPO/elpa" clean elpa_bench 2>&1 | tail -5
EB="$REPO/elpa/elpa_bench"
chmod +x "$EB"
run_elpa scalar FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC
run_elpa stock  FLEXIBLAS=OPENBLAS
run_elpa patched FLEXIBLAS="$RVV"

echo "=== ScaLAPACK ==="
module --ignore_cache load ScaLAPACK/2.2.2-gompi-2025b-fb
make -C "$REPO/scalapack" clean scalapack_bench CC=mpicc 2>&1 | tail -5
SB="$REPO/scalapack/scalapack_bench"
chmod +x "$SB"
run_scalapack scalar FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC
run_scalapack patched FLEXIBLAS="$RVV"

echo "=== GROMACS ==="
if pgrep -f "gmx.*mdrun" >/dev/null; then
  echo "waiting for prior mdrun..."
  while pgrep -f "gmx.*mdrun" >/dev/null; do sleep 30; done
fi
bash "$REPO/gromacs/run-gmx-fft-ab.sh" v2 2>&1 | tail -40

echo "=== IME ==="
taskset -c 0 "$HOME/ime-bench/ime-bench" 512 512 512

echo "=== FFTW (default planner medians) ==="
for s in 256 4096 65536; do
  echo "size $s:"
  grep "size=$s " "$HOME/fftw-proper.log" 2>/dev/null | grep default
done

echo "V2 DONE $(date -Iseconds)"
