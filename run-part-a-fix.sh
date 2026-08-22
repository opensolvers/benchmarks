#!/bin/bash
# Fixed Part A re-verify (single shell, no nested module init).
set +e
LOG="${LOG:-$HOME/logs/part-a-fix-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "FIX START $(date -Iseconds) log=$LOG"

export LMOD_IGNORE_CACHE=yes
export EESSI_VERSION_OVERRIDE=2025.06-001
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module --ignore_cache load foss/2025b

REPO=${REPO:-$HOME/part-a-bench}
RVV=${RVV_LIB:-$HOME/libopenblas_x60_eb_fixed.so}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

run_dgemm() {
  local tag=$1; shift
  echo "--- DGEMM $tag ---"
  env "$@" "$OB/bench_dgemm" 2048
}

echo "=== OpenBLAS ==="
OB=~/openblas-bench2
mkdir -p "$OB"
cp "$REPO/OpenBLAS/bench_dgemm.c" "$REPO/OpenBLAS/difftest.c" "$OB/"
gcc -O2 "$OB/bench_dgemm.c" -o "$OB/bench_dgemm" -lflexiblas
gcc -O2 "$OB/difftest.c" -o "$OB/difftest" -ldl -lm
STOCK=$(find /cvmfs/dev.eessi.io/riscv -path '*/OpenBLAS/*/lib/libopenblas.so' 2>/dev/null | head -1)
echo "STOCK=$STOCK"
run_dgemm scalar FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC
run_dgemm stock  FLEXIBLAS=OPENBLAS
run_dgemm patched FLEXIBLAS="$RVV"
echo "--- difftest stock ---"
"$OB/difftest" "$STOCK" | grep -E 'dgemv|dgemm|nan'
echo "--- difftest patched ---"
"$OB/difftest" "$RVV" | grep -E 'dgemv|dgemm|nan'

echo "=== BLIS ==="
cd "$REPO/BLIS"
BLIS_PREFIX=$HOME/blis-install OPENBLAS_LIB=$RVV SIZES="2048" THREADS="1 8" bash run-ab.sh

echo "=== ELPA ==="
module --ignore_cache load ELPA/2025.06.002-foss-2025b
make -C "$REPO/elpa" clean all 2>&1 | tail -3
EB="$REPO/elpa/elpa_bench"
run_elpa() {
  local tag=$1; shift
  echo "--- ELPA $tag ---"
  timeout 180 env OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 "$@" \
    mpirun --bind-to none -np 1 "$EB" 3000
}
run_elpa scalar FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC
run_elpa stock  FLEXIBLAS=OPENBLAS
run_elpa patched FLEXIBLAS="$RVV"

echo "=== ScaLAPACK ==="
module --ignore_cache load ScaLAPACK/2.2.2-gompi-2025b-fb
make -C "$REPO/scalapack" clean all CC=mpicc 2>&1 | tail -3
SB="$REPO/scalapack/scalapack_bench"
run_scalapack() {
  local tag=$1; shift
  echo "--- ScaLAPACK $tag ---"
  timeout 300 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$@" \
    mpirun --bind-to core -np 8 "$SB" 3000 64 2
}
run_scalapack scalar FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC
run_scalapack patched FLEXIBLAS="$RVV"

echo "=== GROMACS ==="
bash "$REPO/gromacs/run-gmx-fft-ab.sh" part-a-fix 2>&1 | tail -30

echo "=== IME ==="
if [[ -x "$HOME/ime-bench/ime-bench" ]]; then
  taskset -c 0 "$HOME/ime-bench/ime-bench" 512 512 512
else
  echo "SKIP ime-bench binary"
fi

echo "=== FFTW summary (from running sweep) ==="
if [[ -f "$HOME/fftw-proper.log" ]]; then
  grep -E 'size=(256|4096|65536).*med=' "$HOME/fftw-proper.log" | tail -20
fi

echo "FIX DONE $(date -Iseconds)"
