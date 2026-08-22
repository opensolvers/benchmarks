#!/bin/bash
# Part A end-to-end re-verify on Orange Pi RV2 — runs all open harnesses.
set +e
LOG="${LOG:-$HOME/logs/part-a-verify-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "PART-A START $(date -Iseconds) log=$LOG"

export LMOD_IGNORE_CACHE=yes
export EESSI_VERSION_OVERRIDE=2025.06-001
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash

RVV_LIB=${RVV_LIB:-$HOME/libopenblas_x60_eb_fixed.so}
REPO=${REPO:-$HOME/part-a-bench}

run_section() {
  echo ""
  echo "######################################################################"
  echo "### $1"
  echo "######################################################################"
}

# --- OpenBLAS ---
run_section "OpenBLAS"
if [[ -x "$REPO/OpenBLAS/run-openblas-ab.sh" ]]; then
  OB_SRC="$REPO/OpenBLAS" bash "$REPO/OpenBLAS/run-openblas-ab.sh"
else
  echo "SKIP OpenBLAS harness missing"
fi

# --- BLIS vs OpenBLAS (link A/B, N=2048 only) ---
run_section "BLIS"
if [[ -f "$REPO/BLIS/run-ab.sh" && -f "$HOME/blis-install/lib/libblis.a" ]]; then
  BLIS_PREFIX=$HOME/blis-install \
  OPENBLAS_LIB=${OPENBLAS_LIB:-$RVV_LIB} \
  SIZES="2048" THREADS="1 8" \
  bash "$REPO/BLIS/run-ab.sh"
else
  echo "SKIP BLIS"
fi

# --- FFTW r5v vs scalar (subset sizes, no patient) ---
run_section "FFTW"
if [[ -f "$REPO/fftw/bench-fftw-ab.sh" && -d "$HOME/fftwbuild/src-r5v" ]]; then
  SIZES="256 4096 65536" bash "$REPO/fftw/bench-fftw-ab.sh"
  tail -40 "$HOME/fftw-proper.log"
else
  echo "SKIP FFTW"
fi

# --- ELPA ---
run_section "ELPA"
if [[ -f "$REPO/elpa/Makefile" ]]; then
  module --ignore_cache load ELPA/2025.06.002-foss-2025b
  make -C "$REPO/elpa" clean all 2>&1 | tail -5
  EB="$REPO/elpa/elpa_bench"
  for tag envs in \
    "scalar FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC" \
    "stock FLEXIBLAS=OPENBLAS" \
    "patched FLEXIBLAS=$RVV_LIB"; do
    set -- $tag $envs; t=$1; shift
    echo "--- ELPA [$t] ---"
    env OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 "$@" \
      mpirun --bind-to none -np 1 "$EB" 3000
  done
else
  echo "SKIP ELPA"
fi

# --- ScaLAPACK ---
run_section "ScaLAPACK"
if [[ -f "$REPO/scalapack/Makefile" ]]; then
  module --ignore_cache load ScaLAPACK/2.2.2-gompi-2025b-fb
  make -C "$REPO/scalapack" clean all CC=mpicc 2>&1 | tail -5
  SB="$REPO/scalapack/scalapack_bench"
  for tag envs in \
    "scalar FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC" \
    "patched FLEXIBLAS=$RVV_LIB"; do
    set -- $tag $envs; t=$1; shift
    echo "--- ScaLAPACK [$t] ---"
    timeout 300 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$@" \
      mpirun --bind-to core -np 8 "$SB" 3000 64 2
  done
else
  echo "SKIP ScaLAPACK"
fi

# --- GROMACS FFT ---
run_section "GROMACS"
if [[ -x "$REPO/gromacs/run-gmx-fft-ab.sh" && -f "$HOME/gmx-bench/md.tpr" ]]; then
  bash "$REPO/gromacs/run-gmx-fft-ab.sh" part-a 2>&1 | tail -40
else
  echo "SKIP GROMACS"
fi

# --- IME ---
run_section "IME"
if [[ -x "$HOME/ime-bench/ime-bench" ]]; then
  taskset -c 0 "$HOME/ime-bench/ime-bench" 512 512 512
elif [[ -f "$REPO/ime/Makefile" ]]; then
  make -C "$REPO/ime" board 2>&1 | tail -10
  taskset -c 0 "$REPO/ime/ime-bench" 512 512 512
else
  echo "SKIP IME"
fi

echo "PART-A DONE $(date -Iseconds)"
