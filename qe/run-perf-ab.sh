#!/bin/bash
# run-perf-ab.sh <input> [np] - Quantum ESPRESSO pw.x BLAS *performance* A/B:
#   scalar baseline vs vector RVV OpenBLAS, swapped via FlexiBLAS under one
#   unchanged pw.x. Companion to run-qe-ab.sh (which is a *correctness* A/B and
#   also runs the stock-RVV case that NaN-aborts); this one drops the broken
#   case and focuses on wall time on a GEMM-heavy input.
#
# A 2-atom SCF is startup/FFT-bound and shows ~0 BLAS signal; use a Si supercell
# (gen_si_supercell.py) so dense level-3 GEMM (calbec, subspace rotation,
# rdiaghg) is a real fraction of the run. -ndiag 1 pins the subspace
# diagonalization to serial LAPACK->OpenBLAS so the FlexiBLAS swap actually
# affects it (else a linked ELPA/ScaLAPACK would hide it).
#
# Usage:  RVV_LIB=/path/to/vector/libopenblas.so ./run-perf-ab.sh si-super-64.in [np]
# Env:    RVV_LIB       vector OpenBLAS .so (if unset, only the scalar run is done)
#         MPIRUN_FLAGS  extra mpirun flags (e.g. --allow-run-as-root when root)
# Requires a loaded QE module providing pw.x + mpirun + FlexiBLAS, e.g.:
#   module load QuantumESPRESSO/7.5-foss-2025b
set -u
IN=${1:?usage: [RVV_LIB=...] run-perf-ab.sh input.in [np]}
NP=${2:-4}
RVV_LIB=${RVV_LIB:-}
command -v pw.x   >/dev/null || { echo "ERROR: pw.x not on PATH (module load QuantumESPRESSO/...)"; exit 1; }
command -v mpirun >/dev/null || { echo "ERROR: mpirun not on PATH"; exit 1; }
[ -f "$IN" ] || { echo "ERROR: input '$IN' not found"; exit 1; }
base=$(basename "$IN" .in)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

run(){
  tag=$1; shift
  out="out.$base.$tag"
  env "$@" mpirun ${MPIRUN_FLAGS:-} --bind-to core -np "$NP" pw.x -ndiag 1 -in "$IN" > "$out" 2>&1
  rc=$?
  echo "[$tag] rc=$rc  $(grep -aE 'PWSCF' "$out" | grep -aiE 'WALL' | tail -1)"
}

echo "=== QE BLAS perf A/B  input=$IN np=$NP ==="
run scalar FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC
if [ -n "$RVV_LIB" ]; then
  run patched FLEXIBLAS="$RVV_LIB"
else
  echo "(set RVV_LIB=/path/to/vector/libopenblas.so to also run the patched-RVV case)"
fi

echo "--- per-routine timers (BLAS routines speed up, FFT does not) ---"
for t in scalar patched; do
  out="out.$base.$t"
  [ -f "$out" ] && { echo "[$t]"; grep -aE 'PWSCF +:|calbec +:| rdiaghg +:|regterg +:|vloc_psi +:|fftw +:' "$out"; }
done
