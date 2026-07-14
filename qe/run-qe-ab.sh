#!/bin/bash
# run-qe-ab.sh - Quantum ESPRESSO pw.x BLAS-backend A/B (correctness):
#                stock-RVV vs scalar vs patched-RVV, swapped via FlexiBLAS under
#                one unchanged pw.x. Shows whether a buggy vector BLAS breaks a
#                real DFT SCF (NaN / no convergence / abort).
#
# Usage:  RVV_LIB=/path/to/vector/libopenblas.so ./run-qe-ab.sh input.in [np]
# Env:    NP           MPI ranks (positional [np], default 8)
#         RVV_LIB      patched/vector OpenBLAS .so; if unset the [C] case is skipped
#         MPIRUN_FLAGS extra mpirun flags (e.g. --allow-run-as-root when root)
# Requires a loaded QE module providing pw.x + mpirun + FlexiBLAS, e.g.:
#   module load QuantumESPRESSO/7.5-foss-2025b
# pw.x reads pseudopotentials from pseudo_dir in the input's &control.
#
# Per backend it prints: exit code, wall time, whether SCF converged, a suspect
# counter (nan / not-converged / QE error), and the final total energy - so a
# broken vector BLAS shows up as nonzero suspect / missing energy / NaN.
set -u
IN=${1:?usage: RVV_LIB=... run-qe-ab.sh input.in [np]}
NP=${2:-8}
RVV_LIB=${RVV_LIB:-}
command -v pw.x   >/dev/null || { echo "ERROR: pw.x not on PATH (module load QuantumESPRESSO/...)"; exit 1; }
command -v mpirun >/dev/null || { echo "ERROR: mpirun not on PATH"; exit 1; }
[ -f "$IN" ] || { echo "ERROR: input '$IN' not found"; exit 1; }
base=$(basename "$IN" .in)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

run(){
  tag=$1; shift
  out="out.$base.$tag"
  t0=$(date +%s)
  env "$@" mpirun ${MPIRUN_FLAGS:-} --bind-to core -np "$NP" pw.x -in "$IN" > "$out" 2>&1
  rc=$?
  t1=$(date +%s)
  ener=$(grep -aE '^!' "$out" | tail -1)
  conv=$(grep -acE 'convergence has been achieved' "$out")
  bad=$(grep -aciE 'nan|convergence NOT|not converged|stopping|Error in routine' "$out")
  echo "[$tag] rc=$rc time=$((t1-t0))s converged=$conv suspect=$bad  ${ener:-<no-energy-line>}"
}

echo "=== [A] stock CVMFS RVV OpenBLAS (unpatched ZVL256B gemv_n on X60) ==="
run stockRVV FLEXIBLAS=OPENBLAS
echo "=== [B] scalar baseline (same OpenBLAS backend, RISCV64_GENERIC coretype) ==="
run scalar FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC
if [ -n "$RVV_LIB" ]; then
  echo "=== [C] patched x60 RVV (FLEXIBLAS=$RVV_LIB) ==="
  run patchedRVV FLEXIBLAS="$RVV_LIB"
else
  echo "(set RVV_LIB=/path/to/vector/libopenblas.so to also run the patched-RVV case)"
fi
