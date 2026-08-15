#!/usr/bin/env bash
# A/B PETSc MatMult vs hand CSR/stencil SpMV (scalar + RVV) on RV2.
set -euo pipefail

N="${N:-800}"
REPS="${REPS:-20}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUTDIR="${OUTDIR:-$ROOT/results}"
mkdir -p "$OUTDIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUMMARY="$OUTDIR/petsc-spmv-rvv-ab-$STAMP.txt"

export EESSI_VERSION_OVERRIDE="${EESSI_VERSION_OVERRIDE:-2025.06-001}"
export EESSI_USER_INSTALL="${EESSI_USER_INSTALL:-$HOME/eessi-overlay}"
export EESSI_NO_MODULE_PURGE_ON_INIT=1
export PS1="${PS1:-}"
set +u
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load EasyBuild/5.3.1 EESSI-extend/2025.06-easybuild foss/2025b PETSc/3.24.0-foss-2025b
set -u
export LD_LIBRARY_PATH="${EBROOTGCCCORE}/lib64:${EBROOTPETSC}/lib:${EBROOTFLEXIBLAS}/lib:${LD_LIBRARY_PATH:-}"

cd "$ROOT"
make petsc_spmv_rvv_bench

{
  echo "date_utc=$STAMP"
  echo "host=$(hostname)"
  echo "n=$N reps=$REPS"
  echo "petsc=$EBROOTPETSC"
  echo "compiler=$(mpicc -show | head -c 200)"
  echo
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENBLAS_CORETYPE=RISCV64_GENERIC \
    ./petsc_spmv_rvv_bench "$N" "$REPS"
} | tee "$SUMMARY"

echo "SUMMARY=$SUMMARY"
