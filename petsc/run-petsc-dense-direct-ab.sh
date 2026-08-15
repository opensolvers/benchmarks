#!/usr/bin/env bash
# FlexiBLAS A/B for denser PETSc probes: dense MatMult/CG + sparse-direct LU.
set -euo pipefail

THREADS="${THREADS:-8}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUTDIR="${OUTDIR:-$ROOT/results}"
mkdir -p "$OUTDIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUMMARY="$OUTDIR/petsc-dense-direct-ab-$STAMP.txt"

DENSE_N="${DENSE_N:-2048}"
DENSE_CG_N="${DENSE_CG_N:-1024}"
MUMPS_2D_N="${MUMPS_2D_N:-200}"
MUMPS_3D_N="${MUMPS_3D_N:-40}"
SUPERLU_2D_N="${SUPERLU_2D_N:-200}"
UMFPACK_2D_N="${UMFPACK_2D_N:-200}"
REPS="${REPS:-3}"

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
make clean
make all

PATCHED=""
for cand in \
  "$HOME/libopenblas_x60_eb_fixed.so" \
  "$HOME/ob-rvv/libopenblas.so" \
  "$HOME/libopenblas_x60_eb.so" \
  "$HOME/eessi-x60/versions/2025.06-001/software/linux/riscv64/generic/software/OpenBLAS/0.3.30-GCC-14.3.0-x60/lib/libopenblas.so"
do
  if [[ -e "$cand" ]]; then PATCHED="$cand"; break; fi
done

STOCK_OB="${EBROOTOPENBLAS:-}/lib/libopenblas.so"
if [[ ! -e "$STOCK_OB" ]]; then
  STOCK_OB="$(find /cvmfs/dev.eessi.io/riscv -path '*/OpenBLAS/*/lib/libopenblas.so' 2>/dev/null | head -1 || true)"
fi

{
  echo "date_utc=$STAMP"
  echo "host=$(hostname)"
  echo "threads=$THREADS reps=$REPS"
  echo "dense_n=$DENSE_N dense_cg_n=$DENSE_CG_N"
  echo "mumps_2d_n=$MUMPS_2D_N mumps_3d_n=$MUMPS_3D_N"
  echo "superlu_2d_n=$SUPERLU_2D_N umfpack_2d_n=$UMFPACK_2D_N"
  echo "petsc=$EBROOTPETSC"
  echo "stock_openblas=$STOCK_OB"
  echo "patched_openblas=$PATCHED"
  echo "ldd_petsc:"
  ldd "$EBROOTPETSC/lib/libpetsc.so" | grep -Ei 'mumps|superlu|umfpack|cholmod|flexi|openblas|blas' || true
  echo
} | tee "$SUMMARY"

# run_case LABEL ENVVAR=VAL ... -- BIN ARG...
run_case() {
  local label="$1"; shift
  local logfile="$OUTDIR/run_${label}_$STAMP.log"
  local env_args=()
  local cmd_args=()
  local seen_sep=0
  for a in "$@"; do
    if [[ "$a" == "--" ]]; then
      seen_sep=1
      continue
    fi
    if [[ $seen_sep -eq 0 ]]; then
      env_args+=("$a")
    else
      cmd_args+=("$a")
    fi
  done
  echo "==== $label ====" | tee -a "$SUMMARY"
  echo "env: ${env_args[*]}" | tee -a "$SUMMARY"
  echo "cmd: ${cmd_args[*]}" | tee -a "$SUMMARY"
  set +e
  env "${env_args[@]}" "${cmd_args[@]}" >"$logfile" 2>"$logfile.err"
  local rc=$?
  set -e
  cat "$logfile" | tee -a "$SUMMARY"
  if [[ -s "$logfile.err" ]]; then
    echo "--- stderr ---" | tee -a "$SUMMARY"
    tail -50 "$logfile.err" | tee -a "$SUMMARY"
  fi
  if [[ $rc -ne 0 ]]; then
    echo "FAILED rc=$rc" | tee -a "$SUMMARY"
  fi
  echo | tee -a "$SUMMARY"
}

COMMON=(OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS")

run_backend() {
  local blabel="$1"; shift
  # remaining are extra env assignments
  local extra=("$@")

  run_case "dense_mult_${blabel}" \
    "${COMMON[@]}" "${extra[@]}" -- \
    "$ROOT/petsc_dense_bench" "$DENSE_N" "$REPS" mult

  run_case "dense_cg_${blabel}" \
    "${COMMON[@]}" "${extra[@]}" -- \
    "$ROOT/petsc_dense_bench" "$DENSE_CG_N" "$REPS" cg

  run_case "mumps_2d_${blabel}" \
    "${COMMON[@]}" "${extra[@]}" -- \
    "$ROOT/petsc_direct_bench" "$MUMPS_2D_N" "$REPS" mumps 2

  run_case "mumps_3d_${blabel}" \
    "${COMMON[@]}" "${extra[@]}" -- \
    "$ROOT/petsc_direct_bench" "$MUMPS_3D_N" "$REPS" mumps 3

  run_case "superlu_2d_${blabel}" \
    "${COMMON[@]}" "${extra[@]}" -- \
    "$ROOT/petsc_direct_bench" "$SUPERLU_2D_N" "$REPS" superlu_dist 2

  run_case "umfpack_2d_${blabel}" \
    "${COMMON[@]}" "${extra[@]}" -- \
    "$ROOT/petsc_direct_bench" "$UMFPACK_2D_N" "$REPS" umfpack 2
}

run_backend scalar OPENBLAS_CORETYPE=RISCV64_GENERIC
if [[ -n "$STOCK_OB" && -e "$STOCK_OB" ]]; then
  run_backend stock_rvv "FLEXIBLAS=$STOCK_OB"
fi
if [[ -n "$PATCHED" ]]; then
  run_backend patched_rvv "FLEXIBLAS=$PATCHED"
fi

echo "SUMMARY=$SUMMARY"
