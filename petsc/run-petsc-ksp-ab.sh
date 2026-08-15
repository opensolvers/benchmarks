#!/usr/bin/env bash
# FlexiBLAS A/B for petsc_ksp_bench on Orange Pi RV2 (SpaceMiT X60).
set -euo pipefail

N="${N:-400}"
REPS="${REPS:-3}"
THREADS="${THREADS:-8}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUTDIR="${OUTDIR:-$ROOT/results}"
mkdir -p "$OUTDIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUMMARY="$OUTDIR/petsc-ksp-flexiblas-ab-$STAMP.txt"

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
make

BIN="$ROOT/petsc_ksp_bench"
test -x "$BIN"

# Prefer patched RVV OpenBLAS if present (gemv_n fix).
PATCHED=""
for cand in \
  "$HOME/libopenblas_x60_eb_fixed.so" \
  "$HOME/ob-rvv/libopenblas.so" \
  "$HOME/libopenblas_x60_eb.so" \
  "$HOME/eessi-x60/versions/2025.06-001/software/linux/riscv64/generic/software/OpenBLAS/0.3.30-GCC-14.3.0-x60/lib/libopenblas.so" \
  "$HOME/libopenblas_tuned.so"
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
  echo "n=$N reps=$REPS threads=$THREADS"
  echo "petsc=$EBROOTPETSC"
  echo "flexiblas=$EBROOTFLEXIBLAS"
  echo "stock_openblas=$STOCK_OB"
  echo "patched_openblas=$PATCHED"
  echo "ldd_petsc_blas:"
  ldd "$EBROOTPETSC/lib/libpetsc.so" | grep -Ei 'blas|flexi|openblas|lapack' || true
  echo
} | tee "$SUMMARY"

run_one() {
  local label="$1"; shift
  local logfile="$OUTDIR/run_${label}_$STAMP.log"
  echo "==== $label ====" | tee -a "$SUMMARY"
  echo "cmd: $*" | tee -a "$SUMMARY"
  set +e
  env "$@" "$BIN" "$N" "$REPS" >"$logfile" 2>"$logfile.err"
  local rc=$?
  set -e
  cat "$logfile" | tee -a "$SUMMARY"
  if [[ -s "$logfile.err" ]]; then
    echo "--- stderr ---" | tee -a "$SUMMARY"
    cat "$logfile.err" | tee -a "$SUMMARY"
  fi
  if [[ $rc -ne 0 ]]; then
    echo "FAILED rc=$rc" | tee -a "$SUMMARY"
  fi
  echo | tee -a "$SUMMARY"
}

COMMON=(OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS")

# 1) Scalar baseline (same stock .so, force generic kernels)
run_one scalar "${COMMON[@]}" OPENBLAS_CORETYPE=RISCV64_GENERIC

# 2) Stock default dispatch (often ZVL256B RVV on X60; may NaN/hang)
if [[ -n "$STOCK_OB" && -e "$STOCK_OB" ]]; then
  run_one stock_rvv "${COMMON[@]}" FLEXIBLAS="$STOCK_OB"
else
  run_one stock_default "${COMMON[@]}"
fi

# 3) Patched RVV if available
if [[ -n "$PATCHED" ]]; then
  run_one patched_rvv "${COMMON[@]}" FLEXIBLAS="$PATCHED"
else
  echo "==== patched_rvv SKIPPED (no patched lib found) ====" | tee -a "$SUMMARY"
fi

echo "SUMMARY=$SUMMARY"
