#!/bin/bash
# MODFLOW 6 FlexiBLAS A/B on Orange Pi RV2 (EESSI riscv 20240402 / foss-2023b).
#
# Parallel mode (`mf6 -p`) activates the PETSc linear solver (FlexiBLAS via
# PETSc). Workload: USGS ex-gwf-lgrv-lgr (~55 s serial IMS baseline; PETSc path
# timed here).
#
# Prerequisites:
#   wget -O ~/mf6-bench/mf6examples.zip \
#     https://github.com/MODFLOW-ORG/modflow6-examples/releases/download/current/mf6examples.zip
#
# Usage: MF6_MODEL=ex-gwf-lgrv-lgr MF6_NP=1 bash run-modflow-blas-ab.sh
set +e
LOG="${LOG:-$HOME/logs/modflow-blas-ab-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "START $(date -Iseconds) log=$LOG"

export LMOD_IGNORE_CACHE=yes
export LMOD_CACHED_LOADS=no
# shellcheck disable=SC1091
source /cvmfs/riscv.eessi.io/versions/20240402/init/bash
module --ignore_cache load MODFLOW/6.4.4-foss-2023b
echo "mf6=$(command -v mf6)"
echo "EBROOTMODFLOW=$EBROOTMODFLOW"
echo "EBROOTFLEXIBLAS=$EBROOTFLEXIBLAS"
echo "EBROOTPETSC=$EBROOTPETSC"

MODEL=${MF6_MODEL:-ex-gwf-lgrv-lgr}
ZIP=${MF6_ZIP:-$HOME/mf6-bench/mf6examples.zip}
WORK=${WORK:-$HOME/mf6-bench/ab}
NP=${MF6_NP:-1}
RVV_LIB=${RVV_LIB:-$HOME/libopenblas_x60_eb_fixed.so}

if [[ ! -f "$ZIP" ]]; then
  echo "FATAL: missing $ZIP"
  exit 1
fi

mkdir -p "$WORK"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

run_one() {
  local tag=$1
  shift
  local dir="$WORK/run-$tag"
  rm -rf "$dir"
  mkdir -p "$dir"
  ( cd "$dir" && unzip -qo "$ZIP" "$MODEL/*" )
  local rundir="$dir/$MODEL"
  # MF6 6.4.4 parallel mode picks PETSc KSP (CG + IMS PC shell). Custom
  # -pc_type lu/mumps options in .petscrc are often left unused; leave empty.
  : >"$rundir/.petscrc"

  echo ""
  echo "========== [$tag] MODEL=$MODEL NP=$NP env: $* =========="
  echo "rundir=$rundir"
  local t0 t1 rc
  t0=$(date +%s.%N)
  (
    cd "$rundir" || exit 99
    # shellcheck disable=SC2086
    env "$@" mpirun -np "$NP" mf6 -p >mf6.stdout 2>mf6.stderr
  )
  rc=$?
  t1=$(date +%s.%N)
  python3 -c "print('WALL_S=%.3f RC=%d tag=%s' % ($t1-$t0, $rc, '$tag'))"
  grep -E "Elapsed run time:|Normal termination|PETSc Linear|PETSc linear|ERROR REPORT" \
    "$rundir/mf6.stdout" 2>/dev/null | head -20
  if [[ -f "$rundir/mfsim.lst" ]]; then
    grep -E "PERCENT DISCREPANCY" "$rundir/mfsim.lst" | tail -4
  fi
  grep -iE "error|Error|FATAL" "$rundir/mf6.stderr" "$rundir/mf6.stdout" 2>/dev/null | head -15
  echo "RC=$rc"
}

# smoke that -p works on this build
echo "---- PETSc mode smoke (twri, optional) ----"

run_one scalar FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC
run_one stock  FLEXIBLAS=OPENBLAS
if [[ -f "$RVV_LIB" ]]; then
  run_one patched FLEXIBLAS="$RVV_LIB"
else
  echo "SKIP patched: no $RVV_LIB"
fi

echo "DONE $(date -Iseconds)"
