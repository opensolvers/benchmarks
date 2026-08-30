#!/bin/bash
# QE FFT wisdom A/B on r5v libfftw3 (serial or MPI):
#   1) stock ESTIMATE (baseline)
#   2) ESTIMATE->$PLAN collect + export wisdom (slow planning once)
#   3) ESTIMATE + imported wisdom only (fast) via wisdom-preload
#   4) ESTIMATE->$PLAN + imported wisdom (warm planner)
#
# Usage: bash run-qe-fft-wisdom-ab.sh [input.in] [label]
# Env:   NP=1|4|...                 MPI ranks (default 1 = serial pw.x)
#        FFTW_EST2MEAS_FLAGS=measure|patient|exhaustive  (default: measure)
#        SKIP_ESTIMATE=1            skip step 1
#        PW=...                     override pw.x (serial default; MPI uses overlay)
#        MPIRUN_FLAGS=...           extra mpirun flags
set -euo pipefail
IN=${1:-si-super-64.in}
LABEL=${2:-wisdom}
PLAN=${FFTW_EST2MEAS_FLAGS:-measure}
NP=${NP:-1}
BENCH=${BENCH:-$HOME/qe-bench}
R5V=${R5V:-$HOME/fftwbuild/src-r5v/.libs/libfftw3.so.3.6.10}
SRC=${SRC:-$HOME/fftwbuild/src-r5v}
HERE=$(cd "$(dirname "$0")" && pwd)
WISDIR=${WISDIR:-$HOME/fftw-wisdom}
mkdir -p "$WISDIR" "$HOME/logs"

EESSI_SW=/cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/software
QE_MPI=${QE_MPI:-$HOME/eessi-overlay/versions/2025.06-001/software/linux/riscv64/generic/software/QuantumESPRESSO/7.5-foss-2025b}
OMP_ROOT=$EESSI_SW/OpenMPI/5.0.8-GCC-14.3.0
GCC_ROOT=$EESSI_SW/GCCcore/14.3.0

setup_serial_env() {
  set +e
  set +u
  # shellcheck disable=SC1091
  source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
  module use /cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/modules/all
  module --ignore_cache load GCC/14.3.0 2>/dev/null
  module --ignore_cache load FFTW/3.3.10-GCC-14.3.0 2>/dev/null
  module --ignore_cache load FlexiBLAS/3.4.5-GCC-14.3.0 2>/dev/null
  set -u
  set -e
  export LD_LIBRARY_PATH=$SRC/.libs:${EBROOTFFTW:+$EBROOTFFTW/lib:}${EBROOTFLEXIBLAS:+$EBROOTFLEXIBLAS/lib:}${EBROOTOPENBLAS:+$EBROOTOPENBLAS/lib:}${EBROOTGCCCORE:+$EBROOTGCCCORE/lib64:}${LD_LIBRARY_PATH:-}
  PW=${PW:-$HOME/qe-serial/pw.x}
}

setup_mpi_env() {
  # EESSI lmod arch check is flaky on this board — wire foss deps by path.
  # Put r5v FIRST so NEEDED libfftw3.so.3 resolves to one copy (not stock+preload).
  local libs=(
    "$SRC/.libs"
    "$OMP_ROOT/lib64"
    "$EESSI_SW/FlexiBLAS/3.4.5-GCC-14.3.0/lib64"
    "$EESSI_SW/FFTW.MPI/3.3.10-gompi-2025b/lib64"
    "$EESSI_SW/FFTW/3.3.10-GCC-14.3.0/lib64"
    "$EESSI_SW/ScaLAPACK/2.2.2-gompi-2025b-fb/lib64"
    "$EESSI_SW/HDF5/1.14.6-gompi-2025b/lib64"
    "$EESSI_SW/ELPA/2025.06.002-foss-2025b/lib64"
    "$EESSI_SW/libxc/7.0.0-GCC-14.3.0/lib"
    "$EESSI_SW/hwloc/2.12.1-GCCcore-14.3.0/lib64"
    "$EESSI_SW/libevent/2.1.12-GCCcore-14.3.0/lib64"
    "$EESSI_SW/libfabric/2.1.0-GCCcore-14.3.0/lib64"
    "$EESSI_SW/numactl/2.0.19-GCCcore-14.3.0/lib64"
    "$EESSI_SW/PMIx/5.0.8-GCCcore-14.3.0/lib64"
    "$EESSI_SW/UCX/1.19.0-GCCcore-14.3.0/lib64"
    "$EESSI_SW/UCC/1.4.4-GCCcore-14.3.0/lib64"
    "$EESSI_SW/libaec/1.1.4-GCCcore-14.3.0/lib64"
    "$GCC_ROOT/lib64"
  )
  local IFS=:
  export LD_LIBRARY_PATH="${libs[*]}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PATH=$OMP_ROOT/bin:$QE_MPI/bin:$GCC_ROOT/bin:$PATH
  PW=${PW:-$QE_MPI/bin/pw.x}
  MPIRUN=${MPIRUN:-$OMP_ROOT/bin/mpirun}
  command -v "$MPIRUN" >/dev/null || MPIRUN=$OMP_ROOT/bin/mpirun
}

if [ "$NP" -gt 1 ]; then
  setup_mpi_env
else
  setup_serial_env
fi

export FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export FFTW_EST2MEAS_FLAGS=$PLAN
export FFTW_R5V_SO=$R5V

GCC14BIN=$GCC_ROOT/bin
if [ -x "$GCC14BIN/gcc" ]; then
  export PATH=$GCC14BIN:$PATH
  export LIBRARY_PATH=$GCC_ROOT/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}
  export LD_LIBRARY_PATH=$GCC_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
CC=${CC:-gcc}

[ -f "$R5V" ] && [ -x "$PW" ] || { echo "missing r5v or pw.x ($PW)"; exit 1; }
INC=$SRC/api
$CC -O2 -fPIC -shared -I"$INC" "$HERE/fftw-est2meas-interposer.c" \
  -o "$WISDIR/libfftw-est2meas.so" -ldl
$CC -O2 -fPIC -shared -I"$INC" "$HERE/fftw-wisdom-preload.c" \
  -o "$WISDIR/libfftw-wisdom-preload.so" -ldl
$CC -O2 -I"$INC" "$HERE/merge-fftw-wisdom.c" -o "$WISDIR/merge-fftw-wisdom" \
  -L"$SRC/.libs" -lfftw3 -Wl,-rpath,"$SRC/.libs"
echo "built interposers with $($CC --version | head -1)"
echo "NP=$NP PW=$PW plan=$PLAN"
WISFILE=$WISDIR/wisdom-from-qe-${LABEL}.fftw
LOG=$HOME/logs/qe-fft-wisdom-${LABEL}-$(date +%Y%m%d-%H%M%S).log

run_pw() {
  local preload=$1 out=$2
  if [ "$NP" -gt 1 ]; then
    env LD_PRELOAD="$preload" \
        FFTW_WISDOM_FILE="${FFTW_WISDOM_FILE:-}" \
        FFTW_WISDOM_OUT="${FFTW_WISDOM_OUT:-}" \
        FFTW_EST2MEAS_FLAGS="$PLAN" \
        FFTW_R5V_SO="$R5V" \
        FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC \
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "$MPIRUN" ${MPIRUN_FLAGS:-} --bind-to core -np "$NP" \
        "$PW" -ndiag 1 -in "$IN" > "$out" 2> "$out.err"
  else
    env LD_PRELOAD="$preload" \
        FFTW_WISDOM_FILE="${FFTW_WISDOM_FILE:-}" \
        FFTW_WISDOM_OUT="${FFTW_WISDOM_OUT:-}" \
        FFTW_EST2MEAS_FLAGS="$PLAN" \
        FFTW_R5V_SO="$R5V" \
        "$PW" -in "$IN" > "$out" 2> "$out.err"
  fi
}

# Always preload r5v so execute/plan symbols win over RPATH stock FFTW.
preload_est2meas() { echo "$WISDIR/libfftw-est2meas.so:$R5V"; }
preload_wisdom() { echo "$WISDIR/libfftw-wisdom-preload.so:$R5V"; }
preload_r5v_only() { echo "$R5V"; }

run_one() {
  local tag=$1 preload=$2 out=$3
  echo "===== [$tag] ====="
  rm -rf "$BENCH/out"
  cd "$BENCH"
  run_pw "$preload" "$out" || {
      echo "FAILED $tag"; tail -40 "$out.err" "$out"; return 1; }
  local etot fftw pwscf init_run
  etot=$(grep -E '! *total energy' "$out" | tail -1 | awk '{print $(NF-1)}')
  wall_field() {
    grep -E "^ *$1 " "$out" | tail -1 | awk '{
      for (i=1;i<=NF;i++) if ($i=="WALL") {
        if ($(i-1) ~ /s$/ && i>2 && $(i-2) ~ /m$/) print $(i-2), $(i-1), "WALL";
        else if ($(i-1) ~ /s$/) print $(i-1), "WALL";
      }
    }'
  }
  fftw=$(wall_field fftw)
  pwscf=$(wall_field PWSCF)
  init_run=$(wall_field init_run)
  echo "  energy=${etot:-?}  fftw=${fftw:-?}  init_run=${init_run:-?}  PWSCF=${pwscf:-?}"
  grep -E 'fftw-est2meas|fftw-wisdom-preload' "$out.err" | head -20 || true
}

{
  echo "################ QE FFT wisdom A/B : $IN ($LABEL) plan=$PLAN NP=$NP ################"
  echo "r5v=$R5V"
  echo "pw=$PW"
  echo
  unset FFTW_WISDOM_FILE FFTW_WISDOM_OUT
  if [ "${SKIP_ESTIMATE:-0}" != "1" ]; then
    run_one "1-estimate" "$(preload_r5v_only)" "$BENCH/qe-fft-${LABEL}-1-estimate.out"
    echo
  else
    echo "===== [1-estimate] SKIPPED (SKIP_ESTIMATE=1) ====="
    echo
  fi
  export FFTW_WISDOM_OUT=$WISFILE
  unset FFTW_WISDOM_FILE
  rm -f "$WISFILE" "$WISFILE".rank*
  run_one "2-${PLAN}-collect" "$(preload_est2meas)" \
    "$BENCH/qe-fft-${LABEL}-2-collect.out"
  unset FFTW_WISDOM_OUT
  # MPI: merge per-rank wisdom into one file (serial process, no race)
  shopt -s nullglob
  ranks=( "$WISFILE".rank* )
  shopt -u nullglob
  if [ ${#ranks[@]} -gt 0 ]; then
    "$WISDIR/merge-fftw-wisdom" "$WISFILE" "${ranks[@]}" || {
      echo "WARN: wisdom merge failed"; ls -la "$WISFILE".rank* || true; }
  elif [ ! -f "$WISFILE" ]; then
    echo "WARN: no wisdom file produced"
  fi
  echo "  collected wisdom: $(wc -c < "$WISFILE" 2>/dev/null || echo 0) bytes"
  echo
  export FFTW_WISDOM_FILE=$WISFILE
  run_one "3-estimate+wisdom" "$(preload_wisdom)" \
    "$BENCH/qe-fft-${LABEL}-3-wisdom.out"
  echo
  run_one "4-${PLAN}+wisdom" "$(preload_est2meas)" \
    "$BENCH/qe-fft-${LABEL}-4-planwis.out"
  echo "DONE"
} 2>&1 | tee "$LOG"
echo "LOG=$LOG"
