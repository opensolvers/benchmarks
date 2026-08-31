#!/bin/bash
# Opt v3: cao retune (#4), k-space loop patch (#5), FFTW RVV A/B (#6).
set +e
LOG="${LOG:-$HOME/logs/espresso-opt-v3.log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "START $(date -Iseconds)"

HERE=$(cd "$(dirname "$0")" && pwd)
OPT=${OPT:-$HOME/espresso-opt}
# shellcheck disable=SC1090
[ -f "$HOME/espresso-opt-env.sh" ] && . "$HOME/espresso-opt-env.sh"
export PATH="$OPT/bin:$PATH"
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
export PYTHONPATH="$OPT/lib/python${PYVER}/site-packages:${PYTHONPATH:-}"
SCALAR=${SCALAR:-$HOME/fftwbuild/src-scalar/.libs/libfftw3.so.3.6.10}
R5V=${R5V:-$HOME/fftwbuild/libfftw3-r5v-xorconj.so}
[ -f "$R5V" ] || R5V=$HOME/fftwbuild/src-r5v/.libs/libfftw3.so.3.6.10
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENBLAS_CORETYPE=RISCV64_GENERIC
export ESP_BOX=24 ESP_DENSITY=0.4 ESP_STEPS=${ESP_STEPS:-200} ESP_ACCURACY=1e-3 ESP_SKIN=0.4
export LD_PRELOAD="$SCALAR"

bench_one() {
  local tag=$1
  local fftw=$2
  echo ""
  echo "========== [$tag] LD_PRELOAD=$(basename "$fftw") cao=$ESP_CAO mesh=$ESP_MESH =========="
  LD_PRELOAD="$fftw" "$OPT/bin/pypresso" "$HERE/mpi_dense_large.py"
}

echo "=== #4 cao tune (one pypresso per cao) ==="
BEST_WALL=1e99
BEST_PIN=""
for cao in 4 5 6; do
  echo "--- tune cao=$cao ---"
  out=$(ESP_CAO=$cao "$OPT/bin/pypresso" "$HERE/p3m_cao_tune.py" 2>&1)
  echo "$out"
  pin=$(echo "$out" | grep "^PIN " | tail -1)
  wall=$(echo "$pin" | sed -n 's/.*wall=\([0-9.]*\).*/\1/p')
  if [ -n "$wall" ]; then
    better=$(awk -v w="$wall" -v b="$BEST_WALL" 'BEGIN{print (w+0<b+0)?1:0}')
    if [ "$better" = 1 ]; then
      BEST_WALL=$wall
      BEST_PIN=$pin
    fi
  fi
done

if [ -z "$BEST_PIN" ]; then
  echo "CAO_TUNE_FAIL"
  exit 1
fi
echo "BEST $BEST_PIN"
ESP_CAO=$(echo "$BEST_PIN" | sed -n 's/.*cao=\([0-9]*\).*/\1/p')
ESP_MESH=$(echo "$BEST_PIN" | sed -n 's/.*mesh=\([0-9]*\).*/\1/p')
ESP_RCUT=$(echo "$BEST_PIN" | sed -n 's/.*rcut=\([0-9.]*\).*/\1/p')
ESP_ALPHA=$(echo "$BEST_PIN" | sed -n 's/.*alpha=\([0-9.e+-]*\).*/\1/p')
export ESP_CAO ESP_MESH ESP_RCUT ESP_ALPHA ESP_TUNE=0
echo "SELECTED cao=$ESP_CAO mesh=$ESP_MESH rcut=$ESP_RCUT alpha=$ESP_ALPHA best_wall=$BEST_WALL"

echo ""
echo "=== #6 FFTW scalar vs r5v at best tuned cao (3 reps) ==="
for rep in 1 2 3; do
  bench_one "scalar_r${rep}" "$SCALAR"
  bench_one "r5v_r${rep}" "$R5V"
done

echo ""
echo "=== v2 reference pinned cao=6 scalar ==="
export ESP_MESH=32 ESP_CAO=6 ESP_RCUT=3.4847 ESP_ALPHA=0.751886 ESP_TUNE=0
bench_one "v2ref_scalar" "$SCALAR"

echo "DONE $(date -Iseconds) LOG=$LOG"
