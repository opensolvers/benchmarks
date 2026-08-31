#!/bin/bash
# Skin sweep on opt Espresso (dense_large), then multi-rep A/B vs EESSI.
set +e
LOG="${LOG:-$HOME/logs/espresso-opt-v2-followup.log}"
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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENBLAS_CORETYPE=RISCV64_GENERIC
export LD_PRELOAD="$SCALAR"
export ESP_BOX=24 ESP_DENSITY=0.4 ESP_STEPS=${ESP_STEPS:-200} ESP_ACCURACY=1e-3
export ESP_MESH=32 ESP_CAO=6 ESP_RCUT=3.4847 ESP_ALPHA=0.751886 ESP_TUNE=0

echo "=== skin sweep (opt dense_large) ==="
BEST_SKIN=0.4
BEST_WALL=1e99
for s in 0.4 0.6 0.8 1.0 1.2 1.6; do
  echo "--- ESP_SKIN=$s ---"
  out=$(ESP_SKIN=$s "$OPT/bin/pypresso" "$HERE/mpi_dense_large.py" 2>&1)
  echo "$out" | tail -3
  wall=$(echo "$out" | sed -n 's/.*wall=\([0-9.]*\)s.*/\1/p' | tail -1)
  if [ -n "$wall" ]; then
    awk -v w="$wall" -v b="$BEST_WALL" -v s="$s" -v bs="$BEST_SKIN" \
      'BEGIN{ if (w+0 < b+0) printf "BEST skin=%s wall=%s\n", s, w }'
    # update best in shell
    better=$(awk -v w="$wall" -v b="$BEST_WALL" 'BEGIN{print (w+0<b+0)?1:0}')
    if [ "$better" = 1 ]; then
      BEST_WALL=$wall
      BEST_SKIN=$s
    fi
  fi
done
echo "SELECTED_SKIN=$BEST_SKIN BEST_WALL=$BEST_WALL"
export ESP_SKIN=$BEST_SKIN

echo "=== A/B multi-rep with ESP_SKIN=$ESP_SKIN ==="
LOG=$HOME/logs/espresso-opt-v2-ab.log ESP_CASE=both ESP_REPS=3 \
  ESP_SKIN=$BEST_SKIN bash "$HERE/run-espresso-opt-ab.sh"

echo "DONE $(date -Iseconds) SELECTED_SKIN=$BEST_SKIN LOG=$LOG"
