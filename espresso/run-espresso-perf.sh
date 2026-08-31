#!/bin/bash
# Profile Espresso P3M with perf cpu-clock (kernel tools mismatch OK for software events).
# Env: ESP_CASE=dense_large|lattice512  ESP_BUILD=eessi|opt  ESP_STEPS=...
set +e
BUILD=${ESP_BUILD:-eessi}
CASE=${ESP_CASE:-dense_large}
LOG="${LOG:-$HOME/logs/espresso-perf-${BUILD}-${CASE}.log}"
DATA="${DATA:-$HOME/logs/espresso-perf-${BUILD}-${CASE}.data}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "START $(date -Iseconds) build=$BUILD case=$CASE log=$LOG data=$DATA"

PERF=${PERF:-/usr/lib/linux-riscv-6.17-tools-6.17.0-38/perf}
export EESSI_VERSION_OVERRIDE=2025.06-001
export EESSI_USER_INSTALL="${EESSI_USER_INSTALL:-$HOME/eessi-overlay}"
export EESSI_NO_MODULE_PURGE_ON_INIT=1
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load foss/2025b 2>/dev/null || true

HERE=$(cd "$(dirname "$0")" && pwd)
SCALAR=${SCALAR:-$HOME/fftwbuild/src-scalar/.libs/libfftw3.so.3.6.10}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENBLAS_CORETYPE=RISCV64_GENERIC
export LD_PRELOAD="$SCALAR"

case "$CASE" in
  lattice512)
    export ESP_N=512 ESP_BOX=20 ESP_STEPS=${ESP_STEPS:-600} ESP_ACCURACY=1e-3
    export ESP_MESH=18 ESP_CAO=5 ESP_RCUT=4.4896 ESP_ALPHA=0.53745 ESP_TUNE=0
    SCRIPT=$HERE/p3m_lj.py
    ;;
  dense_large)
    export ESP_BOX=24 ESP_DENSITY=0.4 ESP_STEPS=${ESP_STEPS:-250} ESP_ACCURACY=1e-3
    export ESP_MESH=32 ESP_CAO=6 ESP_RCUT=3.4847 ESP_ALPHA=0.751886 ESP_TUNE=0
    SCRIPT=$HERE/mpi_dense_large.py
    ;;
  *)
    echo "unknown ESP_CASE=$CASE" >&2
    exit 1
    ;;
esac

if [ "$BUILD" = opt ]; then
  # shellcheck disable=SC1090
  [ -f "$HOME/espresso-opt-env.sh" ] && . "$HOME/espresso-opt-env.sh"
  OPT=${OPT:-$HOME/espresso-opt}
  export PATH="$OPT/bin:$PATH"
  PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  export PYTHONPATH="$OPT/lib/python${PYVER}/site-packages:${PYTHONPATH:-}"
  PYPRESSO="$OPT/bin/pypresso"
else
  module load ESPResSo/4.2.2-foss-2025b 2>/dev/null || module load ESPResSo/4.2.2-foss-2025b
  PYPRESSO="$(command -v pypresso)"
fi

echo "CASE=$CASE BUILD=$BUILD STEPS=$ESP_STEPS PERF=$PERF SCRIPT=$SCRIPT PYPRESSO=$PYPRESSO"
"$PYPRESSO" -c "import espressomd; print('features', sorted(espressomd.features()))" 2>/dev/null | tail -1
rm -f "$DATA"

# Record whole run (setup is small vs MD for dense_large).
$PERF record -e cpu-clock -F ${ESP_PERF_FREQ:-99} -g -o "$DATA" -- \
  "$PYPRESSO" "$SCRIPT"
echo "RECORD_RC=$?"

ESP_CORE="$("$PYPRESSO" -c 'import espressomd,os; print(os.path.join(os.path.dirname(espressomd.__file__), "Espresso_core.so"))')"
echo "ESP_CORE=$ESP_CORE"

echo ""
echo "===== TOP (no children) ====="
$PERF report -i "$DATA" --stdio --no-children --percent-limit 0.5 2>/dev/null | head -80

echo ""
echo "===== TOP (children / callers) ====="
$PERF report -i "$DATA" --stdio --children --percent-limit 1.0 2>/dev/null | head -80

echo ""
echo "===== FILTER p3m|fft|coulomb|assign|lj|force|verlet|erfc|add_non|function ====="
$PERF report -i "$DATA" --stdio --no-children --symbols 2>/dev/null | \
  grep -iE 'p3m|fft|coulomb|assign|lj|force|verlet|erfc|ewald|short_range|integrate|langevin|add_non|function|invoke|ForceKernel' | head -80

echo ""
echo "===== DSO overhead ====="
$PERF report -i "$DATA" --stdio --no-children -s dso --percent-limit 0.3 2>/dev/null | head -40

echo "DONE $(date -Iseconds)"
echo "LOG=$LOG DATA=$DATA"
