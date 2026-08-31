#!/bin/bash
# Survey Coulomb / P3M share across ESPResSo model classes (scalar FFTW).
# One pypresso process per model — ESPResSo allows only one System instance.
set +e
LOG="${LOG:-$HOME/logs/espresso-hotpath-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "START $(date -Iseconds) log=$LOG"

export EESSI_VERSION_OVERRIDE=2025.06-001
export EESSI_USER_INSTALL="${EESSI_USER_INSTALL:-$HOME/eessi-overlay}"
export EESSI_NO_MODULE_PURGE_ON_INIT=1
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load ESPResSo/4.2.2-foss-2025b 2>/dev/null || module load ESPResSo/4.2.2-foss-2025b

HERE=$(cd "$(dirname "$0")" && pwd)
SCALAR=${SCALAR:-$HOME/fftwbuild/src-scalar/.libs/libfftw3.so.3.6.10}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENBLAS_CORETYPE=RISCV64_GENERIC
export ESP_HOT_STEPS=${ESP_HOT_STEPS:-150}
export ESP_HOT_FORCE_LOOPS=${ESP_HOT_FORCE_LOOPS:-30}

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=(lattice512 sample_p3m dense_wca salt_box50 electrophoresis dense_large)
fi

echo "pypresso=$(command -v pypresso)"
echo "SCALAR=$SCALAR"
echo "MODELS=${MODELS[*]}"
echo "ESP_HOT_STEPS=$ESP_HOT_STEPS ESP_HOT_FORCE_LOOPS=$ESP_HOT_FORCE_LOOPS"

for m in "${MODELS[@]}"; do
  echo ""
  echo "######## process model=$m ########"
  LD_PRELOAD="$SCALAR" pypresso "$HERE/hotpath_models.py" "$m"
  echo "RC_$m=$?"
done

echo ""
echo "===== COMBINED (grep from this log) ====="
grep -E "^MD |^P3M tuned|^===== " "$LOG" || true
echo "DONE $(date -Iseconds)"
echo "LOG=$LOG"
