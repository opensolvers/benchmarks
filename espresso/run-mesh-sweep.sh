#!/bin/bash
# Sweep P3M mesh at fixed cao/r_cut/alpha (one pypresso per mesh).
set +e
LOG="${LOG:-$HOME/logs/espresso-mesh-sweep-$(date +%Y%m%d-%H%M%S).log}"
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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENBLAS_CORETYPE=RISCV64_GENERIC
export ESP_HOT_STEPS=${ESP_HOT_STEPS:-300}
MESHES=${ESP_MESHES:-12,18,24,32,48}

echo "SCALAR=$SCALAR STEPS=$ESP_HOT_STEPS MESHES=$MESHES"
IFS=',' read -r -a arr <<<"$MESHES"
for m in "${arr[@]}"; do
  echo ""
  echo "######## mesh=$m ########"
  ESP_MESH=$m LD_PRELOAD="$SCALAR" pypresso "$HERE/mesh_fft_sweep.py"
  echo "RC_$m=$?"
done
echo "DONE $(date -Iseconds)"
echo "LOG=$LOG"
