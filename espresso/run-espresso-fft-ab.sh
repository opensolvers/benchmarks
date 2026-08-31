#!/bin/bash
# FFT-axis A/B for ESPResSo P3M: hold BLAS constant (FlexiBLAS scalar OpenBLAS),
# swap ONLY libfftw3.so.3 via LD_PRELOAD (scalar vs r5v/RVV).
#
# Usage:
#   ESP_N=512 ESP_STEPS=200 ESP_BOX=20 ESP_ACCURACY=1e-3 bash run-espresso-fft-ab.sh
# Fair A/B (same mesh both sides): also set ESP_MESH ESP_CAO ESP_RCUT.
set +e
LOG="${LOG:-$HOME/logs/espresso-fft-ab-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "START $(date -Iseconds) log=$LOG"

export EESSI_VERSION_OVERRIDE=2025.06-001
export EESSI_USER_INSTALL="${EESSI_USER_INSTALL:-$HOME/eessi-overlay}"
export EESSI_NO_MODULE_PURGE_ON_INIT=1
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load ESPResSo/4.2.2-foss-2025b 2>/dev/null || \
  module load ESPResSo/4.2.2-foss-2025b
echo "pypresso=$(command -v pypresso)"

HERE=$(cd "$(dirname "$0")" && pwd)
SCALAR=${SCALAR:-$HOME/fftwbuild/src-scalar/.libs/libfftw3.so.3.6.10}
R5V=${R5V:-$HOME/fftwbuild/libfftw3-r5v-xorconj.so}
# fall back to live r5v tree if snapshot missing
[ -f "$R5V" ] || R5V=$HOME/fftwbuild/src-r5v/.libs/libfftw3.so.3.6.10
SCRIPT=${SCRIPT:-$HERE/p3m_lj.py}
export ESP_N=${ESP_N:-512}
export ESP_STEPS=${ESP_STEPS:-200}
export ESP_BOX=${ESP_BOX:-20}
export ESP_ACCURACY=${ESP_ACCURACY:-1e-3}
# optional pins (passed through to p3m_lj.py)
[ -n "${ESP_MESH:-}" ] && export ESP_MESH
[ -n "${ESP_CAO:-}" ] && export ESP_CAO
[ -n "${ESP_RCUT:-}" ] && export ESP_RCUT
[ -n "${ESP_ALPHA:-}" ] && export ESP_ALPHA
[ -n "${ESP_TUNE:-}" ] && export ESP_TUNE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENBLAS_CORETYPE=RISCV64_GENERIC

echo "SCALAR=$SCALAR"
echo "R5V=$R5V"
echo "ESP_N=$ESP_N ESP_STEPS=$ESP_STEPS ESP_BOX=$ESP_BOX ESP_ACCURACY=$ESP_ACCURACY"
echo "ESP_MESH=${ESP_MESH:-} ESP_CAO=${ESP_CAO:-} ESP_RCUT=${ESP_RCUT:-} ESP_ALPHA=${ESP_ALPHA:-} ESP_TUNE=${ESP_TUNE:-1}"

run_one() {
  local tag=$1 lib=$2
  echo ""
  echo "========== [$tag] LD_PRELOAD=$(basename "$lib") N=$ESP_N STEPS=$ESP_STEPS ACC=$ESP_ACCURACY =========="
  LD_PRELOAD="$lib" pypresso "$SCRIPT"
  echo "RC=$?"
}

run_one scalar "$SCALAR"
run_one r5v    "$R5V"
echo "DONE $(date -Iseconds)"
echo "LOG=$LOG"
