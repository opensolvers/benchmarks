#!/bin/bash
# FFT-axis A/B for ESPResSo P3M: hold BLAS constant (FlexiBLAS scalar OpenBLAS),
# swap ONLY libfftw3.so.3 via LD_PRELOAD (scalar vs r5v/RVV).
#
# Usage: ESP_N=512 ESP_STEPS=200 ESP_BOX=20 bash run-espresso-fft-ab.sh
set +e
LOG="${LOG:-$HOME/logs/espresso-fft-ab-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "START $(date -Iseconds) log=$LOG"

export EESSI_VERSION_OVERRIDE=2025.06-001
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load ESPResSo/4.2.2-foss-2025b
echo "pypresso=$(command -v pypresso)"

HERE=$(cd "$(dirname "$0")" && pwd)
SCALAR=${SCALAR:-$HOME/fftwbuild/src-scalar/.libs/libfftw3.so.3.6.10}
R5V=${R5V:-$HOME/fftwbuild/src-r5v/.libs/libfftw3.so.3.6.10}
SCRIPT=${SCRIPT:-$HERE/p3m_lj.py}
export ESP_N=${ESP_N:-512}
export ESP_STEPS=${ESP_STEPS:-200}
export ESP_BOX=${ESP_BOX:-20}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENBLAS_CORETYPE=RISCV64_GENERIC

run_one() {
  local tag=$1 lib=$2
  echo ""
  echo "========== [$tag] LD_PRELOAD=$(basename "$lib") N=$ESP_N STEPS=$ESP_STEPS =========="
  LD_PRELOAD="$lib" pypresso "$SCRIPT"
  echo "RC=$?"
}

run_one scalar "$SCALAR"
run_one r5v    "$R5V"
echo "DONE $(date -Iseconds)"
