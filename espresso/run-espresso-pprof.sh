#!/bin/bash
# Sample Espresso P3M hot functions via gperftools (perf not available on 6.6.63-ky).
set +e
LOG="${LOG:-$HOME/logs/espresso-pprof-$(date +%Y%m%d-%H%M%S).log}"
PROF="${PROF:-$HOME/logs/espresso-pprof.prof}"
mkdir -p "$(dirname "$LOG")" "$(dirname "$PROF")"
exec >>"$LOG" 2>&1
echo "START $(date -Iseconds) log=$LOG prof=$PROF"

export EESSI_VERSION_OVERRIDE=2025.06-001
export EESSI_USER_INSTALL="${EESSI_USER_INSTALL:-$HOME/eessi-overlay}"
export EESSI_NO_MODULE_PURGE_ON_INIT=1
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load ESPResSo/4.2.2-foss-2025b 2>/dev/null || module load ESPResSo/4.2.2-foss-2025b

HERE=$(cd "$(dirname "$0")" && pwd)
CASE=${ESP_CASE:-dense_large}
SCALAR=${SCALAR:-$HOME/fftwbuild/src-scalar/.libs/libfftw3.so.3.6.10}
PROFLIB=${PROFLIB:-/usr/lib/riscv64-linux-gnu/libprofiler.so.0}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENBLAS_CORETYPE=RISCV64_GENERIC
export CPUPROFILE="$PROF"
export CPUPROFILE_FREQUENCY=${CPUPROFILE_FREQUENCY:-200}

case "$CASE" in
  lattice512)
    export ESP_N=512 ESP_BOX=20 ESP_STEPS=${ESP_STEPS:-800} ESP_ACCURACY=1e-3
    export ESP_MESH=18 ESP_CAO=5 ESP_RCUT=4.4896 ESP_ALPHA=0.53745 ESP_TUNE=0
    SCRIPT=$HERE/p3m_lj.py
    ;;
  dense_large)
    export ESP_BOX=24 ESP_DENSITY=0.4 ESP_STEPS=${ESP_STEPS:-300} ESP_ACCURACY=1e-3
    export ESP_MESH=32 ESP_CAO=6 ESP_RCUT=3.4847 ESP_ALPHA=0.751886 ESP_TUNE=0
    SCRIPT=$HERE/mpi_dense_large.py
    ;;
  *) echo "bad CASE"; exit 2 ;;
esac

echo "CASE=$CASE SCRIPT=$SCRIPT STEPS=${ESP_STEPS}"
echo "PROFLIB=$PROFLIB SCALAR=$SCALAR"

rm -f "$PROF"
# profiler first so it wraps the process; FFTW second for Espresso calls
LD_PRELOAD="$PROFLIB:$SCALAR" pypresso "$SCRIPT"
echo "RC=$?"
ls -la "$PROF"

ESP_CORE=$(python3 - <<'PY'
import espressomd, os
print(os.path.join(os.path.dirname(espressomd.__file__), "Espresso_core.so"))
PY
)
echo "ESP_CORE=$ESP_CORE"

echo ""
echo "===== google-pprof --text (top 40) ====="
google-pprof --text --lines "$ESP_CORE" "$PROF" 2>/dev/null | head -60 || \
  google-pprof --text "$(command -v python3)" "$PROF" 2>/dev/null | head -60

echo ""
echo "===== filtered: p3m|fft|coulomb|assign|lj|force|verlet ====="
google-pprof --text --lines "$ESP_CORE" "$PROF" 2>/dev/null | \
  grep -iE 'p3m|fft|coulomb|assign|spread|gather|k_space|lj|force|verlet|erfc|ewald|dipole' | head -50

echo ""
echo "===== by shared object (grep profile raw via pprof --list not needed) ====="
google-pprof --text --addresses "$ESP_CORE" "$PROF" 2>/dev/null | head -5
# crude DSO breakdown via nm-resolved text
google-pprof --text "$ESP_CORE" "$PROF" 2>/dev/null | awk '
  NR<=3 {next}
  /^[[:space:]]*[0-9]/ {
    # keep cumulative lines
    print
  }
' | head -40

echo "DONE $(date -Iseconds)"
echo "LOG=$LOG PROF=$PROF"
