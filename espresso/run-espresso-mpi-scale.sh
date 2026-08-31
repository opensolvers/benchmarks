#!/bin/bash
# MPI strong-scaling for ESPResSo P3M (Coulomb hot path).
# Usage: ESP_CASE=lattice512|dense_large bash run-espresso-mpi-scale.sh
set +e
LOG="${LOG:-$HOME/logs/espresso-mpi-scale-$(date +%Y%m%d-%H%M%S).log}"
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
CASE=${ESP_CASE:-lattice512}
NPROCS_LIST=${ESP_NPROCS:-1,2,4,8}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENBLAS_CORETYPE=RISCV64_GENERIC

# Keep FFTW fixed (scalar) — this is an MPI axis, not FFTW A/B.
SCALAR=${SCALAR:-$HOME/fftwbuild/src-scalar/.libs/libfftw3.so.3.6.10}
export LD_PRELOAD="$SCALAR"

case "$CASE" in
  lattice512)
    export ESP_N=512 ESP_BOX=20 ESP_ACCURACY=1e-3
    export ESP_STEPS=${ESP_STEPS:-400}
    # pin tuned params so ranks don't retune differently
    export ESP_MESH=18 ESP_CAO=5 ESP_RCUT=4.4896 ESP_ALPHA=0.53745 ESP_TUNE=0
    SCRIPT=$HERE/p3m_lj.py
    ;;
  dense_large)
    export ESP_BOX=24 ESP_DENSITY=0.4 ESP_ACCURACY=1e-3
    export ESP_STEPS=${ESP_STEPS:-200}
    # pins from hotpath survey3 (np-invariant)
    export ESP_MESH=32 ESP_CAO=6 ESP_RCUT=3.4847 ESP_ALPHA=0.751886 ESP_TUNE=0
    SCRIPT=$HERE/mpi_dense_large.py
    ;;
  *)
    echo "unknown ESP_CASE=$CASE"; exit 2
    ;;
esac

echo "CASE=$CASE SCRIPT=$SCRIPT NPROCS_LIST=$NPROCS_LIST"
echo "ESP_N=${ESP_N:-} ESP_STEPS=${ESP_STEPS:-$ESP_HOT_STEPS} ESP_BOX=${ESP_BOX:-}"
echo "mesh=${ESP_MESH:-tune} LD_PRELOAD=$(basename "$LD_PRELOAD")"
echo "mpirun=$(command -v mpirun) pypresso=$(command -v pypresso)"

IFS=',' read -r -a NPS <<<"$NPROCS_LIST"
for np in "${NPS[@]}"; do
  echo ""
  echo "========== np=$np =========="
  # bind loosely; RV2 has 8 cores
  mpirun -np "$np" --bind-to core --map-by core \
    pypresso "$SCRIPT"
  echo "RC_np$np=$?"
done

echo "DONE $(date -Iseconds)"
echo "LOG=$LOG"
