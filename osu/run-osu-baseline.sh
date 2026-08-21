#!/bin/bash
set +e
LOG="$HOME/logs/osu-baseline-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$HOME/logs" "$HOME/osu-bench"
exec >>"$LOG" 2>&1
echo "START $(date -Iseconds) log=$LOG"

export EESSI_VERSION_OVERRIDE=2025.06-001
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load OSU-Micro-Benchmarks/7.5.1-gompi-2025b
echo "osu_latency=$(command -v osu_latency)"
echo "mpirun=$(command -v mpirun)"

# On-node shared-memory baseline (no external NIC in the path for 2 ranks).
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader
# OpenMPI 5 / PRRTE sometimes needs this on SBCs run as root; we are orangepi.
MPIRUN="mpirun --bind-to core"

run() {
  local label=$1; shift
  echo ""
  echo "========== $label =========="
  echo "CMD: $*"
  "$@"
  echo "RC=$?"
}

# Point-to-point (2 ranks)
run "osu_latency np=2"  $MPIRUN -np 2 osu_latency
run "osu_bw np=2"       $MPIRUN -np 2 osu_bw
run "osu_bibw np=2"     $MPIRUN -np 2 osu_bibw

# Collectives (8 ranks = full X60)
run "osu_allreduce np=8" $MPIRUN -np 8 osu_allreduce
run "osu_bcast np=8"     $MPIRUN -np 8 osu_bcast
run "osu_alltoall np=8"  $MPIRUN -np 8 osu_alltoall

echo "DONE $(date -Iseconds)"
