#!/bin/bash
# run-lammps-bench.sh - LAMMPS canonical-benchmark matrix on one CPU node.
#
# Runs the 5 standard LAMMPS `bench/` inputs (lj, eam, chain, chute, rhodo, all
# 32000 atoms / 100 steps at default size) across three execution modes:
#   serial  - 1 core, no accelerator package
#   kokkos8 - Kokkos/OpenMP, 8 threads (-k on t 8 -sf kk)
#   mpi8    - 8 MPI ranks (mpirun -np 8)
# and emits a CSV (loop time + katom-step/s throughput) plus a serial-relative
# speedup. Written for the RVV-enabled Kokkos LAMMPS built on the Orange Pi RV2
# (SpaceMiT X60) from the dev.eessi.io/riscv foss-2025b custom easyconfig (see
# README), but works on any LAMMPS with the KOKKOS package + an MPI launcher.
#
# Requires: a LAMMPS `lmp` on PATH (or set $LMP), an `mpirun` (or set $MPIRUN),
# and the 5 bench inputs + data files staged in the current directory:
#   in.lj in.eam in.chain in.chute in.rhodo
#   Cu_u3.eam data.chain data.chute data.rhodo
# (copy them from the LAMMPS source tree's `bench/` directory).
#
# Usage:  [LMP=/path/to/lmp] [MPIRUN=/path/to/mpirun] ./run-lammps-bench.sh
set -u

LMP=${LMP:-$(command -v lmp)}
MPIRUN=${MPIRUN:-$(command -v mpirun)}
NP=${NP:-8}
THREADS=${THREADS:-8}

[ -x "$LMP" ]    || { echo "FATAL: lmp not found (set \$LMP)" >&2; exit 1; }
[ -x "$MPIRUN" ] || { echo "FATAL: mpirun not found (set \$MPIRUN)" >&2; exit 1; }
echo "lmp=$LMP mpirun=$MPIRUN np=$NP threads=$THREADS" >&2

OUT=${OUT:-results.csv}
echo "benchmark,mode,atoms,steps,loop_s,timesteps_per_s,katom_step_s" > "$OUT"

parse() {  # $1=bench $2=mode ; reads lmp output on stdin
  awk -v b="$1" -v m="$2" '
    /Loop time of/ {
      loop=$4
      for(i=1;i<=NF;i++){ if($i=="steps"){steps=$(i-1)} if($i=="atoms"){atoms=$(i-1)} }
    }
    /Performance:/ {
      for(i=1;i<=NF;i++){
        if($i=="timesteps/s")  tps=$(i-1)
        if($i=="katom-step/s") kas=$(i-1)
        if($i=="Matom-step/s") kas=$(i-1)*1000
      }
    }
    END { printf "%s,%s,%s,%s,%s,%s,%s\n", b,m,atoms,steps,loop,tps,kas }'
}

for b in lj eam chain chute rhodo; do
  echo ">>> $b serial"  >&2
  "$LMP" -in in.$b 2>/dev/null | parse "$b" serial >> "$OUT"
  echo ">>> $b kokkos$THREADS" >&2
  OMP_NUM_THREADS=$THREADS "$LMP" -k on t $THREADS -sf kk -in in.$b 2>/dev/null | parse "$b" kokkos$THREADS >> "$OUT"
  echo ">>> $b mpi$NP" >&2
  "$MPIRUN" -np $NP "$LMP" -in in.$b 2>/dev/null | parse "$b" mpi$NP >> "$OUT"
done
echo "ALL DONE -> $OUT" >&2
