#!/bin/bash
# PLUMED FlexiBLAS A/B on SpaceMiT X60:
#   driver + CONTACT_MATRIX + SPRINT (NxN eig) under scalar vs patched RVV OpenBLAS.
#
# Usage: PLUMED_N=200 PLUMED_FRAMES=20 bash run-plumed-blas-ab.sh
set +e
LOG="${LOG:-$HOME/logs/plumed-blas-ab-$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "START $(date -Iseconds) log=$LOG"

export EESSI_VERSION_OVERRIDE=2025.06-001
# shellcheck disable=SC1091
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load PLUMED/2.9.4-foss-2025b
echo "plumed=$(command -v plumed)"
if ! command -v plumed >/dev/null; then
  echo "FATAL: plumed not on PATH after module load"
  echo "MODULEPATH=$MODULEPATH"
  exit 1
fi
echo "EBROOTPLUMED=$EBROOTPLUMED"
echo "EBROOTFLEXIBLAS=$EBROOTFLEXIBLAS"

HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$HOME/plumed-bench}
N=${PLUMED_N:-200}
FRAMES=${PLUMED_FRAMES:-20}
SEED=${PLUMED_SEED:-42}
RVV_LIB=${RVV_LIB:-$HOME/libopenblas_x60_eb_fixed.so}
# contact cutoffs in NATURAL (LJ sigma=1) units; ~ first-neighbour shell
R0=${PLUMED_R0:-1.2}
DMAX=${PLUMED_DMAX:-2.0}

mkdir -p "$WORK"
cd "$WORK"
cp -f "$HERE/gen_traj.py" "$HERE/plumed-sprint.dat.in" "$WORK/" 2>/dev/null || true
# allow running from scp'd harness dir
[[ -f "$HERE/gen_traj.py" ]] || HERE="$WORK"

python3 "$HERE/gen_traj.py" -o traj.xyz -n "$N" -f "$FRAMES" --seed "$SEED"
sed -e "s/__NATOMS__/$N/g" -e "s/__R0__/$R0/g" -e "s/__DMAX__/$DMAX/g" \
  "$HERE/plumed-sprint.dat.in" > plumed.dat
echo "---- plumed.dat ----"
cat plumed.dat

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PLUMED_NUM_THREADS=1

run_one() {
  local tag=$1
  shift
  echo ""
  echo "========== [$tag] N=$N FRAMES=$FRAMES env: $* =========="
  rm -f COLVAR timing.$tag
  # wall-clock around driver; also ask PLUMED for its timer if present
  local t0 t1
  t0=$(date +%s.%N)
  env "$@" plumed driver --ixyz traj.xyz --plumed plumed.dat --timestep 1.0 \
    >"out.$tag" 2>"err.$tag"
  local rc=$?
  t1=$(date +%s.%N)
  python3 - "$t0" "$t1" "$rc" "$tag" <<'PY'
import sys
t0,t1,rc,tag=sys.argv[1:5]
dt=float(t1)-float(t0)
print(f"WALL_S={dt:.3f} RC={rc} tag={tag}")
PY
  echo "COLVAR_HEAD:"
  head -3 COLVAR 2>/dev/null || true
  # checksum: sum of sprint coords on last frame
  python3 - <<'PY'
import math,sys
try:
  lines=[l for l in open("COLVAR") if l.strip() and not l.startswith("#")]
  vals=[float(x) for x in lines[-1].split()[1:]]
  s=sum(vals); n=len(vals)
  print(f"CHECKSUM sum={s:.6f} ncomp={n} mean={s/n if n else float('nan'):.6f}")
except Exception as e:
  print("CHECKSUM_FAIL", e)
PY
  grep -E "ERROR|error:|Timing|seconds" "err.$tag" | head -20
  echo "RC=$rc"
}

run_one scalar FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC
if [[ -f "$RVV_LIB" ]]; then
  run_one patched FLEXIBLAS="$RVV_LIB"
else
  echo "SKIP patched: RVV_LIB=$RVV_LIB missing"
fi

echo "DONE $(date -Iseconds)"
