#!/usr/bin/env bash
# Time stock lj/cut vs lj/cut/rvv on RV2 (Pair-dominated melt).
set -eo pipefail
cd "$(dirname "$0")"

export EESSI_VERSION_OVERRIDE="${EESSI_VERSION_OVERRIDE:-2025.06-001}"
export EESSI_NO_MODULE_PURGE_ON_INIT="${EESSI_NO_MODULE_PURGE_ON_INIT:-}"
if ! command -v mpicxx >/dev/null 2>&1 || ! mpicxx --version 2>/dev/null | grep -q '14\.3'; then
  # shellcheck disable=SC1091
  source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
  module load foss/2025b
fi

LAMMPS_PREFIX="${LAMMPS_PREFIX:-$HOME/eessi-overlay/versions/2025.06-001/software/linux/riscv64/generic/software/LAMMPS/22Jul2025_update4-foss-2025b-kokkos}"
LMP="${LMP:-$LAMMPS_PREFIX/bin/lmp}"
PLUGIN="$PWD/ljcutrvvplugin.so"

make -j"$(nproc)" LAMMPS_PREFIX="$LAMMPS_PREFIX" >/dev/null

write_in() {
  local tag=$1
  local style=$2
  local outfile="in.bench_$tag"
  {
    if [[ "$style" == *rvv* ]]; then
      echo "plugin load $PLUGIN"
    fi
    cat <<EOF
units lj
atom_style atomic
boundary p p p
lattice fcc 0.8442
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.44 87287 loop geom
neighbor 0.3 bin
neigh_modify delay 0 every 1 check yes
pair_style $style 2.5
pair_coeff 1 1 1.0 1.0 2.5
fix 1 all nve
fix 2 all langevin 1.0 1.0 10.0 48279
# Force-only timing: no pressure/virial tax on Pair
thermo 0
thermo_style custom step temp pe
timer full
run 50
unfix 2
run 300
EOF
  } > "$outfile"
}

run_one() {
  local tag=$1
  local style=$2
  write_in "$tag" "$style"
  echo "=== bench $tag ($style) ==="
  OMP_NUM_THREADS=1 taskset -c 0 "$LMP" -in "in.bench_$tag" > "log.bench_$tag" 2>&1
  grep -E 'Loop time of|^\s*Pair\s+\||Created [0-9]+ atoms' "log.bench_$tag" | tail -20
  echo
}

run_one stock lj/cut
run_one rvv lj/cut/rvv

python3 - <<'PY'
import re
from pathlib import Path

def events(path):
    lines = Path(path).read_text().splitlines()
    out = []
    for i, line in enumerate(lines):
        m = re.search(
            r"Loop time of ([0-9.]+) on \d+ procs for (\d+) steps with (\d+) atoms",
            line,
        )
        if not m:
            continue
        loop_t, steps, atoms = float(m.group(1)), int(m.group(2)), int(m.group(3))
        pair_avg = pair_pct = None
        for j in range(i + 1, min(i + 40, len(lines))):
            pm = re.match(
                r"\s*Pair\s+\|\s+([0-9.eE+-]+)\s+\|\s+([0-9.eE+-]+)\s+\|"
                r"\s+([0-9.eE+-]+)\s+\|\s+[0-9.]+\s+\|\s+([0-9.]+)",
                lines[j],
            )
            if pm:
                pair_avg = float(pm.group(2))
                pair_pct = float(pm.group(4))
                break
        out.append((steps, atoms, loop_t, pair_avg, pair_pct))
    return out

def pick(ev):
    for e in reversed(ev):
        if e[0] >= 200:
            return e
    return ev[-1] if ev else None

es, er = events("log.bench_stock"), events("log.bench_rvv")
print("stock:", es)
print("rvv:  ", er)
s, r = pick(es), pick(er)
if not s or not r:
    raise SystemExit("missing runs")
print()
print(f"atoms={s[1]}  steps={s[0]}")
print(f"Loop  stock={s[2]:.4f}s  rvv={r[2]:.4f}s  speedup={s[2]/r[2]:.3f}x")
if s[3] is not None and r[3] is not None and r[3] > 0:
    print(f"Pair  stock={s[3]:.4f}s ({s[4]}%)  rvv={r[3]:.4f}s ({r[4]}%)  speedup={s[3]/r[3]:.3f}x")
else:
    print("Pair timer rows missing; showing Loop only")
PY
