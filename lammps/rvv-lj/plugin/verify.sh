#!/usr/bin/env bash
# A/B verify lj/cut vs lj/cut/rvv inside LAMMPS (forces + PE).
set -eo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

export EESSI_VERSION_OVERRIDE="${EESSI_VERSION_OVERRIDE:-2025.06-001}"
export EESSI_NO_MODULE_PURGE_ON_INIT="${EESSI_NO_MODULE_PURGE_ON_INIT:-}"
if ! command -v mpicxx >/dev/null 2>&1 || ! mpicxx --version 2>/dev/null | grep -q '14\.3'; then
  # shellcheck disable=SC1091
  source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
  module load foss/2025b
fi

LAMMPS_PREFIX="${LAMMPS_PREFIX:-$HOME/eessi-overlay/versions/2025.06-001/software/linux/riscv64/generic/software/LAMMPS/22Jul2025_update4-foss-2025b-kokkos}"
LMP="${LMP:-$LAMMPS_PREFIX/bin/lmp}"
PLUGIN="$DIR/ljcutrvvplugin.so"

make -j"$(nproc)" LAMMPS_PREFIX="$LAMMPS_PREFIX"

run_case() {
  local tag=$1
  local style=$2
  local logfile="log.$tag"
  local dump="forces.$tag.dump"
  rm -f "$logfile" "$dump"

  {
    echo "variable style string $style"
    if [[ "$style" == *rvv* ]]; then
      echo "plugin load $PLUGIN"
    fi
    cat <<EOF
units           lj
atom_style      atomic
boundary        p p p

lattice         fcc 0.8442
region          box block 0 4 0 4 0 4
create_box      1 box
create_atoms    1 box
mass            1 1.0

velocity        all create 1.44 87287 loop geom

neighbor        0.3 bin
neigh_modify    delay 0 every 1 check yes

pair_style      \${style} 2.5
pair_coeff      1 1 1.0 1.0 2.5

fix             1 all nve
fix             2 all langevin 1.0 1.0 10.0 48279

thermo          50
thermo_style    custom step temp pe ke etotal press

run             20
unfix           2

# force evaluation only — dump PE + per-atom forces
run             0
variable        pe equal pe
print           "VERIFY_PE \${pe}"
write_dump      all custom $dump id type x y z fx fy fz modify sort id
EOF
  } > "in.$tag"

  echo "=== running $tag ($style) ==="
  taskset -c 0 "$LMP" -in "in.$tag" > "$logfile" 2>&1
  grep VERIFY_PE "$logfile" || { echo "missing VERIFY_PE in $logfile"; tail -40 "$logfile"; exit 1; }
}

run_case stock "lj/cut"
run_case rvv "lj/cut/rvv"

python3 - <<'PY'
import math, re, sys
from pathlib import Path

def pe(log):
    for line in Path(log).read_text().splitlines():
        if line.startswith("VERIFY_PE"):
            return float(line.split()[1])
    raise SystemExit(f"no VERIFY_PE in {log}")

def forces(path):
    # LAMMPS custom dump: ITEM: ATOMS id type x y z fx fy fz
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines) and not lines[i].startswith("ITEM: ATOMS"):
        i += 1
    if i >= len(lines):
        raise SystemExit(f"no ATOMS in {path}")
    out = {}
    for line in lines[i+1:]:
        if line.startswith("ITEM:"):
            break
        p = line.split()
        if len(p) < 8:
            continue
        aid = int(p[0])
        out[aid] = (float(p[5]), float(p[6]), float(p[7]))
    return out

pe_s, pe_r = pe("log.stock"), pe("log.rvv")
fs, fr = forces("forces.stock.dump"), forces("forces.rvv.dump")
ids = sorted(set(fs) | set(fr))
if set(fs) != set(fr):
    print("FAIL: atom id mismatch", file=sys.stderr)
    sys.exit(1)

max_abs = 0.0
sum_sq = 0.0
norm_s = 0.0
for aid in ids:
    a, b = fs[aid], fr[aid]
    for k in range(3):
        d = a[k] - b[k]
        max_abs = max(max_abs, abs(d))
        sum_sq += d * d
        norm_s += a[k] * a[k]
rms = math.sqrt(sum_sq / (3 * len(ids)))
rel = math.sqrt(sum_sq) / (math.sqrt(norm_s) + 1e-300)
dpe = abs(pe_s - pe_r)

print(f"atoms          {len(ids)}")
print(f"PE stock       {pe_s:.16g}")
print(f"PE rvv         {pe_r:.16g}")
print(f"|ΔPE|          {dpe:.6e}")
print(f"max |Δf|       {max_abs:.6e}")
print(f"rms |Δf|       {rms:.6e}")
print(f"rel ||Δf||     {rel:.6e}")

# Tight for double LJ; allow tiny PE drift from reduction order
ok = (max_abs < 1e-10) and (dpe < 1e-8)
print("RESULT         " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PY
