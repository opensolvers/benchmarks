#!/bin/bash
# Prep a small PME/FFT-heavy GROMACS system on the RV2: a cubic box of SPC water,
# energy-minimized, then a short MD .tpr with PME electrostatics. PME => real 3D
# FFTs through libfftw3f => the FFT-axis A/B subject. Serial, mixed precision.
# lmod BEFORE any set -e (EESSI gotcha).

source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash >/dev/null 2>&1
module use /cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/modules/all >/dev/null 2>&1
module load GROMACS/2026.2-foss-2025b >/dev/null 2>&1
export LD_LIBRARY_PATH=$EBROOTGROMACS/lib:$EBROOTGROMACS/lib64:$EBROOTFFTW/lib:$EBROOTFLEXIBLAS/lib:$EBROOTOPENBLAS/lib:$EBROOTGCCCORE/lib64:$LD_LIBRARY_PATH
set -eo pipefail

BENCH=$HOME/gmx-bench
rm -rf "$BENCH"; mkdir -p "$BENCH"; cd "$BENCH"
GMX="gmx --quiet"
TOP=$EBROOTGROMACS/share/gromacs/top

# 1) minimal topology (SPC water); gmx solvate appends the correct SOL count.
cat > topol.top <<EOF
#include "$TOP/oplsaa.ff/forcefield.itp"
#include "$TOP/oplsaa.ff/spc.itp"
[ system ]
SPC water box
[ molecules ]
EOF

# 2) define an empty cubic box, then fill it with SPC water. A 5.6 nm box holds
#    ~5800 waters (~17k atoms) -> a non-trivial PME FFT grid. solvate writes the
#    SOL line into topol.top automatically (canonical, no manual counting).
$GMX insert-molecules -box 5.6 5.6 5.6 -nmol 0 -o empty.gro >/dev/null 2>&1 || \
  printf "empty\n    0\n   5.6 5.6 5.6\n" > empty.gro
$GMX solvate -cp empty.gro -cs spc216.gro -box 5.6 5.6 5.6 -o box27.gro -p topol.top >solvate.log 2>&1
NW=$(grep -E '^SOL' topol.top | tail -1 | awk '{print $2}')
echo "waters=$NW"

# 3) energy-minimization mdp
cat > em.mdp <<'EOF'
integrator  = steep
nsteps      = 200
emtol       = 1000
coulombtype = PME
rcoulomb    = 0.9
rvdw        = 0.9
rlist       = 0.9
fourierspacing = 0.12
pbc         = xyz
EOF

# 4) short MD mdp — PME, fixed step count so both A/B runs do identical FFT work
cat > md.mdp <<'EOF'
integrator      = md
nsteps          = 2000
dt              = 0.002
nstlist         = 10
cutoff-scheme   = Verlet
coulombtype     = PME
rcoulomb        = 0.9
rvdw            = 0.9
rlist           = 0.9
fourierspacing  = 0.12
pme-order       = 4
tcoupl          = v-rescale
tc-grps         = System
tau-t           = 0.1
ref-t           = 300
gen-vel         = yes
gen-temp        = 300
constraints     = h-bonds
EOF

# 5) EM then produce the MD .tpr
$GMX grompp -f em.mdp -c box27.gro -p topol.top -o em.tpr -maxwarn 2 >grompp_em.log 2>&1
$GMX mdrun -deffnm em -ntmpi 1 -ntomp 1 -nb cpu -pme cpu >mdrun_em.log 2>&1
$GMX grompp -f md.mdp -c em.gro -p topol.top -o md.tpr -maxwarn 2 >grompp_md.log 2>&1

echo "=== PME grid (from grompp_md.log) ==="
grep -iE 'fourier|grid|PME' grompp_md.log | head
echo "=== md.tpr ready: $(ls -la md.tpr) ==="
