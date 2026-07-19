#!/bin/bash
# FFT-axis A/B for GROMACS mdrun (PME): hold BLAS constant (FlexiBLAS scalar
# OpenBLAS), swap ONLY libfftw3f.so.3 via LD_PRELOAD (scalar vs r5v/RVV).
# Serial (1 rank, 1 thread), CPU PME. GROMACS 2026.2 build reports "SIMD: None",
# so PME 3D-FFT via libfftw3f is the isolated variable. Captures the mdrun cycle
# table (PME 3D-FFT, PME mesh, total Wall) + potential energy for correctness.
#
# Usage: bash run-gmx-fft-ab.sh <label>
LABEL=${1:-run}
BENCH=$HOME/gmx-bench
TPR=$BENCH/md.tpr

SCALAR=$HOME/fftwbuild-single/src-scalar/.libs/libfftw3f.so.3.6.10
R5V=$HOME/fftwbuild-single/src-r5v/.libs/libfftw3f.so.3.6.10

source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash >/dev/null 2>&1
module use /cvmfs/dev.eessi.io/riscv/versions/2025.06-001/software/linux/riscv64/generic/modules/all >/dev/null 2>&1
module load GROMACS/2026.2-foss-2025b FlexiBLAS/3.4.5-GCC-14.3.0 >/dev/null 2>&1
set -eo pipefail
export LD_LIBRARY_PATH=$EBROOTGROMACS/lib:$EBROOTGROMACS/lib64:$EBROOTFFTW/lib:$EBROOTFLEXIBLAS/lib:$EBROOTOPENBLAS/lib:$EBROOTGCCCORE/lib64:$LD_LIBRARY_PATH
export FLEXIBLAS=OPENBLAS OPENBLAS_CORETYPE=RISCV64_GENERIC OMP_NUM_THREADS=1
GMX="gmx --quiet"

cd "$BENCH"
test -f "$SCALAR" || { echo "FATAL: scalar libfftw3f missing: $SCALAR"; exit 1; }
test -f "$R5V"    || { echo "FATAL: r5v libfftw3f missing: $R5V"; exit 1; }
test -f "$TPR"    || { echo "FATAL: md.tpr missing: $TPR"; exit 1; }

run_one() {
  local tag=$1 lib=$2 pfx=$3
  echo "===== [$tag] LD_PRELOAD=$(basename "$lib") ($(stat -c%s "$lib") bytes) ====="
  rm -f "${pfx}".log "${pfx}".edr "${pfx}".gro "${pfx}".cpt "${pfx}".trr 2>/dev/null || true
  env LD_PRELOAD="$lib" $GMX mdrun -s "$TPR" -deffnm "$pfx" \
      -ntmpi 1 -ntomp 1 -nb cpu -pme cpu -pin off -nsteps 2000 \
      > "${pfx}.stdout" 2>&1 || { echo "  RUN FAILED ($tag)"; tail -25 "${pfx}.stdout"; return 1; }
  # potential energy (mean) for correctness comparison
  local pe
  pe=$(printf 'Potential\n' | env LD_PRELOAD="$lib" $GMX energy -f "${pfx}.edr" -o "${pfx}-pot.xvg" 2>/dev/null \
        | grep -iE '^Potential' | awk '{print $2}')
  echo "  Potential energy (avg) = ${pe:-?} kJ/mol"
  # cycle/time accounting from md.log
  echo "  --- md.log cycle table (PME rows + wall) ---"
  grep -E 'PME 3D-FFT|PME mesh|PME spread|PME gather|PME solve| Wall t|Performance' "${pfx}.log" \
    | sed 's/^/    /' || true
}

echo "#################### GROMACS FFT-axis A/B : md.tpr ($LABEL) ####################"
echo "gmx = $($GMX --version 2>/dev/null | grep -i 'GROMACS version' | head -1)"
echo "BLAS = FlexiBLAS->OPENBLAS RISCV64_GENERIC (constant); 1 rank / 1 thread; CPU PME"
echo
run_one scalar "$SCALAR" "$BENCH/gmx-fft-${LABEL}-scalar"
echo
run_one r5v    "$R5V"    "$BENCH/gmx-fft-${LABEL}-r5v"
echo
echo "===== full logs saved: gmx-fft-${LABEL}-{scalar,r5v}.log ====="
