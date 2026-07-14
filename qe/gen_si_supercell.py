#!/usr/bin/env python3
"""Generate a Quantum ESPRESSO pw.x SCF input for an n x n x n supercell of
diamond Silicon (conventional 8-atom cubic cell), gamma point, using the
Si.pz-vbc.UPF LDA norm-conserving pseudo. Designed as a BLAS(DGEMM)-heavy
perf benchmark: large npw x nbnd => dense level-3 GEMM dominates.

Usage: gen_si_supercell.py N [ecutwfc] [electron_maxstep]
  N                supercell replication (N^3 * 8 atoms)
  ecutwfc          plane-wave cutoff in Ry   (default 22)
  electron_maxstep capped SCF iterations      (default 4, bounded runtime)
Prints the input to stdout.
"""
import sys

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
ecut = float(sys.argv[2]) if len(sys.argv) > 2 else 22.0
maxstep = int(sys.argv[3]) if len(sys.argv) > 3 else 4

a0 = 10.26  # bohr, conventional diamond-Si lattice constant (~5.43 Angstrom)
base = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0),
        (0.25, 0.25, 0.25), (0.25, 0.75, 0.75),
        (0.75, 0.25, 0.75), (0.75, 0.75, 0.25)]

atoms = []
for i in range(n):
    for j in range(n):
        for k in range(n):
            for (x, y, z) in base:
                atoms.append(((x + i) / n, (y + j) / n, (z + k) / n))

nat = len(atoms)
nelec = nat * 4                       # Si: 4 valence electrons
nbnd = nelec // 2 + max(8, nelec // 40)  # occupied + ~5% empty buffer
alat = a0 * n

print(f"""&control
    calculation = 'scf'
    prefix      = 'si{nat}'
    pseudo_dir  = '.'
    outdir      = './out'
    verbosity   = 'high'
    disk_io     = 'none'
/
&system
    ibrav       = 1
    celldm(1)   = {alat:.4f}
    nat         = {nat}
    ntyp        = 1
    ecutwfc     = {ecut}
    nbnd        = {nbnd}
/
&electrons
    electron_maxstep  = {maxstep}
    scf_must_converge = .false.
    diagonalization   = 'david'
    mixing_beta       = 0.7
    conv_thr          = 1.0d-6
/
ATOMIC_SPECIES
 Si 28.0855 Si.pz-vbc.UPF
ATOMIC_POSITIONS crystal""")
for (x, y, z) in atoms:
    print(f" Si {x:.8f} {y:.8f} {z:.8f}")
print("K_POINTS gamma")
