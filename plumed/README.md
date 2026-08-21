# PLUMED — FlexiBLAS A/B (SpaceMiT X60 / Orange Pi RV2)

Enhanced-sampling / CV library [`PLUMED/2.9.4-foss-2025b`](https://www.plumed.org/)
on EESSI `2025.06-001`. Linked against FlexiBLAS (`-lflexiblas`); this harness
swaps the OpenBLAS backend under an unchanged `plumed driver`.

## Workload

Per-frame **SPRINT** on a `CONTACT_MATRIX` for N particles (largest-eigenvalue
topological CVs → dense NxN LAPACK via FlexiBLAS):

```
d: DENSITY SPECIES=1-N
mat: CONTACT_MATRIX ATOMS=d SWITCH={RATIONAL R_0=1.2 D_MAX=2.0}
ss: SPRINT MATRIX=mat
```

Trajectory: synthetic cubic LJ liquid (`gen_traj.py`). Note: bare
`CONTACT_MATRIX ATOMS=1-N` + `SPRINT` segfaults on this build; the `DENSITY`
wrapper is required.

## Result (2026-08-21)

| Backend | Wall (N=200, 20 frames) |
|---------|------------------------:|
| scalar (`OPENBLAS_CORETYPE=RISCV64_GENERIC`) | 80.36 s |
| patched RVV (`libopenblas_x60_eb_fixed.so`) | **64.11 s** |

**Speedup patched vs scalar: 1.25×.**  
Checksums match (`sum(ss.*)=975.389111` on last frame).

Log: `~/logs/plumed-blas-ab-20260821-181833.log` on the board.

## Reproduce

```bash
# on RV2
PLUMED_N=200 PLUMED_FRAMES=20 bash ~/plumed-harness/run-plumed-blas-ab.sh
# RVV_LIB defaults to ~/libopenblas_x60_eb_fixed.so
```
