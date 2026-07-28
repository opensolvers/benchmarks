# Kokkos / LAMMPS-RVV — results table

Hardware: Orange Pi RV2, SpaceMiT X60 @ 1.6 GHz, RVV 1.0 VLEN=256.  
Toolchain for published rows: **EESSI** `GCC/14.3.0` via

```text
EESSI_VERSION_OVERRIDE=2025.06-001
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load GCC/14.3.0
```

(Apps from `/cvmfs/dev.eessi.io/riscv`; init from `software.eessi.io`.)

## LJ/cut microkernel (`lammps/rvv-lj`)

Force-on-i only; single type; `lj1=48`, `lj2=24`, `cut=2.5`; `taskset -c 0`.

| Date | Compiler | n | nnz | rounds | max\|Δf\| | scalar ns/pair | RVV ns/pair | speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-07-28 | EESSI GCC 14.3.0 | 2048 | 98426 | 50 | 6.75e-14 | 58.7 | 35.8 | **1.64×** |
| 2026-07-28 | EESSI GCC 14.3.0 | 4096 | 196644 | 30 | 4.10e-14 | 58.9 | 36.4 | **1.62×** |
| 2026-07-28 | system GCC 13.3 | 2048 | 98426 | 50 | 6.75e-14 | 58.2 | 36.5 | 1.59× |

## Stock LAMMPS Kokkos (no hand RVV)

| Case | Notes | Result |
| --- | --- | --- |
| `lmp -h` | OpenMP+Serial, Kokkos 4.6.2, FFTW3 | OK |
| `in.melt` | 4000 atoms, overlay binary | PASSED (~42 steps/s @ 1×1 in one run) |
| Timer split (lj / melt class) | Pair ~84–87%, Neigh ~8–12% | Pair-dominated |
| Kokkos SIMD | No RVV backend in 4.6.2 | Scalar ops in Pair functor |

## Interpretation

- **1.6×** on the isolated LJ Pair math (after SoA packing) is real vs naive scalar.
- **In-LAMMPS LJ** lands near **1.02×** vs stock — auto-vec + gather tax.
- **In-LAMMPS EAM** is the stronger win: **1.27×** vs `eam`, still behind `eam/opt`
  (1.36×). Pair ~96% of wall.
- Stock Kokkos on this board is a **portable OpenMP** vehicle, not an RVV
  vectorizer — same lesson as GROMACS before `impl_riscv_rvv`.

## EAM plugin (`lammps/rvv-eam`) — in-LAMMPS Pair

| Date | Styles | atoms / steps | `eam` | `eam/rvv` | `eam/opt` | rvv/eam | forces |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 2026-07-28 | Cu_u3, 1 core, force-only | 864 / 100 | 0.780 s | 0.614 s | 0.574 s | **1.27×** | bit-exact |

## LJ plugin (`lammps/rvv-lj`) — in-LAMMPS Pair

| Date | Case | Pair speedup vs stock `lj/cut` |
| --- | --- | ---: |
| 2026-07-28 | 4000 atoms, indexed-gather path, 5 reps | **~1.02×** (0.99–1.04×) |
