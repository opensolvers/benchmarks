# Next: hook RVV LJ into LAMMPS PairLJCut

Microkernel in `rvv-lj/` is correct (~1e-14) and **~1.61×** vs naive scalar on X60
with **EESSI GCC 14.3.0** (n=4k, ~48 neigh/atom, force-on-i only).

## Verified in LAMMPS (plugin)

`plugin/` builds `ljcutrvvplugin.so` → pair style **`lj/cut/rvv`** against the board
overlay LAMMPS (`22Jul2025_update4-foss-2025b-kokkos`), no full rebuild.

```bash
cd plugin
# EESSI init + module load foss/2025b
./verify.sh    # correctness
./bench.sh     # Pair timing vs stock
```

### Correctness (256-atom melt)

| metric | stock `lj/cut` | `lj/cut/rvv` |
| --- | --- | --- |
| PE / forces | — | **bit-exact** (PASS) |

### Performance vs stock `lj/cut` (4000 atoms, 300 steps, 1 core, force-only)

| revision | Pair speedup |
| --- | --- |
| SoA pack + per-neigh params | **0.83–0.95×** (slower) |
| RVV indexed gather (`vloxei64`) + scalar apply | **~1.02×** (5 reps: 0.99–1.04×) |

Microbench **1.61×** does **not** translate in-app: stock is already
`-march=…cv…` auto-vec’d, and half-list newton still needs a scalar `f[j]`
scatter. Indexed gather closed most of the SoA tax; remaining headroom is small.

## EESSI RISC-V reminder

```bash
export EESSI_VERSION_OVERRIDE=2025.06-001
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load foss/2025b   # or GCC/14.3.0
```

## Still open

- Squeeze more from indexed path (approx `1/rsq`, fewer stores, OMP)
- Neigh-list / other hotspots if Pair is near parity
- Kokkos RVV backend (longer arc)
