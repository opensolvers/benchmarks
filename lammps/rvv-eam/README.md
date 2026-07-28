# LAMMPS EAM — hand RVV plugin (`eam/rvv`)

Cubic-spline EAM density + force with RVV indexed table gathers on SpaceMiT X60.

## Board results (Cu `Cu_u3.eam`, 864 atoms, 100 steps, 1 core, force-only)

| Style | Pair time | vs `eam` |
| --- | ---: | ---: |
| `eam` | 0.780 s | 1.00× |
| **`eam/rvv`** | 0.614 s | **1.27×** |
| `eam/opt` | 0.574 s | 1.36× |

Forces vs stock: **bit-exact** after `vfcvt_rtz` (C `(int)` truncation).

Pair is ~96% of wall on this case. `eam/opt` (templated scalar OPT package)
still wins; RVV beats unoptimized `eam` but not OPT yet.

## Build / run (RV2)

```bash
export EESSI_VERSION_OVERRIDE=2025.06-001
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load foss/2025b
cd plugin && make CXX=mpicxx
# in LAMMPS input:
#   plugin load /path/to/eamrvvplugin.so
#   pair_style eam/rvv
#   pair_coeff 1 1 Cu_u3.eam
```

Fast path: single atom type, `!eflag && !vflag`. Otherwise falls back to `PairEAM::compute`.
