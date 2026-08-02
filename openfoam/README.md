# OpenFOAM — motorBike / `simpleFoam` RVV kernel A/B (Orange Pi RV2)

OpenFOAM **v2506** on the Orange Pi RV2 (SpaceMiT K1 / X60 family, RVV 1.0),
under the EESSI RISC-V overlay. This directory records **measured** A/Bs on the
canonical **motorBike** tutorial (`simpleFoam`, 4 MPI ranks, existing mesh,
`endTime=50`).

> **Change one variable.** Axes: GCC auto-vectorize (`-ftree-vectorize`), then
> hand RVV **Amul**, then hand RVV **Gauss–Seidel** face loops — each held
> against an otherwise identical solve.

> **Bottom line:** neither GCC auto-vec nor the hand RVV Amul / GS gather paths
> win on this board. Sparse gather on short LDU/CSR rows regresses; GS still
> dominates wall time and stays hard to SIMD without recolouring.

Raw on-board summaries: [`results/`](results).

---

## Setup (what was measured)

| Piece | Value |
|---|---|
| Board | Orange Pi RV2 · `orangepi@192.168.3.112` · ~7.7 GiB |
| Stack | EESSI `2025.06-001` + user overlay OpenFOAM `v2506-foss-2025b-noPV` |
| Case | `~/openfoam-runs/motorBike` · `decomposePar` hierarchical `n (2 2 1)` · **4 ranks** |
| Solver | `simpleFoam` · smoothSolver+GS (U/k/ω) · GAMG+GS (p) |
| Timing | OpenFOAM `ExecutionTime` + wall; optional `FOAM_RVV_PROFILE=1` |

Hand kernels lived in a **thin** `libOpenFOAM.so` (interposes Amul / `sumProd` /
GS) over baseline `libOpenFOAM-impl.so` — no full `src/OpenFOAM` rebuild for the
RVV A/Bs. Env gates:

| Env | Role |
|---|---|
| `FOAM_RVV_KERNELS=0/1` | Master off/on (default on if unset) |
| `FOAM_RVV_AMUL=0/1` | CSR-gather Amul (default follows kernels) |
| `FOAM_RVV_GS=0/1` | GS face gather/scatter for cells with ≥4 faces |
| `FOAM_RVV_PROFILE=1` | Print Amul / GS / dot / axpy seconds at exit |

`LD_LIBRARY_PATH` must put GCCcore 14 `lib64` first (overlay `libstdc++`).

---

## 1. Auto-vectorize A/B (`-ftree-vectorize`)

Same mesh/solve; only `c++OPT` ± `-ftree-vectorize` (full host `-march=…v…`).

| Variant | ExecutionTime | WALL |
|---|---|---|
| `-ftree-vectorize` | **299.85 s** | 344.1 s |
| `-fno-tree-vectorize` | **300.74 s** | 345.4 s |

**~0%** — auto-vec does not move motorBike. Source: [`results/motorbike-ab-results.txt`](results/motorbike-ab-results.txt).

---

## 2. Hand RVV Amul A/B

CSR-style gather Amul (`vluxei32` + mul + reduce) vs scalar Amul. Profile shows
**GS > Amul** on this case.

| Run | Env | WALL | ExecTime (last) | Amul (4 ranks) | GS (4 ranks) |
|---|---|---|---|---|---|
| **RVV Amul on** | `KERNELS=1` (Amul+GS copy) | **363.8 s** | **319.6 s** | **32.5–37.0 s** / 4248 | 56–65 s / 3830 |
| **RVV off** | `KERNELS=0` | **351.3 s** | **307.1 s** | **20.7–25.5 s** / 4248 | 56–65 s / 3830 |

Hand RVV Amul is **~50% slower** than scalar (~35 s vs ~23 s mean) → whole solve
**~3–4% slower**. GS unchanged. Source: [`results/motorbike-rvv-ab-summary.txt`](results/motorbike-rvv-ab-summary.txt).

---

## 3. Hand RVV Gauss–Seidel A/B

Inner face loops only: accumulate via gather-dot, distribute via gather/scatter
(`nFaces ≥ 4`). **Amul forced scalar** (`FOAM_RVV_AMUL=0`) so the axis is GS alone.

| Run | Env | WALL | ExecTime (last) | Amul | GS (4 ranks) |
|---|---|---|---|---|---|
| **GS RVV on** | `KERNELS=1 AMUL=0 GS=1` | **389 s** | **344.9 s** | 22.8–28.1 s | **66.6–75.5 s** / 3830 |
| **All off** | `KERNELS=0` | **387 s** | **342.7 s** | 22.9–28.4 s | **61.9–72.2 s** / 3830 |

GS mean ~**71 s vs ~67 s** (~5–6% slower); full solve flat / slightly worse.
Source: [`results/motorbike-rvv-gs-ab-summary.txt`](results/motorbike-rvv-gs-ab-summary.txt).

---

## Why gather RVV loses here

motorBike’s hot linear-algebra is **sparse**:

- **Amul:** irregular `psi[idx[i]]` gathers; short CSR rows.
- **GS:** sequential cell sweep; only a handful of faces per cell (typically ~4–6).
  RVV setup + gather/scatter overhead beats a tight scalar face loop on this VLEN.

Same lesson as “auto-vec did nothing”: the loops are not contiguous dense SIMD.
Useful next levers are **algorithmic** (multicolour / Jacobi-like smoothers,
better matrix layout), not more gather microkernels on short rows.

---

## Reproduce (board)

```bash
export EESSI_VERSION_OVERRIDE=2025.06-001
export EESSI_USER_INSTALL=$HOME/eessi-overlay
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load EasyBuild/5.3.1 EESSI-extend/2025.06-easybuild foss/2025b
module load METIS/5.1.0-GCCcore-14.3.0 SCOTCH/7.0.10-gompi-2025b

OF=$EESSI_USER_INSTALL/versions/2025.06-001/software/linux/riscv64/generic/software/OpenFOAM/v2506-foss-2025b-noPV/OpenFOAM-v2506
set +u; source "$OF/etc/bashrc"; set -u
export LD_LIBRARY_PATH="$EBROOTGCCCORE/lib64:$FOAM_LIBBIN:${LD_LIBRARY_PATH:-}"
export PATH="$FOAM_APPBIN:$PATH"

cd ~/openfoam-runs/motorBike
# Amul A/B
env FOAM_RVV_KERNELS=1 FOAM_RVV_PROFILE=1 mpirun -np 4 simpleFoam -parallel
env FOAM_RVV_KERNELS=0 FOAM_RVV_PROFILE=1 mpirun -np 4 simpleFoam -parallel
# GS-only A/B
env FOAM_RVV_KERNELS=1 FOAM_RVV_AMUL=0 FOAM_RVV_GS=1 FOAM_RVV_PROFILE=1 \
  mpirun -np 4 simpleFoam -parallel
```

Helpers used on-board: `~/openfoam-eb/rebuild-rvv-motorbike-short.sh`,
`~/openfoam-eb/run-rvv-ab.sh`, `~/openfoam-eb/run-rvv-gs-ab.sh`.

---

## Date

Measured **2026-08-01** on RV2.
