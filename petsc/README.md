# PETSc — sparse KSP / dense / direct A/Bs (Orange Pi RV2)

PETSc probes on the Orange Pi RV2 (SpaceMiT X60 / Ky X1, RVV 1.0, VLEN=256)
that A/B the **BLAS / LAPACK backend** through FlexiBLAS.

> **Change one variable.** Hold problem + solver fixed; swap only the FlexiBLAS
> backend (patched OpenBLAS vs stock RVV vs scalar). Check finite residuals
> before trusting wall time.

| File | What it does |
|---|---|
| `petsc_ksp_bench.c` | Sparse AIJ 2D Laplacian, Jacobi-CG |
| `petsc_dense_bench.c` | `MATDENSE` MatMult + dense CG |
| `petsc_direct_bench.c` | Sparse-direct LU: MUMPS / SuperLU_DIST / UMFPACK |
| `Makefile` | Builds all three against PETSc + FlexiBLAS |
| `run-petsc-ksp-ab.sh` | FlexiBLAS A/B for Jacobi-CG |
| `run-petsc-dense-direct-ab.sh` | FlexiBLAS A/B for dense + direct suite |

---

## Status (2026-08-14)

| Item | State |
|---|---|
| Board `orangepi@192.168.1.37` | **Reachable** (`orangepirv2`) |
| Overlay module | `PETSc/3.24.0-foss-2025b` (+ SuiteSparse, Hypre, SuperLU_DIST, MUMPS, PnetCDF) |
| Jacobi-CG A/B | **Done** (~1.06×) |
| Dense + direct A/B | **Done** — dense MatMult **~1.70×**; stock RVV **NaN** on dense / SuperLU / UMFPACK |
| SLEPc | Not installed this cycle |

---

## Hardware / stack

| Piece | Value |
|---|---|
| Board | Orange Pi RV2 · SpaceMiT X60 · 8× @ ~1.6 GHz |
| Host | `orangepi@192.168.1.37` |
| PETSc | overlay `PETSc/3.24.0-foss-2025b` (links **FlexiBLAS**) |
| Toolchain | EESSI `2025.06-001` `foss/2025b` |

---

## Build / run

```bash
export EESSI_VERSION_OVERRIDE=2025.06-001
export EESSI_USER_INSTALL=$HOME/eessi-overlay
export EESSI_NO_MODULE_PURGE_ON_INIT=1
source /cvmfs/software.eessi.io/versions/2025.06/init/lmod/bash
module load EasyBuild/5.3.1 EESSI-extend/2025.06-easybuild foss/2025b PETSc/3.24.0-foss-2025b
export LD_LIBRARY_PATH="$EBROOTGCCCORE/lib64:$EBROOTPETSC/lib:$EBROOTFLEXIBLAS/lib:${LD_LIBRARY_PATH:-}"

cd ~/petsc-bench   # or this directory
make all
./run-petsc-ksp-ab.sh
./run-petsc-dense-direct-ab.sh
```

---

## Results — sparse Jacobi-CG (weak BLAS lever)

2D Laplacian **n=400** (160k dofs), Jacobi+CG, 8 threads, best of 3.

| backend | BEST WALL | its | finite |
|---|---:|---:|---:|
| scalar | **11.024 s** | 734 | 1 |
| stock RVV | **10.571 s** (~1.04×) | 734 | 1 |
| patched RVV | **10.374 s** (~1.06×) | 734 | 1 |

Log: [`results/petsc-ksp-flexiblas-ab-20260814T112345Z.txt`](results/petsc-ksp-flexiblas-ab-20260814T112345Z.txt).

---

## Results — denser probes (2026-08-14)

8 threads, best of 3. Log:
[`results/petsc-dense-direct-ab-20260814T182509Z.txt`](results/petsc-dense-direct-ab-20260814T182509Z.txt).

### Dense MatMult (`MATDENSE` n=2048)

| backend | BEST WALL | GF/s | finite |
|---|---:|---:|---:|
| scalar | 0.01020 s | 0.823 | 1 |
| stock RVV | 0.0280 s | 0.300 | **0** (`\|y\|=nan`) |
| patched RVV | **0.00600 s** | **1.397** | 1 |

Patched / scalar ≈ **1.70×**. Stock hits the known OpenBLAS `gemv_n` NaN bug.

### Dense CG (n=1024, `PCNONE`)

| backend | BEST WALL | its | finite |
|---|---:|---:|---:|
| scalar | 0.0232 s | 3 | 1 |
| stock RVV | 0.0074 s | 1 | **0** (diverged / NaN path) |
| patched RVV | **0.00660 s** | 3 | 1 |

Patched / scalar ≈ **3.5×** (tiny iteration count; still shows BLAS on the MatMult path).

### Sparse direct LU

| solver | problem | scalar | stock RVV | patched RVV |
|---|---|---:|---:|---:|
| **MUMPS** | 2D n=200 (40k dofs) | **0.097 s** ✓ | 0.112 s ✓ | 0.112 s ✓ |
| **MUMPS** | 3D n=40 (64k dofs) | **0.586 s** ✓ | 0.635 s ✓ | 0.633 s ✓ |
| **SuperLU_DIST** | 2D n=200 | **0.030 s** ✓ | 0.032 s **NaN** | 0.032 s ✓ |
| **UMFPACK** | 2D n=200 | **0.038 s** ✓ | 0.178 s **NaN** | 0.038 s ✓ |

### Takeaways

1. **Dense PETSc paths** expose FlexiBLAS clearly: patched RVV wins; **stock RVV corrupts**.
2. **SuperLU_DIST / UMFPACK** need the patched OpenBLAS for correctness (same `gemv_n` class of failure as HPL/ELPA).
3. **MUMPS** stayed finite on stock at these sizes but showed **no** patched speedup (analysis / ordering / smaller dense fronts dominate wall time here). Larger 3D problems would be the next lever if chasing MUMPS GFLOPs.
4. **Jacobi-CG AIJ** remains a weak BLAS A/B (~1.06×) — use dense or direct for backend validation.

---

## Next steps

1. Optional: larger MUMPS 3D (`MUMPS_3D_N=60+`) for a stronger frontal BLAS-3 signal.
2. Optional: SLEPc dense eigenprobe.
3. Commit harness + results when ready.
