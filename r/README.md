# R — FlexiBLAS A/B (SpaceMiT X60 / Orange Pi RV2)

`R/4.4.1-gfbf-2023b` on EESSI `riscv.eessi.io` **20240402**. Matrix ops dispatch
into the same FlexiBLAS stack as [`numpy/`](../numpy) and [`armadillo/`](../armadillo).

## Kernels

| Kernel | Call | Backend | Metric |
|--------|------|---------|--------|
| GEMM | `A %*% B` | BLAS `dgemm` | GFLOP/s |
| EIGEN | `eigen(S, symmetric=TRUE)` | LAPACK `dsyev*` | seconds |

## Results (2026-08-22)

8 threads, `N_dgemm=2048`, `N_eig=1024`. Eigenvalue sum checksum matches across tags.

| Tag | GEMM GFLOP/s | EIGEN s | Wall s |
|-----|-------------:|--------:|-------:|
| scalar (`OPENBLAS_CORETYPE=RISCV64_GENERIC`) | 4.47 | 1.72 | 29.41 |
| stock | 4.42 | 1.62 | 29.38 |
| patched (`~/libopenblas_x60_eb_fixed.so`) | **8.03** | **1.19** | 21.94 |

Patched vs scalar: GEMM **1.80×**, EIGEN **1.45×**, wall **1.34×**.

Log: `~/logs/r-blas-ab-20260822-073644.log`

## Reproduce

```bash
module load R/4.4.1-gfbf-2023b
bash run-r-blas-ab.sh
# or: OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 Rscript bench_r.R 2048 1024
```
