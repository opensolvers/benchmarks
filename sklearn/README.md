# scikit-learn — FlexiBLAS A/B (SpaceMiT X60 / Orange Pi RV2)

`scikit-learn/1.4.0-gfbf-2023b` on EESSI `riscv.eessi.io` **20240402**. Estimators
dispatch into NumPy → FlexiBLAS; A/B swaps the OpenBLAS backend like
[`numpy/`](../numpy) and [`armadillo/`](../armadillo).

## Kernels

| Kernel | Call | Notes |
|--------|------|-------|
| PCA | `PCA(svd_solver="full").fit_transform` | dense SVD / GEMM |
| Ridge | `Ridge(solver="cholesky").fit` | LAPACK/BLAS |

## Results (2026-08-22)

8 threads, `N=8000`, `D=512`, `K=64`. Checksums match across tags.

| Tag | PCA s | Ridge s | Wall s |
|-----|------:|--------:|-------:|
| scalar (`OPENBLAS_CORETYPE=RISCV64_GENERIC`) | 8.33 | 0.99 | 25.91 |
| stock | 8.50 | 0.99 | 21.50 |
| patched (`~/libopenblas_x60_eb_fixed.so`) | **6.80** | **0.52** | 19.20 |

Patched vs scalar: PCA **1.22×**, Ridge **1.90×**.

Log: `~/logs/sklearn-blas-ab-20260822-040154.log`

## Reproduce

```bash
module load scikit-learn/1.4.0-gfbf-2023b
bash run-sklearn-blas-ab.sh
```
