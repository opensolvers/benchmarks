# Armadillo — FlexiBLAS A/B (SpaceMiT X60 / Orange Pi RV2)

C++ template linear algebra (`Armadillo/12.8.0-foss-2023b`) on EESSI
`riscv.eessi.io` **20240402**. Links FlexiBLAS; A/B swaps the OpenBLAS backend
the same way as [`numpy/`](../numpy).

## Kernels

| Kernel | Call | Backend | Metric |
|--------|------|---------|--------|
| DGEMM | `C = A * B` | BLAS `dgemm` | GFLOP/s |
| EIG | `eig_sym(S)` | LAPACK `dsyev*` | seconds |

## Results (2026-08-22)

8 threads, `N_dgemm=2048`, `N_eig=1024`, finite on all tags.

| Tag | DGEMM GFLOP/s | EIG s | Wall s |
|-----|-------------:|------:|-------:|
| scalar (`OPENBLAS_CORETYPE=RISCV64_GENERIC`) | 4.55 | 1.42 | 18.75 |
| stock (default FlexiBLAS OpenBLAS) | 4.66 | 1.39 | 18.34 |
| patched (`~/libopenblas_x60_eb_fixed.so`) | **8.27** | **0.87** | 11.30 |

Patched vs scalar: DGEMM **1.82×**, EIG **1.63×**, wall **1.66×**.
Stock ≈ scalar (stock OpenBLAS path here is effectively scalar for this size).

Log: `~/logs/armadillo-blas-ab-20260822-035201.log`

## Reproduce

```bash
module load Armadillo/12.8.0-foss-2023b   # 20240402 stack
bash run-armadillo-blas-ab.sh
# or: g++ -O2 -std=c++17 bench_arma.cpp -o bench_arma -larmadillo
#     OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 ./bench_arma 2048 1024
```
