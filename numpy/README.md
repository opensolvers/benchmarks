# NumPy BLAS/LAPACK backend benchmark

A tiny script that times two dense linear-algebra kernels through NumPy and
checks the results are finite. It is a real-application proxy for the
**performance and numerical correctness of the BLAS/LAPACK backend** that NumPy
(and SciPy) dispatch into:

| Kernel | NumPy call | Backend routine | Metric |
|---|---|---|---|
| DGEMM | `A @ B` | level-3 BLAS `dgemm` | GFLOP/s |
| EIGH | `np.linalg.eigvalsh(S)` | LAPACK `dsyevd` | seconds |

Unlike a raw `dgemm` micro-benchmark, `eigvalsh` also exercises the BLAS-2 /
reduction paths (tridiagonalization) that a buggy vector kernel can quietly turn
into `NaN` - so the `finite` flag is a genuine correctness gate, not decoration.

## Requirements

A NumPy linked against a FlexiBLAS (or OpenBLAS) backend. On an EESSI/EasyBuild
stack, e.g.:

```bash
module load SciPy-bundle/2025.07-gfbf-2025b   # provides numpy on top of FlexiBLAS
```

## Run

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python3 bench_blas.py [dgemm_N] [eigh_N]
# defaults: dgemm_N=4096  eigh_N=2048
```

## A/B a BLAS backend with FlexiBLAS (no reinstall)

Because NumPy here goes through the FlexiBLAS hub, the actual BLAS implementation
can be switched per run with environment variables - identical to the C
benchmarks in this repo:

```bash
# scalar baseline (force the generic, non-vector OpenBLAS kernels)
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OPENBLAS_CORETYPE=RISCV64_GENERIC \
  python3 bench_blas.py

# an alternative OpenBLAS .so as the backend
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 FLEXIBLAS=/path/to/libopenblas.so \
  python3 bench_blas.py
```

## Example result - RISC-V SpaceMiT X60 (Orange Pi RV2, K1, 8x X60 @ ~1.6 GHz)

8 threads, scalar vs a vector (RVV `ZVL256B`) OpenBLAS with the `gemv_n` NaN fix,
via FlexiBLAS:

| Kernel | scalar (`RISCV64_GENERIC`) | RVV `ZVL256B` (patched) | speedup |
|---|---:|---:|---:|
| DGEMM N=4096 | 4.77 GFLOP/s | 11.52 GFLOP/s | **2.4x** |
| EIGH N=2048 | 10.54 s | 6.72 s | **1.6x** |

Both backends produce finite results here (the patched vector build is correct
through real LAPACK). The pure `dgemm` speedup (2.4x) is larger than the
eigensolver's (1.6x) because the eigensolver mixes BLAS-3 with latency-bound
BLAS-2 tridiagonalization - the same reason a full eigensolver (see `elpa/`) is a
more conservative, more representative probe than raw `dgemm`.
