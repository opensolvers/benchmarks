# BLAS / DGEMM verification

Two small programs to check both the **performance** and the **numerical
correctness** of a BLAS backend, and - when something is wrong - to localize
*which* kernel is broken. Written while bringing up a RISC-V RVV (SpaceMiT X60 /
K1) OpenBLAS build, where a vector kernel silently returned `NaN`.

| File | What it does |
|---|---|
| `bench_dgemm.c` | Times square `C = A*B` (level-3 `dgemm`, 3 reps), reports GFLOP/s, prints `C[0]` so garbage is visible. |
| `difftest.c` | `dlopen`s a BLAS `.so` and runs level-1/2/3 routines on fixed inputs, printing `sum / sumsq / nan / inf` per routine for differential comparison. |

Both are single C files, MIT-licensed, and select/switch the backend at runtime
via FlexiBLAS (`FLEXIBLAS=...`) or OpenBLAS (`OPENBLAS_CORETYPE`,
`OPENBLAS_NUM_THREADS`) - no recompilation needed to A/B two backends.

## Build

```bash
gcc -O2 bench_dgemm.c -o bench_dgemm -lflexiblas   # or -lopenblas
gcc -O2 difftest.c    -o difftest    -ldl -lm
```

## Run

```bash
# performance (pick threads + backend via env)
OPENBLAS_NUM_THREADS=8 ./bench_dgemm 4096

# correctness of a specific backend .so
./difftest /path/to/libopenblas.so
```

## A/B a backend with FlexiBLAS (no rebuild)

```bash
# scalar baseline
OPENBLAS_CORETYPE=RISCV64_GENERIC ./bench_dgemm 2048
OPENBLAS_CORETYPE=RISCV64_GENERIC ./difftest  /path/to/libopenblas.so
# vector / alternative backend
FLEXIBLAS=/path/to/libopenblas.so ./bench_dgemm 2048
./difftest                        /path/to/vector-libopenblas.so
```

## Example - the OpenBLAS 0.3.30 RISC-V `gemv_n` NaN bug (SpaceMiT X60)

`difftest` run against the **stock** EESSI-CVMFS OpenBLAS 0.3.30 (a
`DYNAMIC_ARCH` build that dispatches the `ZVL256B` RVV kernels on the X60), the
same lib forced to scalar, and a **patched** RVV build:

| backend / dispatch | `dgemv` nan | `dgemm` nan | `dtrsm` nan | `dgemv` sum |
|---|--:|--:|--:|---|
| stock CVMFS, **default RVV** (`ZVL256B`) | **192** | 0 | 0 | 198.94 (wrong) |
| stock CVMFS, forced scalar (`RISCV64_GENERIC`) | 0 | 0 | 0 | 42.06549 (reference) |
| **patched RVV** (`gemv_n` fix) | 0 | 0 | 0 | 42.06549 (matches reference) |

The differential test pins the fault to `dgemv` alone - `dgemm` and `dtrsm` are
correct even on the broken RVV build. That is exactly why a plain `dgemm`
benchmark looks fine while **HPL fails with a NaN residual**: HPL's panel
factorization leans on `dgemv`. The patched vector build reproduces the scalar
reference bit-for-bit.

Performance on the same machine (`bench_dgemm`, 1 core, N=2048, via FlexiBLAS):

| backend | GFLOP/s | `C[0]` |
|---|--:|--:|
| scalar (`RISCV64_GENERIC`) | 1.16 | 245.24 |
| patched RVV (`ZVL256B`) | 2.62 | 245.24 |

i.e. ~2.3x faster and numerically identical. (Root cause: OpenBLAS 0.3.30
`kernel/riscv64/gemv_n_vector.c` zeroes an *uninitialized* vector accumulator;
fixed upstream after 0.3.30.)
