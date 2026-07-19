# ELPA eigensolver benchmark

A tiny, single-process benchmark that times a dense real-symmetric eigenproblem
solved by [ELPA](https://elpa.mpcdf.mpg.de/) (1-stage solver) and checks that the
computed eigenvalues are **finite**.

It is a deliberately small proxy for the **performance and numerical correctness
of the underlying BLAS/LAPACK backend**. ELPA's tridiagonalization and
back-transform are dominated by BLAS-2/BLAS-3 (`dsymv`, `dsyr2k`, `dgemm`,
`dtrmm`), so swapping the BLAS backend (e.g. through [FlexiBLAS](https://github.com/mpimd-csc/flexiblas))
changes both the runtime and - if the backend is buggy - whether the eigenvalues
come back as numbers at all.

## Why an eigensolver as a BLAS probe

A raw `dgemm` micro-benchmark tells you the BLAS-3 peak, but real applications
spend time in a *mix* of BLAS-3 and latency-bound BLAS-2 plus non-BLAS work. A
dense eigensolver is a good middle ground: it is a real HPC kernel (DFT codes
such as CP2K, VASP, ELSI use ELPA), it exercises `gemv`/`symv` as well as `gemm`,
and a single wrong reduction anywhere shows up as a `NaN` eigenvalue. So the
`finite` flag doubles as a correctness gate and the wall-time as a speed metric.

## Build

Load an EESSI/EasyBuild ELPA module (which pulls in ScaLAPACK, FlexiBLAS and an
MPI), then `make`:

```bash
module load ELPA/2025.06.002-foss-2025b
make
```

The `Makefile` reads `$EBROOTELPA`, `$EBROOTSCALAPACK` and `$EBROOTFLEXIBLAS` and
supports both the OpenMP (`libelpa_openmp`) and plain (`libelpa`) ELPA builds.

## Run

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  mpirun --bind-to none -np 1 ./elpa_bench [na] [nev] [nblk]
# defaults: na=3000  nev=na  nblk=64
```

This is a **shared-memory** benchmark: it runs as a single MPI rank on a trivial
1x1 process grid, and all parallelism comes from the threaded BLAS backend.
`--bind-to none` is important - without it MPI pins the single rank to one core
and the threaded BLAS cannot spread out.

Output is one line:

```
ELPA 1stage na=3000 nev=3000 nblk=64: 34.81 s  ev0=-31.44649 evN=1499.70715 finite=1
```

Lower time = faster backend. `finite=0` means the backend produced a `NaN`
eigenvalue (a correctness failure); the program also exits non-zero in that case
so it can be used as a CI gate.

## A/B a BLAS backend with FlexiBLAS (no rebuild)

Because the binary links the FlexiBLAS hub, you can switch the actual BLAS
implementation per run with environment variables:

```bash
# scalar baseline (force the generic, non-vector OpenBLAS kernels)
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OPENBLAS_CORETYPE=RISCV64_GENERIC \
  mpirun --bind-to none -np 1 ./elpa_bench 3000

# an alternative OpenBLAS .so as the backend
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 FLEXIBLAS=/path/to/libopenblas.so \
  mpirun --bind-to none -np 1 ./elpa_bench 3000
```

## Example result - RISC-V SpaceMiT X60 (Orange Pi RV2, K1, 8x X60 @ ~1.6 GHz)

`na=3000`, 1-stage, 8 threads, three OpenBLAS backends via FlexiBLAS:

| Backend | Time | Correctness |
|---|---:|---|
| stock CVMFS OpenBLAS 0.3.30, default RVV (`ZVL256B`) dispatch | 42.17 s | **`ev0=nan`, finite=0 - FAIL** |
| scalar (`OPENBLAS_CORETYPE=RISCV64_GENERIC`) | 54.92 s | finite=1 |
| RVV `ZVL256B` with the `gemv_n` NaN fix backported | **34.81 s** | finite=1 |

Two conclusions:

1. **Correctness.** The stock vector `gemv_n` kernel in OpenBLAS 0.3.30 returns a
   `NaN` (a known bug fixed upstream after 0.3.30), which corrupts the eigensolve
   - so the default vector backend fails this benchmark on the X60. The patched
   vector build reproduces the scalar eigenvalues exactly.
2. **Speed.** With the fix in place the vector backend is **1.58x** faster than
   the scalar baseline (54.92 / 34.81). The broken run being "faster" than scalar
   is meaningless - its output is garbage.

The speedup is smaller than a pure `dgemm` A/B on the same machine (~2.4x),
exactly because the eigensolver mixes BLAS-3 with latency-bound BLAS-2
tridiagonalization - which is the point of using it as a more representative
probe.

## Cross-board confirmation - Banana Pi BPI-F3 (same K1 / X60 SoC)

`na=3000`, 8 threads on a [Banana Pi BPI-F3](https://www.banana-pi.org/)
(SpaceMiT K1, 8x X60 @ 1.6 GHz), EESSI `2025.06-001` - the same three backends,
the same story as the RV2:

| Backend | Time | Correctness |
|---|---:|---|
| stock CVMFS OpenBLAS 0.3.30, default RVV (`ZVL256B`) | 37.93 s | **`ev0=nan`, finite=0 - FAIL** |
| scalar (`OPENBLAS_CORETYPE=RISCV64_GENERIC`) | 50.42 s | finite=1 |
| RVV `ZVL256B` with the `gemv_n` fix backported | **34.83 s** | finite=1 |

Patched vector is **1.45x** faster than scalar (50.42 / 34.83) and returns
`ev0=-31.44649 evN=1499.70715`, bit-identical to the scalar reference; the stock
vector backend NaNs the eigensolve. A second K1 board, same conclusion.

## Files

- `elpa_bench.c` - the benchmark (single source file, MIT licensed).
- `Makefile` - builds against a loaded ELPA module.
