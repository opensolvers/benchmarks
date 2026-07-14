# ScaLAPACK pdsyev eigensolver benchmark

A distributed dense real-symmetric eigenproblem solved by ScaLAPACK `PDSYEV` on a
2D block-cyclic process grid, as a **pure-MPI** proxy for the performance and
numerical correctness of the (node-local) BLAS backend. Companion to
[`../elpa`](../elpa): the *same* problem, a different library and parallelism
model - ScaLAPACK runs many MPI ranks each with single-threaded BLAS, whereas the
ELPA driver runs one rank with threaded BLAS.

Each rank fills only its local block-cyclic entries of a deterministic symmetric,
diagonally-dominant matrix (global-index math, no scatter), sets up the BLACS
grid + `descinit_` + `pdsyev_`, and rank 0 prints the solve time and a finiteness
check. Exits non-zero on a non-finite eigenvalue (CI correctness gate).

## Build

```bash
module load ScaLAPACK/2.2.2-gompi-2025b-fb
make
```

## Run

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  mpirun --bind-to core -np 8 ./scalapack_bench [na] [nb] [nprow]
# defaults: na=3000  nb=64  grid auto-factored near-square (nprow x np/nprow)
```

Pure-MPI: one thread per rank, parallelism from the process grid. `--bind-to core`
pins each rank to a core.

## A/B a BLAS backend with FlexiBLAS (no rebuild)

```bash
# scalar baseline
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENBLAS_CORETYPE=RISCV64_GENERIC \
  mpirun --bind-to core -np 8 ./scalapack_bench 3000 64 2
# vector / alternative backend
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FLEXIBLAS=/path/to/libopenblas.so \
  mpirun --bind-to core -np 8 ./scalapack_bench 3000 64 2
```

## Example result - RISC-V SpaceMiT X60 (Orange Pi RV2, K1, 8x X60 @ ~1.6 GHz)

`na=3000`, `2x4` grid, 8 ranks x 1 thread, three OpenBLAS backends via FlexiBLAS:

| backend | time | result |
|---|---:|---|
| stock CVMFS 0.3.30, default RVV (`ZVL256B`) | - | **HANG** (killed at 200 s) |
| scalar (`RISCV64_GENERIC`) | 116.87 s | finite=1 |
| RVV `ZVL256B` + `gemv_n` NaN fix | 107.23 s | finite=1 (identical eigenvalues) |

The unpatched vector `gemv_n` NaN (bug fixed upstream after 0.3.30) doesn't just
corrupt the result here - it **hangs** the solver, because `pdsyev`'s serial
tridiagonal QR iteration never converges on a `NaN`. The patched build is correct
(eigenvalues identical to scalar) and **1.09x** faster.

That modest speedup - versus ~2.4x for a raw `dgemm` A/B and ~1.6x for a threaded
eigensolver on the same machine - is expected and is the point of including this
benchmark: at 8 ranks the node-local BLAS blocks are small and MPI communication
plus the serial tridiagonal solve dominate wall time, so the RVV-accelerable
BLAS-3 fraction is small. It is the conservative, communication-bound end of the
BLAS-backend spectrum.
