#!/usr/bin/env python3
"""
bench_blas.py - NumPy BLAS+LAPACK backend micro-benchmark.

Times two dense linear-algebra kernels through NumPy and reports throughput
plus a finiteness check, as a real-application proxy for the underlying
BLAS/LAPACK backend (NumPy dispatches into it for @ / matmul and np.linalg):

  DGEMM : C = A @ B     (N x N, level-3 BLAS dgemm)         -> GFLOP/s
  EIGH  : eigvalsh(S)   (N x N symmetric, LAPACK dsyevd)    -> seconds

These are what real NumPy/SciPy code actually spends time in, so the benchmark
doubles as a correctness gate: finite=False means the backend produced NaN/Inf.

Run:
  OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python3 bench_blas.py [dgemm_N] [eigh_N]
  defaults: dgemm_N=4096  eigh_N=2048

A/B a BLAS backend via FlexiBLAS (no reinstall), exactly as for the C benchmarks
in this repo:
  scalar : OPENBLAS_CORETYPE=RISCV64_GENERIC python3 bench_blas.py
  vector : FLEXIBLAS=/path/to/libopenblas.so python3 bench_blas.py

SPDX-License-Identifier: MIT
"""
import sys
import time
import numpy as np


def best(fn, reps):
    b = 1e18
    r = None
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn()
        b = min(b, time.perf_counter() - t0)
    return b, r


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    rng = np.random.default_rng(0)

    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    A @ B  # warmup
    t, C = best(lambda: A @ B, 3)
    print(f"  DGEMM N={n}: {t:6.2f} s   {2 * n**3 / t / 1e9:6.2f} GFLOP/s   finite={bool(np.isfinite(C[0, 0]))}")

    S = rng.standard_normal((m, m))
    S = S @ S.T  # symmetric positive semidefinite
    t, w = best(lambda: np.linalg.eigvalsh(S), 1)
    print(f"  EIGH  N={m}: {t:6.2f} s   (LAPACK dsyevd)          finite={bool(np.isfinite(w).all())}")


if __name__ == "__main__":
    main()
