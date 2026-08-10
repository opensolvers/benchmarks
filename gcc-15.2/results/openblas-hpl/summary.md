# OpenBLAS / HPL mtune A/B (GCC 15.2-x60 on Orange Pi RV2)

## Setup

- Host: `orangepi@192.168.1.37`, performance governor @ 1.6 GHz
- Compiler: `/home/orangepi/gcc-tune/install/gcc-15.2.0-x60/bin/gcc` (+ EESSI gfortran/libgomp)
- OpenBLAS: `0.3.33.dev` from `~/ob-rvv`, `TARGET=RISCV64_ZVL256B`, `NO_SHARED=1`, `NO_LAPACK=1`, `USE_OPENMP=1`
- Flags: `-O3 -march=rv64gcv_zba_zbb_zbc_zvl256b -mabi=lp64d` + `-mtune=generic-ooo` vs `-mtune=spacemit-x60`
- DGEMM: static-linked, `CLOCK_MONOTONIC_RAW`, `taskset -c 0`, `OPENBLAS_NUM_THREADS=1`
- Two interleaved passes (ooo→x60→x60→ooo); table averages both passes
- HPL 2.3: linked against each static OpenBLAS (CBLAS); HPL itself built with EESSI OpenMPI/GCC 14.3; N=3000, NB=192, P×Q=2×4, 8 ranks

## DGEMM (OpenBLAS)

| N | ooo mean GF/s | x60 mean GF/s | Δ% (x60 vs ooo) | ooo best | x60 best |
|---|---------------|---------------|-----------------|----------|----------|
| 512 | 2.224 | 2.272 | +2.19% | 2.235 | 2.277 |
| 1024 | 2.275 | 2.326 | +2.25% | 2.277 | 2.330 |
| 2048 | 2.296 | 2.383 | +3.79% | 2.297 | 2.384 |

Checksums matched across mtune variants for all sizes: **yes**.

## HPL

| mtune | N | NB | P×Q | Time (s) | Gflops | Residual |
|-------|---|----|-----|----------|--------|----------|
| ooo | 3000 | 192 | 2×4 | 3.40 | 5.2989e+00 | PASSED |
| x60 | 3000 | 192 | 2×4 | 3.37 | 5.3408e+00 | PASSED |

HPL Δ (x60 vs ooo): **+0.79%** (5.2989 → 5.3408 Gflops).

## Caveats

- Prior shared build failed (`exports/` missing after incomplete rsync); rebuilt with `NO_SHARED=1` static `.a` only.
- OpenBLAS default `make` utest link fails under EESSI LDFLAGS (`/lib64/libc.so.6`); builds used `make libs` + install.
- gcc-15.2-x60 install has no libgomp; DGEMM benches linked against EESSI GCCcore 14.3 `-lgomp`.
- Absolute DGEMM GF/s (~2.2–2.4 single-thread) are modest for ZVL256B; A/B isolates mtune only.
- HPL host code is not mtune-A/B'd (EESSI mpicc); only the linked OpenBLAS differs.
- HPL N=3000 is a modest sanity run (~few seconds), not peak-system sizing.
